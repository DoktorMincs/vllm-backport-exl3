# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


def conf_cut_thresholds() -> tuple[float, float] | None:
    """`VLLM_DSPARK_CONF_CUT="conf_min,cum_min"` -> (conf_min, cum_min).

    Regra copiada do motor Maya (src/ds4/ds4_main.cpp:410-418): um draft i so'
    e' verificado enquanto `conf[i] >= conf_min` E o produto das confiancas
    ate' i `>= cum_min`; no primeiro que falha, o resto da cadeia e' descartado.
    Sem a env (default) devolve None e nada muda.
    """
    raw = os.environ.get("VLLM_DSPARK_CONF_CUT", "").strip()
    if raw.lower() in ("", "0", "false", "no", "off"):
        return None
    try:
        parts = [p for p in raw.replace(";", ",").split(",") if p.strip()]
        conf_min = float(parts[0]) if parts else 0.5
        cum_min = float(parts[1]) if len(parts) > 1 else conf_min
    except ValueError:
        return None
    return conf_min, cum_min


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.confidence_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        self._skip = False

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        self._skip = False
        needs_cut = (
            confidence is not None and conf_cut_thresholds() is not None
        )
        if not input_batch.has_structured_output_reqs and not needs_cut:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            self.confidence_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        # For the confidence cut, both the ids and the confidences come back.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            if needs_cut:
                assert confidence is not None
                n = len(self.req_ids)
                self.confidence_np = async_copy_to_np(
                    confidence[:n, : self.num_draft_tokens].contiguous()
                )
                confidence.record_stream(self.copy_stream)
            else:
                self.confidence_np = None
            self.copy_event.record()

    def set_skip_batch(self, input_batch: InputBatch) -> None:
        """Passo de descanso (VLLM_DSPARK_REST): nada a verificar.

        Devolver lista vazia por requisicao faz o scheduler agendar so' o token
        novo; os drafts velhos ficam no `req_states` e nao sao usados.
        """
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = 0
        self.draft_tokens_np = None
        self.confidence_np = None
        self._skip = True

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self._skip:
            self._skip = False
            return DraftTokenIds(self.req_ids, [[] for _ in self.req_ids])
        cut = conf_cut_thresholds()
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
            if cut is not None and self.confidence_np is not None:
                conf_min, cum_min = cut
                trimmed: list[list[int]] = []
                for i, row in enumerate(draft_token_ids):
                    conf = self.confidence_np[i]
                    prod = 1.0
                    k = 0
                    for j in range(min(len(row), conf.shape[0])):
                        c = float(conf[j])
                        prod *= c
                        if c >= conf_min and prod >= cum_min:
                            k = j + 1
                        else:
                            break
                    trimmed.append(row[:k])
                draft_token_ids = trimmed
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
