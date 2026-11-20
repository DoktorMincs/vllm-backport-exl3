# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

from typing import Any

import os
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import record_function

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.markov_argmax import (
    FusedMarkovSampler,
    build_fused_markov_sampler,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.draft_token_confidence_probs = torch.empty_like(
            self.draft_tokens, dtype=torch.float32
        )
        self.enable_adaptive_verification = (
            self.speculative_config.enable_adaptive_verification
        )
        # Corte por confianca (regra do motor Maya): VLLM_DSPARK_CONF_CUT=
        # "conf_min,cum_min". Precisa das confiancas mesmo com a verificacao
        # adaptativa desligada; o corte em si acontece no scheduler (via
        # DraftTokensHandler), nao aqui.
        self.conf_cut_enabled = os.environ.get(
            "VLLM_DSPARK_CONF_CUT", ""
        ).strip().lower() not in ("", "0", "false", "no", "off")
        # As confiancas passam a ser calculadas sempre que o corte pedir.
        self._need_confidence = (
            self.enable_adaptive_verification or self.conf_cut_enabled
        )

        # Fused Markov chain (greedy vocab-sharded path + hybrid prob fallback);
        # built in load_draft_model once the head's weights and shard geometry exist.
        self._fused_markov: FusedMarkovSampler | None = None
        self._prob_draft_hybrid = (
            os.environ.get("VLLM_DSPARK_PROB_DRAFT", "1") != "0"
            and self.use_local_argmax_reduction
            and self.draft_logits is not None
        )
        self._sampler: Any = None
        self._markov_w2_full: torch.Tensor | None = None
        self._base_logits_shard_static: torch.Tensor | None = None
        self._draft_vocab_size: int = self.vocab_size
        # Diagnostico de tempo (VLLM_M3_TIME_DRAFT=1): [logits, markov, passos].
        self._t_dbg: list[float] = [0.0, 0.0, 0.0]

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        if model.draft_id_to_target_id is not None:
            self._draft_vocab_size = int(model.draft_id_to_target_id.shape[0])
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        if self.enable_adaptive_verification and model.model.confidence_head is None:
            raise ValueError(
                "Adaptive verification needs a DSpark checkpoint with a confidence "
                "head, and this one has none. Pass "
                "enable_adaptive_verification=false in the speculative config to verify"
                " a fixed number of drafts instead."
            )
        if (
            self.use_local_argmax_reduction
            and (self.draft_logits is None or self._prob_draft_hybrid)
            and envs.VLLM_DSPARK_FUSED_MARKOV
        ):
            self._fused_markov = build_fused_markov_sampler(
                model, self.max_num_reqs, self.num_speculative_steps, self.device
            )
            if self._prob_draft_hybrid and self._fused_markov is not None:
                op = self._fused_markov.op
                shard_width = op.w2.shape[0]
                self._base_logits_shard_static = torch.empty(
                    (self.max_num_reqs, self.num_speculative_steps, shard_width),
                    dtype=op.w2.dtype,
                    device=self.device,
                )
                w2_full = (
                    tensor_model_parallel_all_gather(op.w2, dim=0)
                    if op.tp_size > 1
                    else op.w2
                )
                self._markov_w2_full = w2_full[: self._draft_vocab_size].contiguous()
        return model

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if self.draft_logits is None:
            return self.model.map_draft_to_target(logits.argmax(dim=-1))

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        return gumbel_sample(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )

    def _validate_local_argmax_reduction(self) -> None:
        # DSpark's greedy selection spans the SUM of two vocab-parallel heads
        # (base draft logits + the Markov transition bias), so it needs a
        # different pair of hooks than the base class's single get_top_tokens.
        if not self.use_local_argmax_reduction:
            return
        if (
            self.speculative_config.draft_sample_method == "probabilistic"
            and not self._prob_draft_hybrid
        ):
            raise ValueError(
                "use_local_argmax_reduction is not compatible with "
                "draft_sample_method='probabilistic': verification needs whole "
                "processed logit rows, which is exactly what the reduction "
                "avoids materializing."
            )
        missing = [
            name
            for name in ("compute_draft_logits_shard", "select_draft_token_shard")
            if not hasattr(self.model, name)
        ]
        if missing:
            raise ValueError(
                "use_local_argmax_reduction is enabled but DSpark draft model "
                f"{self.model.__class__.__name__} does not implement "
                f"{', '.join(missing)}()."
            )
        logger.info(
            "DSpark: vocab-sharded Markov head with local argmax reduction "
            "(communication: O(2*tp_size) vs O(vocab_size) per draft position)."
        )

    def _debug_draft_attn(self, attn_metadata, num_reqs: int) -> None:
        """Mostra o que a atencao do draft vai ler (janela x ring gravado).

        Se `seq_lens` for pequeno (so' o bloco) ou o `block_table` nao tiver as
        posicoes do contexto, o draft esta' prevendo so' a partir do token
        ancora -- que e' o que uma aceitacao de ~17% sugere.
        """
        n = getattr(self, "_dbg_attn_n", 0)
        if n >= 6:
            return
        self._dbg_attn_n = n + 1
        md = attn_metadata
        if isinstance(md, (list, tuple)):
            md = md[0] if md else None
        if not isinstance(md, dict) or not md:
            logger.info("M3-DEBUG draft attn#%d: metadata vazio (%s)", n, type(md))
            return
        parts = []
        for prefix, meta in list(md.items())[:3]:
            seq = getattr(meta, "seq_lens", None)
            bt = getattr(meta, "block_table", None)
            sl = getattr(meta, "slot_mapping", None)
            parts.append(
                "%s: seq_len=%s bt=%s slots=%s"
                % (
                    str(prefix)[-40:],
                    None if seq is None else seq[:num_reqs].tolist(),
                    None if bt is None else tuple(bt.shape),
                    None if sl is None else int(sl.numel()),
                )
            )
        logger.info(
            "M3-DEBUG draft attn#%d reqs=%d groups=%d %s",
            n,
            num_reqs,
            len(md),
            " | ".join(parts),
        )

    def _dump_taps(
        self,
        aux_hidden_states: list[torch.Tensor] | None,
        main_x: torch.Tensor,
        num_target_tokens: int,
    ) -> None:
        """Grava (taps crus, main_x) por passo para o oraculo do drafter.

        Formato (little endian), concatenado no mesmo arquivo:
            int32 n_targets, n_taps, hidden, step
            int32[n_targets]              positions
            float32[n_targets*n_taps*hidden]  aux
            float32[n_targets*hidden]     main_x
        Um <prefix>.meta.json ao lado descreve `tap_layer_ids`.
        """
        import json

        import numpy as np

        if not aux_hidden_states:
            return
        if get_tensor_model_parallel_rank() != 0:
            return
        prefix = os.environ["VLLM_DSPARK_TAPS_DUMP"]
        n = int(num_target_tokens)
        if n <= 0:
            return
        aux = torch.cat(aux_hidden_states, dim=-1)[:n].float()
        mx = main_x[:n].float()
        pos = self.context_positions[:n].to(torch.int32)
        n_taps = int(aux.shape[-1] // self.draft_model_config.get_hidden_size())
        hidden = int(aux.shape[-1] // n_taps)
        step = getattr(self, "_dbg_dump_n", 0)
        self._dbg_dump_n = step + 1
        a = aux.cpu().numpy()
        m = mx.cpu().numpy()
        p = pos.cpu().numpy()
        meta = prefix + ".meta.json"
        if not os.path.exists(meta):
            with open(meta, "w") as f:
                json.dump(
                    {
                        "tap_layer_ids": list(
                            getattr(self.draft_model_config.hf_config,
                                    "eagle_aux_hidden_state_layer_ids", [])
                        ),
                        "hidden": hidden,
                        "n_taps": n_taps,
                        "window_env": os.environ.get("DSPARK_TAP_WIN", "1,2,3"),
                    },
                    f,
                )
        with open(prefix, "ab") as f:
            f.write(struct.pack("<4i", n, n_taps, hidden, step))
            f.write(p.tobytes())
            f.write(a.tobytes())
            f.write(m.tobytes())
        big = int((a.__abs__() > 65504.0).sum())
        logger.info(
            "M3-DUMP step=%d n=%d n_taps=%d taps_absmax=%.1f gt65504=%d "
            "main_x[absmax=%.3f std=%.3f] pos0=%s",
            step,
            n,
            n_taps,
            float(np.abs(a).max()),
            big,
            float(np.abs(m).max()),
            float(m.std()),
            p[:4].tolist(),
        )

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        # Under the local-argmax reduction these are this rank's vocab columns
        # only, so `vocab_size` below is the shard width -- nothing in this
        # method reads it as a vocab id.
        with record_function("dspark::draft_logits"):
            _t0 = None
            if envs.VLLM_M3_TIME_DRAFT and not torch.cuda.is_current_stream_capturing():
                torch.cuda.synchronize()
                _t0 = time.perf_counter()
            if self.use_local_argmax_reduction:
                base_logits = self.model.compute_draft_logits_shard(sample_hidden)
            else:
                base_logits = self.model.compute_draft_logits(sample_hidden)
            if _t0 is not None:
                torch.cuda.synchronize()
                self._t_dbg[0] += time.perf_counter() - _t0
                _t0 = time.perf_counter()
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
        # Contador (nao one-shot): o warmup consumia o diagnostico antes da
        # primeira requisicao real.
        _n_dbg = getattr(self, "_dbg_n", 0)
        if envs.VLLM_M3_DEBUG_DRAFT and _n_dbg < int(
            os.environ.get("VLLM_DSPARK_LOGITS_N", "6")
        ):
            self._dbg_n = _n_dbg + 1
            l0 = base_logits[0].float()
            logger.info(
                "M3-DEBUG draft logits#%d: shape=%s absmax=%.3f mean=%.4f "
                "argmax_per_step=%s anchor_id=%s",
                _n_dbg,
                tuple(base_logits.shape),
                l0.abs().max().item(),
                l0.mean().item(),
                l0.argmax(dim=-1).tolist(),
                self.input_buffers.input_ids[self._anchor_idx[:1]].tolist(),
            )

        # So' o rank 0 do TP escreve: os dois ranks apendam no mesmo arquivo e
        # os registros se intercalam (o leitor perde o alinhamento).
        if (
            os.environ.get("VLLM_DSPARK_TAPS_DUMP")
            and get_tensor_model_parallel_rank() == 0
            and not torch.cuda.is_current_stream_capturing()
        ):
            # Par do dump dos taps: os head_hidden e os drafts que o vLLM
            # realmente produziu, para o oraculo comparar linha a linha.
            _prefix = os.environ["VLLM_DSPARK_TAPS_DUMP"] + ".draft.bin"
            _step = getattr(self, "_dbg_draft_n", 0)
            self._dbg_draft_n = _step + 1
            _anchor = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
            _toks = self.draft_tokens[:num_reqs]
            with open(_prefix, "ab") as f:
                f.write(
                    struct.pack("<4i", _step, num_reqs, n_spec,
                                sample_hidden.shape[-1])
                )
                f.write(_anchor.to(torch.int32).cpu().numpy().tobytes())
                f.write(_toks.to(torch.int32).cpu().numpy().tobytes())
                f.write(sample_hidden.float().cpu().numpy().tobytes())

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        if self._fused_markov is not None and not self.conf_cut_enabled:
            if self._base_logits_shard_static is not None:
                self._base_logits_shard_static[:num_reqs].copy_(base_logits)
            # Same chain as the loop below, with each step's seven kernels and
            # three [num_reqs, shard_width] round trips collapsed into one
            # pass over markov_w2. Writes draft_tokens directly.
            self._fused_markov.sample(num_reqs, base_logits, prev, self.draft_tokens)
            if _t0 is not None:
                self._dbg_markov_time(_t0)
            return

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self._need_confidence:
                confidence_markov_embeds.append(markov_embed)
            if self.use_local_argmax_reduction:
                # Greedy-only (enforced in _validate_local_argmax_reduction):
                # the argmax over base + bias reduces to one (value, id) pair
                # per rank, so neither head materializes a full-vocab row.
                draft_sampled_i = self.model.map_draft_to_target(
                    self.model.select_draft_token_shard(markov_embed, base_logits[:, i])
                )
                self.draft_tokens[:num_reqs, i] = draft_sampled_i
                prev = draft_sampled_i
                continue
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self._need_confidence:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )
        if _t0 is not None:
            self._dbg_markov_time(_t0)

    def _post_propose_sample(self, input_batch: Any, dummy_run: bool) -> None:
        if (
            dummy_run
            or not self._prob_draft_hybrid
            or self.draft_logits is None
            or self._base_logits_shard_static is None
            or self._markov_w2_full is None
            or self._sampler is None
        ):
            return

        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return
        idx_np = input_batch.idx_mapping_np
        ss = self._sampler.sampling_states
        temp_np = ss.temperature.np[idx_np]
        if np.all(temp_np == 0.0):
            # Greedy batch: FusedMarkovSampler already wrote exact argmax draft
            # tokens inside the FULL CUDA graph, and RejectionSampler skips
            # draft_logits whenever temp == 0.0.
            return

        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        shard_slice = self._base_logits_shard_static[:num_reqs]
        full_base = (
            tensor_model_parallel_all_gather(shard_slice, dim=-1)
            if self._fused_markov is not None and self._fused_markov.op.tp_size > 1
            else shard_slice
        )[..., : self._draft_vocab_size]

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        has_filter = (
            self._d2t_scatter_index is None
            and bool(
                np.any(ss.min_p.np[idx_np] > 0.0)
                or np.any(ss.top_k.np[idx_np] != ss.vocab_size)
                or np.any(ss.top_p.np[idx_np] != 1.0)
            )
        )
        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            bias = F.linear(markov_embed, self._markov_w2_full)
            logits_i = full_base[:, i] + bias
            if has_filter:
                req_idx_gpu = idx_map[:, i]
                scaled = logits_i.float()
                ss.apply_temperature(scaled, req_idx_gpu, idx_np)
                ss.apply_min_p(scaled, req_idx_gpu, idx_np)
                scaled = ss.apply_top_k_top_p(scaled, req_idx_gpu, idx_np)
                logits_i = logits_i.masked_fill(scaled == float("-inf"), float("-inf"))
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

    def _dbg_markov_time(self, t_markov_start: float) -> None:
        """Acumula o tempo da cadeia Markov e loga a cada 10 passos."""
        torch.cuda.synchronize()
        self._t_dbg[1] += time.perf_counter() - t_markov_start
        self._t_dbg[2] += 1
        if self._t_dbg[2] % 10 == 0:
            n = self._t_dbg[2]
            logger.info(
                "M3-TIME draft detail (sync, %d passos): logits_lm_head=%.2f ms "
                "markov=%.2f ms",
                n,
                1000 * self._t_dbg[0] / n,
                1000 * self._t_dbg[1] / n,
            )
            self._t_dbg[0] = 0.0
            self._t_dbg[1] = 0.0

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to ``-inf``. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self._need_confidence:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self._need_confidence:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        # NOTE: these scopes are host-side, so they only resolve when the step
        # runs eagerly. Under CUDA-graph replay the whole step is one launch and
        # the ranges do not reappear -- profile with cudagraphs off to use them.
        with record_function("dspark::draft_backbone"):
            head_hidden = self._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        with record_function("dspark::markov_sampling"):
            self._sample_sequential(num_reqs, head_hidden)
