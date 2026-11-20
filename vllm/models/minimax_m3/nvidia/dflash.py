# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark drafter for MiniMax-M3.1-Flash (``DSparkMiniMaxDraftModel``).

O checkpoint do drafter (``<pack>/dspark``) e' um DFlash denso de 5 camadas:
um ``fc`` que funde os estados de 5 camadas do alvo (3, 17, 31, 45, 59 ->
5 x 6144), 5 camadas densas (GQA 64/4 cabecas, head_dim 128, RoPE parcial 0.5,
qk-norm Gemma por cabeca, janela deslizante 4096, MLP SwiGLU-OAI de 12288, sem
MoE) e as cabecas DSpark (markov rank 256 + confidence). Nao traz embed_tokens
nem lm_head: o lm_head e' compartilhado com o alvo e a tabela de embedding e'
copiada do pack do alvo na carga (sob PP o estagio do drafter nao tem o
embedding do alvo, entao compartilhar e' impossivel: ver ``_load_target_embed``).

Reusa a maquinaria DFlash/DSpark do backport (``DFlashQwen3Model``,
``Qwen3DSparkForCausalLM``). O que muda em relacao ao Qwen3 e' a numerica das
camadas (norma estilo Gemma -- ``(1 + w)`` --, ativacao swigluoai com beta=1,
RoPE parcial) e o loader, que desquantiza MXFP8 em bf16 na carga: o drafter tem
2,3 GB e em sm_86 nao ha kernel MXFP8 usavel nas ativacoes fp16 do motor.

Atencao ao cache de contexto do drafter (``precompute_and_store_context_kv``):
o caminho herdado usa ``ops.rms_norm``, que NAO aplica o ``(1 + w)`` da norma
Gemma. ``_build_fused_kv_buffers`` abaixo soma 1 nos dois vetores de norma
usados por esse caminho.
"""

import json
import os
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3DecoderLayer,
    DFlashQwen3Model,
    _get_dflash_fc_input_size,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.transformers_utils.config import set_default_rope_theta

from .model import MiniMAXGemmaRMSNorm

logger = init_logger(__name__)

_SCALE_SUFFIX = ".weight_scale_inv"
_EMBED_KEYS = (
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
)


def _dequant_mxfp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """F8_E4M3 + escala E8M0 por bloco de 32 no eixo K -> bf16.

    ``weight_scale_inv`` e' uint8 com vies 127 (valor = 2**(u - 127)) e forma
    [out, in/32]; o peso dequantizado e' ``w * scale``.
    """
    n, k = int(weight.shape[0]), int(weight.shape[-1])
    s = scale_inv.to(torch.float32)
    s = torch.exp2(s - 127.0)
    if s.shape[1] * 32 < k:  # cauda incompleta: completa com 1.0
        pad = torch.ones((n, (k + 31) // 32 - s.shape[1]), dtype=s.dtype)
        s = torch.cat([s, pad], dim=1)
    w = weight.to(torch.float32).reshape(n, -1, 32)
    w = w * s[:, : w.shape[1], None]
    return w.reshape(n, k).to(torch.bfloat16)


def _dequant_stream(weights: Iterable[tuple[str, torch.Tensor]]):
    """Materializa o stream e desquantiza os pesos MXFP8 (escala vem depois)."""
    items = list(weights)
    scales = {
        name[: -len(_SCALE_SUFFIX)]: t
        for name, t in items
        if name.endswith(_SCALE_SUFFIX)
    }
    n_deq = 0
    for name, t in items:
        if name.endswith(_SCALE_SUFFIX):
            continue
        if name.endswith(".weight"):
            scale = scales.get(name[: -len(".weight")])
            if scale is not None:
                t = _dequant_mxfp8(t, scale)
                n_deq += 1
        yield name, t
    if n_deq:
        logger.info("DSpark drafter: %d pesos MXFP8 desquantizados para bf16", n_deq)


def _load_target_embed(pack_path: str | None) -> torch.Tensor | None:
    """Tabela de embedding do alvo, lida do pack EXL3/alvo.

    O drafter do M3 nao traz ``embed_tokens`` e, com PP>1, o estagio que roda o
    drafter nao tem o embedding do alvo (ele mora no estagio 0), entao o
    caminho de compartilhamento do backport nao se aplica. Ler do disco custa
    ~2,4 GB por rank no load e deixa o drafter autocontido.
    """
    if not pack_path or not os.path.isdir(pack_path):
        return None
    files: list[str] = []
    index_path = os.path.join(pack_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        for key in _EMBED_KEYS:
            fname = weight_map.get(key)
            if fname:
                files.append(os.path.join(pack_path, fname))
                break
    if not files:
        single = os.path.join(pack_path, "model.safetensors")
        if os.path.exists(single):
            files.append(single)
    if not files:
        logger.warning("DSpark drafter: sem tabela de embedding em %s", pack_path)
        return None

    from safetensors import safe_open

    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for key in _EMBED_KEYS:
                if key in keys:
                    logger.info("DSpark drafter: embed_tokens do alvo vindo de %s", key)
                    return f.get_tensor(key)
    logger.warning("DSpark drafter: embed_tokens nao encontrado no pack do alvo")
    return None


class _FlatGemmaRMSNorm(MiniMAXGemmaRMSNorm):
    """Norma Gemma que aceita entrada >2-D (o kernel flashinfer so' ve 2-D).

    ``DFlashQwen3Attention`` aplica o qk-norm numa visao ``[T, cabecas, hd]``;
    o ``gemma_rmsnorm`` do flashinfer exige ``(tokens, hidden)`` e dispara
    device-side assert com 3-D. Achata, normaliza e devolve a forma original.
    """

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None or x.dim() == 2:
            return super().forward(x, residual)
        shape = x.shape
        return super().forward(x.reshape(-1, shape[-1])).view(shape)


class DFlashMiniMaxM3MLP(nn.Module):
    """MLP denso com a ativacao do M3: gate * sigmoid(alpha*gate) * (up+beta)."""

    def __init__(
        self,
        config,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        # swigluoai: beta e' 1.0 (o config do M3 omite o campo e o default da
        # classe de configuracao do backport tambem e' 1.0).
        self.act_fn = SiluAndMulWithClamp(
            swiglu_limit=config.swiglu_limit,
            alpha=config.swiglu_alpha,
            beta=getattr(config, "swiglu_beta", 1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DFlashMiniMaxM3DecoderLayer(DFlashQwen3DecoderLayer):
    """Camada do drafter: atencao (GQA + qk-norm Gemma + RoPE parcial) + MLP."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=10000000)

        dflash_config = getattr(config, "dflash_config", None) or {}
        add_swa_attention_sink_bias = dflash_config.get(
            "attention_sink_bias",
            getattr(config, "add_swa_attention_sink_bias", False),
        )
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        is_neox_style = getattr(config, "is_neox_style", True)

        # bf16/fp16 puro: o MXFP8 do checkpoint e' desquantizado no loader,
        # entao quant_config=None aqui e' proposital (nao herdar o do alvo nem
        # o nome "mxfp8" que o proprio checkpoint do drafter declara).
        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            add_swa_attention_sink_bias=add_swa_attention_sink_bias,
            sliding_window=sliding_window,
            causal=causal,
            is_neox_style=is_neox_style,
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=None,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": getattr(config, "partial_rotary_factor", 1.0),
            },
            prefix=f"{prefix}.self_attn",
        )
        # qk-norm estilo Gemma (use_gemma_norm == True tambem no qk-norm),
        # aplicado numa visao [T, cabecas, hd] -> precisa do wrapper achatado.
        self.self_attn.q_norm = _FlatGemmaRMSNorm(
            self.self_attn.head_dim, eps=config.rms_norm_eps
        )
        self.self_attn.k_norm = _FlatGemmaRMSNorm(
            self.self_attn.head_dim, eps=config.rms_norm_eps
        )

        ffn = (
            getattr(config, "dspark_ffn_hidden_size", None)
            or getattr(config, "dense_intermediate_size", None)
            or config.intermediate_size
        )
        self.mlp = DFlashMiniMaxM3MLP(
            config,
            intermediate_size=int(ffn),
            quant_config=None,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = MiniMAXGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiniMAXGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class DFlashMiniMaxM3Model(DFlashQwen3Model):
    """Backbone DFlash do M3 (5 camadas densas) + cabecas DSpark."""

    decoder_layer_cls = DFlashMiniMaxM3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        config = self.config
        # O drafter do M3 usa norma Gemma nas duas normas do backbone.
        self.hidden_norm = MiniMAXGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.norm = MiniMAXGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # O fc do contexto tambem e' bf16 puro (esta na ignored_layers do
        # checkpoint MXFP8): refaz sem o quant_config do drafter.
        if self.use_aux_hidden_state:
            self.fc = ReplicatedLinear(
                input_size=_get_dflash_fc_input_size(vllm_config),
                output_size=config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )

        # Cabecas DSpark (markov rank 256 + confidence 6400 = 6144 + 256).
        markov_rank = (
            getattr(config, "markov_rank", None)
            or getattr(config, "dspark_markov_rank", None)
            or 256
        )
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            int(markov_rank),
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=None,
            retain_weight_for_gather=(
                getattr(config, "dspark_draft_topk", None) is not None
            ),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", False):
            with_markov = getattr(config, "confidence_head_with_markov", False)
            input_dim = config.hidden_size
            if with_markov:
                input_dim += int(markov_rank)
            self.confidence_head = DSparkConfidenceHead(
                input_dim,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                with_markov=with_markov,
            )

    def _build_fused_kv_buffers(self) -> None:
        """Buffers do cache de contexto, com a norma Gemma aplicada.

        ``precompute_and_store_context_kv`` roda ``ops.rms_norm`` direto nos
        vetores de norma; a norma Gemma e' ``(1 + w)``, entao os dois vetores
        usados ali sao deslocados de 1 uma unica vez.
        """
        super()._build_fused_kv_buffers()
        # Ambos os vetores sao reconstruidos crus a cada chamada, entao o
        # deslocamento e' seguro (e' idempotente em relacao a super()).
        self._hidden_norm_weight = self._hidden_norm_weight + 1.0
        self._k_norm_weights = (self._k_norm_weights + 1.0).contiguous()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(_dequant_stream(weights))


class DSparkMiniMaxM3ForCausalLM(Qwen3DSparkForCausalLM):
    """Entrada de registro para o drafter do MiniMax-M3.1-Flash."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        # O config.json do drafter e' o formato multimodal (MiniMaxM3Config),
        # com os campos do backbone em text_config. As classes DFlash leem o
        # hf_config cru, entao fixa o text_config aqui uma unica vez. O
        # discriminador e' num_attention_heads: o wrapper VL carrega
        # hidden_size/num_hidden_layers no topo, mas nao as cabecas.
        hf_config = self.draft_model_config.hf_config
        if not hasattr(hf_config, "num_attention_heads"):
            text_config = (
                hf_config.get_text_config()
                if hasattr(hf_config, "get_text_config")
                else None
            )
            if text_config is not None and text_config is not hf_config:
                hf_config = text_config
                self.draft_model_config.hf_config = hf_config
        self.config = hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        self._target_pack = vllm_config.model_config.model

        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()
        self.model = DFlashMiniMaxM3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != self.target_vocab_size:
            # Vocabulario proprio: precisa da cabeca e do mapa d2t.
            self.lm_head = ParallelLMHead(
                self.config.draft_vocab_size,
                self.config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            # Vocabulario cheio: load_dspark_model aponta o lm_head do alvo
            # para ca' e apaga este. Nao instanciar evita 1,2 GiB/GPU de pico
            # durante a construcao -- o estagio PP que roda o drafter ja'
            # esta no limite dos 24 GiB.
            self.lm_head = None
            self.draft_id_to_target_id = None
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self._start_layer_id = target_layer_num

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Renomeia o checkpoint do M3 para o layout DFlash do backport.

        O arquivo usa ``language_model.model.dspark.layers.N.decoder_layer.*``
        (N = 0..4, o drafter) e ``final_norm``; o modelo monta a camada como
        ``layers.N.*`` -- sem o segmento ``decoder_layer``, e com ``start_layer_id``
        deslocando apenas o *prefixo* dos nomes de camada do KV cache, nao o
        caminho do state dict -- e chama a norma final de ``norm``. A
        desquantizacao MXFP8 acontece depois, no modelo. O ``embed_tokens`` nao
        existe no checkpoint: e' injetado do pack do alvo.
        """

        def _fix(items):
            n_embed = 0
            for name, t in items:
                n = name
                for pre in (
                    "language_model.model.dspark.",
                    "language_model.model.",
                    "model.dspark.",
                    "dspark.",
                ):
                    if n.startswith(pre):
                        n = n[len(pre):]
                        break
                n = n.replace(".decoder_layer.", ".", 1)
                if n.startswith("final_norm."):
                    n = "norm." + n[len("final_norm."):]
                if "embed_tokens" in n:
                    n_embed += 1
                yield n, t
            if not n_embed:
                embed = _load_target_embed(self._target_pack)
                if embed is not None:
                    yield "embed_tokens.weight", embed

        return super().load_weights(_fix(weights))
