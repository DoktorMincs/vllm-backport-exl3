#!/usr/bin/env python3
"""Ensaio a seco (sem servir): monta EngineConfig + quant config EXL3 do M3.

Exercita exatamente o caminho que o api_server usa antes de tocar a GPU em
serio: parse do config HF -> arquitetura -> QuantizationConfig do plugin ->
particao de PP. Nenhum peso e lido, nenhuma VRAM e alocada.

Uso: python -I dryrun_config.py [--tp 1] [--pp 10] [--maxlen 8192]
"""
from __future__ import annotations

import os as _os

EXL3_ROOT = _os.environ.get("EXL3_PORT", _os.path.expanduser("~/exl3-port"))
MODEL_DIR = _os.environ.get(
    "MODEL_DIR",
    _os.path.expanduser("~/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw"),
)
MM3_REF = _os.environ.get("MM3_REF", _os.path.expanduser("~/mm3-oracle/ref.py"))
VLLM_REPO = _os.environ.get("VLLM_REPO", _os.path.expanduser("~/vllm-backport"))


import argparse
import os
import sys
import traceback

MODEL = MODEL_DIR

fails: list[str] = []


def step(name: str, fn):
    try:
        value = fn()
        print(f"OK    {name}: {value}")
        return value
    except Exception as exc:  # noqa: BLE001
        fails.append(name)
        print(f"FALHA {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=6)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=10)
    ap.add_argument("--maxlen", type=int, default=8192)
    args = ap.parse_args()

    if args.pp == 10 and args.tp == 1:
        os.environ.setdefault("VLLM_PP_LAYER_PARTITION", "7,5,5,7,7,7,7,5,5,5")
    os.environ.setdefault("VLLM_EXL3_MOE_KERNEL", "native")
    os.environ.setdefault("VLLM_EXL3_Q4KV", "0")
    os.environ.setdefault("VLLM_EXL3_INDEXER_MXFP8", "0")
    os.environ.setdefault("VLLM_EXL3_MINIMAX_M3", "1")

    def _env():
        from vllm.plugins import load_general_plugins

        load_general_plugins()
        return "plugins carregados"

    step("load_general_plugins", _env)

    def _engine():
        from vllm.engine.arg_utils import EngineArgs

        return EngineArgs(
            model=MODEL,
            served_model_name="minimax-m3.1-flash",
            quantization="exl3",
            tensor_parallel_size=args.tp,
            pipeline_parallel_size=args.pp,
            max_model_len=args.maxlen,
            gpu_memory_utilization=0.93,
            enforce_eager=True,
            language_model_only=True,
            trust_remote_code=False,
        )

    engine_args = step("EngineArgs", _engine)
    if engine_args is None:
        return 1

    cfg = step("create_engine_config", engine_args.create_engine_config)
    if cfg is None:
        return 1

    mc = cfg.model_config
    step(
        "model",
        lambda: f"arch={mc.architectures} hf={type(mc.hf_config).__name__} "
        f"text={type(mc.hf_text_config).__name__} maxlen={mc.max_model_len} "
        f"quant={mc.quantization} dtype={mc.dtype}",
    )
    step(
        "world",
        lambda: f"tp={cfg.parallel_config.tensor_parallel_size} "
        f"pp={cfg.parallel_config.pipeline_parallel_size} "
        f"world={cfg.parallel_config.world_size}",
    )

    def _text():
        t = mc.hf_text_config
        return (
            f"layers={t.num_hidden_layers} hidden={t.hidden_size} "
            f"heads={t.num_attention_heads}/{t.num_key_value_heads} "
            f"act={t.hidden_act} moe_freq_sum={sum(t.moe_layer_freq)} "
            f"sparse_dims={t.sparse_attention_config.get('sparse_index_dim')} "
            f"topk_blocks={t.sparse_attention_config.get('sparse_topk_blocks')}"
        )

    step("text_config", _text)
    step(
        "multimodal",
        lambda: (
            None
            if mc.multimodal_config is None
            else f"language_model_only={mc.multimodal_config.language_model_only} "
            f"limit_per_prompt={mc.multimodal_config.limit_per_prompt}"
        ),
    )

    def _arch():
        from vllm.model_executor.models.registry import ModelRegistry

        reg = ModelRegistry
        arch = mc.architectures[0]
        supported = arch in reg.get_supported_archs()
        cls = None
        try:  # API varia entre bases (0.28: resolve_model_cls(archs, model_config))
            cls = reg.resolve_model_cls([arch], mc)[0]
        except Exception:  # noqa: BLE001
            try:
                cls = reg.resolve_model_cls([arch])[0]
            except Exception:  # noqa: BLE001
                cls = None
        if cls is None:
            return f"{arch} supported={supported} (classe nao resolvida pela API deste build)"
        return (
            f"{cls.__module__}.{cls.__name__} supported={supported} "
            f"SupportsPP={getattr(cls, 'supports_pp', '?')}"
        )

    step("get_model_architecture", _arch)

    def _qc():
        from vllm.model_executor.model_loader.weight_utils import get_quant_config
        from vllm.config.load import LoadConfig

        qc = get_quant_config(mc, LoadConfig())
        keys = sorted(k for k in vars(qc) if not k.startswith("_"))
        vals = {k: vars(qc)[k] for k in keys if isinstance(vars(qc)[k], (int, str, float, bool, type(None)))}
        nr = getattr(qc, "non_routed_exl3", None) or {}
        mods = nr.get("modules", []) if isinstance(nr, dict) else []
        return (
            f"{type(qc).__module__}.{type(qc).__name__} escalares={vals} "
            f"non_routed_modules={len(mods)} {sorted(mods)} "
            f"layer_bits={nr.get('layer_bits') if isinstance(nr, dict) else None}"
        )

    step("get_quant_config(exl3)", _qc)

    def _pp():
        from vllm.distributed.utils import get_pp_indices

        n = mc.hf_text_config.num_hidden_layers
        parts = [get_pp_indices(n, r, args.pp) for r in range(args.pp)]
        return " ".join(f"{a}-{b}" for a, b in parts) + f" (soma={sum(b - a for a, b in parts)})"

    step("pp_indices", _pp)

    print("\nRESULTADO:", "OK" if not fails else f"FALHAS: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
