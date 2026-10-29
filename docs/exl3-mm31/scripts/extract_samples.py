#!/usr/bin/env python3
"""Extrai amostras reais do pack MiniMax-M3.1 EXL3 (sem carregar os 184 GB).

Le o ledger `quantization_config.json -> tensor_storage` para descobrir quais
tensores existem por classe de bits e usa `safetensors.safe_open` (mmap/lazy)
para carregar apenas os tensores escolhidos.

Uso:
    python -I extract_samples.py <model_dir> <out_dir>

Saida: <out_dir>/samples.json + um .npz por amostra (suh/svh/mul1/trellis).
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


import json
import os
import sys

import numpy as np
from safetensors import safe_open

CLASSES = {
    "expert_K3": lambda n, b: "experts" in n and b == 3,
    "expert_K4": lambda n, b: "experts" in n and b == 4,
    "dense_K5": lambda n, b: ".mlp." in n and b == 5,
    "attn_K6": lambda n, b: "self_attn" in n and "proj" in n and b == 6,
    "lm_head_K6": lambda n, b: n.endswith("lm_head") and b == 6,
    "bf16_norm": lambda n, b: b is None and "norm" in n,
}


def main() -> int:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(model_dir, "quantization_config.json")) as fh:
        ledger = json.load(fh)["tensor_storage"]
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]

    picks: dict[str, tuple[str, int | None]] = {}
    for name, info in ledger.items():
        bits = info.get("bits_per_weight")
        for cls, pred in CLASSES.items():
            if cls in picks:
                continue
            if pred(name, bits):
                picks[cls] = (name, bits)

    manifest = {}
    for cls, (name, bits) in picks.items():
        suffixes = ["suh", "svh", "mul1", "trellis"] if bits is not None else [""]
        arrays = {}
        for suf in suffixes:
            key = f"{name}.{suf}" if suf else name
            shard = weight_map.get(key)
            if shard is None:
                print(f"[skip] {key}: ausente no index")
                continue
            with safe_open(os.path.join(model_dir, shard), framework="numpy") as f:
                arrays[suf or "weight"] = f.get_tensor(key)
        if not arrays:
            continue
        out = os.path.join(out_dir, f"{cls}.npz")
        np.savez(out, **arrays)
        manifest[cls] = {
            "tensor": name,
            "bits": bits,
            "file": os.path.basename(out),
            "shapes": {k: list(v.shape) for k, v in arrays.items()},
            "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        }
        mul1 = arrays.get("mul1")
        print(
            f"{cls:11} bits={bits} {name}\n"
            f"            shapes={manifest[cls]['shapes']}"
            + (f"\n            mul1={int(mul1.reshape(-1)[0])}" if mul1 is not None else "")
        )

    with open(os.path.join(out_dir, "samples.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\n{len(manifest)} amostras em {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
