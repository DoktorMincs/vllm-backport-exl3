#!/usr/bin/env python3
"""Extrai tensores EXL3 especificos do pack (mmap, sem carregar os 184 GB).

`extract_samples.py` pega o PRIMEIRO tensor de cada classe — o que fez a amostra
K3 cair sempre em `layers.17.experts.0.w2`. Para testar o caminho `w1/w3` em K3
(camadas 18..45, que so aparecem fora da L17 mixed-K) e' preciso escolher o nome.

Uso: extract_one.py <model_dir> <out_dir> <prefixo> [<prefixo> ...]
     ex.: extract_one.py $MODEL_DIR/... artifacts/samples_k3w \\
            language_model.model.layers.20.block_sparse_moe.experts.0.w1
Saida: <out_dir>/samples.json + um .npz por tensor (suh/svh/mul1/trellis).
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


def main() -> int:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    prefixes = sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as fh:
        index = json.load(fh)
    weight_map = index["weight_map"]

    manifest = {}
    for pfx in prefixes:
        arrays = {}
        shapes = {}
        dtypes = {}
        for suf in ("suh", "svh", "mul1", "trellis"):
            key = f"{pfx}.{suf}"
            shard = weight_map.get(key)
            if shard is None:
                raise SystemExit(f"sem {key} no indice")
            with safe_open(os.path.join(model_dir, shard), framework="np") as f:
                arr = f.get_tensor(key)
            arrays[suf] = arr
            shapes[suf] = list(arr.shape)
            dtypes[suf] = str(arr.dtype)
        name = pfx.split("layers.")[-1].replace(".", "_")
        fn = f"{name}.npz"
        np.savez(os.path.join(out_dir, fn), **arrays)
        manifest[name] = {"tensor": pfx, "file": fn, "shapes": shapes, "dtypes": dtypes}
        print(f"{name:40} trellis={shapes['trellis']} K={shapes['trellis'][-1]//16}", flush=True)

    with open(os.path.join(out_dir, "samples.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("wrote", os.path.join(out_dir, "samples.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
