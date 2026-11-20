#!/usr/bin/env python3
"""Roteamento no ponto de divergencia: o vLLM escolheu outros experts?

Le o `ffn_in` dumpado pelo vLLM e o gate BF16 do pack, aplica a formula do
oraculo (sigmoid(x@gate.T) + e_score_correction_bias, top-4, renormaliza) e
imprime, por posicao, o conjunto top-4 e as margens na fronteira 4/5.

  - margem grande -> o vLLM escolheu experts DIFERENTES do que o gate manda:
    o erro esta na orquestracao do MoE (dispatcher do plugin), nao no kernel.
  - margem minuscula (~1e-3) -> empate: o roteamento e' instavel por natureza
    e qualquer ruido troca o expert.

Uso: route_cmp.py --dump logs/dump_big --layer 18 --pos 10,100,205,209
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
import json
import pathlib

import numpy as np
from safetensors import safe_open

MODEL = MODEL_DIR
PFX = "language_model.model."


def load_tensor(key: str) -> np.ndarray:
    idx = json.load(open(pathlib.Path(MODEL) / "model.safetensors.index.json"))
    shard = idx["weight_map"][key]
    with safe_open(str(pathlib.Path(MODEL) / shard), framework="np") as f:
        return f.get_tensor(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--pos", default="10,100,205,209")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    L = a.layer
    poss = [int(p) for p in a.pos.split(",") if p]

    x = np.load(pathlib.Path(a.dump) / f"L{L:02d}_ffn_in.npz")["x"].astype(np.float64)
    gate = load_tensor(f"{PFX}layers.{L}.block_sparse_moe.gate.weight").astype(np.float64)
    bias = load_tensor(f"{PFX}layers.{L}.block_sparse_moe.e_score_correction_bias").astype(np.float64)
    print(f"x {x.shape}  gate {gate.shape}  bias {bias.shape}")

    s = 1.0 / (1.0 + np.exp(-(x @ gate.T)))
    sc = s + bias
    top4 = np.argsort(-sc, axis=1)[:, :4]
    srt = np.sort(-sc, axis=1)
    margin = -(srt[:, 4] - srt[:, 3])

    for p in poss:
        print(f"\npos {p}: top4={sorted(top4[p].tolist())}  margem(4o-5o)={margin[p]:.3e}")
        order = np.argsort(-sc[p])[: a.top]
        print("   " + "  ".join(f"{int(e)}:{sc[p][e]:.5f}" for e in order))
        w = s[p][top4[p]]
        w = w / w.sum()
        print("   pesos normalizados: " + "  ".join(f"{float(v):.4f}" for v in w))

    print(f"\nmargem minima no lote: {margin.min():.3e}  mediana: {np.median(margin):.3e}")
    print(f"posicoes com margem < 1e-3: {(margin < 1e-3).sum()} de {margin.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
