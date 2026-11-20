#!/usr/bin/env python3
"""Compara um dump do vLLM (prompt longo) com a referencia do ref_big.py.

O dump tem uma linha por token do batch (linha r = posicao r); a referencia tem
so as posicoes pedidas. Mostra cos(estado) por camada em cada posicao — se as
posicoes iniciais (10, 100) ja divergirem, o problema e de kernel/atencao em
escala; se so as ultimas divergirem, e acumulo/caos.

Uso: compare_big.py --ref logs/ref_big.npz --dump logs/dump_big [--every 1]
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
import pathlib

import numpy as np


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def load(dump: str, L: int, name: str, row: int):
    p = pathlib.Path(dump) / f"L{L:02d}_{name}.npz"
    if not p.exists():
        return None
    arr = np.load(p)["x"]
    if row >= arr.shape[0]:
        return None
    return arr[row].astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    z = np.load(args.ref)
    hs, pos = z["hs"], z["pos"].tolist()
    nL = hs.shape[0]
    print(f"ref: {nL} camadas, posicoes {pos}")
    head = "  L " + " ".join(f"{('pos' + str(p)):>12}" for p in pos)
    print(head)
    for L in range(0, nL, args.every):
        cells = []
        for j, p in enumerate(pos):
            res = load(args.dump, L, "residual", p)
            ffn = load(args.dump, L, "ffn_out", p)
            att = load(args.dump, L, "attn_out", p)
            if res is None or ffn is None:
                cells.append(f"{'-':>12}")
                continue
            st = res + ffn + (att if att is not None else 0.0)
            cells.append(f"{cos(st, hs[L][j]):>12.6f}")
        print(f"{L:>3} " + " ".join(cells), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
