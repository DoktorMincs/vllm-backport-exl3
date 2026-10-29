#!/usr/bin/env python3
"""Compara o dump por camada do vLLM (logs/dump3/*.npz) com o oraculo ref.py.

O ref.py grava `hs` = fluxo residual por camada (convencao dele). O vLLM grava,
por camada: normed_in, attn_out, residual (pos-atencao, pre-FFN), ffn_in, ffn_out.
A convencao do decoder layer do vLLM deixa o FFN sem somar, entao o estado que
sai da camada L e `residual_L + ffn_out_L`.

Como a indexacao do `hs` do ref depende do modo (--layers N), o script testa
varios candidatos por camada e mostra o melhor casamento, sem presumir.

Uso: compare_ref.py --ref REF.npz --dump DIR [--pos -1]
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
import glob
import os

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def load(dump: str, L: int, name: str):
    p = os.path.join(dump, f"L{L:02d}_{name}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["x"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--pos", type=int, default=-1)
    args = ap.parse_args()

    z = np.load(args.ref, allow_pickle=False)
    print("ref keys:", {k: z[k].shape for k in z.files})
    hs = z["hs"] if "hs" in z.files else None
    if hs is None:
        print("!! ref sem fluxo residual")
        return 2
    print(f"hs shape={hs.shape}   (pos comparada: {args.pos})")

    files = sorted(glob.glob(os.path.join(args.dump, "L*_ffn_out.npz")))
    if not files:
        print(f"!! nenhum dump do vLLM em {args.dump}")
        return 2
    layers = sorted(int(os.path.basename(f).split("_")[0][1:]) for f in files)
    print(f"camadas com dump do vLLM: {layers[0]}..{layers[-1]} ({len(layers)})")

    emb = os.path.join(args.dump, "embed.npz")
    if os.path.exists(emb):
        e = np.load(emb)["x"]
        print("embed: " + "  ".join(
            f"cos(hs[{i}])={cos(e[args.pos], hs[i][args.pos]):.4f}"
            for i in range(min(2, hs.shape[0]))
        ))

    print(f"{'L':>3} {'cos(estado,hs[L])':>19} {'cos(estado,hs[L+1])':>20} "
          f"{'cos(attn,hs[L])':>17} {'|vllm|':>9} {'|ref|':>9} {'rms v/ref':>10}")
    first_bad = None
    for L in layers:
        res = load(args.dump, L, "residual")
        ffn = load(args.dump, L, "ffn_out")
        att = load(args.dump, L, "attn_out")
        if res is None or ffn is None:
            continue
        state = res[args.pos] + ffn[args.pos]
        c_same = cos(state, hs[L][args.pos]) if L < hs.shape[0] else float("nan")
        c_next = (cos(state, hs[L + 1][args.pos])
                  if L + 1 < hs.shape[0] else float("nan"))
        ca = cos(att[args.pos], hs[L][args.pos]) if att is not None else float("nan")
        best = max(c_same, c_next)
        if best < 0.99 and first_bad is None:
            first_bad = L
        refn = np.linalg.norm(hs[min(L, hs.shape[0] - 1)][args.pos])
        vrms = np.linalg.norm(state) / np.sqrt(state.shape[0])
        rrms = refn / np.sqrt(hs.shape[-1])
        print(f"{L:>3} {c_same:>19.6f} {c_next:>20.6f} {ca:>17.6f} "
              f"{np.linalg.norm(state):>9.4g} {refn:>9.4g} "
              f"{vrms / max(rrms, 1e-12):>10.3f}")
    print()
    print(f"primeira camada SEM casamento (cos < 0.99): {first_bad}")
    return 0 if first_bad is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
