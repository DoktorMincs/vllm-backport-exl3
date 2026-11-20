#!/usr/bin/env python3
"""Compara o dump por camada do vLLM (VLLM_M3_DUMP) com o oraculo ref.py (60 camadas).

O ref.py grava `hs` = fluxo residual por camada (depois da camada). O vLLM grava,
por camada: normed_in, attn_out, residual (pos-atencao, pre-FFN), ffn_in, ffn_out;
o estado que sai da camada L e `residual_L + ffn_out_L`.

Uso: compare_full.py --ref ref_full.npz --dump DIR [--pos -1] [--top 8]
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
    ap.add_argument("--pos", type=int, default=-1,
                    help="posicao no oraculo (ref.py)")
    ap.add_argument("--row", type=int, default=None,
                    help="linha no dump do vLLM (default: --pos); o batch do "
                         "vLLM vem paddado, entao 4 = 5o token do prompt")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    row = args.pos if args.row is None else args.row

    z = np.load(args.ref, allow_pickle=False)
    hs = z["hs"] if "hs" in z.files else None
    if hs is None:
        print("!! ref sem fluxo residual (--no-hs?)")
        return 2
    print(f"ref: hs={hs.shape}  logits={z['logits'].shape if 'logits' in z.files else None}  ids={z['ids'].tolist()}")
    if "logits" in z.files:
        lg = z["logits"]
        top = np.argsort(-lg[args.pos])[: args.top]
        print(f"oraculo top{args.top} (pos {args.pos}): "
              + ", ".join(f"{int(t)}:{float(lg[args.pos][t]):.3f}" for t in top))

    files = sorted(glob.glob(os.path.join(args.dump, "L*_ffn_out.npz")))
    if not files:
        print(f"!! nenhum dump do vLLM em {args.dump}")
        return 2
    layers = sorted(int(os.path.basename(f).split("_")[0][1:]) for f in files)
    n_ref = hs.shape[0]
    print(f"dump: {len(layers)} camadas ({layers[0]}..{layers[-1]}); ref: {n_ref}")

    # mapa linha->posicao: o batch do vLLM vem paddado (5 tokens viram 16 linhas
    # no perfil). O batch e uma sequencia so, entao linha r = posicao r; as
    # linhas de padding nao casam (NaN). A diagonal e impressa para conferir.
    res0 = load(args.dump, layers[0], "residual")
    ffn0 = load(args.dump, layers[0], "ffn_out")
    att0 = load(args.dump, layers[0], "attn_out")
    if res0 is not None and ffn0 is not None:
        st0 = res0 + ffn0 + (att0 if att0 is not None else 0.0)
        npos = hs.shape[1]
        diag = [cos(st0[r], hs[r][layers[0]]) if r < npos else float("nan")
                for r in range(st0.shape[0])]
        print("cos diagonal (linha r <-> pos r): "
              + ", ".join(f"{r}:{c:.4f}" for r, c in enumerate(diag)))
        if args.row is None:
            args.row = min(npos, st0.shape[0]) - 1
    print(f"comparando: linha {args.row} do dump <-> posicao {args.pos} do oraculo")

    print(f"{'L':>3} {'cos(state,hs[L])':>17} {'cos(attn,dh)':>13} "
          f"{'|state|':>10} {'|hs[L]|':>10} {'ratio':>7}")
    first_bad = None
    for L in layers:
        res = load(args.dump, L, "residual")
        ffn = load(args.dump, L, "ffn_out")
        att = load(args.dump, L, "attn_out")
        if res is None or ffn is None:
            continue
        state = res[args.row] + ffn[args.row]
        if att is not None:
            # o `residual` dumpado e o fluxo ANTES da atencao: o add acontece
            # dentro do fused_allreduce_gemma_rms_norm seguinte, entao o estado
            # que sai da camada e residual + attn_out + ffn_out.
            state = state + att[args.row]
        c = cos(state, hs[L][args.pos]) if L < n_ref else float("nan")
        # a atencao isolada contra o delta do fluxo do oraculo (h[L] - h[L-1])
        ca = float("nan")
        if att is not None and 0 < L <= n_ref - 1:
            dh = hs[L][args.pos] - hs[L - 1][args.pos]
            ca = cos(att[args.row], dh)
        ratio = (np.linalg.norm(state) / max(np.linalg.norm(hs[L][args.pos]), 1e-12)
                 if L < n_ref else float("nan"))
        if c < 0.995 and first_bad is None:
            first_bad = L
        print(f"{L:>3} {c:>17.6f} {ca:>13.5f} {np.linalg.norm(state):>10.4g} "
              f"{np.linalg.norm(hs[L][args.pos]) if L < n_ref else float('nan'):>10.4g} "
              f"{ratio:>7.4f}")

    print()
    print(f"primeira camada com cos < 0.995: {first_bad}")
    return 0 if first_bad is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
