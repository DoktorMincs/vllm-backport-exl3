#!/usr/bin/env python3
"""Erro relativo por COMPONENTE (atencao e FFN) do vLLM contra o oraculo.

Usa logs/ref_parts*.npz (ref_parts.py: attn/ffn por camada) e o dump do vLLM
(L*_attn_out / L*_ffn_out). Mostra ||vllm-ref||/||ref|| por camada, que separa
"erro de atencao" de "erro de MoE/FFN" quando o fluxo residual ja bateu.

Uso: compare_parts.py --ref logs/ref_parts20.npz --dump logs/dump3 [--row 4] [--pos 4]
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

import numpy as np


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / max(nb, 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--pos", type=int, default=4)
    args = ap.parse_args()

    z = np.load(args.ref)
    att_r, ffn_r = z["attn"], z["ffn"]
    nL = att_r.shape[0]
    print(f"ref: {nL} camadas, pos {args.pos}; dump row {args.row}")
    print(f"{'L':>3} {'relerr(attn)':>13} {'relerr(ffn)':>12} "
          f"{'|attn|':>10} {'|ffn|':>10}")
    for L in range(nL):
        pa = os.path.join(args.dump, f"L{L:02d}_attn_out.npz")
        pf = os.path.join(args.dump, f"L{L:02d}_ffn_out.npz")
        if not (os.path.exists(pa) and os.path.exists(pf)):
            continue
        av = np.load(pa)["x"][args.row].astype(np.float64)
        fv = np.load(pf)["x"][args.row].astype(np.float64)
        ar = att_r[L][args.pos].astype(np.float64)
        fr = ffn_r[L][args.pos].astype(np.float64)
        print(f"{L:>3} {relerr(av, ar):>13.3e} {relerr(fv, fr):>12.3e} "
              f"{np.linalg.norm(ar):>10.4g} {np.linalg.norm(fr):>10.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
