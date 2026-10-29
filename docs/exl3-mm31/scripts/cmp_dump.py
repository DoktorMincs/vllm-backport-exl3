#!/usr/bin/env python3
"""Compara dois dumps de tensor do M3 (VLLM_M3_DUMP) tag a tag.

Mostra cos e erro relativo por tag, na ordem do modelo, e aponta o primeiro
componente que diverge — e' assim que se separa "embed errado" de "atencao
errada" de "MoE errado" sem o oraculo.

Aceita arquivos com sufixo de rank (`.r0.npz`, gravado quando TP>1) e compara
so' as `--rows` primeiras linhas (o batch do vLLM vem paddado: 210 -> 256).

Uso: cmp_dump.py <dirA> <dirB> [--rank N] [--rows 210] [--tol 0.999] [--all]
"""

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
import re

import numpy as np

ORDER = ["embed", "normed_in", "attn_out", "residual", "ffn_in", "ffn_out"]
_RANK = re.compile(r"^(.+)\.r(\d+)$")


def scan(d):
    """tag base -> arquivo (prefere o menor rank; --rank escolhe outro)."""
    out = {}
    rank_of = {}
    for f in os.listdir(d):
        if not f.endswith(".npz"):
            continue
        base, rank = f[:-4], 0
        m = _RANK.match(base)
        if m:
            base, rank = m.group(1), int(m.group(2))
        if base not in out or rank < rank_of[base]:
            out[base], rank_of[base] = f, rank
    return out, rank_of


def ordered(tags):
    def key(t):
        m = re.match(r"L(\d+)_(.+)", t)
        if m:
            comp = m.group(2)
            return (int(m.group(1)), ORDER.index(comp) if comp in ORDER else 99)
        return (-1, 0)
    return sorted(tags, key=key)


def load(d, fname, rows):
    with np.load(os.path.join(d, fname)) as z:
        arr = z["x"] if "x" in z else z[list(z.keys())[0]]
    arr = arr.astype(np.float64)
    if rows:
        arr = arr[:rows]
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rank", type=int, default=None, help="rank a usar em B")
    ap.add_argument("--rows", type=int, default=210, help="0 = todas as linhas")
    ap.add_argument("--tol", type=float, default=0.999)
    ap.add_argument("--all", action="store_true", help="imprime todas as linhas")
    args = ap.parse_args()

    fa, _ = scan(args.a)
    fb, rb = scan(args.b)
    if args.rank is not None:
        # reconstroi o nome do arquivo do rank pedido
        for base in list(fb):
            cand = "%s.r%d.npz" % (base, args.rank)
            if os.path.exists(os.path.join(args.b, cand)):
                fb[base] = cand
    common = [t for t in ordered(fa) if t in fb]
    if not common:
        print("nenhuma tag em comum; A=%d tags B=%d tags" % (len(fa), len(fb)))
        return 2
    first_bad = None
    for t in common:
        x, y = load(args.a, fa[t], args.rows), load(args.b, fb[t], args.rows)
        if x.shape != y.shape:
            print("%-20s SHAPE %s vs %s" % (t, x.shape, y.shape))
            first_bad = first_bad or t
            continue
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if not np.isfinite(nx) or not np.isfinite(ny) or nx == 0 or ny == 0:
            print("%-20s NORMA invalida A=%g B=%g" % (t, nx, ny))
            first_bad = first_bad or t
            continue
        cos = float(np.sum(x * y) / (nx * ny))
        rel = float(np.linalg.norm(x - y) / ny)
        flag = "" if cos >= args.tol else "  <== DIVERGE"
        if args.all or cos < args.tol or t == common[0]:
            print("%-20s cos=%.6f relerr=%.4f%s" % (t, cos, rel, flag))
        if cos < args.tol and first_bad is None:
            first_bad = t
    print("\nprimeiro divergente: %s"
          % (first_bad or "nenhum (tudo >= %.4f)" % args.tol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
