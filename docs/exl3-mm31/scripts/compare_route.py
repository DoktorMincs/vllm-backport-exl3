#!/usr/bin/env python3
"""Compara o roteamento do vLLM (dump do plugin) com o route_out do oraculo.

O plugin grava <dir>/<prefix>.npz com ids [T,k] e weights [T,k] por camada
(VLLM_EXL3_DUMP_ROUTE + portao ARM). O oraculo (ref_parts.py) grava
route [L,2,T,k] = (top ids, pesos) por camada.

Uso: compare_route.py --ref logs/ref_parts20r.npz --dump logs/dump3 [--row 4] [--pos 4]
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
import re

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--pos", type=int, default=4)
    args = ap.parse_args()

    z = np.load(args.ref)
    route = z["route"]  # [L, 2, T, k]
    files = glob.glob(os.path.join(args.dump, "*layers.*.npz"))
    print(f"arquivos de rota no dump: {len(files)}")
    by_layer = {}
    for f in files:
        m = re.search(r"layers\.(\d+)\.", os.path.basename(f))
        if m:
            by_layer[int(m.group(1))] = f
    print(f"{'L':>3} {'ids vLLM':<26} {'ids oraculo':<26} {'==':>3} "
          f"{'w vLLM':<30} {'w oraculo':<30}")
    n_same = 0
    for L in sorted(by_layer):
        if L >= route.shape[0]:
            continue
        d = np.load(by_layer[L])
        ids = d["ids"][args.row]
        w = d["weights"][args.row]
        top = route[L][0][args.pos].astype(np.int64)
        wt = route[L][1][args.pos].astype(np.float64)
        if top.shape[0] == 1 and int(top[0]) == 0 and route[L][1].max() == 0:
            continue  # camada densa
        same = set(ids.tolist()) == set(top.tolist())
        n_same += int(same)
        print(f"{L:>3} {str(sorted(ids.tolist())):<26} {str(sorted(top.tolist())):<26} "
              f"{str(same):>5} "
              f"{str([round(float(x), 3) for x in w]):<30} "
              f"{str([round(float(x), 3) for x in wt]):<30}")
    print(f"\ncamadas com o MESMO conjunto de experts: {n_same}/{len(by_layer)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
