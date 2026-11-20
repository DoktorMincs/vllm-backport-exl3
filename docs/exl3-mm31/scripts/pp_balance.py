#!/usr/bin/env python3
"""Busca uma particao PP contigua e balanceada pelos bytes reais do pack.

Le os headers dos safetensors (nao carrega pesos) e, para uma dada
quantidade de estagios, escolhe cortes contiguos que minimizam o maximo por
estagio (programacao dinamica). Mostra o valor por GPU considerando TP.

Uso: pp_balance.py --stages 5 --tp 2
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
import os
import struct

M = MODEL_DIR
GIB = 1 << 30


def shard_sizes(path: str) -> dict[str, int]:
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        off = info.get("data_offsets")
        if off:
            out[name] = int(off[1]) - int(off[0])
    return out


def collect() -> tuple[list[int], dict[str, int]]:
    index = json.load(open(os.path.join(M, "model.safetensors.index.json")))
    weight_map: dict[str, str] = index["weight_map"]
    per_file: dict[str, dict[str, int]] = {}
    for name, fname in weight_map.items():
        if fname not in per_file:
            per_file[fname] = shard_sizes(os.path.join(M, fname))
    sizes = {n: per_file[f][n] for n, f in weight_map.items()}

    layers = [0] * 60
    other: dict[str, int] = {"embed": 0, "lm_head": 0, "norm": 0, "skip": 0}
    for name, nbytes in sizes.items():
        parts = name.split(".")
        if "layers" in parts:
            i = parts.index("layers")
            layers[int(parts[i + 1])] += nbytes
        elif name.startswith(("vision_tower", "multi_modal", "patch_merge")):
            other["skip"] += nbytes
        elif name.endswith("embed_tokens.weight"):
            other["embed"] += nbytes
        elif "lm_head" in parts:
            other["lm_head"] += nbytes
        elif name.endswith("model.norm.weight"):
            other["norm"] += nbytes
        else:
            other["skip"] += nbytes
    return layers, other


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=5)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--util", type=float, default=0.93)
    ap.add_argument("--ram-gib", type=float, default=24.0)
    args = ap.parse_args()

    layers, other = collect()
    n, s = len(layers), args.stages
    # Custo do estagio k com corte contiguo [i, j) : soma das camadas + extras
    pref = [0]
    for b in layers:
        pref.append(pref[-1] + b)

    def stage_bytes(i: int, j: int, k: int) -> int:
        total = pref[j] - pref[i]
        if k == 0:
            total += other["embed"]
        if k == s - 1:
            total += other["lm_head"] + other["norm"]
        return total

    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(s + 1)]
    cut = [[0] * (n + 1) for _ in range(s + 1)]
    best[0][0] = 0
    for k in range(1, s + 1):
        for j in range(1, n + 1):
            for i in range(0, j):
                if best[k - 1][i] == INF:
                    continue
                val = max(best[k - 1][i], stage_bytes(i, j, k - 1))
                if val < best[k][j]:
                    best[k][j] = val
                    cut[k][j] = i
    cuts = []
    j = n
    for k in range(s, 0, -1):
        i = cut[k][j]
        cuts.append((i, j))
        j = i
    cuts.reverse()

    print(f"TP={args.tp}  estagios={s}  max/GPU=%.2f GiB" % (best[s][n] / GIB / args.tp))
    parts = []
    for k, (i, j) in enumerate(cuts):
        b = stage_bytes(i, j, k)
        parts.append(j - i)
        print(
            f"  stage {k}: {j-i:2d} camadas {i:2d}-{j-1:2d}  "
            f"{b/GIB:6.2f} GiB total  {b/GIB/args.tp:6.2f} GiB/GPU"
        )
    usable = args.ram_gib * args.util
    head = usable - best[s][n] / GIB / args.tp
    print(f"particao: {','.join(map(str, parts))}")
    print(f"util {args.util}: {usable:.1f} GiB uteis -> sobram {head:.2f} GiB/GPU "
          f"para KV cache, ativacao e buffers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
