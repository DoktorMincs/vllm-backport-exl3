#!/usr/bin/env python3
"""Orcamento de VRAM por estagio de PP, lendo os headers dos safetensors.

Nao carrega pesos: le o JSON de header de cada shard (8 bytes de tamanho +
JSON com data_offsets) e soma os bytes reais de cada tensor. Inclui o
embed_tokens no estagio 0 e lm_head/norm final no ultimo. Ignora a torre de
visao (--language-model-only).

Uso: python -I pp_budget.py [--partition 7,5,5,7,7,7,7,5,5,5]
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
import collections
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="7,5,5,7,7,7,7,5,5,5")
    args = ap.parse_args()

    index = json.load(open(os.path.join(M, "model.safetensors.index.json")))
    weight_map: dict[str, str] = index["weight_map"]

    per_file: dict[str, dict[str, int]] = {}
    for name, fname in weight_map.items():
        if fname not in per_file:
            per_file[fname] = shard_sizes(os.path.join(M, fname))
    sizes = {n: per_file[f][n] for n, f in weight_map.items()}

    layers: dict[int, int] = collections.defaultdict(int)
    other: dict[str, int] = collections.defaultdict(int)
    for name, nbytes in sizes.items():
        parts = name.split(".")
        if "layers" in parts:
            i = parts.index("layers")
            try:
                layers[int(parts[i + 1])] += nbytes
            except (IndexError, ValueError):
                other["?".join(parts)] += nbytes
        elif name.startswith("vision_tower") or name.startswith("multi_modal") or name.startswith("patch_merge"):
            other["VISAO(ignorada)"] += nbytes
        elif name.endswith("embed_tokens.weight"):
            other["embed_tokens"] += nbytes
        elif "lm_head" in parts:
            other["lm_head"] += nbytes
        elif name.endswith("model.norm.weight"):
            other["norm_final"] += nbytes
        else:
            other[name] += nbytes

    print("Tensores por grupo (GiB):")
    for k, v in sorted(other.items(), key=lambda kv: -kv[1]):
        print(f"  {v/GIB:8.3f}  {k}")
    print(f"\nCamadas: {len(layers)}  total {sum(layers.values())/GIB:.2f} GiB")
    k_by_layer = {i: layers[i] / GIB for i in sorted(layers)}
    print("  5 maiores:", [(i, round(v, 3)) for i, v in sorted(k_by_layer.items(), key=lambda kv: -kv[1])[:5]])

    parts = [int(x) for x in args.partition.split(",")]
    assert sum(parts) == len(layers), (sum(parts), len(layers))
    print(f"\nParticao {args.partition} (soma {sum(parts)}):")
    start = 0
    totals = []
    for stage, count in enumerate(parts):
        idxs = range(start, start + count)
        b = sum(layers[i] for i in idxs)
        extra = []
        if stage == 0:
            b += other["embed_tokens"]
            extra.append(f"embed {other['embed_tokens']/GIB:.2f}")
        if stage == len(parts) - 1:
            b += other["lm_head"] + other["norm_final"]
            extra.append(f"lm_head {other['lm_head']/GIB:.2f}")
        totals.append(b)
        print(
            f"  stage {stage}: {count} camadas {start:2d}-{start+count-1:2d}  "
            f"{b/GIB:6.2f} GiB  {' '.join(extra)}"
        )
        start += count
    mx = max(totals)
    print(f"\nMaximo {mx/GIB:.2f} GiB  minimo {min(totals)/GIB:.2f} GiB  media {sum(totals)/len(totals)/GIB:.2f} GiB")
    print(f"Sobra em 24 GB a 0.93 ({24*0.93:.1f} GiB uteis): {(24*0.93 - mx/GIB):.2f} GiB no pior estagio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
