#!/usr/bin/env python3
"""Orcamento de VRAM por estagio PP para o pack vLLM do DS4.1 (EXL3 + overlay).

Le os headers dos safetensors (nao carrega pesos) do pack convertido e mostra,
para uma particao contigua, os bytes por GPU. O que o vLLM *nao* carrega e'
zerado aqui: o mapper descarta mtp/vision/aligner/image_* e as tabelas do
Engram vem do disco (register_skip_weight_names).

Uso: pack_budget.py [--pack DIR] [--stages 5] [--tp 2] [--util 0.96]
                    [--gpu-gib 24] [--partition 8,8,9,8,7]
"""
from __future__ import annotations

import argparse
import json
import os
import struct

GIB = 1 << 30
DROP_PREFIX = ("mtp.", "vision.", "aligner.", "image_")
DROP_SUBSTR = (".engram.embed.",)


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


def collect(pack: str, tp: int) -> tuple[dict[int, int], dict[str, int]]:
    idx = json.load(open(os.path.join(pack, "model.safetensors.index.json")))
    wmap: dict[str, str] = idx["weight_map"]
    files: dict[str, dict[str, int]] = {}
    layers: dict[int, int] = {}
    other = {"embed": 0, "head": 0, "norm": 0, "replicated": 0, "skipped": 0}
    dropped = 0
    for name, fname in wmap.items():
        if name.startswith(DROP_PREFIX) or any(s in name for s in DROP_SUBSTR):
            dropped += shard_sizes_file(files, pack, fname, name)
            continue
        nbytes = shard_sizes_file(files, pack, fname, name)
        parts = name.split(".")
        if "layers" in parts:
            idx_i = parts.index("layers")
            layer = int(parts[idx_i + 1])
            layers[layer] = layers.get(layer, 0) + nbytes
        elif name.startswith("head.") or name.startswith("lm_head."):
            other["head"] += nbytes
        elif name.startswith("embed."):
            other["embed"] += nbytes
        elif name.startswith("norm.") or name.startswith("hc_expand"):
            other["norm"] += nbytes
        else:
            other["replicated"] += nbytes
    return layers, other, dropped


OVERLAY_SIZES = {
    "wo_a": 8 * 1024 * 4096 * 2,      # [8*1024, 4096] bf16
    "head": 129280 * 5120 * 2,        # [vocab, hidden] bf16
}


def shard_sizes_file(cache: dict, pack: str, fname: str, name: str) -> int:
    path = os.path.join(pack, fname)
    if not os.path.exists(path) and fname.endswith("vllm-overlay.safetensors"):
        return OVERLAY_SIZES["head"] if name == "head.weight" else OVERLAY_SIZES["wo_a"]
    if fname not in cache:
        cache[fname] = shard_sizes(path)
    return cache[fname].get(name, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pack",
        default=os.environ.get(
            "OURS_VLLM", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-1017-vllm")
        ),
    )
    ap.add_argument("--stages", type=int, default=5)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--util", type=float, default=0.96)
    ap.add_argument("--gpu-gib", type=float, default=24.0)
    ap.add_argument("--partition", default="")
    ap.add_argument("--reserve-gib", type=float, default=0.35,
                    help="contexto CUDA + driver por GPU")
    args = ap.parse_args()

    layers, other, dropped = collect(args.pack, args.tp)
    n = max(layers) + 1
    per_layer = [layers.get(i, 0) for i in range(n)]
    total_layer = sum(per_layer)

    print(f"camadas {n}: {total_layer/GIB:.2f} GiB (todas)  "
          f"media {total_layer/n/GIB:.3f} GiB/camada")
    print(f"head {other['head']/GIB:.2f}  embed {other['embed']/GIB:.2f}  "
          f"norm/etc {other['norm']/GIB:.2f}  replicado {other['replicated']/GIB:.3f}")
    print(f"descartado pelo mapper (mtp/vision/aligner/engram-embed): {dropped/GIB:.2f} GiB")

    if args.partition:
        parts = [int(x) for x in args.partition.split(",")]
        assert sum(parts) == n, f"particao soma {sum(parts)} != {n}"
        cuts, acc = [], 0
        for p in parts:
            cuts.append((acc, acc + p))
            acc += p
    else:
        # DP: minimiza o maximo por estagio
        pref = [0]
        for b in per_layer:
            pref.append(pref[-1] + b)
        s = args.stages
        INF = float("inf")
        best = [[INF] * (n + 1) for _ in range(s + 1)]
        cut = [[0] * (n + 1) for _ in range(s + 1)]
        best[0][0] = 0
        for k in range(1, s + 1):
            for j in range(1, n + 1):
                for i in range(0, j):
                    if best[k - 1][i] == INF:
                        continue
                    val = max(best[k - 1][i], pref[j] - pref[i])
                    if val < best[k][j]:
                        best[k][j] = val
                        cut[k][j] = i
        cuts, j = [], n
        for k in range(s, 0, -1):
            i = cut[k][j]
            cuts.append((i, j))
            j = i
        cuts.reverse()
        parts = [j - i for i, j in cuts]

    usable = args.gpu_gib * args.util - args.reserve_gib
    worst = 0.0
    for k, (i, j) in enumerate(cuts):
        b = sum(per_layer[i:j])
        extra = other["head"] + other["norm"]
        if k == 0:
            extra += other["embed"]
        per_gpu = (b + extra) / args.tp
        worst = max(worst, b / args.tp)
        print(
            f"  stage {k}: {j-i:2d} camadas {i:2d}-{j-1:2d}  "
            f"{(b+extra)/GIB:6.2f} GiB  {per_gpu/GIB:6.2f} GiB/GPU  "
            f"-> sobra {usable - per_gpu/GIB:6.2f} GiB p/ KV+ativacao"
        )
    print(f"particao: {','.join(map(str, parts))}")
    print(f"util {args.util} em {args.gpu_gib:.0f} GiB = {usable:.2f} GiB uso maximo/GPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
