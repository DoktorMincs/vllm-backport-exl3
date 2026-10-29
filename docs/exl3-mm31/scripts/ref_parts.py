#!/usr/bin/env python3
"""Oraculo por componente: atencao e FFN por camada, posicao a posicao.

Reusa os primitivos do ref.py validado (attention/ffn/gemma_rms) sem tocar no
arquivo dele, e grava `attn` e `ffn` por camada para isolar onde o vLLM diverge
(atencao x MoE/densa) quando o fluxo residual ja bateu ate a camada N.

Uso: ref_parts.py --layers 20 --out logs/ref_parts.npz [--ids 758,5505,300,5969,355]
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
import importlib.util
import pathlib
import sys

import numpy as np

REF = MM3_REF
MODEL = MODEL_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ids", default="758,5505,300,5969,355")
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("refmm3", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import torch

    ref = mod.Ref(pathlib.Path(a.model), a.dev, False)
    g = ref.g
    ids = [int(x) for x in a.ids.replace(",", " ").split()]
    T = len(ids)

    # embeddings exatamente como o forward do ref.py
    rows = np.array(ref.st.raw(mod.PFX + "embed_tokens.weight").view(np.uint16)).reshape(g.vocab, g.dim)[ids]
    h = torch.from_numpy(rows.astype(np.int32) << 16).view(torch.float32).to(a.dev)
    pos = torch.arange(T, device=a.dev)

    hs, atts, ffns = [], [], []
    route = []
    for i in range(a.layers):
        w = ref.layer_weights(i)
        att = mod.attention(g, w, mod.gemma_rms(h, w["in_norm"], g.eps), pos, g.sparse[i])
        h = h + att
        ro: list = []
        f = mod.ffn(ref, g, i, w, mod.gemma_rms(h, w["post_norm"], g.eps), route_out=ro)
        h = h + f
        hs.append(h.cpu().numpy())
        atts.append(att.cpu().numpy())
        ffns.append(f.cpu().numpy())
        if ro:
            top, wt = ro[0]
            route.append(np.stack([top.numpy(), wt.numpy()]))  # [2, T, k]
            print(f"layer {i:2d} |h| {h.norm(dim=-1).mean().item():.2f} "
                  f"route[T-1]={top[-1].tolist()} w={[round(float(v), 4) for v in wt[-1]]}",
                  flush=True)
        else:
            route.append(np.zeros((2, T, g.k_exp)))
            print(f"layer {i:2d} |h| {h.norm(dim=-1).mean().item():.2f} (densa)", flush=True)
        del w
        torch.cuda.empty_cache()

    np.savez(a.out, ids=np.array(ids), hs=np.stack(hs), attn=np.stack(atts),
             ffn=np.stack(ffns), route=np.stack(route), pos=np.arange(T))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
