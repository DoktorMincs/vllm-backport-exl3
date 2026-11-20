#!/usr/bin/env python3
"""Onde a camada 18 do vLLM se separa do oraculo? Atencao ou MoE?

Parte do estado L17 dumpado pelo vLLM (que bate com o oraculo a 1e-6) e roda a
camada 18 do oraculo peca por peca, comparando com os tensores dumpados:

  attn_out  <- attention(gemma_rms(h, in_norm))
  ffn_in    = gemma_rms(h + attn, post_norm)
  ffn_out   = ffn(ffn_in)

Se a atencao bate e o FFN nao, o problema esta no MoE da L18 (o primeiro layer
uniforme K3). Se a atencao ja diverge, e' a atencao esparsa em 210 tokens.

Uso: replay_components.py --dump logs/dump_big --ids logs/ids_big.json --layer 18 --dev cuda:1
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
import json
import pathlib

import numpy as np

REF = MM3_REF
MODEL = MODEL_DIR
PFX = "language_model.model."


def dump_state(dump: str, L: int, nrows: int) -> np.ndarray:
    def ld(name):
        p = pathlib.Path(dump) / f"L{L:02d}_{name}.npz"
        return np.load(p)["x"][:nrows].astype(np.float64) if p.exists() else 0.0
    return ld("residual") + ld("attn_out") + ld("ffn_out")


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def relerr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--pos", default="10,100,205,209")
    ap.add_argument("--dev", default="cuda:1")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("refmm3", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import torch

    raw = json.load(open(a.ids))
    ids = raw["ids"] if isinstance(raw, dict) else raw
    T = len(ids)
    L = a.layer
    poss = [int(p) for p in a.pos.split(",") if p]

    h_np = dump_state(a.dump, L - 1, T)
    ref = mod.Ref(pathlib.Path(a.model), a.dev, False)
    g = ref.g
    h = torch.from_numpy(np.ascontiguousarray(h_np, dtype=np.float32)).to(a.dev)
    pos = torch.arange(T, device=a.dev)
    eps = g.eps

    def dump(name, rows=T):
        p = pathlib.Path(a.dump) / f"L{L:02d}_{name}.npz"
        return np.load(p)["x"][:rows].astype(np.float64) if p.exists() else None

    lx = ref
    w = ref.layer_weights(L)
    xn = mod.gemma_rms(h, w["in_norm"], eps)
    att = mod.attention(g, w, xn, pos, g.sparse[L])

    att_np = att.detach().cpu().numpy().astype(np.float64)
    d_att = dump("attn_out")
    d_fin = dump("ffn_in")
    d_fout = dump("ffn_out")
    d_res = dump("residual")
    print(f"L{L} a partir do estado L{L-1} do dump ({T} linhas)", flush=True)
    print(f"  atencao : cos={cos(att_np, d_att):.6f} relerr={relerr(att_np, d_att):.3e}", flush=True)

    h2 = h + att
    fn = mod.gemma_rms(h2, w["post_norm"], eps)
    fn_np = fn.detach().cpu().numpy().astype(np.float64)
    print(f"  ffn_in  : cos={cos(fn_np, d_fin):.6f} relerr={relerr(fn_np, d_fin):.3e}", flush=True)

    fo = mod.ffn(lx, g, L, w, fn)
    fo_np = fo.detach().cpu().numpy().astype(np.float64)
    print(f"  ffn_out : cos={cos(fo_np, d_fout):.6f} relerr={relerr(fo_np, d_fout):.3e}", flush=True)
    h_np_l = h.detach().cpu().numpy().astype(np.float64)
    print(f"  residual: cos={cos(h_np_l, d_res):.6f} relerr={relerr(h_np_l, d_res):.3e}", flush=True)
    print("\npor posicao:", flush=True)
    print(f"{'pos':>5} {'cos_attn':>10} {'re_attn':>9} {'cos_ffnin':>10} {'re_ffnin':>9} "
          f"{'cos_ffnout':>11} {'re_ffnout':>10}", flush=True)
    for p in poss:
        print(f"{p:>5} {cos(att_np[p], d_att[p]):>10.6f} {relerr(att_np[p], d_att[p]):>9.2e} "
              f"{cos(fn_np[p], d_fin[p]):>10.6f} {relerr(fn_np[p], d_fin[p]):>9.2e} "
              f"{cos(fo_np[p], d_fout[p]):>11.6f} {relerr(fo_np[p], d_fout[p]):>10.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
