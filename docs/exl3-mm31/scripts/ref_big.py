#!/usr/bin/env python3
"""Referencia do oraculo para um prompt longo (testa prefill/atencao em escala).

Le os ids de um JSON (`{"ids": [...]}`), roda o forward do ref.py numa lista de
posicoes de interesse e grava:
  hs      [L, P, dim]  fluxo residual nas posicoes `--pos`
  logits  [P, vocab]   logits nessas mesmas posicoes
  pos     [P]
  ids     [T]

Uso: ref_big.py --ids logs/ids_big.json --pos -1 --out logs/ref_big.npz
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ids", required=True, help='json {"ids": [...]} ou lista solta')
    ap.add_argument("--pos", default="-1", help="posicoes (virgula; negativo = do fim)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default="cuda:0")
    # --emulate reproduz os arredondamentos do engine de referencia (entradas FP16 nos
    # produtos EXL3, somas F32, KV em BF16). Serve para comparar o vLLM fp16 com
    # a MESMA numerica que a producao usa, em vez do F32 puro.
    ap.add_argument("--emulate", action="store_true")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("refmm3", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import torch

    raw = json.load(open(a.ids))
    ids = raw["ids"] if isinstance(raw, dict) else raw
    T = len(ids)
    pos = [int(p) % T for p in a.pos.split(",")]

    ref = mod.Ref(pathlib.Path(a.model), a.dev, a.emulate)
    print(f"T={T}  posicoes={pos}  emulate={a.emulate}", flush=True)
    logits, hs_list, _ = mod.forward(
        ref, ids, hs_at=torch.tensor(pos, device=a.dev), log=lambda s: print(s, flush=True)
    )
    hs = torch.stack([h for h in hs_list])          # [60, P, H]
    lg = logits[torch.tensor(pos, device=logits.device)]
    np.savez(a.out, ids=np.array(ids), pos=np.array(pos),
             hs=hs.detach().cpu().numpy(), logits=lg.detach().cpu().numpy())
    print("wrote", a.out, "hs", tuple(hs.shape), "logits", tuple(lg.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
