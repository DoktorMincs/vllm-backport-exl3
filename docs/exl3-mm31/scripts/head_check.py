#!/usr/bin/env python3
"""Confere a cabeca (norm final + lm_head) de um dump do vLLM contra o oraculo.

Duas perguntas, uma por modo:

  --mode state   o `final_normed` do vLLM ja' esta' certo? (aplica o lm_head
                 do oraculo sobre ele e compara com os logits do oraculo)
  --mode local   os logits *locais* de cada rank batem com a fatia de vocab
                 que aquele rank deveria cobrir?

Referencia: `ref_big.npz` (`logits` [P, vocab], `pos` [P]).

Uso:
  head_check.py --ref logs/ref_big.npz --pos 209 --mode state \
                --state logs/dump_tp2h2/final_normed.r8.npz --row 209
  head_check.py --ref logs/ref_big.npz --pos 209 --mode local \
                --local logs/dump_tp2h2/logits_local.r8.npz --vocab-split 100032
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


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=EXL3_ROOT + '/logs/ref_big.npz')
    ap.add_argument("--pos", type=int, default=209)
    ap.add_argument("--mode", choices=["state", "local"], required=True)
    ap.add_argument("--state", default="")
    ap.add_argument("--local", default="")
    ap.add_argument("--row", type=int, default=209)
    ap.add_argument("--vocab-split", type=int, default=100032)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    z = np.load(a.ref)
    idx = list(z["pos"]).index(a.pos)
    ref_logits = z["logits"][idx].astype(np.float64)
    top = np.argsort(-ref_logits)[:5]
    print(f"oraculo pos{a.pos} top5: "
          + ", ".join(f"{int(i)}:{ref_logits[i]:.2f}" for i in top))

    if a.mode == "state":
        h = np.load(a.state)["x"][a.row].astype(np.float64)
        print(f"final_normed row {a.row}: |h|={np.linalg.norm(h):.4f}")
        spec = importlib.util.spec_from_file_location("refmm3", REF)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import torch
        ref = mod.Ref(pathlib.Path(a.model), a.dev, False)
        head = ref.W("language_model.lm_head")
        ht = torch.from_numpy(np.ascontiguousarray(h, dtype=np.float32))[None].to(a.dev)
        lg = mod.lin(ht, head).float().cpu().numpy()[0]
        del head
        torch.cuda.empty_cache()
    else:
        lg = np.load(a.local)["x"]
        lg = lg.reshape(-1, lg.shape[-1])[0].astype(np.float64)
        lo = a.rank * a.vocab_split
        hi = min(lo + lg.shape[0], ref_logits.shape[0])
        ref_logits = ref_logits[lo:hi]
        print(f"fatia do rank {a.rank}: [{lo}:{hi}] ({hi - lo} colunas)")

    lg = np.asarray(lg, np.float64).ravel()
    n = min(lg.size, ref_logits.size)
    lg, rl = lg[:n], ref_logits[:n]
    print(f"cos={cos(lg, rl):.6f}  |vllm|={np.linalg.norm(lg):.3f} "
          f"|oraculo|={np.linalg.norm(rl):.3f}")
    print(f"max vllm={lg.max():.3f} (id {int(np.argmax(lg))})  "
          f"max oraculo={rl.max():.3f} (id {int(np.argmax(rl))})")
    ot = np.argsort(-lg)[:5]
    print("vllm top5: " + ", ".join(f"{int(i)}:{lg[i]:.2f}" for i in ot))
    # erro relativo por bloco de 128 colunas: mostra se o defeito e' local
    # (uma faixa) ou global (shard inteiro errado).
    if n >= 128:
        blocks = [(b, float(np.linalg.norm(lg[b:b + 128] - rl[b:b + 128])
                            / max(np.linalg.norm(rl[b:b + 128]), 1e-9)))
                  for b in range(0, n - 127, 128)]
        worst = sorted(blocks, key=lambda x: -x[1])[:3]
        print("piores blocos de 128 (offset, relerr): "
              + ", ".join(f"{b}:{e:.3f}" for b, e in worst))
        print(f"blocos: {len(blocks)}  relerr mediano={np.median([e for _, e in blocks]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
