#!/usr/bin/env python3
"""Compara a distribuicao do 1o token: oraculo (ref_big) x vLLM (req_big_1.json).

Ambos no MESMO prompt longo. Se as duas distribuicoes concordarem, o prefill em
escala (atencao multi-bloco, indexer, kernels de MoE) esta fiel; se divergirem,
o erro esta antes da cabeca.

Uso: compare_first_token.py --ref logs/ref_big.npz --req logs/req_big_1.json [--pos 209]
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
import pathlib

import numpy as np
import torch
from tokenizers import Tokenizer

MODEL = MODEL_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--req", required=True)
    ap.add_argument("--pos", default="")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    z = np.load(a.ref)
    pos = z["pos"].tolist()
    j = pos.index(int(a.pos)) if a.pos else len(pos) - 1
    lg = z["logits"][j].astype(np.float64)
    lp = torch.log_softmax(torch.from_numpy(lg), -1).numpy()
    order = np.argsort(-lg)[:a.top]

    req = json.load(open(a.req))
    tops = (req.get("top_logprobs") or [{}])[0] or {}
    tk = Tokenizer.from_file(str(pathlib.Path(MODEL) / "tokenizer.json"))
    dec = lambda s: tk.decode([s])

    print(f"posicao {pos[j]}  (oraculo: logits -> log_softmax)")
    print("oraculo:", ", ".join(f"{int(t)}:{dec(int(t))!r}:{lp[t]:.3f}" for t in order))
    print("vLLM   :", ", ".join(f"{k!r}:{v:.3f}" for k, v in
                                sorted(tops.items(), key=lambda kv: -kv[1])))
    vllm_top = max(tops.items(), key=lambda kv: kv[1])[0] if tops else None
    print(f"top-1 oraculo={dec(int(order[0]))!r}  top-1 vLLM={vllm_top!r}")
    print(f"oraculo logprob do seu proprio top-1: {lp[int(order[0])]:.3f}")
    if vllm_top is not None:
        idx = tk.encode(vllm_top, add_special_tokens=False).ids
        if len(idx) == 1:
            print(f"oraculo logprob do top-1 do vLLM ({vllm_top!r}): {lp[idx[0]]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
