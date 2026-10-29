#!/usr/bin/env python3
"""Request deterministico para comparar com o oraculo ref.py (mesmos token ids).

1. /tokenize de "The capital of France is" (confere contra os ids do ref.py);
2. /v1/completions com ESSES ids como prompt, max_tokens=1, greedy, logprobs=8
   -> a distribuicao do 1o token gerado, que o ref.py tambem produz.

Uso: request_ref.py [--port 3001] [--text "The capital of France is"]
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
import urllib.request

REF_IDS = [758, 5505, 300, 5969, 355]


def post(base: str, path: str, body: dict, timeout: float = 900.0):
    req = urllib.request.Request(
        f"http://{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--text", default="The capital of France is")
    ap.add_argument("--model", default="minimax-m3.1-flash")
    args = ap.parse_args()
    base = f"{args.host}:{args.port}"

    tok = post(base, "/tokenize", {"model": args.model, "prompt": args.text})
    ids = tok.get("tokens")
    print(f"/tokenize -> {ids}  (ref.py: {REF_IDS})  igual={ids == REF_IDS}")

    out = post(base, "/v1/completions", {
        "model": args.model,
        "prompt": ids if ids else REF_IDS,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 8,
        "echo": False,
    })
    ch = (out.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    toks = lp.get("tokens") or []
    tops = (lp.get("top_logprobs") or [{}])[0] or {}
    print(f"1o token: {toks!r}")
    print("top8:", sorted(tops.items(), key=lambda kv: -kv[1]))
    print("usage:", out.get("usage"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
