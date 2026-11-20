#!/usr/bin/env python3
"""Prefill-only: manda ids e imprime a distribuicao da proxima token (top-k).

A token gerada com max_tokens=1 vem do PREFILL, entao serve para checar o
estado final do prefill sem passar pelo caminho de decode.

Uso: probe_ids.py --ids logs/ids_big.json [--steps 1] [--topk 8] [--temp 0]
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


def post(base: str, body: dict, timeout: float = 3600.0):
    req = urllib.request.Request(
        f"http://{base}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()
    base = f"{a.host}:{a.port}"

    raw = json.load(open(a.ids))
    ids = raw["ids"] if isinstance(raw, dict) else raw
    out = post(base, {"model": a.model, "prompt": ids, "max_tokens": a.steps,
                      "temperature": a.temp, "logprobs": a.topk,
                      "return_token_ids": True})
    ch = (out.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    print("prompt_tokens:", (out.get("usage") or {}).get("prompt_tokens"))
    for i, top in enumerate(lp.get("top_logprobs") or []):
        print(f"passo {i}: token={(lp.get('tokens') or ['?'])[i]!r}")
        for k, v in sorted((top or {}).items(), key=lambda kv: -kv[1]):
            print("   %-16r %7.3f" % (k, v))
    print("token_ids:", ch.get("token_ids"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
