#!/usr/bin/env python3
"""Sonda de completude crua (sem chat template) — o mesmo prompt do oraculo.

Manda "The capital of France is" como texto (ou ids), greedy, e imprime o texto
gerado. Serve para julgar coerencia e para comparar bf16 x fp16 x oraculo no
mesmo protocolo do ref.py.

Uso: complete_probe.py [--port 3001] [--max-tokens 48] [--text "..."]
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


def post(base: str, path: str, body: dict, timeout: float = 1800.0):
    req = urllib.request.Request(
        f"http://{base}{path}",
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
    ap.add_argument("--text", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args()
    base = f"{a.host}:{a.port}"

    out = post(base, "/v1/completions", {
        "model": a.model,
        "prompt": a.text,
        "max_tokens": a.max_tokens,
        "temperature": 0,
    })
    ch = (out.get("choices") or [{}])[0]
    print(f"prompt: {a.text!r}")
    print(f"texto : {ch.get('text')!r}")
    print(f"uso   : {out.get('usage')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
