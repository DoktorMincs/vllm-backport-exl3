#!/usr/bin/env python3
"""Sondas de chat com a amostragem recomendada pelo modelo (temp/top_p).

O `generation_config.json` do pack diz `do_sample: true, temperature 1.0,
top_p 0.95` — greedy (temperature 0) nao e o modo previsto do modelo e entra em
repeticao. Esta sonda usa temp/top_p do config e imprime o texto inteiro.

Uso: chat_sample.py [--port 3001] [--ntok 200] [--temp 1.0] [--top-p 0.95]
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
import secrets
import urllib.request

PROBES = [
    ("pt-br", "Explique em português do Brasil, em no máximo 4 frases, por que o céu é azul."),
    ("codigo", "Escreva uma função Python chamada `n_primos` que devolve os n primeiros "
               "números primos, com docstring e tratamento de n<=0. Só o código."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--ntok", type=int, default=200)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    a = ap.parse_args()
    base = f"{a.host}:{a.port}"

    for nome, prompt in PROBES:
        nonce = secrets.token_hex(4)
        body = {
            "model": a.model, "max_tokens": a.ntok, "temperature": a.temp,
            "top_p": a.top_p,
            "messages": [
                {"role": "system", "content": f"Você é um assistente preciso. (sessão {nonce})"},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(f"http://{base}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.loads(r.read())
        ch = (out.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        print(f"\n=== {nome} (temp {a.temp} top_p {a.top_p}) "
              f"usage={out.get('usage')} finish={ch.get('finish_reason')}")
        print("reasoning:", (msg.get("reasoning_content") or "")[:400] or "(vazio)")
        print("content:", msg.get("content"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
