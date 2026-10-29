#!/usr/bin/env python3
"""Agulha no palheiro: mede se o KV em fp8 ainda entrega o fato enterrado.

Coloca uma frase com um codigo unico em uma profundidade escolhida de um
prompt de ~N tokens e pede o codigo. E' a medida de qualidade que o KV fp8
precisa passar antes de ser aceito (o usuario aceitou perda pequena, medida).

Uso: needle.py --port 3001 --size 60000 --depth 0.5 [--depth 0.9]
"""

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
import sys
import time
import uuid

sys.path.insert(0, EXL3_ROOT + '/scripts')
from bench_prefill import FILLER  # noqa: E402
import urllib.request  # noqa: E402

SECRET = "ZQ7X-4412"
NEEDLE = ("\n\nATENCAO: o codigo de acesso do cofre de seguranca da sala 12 "
          "e' %s. Guarde este codigo.\n\n" % SECRET)
QUESTION = ("\n\nQual e' o codigo de acesso do cofre de seguranca da sala 12? "
            "Responda apenas o codigo.")


def build(size_tokens, depth, nonce):
    target_chars = int(size_tokens * 3.55)
    parts = ["[ref %s]\n\n" % nonce]
    size = len(parts[0])
    i = 0
    while size < target_chars:
        block = "Item %d: %s" % (i, FILLER)
        if size >= target_chars * depth and NEEDLE not in "".join(parts[-1:]):
            block = NEEDLE + block
            depth = 2.0  # injeta uma vez so'
        parts.append(block)
        size += len(block)
        i += 1
    parts.append(QUESTION)
    return "".join(parts)


def ask(host, port, model, prompt, timeout):
    body = {
        "model": model,
        "max_tokens": 96,
        "temperature": 0.0,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "http://%s:%d/v1/chat/completions" % (host, port),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    text = (out["choices"][0].get("message") or {}).get("content") or ""
    return {
        "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
        "answer": text.strip()[:120],
        "hit": SECRET in text,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--size", type=int, default=60000)
    ap.add_argument("--depth", type=float, action="append", default=[])
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args()
    depths = args.depth or [0.1, 0.5, 0.9]
    for d in depths:
        prompt = build(args.size, d, uuid.uuid4().hex)
        row = ask(args.host, args.port, args.model, prompt, args.timeout)
        row["depth"] = d
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
