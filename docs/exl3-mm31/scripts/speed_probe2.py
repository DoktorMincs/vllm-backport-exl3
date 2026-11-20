#!/usr/bin/env python3
"""Decode/TTFT por streaming, contando TOKENS (nao chunks).

O speed_probe.py conta chunks SSE com conteudo. Com speculative decoding o
servidor pode mandar varios tokens aceitos no mesmo chunk, entao a contagem
por chunk subestima a taxa (foi o que aconteceu em 27/10: 60 chunks para 128
tokens). Aqui o probe pede `stream_options.include_usage` e usa o
completion_tokens final; o denominador e' o intervalo entre o primeiro e o
ultimo chunk de conteudo.

Uso: speed_probe2.py [--port 3001] [--ntok 128] [--runs 3]
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
import time
import urllib.request
import uuid

PROMPT = (
    "Explique em um paragrafo tecnico como funciona um modelo autoregressivo "
    "decoder-only, citando atencao, MLP e conexoes residuais."
)


def one_run(host, port, model, ntok, temp):
    body = {
        "model": model,
        "max_tokens": ntok,
        "temperature": temp,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {
                "role": "system",
                "content": "Voce e um engenheiro. nonce=" + uuid.uuid4().hex,
            },
            {"role": "user", "content": PROMPT},
        ],
    }
    req = urllib.request.Request(
        "http://%s:%d/v1/chat/completions" % (host, port),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    usage_tokens = None
    text = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                usage_tokens = int(usage["completion_tokens"])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                # Com --reasoning-parser o inicio da geracao vem em `reasoning`
                # (ou `reasoning_content`), nao em `content`. Contar so `content`
                # perde o bloco de raciocinio inteiro (probe reportava "sem tokens").
                piece = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter() - t0
                    chunks += 1
                    text.append(piece)
                    if first is None:
                        first = now
                    last = now
    if first is None or last is None:
        return None
    n = usage_tokens if usage_tokens else chunks
    return {
        "tokens": n,
        "chunks": chunks,
        "ttft_s": first,
        "gen_s": last - first,
        "decode_tok_s": (n - 1) / (last - first) if last > first else None,
        "text": "".join(text)[:150].replace("\n", " "),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--ntok", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    rates = []
    for i in range(args.runs):
        out = one_run(args.host, args.port, args.model, args.ntok, args.temp)
        if not out:
            print("run %d: sem tokens" % i)
            continue
        rates.append(out["decode_tok_s"])
        print(
            "run %d: tokens=%d chunks=%d ttft=%.2fs gen=%.2fs decode=%.2f tok/s"
            % (
                i,
                out["tokens"],
                out["chunks"],
                out["ttft_s"],
                out["gen_s"],
                out["decode_tok_s"],
            )
        )
        print("   inicio: %s" % out["text"])
    if rates:
        print("media decode: %.2f tok/s (%d runs)" % (sum(rates) / len(rates), len(rates)))


if __name__ == "__main__":
    main()
