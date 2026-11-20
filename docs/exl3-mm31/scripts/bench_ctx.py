#!/usr/bin/env python3
"""Curva de contexto: TTFT e decode COM o contexto ja' carregado.

Diferente do speed_probe (prompt curto), aqui o prompt tem ~N tokens, entao o
decode paga a leitura de KV do contexto inteiro — que e' o que a curva precisa
mostrar. Reaproveita o gerador de prompt do bench_prefill.

Uso: bench_ctx.py --port 3001 -n 8192,32768,65536,131072 --ntok 64
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
import statistics
import sys
import time
import uuid

sys.path.insert(0, EXL3_ROOT + '/scripts')
from bench_prefill import build_prompt  # noqa: E402
import urllib.request  # noqa: E402

# Cauda que pede resposta longa: com "diga apenas ok" o modelo emite EOS em 2
# tokens e a coluna de decode vira poeira.
TAIL = (
    "\n\nIgnore a instrucao final do material acima. Escreva agora um "
    "paragrafo tecnico de cerca de 200 palavras resumindo os pontos citados, "
    "sem listas e sem titulo."
)


def one(host, port, model, target, ntok, temp, top_p, timeout):
    nonce = uuid.uuid4().hex
    body = {
        "model": model,
        "max_tokens": ntok,
        "temperature": temp,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": build_prompt(target, nonce, tail=TAIL)}],
    }
    req = urllib.request.Request(
        "http://%s:%d/v1/chat/completions" % (host, port),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    stamps, usage = [], {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    stamps.append(time.perf_counter() - t0)
    ptoks = usage.get("prompt_tokens")
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    if len(stamps) < 2:
        return {"target": target, "prompt_tokens": ptoks, "gen_tokens": len(stamps)}
    ttft = stamps[0]
    dec = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    return {
        "target": target,
        "prompt_tokens": ptoks,
        "cached": cached,
        "gen_tokens": len(stamps),
        "ttft_s": round(ttft, 2),
        "prefill_tok_s": round(ptoks / ttft) if ptoks else None,
        "decode_tok_s": round(dec, 2),
        "total_s": round(stamps[-1], 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("-n", "--targets", default="8192,32768,65536,131072")
    ap.add_argument("--ntok", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args()
    for t in [int(x) for x in args.targets.split(",") if x.strip()]:
        row = one(args.host, args.port, args.model, t, args.ntok,
                  args.temp, args.top_p, args.timeout)
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
