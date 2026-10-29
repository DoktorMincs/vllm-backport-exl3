#!/usr/bin/env python3
"""Mede TTFT e velocidade de decode por streaming (SSE).

Manda uma completion de chat com max_tokens fixo, le os chunks SSE e mede:
  - TTFT: do envio ate o primeiro token;
  - decode: (tokens-1) / (ultimo - primeiro token);
  - total: instante do ultimo token desde o envio.
O prompt leva um nonce para nao reaproveitar prefix cache.

Uso: speed_probe.py [--port 3001] [--ntok 128] [--runs 2]
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


def one_run(host, port, model, ntok, temp, top_p):
    body = {
        "model": model,
        "max_tokens": ntok,
        "temperature": temp,
        "top_p": top_p,
        "stream": True,
        "messages": [
            {"role": "system", "content": "Voce e um engenheiro. nonce=" + uuid.uuid4().hex},
            {"role": "user", "content": PROMPT},
        ],
    }
    req = urllib.request.Request(
        "http://%s:%d/v1/chat/completions" % (host, port),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    stamps = []
    pieces = []
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
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    stamps.append(time.perf_counter() - t0)
                    pieces.append(piece)
    n = len(stamps)
    if n == 0:
        return None
    out = {
        "tokens": n,
        "ttft_s": stamps[0],
        "total_s": stamps[-1],
        "decode_tok_s": None,
        "text": "".join(pieces)[:150].replace("\n", " "),
    }
    if n > 1 and stamps[-1] > stamps[0]:
        out["decode_tok_s"] = (n - 1) / (stamps[-1] - stamps[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--ntok", type=int, default=128)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    for i in range(args.runs):
        res = one_run(args.host, args.port, args.model, args.ntok, args.temp, args.top_p)
        if res is None:
            print("run %d: sem tokens" % i, flush=True)
            continue
        dec = res["decode_tok_s"]
        print(
            "run %d: tokens=%d ttft=%.2fs decode=%s total=%.2fs"
            % (i, res["tokens"], res["ttft_s"], ("%.2f tok/s" % dec) if dec else "n/a", res["total_s"]),
            flush=True,
        )
        print("   inicio: %s" % res["text"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
