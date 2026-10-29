#!/usr/bin/env python3
"""Curva latencia x vazao agregada com C requisicoes concorrentes.

PP=10 sem DBO deixa 9 de 10 GPUs ociosas com 1 usuario; a vazao agregada e'
o unico jeito de usar o pipeline. Mede, para cada C:
  - tok/s agregado (tokens gerados / tempo de parede)
  - p50/p99 de TTFT e de decode por requisicao
  - tokens por requisicao (fixo, para comparar sem a armadilha do N diferente)

Uso: bench_concurrency.py --port 3001 -c 1,2,4,8,16 --ntok 128
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
import threading
import time
import urllib.error
import urllib.request
import uuid

PROMPT = (
    "Escreva um paragrafo tecnico sobre atencao esparsa em modelos "
    "autoregressivos, citando paginas de chave-valor e selecao de blocos."
)


def one_request(host, port, model, ntok, temp, top_p, barrier, out, idx, timeout):
    body = {
        "model": model,
        "max_tokens": ntok,
        "temperature": temp,
        "top_p": top_p,
        "stream": True,
        "messages": [
            {"role": "system", "content": "nonce=" + uuid.uuid4().hex},
            {"role": "user", "content": PROMPT},
        ],
    }
    req = urllib.request.Request(
        "http://%s:%d/v1/chat/completions" % (host, port),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    try:
        barrier.wait(timeout=120)
        t0 = time.perf_counter()
        stamps = []
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
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        stamps.append(time.perf_counter() - t0)
        out[idx] = {"tokens": len(stamps),
                    "ttft_s": stamps[0] if stamps else None,
                    "last_s": stamps[-1] if stamps else None}
    except Exception as exc:  # noqa: BLE001 - queremos o erro no relatorio
        out[idx] = {"error": str(exc)}


def run_point(host, port, model, c, ntok, temp, top_p, timeout):
    barrier = threading.Barrier(c)
    out = [None] * c
    threads = [
        threading.Thread(target=one_request,
                         args=(host, port, model, ntok, temp, top_p, barrier,
                               out, i, timeout))
        for i in range(c)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in out if r and "error" not in r and r["tokens"]]
    errs = [r for r in out if r and "error" in r]
    if not ok:
        return {"c": c, "wall_s": wall, "errors": [e["error"] for e in errs]}
    total_tokens = sum(r["tokens"] for r in ok)
    ttf = sorted(r["ttft_s"] for r in ok)
    dec = sorted((r["tokens"] - 1) / (r["last_s"] - r["ttft_s"])
                 for r in ok if r["tokens"] > 1 and r["last_s"] > r["ttft_s"])

    def pct(vals, p):
        if not vals:
            return None
        k = max(0, min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1)))))
        return vals[k]

    return {
        "c": c,
        "wall_s": round(wall, 2),
        "requests_ok": len(ok),
        "errors": len(errs),
        "tokens_total": total_tokens,
        "aggregate_tok_s": round(total_tokens / wall, 2),
        "ttft_p50": round(pct(ttf, 50), 2),
        "ttft_p99": round(pct(ttf, 99), 2),
        "decode_p50": round(statistics.median(dec), 2) if dec else None,
        "decode_p99": round(pct(dec, 99), 2) if dec else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("-c", "--conc", default="1,2,4,8,16")
    ap.add_argument("--ntok", type=int, default=128)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    for c in [int(x) for x in args.conc.split(",") if x.strip()]:
        row = run_point(args.host, args.port, args.model, c, args.ntok,
                        args.temp, args.top_p, args.timeout)
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
