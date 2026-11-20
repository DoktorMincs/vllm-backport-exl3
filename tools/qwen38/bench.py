#!/usr/bin/env python3
"""Throughput bench for a served Qwen3.8-27B-NVFP4 endpoint.

Usage: bench.py <tag> [port] [model]
Env: CONC (default 4), NTOK (default 192), BENCH_TEXT (file used as the long
prefill prompt; defaults to this directory's RUNBOOK).

Tokens are counted from the server's `usage.completion_tokens`, not from SSE
chunks: Qwen3.8 streams its thinking in `delta.reasoning` and each chunk
carries a few tokens, so counting chunks undercounts decode by ~3x.
"""
import concurrent.futures as cf
import json, os, sys, time, urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 3001
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen38-nvfp4"
BASE = f"http://127.0.0.1:{PORT}"
CONC = int(os.environ.get("CONC", "4"))
NTOK = int(os.environ.get("NTOK", "192"))
TEXT_PATH = os.environ.get(
    "BENCH_TEXT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "docs", "qwen38-nvfp4", "RUNBOOK.md"),
)

PROMPT = ("Escreva um ensaio tecnico detalhado sobre a historia dos "
          "aceleradores de hardware para aprendizado profundo, cobrindo "
          "arquiteturas de 2012 a 2026, com numeros quando possivel. ")

TEXT = open(TEXT_PATH).read()


def stream(prompt, max_tokens=NTOK):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    usage = {}
    seen = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                if (delta.get("content") or delta.get("reasoning")
                        or delta.get("tool_calls")):
                    if ttft is None:
                        ttft = time.time() - t0
                    seen += 1
    total = time.time() - t0
    return ttft, seen, total, usage


def one(prompt, ntok=NTOK):
    ttft, seen, total, u = stream(prompt, ntok)
    n = u.get("completion_tokens", seen)
    dec = (n - 1) / (total - ttft) if n > 1 and total > ttft else 0.0
    pt = u.get("prompt_tokens", 0)
    return {"ttft": ttft, "n": n, "total": total, "dec": dec,
            "pt": pt, "pf": (pt / ttft if ttft else 0), "chunks": seen}


def metric(name):
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith(name + "{"):
                    return float(line.rsplit(" ", 1)[1])
                if line.startswith(name + " "):
                    return float(line.rsplit(" ", 1)[1])
    except Exception:
        return None
    return None


# 1) long prefill (fresh context)
long_prompt = f"[{int(time.time())}] " + TEXT + "\n\nResuma em 3 itens."
r_pf = one(long_prompt, 32)
print(f"[{TAG}] prefill: {r_pf['pt']} tok em {r_pf['ttft']:.2f}s = "
      f"{r_pf['pf']:.0f} tok/s")

# 2) decode C=1 (3 reps, median)
vals = [one(PROMPT) for _ in range(3)]
vals.sort(key=lambda x: x["dec"])
r1 = vals[1]
print(f"[{TAG}] decode C=1: {r1['dec']:.1f} tok/s "
      f"({', '.join(f'{v['dec']:.1f}' for v in vals)})")

# 3) concurrent decode
with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
    rs = list(ex.map(lambda _: one(PROMPT), range(CONC)))
agg = sum(x["n"] for x in rs) / max(x["total"] for x in rs)
per = [x["dec"] for x in rs]
print(f"[{TAG}] decode C={CONC}: {agg:.1f} tok/s agregado "
      f"({', '.join(f'{v:.1f}' for v in per)})")

acc = metric("vllm:spec_decode_num_accepted_tokens_total")
drf = metric("vllm:spec_decode_num_draft_tokens_total")
if acc is not None and drf:
    print(f"[{TAG}] spec decode: {drf:.0f} drafts, {acc:.0f} aceitos, "
          f"taxa {acc / drf:.1%}")
