#!/usr/bin/env python3
"""Mede a VAZAO de prefill (tok/s), nao o TTFT de prompt curto.

O numero antigo ("230-425 tok/s") era TTFT de um prompt de 210 tokens,
dominado por custo fixo por chunk. Aqui o prompt e' grande (2k/8k/32k) e
unico (nonce no inicio, `cached_tokens` conferido = 0), entao o TTFT e'
preenchimento de verdade.

Metodo: max_tokens=3, stream. t1 = TTFT (prefill + 1 passo de decode),
t2-t1 = passo de decode puro -> prefill_s = t1 - passo. Reporta tambem
prompt_tokens/TTFT (cota superior conservadora, sem descontar o passo).

Uso:
  bench_prefill.py --port 3001 -n 2048,8192,32768 --runs 3
  bench_prefill.py --port 3001 -n 8192 --json
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
import urllib.error
import urllib.request
import uuid

FILLER = (
    "o kernel de atencao esparsa le paginas de cento e vinte e oito tokens e o "
    "roteador escolhe quatro de cento e vinte e oito especialistas por camada. "
    "a memoria de valores por token por camada soma dois mil e quarenta e oito "
    "bytes e o pipeline de dez estagios troca ativacoes por pcie a cada passo. "
)


def build_prompt(target_tokens, nonce, tail="\n\nDiga apenas: ok"):
    """Texto variado (sem repeticao literal) com ~target_tokens tokens.

    A repeticao literal e' evitada porque com speculative decoding ou n-gram
    futuro ela mudaria o custo; aqui o conteudo tambem precisa ser unico por
    requisicao para o prefix cache nao reaproveitar nada.
    """
    target_chars = int(target_tokens * 3.55)
    parts = ["[ref %s] Leia o material tecnico a seguir e responda ao final.\n\n" % nonce]
    size = len(parts[0])
    i = 0
    while size < target_chars:
        block = "Item %d: %s" % (i, FILLER)
        parts.append(block)
        size += len(block)
        i += 1
    parts.append(tail)
    return "".join(parts)


def one_run(host, port, model, target_tokens, max_tokens, timeout, temp):
    nonce = uuid.uuid4().hex
    prompt = build_prompt(target_tokens, nonce)
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "http://%s:%d/v1/chat/completions" % (host, port),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    stamps = []
    usage = {}
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
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    stamps.append(time.perf_counter() - t0)
    if not stamps:
        return None

    ptoks = usage.get("prompt_tokens")
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")
              or usage.get("num_cached_tokens") or 0)

    ttft = stamps[0]
    step = None
    if len(stamps) >= 2:
        step = statistics.median([b - a for a, b in zip(stamps, stamps[1:])])
    prefill_s = ttft - step if step else ttft

    return {
        "target_tokens": target_tokens,
        "prompt_tokens": ptoks,
        "cached_tokens": cached,
        "gen_tokens": len(stamps),
        "ttft_s": ttft,
        "decode_step_s": step,
        "prefill_s": prefill_s,
        "prefill_tok_s": (ptoks / prefill_s) if ptoks and prefill_s > 0 else None,
        "prefill_tok_s_ttft": (ptoks / ttft) if ptoks and ttft > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("-n", "--targets", default="2048,8192,32768",
                    help="lista de tokens-alvo do prompt")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--json", action="store_true", help="so' as linhas JSON")
    args = ap.parse_args()

    targets = [int(x) for x in args.targets.split(",") if x.strip()]
    rc = 0
    for target in targets:
        rows = []
        for r in range(args.runs):
            try:
                row = one_run(args.host, args.port, args.model, target,
                              args.max_tokens, args.timeout, args.temp)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                print("n=%d run=%d ERRO %s" % (target, r, exc), flush=True)
                rc = 1
                continue
            if row is None:
                print("n=%d run=%d sem tokens" % (target, r), flush=True)
                rc = 1
                continue
            rows.append(row)
            if not args.json:
                print(
                    "n=%d run=%d prompt=%s cached=%s ttft=%.2fs step=%.3fs "
                    "prefill=%.2fs -> %.0f tok/s (TTFT cru: %.0f)"
                    % (target, r, row["prompt_tokens"], row["cached_tokens"],
                       row["ttft_s"], row["decode_step_s"] or 0.0,
                       row["prefill_s"], row["prefill_tok_s"] or 0.0,
                       row["prefill_tok_s_ttft"] or 0.0),
                    flush=True)
            if row["cached_tokens"]:
                print("  AVISO: prefix cache reaproveitou %d tokens; a medicao "
                      "nao vale como prefill" % row["cached_tokens"], flush=True)
        if rows:
            med = statistics.median([x["prefill_tok_s"] for x in rows if x["prefill_tok_s"]])
            pt = statistics.median([x["prompt_tokens"] for x in rows])
            print(json.dumps({"median_prefill_tok_s": round(med, 1),
                              "prompt_tokens": pt, "target_tokens": target,
                              "runs": len(rows)}), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
