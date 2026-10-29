#!/usr/bin/env python3
"""Smoke + bench do servidor EXL3 (porta propria), sem tocar em producao.

Modo 1 (padrao): 3 sondas de conteudo (conteudo/raciocinio/codigo) com nonce
anti-cache e temperatura 0; imprime TTFT, tok/s de stream e o texto.

Modo 2 (--bench): TTFT e taxa por tamanho de contexto, same idea do
run_bench do projeto: nonce no inicio do prompt, SSE, TTFT = 1o delta de
conteudo nao vazio, decode = (ultimo - primeiro) chunk.

Uso:
  python -I smoke_chat.py [--port 3001] [--model NOME]
  python -I smoke_chat.py --bench 1000,8000 --ntok 128 --reps 2
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
import sys
import time
import urllib.request

CHARS_PER_TOKEN = 3.6


def _post(base: str, body: dict, timeout: float = 900.0):
    req = urllib.request.Request(
        f"http://{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(base: str, body: dict) -> dict:
    """Envia em SSE e devolve metricas + texto."""
    t0 = time.time()
    first = None
    last = None
    texto: list[str] = []
    usage = None
    n_chunks = 0
    with _post(base, body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
                    n_chunks += 1
                    texto.append(piece)
    t1 = time.time()
    out = "".join(texto)
    decode_n = None
    decode_tps = None
    if usage:
        completion = int(usage.get("completion_tokens") or 0)
        decode_n = completion
        if first is not None and last is not None and completion > 1 and last > first:
            decode_tps = (completion - 1) / (last - first)
    return {
        "ttft": None if first is None else first - t0,
        "total": t1 - t0,
        "chunks": n_chunks,
        "n": decode_n,
        "tps": decode_tps,
        "texto": out,
        "usage": usage,
    }


def modelos(base: str) -> list[str]:
    with urllib.request.urlopen(f"http://{base}/v1/models", timeout=30) as r:
        return [d["id"] for d in json.loads(r.read()).get("data", [])]


PROBES = [
    (
        "pt-br",
        "Explique em português do Brasil, em no máximo 4 frases, por que o céu é azul.",
    ),
    (
        "codigo",
        "Escreva uma função Python chamada `n_primos` que devolve os n primeiros "
        "números primos, com docstring e tratamento de n<=0. Só o código.",
    ),
    (
        "probabilidade",
        "Jogo um dado justo de 6 faces duas vezes. Qual é a probabilidade de a soma "
        "ser 7? Mostre a contagem e a fração final.",
    ),
]


def filler(n_chars: int, nonce: str) -> str:
    """Texto neutro de ~n_chars, com nonce no inicio (mata cache de prefixo)."""
    base = (
        "Relatório técnico de acompanhamento. "
        "Os sensores registraram variação estável ao longo do período analisado, "
        "com desvios dentro da faixa esperada para a estação. "
    )
    corpo = base * (n_chars // len(base) + 1)
    return f"[nonce {nonce}] " + corpo[:n_chars]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ntok", type=int, default=128)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--bench", default="")
    ap.add_argument("--max-chars", type=int, default=500)
    args = ap.parse_args()
    base = f"{args.host}:{args.port}"

    ids = modelos(base)
    modelo = args.model or ids[0]
    print(f"servidor {base}  modelos={ids}  usando={modelo}", flush=True)

    if args.bench:
        alvos = [int(x) for x in args.bench.split(",")]
        print("aquecimento...", flush=True)
        stream_chat(
            base,
            {
                "model": modelo,
                "max_tokens": 8,
                "temperature": 0,
                "messages": [{"role": "user", "content": "oi"}],
            },
        )
        print("ctx      rep  ttft_s   dec_tps   n   total_s", flush=True)
        for alvo in alvos:
            for rep in range(args.reps):
                nonce = secrets.token_hex(8)
                txt = filler(int(alvo * CHARS_PER_TOKEN), nonce)
                r = stream_chat(
                    base,
                    {
                        "model": modelo,
                        "max_tokens": args.ntok,
                        "temperature": 0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "messages": [
                            {"role": "user", "content": txt + "\n\nResuma em uma linha."}
                        ],
                    },
                )
                ttft = "-" if r["ttft"] is None else f"{r['ttft']:.3f}"
                tps = "-" if r["tps"] is None else f"{r['tps']:.2f}"
                n = "-" if r["n"] is None else str(r["n"])
                print(f"{alvo:>6} {rep:>4} {ttft:>8} {tps:>9} {n:>4} {r['total']:>8.2f}", flush=True)
        return 0

    falhas = 0
    for nome, prompt in PROBES:
        nonce = secrets.token_hex(6)
        r = stream_chat(
            base,
            {
                "model": modelo,
                "max_tokens": args.ntok,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [
                    {
                        "role": "system",
                        "content": f"Você é um assistente preciso. (sessão {nonce})",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
        )
        txt = r["texto"].strip()
        vazio = not txt
        if vazio:
            falhas += 1
        ttft = "-" if r["ttft"] is None else f"{r['ttft']:.2f}s"
        tps = "-" if r["tps"] is None else f"{r['tps']:.1f} tok/s"
        print(
            f"\n=== {nome}  ttft={ttft}  stream={tps}  n={r['n']}  "
            f"chunks={r['chunks']}{'  VAZIO!' if vazio else ''}"
        )
        print(txt[: args.max_chars].replace("\n", "\n  ") or "(vazio)")
    print()
    print("RESULTADO:", "OK" if falhas == 0 else f"{falhas} sonda(s) vazia(s)")
    return 0 if falhas == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
