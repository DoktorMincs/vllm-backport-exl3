#!/usr/bin/env python3
"""Sonda de ablacao do M3 no vLLM: os 60 layers estao fazendo alguma coisa?

Teste central (independencia de prefixo): se o modelo efetivo tivesse ZERO
layers funcionando, o estado no ultimo token seria so o embedding do ultimo
token e a distribuicao do proximo token seria IDENTICA para qualquer prefixo
que termine igual. Um modelo real muda a distribuicao quando o prefixo muda.

Tambem mede: primeiro token (argmax + top-8 logprobs), recall de contexto
longo, copia literal e taxa de decode.

Uso: probe_ablate.py [--port 3001] [--model NOME] [--long 1500]
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


def post(base: str, body: dict, timeout: float = 900.0):
    req = urllib.request.Request(
        f"http://{base}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def gen(base: str, modelo: str, prompt: str, n: int, topk: int = 8,
        stream: bool = False) -> dict:
    """Uma completion greedy; devolve tokens + logprobs por posicao."""
    body = {
        "model": modelo,
        "prompt": prompt,
        "max_tokens": n,
        "temperature": 0,
        "logprobs": topk,
        "echo": False,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    t0 = time.time()
    tok: list[str] = []
    first = last = None
    per_pos: list[dict] = []
    usage = None
    with post(base, body) as r:
        raw = r.read()
    dt = time.time() - t0
    obj = json.loads(raw)
    if obj.get("usage"):
        usage = obj["usage"]
    for ch in obj.get("choices") or []:
        lp = ch.get("logprobs") or {}
        tok = lp.get("tokens") or []
        for entry in lp.get("top_logprobs") or []:
            per_pos.append(entry or {})
    return {"tokens": tok, "per_pos": per_pos, "usage": usage, "wall": dt}


def show(tag: str, r: dict) -> None:
    n = (r["usage"] or {}).get("completion_tokens")
    tps = None
    if n and n > 1:
        tps = (n - 1) / r["wall"] if r["wall"] else None
    print(f"\n--- {tag}: n={n} wall={r['wall']:.2f}s "
          f"({tps:.2f} tok/s incluindo prefill)" if tps else f"\n--- {tag}")
    if r["per_pos"]:
        first = r["per_pos"][0]
        top = sorted(first.items(), key=lambda kv: -kv[1])[:8]
        print(f"   1o token: {' '.join(r['tokens'][:1])!r}  top8={top}")
    for i, entry in enumerate(r["per_pos"][1:6], start=1):
        top = sorted(entry.items(), key=lambda kv: -kv[1])[:5]
        print(f"   pos{i}: {r['tokens'][i] if i < len(r['tokens']) else '?'!r} top5={top}")
    print(f"   texto: {''.join(r['tokens'])[:220]!r}")


def filler(n_chars: int, nonce: str) -> str:
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
    ap.add_argument("--long", type=int, default=1500, help="tokens de recheio")
    args = ap.parse_args()
    base = f"{args.host}:{args.port}"
    with urllib.request.urlopen(f"http://{base}/v1/models", timeout=30) as r:
        ids = [d["id"] for d in json.loads(r.read()).get("data", [])]
    modelo = args.model or ids[0]
    print(f"servidor {base}  modelos={ids}  usando={modelo}", flush=True)

    # 1) independencia de prefixo: mesmo sufixo, prefixos sem relacao
    p_a = "The capital of France is"
    p_b = "Qzxv wpty blorf. The capital of France is"
    p_c = "The capital of France is"
    r_a = gen(base, modelo, p_a, 1)
    r_b = gen(base, modelo, p_b, 1)
    r_c = gen(base, modelo, p_c, 1)
    show("A prefixo curto", r_a)
    show("B prefixo sem relacao", r_b)
    show("C repeticao de A (sanity/cache)", r_c)

    def top8(r):
        return sorted((r["per_pos"][0] or {}).items(), key=lambda kv: -kv[1])[:8]

    ta, tb, tc = top8(r_a), top8(r_b), top8(r_c)
    same_ab = ta == tb
    same_ac = ta == tc
    print(f"\nprefixo ignorado (A vs B identicos): {same_ab}")
    print(f"A vs C identicos (sanity): {same_ac}")
    if ta and tb:
        delta = [abs(a[1] - b[1]) for a, b in zip(ta, tb) if a[0] == b[0]]
        print(f"|dlogprob| casado A/B: {[round(d, 3) for d in delta]}")

    # 2) recall de contexto longo
    nonce = secrets.token_hex(4)
    segredo = f"BANANA-{nonce}"
    long_prompt = (
        filler(int(args.long * 3.6), nonce)
        + f"\n\nO codigo secreto e {segredo}.\n\nPergunta: qual e o codigo secreto? "
        "Responda apenas o codigo."
    )
    r_l = gen(base, modelo, long_prompt, 12)
    show(f"L recall ctx~{args.long} tok (segredo {segredo})", r_l)
    print(f"   contem segredo: {segredo in ''.join(r_l['tokens'])}")

    # 3) copia literal
    alvo = "QWERTY-54321-ZZZ"
    r_cp = gen(base, modelo, f"Repita exatamente: {alvo}\nResposta:", 14)
    show(f"M copia literal (alvo {alvo})", r_cp)
    print(f"   contem alvo: {alvo in ''.join(r_cp['tokens'])}")

    # 4) decode sustentado
    r_d = gen(base, modelo, "Conte de 1 a 40 em numeros.", 40, topk=1)
    n = (r_d["usage"] or {}).get("completion_tokens") or 0
    print(f"\n--- D decode: n={n} wall={r_d['wall']:.2f}s "
          f"-> {n / r_d['wall']:.2f} tok/s (com prefill junto)")
    print(f"   texto: {''.join(r_d['tokens'])[:200]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
