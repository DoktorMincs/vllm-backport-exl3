#!/usr/bin/env python3
"""Consistencia prefill x decode SEM oraculo.

Gera N tokens com KV cache (1 token por passo) e depois re-prefilla
`ids + gen[:k]` de uma vez pedindo so a proxima token. Se o caminho de decode
(insercao/leitura do KV, kernels m=1) estiver fiel, a distribuicao do passo k+1
tem de bater com a do prefill do mesmo prefixo.

Nao prova que o modelo esta certo, mas isola erro de KV/decode de erro de
kernel de prefill.

Uso: prefix_consistency.py --ids logs/ids_big.json [--steps 4] [--out logs/prefix_cons.json]
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
import urllib.request


def post(base: str, path: str, body: dict, timeout: float = 3600.0):
    req = urllib.request.Request(
        f"http://{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = f"{a.host}:{a.port}"

    raw = json.load(open(a.ids))
    ids = raw["ids"] if isinstance(raw, dict) else raw

    # 1) decode real: um prefill + N passos lendo o KV cache
    out = post(base, "/v1/completions",
               {"model": a.model, "prompt": ids, "max_tokens": a.steps,
                "temperature": 0, "logprobs": a.topk, "return_token_ids": True})
    ch = (out.get("choices") or [{}])[0]
    gen = ch.get("token_ids") or []
    # O backport devolve o objeto Logprobs do OpenAI (tokens/top_logprobs dentro
    # de `logprobs`), nao os campos soltos no choice.
    tops_dec = ((ch.get("logprobs") or {}).get("top_logprobs")) or [{}]
    print(f"decode: {len(gen)} tokens {gen}", flush=True)

    results = []
    for k in range(len(gen)):
        prefix = list(ids) + list(gen[:k])
        out2 = post(base, "/v1/completions",
                    {"model": a.model, "prompt": prefix, "max_tokens": 1,
                     "temperature": 0, "logprobs": a.topk, "return_token_ids": True})
        ch2 = (out2.get("choices") or [{}])[0]
        tops_pf = (((ch2.get("logprobs") or {}).get("top_logprobs")) or [{}])[0] or {}
        dec = tops_dec[k] if k < len(tops_dec) else {}

        print(f"\npasso {k}: gerado {gen[k]!r}", flush=True)
        for t in sorted(set(dec) | set(tops_pf), key=lambda t: -float(dec.get(t, -99.0))):
            d, p = dec.get(t), tops_pf.get(t)
            print(f"   {t!r:>12}  decode={f'{d:7.3f}' if d is not None else '   -   '}"
                  f"  prefill={f'{p:7.3f}' if p is not None else '   -   '}", flush=True)
        d1 = max(dec, key=lambda t: dec[t]) if dec else None
        p1 = max(tops_pf, key=lambda t: tops_pf[t]) if tops_pf else None
        print(f"   top-1 decode={d1!r}  top-1 prefill={p1!r}  "
              f"{'MESMA' if d1 == p1 else 'DIFERENTE'}", flush=True)
        results.append({"step": k, "gen_token": gen[k], "top1_decode": d1,
                        "top1_prefill": p1, "same_top1": d1 == p1,
                        "decode": dec, "prefill": tops_pf})

    if a.out:
        json.dump({"ids": ids, "gen": gen, "steps": results}, open(a.out, "w"))
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
