#!/usr/bin/env python3
"""Completion com ids explicitos e logprobs por passo (prefill + decode).

Serve para dois testes:
- passo 0 vem do PREFILL (uma passada com todos os ids);
- passos >= 1 vem do DECODE (1 token por passo, lendo o KV cache).
Comparando os logprobs por passo com o oraculo da para separar prefill de decode.

Uso: request_ids.py --ids logs/ids_big.json --steps 4
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
    out = post(base, "/v1/completions", {
        "model": a.model, "prompt": ids, "max_tokens": a.steps,
        "temperature": 0, "logprobs": a.topk, "return_token_ids": True,
    })
    ch = (out.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    # OpenAI-style: `logprobs` e' um objeto com tokens/token_logprobs/top_logprobs.
    toks = lp.get("tokens") or []
    tops = lp.get("top_logprobs") or []
    print(f"prompt_tokens={out.get('usage', {}).get('prompt_tokens')}  gerados={ch.get('text')!r}")
    for i, (tok, top) in enumerate(zip(toks, tops)):
        tops_sorted = sorted((top or {}).items(), key=lambda kv: -kv[1])
        print(f"passo {i}: escolhido {tok!r}  top{args.topk}: "
              + ", ".join(f"{k!r}:{v:.3f}" for k, v in tops_sorted))
    gen_ids = ch.get("token_ids")
    print(f"token_ids={gen_ids}")
    if a.out:
        json.dump({"ids": ids, "token_ids": gen_ids, "tokens": toks,
                   "top_logprobs": tops}, open(a.out, "w"))
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
