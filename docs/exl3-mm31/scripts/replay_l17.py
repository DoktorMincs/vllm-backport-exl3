#!/usr/bin/env python3
"""Teste decisivo: oraculo parte do estado da L17 do vLLM e roda ate a L59.

O prefill de 210 tokens do vLLM bate com o oraculo F32 ate a L17 (cos
1.000000), mas na saida o vLLM diz 'Voce' (p~0.12) enquanto o oraculo diz
'<mm:think>' (p~0.99). Duas leituras possiveis:

  (a) as camadas 18..59 do vLLM estao erradas -> o replay do oraculo, partindo
      do MESMO estado, tem de recuperar '<mm:think>';
  (b) o modelo e' caotico nessa escala -> o replay tambem devolve uma
      distribuicao achatada.

Nada aqui toca no servidor: le o dump em disco.

Uso: replay_l17.py --dump logs/dump_big --ids logs/ids_big.json \
                   --out logs/replay_l17.npz [--start 17] [--end 59]
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
import importlib.util
import json
import pathlib

import numpy as np

REF = MM3_REF
MODEL = MODEL_DIR
PFX = "language_model.model."
LM = "language_model.lm_head"


def dump_state(dump: str, L: int, nrows: int) -> np.ndarray:
    def ld(name):
        p = pathlib.Path(dump) / f"L{L:02d}_{name}.npz"
        return np.load(p)["x"][:nrows].astype(np.float64) if p.exists() else 0.0
    return ld("residual") + ld("attn_out") + ld("ffn_out")


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=17)
    ap.add_argument("--end", type=int, default=59)
    ap.add_argument("--pos", type=int, default=-1, help="posicao para a cabeca")
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("refmm3", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import torch

    raw = json.load(open(a.ids))
    ids = raw["ids"] if isinstance(raw, dict) else raw
    T = len(ids)
    pos_i = (a.pos % T)

    h_np = dump_state(a.dump, a.start, T)
    print(f"estado inicial: dump L{a.start}, linhas 0..{T-1}, |h|={np.linalg.norm(h_np):.2f}",
          flush=True)

    ref = mod.Ref(pathlib.Path(a.model), a.dev, False)
    g = ref.g
    h = torch.from_numpy(np.ascontiguousarray(h_np, dtype=np.float32)).to(a.dev)
    pos = torch.arange(T, device=a.dev)

    print(f"{'L':>3} {'cos(replay,dump)':>17} {'relerr':>10}", flush=True)
    for i in range(a.start + 1, a.end + 1):
        w = ref.layer_weights(i)
        h = mod.layer(ref, g, i, w, h, pos)
        del w
        torch.cuda.empty_cache()
        hn = h.detach().cpu().numpy().astype(np.float64)
        dn = dump_state(a.dump, i, T)
        print(f"{i:>3} {cos(hn, dn):>17.6f} "
              f"{np.linalg.norm(hn - dn) / max(np.linalg.norm(dn), 1e-12):>10.2e}", flush=True)

    h = mod.gemma_rms(h, ref.T(PFX + "norm.weight"), g.eps)
    logits = mod.lin(h, ref.W(LM)).float().cpu().numpy()
    np.savez(a.out, logits=logits, ids=np.array(ids), pos=np.array([pos_i]))

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(pathlib.Path(a.model) / "tokenizer.json"))
    dec = lambda t: tk.decode([int(x) for x in t])
    li = logits[pos_i]
    lp = torch.log_softmax(torch.from_numpy(li), -1).numpy()
    order = np.argsort(-li)[:8]
    print(f"\nreplay (do estado L{a.start} do vLLM) top8 em pos {pos_i}:")
    print("   " + ", ".join(f"{int(t)}:{dec([int(t)])!r}:{lp[t]:.3f}" for t in order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
