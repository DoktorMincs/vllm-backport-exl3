#!/usr/bin/env python3
"""Replay do oraculo a partir do estado que o vLLM produziu (teste decisivo).

Se o oraculo, partindo do estado do vLLM depois da camada S, reproduzir a
saida do vLLM (top-1/distribuicao), entao as camadas S+1..59 do vLLM sao
fieis e a divergencia e ruido bf16 amplificado. Se o oraculo voltar a ' Paris',
ha defeito real nas camadas depois de S.

A cada camada o replay imprime cos/relerr contra o estado dumpado pelo vLLM
(a divergencia dentro do replay e a "taxa de caos" do modelo, nao do kernel).

Fontes do estado inicial (--source):
  vllm   residual + attn_out + ffn_out da linha --row em --start
  ref    hs[--start] do ref_full.npz

--noise bf16|fp16|none  arredonda o estado inicial (simula armazenamento)
--noise-rel F           ruido gaussiano de norma relativa F

Uso: replay_from_dump.py --ref logs/ref_full.npz --dump logs/dump3 --start 15 \
                          --source vllm [--end 59] [--row 4] [--pos 4]
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


def _files(dump: str, pattern: str) -> list[pathlib.Path]:
    """Arquivos do padrao, tolerando o sufixo de rank (`.r0.npz`, TP>1)."""
    d = pathlib.Path(dump)
    exact = d / f"{pattern}.npz"
    if exact.exists():
        return [exact]
    return sorted(d.glob(f"{pattern}.r*.npz"))


def dump_state(dump: str, L: int, nrows: int) -> np.ndarray:
    """Estado depois da camada L, linhas 0..nrows-1 (o prefixo causal real).

    Em TP>1 o `attn_out` e' a saida *parcial* de cada rank (row-parallel) e
    precisa ser somada; `residual` e `ffn_out` sao replicados (pos-all-reduce).
    """
    def ld(name, sum_ranks=False):
        files = _files(dump, f"L{L:02d}_{name}")
        if not files:
            return 0.0
        out = 0.0
        for p in files if sum_ranks else files[:1]:
            out = out + np.load(p)["x"][:nrows].astype(np.float64)
        return out
    return ld("residual") + ld("attn_out", sum_ranks=True) + ld("ffn_out")


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def topk_str(vals, dec, k=8) -> str:
    order = np.argsort(-vals)[:k]
    return ", ".join(f"{int(o)}:{dec([int(o)])!r}:{float(vals[o]):.3f}" for o in order)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--start", type=int, default=15)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--source", choices=["vllm", "ref"], default="vllm")
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--pos", type=int, default=4)
    ap.add_argument("--noise", choices=["none", "bf16", "fp16"], default="none")
    ap.add_argument("--noise-rel", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-dump-cmp", action="store_true")
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("refmm3", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import torch

    z = np.load(a.ref)
    T = int(z["ids"].shape[0])
    hs = z["hs"]
    logits_ref = z["logits"]
    pos_i = a.pos % T
    pos = torch.arange(T, device=a.dev)
    cfg = json.loads((pathlib.Path(a.model) / "config.json").read_text())
    n_layers = int((cfg.get("text_config") or cfg)["num_hidden_layers"])
    end = n_layers - 1 if a.end < 0 else a.end

    if a.source == "vllm":
        h_np = dump_state(a.dump, a.start, T)
        tag = f"vLLM estado apos L{a.start} (linhas 0..{T - 1})"
    else:
        h_np = hs[a.start].astype(np.float64)
        tag = f"oraculo hs[{a.start}] (posicoes 0..{T - 1})"
    ref_np = hs[a.start].astype(np.float64)
    rel = np.linalg.norm(h_np - ref_np) / max(np.linalg.norm(ref_np), 1e-12)
    print(f"fonte: {tag}  |h|={np.linalg.norm(h_np):.2f}  delta_rel_vs_oraculo={rel:.3e}  "
          f"camadas {a.start + 1}..{end}")

    h = torch.from_numpy(np.ascontiguousarray(h_np, dtype=np.float32)).to(a.dev)
    if a.noise == "bf16":
        h = h.bfloat16().float()
    elif a.noise == "fp16":
        h = h.half().float()
    if a.noise_rel > 0:
        gen = torch.Generator().manual_seed(a.seed)
        n = torch.randn(h.shape, generator=gen).to(a.dev)
        h = h + n * (a.noise_rel * h.norm() / n.norm())
        print(f"ruido gaussiano relativo {a.noise_rel:g} (seed {a.seed})")

    ref = mod.Ref(pathlib.Path(a.model), a.dev, False)
    g = ref.g
    print(f"{'L':>3} {'cos(replay,dump)':>17} {'relerr':>10} {'|replay|':>10} {'|dump|':>10} {'|ref|':>10}")
    for i in range(a.start + 1, end + 1):
        w = ref.layer_weights(i)
        h = mod.layer(ref, g, i, w, h, pos)
        del w
        torch.cuda.empty_cache()
        hn = h.detach().cpu().numpy().astype(np.float64)
        if not a.skip_dump_cmp:
            try:
                dn = dump_state(a.dump, i, T)
            except FileNotFoundError:
                continue
            print(f"{i:>3} {cos(hn, dn):>17.6f} "
                  f"{np.linalg.norm(hn - dn) / max(np.linalg.norm(dn), 1e-12):>10.2e} "
                  f"{np.linalg.norm(hn):>10.4g} {np.linalg.norm(dn):>10.4g} "
                  f"{np.linalg.norm(hs[i][:T]):>10.4g}", flush=True)

    if end < g.n_layers - 1:
        print("(parou antes da ultima camada; sem logits)")
        return 0

    h = mod.gemma_rms(h, ref.T(PFX + "norm.weight"), g.eps)
    head = ref.W(LM)
    logits = mod.lin(h, head).float().cpu().numpy()
    li = logits[pos_i]
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(pathlib.Path(a.model) / "tokenizer.json"))
    dec = lambda t: tk.decode([int(x) for x in t])
    lp = torch.log_softmax(torch.from_numpy(li), -1).numpy()
    lpr = torch.log_softmax(torch.from_numpy(logits_ref[pos_i]), -1).numpy()
    print(f"\nreplay  top8: {topk_str(li, dec)}")
    print(f"oraculo top8: {topk_str(logits_ref[pos_i], dec)}")
    best = int(np.argmax(logits_ref[pos_i]))
    am = int(np.argmax(li))
    print(f"top-1 oraculo ({best}={dec([best])!r}): logprob replay {lp[best]:.3f} "
          f"vs oraculo {lpr[best]:.3f}")
    print(f"argmax replay={am} ({dec([am])!r})   cos(logits)={cos(li, logits_ref[pos_i]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
