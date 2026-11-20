#!/usr/bin/env python3
"""G1: paridade dos kernels EXL3 em sm86 contra o reconstruct do exllamav3.

Para cada amostra real do pack (K3/K4/K5/K6 + lm_head):
  1. monta uma LinearEXL3 (caminho fused: exl3_gemm/exl3_gemv da ext);
  2. reconstroi a matriz de pesos pelo caminho independente
     (ext.reconstruct + hadamard + suh/svh -> get_weight_tensor);
  3. compara linear(x) com x @ W_ref.

Uso: python parity_check.py [--max-out N] [--samples DIR]
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
import os
import sys

import numpy as np
import torch

SAMPLES = EXL3_ROOT + '/artifacts/samples'


def build(entry, max_out, device):
    z = np.load(os.path.join(SAMPLES, entry["file"]))
    trellis = torch.from_numpy(z["trellis"]).to(device)
    suh = torch.from_numpy(z["suh"]).to(device)
    svh = torch.from_numpy(z["svh"]).to(device)
    mul1 = torch.from_numpy(z["mul1"]).to(device).reshape(1).to(torch.int32)

    in_tiles, out_tiles, k_words = trellis.shape
    if int(suh.numel()) != in_tiles * 16:
        raise SystemExit(f"{entry['tensor']}: suh {suh.numel()} != in {in_tiles*16}")
    if int(svh.numel()) != out_tiles * 16:
        raise SystemExit(f"{entry['tensor']}: svh {svh.numel()} != out {out_tiles*16}")
    if k_words % 16:
        raise SystemExit(f"{entry['tensor']}: k_words {k_words} nao e multiplo de 16")

    keep = min(out_tiles, max(1, max_out // 16))
    if keep < out_tiles:
        trellis = trellis[:, :keep, :].contiguous()
        svh = svh[: keep * 16].contiguous()
    return trellis, suh, svh, mul1, keep * 16


def main() -> int:
    global SAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-out", type=int, default=4096)
    ap.add_argument("--samples", default=SAMPLES)
    ap.add_argument("--dev", default="cuda:0")
    args = ap.parse_args()

    SAMPLES = args.samples
    manifest = json.load(open(os.path.join(SAMPLES, "samples.json")))

    from vllm_exl3.exl3 import make_linear_exl3

    device = torch.device(args.dev)
    torch.manual_seed(0)
    failures = []
    print(f"{'amostra':11} {'K':>2} {'in':>6} {'out':>6}  cos      max_rel   veredito")
    for name, entry in manifest.items():
        if "mul1" not in entry["shapes"]:
            print(f"{name:11} (sem trellis, pulando)")
            continue
        trellis, suh, svh, mul1, out_rows = build(entry, args.max_out, device)
        K = trellis.shape[2] // 16
        lin = make_linear_exl3(
            trellis, suh, svh, None, mul1, out_dtype=torch.float16
        )
        torch.cuda.synchronize()

        W = lin.get_weight_tensor()          # [in, out] fp16, caminho independente
        torch.cuda.synchronize()
        in_features = W.shape[0]

        x = torch.randn(16, in_features, device=device, dtype=torch.float16)
        with torch.no_grad():
            # Convencao do plugin/exllamav3: forward(x, params, out_dtype=...)
            y = lin.forward(x.contiguous(), {}, out_dtype=torch.float32).float()
            y_ref = (x.float() @ W.float())
        torch.cuda.synchronize()

        cos = torch.nn.functional.cosine_similarity(
            y.flatten(), y_ref.flatten(), dim=0
        ).item()
        denom = y_ref.abs().mean().clamp_min(1e-6)
        max_rel = ((y - y_ref).abs().max() / denom).item()
        ok = cos >= 0.9999 and max_rel < 0.05
        if not ok:
            failures.append(name)
        print(
            f"{name:11} {K:>2} {in_features:>6} {out_rows:>6}  "
            f"{cos:.7f}  {max_rel:8.4f}  {'OK' if ok else 'FALHOU'}"
        )
        del lin, W, x, y, y_ref, trellis, suh, svh, mul1
        torch.cuda.empty_cache()

    print()
    if failures:
        print(f"FALHAS: {failures}")
        return 1
    print("PARIDADE OK em todas as amostras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
