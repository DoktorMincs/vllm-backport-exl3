#!/usr/bin/env python3
"""Case A2 (mixed-K) do tools/gb10_minimax_m3_parity.py na geometria real do M3.

Roda a tabela de dispatch que o serve do pack usa (unified exl3_moe_mixedk e
grouped) em lotes de decode e de prefill, comparando com a referencia clampada
SwiGLU-OAI (act 3) e com a referencia SILU. Depois do patch de act 3 no
hadamard_inner.cuh do exllamav3, cosine deve ir a ~1.0 e a SILU continuar baixa.

Uso: parity_m3_mixedk.py
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


import importlib.util
import json
import sys

TOOL = EXL3_ROOT + '/pkgs/vllm-exl3/tools/gb10_minimax_m3_parity.py'


def main() -> int:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0
    spec = importlib.util.spec_from_file_location("parity_m3", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.A_HIDDEN = 6144
    mod.A_INTER = 3072
    mod.A_NEXP = 128
    mod.TOPK = 4
    import torch

    device = torch.device("cuda:0")
    out = mod._case_a_mixedk_run(device)
    for p in out["probes"]:
        print(f"mixed-K mode={p['mode']:<8} tokens={p['tokens']:>4} path={p['path']:<16} "
              f"cos={p['cosine']:.6f} cos_silu={p['cosine_silu_reference']:.6f}", flush=True)
    print(f"tuples={out['tuples']}  pass={out['pass']}", flush=True)
    print(json.dumps(out), flush=True)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
