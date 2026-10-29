#!/usr/bin/env python3
"""Case A do tools/gb10_minimax_m3_parity.py na GEOMETRIA REAL do M3.

O tool fixa A_HIDDEN=4096, A_INTER=1024, A_NEXP=8, TOPK=2 (caso GB10 do autor).
Aqui reusamos o MESMO codigo (mesmos packs sinteticos, mesma referencia
clampada SwiGLU-OAI, mesmo build_exl3_fused_state) apenas trocando as
constantes de modulo para hidden=6144, intermediate=3072, 128 experts, top-4,
que e o que o noso pack roda.

Uso: parity_m3_geometry.py [K ...]     (default: 3 4)
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
import os
import sys

TOOL = EXL3_ROOT + '/pkgs/vllm-exl3/tools/gb10_minimax_m3_parity.py'


def load_tool():
    spec = importlib.util.spec_from_file_location("parity_m3", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    ks = [int(a) for a in argv[1:]] or [3, 4]
    mod = load_tool()
    # geometria real do MiniMax-M3.1-Flash (hidden 6144, intermediate 3072,
    # 128 experts, top-4) + topk do probe de mixed-K
    mod.A_HIDDEN = 6144
    mod.A_INTER = 3072
    mod.A_NEXP = 128
    mod.TOPK = 4
    import torch

    device = torch.device("cuda:0")
    out = []
    for k in ks:
        ent = mod._case_a_run(k, device)
        out.append(ent)
        print(
            f"case A GEOM K={k}: path={ent['path']} cosine={ent['cosine']} "
            f"cosine_silu={ent['cosine_silu_reference']} "
            f"prefill_path={ent['prefill_path']} "
            f"prefill_cosine={ent['prefill_cosine']} pass={ent['pass']}",
            flush=True,
        )
        torch.cuda.empty_cache()
    print(json.dumps({"geometry": {"hidden": mod.A_HIDDEN, "inter": mod.A_INTER,
                                   "nexp": mod.A_NEXP, "topk": mod.TOPK},
                      "case_a": out}), flush=True)
    return 0 if all(e["pass"] for e in out) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
