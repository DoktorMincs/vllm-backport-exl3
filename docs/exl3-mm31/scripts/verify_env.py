#!/usr/bin/env python3
"""Verificação do ambiente: vllm do backport + plugin EXL3 + extensões.

Roda SEM GPU. Uso: python -I verify_env.py
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


import os
import sys

ok = True


def check(name, fn):
    global ok
    try:
        value = fn()
        print(f"OK    {name}: {value}")
        return value
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"FALHA {name}: {type(exc).__name__}: {exc}")
        return None


def _vllm():
    import vllm

    path = os.path.realpath(vllm.__file__)
    assert VLLM_REPO + "/" in path or "exl3-port" in path, path
    return f"{vllm.__version__} @ {path}"


def _torch():
    import torch

    return f"{torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}"


def _plugins():
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    return "general_plugins carregados"


def _quant_config():
    from vllm.model_executor.layers.quantization import get_quantization_config

    cls = get_quantization_config("exl3")
    assert cls is not None, "exl3 nao registrado"
    return f"{cls.__module__}.{cls.__name__}"


def _ext():
    import exllamav3_ext

    names = [n for n in dir(exllamav3_ext) if "mixedk" in n or n.startswith("exl3_moe")]
    assert hasattr(exllamav3_ext, "exl3_moe_mixedk"), "sem exl3_moe_mixedk"
    return f"{exllamav3_ext.__file__} ({len(names)} ops moe)"

def _plugin_c_ext():
    import vllm_exl3_c

    return getattr(vllm_exl3_c, "__file__", "?")


def _plugin_register():
    import vllm_exl3

    vllm_exl3.register()
    return "register() ok"


if __name__ == "__main__":
    check("torch", _torch)
    check("vllm", _vllm)
    check("vllm plugins", _plugins)
    check("quant exl3 registrado", _quant_config)
    check("exllamav3_ext", _ext)
    check("vllm_exl3_c", _plugin_c_ext)
    check("vllm_exl3.register", _plugin_register)
    print("\nRESULTADO:", "OK" if ok else "FALHAS")
    raise SystemExit(0 if ok else 1)
