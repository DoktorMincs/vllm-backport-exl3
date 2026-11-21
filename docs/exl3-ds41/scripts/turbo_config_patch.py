#!/usr/bin/env python3
"""Completa o config.json do pack convertido com as chaves que o DS4.1 do
turboderp nao traz e o nosso drafter exige.

Hoje e' uma chave so': `eagle_aux_hidden_state_layer_ids`. O
`DSparkSpeculator`/`ExtractHiddenStates` le' os taps do drafter dai'
(`v1/spec_decode/extract_hidden_states.py:65-71` levanta ValueError quando a
chave falta) e ela tambem entra na chave de cache do grafo
(`config/speculative.py:626-634`). O nosso pack tem a lista; o do turboderp nao
tem -- o resto do config e' identico (conferido chave a chave em 07/11).

Nao inventa o valor: le' do nosso pack e copia. Se o nosso nao tiver, falha.

Uso: turbo_config_patch.py [--dst DIR] [--ours DIR]
"""
from __future__ import annotations

import argparse
import json
import os

KEYS = ("eagle_aux_hidden_state_layer_ids",)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dst",
        default=os.environ.get(
            "TURBO_DST", os.path.expanduser("~/models/DS41-turbo-2.68bpw-vllm")
        ),
    )
    ap.add_argument(
        "--ours",
        default=os.environ.get(
            "OURS_VLLM",
            os.path.expanduser("~/models/DeepSeek-V4.1-Flash-1017-vllm"),
        ),
    )
    args = ap.parse_args()

    src = json.load(open(os.path.join(args.ours, "config.json")))
    path = os.path.join(args.dst, "config.json")
    if os.path.islink(path):
        raise SystemExit(f"{path} e' symlink; rode make_vllm_pack antes (link_rest "
                         "escreve um config proprio)")
    cfg = json.load(open(path))

    changed = False
    for k in KEYS:
        if k in cfg:
            continue
        if k not in src or src[k] in (None, [], {}):
            raise SystemExit(f"{k} nao esta no nosso config; nao ha o que copiar")
        cfg[k] = src[k]
        changed = True
        print(f"+ {k} = {src[k]}")

    if not changed:
        print("config ja' completo")
        return 0
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"escrito {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
