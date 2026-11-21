#!/usr/bin/env python3
"""Liga as tabelas de Engram que ja temos ao pack vLLM do turboderp.

O pack do turboderp guarda as duas tabelas de Engram
(`layers.{1,14}.engram.embed.{weight,scale}`) num arquivo de 188,83 GiB
(`ngram_embedding.safetensors`) que NAO baixamos -- as nossas duas shards do pack
de 2,9 bpw tem exatamente os mesmos quatro tensores, com shape e dtype identicos
(`F8_E4M3 [384006168,256]` + `F8_E8M0 [384006168,8]`; a de 14 tem 384016682
linhas). Conferido row a row por `pread` aleatorio em 06/11.

`_DiskEngramTable._locate` (vllm/models/deepseek_v4_1/common/engram.py:1031)
procura a chave no `model.safetensors.index.json` e, **se ela nao estiver la**,
varre todos os `*.safetensors` do diretorio do modelo. O index do turboderp nao
tem essas chaves (elas vivem no arquivo separado), entao basta symlinkar as
nossas duas shards dentro do diretorio servido -- zero bytes copiados.

Uso: turbo_prepare.py [--dst DIR] [--ours DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import struct

OURS = os.environ.get("OURS",
                       os.path.expanduser("~/models/DeepSeek-V4.1-Flash-1017"))
EXPECT = {
    "layers.1.engram.embed.weight": ("model-00047-of-00048.safetensors",
                                     "engram-layer1.safetensors"),
    "layers.1.engram.embed.scale": ("model-00047-of-00048.safetensors",
                                    "engram-layer1.safetensors"),
    "layers.14.engram.embed.weight": ("model-00048-of-00048.safetensors",
                                      "engram-layer14.safetensors"),
    "layers.14.engram.embed.scale": ("model-00048-of-00048.safetensors",
                                     "engram-layer14.safetensors"),
}


def header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dst",
        default=os.environ.get(
            "TURBO_DST", os.path.expanduser("~/models/DS41-turbo-2.68bpw-vllm")
        ),
    )
    ap.add_argument("--ours", default=OURS)
    args = ap.parse_args()

    # 1) o index do pack novo nao pode citar as chaves (senao o _locate vai so'
    #    no arquivo citado e morre se ele nao existir/nao tiver o tensor).
    idx_path = os.path.join(args.dst, "model.safetensors.index.json")
    with open(idx_path) as f:
        doc = json.load(f)
    stale = [k for k in doc["weight_map"] if "engram.embed" in k]
    if stale:
        for k in stale:
            doc["weight_map"].pop(k)
        with open(idx_path, "w") as f:
            json.dump(doc, f)
        print(f"removidas {len(stale)} chaves de engram do index")

    # 2) symlink das shards + conferencia de shape/dtype
    for key, (src, link) in EXPECT.items():
        src_path = os.path.join(args.ours, src)
        if not os.path.isfile(src_path):
            raise SystemExit(f"faltando {src_path}")
        meta = header(src_path).get(key)
        if meta is None:
            raise SystemExit(f"{key} nao esta em {src_path}")
        target = os.path.join(args.dst, link)
        if not os.path.lexists(target):
            os.symlink(src_path, target)
        print(f"  {key}: {meta['dtype']} {meta['shape']} -> {link}")
    print("engram ligado ao pack novo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
