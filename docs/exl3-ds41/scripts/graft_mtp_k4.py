#!/usr/bin/env python3
"""Cria um pack do turboderp (2,68bpw) com o DSpark do NOSSO pack (K=4).

Por que (medido 07/11): o MTP do turboderp tem os experts roteados em **K=2**,
o nosso em **K=4**. No mesmo prompt de chat o drafter do turboderp aceita 1,46
token/rodada contra ~2,4 do nosso, o que custa ~30% do decode com spec — o
resto do passo e' igual.

O que muda em relacao ao dst convertido do turboderp:

  shards 1..39          symlink para o pack do turboderp (intocados)
  shards 40 e 41        REESCRITOS sem os tensores `mtp.*` (2,39 + 1,30 GiB)
  mtp-k4-1/2            copia bit-a-bit dos shards 42/43 do nosso pack
                        (100% `mtp.*`, mesmos nomes)
  index                 mtp.* do turboderp -> mtp.* do nosso, apontando os
                        novos arquivos; resto igual
  quantization_config   bloco do draft vem do nosso (layer_bits 40-42 = 4,
                        mtp_bits 4, sem `model.main_proj` EXL3 porque o nosso
                        traz `mtp.0.main_proj.weight` bf16 no overlay),
                        skip_weight_names |= nomes mtp do nosso
  dspark-overlay        o nosso (wo_a + main_proj bf16 do draft K=4)

Uso: graft_mtp_k4.py [--dst DIR] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

TURBO_SRC = os.environ.get(
    "TURBO_SRC", os.path.expanduser("~/models/DS41-turbo-2.68bpw")
)
TURBO_DST = os.environ.get(
    "TURBO_DST", os.path.expanduser("~/models/DS41-turbo-2.68bpw-vllm")
)
OUR_SRC = os.environ.get(
    "OURS", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-1017")
)
OUR_DST = os.environ.get(
    "OURS_VLLM", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-1017-vllm")
)
DST = os.environ.get(
    "TURBO_MTP4_DST",
    os.path.expanduser("~/models/DS41-turbo-2.68bpw-mtp4-vllm"),
)

REWRITE = ("model-00040-of-00041.safetensors", "model-00041-of-00041.safetensors")
OUR_MTP = {
    "model-00042-of-00043.safetensors": "mtp-k4-1.safetensors",
    "model-00043-of-00043.safetensors": "mtp-k4-2.safetensors",
}
DRAFT_LAYERS = ("40", "41", "42")


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def rewrite_without_mtp(src: str, dst: str) -> None:
    """Copia um shard do turboderp sem os tensores `mtp.*`."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out = {}
    with safe_open(src, framework="pt", device="cpu") as f:
        for name in f.keys():
            if name.startswith("mtp."):
                continue
            out[name] = f.get_tensor(name)
    save_file(out, dst, metadata={"format": "pt"})
    print(f"  {os.path.basename(dst)}: {len(out)} tensores, "
          f"{os.path.getsize(dst) / 2**30:.2f} GiB", flush=True)
    del out
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dst = args.dst

    if os.path.exists(os.path.join(dst, "model.safetensors.index.json")) and not args.force:
        raise SystemExit(f"{dst} ja' parece convertido; use --force")
    os.makedirs(dst, exist_ok=True)

    t_idx = load_json(os.path.join(TURBO_DST, "model.safetensors.index.json"))
    o_idx = load_json(os.path.join(OUR_DST, "model.safetensors.index.json"))
    t_map, o_map = t_idx["weight_map"], o_idx["weight_map"]

    # 1) symlinks dos shards que ficam iguais + arquivos pequenos
    files = sorted({v for v in t_map.values()
                    if v.startswith("model-") and v.endswith(".safetensors")})
    for f in files:
        s = os.path.join(TURBO_SRC, f)
        d = os.path.join(dst, f)
        if f in REWRITE or os.path.lexists(d):
            continue
        os.symlink(s, d)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "README.md", ".gitattributes", "LICENSE", "engram-layer1.safetensors",
              "engram-layer14.safetensors"):
        s = os.path.join(TURBO_DST, f)
        d = os.path.join(dst, f)
        if os.path.lexists(s) and not os.path.lexists(d):
            os.symlink(os.path.realpath(s), d)
    for f in ("vllm-overlay.safetensors", "dspark-overlay.safetensors"):
        s = os.path.join(TURBO_DST, f)
        d = os.path.join(dst, f)
        if f == "dspark-overlay.safetensors":
            continue                      # vem do nosso pack
        if os.path.exists(s) and (os.path.islink(d) or not os.path.lexists(d)):
            if os.path.islink(d):
                os.unlink(d)
            shutil.copyfile(s, d)
    print("1) shards symlinkados", flush=True)

    # 2) shards 40/41 sem mtp
    print("2) reescrevendo", REWRITE, flush=True)
    for f in REWRITE:
        rewrite_without_mtp(os.path.join(TURBO_SRC, f), os.path.join(dst, f))

    # 3) MTP do nosso pack: copia bit-a-bit dos dois shards (100% mtp.*)
    print("3) copiando MTP K=4", flush=True)
    for src, new in OUR_MTP.items():
        shutil.copyfile(os.path.join(OUR_SRC, src), os.path.join(dst, new))
        print(f"  {new}: {os.path.getsize(os.path.join(dst, new)) / 2**30:.2f} GiB",
              flush=True)
    # overlay do drafter (wo_a + main_proj bf16), construido do nosso MTP
    shutil.copyfile(os.path.join(OUR_DST, "dspark-overlay.safetensors"),
                    os.path.join(dst, "dspark-overlay.safetensors"))

    # 4) index
    print("4) index", flush=True)
    new_map = {k: v for k, v in t_map.items() if not k.startswith("mtp.")}
    n_mtp = 0
    for k, v in o_map.items():
        if not k.startswith("mtp."):
            continue
        new_map[k] = OUR_MTP.get(v, v)
        n_mtp += 1
    print(f"  mtp do nosso: {n_mtp} tensores")
    missing = [k for k, v in new_map.items()
               if not os.path.exists(os.path.join(dst, v))]
    if missing:
        raise SystemExit(f"index aponta para arquivo ausente: {missing[:5]}")
    dump_json({"metadata": t_idx.get("metadata", {}), "weight_map": new_map},
              os.path.join(dst, "model.safetensors.index.json"))

    # 5) quantization_config: bloco do draft do nosso
    print("5) quantization_config", flush=True)
    tq = load_json(os.path.join(TURBO_DST, "quantization_config.json"))
    oq = load_json(os.path.join(OUR_DST, "quantization_config.json"))
    tq["layer_bits"].update({k: oq["layer_bits"][k] for k in DRAFT_LAYERS})
    cov = tq["non_routed_exl3"]["layers"]
    for k in [k for k in cov if k == "model.main_proj"
              or k.startswith(tuple(f"model.layers.{i}." for i in DRAFT_LAYERS))]:
        cov.pop(k)
    cov.update({k: v for k, v in oq["non_routed_exl3"]["layers"].items()
                if k.startswith(tuple(f"model.layers.{i}." for i in DRAFT_LAYERS))})
    tq["mtp_bits"] = oq.get("mtp_bits")
    tq["mtp_experts"] = oq.get("mtp_experts")
    skip = set(tq.get("skip_weight_names") or [])
    skip |= {k for k in (oq.get("skip_weight_names") or []) if k.startswith("mtp.")}
    tq["skip_weight_names"] = sorted(skip)
    tq["model_dir"] = dst
    dump_json(tq, os.path.join(dst, "quantization_config.json"))
    print(f"  cov draft: {len(cov)} prefixos, layer_bits 40-42="
          f"{[tq['layer_bits'][k] for k in DRAFT_LAYERS]}, "
          f"skip={len(skip)}", flush=True)

    # 6) config (o do turboderp convertido ja' tem os taps do drafter)
    cfg = load_json(os.path.join(TURBO_DST, "config.json"))
    dump_json(cfg, os.path.join(dst, "config.json"))
    print("6) config copiado")
    print(f"pronto: {dst}")
    print("  boot: PACK=%s ./scripts/serve_turbo_ds41.sh 3000 65536" % dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
