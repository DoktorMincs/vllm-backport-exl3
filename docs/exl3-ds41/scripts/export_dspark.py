#!/usr/bin/env python3
"""Exporta o drafter DSpark (`mtp.*`) do pack exllamav3 para o pack do vLLM.

Por que existe: o `make_vllm_pack.py` descarta `mtp.*` (7,7 GiB) porque o
mapper do vLLM nao conhecia o drafter. O fork agora tem
`DSparkDeepseekV4ForCausalLM` (registrado como DSparkV41DraftModel) e ele LE
o mesmo checkpoint, remapeando `mtp.{i}.<rest>` -> `model.layers.{40+i}.<rest>`.

O que falta e' o que este script produz:

1. **`attn.wo_a` do draft em bf16.** O pack so' tem as 96 fatias
   `mtp.{i}.attn.wo_a.slice.{0..7}.{trellis,suh,svh,mul1}`, que ficam em
   `skip_weight_names`. O `wo_a` do vLLM e' UMA `ColumnParallelLinear(is_bmm)`
   bf16 `[8*r, hidden]`. Reconstroi com a mesma matematica do `build_wo_a`
   (Weff = M_su @ w @ M_sv por fatia, concat no eixo de saida).
   O tensor sai com o NOME DO CHECKPOINT (`mtp.{i}.attn.wo_a.weight`), nao com
   o nome de runtime: e' o `_remap_dspark_name` que traduz. O loader do alvo
   ignora qualquer `mtp.*`, entao o tensor extra e' inofensivo para ele.

2. **Cobertura EXL3 dos densos do draft.** O `quantization_config.json` do
   pack nao declara nenhum prefixo de draft; com `bf16_as_stored` o plugin
   monta `UnquantizedLinearMethod` e os `*.trellis` ficam sem parametro
   (KeyError em dspark.py). Aqui entram os 16 prefixos pos-fusao com o K real
   lido do trellis, e `layer_bits[40..42]` com o K dos experts roteados.

Uso:
  export_dspark.py --dry-run          # so' le e imprime o que faria
  export_dspark.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_vllm_pack as mp  # noqa: E402  (read_exl3/effective/shard_of)

DST = mp.DST
SRC = mp.SRC
OVERLAY = "dspark-overlay.safetensors"
N_DRAFT = 3
WO_A_GROUPS = mp.WO_A_GROUPS

# exllamav3 (source) -> prefixo pos-fusao do vLLM
DENSE_FAMILIES = (
    # (nomes no pack de origem, prefixo no modulo do vLLM)
    (("attn.wq_a", "attn.wkv"), "attn.fused_wqa_wkv"),
    (("attn.wq_b",), "attn.wq_b"),
    (("attn.wo_b",), "attn.wo_b"),
    (("ffn.shared_experts.w1", "ffn.shared_experts.w3"),
     "ffn.shared_experts.gate_up_proj"),
    (("ffn.shared_experts.w2",), "ffn.shared_experts.down_proj"),
)


def build_wo_a_stage(ext, src_index: dict, stage: int, device, verify: bool):
    """8 fatias -> [8*r, hidden] bf16 (mesmo caminho de build_wo_a)."""
    ps = [mp.read_exl3(src_index, f"mtp.{stage}.attn.wo_a.slice.{g}")
          for g in range(WO_A_GROUPS)]
    in_dim, out_s = ps[0]["in_dim"], ps[0]["out_dim"]
    for g, s in enumerate(ps):
        assert (s["in_dim"], s["out_dim"]) == (in_dim, out_s), (stage, g, "shape")
    parts = []
    for g, s in enumerate(ps):
        weff = mp.effective(ext, s, None, device)
        if verify:
            mp.check(ext, s, weff, device, f"mtp.{stage} slice {g}")
        parts.append(weff)
    full = torch.cat(parts, dim=1)                 # [in, 8*out_s]
    del parts
    return full.t().contiguous().to(torch.bfloat16), in_dim, out_s


def dense_bits(src_index: dict, stage: int) -> dict[str, int]:
    """K real de cada familia densa do draft (le o trellis, nao supoe)."""
    out: dict[str, int] = {}
    for names, target in DENSE_FAMILIES:
        ks = {mp.read_exl3(src_index, f"mtp.{stage}.{n}")["K"] for n in names}
        assert len(ks) == 1, (stage, target, ks)
        out[target] = ks.pop()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="confere cada fatia contra o forward real (lento)")
    args = ap.parse_args()

    dst_index_path = os.path.join(DST, "model.safetensors.index.json")
    with open(dst_index_path) as f:
        dst = json.load(f)
    with open(os.path.join(SRC, "model.safetensors.index.json")) as f:
        src_index = json.load(f)["weight_map"]

    qpath = os.path.join(DST, "quantization_config.json")
    with open(qpath) as f:
        qcfg = json.load(f)
    cov = qcfg.setdefault("non_routed_exl3", {}).setdefault("layers", {})
    layer_bits = qcfg.setdefault("layer_bits", {})
    base_layer = 40  # num_hidden_layers do alvo: o draft vive em 40..42

    # Os trellis de `mtp.{i}.attn.wo_a.slice.{g}` existem nos DOIS packs, mas o
    # `make_vllm_pack` so' os poe em `skip_weight_names` quando o ledger do pack
    # de origem tem entradas `mtp.*` -- o ledger do turboderp nao tem nenhuma
    # (`tensor_storage` com 47.008 chaves, zero `mtp.`). Sem o skip, o iterador
    # de pesos do draft entrega `...slice.0.mul1` e o `dspark.py:585` morre com
    # `KeyError: 'model.layers.0.attn.wo_a.slice.0.mul1'` (b3, 03:04).
    # O overlay logo abaixo substitui esses 24 prefixos por bf16.
    skips = qcfg.setdefault("skip_weight_names", [])
    _known = set(skips)
    _added = 0
    for pref in {k.rsplit(".", 1)[0] for k in src_index
                 if k.startswith("mtp.") and ".attn.wo_a.slice." in k}:
        if pref + ".trellis" not in src_index:
            continue
        for k in src_index:
            if k.rsplit(".", 1)[0] != pref:
                continue
            if k not in _known:
                skips.append(k)
                _known.add(k)
                _added += 1
    if _added:
        qcfg["skip_weight_names"] = sorted(skips)
        print(f"skip_weight_names: +{_added} tensores de mtp.*.attn.wo_a.slice.*")

    # K dos experts roteados por stage (w1/w2/w3 devem bater)
    expert_bits = {}
    for i in range(N_DRAFT):
        ks = {mp.read_exl3(src_index, f"mtp.{i}.ffn.experts.0.w{n}")["K"]
              for n in (1, 2, 3)}
        assert len(ks) == 1, (i, "experts K", ks)
        expert_bits[base_layer + i] = ks.pop()

    bits_plan = {i: dense_bits(src_index, i) for i in range(N_DRAFT)}
    print("K densos do draft:", {i: sorted(set(b.values()))
                                 for i, b in bits_plan.items()})
    print("K dos experts do draft:", expert_bits)

    if args.dry_run:
        for i in range(N_DRAFT):
            print(f"  mtp.{i}.attn.wo_a.weight -> "
                  f"model.layers.{base_layer + i}.attn.wo_a.weight (bf16)")
            for target, k in bits_plan[i].items():
                print(f"  cover model.layers.{base_layer + i}.{target} bits={k}")
        for target, k in dense_bits(src_index, 0).items():
            print(f"  (idem main_proj bits={k})")
        return 0

    import exllamav3_ext as ext  # noqa: E402

    torch.cuda.set_device(args.device)
    tensors: dict[str, torch.Tensor] = {}
    for i in range(N_DRAFT):
        w, in_dim, out_s = build_wo_a_stage(ext, src_index, i,
                                            torch.device(args.device),
                                            args.verify)
        name = f"mtp.{i}.attn.wo_a.weight"
        tensors[name] = w
        print(f"{name}: {tuple(w.shape)} {w.dtype} "
              f"({in_dim} in, {out_s * WO_A_GROUPS} out)", flush=True)
        del w

    out_path = os.path.join(DST, OVERLAY)
    save_file(tensors, out_path)
    print(f"escrito {out_path} ({os.path.getsize(out_path) / (1 << 30):.2f} GiB)")
    del tensors

    for i in range(N_DRAFT):
        dst["weight_map"][f"mtp.{i}.attn.wo_a.weight"] = OVERLAY
    with open(dst_index_path, "w") as f:
        json.dump(dst, f)

    tgt_bits = bits_plan[0]
    for i in range(N_DRAFT):
        layer = base_layer + i
        for target, k in bits_plan[i].items():
            assert k == tgt_bits[target], (i, target, "K inconsistente")
            cov[f"model.layers.{layer}.{target}"] = {"bits": k}
        layer_bits[str(layer)] = expert_bits[layer]

    # main_proj vem de mtp.0 (fora das camadas)
    main_proj_k = mp.read_exl3(src_index, "mtp.0.main_proj")["K"]
    cov["model.main_proj"] = {"bits": main_proj_k}
    print(f"cover model.main_proj bits={main_proj_k}")

    with open(qpath, "w") as f:
        json.dump(qcfg, f, indent=1)
    print(f"escrito {qpath}: {len(cov)} prefixos densos, "
          f"layer_bits={sorted(layer_bits, key=int)[-4:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
