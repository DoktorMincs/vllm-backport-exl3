#!/usr/bin/env python3
"""Converte o pack EXL3 do DS4.1 (layout exllamav3) para o layout do vLLM.

O pack de origem (``$MVPSRC``, default ``~/models/DeepSeek-V4.1-Flash-1017``)
vem do exllamav3. Quatro
familias nao podem ficar como EXL3 no vLLM e viram bf16 aqui:

  1. ``attn.wo_a.slice.{0..7}`` -- 8 lineares separadas (``exl3_mgemm``) contra
     UMA ``ColumnParallelLinear(is_bmm)`` no vLLM, com ``wo_a.weight``
     [8*r, hidden] (o caminho SM8x a le como [n_local_groups, r, hidden]);
  2. ``head`` -- o ``ParallelLMHead`` e' construido SEM quant_config (bf16);
  3. ``attn.indexer.wk`` -- ``ReplicatedLinear(quant_config=None)`` em
     attention.py (o proprio comentario diz que o pack de referencia e' bf16);
  4. ``attn.compressor.wkv|wgate`` -- ``MergedColumnParallelLinear(
     quant_config=None)`` em compressor.py. Aqui os nomes do checkpoint ficam
     por fatia (``compressor.wkv.weight``/``compressor.wgate.weight``), porque e'
     o ``stacked_params_mapping`` do modelo que faz a fusao.

Todo o resto continua EXL3. O mapa linear e' o da pipeline de kernels do
exllamav3 (``had(suh) -> hgemm -> had(svh)``), medido sobre a base identidade:

    Weff = M_su @ w @ M_sv,   M_su = had(I_in, suh),  M_sv = had(I_out, svh)

Verificacao (``--verify``): roda o forward de verdade do exllamav3
(``had_r_128(x,suh) -> x @ w -> had_r_128(y,svh)``) e compara com ``x @ Weff``.
NAO usar ``ext.reconstruct_had_slice`` como oraculo: aquele kernel aplica uma
Hadamard 2D (nos dois eixos), enquanto o forward aplica Hadamard so' no eixo de
entrada e no de saida -- medido, os dois divergem ~1,3 (relativo).

O destino tem symlink para cada shard original, um ``vllm-overlay.safetensors``
novo, um index sem os tensores convertidos e um ``quantization_config.json``
com ``non_routed_exl3`` explicito (o ledger do pack usa nomes do exllamav3 e nao
cobre os parametros fundidos do vLLM).

Uso:
  make_vllm_pack.py --verify [--layers 0,2]
  make_vllm_pack.py [--device cuda:0]
  make_vllm_pack.py --config-only | --index-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.environ.get("MVPSRC", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-1017"))
DST = os.environ.get("MVPDST", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-1017-vllm"))
OVERLAY = os.environ.get("MVPOVERLAY", "vllm-overlay.safetensors")
HEAD_CHUNK = 8192            # colunas de saida por vez no lm_head (128-alinhado)
WO_A_GROUPS = 8

# familias do ledger que viram bf16 (prefixo -> como montar o tensor)
PLAIN_FAMILIES = (
    "attn.indexer.wk",
    "attn.compressor.wkv",
    "attn.compressor.wgate",
)

# exllamav3 -> prefixo do modulo no vLLM pos-mapper/pos-stacking
FAMILY_REWRITE = (
    (".ffn.shared_experts.w1", ".ffn.shared_experts.gate_up_proj"),
    (".ffn.shared_experts.w3", ".ffn.shared_experts.gate_up_proj"),
    (".ffn.shared_experts.w2", ".ffn.shared_experts.down_proj"),
    (".attn.wq_a", ".attn.fused_wqa_wkv"),
    (".attn.wkv", ".attn.fused_wqa_wkv"),
)


# ---------------------------------------------------------------- checkpoint

def load_paths() -> tuple[dict, dict]:
    with open(os.path.join(SRC, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    with open(os.path.join(SRC, "quantization_config.json")) as f:
        qcfg = json.load(f)
    return index, qcfg


def shard_of(index: dict, name: str) -> str:
    shard = index.get(name)
    if shard is None:
        raise KeyError(f"{name} nao esta no index")
    return os.path.join(SRC, shard)


def read_exl3(index: dict, prefix: str) -> dict:
    """Trellis/suh/svh + marcador de codebook de um modulo EXL3."""
    out: dict = {}
    for suffix in ("trellis", "suh", "svh"):
        name = f"{prefix}.{suffix}"
        with safe_open(shard_of(index, name), framework="pt") as f:
            out[suffix] = f.get_tensor(name)
    out["mcg"] = f"{prefix}.mcg" in index
    name = f"{prefix}.mul1"
    if name in index:
        with safe_open(shard_of(index, name), framework="pt") as f:
            out["mul1"] = bool(int(f.get_tensor(name).reshape(1).item()) != 0)
    else:
        out["mul1"] = False
    shape = out["trellis"].shape
    out["in_dim"] = int(shape[0]) * 16
    out["out_dim"] = int(shape[1]) * 16
    out["K"] = int(shape[2]) // 16
    assert out["suh"].numel() == out["in_dim"], (prefix, "suh")
    assert out["svh"].numel() == out["out_dim"], (prefix, "svh")
    return out


# ------------------------------------------------------------------- kernels

def had_identity(ext, dim: int, scale_vec: torch.Tensor) -> torch.Tensor:
    """[dim, dim] fp16: a transformada de Hadamard com os scales do modulo."""
    eye = torch.eye(dim, dtype=torch.half, device=scale_vec.device)
    out = torch.empty_like(eye)
    ext.had_r_128(eye, out, scale_vec, None, 1.0)
    return out


def reconstruct(ext, spec: dict, device, fused: bool) -> torch.Tensor:
    """w [in, out] fp16; fused aplica suh/svh dentro do proprio kernel."""
    w = torch.empty(
        (spec["in_dim"], spec["out_dim"]), dtype=torch.half, device=device
    )
    trellis = spec["trellis"].to(device)
    if fused:
        ext.reconstruct_had_slice(
            w, trellis, spec["suh"].to(device), spec["svh"].to(device),
            spec["K"], spec["mcg"], spec["mul1"], 0,
        )
    else:
        ext.reconstruct(w, trellis, spec["K"], spec["mcg"], spec["mul1"])
    return w


def effective(ext, spec: dict, m_su: torch.Tensor | None, device) -> torch.Tensor:
    """[in, out] fp32 do mapa linear efetivo, pelo `exl3_torch` (torch puro).

    05/11 18:09: o caminho antigo (`ext.reconstruct` do exllamav3 + as duas
    Hadamards aqui) devolve o tile de 16x16 em OUTRA ordem -- cos ~0,006 contra
    o peso real, com o MESMO rms. Foi o que estragou o `vllm-overlay` (04/11) e
    depois o `dspark-overlay` (05/11 09:21/11:28); nos dois casos o `check()`
    era auto-consistente e nao viu. O `exl3_torch` do repo do Maya e' o
    decodificador que bate cos 0,9994 com o bf16 gerado pelo proprio engine.
    """
    if m_su is not None:
        raise RuntimeError("effective(): m_su nao e' mais suportado (nenhum chamador usa)")
    maya_tools = os.environ.get(
        "MAYA_TOOLS", os.path.expanduser("~/project-maya/tools/mm3")
    )
    if maya_tools not in sys.path:
        sys.path.insert(0, maya_tools)
    import exl3_torch as ET  # noqa: E402
    cb = 2 if spec.get("mul1") else 1 if spec.get("mcg") else 0
    w = ET.decode_raw(spec["trellis"].to(device), spec["suh"].to(device),
                      spec["svh"].to(device), cb)     # [out, in] fp32
    return w.t().contiguous()


def check(ext, spec: dict, weff: torch.Tensor, device, label: str) -> None:
    """Confere ``weff`` contra o caminho REAL do forward do exllamav3.

    O oraculo NAO pode ser ``reconstruct_had_slice``: aquele kernel aplica uma
    Hadamard **2D** (nos dois eixos, com suh/svh por fora), enquanto o
    ``exl3_gemm`` do forward aplica Hadamard so' no eixo de entrada e no de
    saida. Medido neste pack: o kernel fundido diverge do forward em ~1,3
    (relativo) e ``diag(suh) H w H diag(svh)`` casa com ele em 9,5e-4 -- ou
    seja, os dois caminhos sao transformadas diferentes. Aqui o teste e' o
    forward de verdade, que fecha em ~7e-4 (ruido de fp16).
    """
    w = reconstruct(ext, spec, device, fused=False)
    suh, svh = spec["suh"].to(device), spec["svh"].to(device)
    torch.manual_seed(0)
    x = torch.randn(16, spec["in_dim"], dtype=torch.half, device=device)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    y = (xh.float() @ w.float()).to(torch.half)
    yh = torch.empty_like(y)
    ext.had_r_128(y, yh, svh, None, 1.0)
    y_eff = x.float() @ weff
    err = (yh.float() - y_eff).abs().max().item() / max(yh.abs().max().item(), 1e-9)
    print(f"    {label}: err_forward={err:.3e}", flush=True)
    assert err < 2e-3, f"{label}: divergencia {err}"


def build_wo_a(ext, index: dict, layer: int, device, verify: bool) -> torch.Tensor:
    """8 slices -> [8*r, hidden] bf16, o layout do wo_a no vLLM."""
    ps = [read_exl3(index, f"layers.{layer}.attn.wo_a.slice.{g}")
          for g in range(WO_A_GROUPS)]
    base = ps[0]
    in_dim, out_s = base["in_dim"], base["out_dim"]
    for g, s in enumerate(ps):
        assert (s["in_dim"], s["out_dim"]) == (in_dim, out_s), (layer, g, "shape")
    # Cada fatia tem o SEU suh/svh (medido: o suh difere entre as 8). O peso
    # efetivo ja' embute a Hadamard de cada uma, entao a concatenacao no eixo
    # de saida e' exata -- nao ha transformada compartilhada a fatorar.
    parts = []
    for g, s in enumerate(ps):
        weff = effective(ext, s, None, device)
        if verify:
            check(ext, s, weff, device, f"slice {g}")
        parts.append(weff)
    full = torch.cat(parts, dim=1)                  # [in, 8*out_s]
    del parts
    return full.t().contiguous().to(torch.bfloat16)  # [out, in]


def build_plain(ext, index: dict, prefix: str, device, verify: bool) -> torch.Tensor:
    """Modulo EXL3 simples -> [out, in] bf16."""
    spec = read_exl3(index, prefix)
    weff = effective(ext, spec, None, device)
    if verify:
        check(ext, spec, weff, device, prefix)
    return weff.t().contiguous().to(torch.bfloat16)


def effective_ext_hadamard(ext, spec: dict, device, n0: int, n1: int) -> torch.Tensor:
    """[in, n1-n0] fp32 pelo decoder do exllamav3 + Hadamard do Python.

    Caminho independente do ``effective()`` (exl3_torch), usado so' como segunda
    opiniao do `head`. Nao usar ``ext.had_r_128`` sobre uma identidade para
    montar m_su/m_sv: em 07/11 o build do pack do turboderp reprovou assim
    (`head.weight cos=-0,005485`) e a causa nao era o pack -- o decode bruto do
    ext esta' certo (aqui, com as Hadamards aplicadas explicitamente, casa cos
    1,000000 com o `effective()`), mas compor as Hadamards pelo kernel nao
    reproduz mais o mapa efetivo na 1.6.5.
    """
    from exllamav3.modules.quant.exl3_lib.quantize import get_hadamard_dt

    trellis = spec["trellis"][:, n0 // 16: n1 // 16, :].to(device).contiguous()
    in_dim = int(trellis.shape[0]) * 16
    w = torch.empty((in_dim, n1 - n0), dtype=torch.half, device=device)
    ext.reconstruct_slice(w, trellis, spec["K"], spec["mcg"], spec["mul1"], 0)
    H = get_hadamard_dt(128, device, torch.float, 1.0 / (128 ** 0.5))
    x = w.float().reshape(-1, 128, n1 - n0)
    x = torch.einsum("ab,nbc->nac", H, x).reshape(-1, n1 - n0)
    x = x * spec["suh"].to(device).float().unsqueeze(1)
    x = (x.reshape(-1, 128) @ H).view(-1, n1 - n0)
    x = x * spec["svh"][n0:n1].to(device).float().unsqueeze(0)
    return x


def build_head(ext, index: dict, device, verify: bool) -> torch.Tensor:
    """lm_head EXL3 K=6 -> [vocab, hidden] bf16, em blocos alinhados de 128.

    Construido pelo ``effective()`` (exl3_torch), o mesmo caminho de
    `wo_a`/`compressor`/`indexer`, e conferido por `effective_ext_hadamard`
    (decode do ext + Hadamards do Python), que e' uma implementacao
    independente. O par (ext, python-H) casa cos 1,000000 com o exl3_torch nos
    dois packs (07/11).
    """
    spec = read_exl3(index, "head")
    in_dim, out_dim = spec["in_dim"], spec["out_dim"]
    trellis = spec["trellis"].to(device)
    out = torch.empty((out_dim, in_dim), dtype=torch.bfloat16, device=device)
    for n0 in range(0, out_dim, HEAD_CHUNK):
        n1 = min(n0 + HEAD_CHUNK, out_dim)
        sub = dict(spec)
        sub["trellis"] = trellis[:, n0 // 16: n1 // 16, :]
        sub["svh"] = spec["svh"][n0:n1]
        sub["out_dim"] = n1 - n0
        weff = effective(ext, sub, None, device)         # [in, chunk], fp32
        out[n0:n1] = weff.t().to(torch.bfloat16)
        if verify:
            alt = effective_ext_hadamard(ext, spec, device, n0, n1)
            x, y = weff.reshape(-1).double(), alt.reshape(-1).double()
            c = float(x @ y / max(x.norm() * y.norm(), 1e-30))
            print(f"    head [{n0}:{n1}]: cos(ext+pyH, torch)={c:.7f}", flush=True)
            assert c > 0.9995, f"head chunk {n0}: divergencia cos={c}"
            del alt
        del weff
    return out


def verify_overlay_torch(tensors: dict[str, torch.Tensor], index: dict) -> int:
    """Confere uma amostra do overlay contra o decodificador torch do Maya.

    Independente do exllamav3: `exl3_torch.decode` reproduz o bf16 que o engine
    do Maya gerou para `layers.{1,14}.engram.wkv` com cos 0,9994. Se este gate
    falhar, o overlay esta' errado e o modelo responde lixo -- foi exatamente o
    que aconteceu com o pack de 04/11 22:53.
    """
    sys.path.insert(0, os.environ.get(
        "MAYA_TOOLS", os.path.expanduser("~/project-maya/tools/mm3")
    ))
    try:
        import exl3_torch as ET  # noqa: E402
        from st import SafeTensors  # noqa: E402
    except ImportError as e:
        print(f"AVISO: decodificador torch indisponivel ({e}); gate pulado")
        return 0
    st = SafeTensors(SRC)
    sample = [n for n in sorted(tensors) if n.endswith(".attn.wo_a.weight")][:3]
    sample += [n for n in sorted(tensors) if ".compressor." in n or ".indexer." in n][:3]
    if "head.weight" in tensors:
        h = tensors["head.weight"]
        sample.append(("head.weight", 0, 128))
    bad = 0
    for item in sample:
        if isinstance(item, tuple):
            name, r0, r1 = item
            tr = ET.tensor(st, "head.trellis", "cpu")[:, r0 // 16:r1 // 16].contiguous()
            ref = ET.decode_raw(tr, ET.tensor(st, "head.suh", "cpu"),
                                ET.tensor(st, "head.svh", "cpu")[r0:r1],
                                ET.codebook(st, "head"))
            got = tensors[name][r0:r1].float()
        else:
            name = item
            prefix = name[: -len(".weight")]
            if ".wo_a.weight" in name:
                parts = [ET.decode(st, f"{prefix}.slice.{g}", "cpu")
                         for g in range(WO_A_GROUPS)]
                ref = torch.cat(parts, dim=0)
            else:
                ref = ET.decode(st, prefix, "cpu")
            got = tensors[name].float()
        a, b = got.reshape(-1).double(), ref.reshape(-1).double()
        c = float(a @ b / max(a.norm() * b.norm(), 1e-30))
        ok = c > 0.9995
        bad += 0 if ok else 1
        print(f"  check {name:44s} cos={c:.6f} {'OK' if ok else 'DIVERGE'}",
              flush=True)
    return bad


# -------------------------------------------------------------------- config

def is_converted(key: str) -> bool:
    if key == "head" or ".wo_a.slice." in key:
        return True
    return any(key.endswith("." + fam) for fam in PLAIN_FAMILIES)


def is_exl3(spec) -> bool:
    return isinstance(spec, dict) and spec.get("quant_format") == "exl3"


def converted_prefixes(ledger: dict) -> list[str]:
    """Prefixos do ledger que o overlay converte -- SO' os que sao EXL3 na origem.

    O pack do turboderp guarda `attn.indexer.wk` e `engram.wkv` como tensores
    DENSOS (F16), ao contrario do nosso (trellis K8/K5). Converter por sufixo,
    sem olhar o ledger, tentaria decodificar um tensor que nao tem trellis e
    apagaria do index um peso que carrega nativamente.
    """
    out: list[str] = []
    for key, spec in ledger.items():
        if not is_exl3(spec):
            continue
        if is_converted(key):
            out.append(key)
    return out


def vllm_coverage(ledger: dict) -> dict[str, dict]:
    """Ledger -> {'model.<prefix>': {'bits': K}} nos nomes pos-mapper do vLLM."""
    cov: dict[str, dict] = {}
    for key, spec in ledger.items():
        if not isinstance(spec, dict) or spec.get("quant_format") != "exl3":
            continue
        if ".experts." in key or is_converted(key):
            continue
        bits = spec.get("bits_per_weight")
        if bits is None:
            continue
        bits = int(bits)
        if not 2 <= bits <= 8:
            continue
        name = key
        for suffix, new in FAMILY_REWRITE:
            if key.endswith(suffix):
                name = key[: -len(suffix)] + new
                break
        full = key if name.startswith("model.") else "model." + name
        prev = cov.get(full)
        if prev is None or bits > prev["bits"]:
            cov[full] = {"bits": bits}
    return cov


def expert_layer_bits(ledger: dict) -> dict[str, int]:
    """K de cada camada para os experts roteados (uniforme dentro da camada)."""
    per: dict[int, set] = {}
    for key, spec in ledger.items():
        if ".experts." not in key or not key.endswith(".w1"):
            continue
        bits = (spec or {}).get("bits_per_weight")
        if bits is not None:
            # Meia-K (2.5) existe e NAO pode ser truncada aqui: e' o que o
            # `layer_bits` do quantization_config leva para os workers.
            v = float(bits)
            per.setdefault(int(key.split(".")[1]), set()).add(
                int(v) if v.is_integer() else v
            )
    out = {}
    for layer, ks in sorted(per.items()):
        if len(ks) != 1:
            raise RuntimeError(f"layer {layer}: K misto entre experts {ks}")
        out[str(layer)] = next(iter(ks))
    return out


def converted_tensor_names(index: dict, converted: list[str] | None = None) -> list[str]:
    """Tensores EXL3 originais que o overlay substitui (ou que nao tem modulo).

    O iterador de pesos do vLLM enumera os tensores **pelo arquivo**
    (``safe_open(...).keys()``), nao pelo ``model.safetensors.index.json``:
    apagar a chave do index nao impede o loader de oferecer ``head.trellis``
    (foi o erro "There is no module or parameter named 'head'"). A lista vai
    em ``skip_weight_names`` no quantization_config e o plugin a registra em
    ``register_skip_weight_names``.
    """
    conv = set(converted) if converted is not None else None
    out = []
    for key in index:
        prefix = key.rsplit(".", 1)[0]
        if conv is not None:
            if prefix in conv:
                out.append(key)
            continue
        if key.startswith("head.") or ".wo_a.slice." in key:
            out.append(key)
        elif any(f".{fam}." in key for fam in PLAIN_FAMILIES):
            out.append(key)
    # Tensores do pack novo que o modelo do v4.1 nao tem parametro nenhum
    # (o turboderp traz `compressor.ape`, que so' o v4.0 usa).
    for key in index:
        if key.startswith("layers.") and ".attn.compressor.ape" in key:
            out.append(key)
    return sorted(out)


def write_quant_config(qcfg: dict, dst: str, index: dict) -> None:
    ledger = qcfg.get("tensor_storage") or {}
    cov = vllm_coverage(ledger)
    lb = expert_layer_bits(ledger)
    base_bits = max(lb.values()) if lb else 3
    cb = qcfg.get("codebook", "mul1")
    cfg = {
        "quant_method": qcfg.get("quant_method", "exl3"),
        "version": qcfg.get("version"),
        "bits": int(base_bits),
        "layer_bits": lb,
        "codebook": cb,
        "mtp_experts": "exl3",
        "non_routed_exl3": {"codebook": cb, "layers": cov},
        # nada de non_routed_quantization: no shim do V4.1 ele liga weight_block_size
        # e faz o modelo tratar o shared expert como fp8 em blocos (padding).
        "non_routed_dtype_policy": "bf16_as_stored",
        # Diretorio do pack para o prescan das arenas de trellis do plugin.
        # Vai aqui (e nao por env) porque o quantization_config e' serializado
        # para todos os workers; a env so' chegou ao rank 0.
        "model_dir": dst,
        "skip_weight_names": converted_tensor_names(
            index, converted_prefixes(ledger)
        ),
        "head_bits": qcfg.get("head_bits"),
        "mtp_bits": qcfg.get("mtp_bits"),
        "out_scales": qcfg.get("out_scales"),
        "calibration": qcfg.get("calibration"),
        "original_quantization_config": qcfg.get("original_quantization_config"),
        "tensor_storage": ledger,
        "vllm_pack_note": (
            "wo_a (8 slices), head, indexer.wk e compressor.wkv/wgate em bf16 "
            "no overlay; non_routed_exl3 com os nomes pos-fusao do vLLM"
        ),
    }
    path = os.path.join(dst, "quantization_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"escrito {path}: {len(cov)} prefixos EXL3 densos, layer_bits={lb}")


def overlay_index(index: dict, extra: dict[str, str],
                  converted: list[str] | None = None) -> dict:
    """Index do destino: sem os tensores convertidos, com o overlay."""
    if converted is not None:
        conv = set(converted)
        new_map = {k: v for k, v in index.items()
                   if k.rsplit(".", 1)[0] not in conv}
        for name in extra:
            new_map[name] = OVERLAY
        return new_map
    new_map = {k: v for k, v in index.items()
               if ".wo_a.slice." not in k and not k.startswith("head.")}
    for fam in PLAIN_FAMILIES:
        suffix = "." + fam + "."
        for key in list(new_map):
            if key.startswith("layers.") and suffix in key:
                del new_map[key]
    for name in extra:
        new_map[name] = OVERLAY
    return new_map


def link_rest() -> None:
    """Symlinka o resto do pack e reescreve o config.json.

    O ``config.json`` do pack embute um bloco ``quantization_config`` que NAO
    tem bits/layer_bits/non_routed_exl3 (e o vLLM le esse bloco em vez do
    sidecar). Aqui ele sai: sem ``quantization_config`` embutido, o vLLM cai no
    arquivo ``quantization_config.json`` do diretorio -- o nosso, completo.
    """
    for entry in sorted(os.listdir(SRC)):
        if entry in ("model.safetensors.index.json", "quantization_config.json",
                     "config.json") or entry.startswith("maya-"):
            continue
        dst = os.path.join(DST, entry)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(SRC, entry), dst)
    with open(os.path.join(SRC, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    path = os.path.join(DST, "config.json")
    if os.path.islink(path):
        os.unlink(path)          # nunca escrever atraves do symlink do source
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1)


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="", help="subconjunto, ex. 0,2")
    ap.add_argument("--verify", action="store_true", help="checa kernel fundido")
    ap.add_argument("--config-only", action="store_true")
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    index, qcfg = load_paths()
    ledger = qcfg["tensor_storage"]
    conv = converted_prefixes(ledger)
    os.makedirs(DST, exist_ok=True)

    if args.config_only:
        write_quant_config(qcfg, DST, index)
        return 0
    if args.index_only:
        write_quant_config(qcfg, DST, index)
        extra = {f"layers.{L}.attn.wo_a.weight": OVERLAY for L in range(40)}
        extra["head.weight"] = OVERLAY
        with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": 0},
                       "weight_map": overlay_index(index, extra, conv)}, f)
        link_rest()
        print(f"index de orcamento escrito em {DST} (overlay ainda ausente)")
        return 0

    import exllamav3_ext as ext

    torch.cuda.set_device(args.device)
    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              if args.layers else list(range(40)))
    full_run = len(layers) == 40
    selected = set(layers)

    t0 = time.time()
    tensors: dict[str, torch.Tensor] = {}

    def add(name: str, t: torch.Tensor, label: str) -> None:
        print(f"{label}: {name} {tuple(t.shape)} {t.dtype} "
              f"{t.numel() * t.element_size() / 2**20:.0f} MiB "
              f"{time.time() - t0:.0f}s", flush=True)
        tensors[name] = t.cpu()
        del t
        torch.cuda.empty_cache()

    for layer in layers:
        add(f"layers.{layer}.attn.wo_a.weight",
            build_wo_a(ext, index, layer, args.device, args.verify), f"L{layer}")

    for key in conv:
        if ".wo_a.slice." in key or key == "head":
            continue          # head tem caminho proprio (build_head, em blocos)
        parts = key.split(".")
        layer = int(parts[1])
        if layer not in selected:
            continue
        # O nome no checkpoint E' o caminho do modulo (layers.N.attn.compressor.wkv);
        # cortar o "attn." aqui geraria layers.N.compressor.wkv e o loader nao
        # acharia o parametro.
        name = f"{key}.weight"
        add(name, build_plain(ext, index, key, args.device, args.verify), f"L{layer}")

    if full_run:
        add("head.weight", build_head(ext, index, args.device, args.verify), "head")

    if args.verify:
        print("verify: nada escrito")
        return 0
    if not full_run:
        print("AVISO: subconjunto -- nada escrito")
        return 0

    out = os.path.join(DST, OVERLAY)
    save_file(tensors, out)
    print(f"escrito {out} ({os.path.getsize(out) / 2**30:.2f} GiB)")

    # Gate de 05/11: o overlay de 04/11 22:53 ficou decorrelacionado (cos ~0,005)
    # porque foi gerado por uma versao anterior deste arquivo e o pack nunca foi
    # regerado quando o `effective()` mudou (o conversor foi editado 23:31, o
    # overlay era de 22:53). `--verify` NAO pega isso: ele compara o forward com
    # o proprio `w`, entao e' auto-consistente com qualquer layout. A unica
    # conferencia que vale e' contra um decodificador independente.
    if os.environ.get("EXL3_SKIP_OVERLAY_CHECK") != "1":
        bad = verify_overlay_torch(tensors, index)
        if bad:
            print(f"ERRO: {bad} tensores do overlay divergem do decodificador "
                  f"torch -> pack NAO confiavel", flush=True)
            return 3

    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 0},
                   "weight_map": overlay_index(index, {k: OVERLAY for k in tensors}, conv)}, f)
    link_rest()
    write_quant_config(qcfg, DST, index)
    print(f"pack vLLM pronto em {DST} em {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
