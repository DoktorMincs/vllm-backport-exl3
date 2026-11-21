# Conversão do pack DS4.1 (exllamav3 → vLLM)

Estes são os scripts que montam o diretório servido pelo qual o engine boota. Todos
os caminhos têm default em `~/…` e podem ser trocados por variável de ambiente —
nada aqui depende de host, unidade systemd ou caminho absoluto.

O pack de origem é o **exllamav3** (o nosso, ou o do
`turboderp/DeepSeek-V4.1-Flash-1017-exl3` @ `2.68bpw`); o destino é um diretório com
symlink para cada shard, um overlay bf16 novo, um index sem os tensores convertidos
e um `quantization_config.json`. **Nenhum peso é distribuído neste repositório** — o
pack é derivado do DeepSeek-V4.1-Flash-1017 e continua sob os termos do modelo.

## Pré-requisitos

| item | para quê |
|---|---|
| pack de origem em formato exllamav3 | entrada (`MVPSRC` / `TURBO_SRC`) |
| `mw00/project-maya`, `tools/mm3` | o oráculo `exl3_torch.py` (`decode_raw`) que confere cada tensor convertido; sem ele `make_vllm_pack.py` não roda |
| plugin `vllm-exl3` já aplicado | `export_dspark.py` lê o ledger do plugin |
| GPU com memória livre | `make_vllm_pack.py` e `export_dspark.py` decodificam na GPU 0 |

```bash
git clone https://github.com/mw00/project-maya ~/project-maya
export MAYA_TOOLS=~/project-maya/tools/mm3
```

## Variáveis de ambiente

| variável | default | o que é |
|---|---|---|
| `EXL3_PORT` / `EXL3_ROOT` | `~/exl3-port` | raiz do ambiente (os scripts precisam estar em `$EXL3_ROOT/scripts`) |
| `MODELS` | `~/models` | diretório dos packs |
| `MVPSRC`, `MVPDST`, `MVPOVERLAY` | `$MODELS/DeepSeek-V4.1-Flash-1017`… | entrada/saída do `make_vllm_pack.py` |
| `TURBO_SRC`, `TURBO_DST` | `$MODELS/DS41-turbo-2.68bpw{,-vllm}` | o pack do turboderp e o destino convertido |
| `OURS`, `OURS_VLLM` | `$MODELS/DeepSeek-V4.1-Flash-1017{,-vllm}` | o nosso pack (doador das tabelas de Engram e do drafter K=4) |
| `PYTHON` | `$EXL3_ROOT/venv/bin/python` | interpretador com torch + safetensors |
| `MAYA_TOOLS` | `~/project-maya/tools/mm3` | onde está o `exl3_torch` |

## Pipeline

```bash
cp docs/exl3-ds41/scripts/*.py docs/exl3-ds41/scripts/*.sh "$EXL3_ROOT/scripts/"
"$EXL3_ROOT/scripts/turbo_pack_build.sh"        # 4 etapas, morre se houver GPU ocupada
"$EXL3_ROOT/scripts/turbo_pack_build.sh" --verify   # idem, com o gate por tensor
```

1. **`make_vllm_pack.py`** — layout exllamav3 → vLLM. Quatro famílias não podem ficar
   em EXL3 no vLLM e viram bf16 aqui: `attn.wo_a.slice.{0..7}` (8 lineares separadas
   contra uma `ColumnParallelLinear(is_bmm)`), `head` (o `ParallelLMHead` é construído
   sem `quant_config`), `attn.indexer.wk` e `attn.compressor.wkv|wgate`. O resto
   continua EXL3. Escreve o overlay, o index e o `quantization_config.json`.
2. **`export_dspark.py`** — overlay do drafter: `attn.wo_a` bf16 reconstruído das 96
   fatias, `main_proj`, e a cobertura EXL3 dos densos do `mtp.*` (sem ela o plugin
   monta `UnquantizedLinearMethod` e o loader morre em `KeyError` no `*.trellis`).
3. **`turbo_prepare.py`** — liga as tabelas de Engram do nosso pack por symlink e tira
   as chaves delas do index (zero bytes copiados; o `ngram_embedding.safetensors` do
   turboderp, 188,83 GiB, não é baixado). Confere `dtype`/`shape` de cada uma.
4. **`turbo_config_patch.py`** — grava no `config.json` os taps do drafter.

Ferramentas avulsas: `pack_budget.py --pack DIR --levels` dá a partição de camadas
ótima (e a memória por estágio); `graft_mtp_k4.py` enxerta o nosso drafter K=4 no pack
do turboderp (o MTP de lá é K=2 e a aceitação cai).

## As duas armadilhas de conversão

- **skip dos `mtp.*.attn.wo_a.slice.*`**: o `make_vllm_pack` só registrava esses 96
  tensores quando o ledger de origem tinha prefixos `mtp.*` — o do turboderp tem
  47.008 chaves e **zero** `mtp.`, e o drafter morria em
  `KeyError: 'model.layers.0.attn.wo_a.slice.0.mul1'`. O `export_dspark.py` agora
  adiciona os nomes por conta própria (gate: achar o `trellis` no cabeçalho).
- **overlay do `head`**: nunca compor `reconstruct_slice` com `had_r_128` sobre a
  identidade (cos −0,0055 no exllamav3 1.6.5). O `build_head` usa `effective()` do
  `exl3_torch` e confere cada bloco de 8192 colunas contra uma segunda implementação
  (decode do ext + Hadamards em Python), cos > 0,9995.

Detalhe dos números e da partição: [`../RUNBOOK.md`](../RUNBOOK.md).
