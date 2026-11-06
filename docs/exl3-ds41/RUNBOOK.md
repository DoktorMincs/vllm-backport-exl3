# RUNBOOK — DeepSeek-V4.1-Flash-1017 EXL3 no vllm-backport (SM80)

Port do pack **EXL3** do DeepSeek-V4.1-Flash-1017 (o mesmo que o Project Maya
serve) para este fork, rodando em **10× RTX 3090 (sm_86, sem P2P)**. Geométrica
vencedora: `TP=2 × PP=5`, `--dtype bfloat16`, KV `fp8_ds_mla`, DSpark (drafter
`mtp.*`) ligado, Engram servido do disco.

Os caminhos abaixo são placeholders — como no [`docs/exl3-mm31`](../exl3-mm31),
copie os scripts para `$EXL3_ROOT/scripts` e ajuste as variáveis.

| variável | o que é |
|---|---|
| `EXL3_ROOT` | raiz do ambiente experimental (venv, logs, artefatos) |
| `MODEL_DIR` | pack convertido (symlinks para o pack exllamav3 + overlays) |
| `VLLM_REPO` | este checkout |
| `PROD_UNIT` / `WATCHDOG_UNIT` | unidades systemd que a janela de GPU para e religa |

## Config de boot

```
--model $MODEL_DIR --quantization exl3 --dtype bfloat16
--tensor-parallel-size 2 --pipeline-parallel-size 5
--max-model-len 98304 --gpu-memory-utilization 0.97
--kv-cache-dtype fp8_ds_mla --max-num-batched-tokens 512 --max-num-seqs 8
--engram-config '{"storage": "disk", "disk_threads": 128}'
--limit-mm-per-prompt '{"image": 0}'          # sem isto o encoder de imagem
                                              # (1025 tok/item) recusa chunk 512
--speculative-config '{"method": "dspark", "num_speculative_tokens": 5}'
# EAGER=0 (CUDA graphs): 7,2x no decode -- sem grafo, 3,5 tok/s.
# VLLM_PP_LAYER_PARTITION=8,8,10,8,6
# env: EXL3_FORCE_MIXEDK=1 VLLM_EXL3_V41_NATIVE_MOE=1 VLLM_EXL3_MOE_TP_ALIGN=128
```

`num_speculative_tokens` **tem que ser múltiplo de `n_predict=5`** do checkpoint
(N=6 é rejeitado no boot; N=10 sobe e é pior). A partição não é livre: o drafter
custa +4,6 GiB no último estágio, e tirar uma camada do `pp2` (10 camadas) para
o `pp4` (6) pede +2,3 GiB/GPU que não cabem no orçamento.

## Números medidos (10×3090 @ 200 W, `--max-model-len 64k`, 1 requisição)

| contexto | TTFT | prefill | decode `/v1/completions` | decode chat | potência (prefill / decode) |
|---|---:|---:|---:|---:|---:|
| 2 k | 1,5 s | 1.331 tok/s | 50,9 tok/s | 41,2 | 1,71 kW / 1,79 kW |
| 8 k | 4,1 s | 1.952 | 51,5 | 37,7 | 1,88 / 1,79 |
| 32 k | 14,0 s | 2.284 | 65,9 | 40,8 | 1,93 / 1,83 |
| 60 k | 25,6 s | 2.275 | 54,2 | 40,1 | 1,95 / 1,81 |
| 87 k | 37,2 s | 2.352 | 45,8 | – | 1,93 / 1,77 |

Concorrência (`--max-num-seqs 8`, prompts simples): **115–143 tok/s** de
agregado contra 30,6 sem drafter. A variação entre boots chega a 8% (o pack e as
tabelas do Engram são lidos do NVMe; depois de alguns boots boa parte está em
page cache) e a primeira medida após o boot sai ~5% alta.

Para comparação, o motor Maya na mesma máquina faz 632/1.218/2.175 tok/s de
prefill e 68/63/74 tok/s de decode em chat: **o port ganha no prefill e perde no
decode em chat** (o handoff de PP em NCCL custa ~44 J/token contra ~21 J/token
do Maya, que usa comunicação IPC própria).

## Contexto e KV

O KV já é **fp8** (`fp8_ds_mla`, ~3 KB/token). O pool escala com
`--max-model-len`:

| maxlen | pool de KV | observação |
|---|---:|---|
| 32.768 | 203.300 tok | |
| 65.536 | 189.032 tok | |
| **98.304** | **126.714 tok** | default; um prompt de 87 k roda inteiro |
| 131.072 | 184.087 tok | só com `util 0.98` **e** o patch abaixo; um prompt de ~118 k derruba o engine (`k_bf16` do indexer sobre o contexto todo) |

O fork também aceita `nvfp4_ds_mla` e `turboquant_*` (KV de 4 bits), que
reduzem o KV mas não o pico de ativação — não são o que limita o contexto.

## Armadilhas (todas custaram medida)

1. **Decodificador EXL3 do pack.** Nenhum caminho do exllamav3
   (`ext.reconstruct`, `reconstruct_had_slice`) devolve o peso certo: o tile de
   16×16 sai em outra ordem (cos ~0,006, mesmo rms), e um `check()` que compare
   o forward com o próprio `w` fecha em 7e-4 e esconde o erro. O decodificador
   confiável é torch puro (`exl3_torch`), validado a cos 0,9994 contra o bf16
   que o engine do Maya gera.
2. **O `dspark-overlay`** (`main_proj` + `attn.wo_a` dos 3 blocos do drafter)
   tinha o mesmo defeito — e `main_proj` é por onde passa TODO o KV de contexto
   do draft. Sintoma: `Drafted: N, Accepted: 0`. Depois de regerado:
   aceitação por posição 1,000/1,000/1,000/0,944/0,944 (5,89 de 6).
3. **Iterador de pesos lê o ARQUIVO, não o índice.** Apagar nomes do
   `model.safetensors.index.json` não impede o loader de oferecê-los; o boot
   morre com `There is no module or parameter named ...`. Use
   `skip_weight_names` no `quantization_config.json` (`register_skip_weight_names`).
4. **Gate multimodal**: `vision_config` no config monta orçamento de encoder de
   imagem e recusa chunk 512 → `--limit-mm-per-prompt '{"image": 0}'`.
5. **O gancho de timing mora no runner V2** (`vllm/v1/worker/gpu/model_runner.py`
   + `gpu/pp_utils.py`); o `gpu_model_runner.py` (V1) nunca executa neste fork.
6. **`coop_autotune` mede dentro da captura** do CUDA graph e derruba com
   "operation not permitted when stream is capturing": em cache miss ele faz
   `cudaStreamSynchronize`. O autotuner tem que escolher sem medir quando
   `cudaStreamIsCapturing()`.
7. **Engram de disco**: o helper C precisa ser compilado dentro do `cache_dir`
   (`os.replace` entre tmpfs e disco dá EXDEV) e o gather usa `step=max(1, ...)`
   — com 128 linhas em 2 threads cada camada custava ~70 ms por passo.
8. **Prefill workspace bf16** (`flashmla_sparse.get_prefill_workspace_size`):
   era dimensionado por `max_model_len` (755 MB por rank a 128 k) mesmo com
   chunk 512. Este branch usa `min(max_model_len, max_num_batched_tokens)`.

## Estado

Funcional e validado contra o motor Maya (paridade de prefill; decode 4x pior
sem o drafter e equivalente com ele, em prompt simples). O que continua em
aberto: o decode não escala com concorrência como no Maya e o custo por token em
chat/thinking é maior (a aceitação do drafter cai de ~0,95 para ~0,72 na
primeira posição quando o modelo está "pensando").
