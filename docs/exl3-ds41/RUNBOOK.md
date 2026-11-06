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

### Medido em 06/11 — a diferença de chat é o CUSTO DO PASSO, não a aceitação

Mesmo prompt de chat, 128 tokens, `scripts/maya_step.py` (lê `drafts A of B
kept` do log dele) contra `scripts/step_by_prompt.py` (lê `/metrics`):

| | port (N=5) | Maya |
|---|---:|---:|
| tok/s chat | 30,7–33,9 | **86,0–86,3** |
| ms/passo | 78–83 | **32–35** |
| tok/rodada | 2,5–2,6 | 2,8–3,1 |
| drafts verificados/rodada | 5 | 2,3–2,5 (corta) |

Os dois entregam ~2,5–3 tokens por rodada: a diferença de 2,5× está no preço do
passo. O Maya roda um processo só (sem all-reduce de TP e sem rendezvous de PP)
e corta os drafts antes de verificar; nós pagamos ~12 coletivos por passo e
verificamos 5. Sobrepor estágios (DBO/microbatch) é a única alavanca de 2× que
sobra e a V2 a recusa com PP>1 + spec (`config/vllm.py:2806-2845`) — e a mesma
lista também recusa DBO com CUDA graphs, que aqui valem 7×; é trabalho de
código, não de configuração.

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

## Envs experimentais (todas default = comportamento de antes)

Nenhuma destas muda o caminho default; o `if` desligado não aloca buffer, não
cria stream e não importa nada pesado.

| env | default | o que faz | veredito |
|---|---|---|---|
| `VLLM_DSPARK_REST` | 0 | depois de uma rodada sem nenhum aceite, o passo seguinte daquela requisição não drafta (regra do Maya) | correto e neutro nesta máquina; o `conc=8` varia 109–184 tok/s entre corridas seguidas, o que engole o efeito |
| `VLLM_DSPARK_CONF_CUT` | vazio | `conf_min,cum_min`; corta os drafts de baixa confiança no scheduler | implementado; não medido (o corte por requisição quebra o decode uniforme dos CUDA graphs) |
| `VLLM_PP_TRANSPORT` | `nccl` | `host` troca o handoff de PP por staging em host (`pp_host.py`, relay por `/dev/shm`) | **mais lento** (46,8 contra 60,6 tok/s): os 7–8 ms/estágio são bolha, não NCCL |
| `VLLM_PP_HOST_*` | — | slots/teto/dir/timeout/eager do transporte acima | — |
| `VLLM_DS41_ENGRAM_TIME` | 0 | decompõe o custo do Engram (d2h/uniq/preads/h2d/wait/hash) a cada 20 passos | em decode: ~5,7 ms por camada por passo, preads dominam |
| `VLLM_DS41_ENGRAM_REUSE_HASH` | 0 | 1 reusa o hash do staging; 2 reusa e confere | correto (bit-idêntico) |
| `VLLM_DS41_ENGRAM_OVERLAP` | 0 | gathers das camadas em threads de trabalho + double buffer (+26 MiB/GPU) | sem ganho medível |
| `VLLM_DS41_ENGRAM_ASYNC_D2H` | 0 | D2H dos ids em buffer pinado + side stream | sem ganho medível |
| `VLLM_DS41_ENGRAM_THREADS` / `_THREAD_FLOOR` | — / 0 | força o nº de threads do helper / capa em 2 abaixo de 256 rows | ferramenta |

## Metodologia de medida (custou a noite de 05→06/11)

- **`conc=8` numa corrida só não vale nada**: no mesmo boot e mesma config,
  seis corridas deram 109,6 / 129,6 / 135,5 / 145,7 / 147,4 / 183,6 tok/s. Use
  `spec_bench.py --reps` e reporte a mediana; ou compare distribuições.
- A primeira medida depois do boot sai baixa (page cache do pack e das tabelas
  do Engram). Aqueça antes de comparar braços.
- O **stream guloso é determinístico** entre boots e entre eager/grafo: um probe
  de 256 tokens com `temperature=0` serve de oráculo de correção para qualquer
  mudança de transporte, grafo ou Engram (`scripts/dspark_probe.py`).
- `scripts/ab_bench.sh <tag>` roda conc1 + conc8 + prefill/TTFT/potência e grava
  `logs/ab-<tag>-*` com cabeçalho de envs e commit.
- **A aceitação varia mais que o passo.** Os mesmos 256 tokens saem a 26 tok/s
  num prompt analítico (2,17 tok/passo) e a 64 tok/s num prompt curto
  (5,43 tok/passo) no MESMO boot. Comparar braços com prompts diferentes mede o
  prompt: use `scripts/step_by_prompt.py`, que fixa 5 prompts e reporta
  tok/s, passo, aceitação e aceitação por posição.

## Decomposição do passo (medida em 06/11)

`scripts/step_by_prompt.py` lê `vllm:spec_decode_num_drafts_total` e
`..._num_accepted_tokens_total`; `ms/passo` sai de `dt / rodadas` — sem profiler
e sem sync, então não perturba o que mede.

| braço | tok/s | ms/passo | tok/passo |
|---|---:|---:|---:|
| sem spec (1 token) | 16,6 | **60** | 1,0 |
| N=1 (k=2) | 25,1 | **76** | 1,9 |
| N=5 (k=6) | 48,9 | **83** | 4,1 |

`passo ≈ 60 (alvo) + 14 (drafter fixo) + 1,7 × (k−1)`: 1,7 ms por token
verificado (cortar drafts por confiança vale ≤6% em chat) e 17% do passo é o
drafter — 80% é o alvo.

**O alvo de 60 ms é sincronização, não GEMM.** Kernel time por GPU em 3,1 s de
decode (nsys, `--cuda-graph-trace=node`): **58–78% do tempo ocupado está dentro
de kernels NCCL** (AllReduce de TP; Broadcast/SendRecv do PP) contra 10–20% de
EXL3 MoE + `reconstruct` + cutlass. O kernel do NCCL gira esperando o par, então
esses milissegundos são rendezvous: ~2 all-reduces por camada × 40 camadas por
passo, num par TP sem P2P.

O protocolo do PP, medido à parte (`VLLM_PP_PROTO_TIME=1`): metadados no gloo
**0,1–0,2 ms**, `isend` do tensor **0,1–1,6 ms**, espera do tensor do passo
anterior **0,01–0,06 ms**. Os 37–60 ms aparecem no `recv` bloqueante porque é
ali que a bolha do pipeline se materializa — **o transporte de PP não custa
nada**; o que custa é o rendezvous. É por isso que trocar NCCL por staging em
host (`VLLM_PP_TRANSPORT=host`) ficou mais lento: atacava o que não era o custo.

**Configuração de NCCL não conserta o rendezvous** (06/11, com a máquina
aquecida): `NCCL_PROTO=LL` 51,70 tok/s / 78,4 ms contra 51,75 / 78,3 do base
(Simple) — neutro no decode e 5–10% **pior** no prefill; `NCCL_MAX_NCHANNELS=1`
52,11 / 77,7; `NCCL_ALGO=Tree` **não sobe** (`NCCL error: invalid usage` no
`determine_available_memory`, não aceita os grupos strided de 2 ranks);
`CUDA_DEVICE_MAX_CONNECTIONS` 8 = 1. A alavanca que sobra é sobrepor
microbatches (DBO), não ajustar coletivo.

## Estado

Funcional e validado contra o motor Maya (paridade de prefill; decode 4x pior
sem o drafter e equivalente com ele, em prompt simples). O que continua em
aberto: o decode não escala com concorrência como no Maya e o custo por token em
chat/thinking é maior (a aceitação do drafter cai de ~0,95 para ~0,72 na
primeira posição quando o modelo está "pensando").
