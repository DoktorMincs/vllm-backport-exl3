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
| **98.304** | **126.714 tok** | pack 2,9 bpw; o de 2,68 bpw dá **1.448.915** |
| 131.072 | 184.087 tok | pack 2,9 bpw, só com `util 0.98` e o patch abaixo; um prompt de ~118 k derruba o engine (`k_bf16` do indexer sobre o contexto todo) |

A partição faz toda a diferença e precisa ser recalculada por pack:
`pack_budget.py --pack <dir> --levels` dá o ótimo contíguo. No pack de 2,68 bpw
com o drafter no último estágio, **`8,9,9,8,6`** dá 1,34 M tokens a 64 k e
1,45 M a 96 k; a partição antiga (`8,8,10,8,6`) deixa o estágio de 10 camadas
no teto de `util` e o pool cai para 111 k.

O fork também aceita `nvfp4_ds_mla` e `turboquant_*` (KV de 4 bits), que
reduzem o KV mas não o pico de ativação — não são o que limita o contexto.

## Pack do turboderp (2,68 bpw) — suporte a K fracionário

`turboderp/DeepSeek-V4.1-Flash-1017-exl3` @ `2.68bpw` (179,27 GiB; as tabelas de
Engram são as nossas, por symlink — `ngram_embedding.safetensors` não é
baixado). Convertido por `scripts/turbo_pack_build.sh`
(`make_vllm_pack.py` → `export_dspark.py` → `turbo_prepare.py` →
`turbo_config_patch.py`) em `/root/models/DS41-turbo-2.68bpw-vllm`.

O bloqueio era **K = 2,5** nos experts roteados das camadas 6–34: a tile tem
40 palavras = `16 × 2,5`. Do lado do plugin: o loader aceita `words % 16 == 8`,
`Exl3Config.layer_bits` guarda 2,5 como float, e a convenção do kernel mudou na
linha 1.6 do exllamav3 — as arrays por expert do `exl3_moe_mixedk` são **meias
unidades** (`k2 = 2*bits + half`) e `k_max` continua em bits
(`(max_k2+1)//2`). Gate em `scripts/frac_mixedk_check.py`: cos 1,000000
(K=2,5 × K=2,5), 0,999703 (K=2,5 × K=3) e **0,0039 com K em bits** — o negativo
prova a convenção. Camadas de K fracionário nunca entram no coop nem no dispatch
`grouped`, deixam `_exl3_k_tuple = None` e não passam pelo caminho nativo; as de
K inteiro continuam no kernel nativo (`EXL3_FORCE_MIXEDK=0` é o certo: forçar
mixedk em tudo piora o passo 16%).

Duas armadilhas de conversão, ambas custaram boot:
- **skip dos `mtp.*.attn.wo_a.slice.*`**: o `make_vllm_pack` só registrava esses
  96 tensores quando o ledger do pack de origem tinha prefixos `mtp.*` — o do
  turboderp tem 47.008 chaves e **zero** `mtp.`, então o drafter morria em
  `KeyError: 'model.layers.0.attn.wo_a.slice.0.mul1'`. `export_dspark.py` agora
  adiciona esses nomes por conta própria (gate: achar `trellis` no cabeçalho).
- **overlay do `head`**: nunca compor `reconstruct_slice` com `had_r_128` sobre
  a identidade (cos −0,0055 no exllamav3 1.6.5). `build_head` usa `effective()`
  do exl3_torch e confere cada bloco de 8192 colunas contra uma segunda
  implementação (decode do ext + Hadamards do Python), cos > 0,9995.

Números (partição `8,9,9,8,6`, DSpark N=5, knobs de host OFF):

| medida | valor |
|---|---|
| pool de KV @65.536 | 1.339.664 tok (20,4×) |
| **pool de KV @98.304** | **1.448.915 tok (14,74×)** |
| decode repetitivo, 256 tok | 59–63 tok/s |
| decode comp (5 prompts) | 39,8 tok/s / 80,9 ms / 2,28 aceitos por rodada |
| decode chat (raciocínio) | 28,8 tok/s / 88,9 ms / 1,56 aceitos |

O MTP do turboderp é **K=2** contra K=4 do nosso pack, e a aceitação cai
(2,28 contra ~2,2–2,4 do nosso pack no chat). Enxertar o nosso drafter K=4
(`scripts/graft_mtp_k4.py`) **não ajuda**: 38,75/27,71 tok/s com passo 82,6/96,2
ms e aceitação 2,235/1,667 — a perda é do modelo principal (experts de 2,5 bits
mudam a distribuição que o drafter vê), não da largura do MTP. O pack sem
enxerto é o que fica.

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

### Comparativo de kernel com o Maya (06/11) — o gargalo é o protocolo

Perfil por sessão (`nsys launch` sem coletar + `start`/`stop`, 512 tokens, 1
requisição, sem spec nos dois) contra o motor Maya na mesma topologia. Por
**token por GPU**: compute 3,88 ms nosso contra 3,26 do Maya (**1,19×**, quase
paridade; 0,45 contra 0,36–0,42 ms por camada) e **46,4 contra 15,7 ms de kernel
no total** — o resto é espera (a coluna de comunicacão soma kernels longos que só
aguardam o par; o custo útil é ~1,4 contra ~0,8 ms/token/GPU).

Três coisas que a atribuição (gate `VLLM_LOG_COLLECTIVE_CALLERS=1`, log
one-shot por call site) mostrou:

- **Não existe taxa de coletiva por camada.** São 1–2 all-gathers lógicos por
  passo por estágio: o **handoff de PP** (`irecv_tensor_dict(...,
  all_gather_group=get_tp_group())`, 163840 elementos — a máquina não tem P2P,
  então cada rank TP recebe metade e completa com o vizinho), o **Engram**
  (`engram.py:1944`, camadas 1 e 14) e os **logits** fatiados no TP
  (`logits_processor._gather_logits`, 8 × 64640, só no último estágio). O
  all-gather soma 2,36 ms/token/GPU (~4 ms de um passo de 60, ~7%), não os 8 ms
  que a média de kernel sugeria.
- O `SendRecv` observado é o *primer* de 1 elemento do handshake do PP
  (`gpu_worker.py:885`); os 6–12 kernels por passo são instanciações do NCCL,
  não chamadas.
- 536 kernels por passo por GPU contra 266 do Maya, com gaps de 0,15 ms entre
  eles: o overhead fixo por estágio existe e é o item mais barato de atacar.

Ou seja: 2,4× de diferença no passo (60 contra ~25 ms) **não** vem de compute nem
de pipelining — o pipeline do Maya é o mesmo (5 estágios em série; um token por
passo não tem como sobrepor). Vem do encadeamento entre estágios e do custo por
salto. Artefatos e método em `$EXL3_ROOT/NSYS-MAYA-VS-PORT-20261106.md` +
`scripts/nsys_{maya,vllm}_capture.sh` e `scripts/nsys_stage_report.py`.

### Latência por salto de estágio — medida na linha do tempo (06/11)

`scripts/stage_timeline.py <sqlite>`: âncora = o segundo `Broadcast` do NCCL
(curto, 1 por passo, alinhado a ~1 ms nos 10 ranks), bloco do estágio = maior
aglomerado de kernels de compute com lacunas ≤ 1 ms (é o replay do CUDA graph —
dentro do grafo não há **nenhuma** lacuna > 1 ms).

| estágio | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| bloco de camadas (ms) | 4,03 | 4,01 | 4,49 | 4,02 | 5,65 |
| início do bloco após a âncora (ms) | 0,00 | 5,87 | 21,76 | 38,92 | 59,31 |
| salto até o próximo (ms) | 1,8 | 11,9 | 14,5 | 16,4 | 17,5 (volta) |

Três conclusões que mudam o plano:

1. **O salto entre estágios é ~0–3 ms** (com a ponte de 5 ms para o bookkeeping,
   os saltos dão −2,3 / −2,6 / −1,4 / −1,1 / **+0,16** ms). Não existe latência
   de handoff para atacar — quando o estágio k termina o k+1 já está pronto. A
   troca de `SendRecv` NCCL por ring em memória (como o `xfer_*` do Maya) **não
   tem o que ganhar**.
2. **O trabalho de camadas é paridade com o Maya** (4,0–5,7 contra 3,2–4,5 ms
   por estágio). O que separa os blocos densos (12–16 ms por estágio) é o
   **bookkeeping eager do próprio estágio na mesma GPU**: ~90 kernels por passo
   (slot mappings, block tables, `_prepare_pos_seq_lens`, `_post_update_*`,
   `_pack_dense_prefix_to_ragged`, …) com lacunas de 0,05–0,8 ms entre eles —
   GPU ociosa esperando o host.
3. **Por passo por GPU** (trace, 512 tokens): kernel total 76,5 / 62,7 / 50,0 /
   30,8 / 10,8 ms, dos quais `Broadcast` (espera a vez) 71,5 / 54,3 / 39,3 /
   19,9 / 0,03; replay de grafo 4,6 / 4,2 / 5,0 / 4,4 / 3,5; `SendRecv`+
   `AllGather` 0,1 / 3,8 / 5,3 / 6,4 / 6,2 (mediana ~0, **média** inflada por uma
   cauda de spins de até 10 ms em ~10% dos passos); `AllReduce` 2,9 ms no total.

⚠️ **O profiler infla o nosso lado** (82,5 ms no trace contra 60,4 ms sem
profiler = 1,37×; o Maya dá 25,6 contra 25,7 = 1,0×), então compare proporções,
nunca milissegundos absolutos.

**Prioridade revisada** (substitui a tabela de alavancas do relatório de 06/11):
1. bookkeeping eager por passo (12–16 ms de espaçamento por estágio) — capturar
   no grafo ou baratear; 2. cauda de 8–10 ms dos coletivos de handoff; 3.
   all-gather de logits/dict (≤2 ms). O "handoff de PP" saiu da lista.

### O passo é limitado pelo HOST, não pela GPU (medido 07/11)

Fechando a pergunta dos 12–16 ms por estágio com só 4–5 ms de GPU: medição
**sem profiler nenhum** (`/proc/<pid>/stat` utime+stime antes/depois), 1024
tokens, no-spec, `max_model_len=32768`:

| grandeza | valor |
|---|---|
| decode | **18,29 tok/s → 54,6 ms/passo** |
| EngineCore (1 thread) | **~31 ms de CPU/passo** (57 % de um core) |
| cada um dos 10 workers | **~36 ms de CPU/passo** (65 % de um core) |
| CPU total por passo | ~0,39 s para 55 ms de parede (24 cores, sem quota) |
| GPU | ~22–25 ms de kernel por passo |

`py-spy` (frames Python, que o `--sample=cpu` do nsys não vê) no EngineCore:
**36 %** dentro de `_process_engine_step` (`core.py:1491`), **~28 %** esperando a
resposta dos workers — e essa espera é *spin* (`MessageQueue.dequeue`,
`acquire_read`, `SpinCondition.wait` → `sched_yield`), mais ~10 % de
`zmq`/`memory_fence`/`timeout_ms`. **`Scheduler.update_from_output` é 1,8 %**: o
scheduler do vLLM não é o custo; o custo é o **caminho de mensagens entre
processos** (1 `SchedulerOutput` e 1 resposta por rank por passo).

`max_concurrent_batches` já é `pp_size + 1 = 6` com async scheduling, mas isso
só adianta o lado do scheduler: **vLLM PP executa stage 0→4 do mesmo token
dentro do passo**, então o passo é a soma das latências de estágio.

Consequência para alavancas: NCCL/DBO/protocolo de PP e o scheduler estão
descartados por medição. Sobra **o trabalho de host por passo dos workers** —
~91 kernels eager por passo com lacunas de 0,05–0,8 ms, ~27 `cudaMemcpyAsync`
(1/4/8/512 B, DtoD) e ~95 launches. Se o host por estágio cair de ~11 para ~5 ms,
o passo vai de 55 para ~35 ms (**1,6×**) e o chat de ~38–41 para ~60 tok/s.
Relatório e método em `$EXL3_ROOT/HOST-BOUND-20261107.md`; ferramentas
`scripts/pyspy_engine.sh`, `scripts/spy_summary.py`, `scripts/spy_tree.py`.

### Grafo FULL para passos microbatched (port do PR #51700)

O `[2/2]` do DBO está no tree (`Ported-from: 73d2a8cf86`): `UBatchRunner` ganhou
`capture_stream`, `begin_capturable_run()`/`finish()`/`abort_pending_run()`, o
`CudaGraphManager` ganhou `ubatch_runner` + `_maybe_ubatch_twin()` (gêmeo de
captura por bucket de decode uniforme) e `_capture_stream(desc)`, e o gate
`_get_dbo_unsupported_features` deixou de recusar DBO com **qualquer** grafo
para recusar só com `PIECEWISE` (DBO captura FULL).

**Continua inerte nesta topologia**: com `dp_size=1` o `maybe_build_ubatch_runner`
devolve `None`, `dispatch_cg_and_sync_dp` sai em `dp_size == 1` e
`use_ubatching` exige `deepep_*`/`nixl_ep` (não instalados). Nenhum caminho novo
é alcançado; a config de produção fica bit-identical.

Duas notas para quando o DBO for ligado (degraus 2–4):
- `VLLM_DSPARK_REST`/`VLLM_DSPARK_CONF_CUT` expandem `decode_query_lens`; com o
  gêmeo, cada bucket passa a gerar **dois** grafos — conferir
  `_capture_mem_samples` e o pool antes de ligar as duas coisas juntas.
- `is_current_stream_capturing()` (usado pelo plugin EXL3 para a rota
  capture-legal) é **thread-local**: durante a captura microbatched o
  `torch.cuda.graph` roda na thread principal e o modelo nas threads de
  microbatch, então os kernels EXL3 dessas threads não se veem "capturando".
  Revisar antes de confiar em `_CAPTURE_SPLIT` sob DBO.

### DBO sob TP puro (degrau 3 — portado, medido, **REPROVADO**)

`Ported-from: 3a7ba0e13cae995a59f92b87c5ad3474d53e5d30` (#63483, "DBO under
pure tensor parallelism") adaptado a este tree, mais a nossa extensão **3b**.

O que entra: `dbo_tp_only` (`use_ubatching and dp_size==1 and not EP and tp>1`),
um communicator NCCL **por stream** (`pynccl._stream_comms`/`bind_stream`/`_comm_for`,
nas 9 coletivas) para os dois microbatches não dividirem o mesmo `ncclComm_t`,
o hand-off compute→comm em `GroupCoordinator._all_reduce_out_place` (dentro do
custom op `vllm.all_reduce`, portanto alcançável de dentro do grafo),
`enable_dbo_tp_overlap`, `prepare_ubatch_streams` (base no-op) e o split local
em `dispatch_cg_and_sync_dp`. `merge_ubatch_outputs` passa a aceitar
`tuple`/`list`/`None` recursivamente — é o que o DSpark precisa no degrau 2.

**3b (nossa):** o upstream devolve `cg_mode=NONE` para o passo dividido ("a
split step runs eager"), e eager aqui vale 7× menos. Com `VLLM_DBO_TP_GRAPH=1`
o split despacha o **gêmeo microbatch** já capturado (`dispatch(...,
num_ubatches=N)`, #51700) e só cai em eager quando não existe gêmeo ou um
microbatch sairia vazio.

Knobs (todos default = comportamento de antes):

| env / flag | default | o que faz |
|---|---|---|
| `--enable-dbo`, `--dbo-decode-token-threshold`, `--dbo-prefill-token-threshold` | off / 32 / 32 | liga o DBO; o threshold de decode tem que caber no bucket do passo (8 reqs × 1 token → 4 ou menos) e o de prefill tem que ser altíssimo (com chunk 512 todo passo de prefill seria dividido) |
| `VLLM_DBO_TP_GRAPH` | 0 | 3b: grafo FULL no passo dividido (0 = comportamento upstream, eager) |
| `VLLM_DBO_ALLOW_PP` | 0 | único ponto que recusa PP>1 sob DBO; ligado, mede-se na topologia real |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 1 nos scripts | **tem que ser ≥2** para as duas streams não serializarem; medido neutro sem DBO |

Medições (06/11, `scripts/serve_dbo_ds41.sh`, `scripts/serve_qwen_dbo.sh`):

| braço | Qwen2-1.5B TP2×PP1, C=8 | DS4.1 TP2×PP5, C=8 (agregado) | passo |
|---|---:|---:|---:|
| A sem DBO | **1052,6 tok/s** | 57,2–72,0 (mediana ~62,8) | 60,4 ms |
| B DBO **eager** (upstream) | 36,7 | 8,6 | 927 ms |
| C DBO **+ grafo** (3b) | 860,8 | 50,0–62,0 (mediana ~62,0) | ~60–111 ms |

- **Correção do split: OK.** No Qwen os braços B e C dão o **mesmo** stream
  guloso (mesmo fingerprint de token ids, 8×64 tokens) e o A difere em 1
  sinônimo em 512 — o grafo microbatched não muda o resultado. No DS4.1 o
  guloso não serve de oráculo: dois boots da MESMA config concordam em
  ~40–49% dos tokens (a paridade com o Maya já divergia em ~5–16 tokens), e os
  três braços saem coerentes.
- **C é 7× o B no DS4.1** (62 contra 8,6 tok/s): prova que o gêmeo do #51700 é
  de fato reexecutado; sem ele o split seria eager e teria esse custo.
- **Mas o split não paga**: em PP=1 (Qwen, onde não há bolha de PP) o C é 18%
  **pior** que o A; com PP=5 o C empata ou perde ~1–14%. O `--enable-dbo` troca
  um all-reduce de TP por um round trip a mais do pipeline e por uma troca de
  stream; num par TP sem P2P o rendezvous de 2 ranks cabe justamente sob o
  GEMM do outro microbatch só em teoria, não nos 60 ms medidos.
- **Gate decidido (ms/passo(C) ≤ 0,90 × ms/passo(A)) não passou → o degrau 2
  (spec decode sob DBO, #61464) NÃO foi feito.** `--enable-dbo` continua
  recusando spec decode; nada disso está no caminho default.

## Estado

Funcional e validado contra o motor Maya (paridade de prefill; decode 4x pior
sem o drafter e equivalente com ele, em prompt simples). **No ar em 07/11**: o
pack de 2,5 bits do turboderp em `0.0.0.0:3000` (ver a seção acima e
`/root/exl3-port/SERVING-NOW.md`). O que continua em aberto: o decode não escala com
concorrência como no Maya e o custo por token em chat/thinking é maior (a
aceitação do drafter cai de ~0,95 para ~0,72 na primeira posição quando o modelo
está "pensando"). DBO sob TP puro foi medido e reprovado (acima); a
bolha/sincronização do passo segue sendo o limite.

**Próxima alavanca de kernel (não medida): `ext.CoopMK`.** O master do
exllamav3 ≥1.6.5 traz um MoE cooperativo de decode com K por camada
(`exllamav3/modules/block_sparse_mlp.py`, env `EXL3_MOE_COOP_MIXEDK`, default
só em sm_121) — nos 3090 ele não é usado por default, e o plugin chama o ext
direto, então isso é trabalho de código (buffers + ordem dos experts). Vale
porque cobriria as camadas de K=2,5 no kernel rápido; o teto é o MoE de decode,
que é ~8–10 ms de um passo de 80 ms com spec — não é a alavanca de 2×.


## KV pool sweep (2026-11-10, pack Mia EAQ v1 2.9 bpw, TP2xPP5, DSpark N=5)

Measured with `sweep_util.sh` (util x PIECEWISE skip), `VLLM_DSPARK_REST` off:

| util | maxlen | KV pool (tokens) | concurrency | prefill 87k | decode C=1 |
|---|---|---|---|---|---|
| 0.97 | 98304 | 126,714 | 1.29x | ok | 50.9 tok/s |
| 0.9775 | 98304 | 249,405 | 2.54x | ok | 51.2 |
| **0.98** | 98304 | **290,302** | **2.95x** | ok, stress ok | 50.9-51.6 |
| 0.98 | 131072 | 184,087 | 1.40x | 122k ok | 52.7 |
| 0.982 | 131072 | 219,951 | 1.68x | OOM in the indexer (26 MiB) | - |
| 0.985 | any | refused: free memory on startup 23.14/23.56 GiB caps util at ~0.982 | | | |

* The pool depends on util, not on skipping the PIECEWISE graphs. `VLLM_CUDAGRAPH_SKIP_PIECEWISE=1`
  returns 0.14 GiB of graph memory but costs ~10% decode (C=1) and ~15% (C=8): leave it off.
* `VLLM_DSPARK_REST=1` at util 0.98 OOMs (40 MiB, dense EXL3 fallback) in the 87k prefill.
* Packing the hybrid KV groups by pooled bytes (vllm-project/vllm#60996, branch `kv-pack-60996`)
  yields identical groups (9,140 bytes/token summed) and an identical pool: not needed here.
* `VLLM_LOG_KV_GROUPS=1` prints the per-group page size and padding at boot.

### max-num-seqs 2 and where the KV pool goes (2026-11-10)

* `SEQS=2` @ util 0.98, maxlen 98304: pool **identical** (290,302), decode C=1 50.3 tok/s (same),
  C=2 48.9 tok/s/request (~98 aggregate). Only the CUDA-graph memory shrinks (0.56 -> ~0.2 GiB
  on most ranks, 0.43 on the worst). Not worth the lost concurrency by itself.
* `VLLM_LOG_KV_GROUPS=1` now also prints tensor sizes, bytes/block and blocks/request. DS4.1 TP2xPP5:
  6 groups (4x SlidingWindowMLA 2 layers, MLA 2 layers, CircularBuffer 1 layer), 7 pool tensors of
  46,080 B per block = **322,560 B per block id**; a 98k request claims 1,173 blocks
  (SWA 4x101, MLA 768, CB 1) = ~378 MB/rank, of which only ~73 MB are pages actually used by the group
  that owns each block: the shared-slab layout wastes ~5x. The structural fix is a per-group pool
  (vllm-project/vllm#44276); packing tweaks (#60996) do not change these groups.
* maxlen 131072 @ util 0.98 SEQS=2: pool 184,087 (1.40x), "weights + non-torch" grows by 0.08-0.17
  GiB per rank versus maxlen 98304 (something is sized by max_model_len); 122k prefill ok.

## KV pool: indexer workspace fix + prefill chunk + `auto` (2026-11-10, night)

Root causes found with `VLLM_DEBUG_WORKSPACE=1` (its logging had a NameError, fixed) and
`VLLM_LOG_KV_GROUPS=1`:

1. `streamed_workspace.eligible()` was always False in the workers: the indexer layers are built while
   `cache_config.block_size` is still the default 16, so the indexer K-gather workspace fell back to
   `max_model_len*40` rows (248 MB on PP0/1, 496 MB on PP2-4 at 98k; it scaled ~2.5-5 KB per token of
   max_model_len). Fixed (accept 16 or 128, re-check in the metadata builder): 7.2 MB.
2. The sparse-prefill gather workspace (`rocm.py` forward_mqa on this platform) then became the largest
   reservation (4 chunks x (L/ratio + window + chunk) x 1 KB): `VLLM_DS41_PREFILL_CHUNK_SIZE=1` cuts it 4x.
3. `VLLM_MAX_MODEL_LEN_CAP` bounds the DERIVED (`auto`) max_model_len; without it the profile sizes the
   workspaces for the model's 1M context.

| config (util 0.98 unless noted) | max_model_len | pool | prefill | notes |
|---|---|---|---|---|
| before (util 0.97) | 98,304 | 126,714 | 87k ok | |
| util 0.98 | 98,304 | 290,302 | 87k ok | |
| + eligible() fix | 98,304 | 358,101 | 87k ok | |
| + chunk 1 | 98,304 | 547,921 (5.57x) | 87k ok | |
| auto, cap 262144, chunk 1, SEQS=2 | 262,144 | 560,408 (2.14x) | 232k ok (1,131 t/s) | decode 51.8 |
| auto, cap 393216, chunk 1, SEQS=2 | 393,216 | 464,915 (1.18x) | ~350k OOM | |
| auto, cap 262144, chunk 1, **SEQS=8 util 0.98** | 262,144 | 560,408 | 232k OOM (2 MiB) | graphs 0.56 GiB |
| auto, cap 262144, chunk 1, **SEQS=8 util 0.975** | 262,144 | 456,107 (1.74x) | 232k ok (1,150 t/s) | stress ok; decode 51.2 C=1, 19.0/req C=8 |

Defaults in `scripts/serve_best_ds41.sh`: UTIL 0.975, SEQS 8, maxlen `auto`, cap 262144, chunk 1.
Decode is unchanged by all of this (host-bound).

## Decode host path (2026-11-10): where the CPU goes, and `VLLM_SHARE_TOKEN_TO_REQ`

`scripts/host_profile.sh` (py-spy 50 Hz on the EngineCore and one worker per stage, production config,
C=1, ~57 ms/step): stage 0 spends 79% of its samples in the Engram D2H sync (`engram.py:1159`), i.e. waiting
for the new token from the last stage (inherent PP latency; the preads are only 3.4%); the EngineCore spends
~75% spinning on the shm queue (off the critical path). The host work on the critical path is ~8.5 ms/step on
stage 0 and ~14 ms/step on the last stage, of which attention-metadata building is the largest item.

* `token_to_req_indices` was rebuilt for every KV cache group (~5x per step). `VLLM_SHARE_TOKEN_TO_REQ=1`
  (`vllm/v1/worker/gpu/attn_utils.py`) reuses the first group's result. 20 s py-spy windows, baseline -> flag:
  `build_attn_metadata` 68 -> 46 samples (stage 0) and 111 -> 85 (stage 4); `token_to_req_indices` 34 -> 14 on
  stage 4; drafter metadata 43 -> 31; decode C=1 56.6 -> 61.0 tok/s (inside the +-8% boot noise).
* Correctness: `VLLM_SHARE_TOKEN_TO_REQ=check` asserts every step (device sync, skipped during graph capture)
  that the shared map equals the one each group would build; a full run (87k prefill, mixed stress, decode) had
  zero divergences. Default in `scripts/serve_best_ds41.sh`: 1.
* Not ported: the batched sliding-window metadata of vllm-project/vllm#60996 (one kernel for the 4 SWA groups);
  it depends on `replay_start` and a model-state hook this fork does not have. Estimated extra gain ~1 ms/step.

### Last PP stage (2026-11-10): `VLLM_SKIP_IDLE_STAGED_WRITES`

The last stage's profile (321 busy samples per 20 s) showed `Sampler.apply_staged_writes` + `copy_to_uva`
at ~9%: each decode step re-copied ~11 per-request arrays (temperature, top_k/top_p/min_p, seeds, logit
bias, bad words, logprob ids) into the UVA ring although they only change in `add_request`.
`VLLM_SKIP_IDLE_STAGED_WRITES=1` (`vllm/v1/worker/gpu/sample/sampler.py`) skips them until a request is added
(the thinking-budget and trace-replay states keep their own guards). Last-stage samples: `apply_staged_writes`
13 -> 0, `copy_to_uva` 14 -> 1, total busy 328 -> 273.
Validation: unit tests; full run with both flags (87k prefill, mixed stress, decode, `check` mode: zero
divergences); `scripts/sampling_semantics.py` (greedy repeatable, top_k=1 == greedy, same seed repeatable,
temperature applied, different seed differs): all OK. Production config with both flags
(`VLLM_SHARE_TOKEN_TO_REQ=1`, `VLLM_SKIP_IDLE_STAGED_WRITES=1`): decode C=1 median 55.0 tok/s (5 reps, was
51.2 before the host-path work, same util/SEQS), C=8 21.2 tok/s/request (was 19.0); both inside boot noise
individually but consistent with the CPU-sample reductions. Defaults: both on in `serve_best_ds41.sh`.
Remaining on the last stage: the drafter attention metadata (~2.4 ms/step) and `sample_tokens`.

## Fase A do último estágio (11/11): `VLLM_RAGGED_DIRECT` + `dummy_q` compartilhado
- `VLLM_RAGGED_DIRECT=1` (default 1 em `serve_best_ds41.sh`): `DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build` empacota os
  índices ragged direto nos buffers persistentes do grafo (sem as 2 cópias D2D de `_copy_ragged_to_graph_buffers`).
  `VLLM_SHARE_DUMMY_Q` (default 1) reaproveita o `dummy_q` zerado entre as camadas do drafter.
- Medido: `sample_tokens` do último estágio 118→102 amostras, passos/22 s 280→302, greedy_gate OK contra
  `dumps/base_tokens_mia_flags0.json`, PPL 28,40±0,17 (ruído), C=1 mediana 56,25 tok/s.
- Fases B (metadata do drafter só-refresh), C (SWA em lote), D (contexto KV do drafter no grafo) ainda NÃO feitas.

## Fase C (11/11): `VLLM_BATCH_SWA_BUILD`
- Pré-passe em `build_attn_metadata` (`attn_utils.py`): um único kernel Triton (`_batched_swa_decode_kernel`, grid
  (tokens, grupos), vetores de ponteiros) preenche `decode_swa_indices/lens` e `is_valid_token` de todos os grupos SWA
  causais; `DeepseekSparseSWAMetadataBuilder.build` pula o próprio kernel/zeramento (`_swa_prebuilt`). Exige
  `VLLM_SHARE_TOKEN_TO_REQ`. Cai no caminho por grupo se: <2 grupos, `max_query_len > decode_threshold`, não-causal
  (drafter), strides/dtypes/janela/block_size diferentes, captura de grafo.
- `=check` roda também o caminho por grupo e compara (sync; só validação): 0 divergências em prefill 87k + estresse misto.
- Medido: `greedy_gate` OK, PPL 28,42±0,06, C=1 mediana 60,8 tok/s; perfil py-spy 78,0→62,8 ms/passo,
  `build_attn_metadata` do último estágio 26,7%→16,7% das amostras. Default 1 em `serve_best_ds41.sh`.
- Teste: `tests/v1/attention/test_batched_swa_decode.py`.

## Fase D (11/11): `VLLM_DRAFT_CTX_IN_GRAPH`
- Em passos pure-decode (`num_target_tokens == num_reqs * (N+1)`, descritor FULL) o contexto KV do drafter
  (`combine_hidden_states` + cópia + `precompute_and_store_context_kv`) roda DENTRO do grafo FULL do drafter
  (`DFlashSpeculator._graph_body`). `aux_hidden_states` vão por `torch.cat(out=)` para um buffer estático
  (`_aux_static`, `(N+1)*max_num_reqs` linhas); linhas de padding do contexto ficam com PAD_SLOT_ID/posição 0.
- Armadilha: o grafo agora SEMPRE contém o insert; em passos FULL que não são pure-decode o insert eager roda
  antes e depois os slots de contexto são preenchidos com PAD para o replay não regravar KV com dados velhos
  (sem isso a decodificação caiu a 16 tok/s e a aceitação desabou).
- O `greedy_gate` NÃO detecta drafter quebrado (o alvo decide a saída): comparar aceitação
  (`scripts/accept_rate.sh`, "Mean acceptance length" nos logs): 4,6996 vs 4,6997 no baseline.
- Medido: gate OK, PPL 28,37±0,19, C=1 mediana 62,9 tok/s; grafo +0,0 GiB visível (0,28–0,39 GiB/rank como antes).

## Carbon Waves 1+2+3 + Vision Tower (2026-11-11)

All changes are individually gated by env vars (`1` = enabled, `0` = legacy fallback); companion plugin/ext changes live on `DoktorMincs/vllm-exl3` (`exl3-ds41-local`) and `docs/exl3-ds41/patches/exllamav3-ds41-sm86.patch`.

1. **Vision Tower (`PP0`-only) + VRAM budgeting (`--limit-mm-per-prompt '{"image": 4}'`, `CHUNK=1056`, `UTIL=0.968`):**
   - `VLLM_VL_SKIP_NON_FIRST_PP_TOWER=1` (`vl_model.py`): instantiates `DeepseekV4ViT` + `DeepseekV4Aligner` (`~0.45 GiB/GPU`) only on `is_first_rank` (`PP0`), saving `0.45 GiB/GPU` across `PP1..PP4`.
   - `EXL3_MAX_RECONSTRUCT_SLICE_N=1024` (`exllamav3/modules/quant/exl3.py`) + `VLLM_EXL3_MOE_TEMP_ROWS=256`: slices `reconstruct_hgemm` along `N <= 1024` (`engram.wkv` peak `-250 MiB`, `PIECEWISE` cudagraph private pool `0.55 -> 0.31 GiB/GPU`) and shrinks `_FUSED_TEMP_CACHE` (`490 -> 61 MiB/GPU`).
2. **Quality & DSpark `<think>` acceptance (`SD-1`, `SD-3`, `SD-4`):**
   - `VLLM_DSPARK_PROB_DRAFT=1` (`dspark/speculator.py`): compact `draft_logits` (`[max_reqs * num_spec, vocab]`, `20.7 MiB`), dynamic `all_greedy` fast path (`use_local_argmax_reduction` in-graph), matched `(top_k, top_p, min_p)` filtering on draft logits for `T > 0`, and lossless joint block verification (`rejection_sample_method="block"`).
   - `VLLM_FP32_MOE_GATE=1` (`deepseek_v4/nvidia/model.py`, `gate_linear.py`, `gemv_triton.py`): keeps all 43 `F16` MoE router weights and logits in `FP32` across all batch sizes (`PPL` `27.885 -> 27.202`).
   - `VLLM_DRAFT_CTX_IN_GRAPH_RELAXED=1` (`dflash/speculator.py`): relaxes `num_target_tokens <= _ctx_per_req * num_reqs_padded` with zeroed tail padding so mixed/first-decode steps stay inside the captured context graph.
3. **Hybrid KV prefix cache & SM86 prefill (`KV-1`, `PRE-1..4`, `DEC-2a/5a/5b`):**
   - `VLLM_FIX_HYBRID_SWA_PREFIX=1` (`kv_cache_coordinator.py`, `kv_cache_utils.py`): aligns hybrid SWA retention/intersection with `eagle_block_drop`, restoring `99.3%` prefix-cache hit rate on `~32k` multi-turn (`TTFT 25.4 s -> 1.68–1.92 s`).
   - `VLLM_SM86_INDEXER_KV_GROUP=1` (`mqa_logits_triton.py`), `VLLM_INDEXER_MIN_SHARD_TOKENS=513` (`indexer.py`, `sparse_attn_indexer.py`), `VLLM_DSV4_LOGITS_ROW_CHUNK=64`, `VLLM_SPARSE_PREFILL_SM86_BK16=1` (`rocm_aiter_mla_sparse.py`, `BLOCK_K=16` -> `2 CTAs/SM` on SM86), and `VLLM_SKIP_PREFILL_PROPOSE=1` (`model_runner.py`).
   - `VLLM_PP_NO_DRAIN_SLEEP=1` (`core.py`), `VLLM_PP_SKIP_EMPTY_AG=1` (`gpu_worker.py`), `VLLM_SPARSE_DECODE_DIRECT_OUT=1` (`rocm_aiter_mla_sparse.py`).
4. **Wave 2 + Wave 3 (`DEC-3A` + `DEC-4`):**
   - `VLLM_EXL3_COOP_SLOTS_MAX=512`, `VLLM_EXL3_COOP_ROWS_MAX=88` (`exl3_moe_coop*` `(my_e << 10) | t` + `vllm_exl3/exl3.py`): eliminates the `slots = 288 > 256` fallback cliff at `C=8` (`M=48, topk=6`), `100%` bit-exact (`torch.equal`).
   - `VLLM_DS41_ENGRAM_FAST_GATHER=1`, `VLLM_DS41_ENGRAM_CACHE_POW2=17` (`engram.py`): persistent C `pread` pthread pool (`256 KiB` stacks, per-worker condvars waking `<=8` threads on decode misses to avoid waking all `128` `disk_threads`), open-addressing deduplication + ascending sort matching `np.unique`, `131,072`-row (`34 MiB`/table) `256`-striped row cache, and current-stream pinned D2H/H2D (`3.74x–4.90x` faster at `T=48`, `100%` bit-exact).

**Measured A/B (`10× RTX 3090`, `TP=2 × PP=5`, `8,8,10,8,6`, Mia v1 pack, isolated `127.0.0.1:3019`):**
- **KV pool (`cap 262,144`):** `477,271` tokens (`1.82x`, Vision `image:4` ON) vs `456,107` text-only baseline; CUDA graphs `0.31–0.41 GiB/GPU`.
- **Quality & Vision:** `greedy_gate` `256/256` OK; `ppl_probe` `27.885 -> 27.202 ± 0.087`; Vision `1/2/4`-image grounding + OCR `100%` pass.
- **Stochastic `<think>` (`T=1.0, top_p=0.95`):** `2.256 -> 2.406–2.514 tok/draft` (`+6.6%..+11.4%`).
- **Decode:** `step_by_prompt` `66.30 -> 62.39` (Wave 1) `-> 58.85 ms/step` (`-11.2%`); `C=1` `70.7–76.6 tok/s` (peak `92.5`); `spec_bench C=8` `170.7` (`155.5` Wave 1) `-> 248.0 tok/s` median (`168.4` cold / `253.2` warm, `+45.3%` vs baseline, `+59.4%` vs Wave 1); 24 unseen `C=8` prompts `117.4 -> 162.3 tok/s` median (`+38.2%`).
- **Long prefill (`201k` actual tokens, Vision ON):** `202.0 s (1,131 tok/s) -> 119.6–128.6 s (1,565–1,683 tok/s, +38%..+49%)`, zero OOM.

