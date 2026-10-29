# EXL3 no vllm-backport — status da execução autônoma

Log cronológico, **mais recente primeiro**. Seções antigas descrevem estados já
superados (o rodapé, por exemplo, ainda cita 3,5 tok/s e DSpark/visão não
exercitados); os números válidos são os do topo e do README.

Atualizado: 2026-10-27 23:30 (America/Sao_Paulo)

## DSpark (drafter do MiniMax-M3) — FUNCIONA: 53,95 tok/s vs 44,8 (+20%, N=2)

Implementado em `vllm/models/minimax_m3/nvidia/dflash.py`
(`DSparkMiniMaxDraftModel` → `DSparkMiniMaxM3ForCausalLM`), registrado em
`registry.py`, com três ajustes de config em `vllm/config/speculative.py`
(exclusão da reescrita DSV4, `draft_hf_overrides={}` e `quantization=None`) e o
backend de atenção do drafter forçado para FLASH_ATTN em
`v1/worker/gpu/spec_decode/dspark/utils.py`.

**Três defeitos derrubaram a aceitação a 0% antes de funcionar** (todos silenciosos):

1. **Peso MXFP8 nunca desquantizado** — `_SCALE_SUFFIX = "weight_scale_inv"`
   fazia `name[:-len(sufixo)]` sobrar `...q_proj.` (com ponto), enquanto a chave
   do peso era `...q_proj`. Nenhum par casava → os valores `float8_e4m3`
   (|w| até 448) iam **crus** para parâmetros fp16. Correção: sufixo com o ponto
   (`".weight_scale_inv"`). Sintoma: modelo roda, texto do alvo é coerente,
   `Drafted: 154 / Accepted: 0`.
2. **Backend errado no drafter** — herdava `MINIMAX_M3_SPARSE_INDEXER` do alvo;
   as 5 camadas densas do drafter não têm pesos de indexer. Forçado o FLASH_ATTN.
3. **Medição**: `speed_probe.py` conta **chunks SSE**, não tokens. Com spec
   decode o servidor manda ~2 tokens por chunk e o probe reportava 18 tok/s
   falso. Usar `scripts/speed_probe2.py` / `bench_concurrency2.py`
   (`usage.completion_tokens`).

| config | tok/s (decode, temp 0) | aceitação |
|---|---|---|
| sem drafter (baseline) | 44,8 | — |
| **DSpark N=2** | **53,95** (+20%) | 0,612/0,401 por posição, L=2,01 |
| DSpark N=3 | 51,6 (+15%) | 0,552/0,368/0,227 por posição, L=2,15, draft acc 38% |
| DSpark N=1 | 47,0 | 0,592 por posição, L=1,59 |
| DSpark N=7 | 42,6 | 0,519/0,154/0,115/0,096/0,058/0,038/0,038, L=2,05 |

N menor também **melhora a aceitação** das posições 1–2 (0,612/0,401 no N=2 vs
0,368/0,227 no N=3 e 0,154/0,115 no N=7): mais slots de ruído degradam o bloco.

**Sweep de N (27/10 20:00) — a curva tem um ótimo em N=2, não é monotônica:**

- **N=1 não paga** (47,0 vs 44,8 do baseline): a aceitação por posição até sobe
  para 0,592, mas os **tokens por passo** caem para `L=1,59` (N=2: 2,01;
  N=3: 2,15). O custo do passo é dominado pelo **fixo** (bolha do PP=5 +
  handoff), não pelas linhas de verify — então quem manda é `L`, e `L` satura
  cedo.
- **N=2 é o melhor single-stream** (52,5 no 1º boot, 54,0 no 2º) e **N=3 fica
  no mesmo patamar** (51,6): a diferença está dentro da variância entre boots.
- **N=3 ainda ganha no agregado**: C=4 dá 53,1 contra 48,5 do N=2 (e 62,9 sem
  drafter). Causa provável: o balde de CUDA graph arredonda as 12 linhas do
  N=2 para o mesmo tamanho das 16 do N=3, então o N=2 paga o mesmo verify e
  entrega menos tokens por passo.
- **O drafter custa vazão agregada a partir de C=4 em qualquer N** (−16% no
  N=3, −23% no N=2): é config de latência single-stream, não de servidor
  concorrente.

**Contexto com o drafter carregado** — o estágio 5 (8 camadas + o drafter) é o
gargalo; o pool de KV é o teto para uma requisição:

| configuração | pool de KV | contexto máx. |
|---|---|---|
| drafter, KV fp16, util 0,92 | 41.638 | ~41k |
| drafter, KV fp16, **util 0,95** | **51.526** | **~51k** |
| sem drafter, KV fp16, util 0,92 | 126.720 | 65.536 |
| sem drafter, KV `fp8_e5m2` | 211.328 | 131.072 |

Ou seja: o drafter come ~60–70% do pool, e subir a util de 0,92 para 0,95
devolve 24% (41.638 → 51.526) sem custo medido.

**`kv_cache_dtype=fp8_e5m2` NÃO É POSSÍVEL com o drafter.** O drafter não tem os
pesos do indexer esparso, então é forçado ao backend `FLASH_ATTN`, e o FA em
sm86 recusa cache fp8: o worker do estágio 5 morre no boot com
`FlashAttentionBackend::validate_configuration: Selected backend
AttentionBackendEnum.FLASH_ATTN is not valid ... FP8 KV cache requires FA3 on
SM90 or FA4 on SM100`. Destravar exigiria um backend denso que aceite
`fp8_e5m2` em sm86 (caminho Triton), a validar em paridade.

**Teto de linhas (importante):** o `lm_head` compartilhado é EXL3 e, com
17–144 linhas, `_dense_forward` cai em `reconstruct` (1,23 GiB/rank). Numa
captura de CUDA graph esse temporário fica **no pool do graph** → OOM. Por isso
spec decode só roda com `max_num_seqs × (N+1) ≤ 16` (N=3 → `--max-num-seqs 4`).
Para concorrência maior seria preciso um `lm_head` bf16 dedicado ao drafter
(1,2 GiB/rank, dobrando suh/svh + janela `lead` do TP).

## KV cache, contexto e reasoning (27/10 21:00)

**A util não estava no limite — a ativação reservada estava.** O boot sem drafter
em `--gpu-memory-utilization 0.95` morre com CUDA OOM no estágio de 14 camadas
quando `max_num_seqs` fica no default (256). Com **`--max-num-seqs 8`** o perfil
de ativação encolhe e o pool sobe muito:

| config (sem drafter, KV `fp8_e5m2`, MB 512) | pool de KV | decode |
|---|---|---|
| util 0,92, ms=256 (config da campanha) | 211.328 | 45,3 |
| **util 0,95, ms=8** | **281.856** (+33%) | 45,1 |
| util 0,95, ms=256 | OOM no boot | — |
| util 0,95, ms=8, partição 12,12,12,12,12 | morre ("No available memory for the cache blocks") | — |

A partição chapada não serve porque o estágio final carrega o `lm_head`: com 12
camadas ele fica com 22,21 GiB de pesos e não sobra cache. **Manter
`12,12,14,12,10`.** Ganho de graça: 70k tokens de pool sem custo de velocidade.

Com 281.856 tokens, **262.144 de contexto passa a caber no pool** (1,07×);
com o drafter carregado o teto continua ~51k.

### Chat template / reasoning parser — `--reasoning-parser minimax_m3`

O M3 delimita o raciocínio com `<mm:think>…</mm:think>` (não ` thinking`/`</think>`).
O `serve_m3.sh` agora passa `--reasoning-parser minimax_m3` por padrão
(`REASONING_PARSER=` desliga). Sem isso a tag cai crua em `content`.

**Havia um segundo defeito, só no streaming:** o `is_reasoning_end(prompt)`
do backport varria o prompt inteiro procurando o par de marcadores, mas o bloco
`<thinking_instructions>` do próprio template escreve
`<mm:think></mm:think>` literalmente — o prompt então "parecia" já ter fechado o
raciocínio e o servidor mandava tudo como `content`, tag inclusive. O
não-streaming nunca sofreu (ele só olha a saída).

Correção (`vllm/reasoning/minimax_m3_reasoning_parser.py`, recorte em
`patches/backport-local-reasoning-m3.diff`; é um subconjunto do diff cumulativo,
não aplicar em separado): só importa o que o template
**prefila no fim do prompt** — `</mm:think>` (thinking desligado) → resposta,
`<mm:think>` (ligado) → raciocínio, nada (adaptive) → o parser decide posição a
posição. Validado nos 4 modos offline e no servidor (streaming e não-streaming).
`chat_template_kwargs: {"thinking_mode": "enabled"|"disabled"|"adaptive"}` é
repassado ao parser.

**Armadilha de medição nova:** com o reasoning parser ligado, os primeiros
tokens vêm em `delta.reasoning`, não em `content` — `speed_probe2.py` contava só
`content` e reportava "sem tokens". Já corrigido.

## Visão e vídeo no backport (23:30)

A produção sempre serviu imagem; o build experimental não, porque o
`serve_m3.sh` passava `--language-model-only`, que zera o limite de **todas** as
modalidades (`multimodal.py:get_limit_per_prompt` devolve 0) — daí o erro do
cliente "At most 0 image(s) may be provided in one prompt". O pack tem a torre
(515 tensores `vision_tower.*` + 4 projectors, em bf16, **fora** do EXL3).

Ligar a visão destravou **três defeitos em série**, todos silenciosos até o
encoder existir de fato:

1. **`--language-model-only`** removido (agora `LANGUAGE_MODEL_ONLY=1` reproduz
   as configs de texto). Limites por `MM_LIMITS` (default `{"image": 2,
   "video": 0}`).
2. **EXL3 reivindicava as lineares da torre** e lia bf16 como trellis:
   `RuntimeError: EXL3 linear shard 0: neither mcg nor mul1 marker is set`. O
   `quantization_config.json` do pack não tem **nenhuma** chave de visão, então
   o certo é `quant_config=None` na construção da `MiniMaxVLVisionModel`.
3. **O dummy de vídeo do profiling era dimensionado por `max_model_len`**, não
   pelo limite do encoder: `get_num_frames_with_most_features` devolvia até 500
   frames com seq_len 262k (o `vision_config` é um **dict** em vLLM, então
   `getattr` não achava `vision_segment_max_frames=4`). O profiling estourava
   dentro do primeiro `layer_norm` da torre — 2,75 GiB num encoder de 300M.
   Correção: capar pelos 4 frames do segmento (o `_apply_max_frames_limit` já
   divide vídeos maiores em segmentos de 4).

| config com visão | pool de KV | contexto |
|---|---|---|
| util 0,96 | **223.488** | **131.072 (1,71×)** |
| util 0,95 | ~199.808 (teto estimado pelo engine) | ~200k |
| util 0,94 | 176.512 | 131.072 (1,35×) |
| sem visão, util 0,95 | 281.856 | 262k cabe |
| sem visão + 262k pedido | — | ✓ |

O encoder é **replicado em todos os estágios de PP** (+0,81 GiB/GPU em cada um
dos 5), e é isso — não a partição — que derruba o pool: `12,12,14,12,10` já está
no piso do balanceamento (o DP com a penalidade da visão continua escolhendo
essa partição). **Vision + 262k não cabe** em 10×3090 com TP=2×PP=5.

Custo de velocidade: nenhum. Decode **45,50 tok/s** (util 0,94) e **45,47**
(util 0,96) com visão, igual ao texto puro (45,1).

**Escala util → pool com a visão carregada:** ~24k tokens por 0,01 de util
(176.512 → 199.808 → 223.488). Para caber 262k com visão seria preciso util 0,98,
que deixa só ~0,5 GiB de folga para os CUDA graphs — não testado.

Validado ponta a ponta: imagem de teste → leu o código "4815" e as formas
(retângulo vermelho, círculo azul); vídeo de 4 frames (695 prompt tokens) →
descreveu os quadros corretos. Patches em
`patches/backport-local-m3-vision.diff` (acumulado vs HEAD).

## TP=2 × PP=5 CONSERTADO — decode 21 → 45,3 tok/s (2,17×)

**A causa raiz do "etim" era o lm_head, não a atenção nem o MoE.** O dump
camada-a-camada provou que **o corpo do modelo em TP=2 é numéricamente perfeito
até a L59** (todos os tensores replicados com cos ≥ 0,9987 contra um TP=1
correto, e `attn_out(rank0)+attn_out(rank1)` batendo 0,9995 com o TP=1). O
defeito estava na cabeça:

- O `lm_head` tem 200.064 linhas e o EXL3 aplica a **Hadamard de saída em blocos
  de 128 colunas**. A fatia do TP=2 (100.032) cai no meio de um bloco
  (100.032 = 781,5 × 128).
- Medido com logits locais por rank: **rank 0 quase perfeito (cos 0,9969) exceto
  o id 99968, com logit 63,9** (49 nats acima de tudo — é o token 'etim' que
  saturava o softmax); **rank 1 inteiro errado (cos −0,50)**.
- Correção no plugin (`_exl3_pad_col_window` + loader e apply): cada rank carrega
  a **janela 128-alinhada inteira** (rank 0: `[0, 100096)`; rank 1:
  `[99968, 200064)`, `lead=64`) e o apply descarta o `lead` antes de concatenar.
- Depois: `<mm:think>` **−0,013** (TP=1: −0,014); logits locais **cos 0,99987**
  nos dois ranks; prefill × decode consistentes (top-1 igual, ≤0,15 nat); chat
  coerente em pt-br; agulha 3/3 a 87k.

| métrica | TP=1×PP=10 | **TP=2×PP=5** | ganho |
|---|---|---|---|
| decode (128 tok) | 21,0 | **45,6** | 2,17× |
| decode (400 tok) | 20,7 | **45,2** | 2,18× |
| prefill 2k | 1.069 | **1.491** | 1,39× |
| prefill 8k | 1.805 | **2.126** | 1,18× |
| prefill 32k | 2.185 | **2.357** | 1,08× |
| agregado C=16 | 56,8 | **84,1** | 1,48× |
| decode a 105k | 25,9 (fp8) | **42,4** | 1,64× |

Config em `scripts/serve_best.sh` (TP=2 × PP=5, EAGER=0, MB=512, cap 230 W).

**Contexto**: 131.072 com KV `fp8_e5m2` (pool 211.328 tokens, 1,61× de
concorrência) é a config medida; 65.536 com fp16 KV (pool 126.720). A curva é
plana (105k: prefill 2.297, decode 42,4). **262k continua não fechando**: com
`--gpu-memory-utilization 0.945` o pool calculado chega a 270.208 tokens (1,03×),
mas um worker morre com **CUDA OOM no startup** (0.96 idem); precisaria reduzir
ativação (MB/cudagraph) — não vale o trade.

## Campanha de performance 27/10 11:00–13:00 (Fases 0–2 do plano)

**Prefill: 349 → 1.069 tok/s em 2k (3,1×), 919 → 1.805 em 8k (1,96×),
1.544 → 2.185 em 28,8k (1,42×). Decode: 21 tok/s em toda config testada.**
Config vencedora em `scripts/serve_best.sh`; números por boot em `logs/tuning.csv`.

### O que moveu o prefill (dois knobs, nenhum deles "óbvio")

1. **`--max-num-batched-tokens 512`** (o default era 2.048). Com PP=10 o prefill
   roda em *chunks* e o pipeline se enche de chunks: `wall ≈ (chunks + 9) ×
   t_stage`. Chunk menor encurta `t_stage` **e** a bolha, então 512 > 1.024 >
   2.048. O contrário também foi medido: **8.192 deu 349 tok/s no prompt de 8k
   (2,6× pior que 2.048)** — o default do vLLM não era um limite aqui, era o
   que mantinha o pipeline alimentado. Custo: vazão agregada em C=16 cai de
   62,1 para 56,8 tok/s; `MB=1024` fica no meio (58,1 e prefill ~1,4×).
2. **Power cap 200 → 230 W/GPU: +21% no prefill de 32k** (1.895 → 2.285 tok/s),
   a 2,06 kW totais, soak de 70 s sem desarme, 85 °C. Decode insensível, como já
   estava registrado. Não persiste no reboot: `scripts/set_cap.sh 230` antes de
   subir, e voltar ao valor de produção depois.

Neutros (medidos, não repetir): `CUDA_DEVICE_MAX_CONNECTIONS=8`,
`VLLM_EXL3_MOE_LPT` (~ruído, mantido), reabilitar o kernel coop via
`EXL3_FORCED_MIXEDK_COOP=1` (decode 21,4 = sem ganho; revertido por risco
numérico). Inertes neste build: `VLLM_SPARSE_RAGGED_FAST_SCAN` (só ROCm) e
`VLLM_EXL3_GROUPED_PREFILL` (módulo de política, não despacha kernel).

### Por que o decode não sai dos 21 tok/s

- Perfil nsys (1 boot instrumentado): **89% do tempo de kernel é NCCL**
  (`ncclDevKernel_Broadcast_RING_LL` + `SendRecv`), só ~9% é compute
  (`moe_exl3` 4,4% + `dense_exl3` 4,2%).
- Durante o decode os **10 workers ficam a ~100% de UM core de CPU cada** —
  é espera ocupada no handoff do pipeline, não trabalho útil.
- Consequência: o teto de um stream é a soma das latências dos 10 estágios
  (~26 ms de compute + ~22 ms de handoff = 48 ms/token). Nenhum knob de kernel
  ou de batch move isso; só um pipeline mais curto (TP=2 × PP=5, ainda quebrado)
  ou speculative decoding (não há drafter neste pack).

### Curva de contexto (o KV quase não pesa)

| contexto | prefill tok/s | decode tok/s | TTFT |
|---|---|---|---|
| 7,5k | 1.744 | 22,7 | 4,3 s |
| 28,8k | 2.149 | 21,3 | 13,4 s |
| 54k | 2.219 | 22,3 | 24,4 s |
| 99,7k (KV fp8_e5m2) | 2.216 | 25,9 | 45,0 s |

Plano nos dois eixos: a atenção esparsa lê só `topk_blocks × 128 = 2.048` tokens
de KV por camada, qualquer que seja o contexto.

**Teto de contexto: 65.536 com KV fp16** (pool 109.184 tokens) ou **131.072 com
KV `fp8_e5m2`** (pool 196.608, 1,5× de concorrência). `fp8` (e4m3) **não
compila** no Triton de sm_86 (`type fp8e4nv not supported`). **262k não cabe**:
exigiria ~9,4 GiB de KV por estágio e há ~3,5. Qualidade com e5m2 medida por
agulha: **3/3 acertos** em 10/50/90% de profundidade a 49k (`scripts/needle.py`).

### Instrumentação nova

`bench_prefill.py` (vazão real de prefill; o número antigo era TTFT de prompt de
210 tokens), `bench_ctx.py` (decode *com* contexto carregado), `bench_concurrency.py`,
`needle.py`, `eval_boot.sh` (bateria padrão por boot), `prof_window.py` (soma de
kernels por janela de tempo no trace nsys), `set_cap.sh`, `serve_best.sh`.
Guard de CPU-offload em `serve_m3.sh` (aborta com exit 97; testado).

## Rodada de 27/10 09:00 — velocidade e memória

**Decode 3,5 → 21 tok/s (6,1×) tirando o `--enforce-eager`.** Geometria TP=1 × PP=10,
`EAGER=0` (torch.compile + CUDA graphs; o backport liga
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` sozinho e loga "Breakable CUDA graph enabled").
Medido com `speed_probe.py` (SSE): **21,6 / 21,0 tok/s** em 128 tokens e
**20,71 / 20,70 tok/s** em 400 tokens (regime estável); TTFT 0,55–0,77 s. A numérica não mudou: prefill de 210 tokens dá
`<mm:think>` **−0,014** (oráculo −0,013, igual ao run eager), e a consistência
prefill × decode bate passo a passo (top-1 idêntico, logprobs a ≤0,2 nat).

Por que o gargalo era esse: `TP=1×PP=5`→ não, medido — **PP 10→5 não mudou nada**
e **TP 1→2 também não** (3,40 tok/s), então o custo é *fixo por camada*
(~4,8 ms/camada/token: lançamento de kernel e Python em modo eager), não banda
nem número de estágios. Grafo ataca exatamente isso.

**OOM do host — causa e correção.** O boot TP=2 encheu 96 GB + swap e o
OOM-killer matou os workers às 08:53 (log corta no meio de um expert, sem
traceback). Causa: `_narrow_tp()` no plugin terminava em `.contiguous()`, o que
em TP>1 materializa uma cópia **anônima** do trellis de cada expert; esse staging
vive até o fim do load → ~1,3 GiB × 12 camadas × 10 ranks ≈ 150 GB. Em TP=1 a
função retorna o tensor intacto e o staging é só *view* do mmap (page cache,
reclamável) — por isso o TP=1 nunca sofreu. Corrigido devolvendo a view
(`EXL3_TP_COPY_SHARD=1` restaura o antigo). Medido depois: **pico anônimo 23,4 GB**,
76 GB livres.

**TP=2 × PP=5 — CONSERTADO em 27/10 16:00 (ver a seção no topo: 45,3 tok/s).**
O texto abaixo é o registro do estado anterior à correção do lm_head.
Depois de destravar
`EXL3_ALLOW_TP_INDEXER=1` e de implementar a geometria padded em TP>1 (o lm_head
de 200.064 vira 100.032 por rank, que não é múltiplo de 128), o servidor sobe,
carrega e serve em 5 estágios × 2 ranks (`VLLM_PP_LAYER_PARTITION=12,12,14,12,10`,
máx. 18,74 GiB/GPU contra 19,31 do TP=1/PP=10). **Mas a saída é ruído**
(`etimetim…`, prefill `'etim'` em vez de `<mm:think>`) e o decode ficou em
3,40 tok/s — sem ganho. Falta localizar o erro de sharding (candidato: caminho
TP da atenção esparsa/indexer, que é código do backport, não do plugin).
Fica registrado como não-produtivo: **para velocidade, o caminho é grafo, não TP**.

## Objetivo

Servir `MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpf` (184 GB, EXL3/EAQ 3,55 bpw) no
**vllm-backport** (wtdcode, v0.13.2.dev4+gced6985fd) com o plugin
`vcruz305/vllm-exl3`, em 10× RTX 3090, sem tocar na produção.

**Status: FUNCIONA.** Sobe, serve e responde de forma coerente — com uma
correção obrigatória no plugin (MoE de K uniforme 3) descrita abaixo.

## Estado atual

- **Serve**: `/v1/models` 200; prefill e decode funcionando; TP=1 × PP=10.
- **Texto coerente** com a amostragem do modelo (temp 1,0 / top_p 0,95): o
  modelo raciocina de fato (`<mm:think>` + plano da resposta + resposta).
- **Correção obrigatória**: `EXL3_FORCE_MIXEDK=1` (patch local no plugin) —
  manda as camadas de MoE de K uniforme pelo kernel mixed-K. Sem ela, as
  camadas 18–45 (K=3 uniforme) produzem saída errada e o texto degrada.
  `serve_m3.sh` já exporta a flag por padrão.

## A causa raiz (medida em 27/10)

1. O prefill de 210 tokens bate com o oráculo F32 **até a L17** em todas as
   posições (cos 1.000000); a L17 inclusive é a única mixed-K.
2. Na L18 — a **primeira camada de K=3 uniforme** — a atenção (cos 1.000000),
   o `ffn_in` (relerr 3e-4) e o roteamento (margens 1,3e-2–7,8e-2) estão
   certos, mas o `ffn_out` sai com **cos 0.32 / relerr 1.67**.
3. A L17 (mixed-K) e as camadas K4 (L3–16, L46–59) batem: é específico do
   **kernel MoE fundido de K uniforme em K=3**.
4. Prova por A/B: partindo do estado L17 **do próprio vLLM**, o oráculo
   devolve `<mm:think>` com logprob −0,013 (p≈0,99); o vLLM devolvia `'Você'`
   (p≈0,12). Com as camadas K3 no loop de referência → −0,014. Com o kernel
   mixed-K → −0,015. Os tensores K3 carregados estão certos.

Detalhe completo em `FINDINGS-20261027.md` (seções 10 e 11); a conclusão
"é caos, não é bug" das seções 1–9 está corrigida na 11.2.

## O que está provado (por componente)

| Item | Resultado |
|---|---|
| Kernels EXL3 densos (K3/K4/K5/K6 + lm_head, amostras reais) | cos ≥ 0,99999 |
| MoE native p2b, mixed-K, fat, ativação (72 testes) | OK |
| Case A2 (mixed-K, act-3) na geometria real, 64 e 512 tokens | cos 0,999994 |
| Case B (qkv fundido + indexer, m=1 e 64) | cos ≥ 0,99997 |
| Atenção esparsa + indexer em 210 tokens (pos 205/209) | cos 1.000000 até L17 |
| Gate do vLLM (fórmula do oráculo sobre o `ffn_in`) | idêntico ao vLLM |
| Camadas 16 e 17 do vLLM (replay do oráculo) | cos 0,999985 / 0,999983 |
| Cabeça (norm final + lm_head sobre o estado L59 do vLLM) | Δ ≤ 0,04 nats |
| **MoE fundido de K uniforme em K=3** | **ERRADO** (cos 0.32 na L18) |

## Patches necessários

1. **Plugin — 3** (todos locais, sem rebuild):
   - indexer fp16 → o cache lateral segue o dtype do qkv;
   - cobertura non-routed do M3 por env (`VLLM_EXL3_MINIMAX_M3=1`);
   - **`EXL3_FORCE_MIXEDK=1`**: camadas de K uniforme pelo kernel mixed-K
     (contorna o defeito do kernel uniforme em K=3).
2. **exllamav3 — 1**: `ACT_SILU_OAI` (act 3) em `hadamard_inner.cuh` (o caminho
   não-coop aplicava SILU). Rebuild necessário.
3. **Backport — 3, só Python**: cache lateral do indexer segue o dtype do modelo
   (`indexer.py`, `attention.py`, `model.py`). Sem isso o boot fp16 morre em
   `insert mode requires index_cache matching qkv dtype or fp8 e4m3`.
   Diff em `patches/backport-local-*.diff`.

## Configuração (`scripts/serve_m3.sh`)

- `DTYPE=float16` — o pack declara bf16, mas o oráculo e o stack de referência
  rodam fp16; bf16 injeta ~0,5% de erro até a L15 e piora o 1º token.
- `EXL3_FORCE_MIXEDK=1` — ver acima (obrigatório).
- `VLLM_EXL3_FAT_THRESHOLD=1e8` — o tier "fat" mediu cos 0,12; desligado.
- `VLLM_EXL3_Q4KV=0`, `VLLM_EXL3_INDEXER_MXFP8=0` (GB10/sm_121).
- `VLLM_PP_LAYER_PARTITION=7,5,5,7,7,7,7,5,5,5`.
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` (o default de 300 s derruba o
  engine quando o dump de depuração está ligado).
- Amostragem recomendada pelo pack: **temp 1,0 / top_p 0,95**.

## Sondas e ferramentas

`smoke_chat.py`, `chat_sample.py`, `complete_probe.py`, `probe_ids.py`,
`prefix_consistency.py`, `request_ids.py`, `ref_big.py`, `compare_big.py`,
`component_l18.py` (atenção x MoE por camada), `route_cmp.py` (margens do gate),
`replay_l17.py` (oráculo a partir do estado do vLLM), `replay_from_dump.py`,
`gate_check.py`, `parity_check.py`, `parity_m3_mixedk.py`, `extract_one.py`,
`prod_chain.sh`, `boot_loop.sh`, `boot_mixedk.sh`.

## Armadilhas

- Warmup do boot: ~4–5 min (com 10 estágios de PP).
- Batch do vLLM vem paddado (210 → 256 linhas); padding vira NaN e não afeta as
  linhas reais (atenção causal).
- `EXL3_FUSED_MOE=0` **não** serve para bissectar aqui: derruba também o
  mixed-K da L17 no `apply_mixed_k` de referência, que exige o store (desligado
  por padrão) → AttributeError. Use `EXL3_UNIFORM_LOOP=1`.
- `systemctl is-active` não prova função: confirmar com geração real.

## Limitações e próximos passos

- **Decode ~3,5 tok/s**: é o custo do PP=10 em `--enforce-eager` (cada token
  atravessa 10 estágios em sequência). Não é do plugin. Candidatos a melhorar,
  em ordem de custo: tirar o `--enforce-eager` (CUDA graphs; o plugin diz que o
  mixedk é seguro para grafo, mas PP+grafo precisa validar), reduzir o número de
  estágios (PP=5 exige TP=2, e o plugin recusa TP>1 no qkv fundido do indexer —
  seria preciso patch), e revisar o `VLLM_PP_LAYER_PARTITION`.
- **Prefill**: ~210 tokens em 0,5–0,9 s (227–425 tok/s), ok.
- **Diferença residual em contexto curto**: em "The capital of France is" o top-1
  agora bate com o oráculo (`' a'`), mas a distribuição é mais achatada
  (−4,89 vs −0,41 do oráculo F32) — consistente com o ruído fp16/F32 em 60
  camadas; o caminho de chat (o uso real) está correto e confiante.
- **DSpark** (drafter) e **visão** não foram exercitados — **todos os números
  acima são sem speculative decoding**. O pack *tem* o drafter
  (`dspark/model-dspark-00001.safetensors`, 2,3 GB, 97 tensores: 35 lineares em
  **MXFP8** F8_E4M3 com escala `weight_scale_inv` + cabeças em BF16, arquitetura
  `DSparkMiniMaxDraftModel`), mas o backport só registra as variantes DeepSeek-V4,
  Qwen3, Kimi K3 e Gemma4 — **não existe `DSparkMiniMaxDraftModel` nem para
  MiniMax-M2**. Além disso o pack principal não traz MTP/EAGLE (0 tensores com
  `mtp`/`eagle`/`nextn`; o registry só tem `MiniMaxM3MTP`, sem pesos) e o caminho
  MXFP8 desta stack é gated a Blackwell (o `VLLM_EXL3_INDEXER_MXFP8` do plugin é
  GB10/sm_121). Habilitar exigiria classe de modelo + plumbing de aceitação, e um
  fallback de dequant para sm_86 — é a maior alavanca que sobrou agora que o
  decode está em 45 tok/s (≈2 tokens aceitos/passo = 70–110 tok/s), mas não é
  uma flag.
- Rascunho de issue para o autor do plugin em `BUG-exl3-moe-uniform-k3.md`
  (publicado neste repositório).
