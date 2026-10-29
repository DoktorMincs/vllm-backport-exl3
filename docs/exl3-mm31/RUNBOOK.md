# RUNBOOK — MiniMax-M3.1-Flash EXL3 no vllm-backport (venv isolado)

Este runbook descreve o ambiente experimental que ficou **separado** do stack
de produção. Os caminhos abaixo são placeholders: copie os scripts de
[`scripts/`](scripts/README.md) para `$EXL3_ROOT/scripts` e ajuste as variáveis.

| variável | default | o que é |
|---|---|---|
| `EXL3_ROOT` (`EXL3_PORT`) | `~/exl3-port` | raiz do ambiente experimental (venv, logs, artefatos) |
| `MODEL_DIR` | `~/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw` | pack EXL3 |
| `VLLM_REPO` | `~/vllm-backport` | checkout do engine |
| `MM3_REF` | `~/mm3-oracle/ref.py` | oráculo F32 (não distribuído) |
| `PROD_UNIT` / `WATCHDOG_UNIT` | vazio | unidades systemd que `window_*` para/religa |

A produção (se você tiver uma) **não é tocada** por nenhum destes passos, exceto
pela janela de GPU, que para e religa `PROD_UNIT`/`WATCHDOG_UNIT` quando
definidas.

## O que está instalado

| Item | Onde | Versão/SHA |
|---|---|---|
| vLLM backport (editable) | `$VLLM_REPO` | `0.13.2.dev4+gced6985fd` (wtdcode, v0.13.1-4-gced6985fd1) |
| venv isolado | `$EXL3_ROOT/venv-028` | torch 2.13.0+cu130, Python 3.12.13 |
| plugin EXL3 (editable) | `$EXL3_ROOT/pkgs/vllm-exl3` | `053c634` + 5 patches locais (abaixo) |
| exllamav3 (wheel sm86) | venv-028 | `1.5.0`, `329e051`, com `exl3_moe_mixedk` |
| pack | `$MODEL_DIR` | 184 GB, 25 shards + visão |

Rollback do venv para a wheel 0.28.0: `builds/vllm-028-sitepkg/` (pacote +
dist-info). O diretório residual `builds/vllm-028-leftover-sitepkg/` são só os
`.bak-*` de patch (não importa).

## Patches locais no backport (obrigatórios para fp16)

O pack declara `bfloat16` no config, então o vLLM auto-escolhe bf16. Rodar fp16
(`DTYPE=float16`, o padrão agora) exige 3 mudanças de Python — sem rebuild:

1. `vllm/models/minimax_m3/nvidia/model.py` (~L585): o dtype do cache lateral do
   indexer segue o dtype do modelo (era `"bf16"` fixo).
2. `vllm/models/minimax_m3/common/indexer.py`: `MiniMaxM3IndexerCache` aceita
   `fp16` **e** o guard do impl Triton aceita `fp16` (os kernels Triton são
   genéricos: `tl.dot` com acumulação fp32).
3. `vllm/config/attention.py`: `IndexerKVDType` ganhou `"fp16"`.

Sem isso o boot morre em
`fused_minimax_m3_qknorm_rope_kv_insert ... insert mode requires index_cache
matching qkv dtype or fp8 e4m3` — o kernel fundido grava o index-K no dtype do
qkv, então cache bf16 sob modelo fp16 é incompatível.
Diff em `patches/backport-exl3-mm31-cumulative.diff` (inclui também a
instrumentação).

## Patches locais no plugin (obrigatórios para este pack)

1. **PATCH #1 — indexer fp16**: o pack guarda `index_q_proj`/`index_k_proj` em fp16
   (sem `weight_scale_inv`), enquanto o plugin espera MXFP8 (payload F8 + escalas E8M0).
   O loader agora manda tensores float direto para as linhas de staging bf16 e
   `_process_indexer_shards` pula escalas ausentes.
2. **PATCH #2 — cobertura M3 por env**: a derivação de cobertura non-routed do M3
   dependia de um marker de processo que só existe no processo do engine. Como o load
   roda nos workers, `VLLM_EXL3_MINIMAX_M3=1` torna a derivação determinística.
3. **PATCH #3 — `EXL3_FORCE_MIXEDK=1` (OBRIGATÓRIO)**: o kernel MoE fundido de
   **K uniforme** (`fused_moe=exl3_moe`, stamp `"batched"`) acerta K=4 e **erra
   K=3** neste pack (L18–45: `ffn_out` com cos 0.32 / relerr 1.67, partindo de um
   `ffn_in` que bate a 3e-4; roteamento com margem 1,3e-2–7,8e-2, não é empate).
   A flag força essas camadas no kernel `exl3_moe_mixedk` (K por expert), que
   acerta; a cópia coop de decode fica desligada nelas. O `serve_m3.sh` já
   exporta a flag. Diagnóstico em `FINDINGS-20261027.md` §§10–12.
4. **PATCH #4 — A/B `EXL3_UNIFORM_LOOP=1`** (só diagnóstico): manda as camadas de
   K uniforme para o loop Python de referência. Mediu `<mm:think>` −0,014 contra
   −0,015 do mixedk e `'Você'` −2,10 do estoque.
5. **PATCH #5 — janela 128-alinhada do `lm_head` em TP>1** (`_exl3_pad_col_window`
   + loader + trim no apply). Sem ele o TP=2 cospe ruído: a fatia 200.064/2 =
   100.032 corta um bloco de 128 da Hadamard de saída do EXL3, então o rank 1 é
   decodificado contra a transformada errada e o último bloco do rank 0 fabrica
   um logit de 63,9 no id 99968 ('etim'). Ver STATUS.md (topo).

O diff cumulativo inclui ainda duas mudanças de suporte que não são correções
numeradas: `_narrow_tp` devolve view (evita a cópia anônima do trellis em TP>1)
e o caminho de TP do indexer (`EXL3_ALLOW_TP_INDEXER`, `VLLM_EXL3_MOE_TP_ALIGN`).
Ver `patches/NOTICE-plugin.md`.

Os patches são um diff sobre o plugin em `053c634` e estão sob AGPL-3.0-only
(`patches/NOTICE-plugin.md` e `patches/LICENSE-AGPL-3.0`).

## Subir o servidor

```bash
$EXL3_ROOT/scripts/run_serve.sh boot1     # log: logs/boot1.log, exit: logs/boot1.exit
```

Parâmetros por ambiente: `PORT` (3001), `MAXLEN` (8192), `TP` (1), `PP` (10),
`GPU_UTIL` (0.93), `DTYPE` (**float16** por padrão; `DTYPE=bf16` volta ao antigo),
`EXL3_FORCE_MIXEDK` (**1** por padrão — correção do MoE K3, ver acima),
`EAGER` (0 = torch.compile + CUDA graphs; 1 = `--enforce-eager`).

### Velocidade

`EAGER=0` é a alavanca: em TP=1 × PP=10 o decode vai de **3,5 → 21 tok/s**
(6,1×), sem mudar a numérica. O backport liga `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
sozinho. Suba com `scripts/boot_graphs.sh <tag>` ou
`EAGER=0 scripts/run_serve.sh <tag>`.

O segundo salto é o TP=2 × PP=5 **depois** do PATCH #5: o TP=2 estava quebrado
pelo corte da Hadamard do `lm_head`, não pelo paralelismo. Corrigido, a config
vencedora é `TP=2 × PP=5` com **45,3 tok/s** de decode, prefill
1.491/2.126/2.357 tok/s em 2k/8k/32k e 84,1 tok/s agregados com C=16
(`scripts/serve_best.sh` / `scripts/boot_tp2pp5.sh`).

### RAM do host: `_narrow_tp` e o OOM

Boot em TP>1 encheu 96 GB + swap e o OOM-killer matou os workers (log corta no
meio de um expert, sem traceback). Causa: `_narrow_tp()` terminava em
`.contiguous()`, copiando o trellis de cada expert para RAM **anônima** em TP>1, e
esse staging vive até o fim do load (~150 GB nos 10 ranks). Agora devolve *view*
do mmap (page cache); `EXL3_TP_COPY_SHARD=1` restaura o antigo. Pico medido
depois da correção: 23,4 GB anônimos. Guarda de RAM durante o load:
`scripts/mem_guard.sh <tag> [piso_MB] [intervalo_s]` (mata **só** o venv
experimental se `MemAvailable` cair abaixo do piso duas amostras seguidas).

Configuração final (10×3090, fp16): prefill 1.491/2.126/2.357 tok/s e decode
45,3 tok/s em TP=2 × PP=5 (`EAGER=0`, MB=512, KV `fp8_e5m2`, cap 230 W).
Geometria padrão TP=1 × PP=10 com
`VLLM_PP_LAYER_PARTITION=7,5,5,7,7,7,7,5,5,5` (máx 19,41 GiB/estágio); no
TP=2 × PP=5: `12,12,14,12,10` (máx 18,74 GiB/GPU).

Verificação: `scripts/watch_log.sh logs/boot1.log "Application startup complete" 3000`
(sai no primeiro sinal), depois `scripts/smoke_chat.py --port 3001`.

## Janela de GPU (parar/religar a produção)

```bash
$EXL3_ROOT/scripts/window_open.sh      # snapshot + sentinela + backstop + para produção
$EXL3_ROOT/scripts/window_keepalive.sh # heartbeat (rodar como tarefa da sessão)
$EXL3_ROOT/scripts/window_close.sh     # derruba experimental, espera VRAM, religa
```

- O backstop é uma unidade transiente (`systemd-run --unit=exl3-window-backstop`),
  **sobrevive à morte da sessão**; dispara por estado: sentinela ausente, sentinela
  com idade > 20 min (heartbeat parado) ou deadline duro de 10 h.
- Ao disparar: `pkill` **só** do venv experimental (`$EXL3_ROOT/venv-028/bin/python`),
  espera a VRAM cair e sobe `PROD_UNIT` + `WATCHDOG_UNIT` (se definidas; sem
  elas, só encerra o experimental).
- Estado/snapshot em `logs/window.log`, `logs/window-backstop.log`,
  `logs/snapshot-pre-janela-*.txt`.

## Diagnóstico

| Sintoma | Onde olhar |
|---|---|
| `vllm.__file__` = None / `No module named vllm.model_executor.layers` | diretório residual `site-packages/vllm` (namespace package). Remover/mover. |
| `flashinfer-cubin version ... does not match` | `FLASHINFER_DISABLE_VERSION_CHECK=1` (já no `serve_m3.sh`) |
| `EXL3 indexer shard N incomplete` | PATCH #1 não aplicado |
| atenção/densa/lm_head com BF16 (peso não inicializado) | PATCH #2 / `VLLM_EXL3_MINIMAX_M3=1` |
| OOM por estágio | `--max-num-batched-tokens 4096`, `--max-model-len 4096`, ou reparticionar (`pp_budget.py`) |

## Ferramentas

A versão sanitizada de todas elas está em [`scripts/`](scripts/README.md)
(copie para `$EXL3_ROOT/scripts`). As principais:

- `scripts/dryrun_config.py` — valida config/arquitetura/quant/PP **sem GPU**.
- `scripts/pp_budget.py` — orçamento de VRAM por estágio pelos bytes reais do pack.
- `scripts/pp_balance.py --stages N --tp T` — melhor partição PP contígua (DP) e
  valor por GPU; foi ele que achou `12,12,14,12,10` = 18,74 GiB/GPU no TP=2×PP=5.
- `scripts/speed_probe2.py` — TTFT e decode por SSE **contando tokens do usage**
  (o `speed_probe.py` conta chunks e subestima com spec decode).
- `scripts/mem_guard.sh <tag> [piso_MB]` — vigia de `MemAvailable` durante o load;
  mata só o venv experimental.
- `scripts/prod_chain_tp.sh <tag>` — prefill-only por ids + chat temp 1,0 +
  consistência prefill×decode, gravando em `logs/prod_chain_<tag>.out`.
- `scripts/boot_graphs.sh <tag>` — TP=1×PP=10 com `EAGER=0`.
- `scripts/serve_best.sh <tag>` — **config vencedora: TP=2 × PP=5**, MB=512,
  EAGER=0, 131k com KV fp8_e5m2, cap 230 W (decode 45,3 tok/s). Números e
  justificativa no cabeçalho e em `STATUS.md`.
- `scripts/cmp_dump.py <dirA> <dirB>` — compara dumps `VLLM_M3_DUMP` tag a tag
  (cos/relerr por componente, tolera sufixo `.rN` de rank e o batch paddado).
  Foi ele que isolou o defeito do `lm_head`: o corpo em TP=2 batia 0,9999.
- `scripts/head_check.py --mode local|state` — confere os logits locais de cada
  rank (ou o `final_normed`) contra o oráculo `ref_big.npz`.
- `scripts/set_cap.sh [230]` — powercap em todas as GPUs (sem lista fixa).
  O stack de referência rodava a 190 W; restaurar depois.
- `scripts/eval_boot.sh <tag>` — bateria padrão por boot (coerência, decode
  128/400, prefill 2k/8k/32k, concorrência 1/4/16) → `logs/eval_<tag>.txt`.
- `scripts/bench_prefill.py` — **vazão de prefill** (prompt grande, nonce,
  `cached_tokens` conferido, desconta um passo de decode).
- `scripts/bench_ctx.py` — TTFT e decode com o contexto já carregado (curva).
- `scripts/bench_concurrency2.py` — curva latência × vazão agregada (conta
  tokens do usage, não chunks).
- `scripts/needle.py` — agulha no palheiro (qualidade de KV quantizado).
- `scripts/prof_window.py --db t.sqlite --window nome=epoch_ini:epoch_fim` —
  soma tempo de kernel por nome dentro de janelas de um trace nsys exportado
  (`nsys export --type sqlite`). Foi assim que se achou os 89% de NCCL no decode.
- `scripts/boot_tp2pp5.sh <tag>` — TP=2×PP=5 (config vencedora desde 27/10 16:00,
  45,3 tok/s de decode; o `serve_best.sh` usa esta).
- `scripts/parity_check.py` — paridade dos kernels EXL3 em sm86 (gate G1).
- `scripts/smoke_chat.py` — 3 sondas de conteúdo + bench TTFT por contexto.
- `scripts/verify_env.py` — ambiente (torch, vllm, plugin, extensões).
- `scripts/ref_big.py` + `scripts/compare_big.py` — oráculo vs VLLM em prompt longo
  (por posição, por camada); `scripts/compare_ref.py` compara um dump com o
  oráculo de referência.
- `scripts/request_ids.py` — prompt por ids com logprobs por passo (separa prefill
  de decode: passo 0 vem do prefill, passos ≥1 do decode).
- `scripts/request_ref.py` — request determinístico para comparar com o oráculo.
- `scripts/replay_from_dump.py` — roda o oráculo a partir do estado que o VLLM
  dumpou (`--source vllm --start N`); é o teste que separa "kernel errado" de
  "o modelo amplifica ruído".
- `scripts/route_cmp.py` — recalcula a escolha de experts com a fórmula do oráculo
  sobre o `ffn_in` do VLLM (isola bug de gate).
- `scripts/chat_sample.py` — sondas de chat com a amostragem recomendada
  (`generation_config.json`: temp 1.0, top_p 0.95).
- `scripts/complete_probe.py` — completion crua (sem chat template).
- `scripts/prefix_consistency.py` — consistência **prefill x decode** sem oráculo:
  gera N tokens com KV cache e re-prefilla `ids + gen[:k]`; se as distribuições
  do passo k baterem, o caminho de decode (inserção/leitura do KV) está fiel.
- `scripts/post_boot_big.sh <tag>` + `DUMPDIR` — sobe a bateria de prompt longo
  (1 passo para a distribuição do prefill e 4 passos para prefill+decode) com o
  dump de até 256 linhas; `scripts/prod_chain.sh <tag>` roda a bateria de
  coerência em modo normal (chat temp 1.0, completion greedy, consistência).

## Armadilha do RPC em passos longos

O backport espera no máximo **300 s** por `sample_tokens`
(`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`). Com `VLLM_M3_DUMP` ligado, um prefill de
210 tokens passou disso (grava ~900 MB de `.npz`) e o engine morreu com
`TimeoutError: RPC call to sample_tokens timed out` — **não** é bug do modelo.
`serve_m3.sh` já exporta `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`; em modo
normal o passo leva frações de segundo.
