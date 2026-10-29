# Ferramentas da campanha EXL3 (versão sanitizada)

Estes são os scripts usados para medir, comparar e operar o
MiniMax-M3.1-Flash EXL3 no `vllm-backport`. É a versão **sanitizada** do
diretório de trabalho usado na campanha: nenhum caminho absoluto da máquina
original, nenhum nome de serviço de produção e nenhum segredo.

Eles não são um produto: muitos assumem a topologia de 10× RTX 3090
(TP=2 × PP=5 ou TP=1 × PP=10), o pack EXL3 de 3,5 bpw e o plugin
`vcruz305/vllm-exl3` com os patches de `../patches/`.

## Variáveis de ambiente

| variável | default | uso |
|---|---|---|
| `EXL3_PORT` | `~/exl3-port` | raiz do ambiente experimental (venv, logs, artefatos) |
| `MODEL_DIR` | `~/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw` | pack EXL3 |
| `VLLM_REPO` | `~/vllm-backport` | checkout do engine (usado por `build_vllm.sh` e `verify_env.py`) |
| `MM3_REF` | `~/mm3-oracle/ref.py` | módulo do oráculo F32 — **não distribuído**; forneça o seu |
| `PROD_DIR` | `~/production` | diretório da produção, só para o snapshot da janela |
| `PROD_UNIT` | vazio | unidade systemd da produção que `window_*` para/religa |
| `WATCHDOG_UNIT` | vazio | unidade do watchdog (opcional) |

Os scripts de servidor também leem `PORT`, `MAXLEN`, `TP`, `PP`, `GPU_UTIL`,
`DTYPE`, `EAGER`, `KV_DTYPE`, `MAX_NUM_BATCHED_TOKENS`, `EXTRA_ARGS`,
`REASONING_PARSER`, `MM_LIMITS` e `LANGUAGE_MODEL_ONLY`. Os defaults estão no
cabeçalho de cada arquivo.

## Instalação

Copie para a raiz do seu ambiente, que é onde os scripts procuram uns aos
outros:

```bash
mkdir -p "$EXL3_PORT/scripts"
cp docs/exl3-mm31/scripts/*.py docs/exl3-mm31/scripts/*.sh "$EXL3_PORT/scripts/"
```

O oráculo (`MM3_REF`) é a única dependência que **não** vai junto: os scripts
`ref_big.py`, `ref_parts.py`, `replay_from_dump.py`, `replay_l17.py`,
`head_check.py` e `component_l18.py` importam o seu `ref.py` por caminho.

## Índice

**Servidor e build:** `serve_m3.sh`, `serve_best.sh`, `serve_dspark.sh`,
`run_serve.sh`, `run_serve_dbg.sh`, `boot_graphs.sh`, `boot_tp2pp5.sh`,
`boot6_fp16.sh`, `boot_loop.sh`, `boot_mixedk.sh`, `boot_nsys.sh`,
`build_vllm.sh`, `build_exllamav3.sh`, `install_plugin.sh`.

**Benchmark e sondas:** `speed_probe.py`, `speed_probe2.py` (conta tokens do
`usage`, não chunks SSE), `bench_prefill.py`, `bench_ctx.py`,
`bench_concurrency.py`, `bench_concurrency2.py`, `eval_boot.sh`,
`smoke_chat.py`, `chat_sample.py`, `complete_probe.py`, `probe_ids.py`,
`probe_ablate.py`, `prefix_consistency.py`, `needle.py`, `request_ids.py`,
`request_ref.py`.

**Dump e paridade:** `cmp_dump.py`, `compare_big.py`, `compare_ref.py`,
`compare_full.py`, `compare_parts.py`, `compare_route.py`, `route_cmp.py`
(recalcula o gate sobre o `ffn_in` do vLLM), `head_check.py`,
`replay_from_dump.py`, `replay_l17.py`, `ref_big.py`, `ref_parts.py`,
`component_l18.py`, `compare_first_token.py`, `extract_samples.py`,
`extract_one.py`, `parity_check.py`, `parity_m3_geometry.py`,
`parity_m3_mixedk.py`, `dryrun_config.py`, `pp_balance.py`, `pp_budget.py`,
`verify_env.py`, `prof_window.py`.

**Operação:** `window_open.sh`, `window_close.sh`, `window_backstop.sh`,
`window_keepalive.sh`, `mem_guard.sh`, `watch_log.sh`, `set_cap.sh`.

## Avisos

- `set_cap.sh` muda o power limit de **todas** as GPUs visíveis.
- `window_open.sh`/`window_close.sh` param e religam `PROD_UNIT`/`WATCHDOG_UNIT`
  se você as definir; sem elas, só mexem no servidor experimental e na
  sentinela. O `window_backstop.sh` é disparado pelo `window_open.sh` via
  `systemd-run` e sobrevive à sessão.
- `mem_guard.sh` mata apenas o Python de `${EXL3_ROOT}/venv-028`; ajuste
  `PY_PAT` se o seu venv tiver outro caminho.
- `verify_env.py` checa o `vllm` importado contra `VLLM_REPO`.

## Licença

Os scripts deste diretório são parte deste repositório e seguem a licença do
projeto (Apache-2.0). Eles chamam o plugin `vllm-exl3`, que é AGPL-3.0, mas não
distribuem código dele.
