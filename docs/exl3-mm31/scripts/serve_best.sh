#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# CONFIG VENCEDORA da campanha de 27/10 (medida; ver logs/tuning.csv).
#
# TP=2 x PP=5 + CUDA graphs. Decode 45,3 tok/s (TP=1 x PP=10 dava 21,0) e
# prefill 1491 / 2126 / 2357 tok/s em 2k/8k/32k (TP=1: 1069/1805/2185).
# Agregado com 16 requisicoes: 84,1 tok/s (TP=1: 56.8).
#
# Por que o TP=2 estava "quebrado" (resolvido em 27/10):
#   O lm_head tem 200.064 linhas e o EXL3 aplica a Hadamard de SAIDA em blocos
#   de 128 colunas. A fatia do TP=2 (100.032) cai no meio de um bloco: o rank 1
#   era decodificado contra a transformada errada (logits locais com cos -0,50
#   contra o oraculo; TUDO errado) e o ultimo bloco do rank 0 (99968..100095,
#   metade padding) fabricava um logit de 63,9 no id 99968 — daí o token 'etim'
#   com p=1.0 em todo prompt. Correcao no plugin (`_exl3_pad_col_window`): cada
#   rank carrega a janela 128-alinhada inteira (rank 1: [99968, 200064), lead=64)
#   e o apply descarta o `lead` antes de concatenar. Depois: `<mm:think>` -0,013
#   (TP=1: -0,014), logits locais cos 0,99987 nos DOIS ranks, prefill x decode
#   consistentes (top-1 igual, <=0,15 nat), agulha 3/3 a 87k com KV fp8.
#
# PREFILL / DECODE, o que cada knob paga:
#   --max-num-batched-tokens 512 : com PP o prefill roda em chunks e o pipeline
#     se enche deles (wall ~ (chunks + pp-1) * t_stage). Chunk menor encurta as
#     duas pontas. Chunk maior e' PIOR: 8192 deu 349 tok/s no 8k (2,6x pior que
#     o default 2048). Custo: agregado em C=16 cai; MB=1024 fica no meio.
#   cap 230 W/GPU : +21% no prefill de 32k (200->230 W), soak de 70 s a 2,06 kW
#     sem desarme, 85 C. NAO persiste no reboot: rode scripts/set_cap.sh 230
#     antes de subir e volte ao seu cap normal depois. Decode e' insensivel.
#   EAGER=0 : CUDA graphs; 6,1x o eager no TP=1 e ~7x no TP=2 (6,6 -> 45,3).
#
# Contexto: MAXLEN 131072 com KV fp8_e5m2 (pool 211.328 tokens no modo texto,
# LANGUAGE_MODEL_ONLY=1; com a torre de visao o pool e' menor) e' a config
# medida; a 105k o decode ainda faz 42,4 tok/s e o prefill 2297 tok/s. fp16 KV
# vai ate' 65536 (pool 126.720). fp8 e4m3 NAO compila no Triton de sm_86. 262k
# com --max-num-seqs 256 NAO fecha: com util 0.945 o pool calculado chega a
# 270.208 tokens (1,03x), mas um worker morre com CUDA OOM no startup; 0.96
# idem. Com --max-num-seqs 8 e util 0.95 o pool vai a 281.856 e 262k cabe (modo
# texto; ver README). Nao insistir sem reduzir o ativacao (MB/cudagraph).
set -uo pipefail
export MAXLEN="${MAXLEN:-131072}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
export EXTRA_ARGS="${EXTRA_ARGS:---max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS}}"
KV_DTYPE="${KV_DTYPE-fp8_e5m2}"   # KV_DTYPE= (vazio) volta ao fp16 e a 65536
if [ -n "${KV_DTYPE}" ]; then EXTRA_ARGS="${EXTRA_ARGS} --kv-cache-dtype ${KV_DTYPE}"; fi
export VLLM_EXL3_MOE_LPT="${VLLM_EXL3_MOE_LPT:-1}"
export EAGER=0
export GPU_UTIL="${GPU_UTIL:-0.92}"
exec ${EXL3_ROOT}/scripts/boot_tp2pp5.sh "${1:-best}"
