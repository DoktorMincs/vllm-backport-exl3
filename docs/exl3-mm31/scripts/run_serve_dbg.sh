#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Sobe o servidor com a instrumentacao de dump ligada (comparacao com o oraculo
# ${MM3_REF}). Os tensores por camada vao para
# logs/dump3/*.npz numa unica chamada pequena (T <= VLLM_M3_DUMP_MAXT).
#
# Uso: run_serve_dbg.sh [tag]
set -uo pipefail
TAG="${1:-boot3}"
DUMP="${DUMPDIR:-${EXL3_ROOT}/logs/dump3}"
rm -rf "$DUMP"
mkdir -p "$DUMP"
export VLLM_M3_DEBUG="${VLLM_M3_DEBUG:-1}"
export VLLM_M3_DEBUG_MAX="${VLLM_M3_DEBUG_MAX:-8}"
export VLLM_M3_DEBUG_ATTN="${VLLM_M3_DEBUG_ATTN:-1}"
export VLLM_M3_DUMP="$DUMP"
export VLLM_M3_DUMP_MAXT="${VLLM_M3_DUMP_MAXT:-16}"
# T exato = 0: grava o primeiro forward pequeno (<= MAXT) de cada tag. O
# profiling/warmup do boot e bem maior (2048) e fica de fora. O batch do
# request chega PADDADO (5 tokens viram 16 linhas), por isso MAXT=16; o
# compare_full.py mapeia linha->posicao pela camada 0.
export VLLM_M3_DUMP_T="${VLLM_M3_DUMP_T:-0}"
# Roteamento por camada (topk ids/pesos) do plugin, mesmo portao ARM.
export VLLM_EXL3_DUMP_ROUTE="$DUMP"
exec ${EXL3_ROOT}/scripts/run_serve.sh "$TAG"
