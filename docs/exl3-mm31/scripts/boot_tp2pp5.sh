#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# TP=2 x PP=5 nas 10 GPUs, com a correcao do MoE K3 e (opcional) coop no
# decode. Requer o patch local EXL3_ALLOW_TP_INDEXER=1 (ja' exportado pelo
# serve_m3.sh quando TP=2/PP=5).
#
#   EXL3_FORCED_MIXEDK_COOP=1 -> decode pelo kernel cooperativo (rapido) e
#   prefill pelo mixedk (correto no K3). Sem a flag, decode tambem no mixedk.
set -uo pipefail
export TP="${TP:-2}"
export PP="${PP:-5}"
export EXL3_FORCE_MIXEDK=1
export EXL3_FUSED_MOE=1
export DTYPE=float16
export VLLM_EXL3_MOE_TP_ALIGN=128
export EXL3_ALLOW_TP_INDEXER=1
export GPU_UTIL="${GPU_UTIL:-0.92}"
exec ${EXL3_ROOT}/scripts/run_serve.sh "${1:-tp2pp5}"
