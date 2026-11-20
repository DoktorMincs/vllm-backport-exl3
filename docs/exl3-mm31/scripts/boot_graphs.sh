#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# TP=1 x PP=10 sem --enforce-eager: torch.compile + CUDA graphs.
# Unica diferenca em relacao a config conhecida-boa (boot13): o grafo.
set -uo pipefail
export TP="${TP:-1}"
export PP="${PP:-10}"
export EAGER=0
export EXL3_FORCE_MIXEDK=1
export EXL3_FUSED_MOE=1
export DTYPE=float16
exec ${EXL3_ROOT}/scripts/run_serve.sh "${1:-graphs}"
