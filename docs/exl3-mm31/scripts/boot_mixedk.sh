#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Forca as camadas de K uniforme pelo kernel mixed-K (K por expert).
# O kernel uniforme `exl3_moe` acerta K=4 e erra K=3 (L18..45); o mixedk trata
# K=3 corretamente (w2 da L17). Patch local: EXL3_FORCE_MIXEDK=1.
set -uo pipefail
export EXL3_FORCE_MIXEDK=1
export EXL3_FUSED_MOE=1
export DTYPE=float16
exec ${EXL3_ROOT}/scripts/run_serve.sh "${1:-boot13}"
