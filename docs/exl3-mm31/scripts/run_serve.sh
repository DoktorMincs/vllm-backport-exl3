#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Roda o servidor com log e marcador de saída (disciplina de vigia).
# Uso: run_serve.sh <tag>   (PORT/MAXLEN/TP/PP/GPU_UTIL por ambiente)
set -uo pipefail
TAG="${1:-boot1}"
LOG=${EXL3_ROOT}/logs/"$TAG".log
MARK=${EXL3_ROOT}/logs/"$TAG".exit
${EXL3_ROOT}/scripts/serve_m3.sh \
  "${PORT:-3001}" "${MAXLEN:-8192}" "${TP:-1}" "${PP:-10}" > "$LOG" 2>&1
rc=$?
echo "$rc" > "$MARK"
echo "=== $TAG exit=$rc $(date -Is) ===" >> "$LOG"
exit $rc
