#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

set -uo pipefail
LOG=${EXL3_ROOT}/logs/build-exllamav3.log
MARK=${EXL3_ROOT}/logs/build-exllamav3.exit
cd ${EXL3_ROOT}/pkgs/exllamav3 || exit 90
TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=6 \
  ${EXL3_ROOT}/venv-028/bin/python setup.py bdist_wheel -d ${EXL3_ROOT}/pkgs/wheels \
  > "$LOG" 2>&1
rc=$?
echo "$rc" > "$MARK"
echo "=== build_exllamav3.sh exit=$rc $(date -Is) ===" >> "$LOG"
exit $rc
