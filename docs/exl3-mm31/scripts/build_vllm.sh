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
LOG=${EXL3_ROOT}/logs/build-vllm-backport.log
MARK=${EXL3_ROOT}/logs/build-vllm-backport.exit
cd ${VLLM_REPO} || exit 90
MAX_JOBS=10 CMAKE_BUILD_PARALLEL_LEVEL=10 VLLM_TARGET_DEVICE=cuda \
  ${EXL3_ROOT}/venv-028/bin/python -m pip install -e . --no-build-isolation --no-deps -v \
  > "$LOG" 2>&1
rc=$?
echo "$rc" > "$MARK"
echo "=== build_vllm.sh exit=$rc $(date -Is) ===" >> "$LOG"
exit $rc
