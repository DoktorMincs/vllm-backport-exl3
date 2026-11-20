#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Instala o plugin vllm-exl3 (modo editavel) no venv-028 e compila
# vllm_exl3_c para sm_86, usando os headers do checkout do exllamav3.
set -uo pipefail
LOG=${EXL3_ROOT}/logs/install-plugin.log
MARK=${EXL3_ROOT}/logs/install-plugin.exit
EXT=${EXL3_ROOT}/pkgs/exllamav3/exllamav3/exllamav3_ext

[ -d "$EXT" ] || { echo "sem headers em $EXT" > "$LOG"; echo 91 > "$MARK"; exit 91; }

cd ${EXL3_ROOT}/pkgs/vllm-exl3 || exit 90
EXL3_EXT_INCLUDE="$EXT" \
TORCH_CUDA_ARCH_LIST=8.6 \
MAX_JOBS=6 \
  ${EXL3_ROOT}/venv-028/bin/python -m pip install -e . \
  --no-build-isolation --no-deps -v \
  > "$LOG" 2>&1
rc=$?
echo "$rc" > "$MARK"
echo "=== install_plugin.sh exit=$rc $(date -Is) ===" >> "$LOG"
exit $rc
