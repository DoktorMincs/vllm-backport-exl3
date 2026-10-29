#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Bateria padrao de avaliacao de um boot de tuning.
# Uso: eval_boot.sh <tag> [porta]
# Grava logs/eval_<tag>.txt com: coerencia, decode (128/400), prefill
# (2k/8k/32k) e curva de concorrencia (1/4/16). Uma linha por boot no
# logs/tuning.csv e' preenchida a mao a partir daqui.
set -uo pipefail
TAG="${1:?tag}"
PORT="${2:-3001}"
PY=${EXL3_ROOT}/venv-028/bin/python
OUT=${EXL3_ROOT}/logs/eval_${TAG}.txt
{
  echo "### $TAG  $(date -Is)"
  echo "--- coerencia"
  "$PY" -I ${EXL3_ROOT}/scripts/smoke_chat.py --port "$PORT" 2>&1 | grep -E "===|RESULTADO"
  echo "--- decode 128"
  "$PY" -I ${EXL3_ROOT}/scripts/speed_probe.py --port "$PORT" --ntok 128 --runs 2 2>&1 | grep "^run"
  echo "--- decode 400"
  "$PY" -I ${EXL3_ROOT}/scripts/speed_probe.py --port "$PORT" --ntok 400 --runs 1 2>&1 | grep "^run"
  echo "--- prefill 2k/8k/32k"
  "$PY" -I ${EXL3_ROOT}/scripts/bench_prefill.py --port "$PORT" -n 2048,8192,32768 --runs 1 2>&1 | grep -E "^n=|median"
  echo "--- concorrencia 1/4/16 (128 tok)"
  "$PY" -I ${EXL3_ROOT}/scripts/bench_concurrency.py --port "$PORT" -c 1,4,16 --ntok 128 2>&1
  echo "### fim $(date -Is)"
} | tee "$OUT"
