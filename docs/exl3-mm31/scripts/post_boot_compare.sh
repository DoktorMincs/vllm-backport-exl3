#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Espera o servidor da porta 3001, dispara o request de referencia e compara o
# dump por camada com o oraculo. Uso: post_boot_compare.sh <ref.npz> <tag>
set -uo pipefail
REF="${1:?ref npz}"
TAG="${2:-boot4}"
DUMP="${DUMPDIR:-${EXL3_ROOT}/logs/dump3}"
LOG=${EXL3_ROOT}/logs/compare_${TAG}.txt
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3001/v1/models 2>/dev/null | grep -q 200; do
  pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null || { echo "server died before ready"; exit 1; }
  sleep 3
done
echo "== API up $(date -Is) =="
# arma o dump so agora: os forwards de profiling do boot tem o mesmo T e seriam
# capturados antes do request (ver _M3_ARM em model.py)
touch $DUMP/ARM
${EXL3_ROOT}/venv-028/bin/python -I ${EXL3_ROOT}/scripts/request_ref.py --port 3001
echo "== dumps =="
ls $DUMP | wc -l
${EXL3_ROOT}/venv-028/bin/python -I ${EXL3_ROOT}/scripts/compare_full.py \
  --ref "$REF" --dump $DUMP > "$LOG" 2>&1
rc=$?
echo "== compare rc=$rc =="
head -n 3 "$LOG"
echo "..."
grep -E "^ *[0-9]+ " "$LOG" | awk 'NR<=3 || $2<0.995 || NR%10==0' | head -30
tail -n 3 "$LOG"
exit $rc
