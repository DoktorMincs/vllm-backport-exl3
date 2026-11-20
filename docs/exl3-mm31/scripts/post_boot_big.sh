#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Espera o servidor, arma o dump e dispara a bateria do prompt longo:
# 1 passo (distribuicao do prefill) e 4 passos (prefill + 3 decodes).
# Uso: post_boot_big.sh <tag>   (DUMPDIR no ambiente)
set -uo pipefail
TAG="${1:-big}"
DUMP="${DUMPDIR:-${EXL3_ROOT}/logs/dump_big}"
PY=${EXL3_ROOT}/venv-028/bin/python
IDS=${EXL3_ROOT}/logs/ids_big.json
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3001/v1/models 2>/dev/null | grep -q 200; do
  pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null || { echo "server died before ready"; exit 1; }
  sleep 3
done
echo "== API up $(date -Is) =="
touch "$DUMP/ARM"
echo "== passo 1 (prefill) =="
$PY -I ${EXL3_ROOT}/scripts/request_ids.py --ids "$IDS" --steps 1 --out ${EXL3_ROOT}/logs/req_${TAG}_1.json
echo "== dumps: $(ls "$DUMP" | wc -l) =="
echo "== 4 passos (prefill + decode) =="
$PY -I ${EXL3_ROOT}/scripts/request_ids.py --ids "$IDS" --steps 4 --out ${EXL3_ROOT}/logs/req_${TAG}_4.json
echo "== fim $(date -Is) =="
