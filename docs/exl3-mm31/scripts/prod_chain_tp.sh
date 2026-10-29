#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Bateria de verificacao para a geometria TP=2 x PP=5 (e para o coop no decode).
#   1) prefill-only por ids: o top-1 tem de ser '<mm:think>' (~ -0.01) — mede o
#      caminho de prefill, que continua no kernel mixedk;
#   2) chat com a amostragem do generation_config (temp 1.0 / top_p 0.95):
#      coerencia + velocidade de decode;
#   3) consistencia prefill x decode: detecta kernel de decode errado (coop)
#      sem precisar do oraculo.
# Uso: prod_chain_tp.sh <tag>
set -uo pipefail
TAG="${1:-tp2pp5}"
PY=${EXL3_ROOT}/venv-028/bin/python
OUT=${EXL3_ROOT}/logs/prod_chain_${TAG}.out
IDS=${EXL3_ROOT}/logs/ids_big.json

: > "$OUT"
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3001/v1/models 2>/dev/null | grep -q 200; do
  pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null || { echo "server died before ready" >> "$OUT"; exit 1; }
  sleep 3
done
echo "== API up $(date -Is) ==" >> "$OUT"

{
  echo "== prefill-only (ids) $(date -Is) =="
  $PY -I ${EXL3_ROOT}/scripts/probe_ids.py --ids "$IDS" --steps 1 --topk 8 --temp 0
  echo "== chat temp1.0/top_p0.95 $(date -Is) =="
  $PY -I ${EXL3_ROOT}/scripts/chat_sample.py --ntok 200 --temp 1.0 --top-p 0.95
  echo "== consistencia prefill x decode $(date -Is) =="
  $PY -I ${EXL3_ROOT}/scripts/prefix_consistency.py --ids "$IDS" --steps 4 \
      --out ${EXL3_ROOT}/logs/prefix_cons_${TAG}.json
} >> "$OUT" 2>&1

echo "== fim $(date -Is) ==" >> "$OUT"
