#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Vigia de RAM do host durante o load experimental.
#
# O boot tp2pp5-coop2 (27/10 08:53) encheu 96 GB de RAM + 16 GB de swap e o
# OOM-killer derrubou tudo. Este guarda amostra MemAvailable e, se ela cair
# abaixo do piso por duas amostras seguidas, mata SO' o venv experimental
# (a producao fica de pe') e registra o pico no log.
#
# Uso: mem_guard.sh <tag> [piso_MB=8000] [intervalo_s=5] [porta=3001]
# A porta entra no pgrep que decide se o servidor ainda existe (o experimental
# agora sobe na 3000 com --served-model-name minimax-m2.7).
set -uo pipefail
TAG="${1:-mem}"
FLOOR="${2:-8000}"
STEP="${3:-5}"
PORT_PAT="${4:-3001}"
LOG=${EXL3_ROOT}/logs/mem_guard_${TAG}.log
PY_PAT=${EXL3_ROOT}/venv-028/bin/python

peak=0
low=0
while :; do
  avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
  anon=$(awk '/^AnonPages:/ {print int($2/1024)}' /proc/meminfo)
  cache=$(awk '/^Cached:/ {print int($2/1024)}' /proc/meminfo)
  swap=$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print int((t-f)/1024)}' /proc/meminfo)
  used=$((anon))
  [ "$used" -gt "$peak" ] && peak=$used
  echo "$(date -Is) avail=${avail}MB anon=${anon}MB cache=${cache}MB swap=${swap}MB peak_anon=${peak}MB" >> "$LOG"
  if [ "${avail:-99999}" -lt "$FLOOR" ]; then
    low=$((low + 1))
    echo "$(date -Is) BAIXO #${low} (avail=${avail} < ${FLOOR})" >> "$LOG"
    if [ "$low" -ge 2 ]; then
      echo "$(date -Is) ABORTANDO: matando o venv experimental" >> "$LOG"
      pkill -TERM -f "$PY_PAT" 2>/dev/null
      sleep 10
      pkill -KILL -f "$PY_PAT" 2>/dev/null
      echo "$(date -Is) guarda encerrou (pico anon=${peak}MB)" >> "$LOG"
      exit 2
    fi
  else
    low=0
  fi
  if ! pgrep -f "[v]llm.entrypoints.openai.api_server.*${PORT_PAT}" >/dev/null \
     && ! pgrep -f "[b]oot_tp2pp5" >/dev/null; then
    echo "$(date -Is) servidor experimental nao esta' rodando; guarda encerrou (pico anon=${peak}MB)" >> "$LOG"
    exit 0
  fi
  sleep "$STEP"
done
