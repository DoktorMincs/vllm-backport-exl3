#!/bin/bash
# Fecha a janela: derruba o servidor experimental, espera a VRAM, remove o
# sentinela e confirma que a producao voltou.
#
# Versao sanitizada: PROD_UNIT/WATCHDOG_UNIT vem do ambiente. Sem PROD_UNIT o
# script apenas encerra o servidor experimental e remove o sentinela.
set -uo pipefail
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"
SENT="${EXL3_ROOT}/logs/WINDOW_OPEN"
LOG="${EXL3_ROOT}/logs/window.log"

unit_state() {  # $1 = unidade; imprime n/a se nao configurada
  if [ -n "$1" ]; then systemctl is-active "$1" 2>&1; else echo "n/a"; fi
}

pkill -f "${EXL3_ROOT}/venv-028/bin/python" 2>/dev/null && sleep 10
pkill -9 -f "${EXL3_ROOT}/venv-028/bin/python" 2>/dev/null
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  echo "$(date -Is) close tentativa $i: VRAM = ${used} MiB" >> "$LOG"
  if [ "$used" -lt 3000 ]; then break; fi
  sleep 5
done

rm -f "$SENT"
echo "$(date -Is) sentinela removido; aguardando backstop" >> "$LOG"
if [ -z "$PROD_UNIT" ]; then
  echo "$(date -Is) PROD_UNIT nao definido; nada a religar" >> "$LOG"
  exit 0
fi
for i in $(seq 1 30); do
  if systemctl is-active --quiet "$PROD_UNIT"; then
    echo "$(date -Is) producao de volta: prod=$(unit_state "$PROD_UNIT") watchdog=$(unit_state "$WATCHDOG_UNIT")" >> "$LOG"
    exit 0
  fi
  sleep 5
done
echo "$(date -Is) AVISO: ${PROD_UNIT} nao voltou em 150 s; subindo manualmente" >> "$LOG"
systemctl start "$PROD_UNIT"
[ -n "$WATCHDOG_UNIT" ] && systemctl start "$WATCHDOG_UNIT"
exit 1
