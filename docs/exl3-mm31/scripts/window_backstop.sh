#!/bin/bash
# Backstop da janela de GPU: gatilho por ESTADO (sentinela + heartbeat),
# nunca por relogio. Roda como unidade transitoria (systemd-run), portanto
# sobrevive a morte da sessao/CLI.
#
# Versao sanitizada: PROD_UNIT/WATCHDOG_UNIT vem do ambiente. Sem elas o
# backstop apenas encerra o servidor experimental quando a sentinela vence.
set -uo pipefail
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"
SENT="${EXL3_ROOT}/logs/WINDOW_OPEN"
LOG="${EXL3_ROOT}/logs/window-backstop.log"
STALE=${STALE_SECS:-1200}     # heartbeat vencido (20 min)
HARD_H=${HARD_HOURS:-10}      # rede final, generosa
DEADLINE=$(( $(date +%s) + HARD_H*3600 ))
echo "$(date -Is) backstop iniciado (stale=${STALE}s deadline=${HARD_H}h)" >> "$LOG"

unit_state() {  # $1 = unidade; imprime n/a se nao configurada
  if [ -n "$1" ]; then systemctl is-active "$1" 2>&1; else echo "n/a"; fi
}

while :; do
  now=$(date +%s)
  if [ ! -f "$SENT" ]; then reason="sentinela ausente"; break; fi
  age=$(( now - $(stat -c %Y "$SENT") ))
  if [ "$age" -gt "$STALE" ]; then reason="sentinela vencido (${age}s)"; break; fi
  if [ "$now" -gt "$DEADLINE" ]; then reason="deadline duro ${HARD_H}h"; break; fi
  sleep 30
done

echo "$(date -Is) RESTAURANDO producao: $reason" >> "$LOG"

# Derruba APENAS os processos do venv experimental (a producao usa outro venv).
pkill -f "${EXL3_ROOT}/venv-028/bin/python" 2>/dev/null && echo "$(date -Is) pkill venv-028 enviado" >> "$LOG"
sleep 10
pkill -9 -f "${EXL3_ROOT}/venv-028/bin/python" 2>/dev/null

# Espera a VRAM liberar (ate 150 s) antes de religar a producao.
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  echo "$(date -Is) restore tentativa $i: VRAM = ${used} MiB" >> "$LOG"
  if [ "$used" -lt 3000 ]; then break; fi
  sleep 5
done

if [ -n "$PROD_UNIT" ]; then systemctl start "$PROD_UNIT"; fi
if [ -n "$WATCHDOG_UNIT" ]; then systemctl start "$WATCHDOG_UNIT"; fi
sleep 5
echo "$(date -Is) estado: prod=$(unit_state "$PROD_UNIT") watchdog=$(unit_state "$WATCHDOG_UNIT")" >> "$LOG"
