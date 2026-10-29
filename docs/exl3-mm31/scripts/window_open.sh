#!/bin/bash
# Abre a janela de GPU: snapshot -> sentinela -> backstop (systemd-run, sobrevive
# a sessao) -> SO DEPOIS para watchdog+producao. A producao volta quando o
# sentinela sumir/vencer.
#
# Versao sanitizada: os caminhos e as unidades de systemd vem do ambiente.
#   EXL3_PORT     raiz do ambiente experimental (default: ~/exl3-port)
#   PROD_UNIT     unidade systemd da producao (vazio = nao mexe em producao)
#   WATCHDOG_UNIT unidade do watchdog (opcional)
#   PROD_DIR      diretorio da producao, apenas para o snapshot (opcional)
set -uo pipefail
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"
SENT="${EXL3_ROOT}/logs/WINDOW_OPEN"
LOG="${EXL3_ROOT}/logs/window.log"
SNAP="${EXL3_ROOT}/logs/snapshot-pre-janela-$(date +%Y%m%d-%H%M%S).txt"

unit_state() {  # $1 = unidade; imprime n/a se nao configurada
  if [ -n "$1" ]; then systemctl is-active "$1" 2>&1; else echo "n/a"; fi
}

{
  echo "== $(date -Is) abrindo janela =="
  if [ -n "${PROD_UNIT}${WATCHDOG_UNIT}" ]; then
    systemctl show -p Id,ActiveState,SubState,ExecMainStartTimestamp,MainPID \
      ${PROD_UNIT} ${WATCHDOG_UNIT} 2>&1
  fi
  nvidia-smi --query-gpu=index,memory.used,memory.total,name --format=csv 2>&1
  echo "--- configs"
  if [ -d "$PROD_DIR" ]; then
    sha256sum "$PROD_DIR"/*.json 2>/dev/null
    echo "--- git"
    git -C "$PROD_DIR" rev-parse --short HEAD 2>&1
  fi
  echo "--- processos"
  ps -eo pid,etime,args | grep -E "[v]llm" | head -20
} > "$SNAP" 2>&1

echo "$(date -Is) snapshot: $SNAP" >> "$LOG"

# 1) Sentinela ANTES de mexer na producao: se algo falhar daqui pra frente, o
#    backstop (ja vivo) ve a sentinela vencida e religa tudo.
rm -f "$SENT"
touch "$SENT"

# 2) Backstop via systemd-run: unidade transitoria, nao morre com a sessao.
systemctl stop exl3-window-backstop.service 2>/dev/null
systemd-run --collect --unit=exl3-window-backstop \
  --description="Backstop da janela GPU EXL3" \
  "${EXL3_ROOT}/scripts/window_backstop.sh" >/dev/null 2>&1
sleep 2
echo "$(date -Is) backstop: $(systemctl is-active exl3-window-backstop 2>&1)" >> "$LOG"
if ! systemctl is-active --quiet exl3-window-backstop; then
  echo "$(date -Is) ERRO: backstop nao subiu; nao vou parar a producao" >> "$LOG"
  rm -f "$SENT"
  exit 2
fi

# 3) Para o watchdog primeiro (senao ele reage ao stop da producao), depois a
#    producao.
if [ -n "${WATCHDOG_UNIT}${PROD_UNIT}" ]; then
  systemctl stop ${WATCHDOG_UNIT} ${PROD_UNIT}
  sleep 8
fi

# 4) Espera as GPUs esvaziarem; aborta (restaurando) se nao esvaziarem.
free_ok=0
used=-1
for i in $(seq 1 24); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  echo "$(date -Is) tentativa $i: VRAM usada total = ${used} MiB" >> "$LOG"
  if [ "$used" -lt 3000 ]; then free_ok=1; break; fi
  sleep 5
done

if [ "$free_ok" != "1" ]; then
  echo "$(date -Is) ERRO: VRAM nao liberou (${used} MiB); removendo sentinela p/ o backstop religar" >> "$LOG"
  rm -f "$SENT"
  exit 3
fi

echo "$(date -Is) JANELA ABERTA (prod=$(unit_state "$PROD_UNIT") backstop=$(systemctl is-active exl3-window-backstop))" >> "$LOG"
