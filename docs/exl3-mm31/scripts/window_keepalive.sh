#!/bin/bash
# Heartbeat da janela: mantem o sentinela fresco enquanto a sessao vive.
# Quando este processo morre (sessao encerrada), o sentinela vence e o
# backstop destacado religa a producao.
set -uo pipefail
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
SENT="${EXL3_ROOT}/logs/WINDOW_OPEN"
for _ in $(seq 1 720); do   # 12 h de heartbeat no maximo
  [ -f "$SENT" ] || exit 0
  touch "$SENT"
  sleep 60
done
