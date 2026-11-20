#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Vigia de log: acorda no primeiro sinal (ok OU falha), sem polling manual.
# Uso: watch_log.sh <log> <regex-ok> [segundos-max] [exit-esperado]
set -uo pipefail
LOG="$1"; PAT="$2"; MAX="${3:-3600}"; MARK="${LOG%.log}.exit"
ERR='Traceback \(most recent call last\)|CUDA out of memory|torch.OutOfMemoryError|Engine core initialization failed|EngineCore failed|RuntimeError: |AssertionError|Killed|Segmentation fault|startup failed'
for _ in $(seq 1 "$MAX"); do
  if grep -qE "$PAT" "$LOG" 2>/dev/null; then
    echo "OK-PATTERN: $PAT"; exit 0
  fi
  if grep -qE "$ERR" "$LOG" 2>/dev/null; then
    echo "ERRO no log:"; grep -nE "$ERR" "$LOG" | head -5; exit 2
  fi
  if [ -f "$MARK" ]; then
    echo "processo terminou exit=$(cat "$MARK" 2>/dev/null)"; exit 3
  fi
  sleep 1
done
echo "TIMEOUT (${MAX}s) sem '$PAT'"
exit 4
