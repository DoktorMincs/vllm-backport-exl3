#!/bin/bash
# Aplica um powercap em TODAS as GPUs que o driver enxergar (nunca lista fixa:
# em 23/10 uma lista 0..7 deixou as GPUs 8 e 9 a 170 W e o prefill caiu).
# Uso: set_cap.sh [230]
# O stack de referencia rodava a 190/200 W; restaure o seu cap depois da
# campanha.
set -uo pipefail
WATTS="${1:-230}"
n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for i in $(seq 0 $((n - 1))); do
  nvidia-smi -i "$i" -pl "$WATTS" >/dev/null 2>&1 || echo "falha na GPU $i" >&2
done
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader
