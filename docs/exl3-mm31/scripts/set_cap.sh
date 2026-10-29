#!/bin/bash
# Aplica um powercap em TODAS as GPUs que o driver enxergar (nunca lista fixa:
# em 23/10 uma lista 0..7 deixou as GPUs 8 e 9 a 170 W e o prefill caiu).
# Uso: set_cap.sh [230] [--yes]
# O stack de referencia rodava a 190/200 W; restaure o seu cap depois da
# campanha. Em host compartilhado isto atinge a producao: o script mostra as
# GPUs e pede confirmacao antes de aplicar (--yes pula o prompt, para
# automacao).
set -uo pipefail
WATTS=230
YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes) YES=1; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    -*) echo "opcao desconhecida: $1" >&2; exit 2 ;;
    *) WATTS="$1"; shift ;;
  esac
done

n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$n" -eq 0 ]; then
  echo "nenhuma GPU visivel" >&2
  exit 1
fi

echo "power limit alvo: ${WATTS} W em ${n} GPU(s):"
nvidia-smi --query-gpu=index,name,power.limit --format=csv,noheader
if [ "$YES" != "1" ]; then
  if [ -t 0 ]; then
    read -r -p "confirma? [y/N] " ans || ans=""
    case "$ans" in
      y|Y|yes|YES) ;;
      *) echo "cancelado" >&2; exit 1 ;;
    esac
  else
    echo "ERRO: modo nao interativo exige --yes" >&2
    exit 2
  fi
fi

for i in $(seq 0 $((n - 1))); do
  nvidia-smi -i "$i" -pl "$WATTS" >/dev/null 2>&1 || echo "falha na GPU $i" >&2
done
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader
