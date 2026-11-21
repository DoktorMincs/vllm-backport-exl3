#!/bin/bash
# Pipeline completo do pack do turboderp (2,68 bpw) para o nosso fork.
#
#   1. make_vllm_pack.py  : layout exllamav3 -> layout do vLLM (overlay bf16 de
#                           wo_a/head/indexer.wk/compressor.*, index, config).
#                           Precisa de GPU (had_r_128 e o decode das fatias).
#   2. export_dspark.py   : overlays do drafter (attn.wo_a bf16 + main_proj) e a
#                           cobertura EXL3 dos densos do MTP.
#   3. turbo_prepare.py   : liga as nossas tabelas de Engram por symlink e tira
#                           as chaves delas do index (zero bytes copiados).
#
# Rodar SEM nenhum serve ocupando as GPUs (a conversion decodifica na GPU 0).
# Uso: turbo_pack_build.sh [--verify] [--gpu 0]
set -uo pipefail
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODELS="${MODELS:-$HOME/models}"
cd "$EXL3_ROOT" || exit 1

SRC="${TURBO_SRC:-$MODELS/DS41-turbo-2.68bpw}"
DST="${TURBO_DST:-$MODELS/DS41-turbo-2.68bpw-vllm}"
PY="${PYTHON:-$EXL3_ROOT/venv/bin/python}"
GPU=0
VERIFY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --verify) VERIFY="--verify"; shift ;;
    --gpu) GPU="$2"; shift 2 ;;
    *) echo "arg desconhecido: $1" >&2; exit 2 ;;
  esac
done

# GPU livre antes de comecar: um serve rodando com util 0.97 nao deixa espaco
# e a conversao morre no meio (ou derruba o serve).
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" != "0" ]; then
  echo "ERRO: $busy processo(s) usando GPU; derrube o serve antes." >&2
  exit 3
fi

export MVPSRC="$SRC" MVPDST="$DST" MVPOVERLAY="vllm-overlay.safetensors"
mkdir -p "$DST"

echo "=== 1/4 make_vllm_pack $(date -Is) ==="
"$PY" scripts/make_vllm_pack.py --device "cuda:$GPU" $VERIFY || {
  echo "FALHOU make_vllm_pack" >&2; exit 10; }

echo "=== 2/4 export_dspark $(date -Is) ==="
# O export_dspark le mp.DST/mp.SRC, que vem das mesmas envs.
"$PY" scripts/export_dspark.py --device "cuda:$GPU" || {
  echo "FALHOU export_dspark" >&2; exit 11; }

echo "=== 3/4 turbo_prepare $(date -Is) ==="
"$PY" scripts/turbo_prepare.py --dst "$DST" || {
  echo "FALHOU turbo_prepare" >&2; exit 12; }

echo "=== 4/4 config do turboderp (taps do drafter) $(date -Is) ==="
"$PY" scripts/turbo_config_patch.py --dst "$DST" || {
  echo "FALHOU turbo_config_patch" >&2; exit 13; }

echo "=== pronto $(date -Is) ==="
du -sh "$DST"
