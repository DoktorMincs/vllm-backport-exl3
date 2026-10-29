#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Boot fp16: o oraculo (ref.py) e o stack de referencia usam fp16 nas ativacoes; o plugin EXL3
# converte para fp16 dentro dos kernels de qualquer forma, mas o fluxo residual
# em bf16 (dtype auto do config) difere do oraculo a partir da camada 16.
# Espera o replay decisivo terminar (gatilho por estado, nao por relogio),
# sobe o servidor com DTYPE=float16 e repete a comparacao por camada + rotas.
set -uo pipefail
MARK=${EXL3_ROOT}/logs/replay/from_vllm_L15.log
until grep -q "argmax replay" "$MARK" 2>/dev/null; do sleep 3; done
while pgrep -f "[r]eplay_from_dump.py" >/dev/null; do sleep 2; done
echo "== replay terminou $(date -Is) =="

# Antes de subir: a cabeca (norm final + lm_head K6) e testada a partir do
# estado L59 do proprio vLLM. Se o oraculo, partindo do estado do vLLM,
# devolver uma distribuicao CONFORTANTE (' a'), o lm_head do vLLM esta errado;
# se devolver a mesma distribuicao achatada do vLLM, a cabeca esta fiel e a
# diferenca vem do estado (caos).
${EXL3_ROOT}/venv-028/bin/python -I ${EXL3_ROOT}/scripts/replay_from_dump.py \
  --ref ${EXL3_ROOT}/logs/ref_full.npz --dump ${EXL3_ROOT}/logs/dump3 \
  --start 59 --source vllm --skip-dump-cmp \
  > ${EXL3_ROOT}/logs/replay/head_from_vllm_L59.log 2>&1
echo "== head check pronto $(date -Is) =="

echo "== subindo fp16 $(date -Is) =="

export DUMPDIR=${EXL3_ROOT}/logs/dump_fp16
export DTYPE=float16
nohup ${EXL3_ROOT}/scripts/run_serve_dbg.sh boot6 > ${EXL3_ROOT}/logs/boot6.driver.out 2>&1 &
DUMPDIR="$DUMPDIR" ${EXL3_ROOT}/scripts/post_boot_compare.sh \
  ${EXL3_ROOT}/logs/ref_full.npz boot6
