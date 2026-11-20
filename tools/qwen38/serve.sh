#!/bin/bash
# Serve Qwen3.8-27B-NVFP4-QAD (NVFP4 weights) on SM86 through this fork.
#
# Usage: tools/qwen38/serve.sh [tp] [port] [max_model_len] [max_num_seqs] [extra vllm args...]
#
# Env:
#   MODEL_DIR   target checkpoint (default /root/models/Qwen3.8-27B-NVFP4-QAD)
#   VLLM_BIN    vllm entrypoint (default: vllm from PATH)
#   GPU_UTIL    gpu-memory-utilization (default 0.90)
#   DRAFT       DFlash2 drafter dir; enables speculative decoding when set
#   NUM_SPEC    speculative tokens per step (default 7 -> block of 8)
#   DRAFT_TP    draft_tensor_parallel_size (optional)
#
# NVFP4 on SM86 is served weight-only: MLP+lm_head through Marlin W4A16 and
# FP8 attention/GDN through MarlinFP8 W8A16. The CUTLASS FP8 kernel advertises
# support on every CUDA arch and then fails at runtime on sm_86, so it must be
# disabled explicitly.
set -u
TP=${1:-1}; PORT=${2:-3001}; MAXLEN=${3:-4096}; SEQS=${4:-4}
if [ $# -ge 4 ]; then shift 4; else shift $#; fi
V=${VLLM_BIN:-vllm}
MODEL_DIR=${MODEL_DIR:-/root/models/Qwen3.8-27B-NVFP4-QAD}
export VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
SPEC=()
if [ -n "${DRAFT:-}" ]; then
  SPEC_JSON="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":${NUM_SPEC:-7}"
  if [ -n "${DRAFT_TP:-}" ]; then
    SPEC_JSON="$SPEC_JSON,\"draft_tensor_parallel_size\":${DRAFT_TP}"
  fi
  SPEC=(--speculative-config "$SPEC_JSON}")
fi
exec "$V" serve "$MODEL_DIR" \
  --served-model-name qwen38-nvfp4 \
  --dtype bfloat16 \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$SEQS" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --port "$PORT" \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  "${SPEC[@]}" \
  "$@"
