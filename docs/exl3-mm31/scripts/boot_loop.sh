#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# A/B: camadas de K uniforme (todas menos a L17) pelo loop Python de referencia.
#
# Motivacao (27/10): partindo do estado L17 do vLLM (que bate com o oraculo a
# 1e-6), na L18 a atencao e o ffn_in batem a ~3e-4 mas o ffn_out diverge com
# cos 0.32 / relerr 1.67. A L18 e' a primeira camada MoE de K uniforme 3
# (3..16 e 46..59 sao K4 e batem; a 17 e' mixed-K e bate). O roteamento da L18
# tem margem 1.3e-2..7.8e-2, entao nao e' desempate de gate.
#
# Liga EXL3_UNIFORM_LOOP=1 (patch local no plugin), que manda SO' as camadas de
# K uniforme para apply_exl3_python_loop. Se o loop reproduzir o oraculo na L18,
# o defeito esta no caminho fundido `batched` (exl3_moe); se nao, esta no
# carregamento dos trellis K3 ou no roteamento/escala.
set -uo pipefail
export EXL3_FORCE_MIXEDK=0   # o A/B do loop so' faz sentido com o mixedk desligado
export EXL3_UNIFORM_LOOP=1
export EXL3_FUSED_MOE=1
export DTYPE=float16
exec ${EXL3_ROOT}/scripts/run_serve.sh "${1:-boot12}"
