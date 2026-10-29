#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# CONFIG COM DSpark (drafter do MiniMax-M3) — medida em 27/10.
#
# Decode 54,0 tok/s vs 44,8 sem drafter (+20%), temp 0, 128 tokens.
# Aceitacao por posicao 0,612/0,401 (L=2,01), draft acceptance 59% na pos 0.
#
# As tres restricoes que tornam esta config o que ela e':
#   --num-speculative-tokens 2 : N=3 empata (51,6), N=1 PERDE (47,0) e N=7
#     perde feio (42,6). O custo do passo e' dominado pelo fixo (bolha do PP),
#     entao o que manda e' L (tokens por passo): 2,01 (N=2) / 2,15 (N=3) /
#     1,59 (N=1). N=3 e' melhor no AGREGADO (C=4: 53,1 vs 48,5) porque o balde
#     de CUDA graph arredonda 12 linhas para o tamanho de 16.
#     N menor tambem ACEITA MELHOR nas posicoes 1-2 (0,401 no N=2 vs 0,227 no
#     N=3 vs 0,115 no N=7): menos slots de ruido degradam o bloco do drafter.
#   --max-num-seqs 4 : TETO DURO. O lm_head e' o EXL3 do alvo e, com 17..144
#     linhas, o plugin cai em `reconstruct` (materializa 1,23 GiB/rank). Numa
#     captura de CUDA graph esse temporario fica no pool do graph -> OOM no
#     startup. Limite: max_num_seqs * (N+1) <= 16 (4 * 4 = 16).
#   MAXLEN 32768 : o estagio do drafter (8 camadas + drafter) fica apertado;
#     contextos maiores nao foram medidos nessa particao.
#
# Custo medido: em C=4 o agregado cai de 62,9 (sem drafter) para 48,5-53,1.
# Ou seja: config de LATENCIA single-stream, nao de servidor concorrente.
#
# Medir SEMPRE com scripts/speed_probe2.py / bench_concurrency2.py: o
# speed_probe.py conta chunks SSE e com spec decode o servidor manda ~2 tokens
# por chunk (reporta ~18 tok/s falso).
set -uo pipefail
export MAXLEN="${MAXLEN:-32768}"
export EAGER=0
export GPU_UTIL="${GPU_UTIL:-0.92}"
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-13,13,14,12,8}"
DSPARK="${MODEL_DIR}/dspark"
export EXTRA_ARGS="--max-num-batched-tokens 512 --max-num-seqs 4 \
--speculative-config {\"method\":\"dspark\",\"model\":\"${DSPARK}\",\"num_speculative_tokens\":${NSPEC:-2}}"
exec ${EXL3_ROOT}/scripts/boot_tp2pp5.sh "${1:-dspark}"
