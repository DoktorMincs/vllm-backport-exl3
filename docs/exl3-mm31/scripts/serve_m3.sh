#!/bin/bash

# Sanitized for publication: paths and production units come from the environment.
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm-backport}"
MM3_REF="${MM3_REF:-$HOME/mm3-oracle/ref.py}"
PROD_DIR="${PROD_DIR:-$HOME/production}"
PROD_UNIT="${PROD_UNIT:-}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-}"

# Sobe o MiniMax-M3.1-Flash EXL3 no vllm-backport (venv isolado).
#
# Geometria padrao: TP=1 x PP=10 (o pack de 176 GiB nao cabe em 5 GPUs com
# TP=1). PP=10 divide 60 camadas.
#
# ATENCAO — TP=2 x PP=5 subiu quebrado em 27/10 (ruido: 'etimetim...'): a fatia
# do lm_head (200064 -> 100032 por rank) cortava um bloco de 128 da Hadamard de
# saida do EXL3, e o rank 1 decodificava contra a transformada errada. Corrigido
# no plugin (`_exl3_pad_col_window`) e verificado; depois disso TP=2 x PP=5 e' a
# config vencedora (45,3 tok/s, prefill 1491/2126/2357). Ver STATUS.md e
# serve_best.sh.
#
# VLLM_PP_LAYER_PARTITION balanceia pelo tamanho real do pack (K4 ~3.5 GiB,
# K3 ~2.66 GiB): maximo 19.15 GiB/estagio em vez de 21.7 do split uniforme.
#
# Uso: serve_m3.sh [porta] [max_model_len] [tp] [pp]
set -uo pipefail

# MODEL=... seleciona outro pack (ex.: o 3.06 bpw, que tem 153 GiB de camadas
# contra 177 do 3.5 e particao PP propria: 13,12,12,12,11).
MODEL="${MODEL:-${MODEL_DIR}}"
PY=${EXL3_ROOT}/venv-028/bin/python
PORT="${1:-3001}"
MAXLEN="${2:-8192}"
TP="${3:-1}"
PP="${4:-10}"
HOST="${HOST:-0.0.0.0}"
API_KEY="${API_KEY:-}"

# Balanceamento por bytes reais do pack (60 camadas, contiguas):
# 0-6 7 | 7-11 5 | 12-16 5 | 17-23 7 | 24-30 7 | 31-37 7 | 38-44 7 |
# 45-49 5 | 50-54 5 | 55-59 5   -> max 19.15 GiB (stage 3), min 16.65
if [ "$PP" = "10" ] && [ "$TP" = "1" ]; then
  export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-7,5,5,7,7,7,7,5,5,5}"
fi
# TP=2 x PP=5 (10 GPUs): 12 camadas/estagio por GPU = 176 GiB / 10 GPUs, com
# maximo 18.74 GiB por GPU (pp_balance.py --stages 5 --tp 2), menos que os
# 19.31 do TP=1/PP=10. O split 12,12,14,12,10 existe porque o ultimo estagio
# leva o lm_head e o primeiro o embed.
if [ "$PP" = "5" ] && [ "$TP" = "2" ]; then
  export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-12,12,14,12,10}"
  # A unica barreira no plugin era a guarda do indexer (patch local
  # EXL3_ALLOW_TP_INDEXER); os experts roteados ja' sao TP-cientes.
  export EXL3_ALLOW_TP_INDEXER="${EXL3_ALLOW_TP_INDEXER:-1}"
  export VLLM_EXL3_MOE_TP_ALIGN="${VLLM_EXL3_MOE_TP_ALIGN:-128}"
fi

export VLLM_EXL3_MOE_KERNEL="${VLLM_EXL3_MOE_KERNEL:-native}"
# CORRECAO 27/10 (medida): o kernel MoE fundido de K UNIFORME (`exl3_moe`,
# caminho "batched") acerta K=4 e ERRA K=3 neste pack. Partindo do estado L17
# do proprio vLLM (que bate com o oraculo a 1e-6), a L18 — a primeira camada de
# K3 uniforme — tem atencao cos 1.000000 e ffn_in relerr 3e-4, mas ffn_out cos
# 0.32 / relerr 1.67, com margem de roteamento 1.3e-2..7.8e-2 (nao e' empate).
# Com as camadas K3 no loop de referencia o prefill volta a '<mm:think>' p=0,99
# (oraculo: 0,99). O kernel mixed-K (K por expert) acerta K=3, entao mandamos
# as camadas de K uniforme por ele (patch local EXL3_FORCE_MIXEDK=1).
export EXL3_FORCE_MIXEDK="${EXL3_FORCE_MIXEDK:-1}"
# EXL3_FUSED_MOE=0 troca o apply do MoE pelo loop Python por expert (caminho
# de referencia). Serve para bissectar kernel fundido x orquestracao.
export EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
# apply_exl3_batched_fat (tier "fat", >=256 linhas por expert) mediu cos 0.12
# contra a referencia na geometria do M3 -> numerico errado. Threshold enorme
# desliga o tier; as linhas voltam ao exl3_moe fundido (tiles m32/m64).
export VLLM_EXL3_FAT_THRESHOLD="${VLLM_EXL3_FAT_THRESHOLD:-100000000}"
export VLLM_EXL3_Q4KV="${VLLM_EXL3_Q4KV:-0}"          # q4kv e GB10/sm_121
export VLLM_EXL3_INDEXER_MXFP8="${VLLM_EXL3_INDEXER_MXFP8:-0}"  # idem
# PATCH #2: o marker de arquitetura do plugin so e setado no processo do
# engine (ModelConfig.__post_init__ -> override_quantization_method); o load
# acontece nos workers, que nunca chamam esse hook. Sem esta flag o M3 cai no
# escopo default "glm53_routed_experts_only" e atencao/densa/lm_head ficariam
# BF16 (pesos do pack sao trellis) -> load quebrado.
export VLLM_EXL3_MINIMAX_M3="${VLLM_EXL3_MINIMAX_M3:-1}"
# O venv veio do stack de referencia com flashinfer 0.6.16.post3 e
# flashinfer-cubin 0.6.13.
# O backport importa flashinfer no module-load de um passe de fusao
# (rms_quant_fusion) que nos nao usamos (--enforce-eager), e o import morre no
# version check. Bypass documentado; nenhum kernel flashinfer entra no caminho.
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
# O RPC "sample_tokens" espera no maximo VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
# (default 300 s). Com o dump de depuracao ligado um prefill de 210 tokens
# passou de 300 s (grava ~900 MB de npz) e o engine morreu com
# "TimeoutError: RPC call to sample_tokens timed out" (boot9, 27/10 02:56).
# 1800 s cobre o dump; em producao normal o passo leva ~0,2-0,5 s.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
# NAO definir VLLM_EXL3_MODEL_DIR: com ele o from_config leria o ledger
# quantization_config.json (prefixos language_model.*) e a cobertura
# non_routed casaria errado; sem ele vale o caminho MiniMax-M3 por arquitetura.
unset VLLM_EXL3_MODEL_DIR

# DTYPE: fp16 por padrao. O pack declara bfloat16 no config, mas o oraculo
# (ref.py) e o stack de referencia rodam fp16, e o bf16 do fluxo residual injeta ~0,5%
# de erro ate a camada 15 — o suficiente para desempatar roteamentos (margens de
# ~1e-3) e jogar a trajetoria para fora da bacia de referencia. Medido em
# 27/10: fp16 deu distribuicao sensata no 1o token (' The'/' a'/' rich'),
# bf16 deu ':' com top-8 achatado. DTYPE=bf16 volta ao comportamento antigo.
DTYPE_ARGS=()
if [ -n "${DTYPE:-float16}" ]; then DTYPE_ARGS=(--dtype "${DTYPE:-float16}"); fi

# EAGER=0 (PADRAO agora) tira o --enforce-eager e libera torch.compile + CUDA
# graphs: decode 3,5 -> 21 tok/s (6,1x) em 27/10, com a mesma numerica (prefill
# '<mm:think>' -0,014 e consistencia prefill x decode top-1 identica). EAGER=1
# volta ao modo antigo, ~3,5 tok/s. TP 1->2 e PP 10->5 foram medidos e nao
# mudaram nada, entao o custo e' overhead fixo por camada — grafo e' a alavanca.
EAGER_ARGS=()
if [ "${EAGER:-0}" != "0" ]; then EAGER_ARGS=(--enforce-eager); fi

# ---------------------------------------------------------------------------
# GUARD — zero CPU offload de pesos dos pesos (requisito duro da campanha).
# Qualquer um destes caminhos tira peso do modelo da VRAM e passa a le-lo da RAM
# do host ou do page cache: a medicao deixaria de ser "o modelo cabe nas 3090".
# Aborta o boot em vez de degradar em silencio.
offload_hits=()
[ "${VLLM_PLE_CPU_OFFLOAD:-0}" != "0" ] && \
  offload_hits+=("VLLM_PLE_CPU_OFFLOAD=${VLLM_PLE_CPU_OFFLOAD}")
[ "${VLLM_EXL3_ATS_ALIAS:-0}" != "0" ] && \
  offload_hits+=("VLLM_EXL3_ATS_ALIAS=${VLLM_EXL3_ATS_ALIAS} (aliasa experts em RAM)")
[ "${VLLM_EXL3_NGRAM_TABLE:-}" = "disk" ] && \
  offload_hits+=("VLLM_EXL3_NGRAM_TABLE=disk (gather do page cache)")
while IFS= read -r _name; do
  [ -n "$_name" ] && offload_hits+=("${_name}=${!_name}")
done < <(compgen -v | grep '^VLLM_WEIGHT_OFFLOADING_' || true)
case " $* " in
  *"--cpu-offload-gb"*) offload_hits+=("argumento --cpu-offload-gb") ;;
esac
if [ "${#offload_hits[@]}" -gt 0 ]; then
  echo "ERRO: CPU offload de pesos detectado — este boot esta' PROIBIDO:" >&2
  printf '  %s\n' "${offload_hits[@]}" >&2
  echo "Desligue essas variaveis e suba de novo." >&2
  exit 97
fi

# Perfil de kernels: NSYS_OUT=<prefixo> sobe o servidor sob o Nsight Systems
# (o boot inteiro entra no trace; filtre as fases depois com
# `nsys stats --filter-time`). --trace-fork-before-exec e' obrigatorio porque
# os 10 workers de PP sao processos filhos, e --cuda-graph-trace=node e'
# obrigatorio para ver dentro dos grafos (decode e' 100% grafo).
NSYS_ARGS=()
if [ -n "${NSYS_OUT:-}" ]; then
  # NSYS_GRAPH=0 tira o `--cuda-graph-trace=node`: o trace fica ~10x menor
  # (o replay de grafo gera ~30k kernels/s) e serve para medir PREFILL, que
  # nao roda em grafo. Use node so quando o alvo for o decode.
  GRAPH_ARG=()
  [ "${NSYS_GRAPH:-1}" = "1" ] && GRAPH_ARG=(--cuda-graph-trace=node)
  NSYS_ARGS=(nsys profile --force-overwrite true -o "$NSYS_OUT" \
    -t cuda "${GRAPH_ARG[@]}" --sample=none --cpuctxsw=none \
    --trace-fork-before-exec=true)
fi

# EXTRA_ARGS: flags extras do vLLM para experimentos de tuning (separadas por
# espaco). Ex.: EXTRA_ARGS="--max-num-batched-tokens 8192 --max-num-seqs 32"
EXTRA=()
if [ -n "${EXTRA_ARGS:-}" ]; then read -r -a EXTRA <<< "${EXTRA_ARGS}"; fi

# O M3 fecha o reasoning com <mm:think>...</mm:think>, nao com  thinking/response.
# Sem --reasoning-parser o texto bruto cai em `content` e os clientes mostram a
# tag; o parser `minimax_m3` (vllm/reasoning/minimax_m3_reasoning_parser.py) manda
# o bloco para `reasoning_content`. REASONING_PARSER= (vazio) desliga.
REASONING_PARSER="${REASONING_PARSER-minimax_m3}"
REASONING=()
if [ -n "$REASONING_PARSER" ]; then
  REASONING=(--reasoning-parser "$REASONING_PARSER")
fi

# Vision tower. `--language-model-only` zera o limite de TODAS as modalidades
# (multimodal.py:get_limit_per_prompt devolve 0), e o sintoma no cliente e'
# "At most 0 image(s) may be provided in one prompt". O pack TEM a torre
# (515 tensores vision_tower.* + 4 projectors, em bf16, fora do EXL3).
#   LANGUAGE_MODEL_ONLY=1 -> reproduz as configs de texto (pool maior).
#   MM_LIMITS (default '{"image": 2}') controla quantas imagens por prompt.
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
MM_ARGS=()
if [ "$LANGUAGE_MODEL_ONLY" = "1" ]; then
  MM_ARGS=(--language-model-only)
else
  # Default in a separate assignment: the closing brace of the JSON would
  # otherwise be eaten by the ${VAR:-...} expansion and vLLM gets '{"image": 2}}'.
  MM_LIMIT_SPEC="${MM_LIMITS:-}"
  if [ -z "$MM_LIMIT_SPEC" ]; then MM_LIMIT_SPEC='{"image": 2, "video": 1}'; fi
  MM_ARGS=(--limit-mm-per-prompt "$MM_LIMIT_SPEC")
fi

# Bind e autenticacao: o default e' 0.0.0.0 sem API key. Em rede compartilhada,
# defina HOST=127.0.0.1 (ou API_KEY=...) ou coloque um proxy/firewall na frente.
AUTH_ARGS=()
if [ -n "$API_KEY" ]; then AUTH_ARGS=(--api-key "$API_KEY"); fi

exec "${NSYS_ARGS[@]}" "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "${SERVED_NAME:-minimax-m3.1-flash}" \
  "${REASONING[@]}" \
  --quantization exl3 \
  "${DTYPE_ARGS[@]}" \
  --tensor-parallel-size "$TP" \
  --pipeline-parallel-size "$PP" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "${GPU_UTIL:-0.93}" \
  "${EAGER_ARGS[@]}" \
  "${MM_ARGS[@]}" \
  --port "$PORT" \
  --host "$HOST" \
  "${AUTH_ARGS[@]}" \
  "${EXTRA[@]}"
