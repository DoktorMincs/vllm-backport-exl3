# RUNBOOK — Qwen3.8-27B-NVFP4-QAD em SM86 (RTX 3090)

Serve o `nvidia/Qwen3.8-27B-NVFP4-QAD` (NVFP4, ~19,9 GB) em GPUs Ampere sem
suporte nativo a NVFP4. O caminho é **weight-only**: MLP + `lm_head` vão pelo
Marlin **W4A16**, atenção/GDN FP8 pelo **MarlinFP8 W8A16**, e a torre de visão
fica em BF16. Não há kernel NVFP4 nativo — o formato é só o empacotamento dos
pesos.

## Armadilha obrigatória

```
export VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel
```

`CutlassFP8ScaledMMLinearKernel.is_supported` responde "sim" em qualquer CUDA e
quebra em runtime no sm_86. Sem a env, o engine não sobe.

## Subir

```
MODEL_DIR=/caminho/Qwen3.8-27B-NVFP4-QAD tools/qwen38/serve.sh 2 3001 32768 4
```

`serve.sh [tp] [porta] [max_model_len] [max_num_seqs] [args...]`; env `GPU_UTIL`
(default 0.90), `DRAFT` (diretório do drafter DFlash2), `NUM_SPEC` (default 7),
`DRAFT_TP`. A visão fica ligada por padrão (`--limit-mm-per-prompt
'{"image":4,"video":0}'`).

## Medido (10× RTX 3090, 2026-11-20)

| config | prefill | decode C=1 | C=4 agregado | aceitação |
|---|---|---|---|---|
| TP=1 | 576 tok/s | 17,9 | — (C=2: 33,6) | — |
| TP=2 | 1.083 | 36,8 | 120,9 | — |
| TP=4 | 1.385 | 67,4 | 162,9 | — |
| **TP=2 + DFlash2** | 1.171 | **105,7** | **230,3** | 48,6% |
| TP=4 + DFlash2 | 1.213 | 93,0 | 234,6 | 46,3% |

- TP=1 não hospeda o drafter (21,7 GB da base + 3,9 GB do drafter > 24 GB).
- TP=2 + DFlash2 é o ponto doce: o passo de draft é latência-bound, então o
  all-reduce 4-way do TP=4 custa mais do que a banda extra rende.
- Bench: `tools/qwen38/bench.py <tag> [porta]` — conta tokens pelo
  `usage.completion_tokens`; o modelo emite o texto em `delta.reasoning`, e
  contar chunks subconta o decode em ~3×.

## DFlash2 (`Qwen3.8-27B-DFlash2`)

`DFlash2DraftModel`, 5 camadas, hidden 5120, bloco de 8 (`num_speculative_tokens=7`),
`selector_top_k=16`, `target_layer_ids [5,19,33,47,61]`. Três correções foram
necessárias neste fork:

1. `vllm/config/vllm.py` — o refactor #53183 inverteu a lista de versões que
   forçam o Model Runner V2 e passou a **rejeitar** `dflash2 drafts`. O
   DFlash2 *exige* V2 (o seletor de candidatos só existe no speculator V2).
2. `vllm/v1/worker/gpu/model_runner.py` — o hook de confiança do DSpark lia
   `speculator.draft_token_confidence_probs` sem guarda; só o DSpark tem esse
   atributo, então qualquer outro speculator morria com `AttributeError`.
3. `spec_decode/dflash/speculator.py` — `propose` reusava o
   `BatchExecutionDescriptor` do grafo do **target**. O drafter precisa de
   exatamente `num_reqs * num_query_per_req` linhas, mas o bucket eager
   arredonda para 16 e o `query_start_loc` é clampado em `num_reqs`, então o
   FlashInfer rejeitava (`q.shape[0] (16) != qo_indptr[-1] (8)`). Quando a
   forma não bate, o drafter agora roda eager com a contagem exata.

`VLLM_DF2_DEBUG=1` loga a forma de cada passo de draft.
