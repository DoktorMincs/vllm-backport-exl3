# vllm-backport-exl3

A fork of [`wtdcode/vllm-backport`](https://github.com/wtdcode/vllm-backport) carrying
the engine work needed to serve three model families on **10× RTX 3090 (sm_86, 24 GB,
PCIe, no NVLink)**:

| model | on disk | geometry | headline, all measured on this machine |
|---|---|---|---|
| **MiniMax-M3.1-Flash-EXL3-EAQ** | 184 GB (3.5 bpw) / 160 GB (3.06 bpw) | TP=2 × PP=5 | 45.6 tok/s single stream, 131k ctx, 84.1 tok/s aggregate |
| **DeepSeek-V4.1-Flash-1017-EXL3** | Mia-AiLab v1 EXL3 pack | TP=2 × PP=5 | 512k ctx (574,853-token pool), 1,434 tok/s prefill, 74.1 / 248.0 tok/s at C=1 / C=8 |
| **Qwen3.8-27B-NVFP4-QAD** | 19.9 GB | TP=1/2/4 | 17.9 / 36.8 / 67.4 tok/s; **105.7** with the DFlash2 drafter at TP=2 |

The integration branch is **`master`**; the per-model history is also kept in
`exl3-mm31-flash` (MiniMax) and `vllm-backport-exl3-ds41` (DeepSeek + Qwen3.8), which
`master` merges. The fork descends from `ced6985fd1`, a commit authored in
[`avtc/vllm-backport`](https://github.com/avtc/vllm-backport) (itself a fork of
`wtdcode/vllm-backport`) and present in both.

This is not a general-purpose serving recipe. It is the record of making models that were
never meant for 24 GB cards run correctly and fast on ten of them, where "correctly" was
the hard part: most of what we hit loaded, booted and answered — with the wrong number,
the wrong text, or zero acceptance.

---

# MiniMax-M3.1-Flash-EXL3

Two packs boot and serve on this stack (images and video were verified end to end on the
3.5 bpw pack; the 3.06 bpw pack was validated for text, pool size and speed):

| pack | size | layers | bits |
|---|---|---|---|
| `MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw` | 184 GB | 177.4 GiB | 3.55 avg, 6-bit heads, mixed K3/K4 |
| `MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw` | 160 GB | 153.2 GiB | 3.06 avg, 8-bit heads, uniform K3 |

Quantization comes from the [`vllm-exl3`](https://github.com/vcruz305/vllm-exl3) plugin,
which needed local patches — those are distributed **as diffs only**
(`docs/exl3-mm31/patches/`), since the plugin is AGPL-3.0-only. The engineering record —
every defect, measurement and dead end — is in
[`docs/exl3-mm31/`](docs/exl3-mm31/README.md).

## Results

TP=2 × PP=5 across the 10 GPUs, fp16 weights, `--kv-cache-dtype fp8_e5m2`,
`--max-num-batched-tokens 512`, CUDA graphs (`cudagraph_mode=FULL_AND_PIECEWISE`).

| regime | measured |
|---|---|
| decode, single stream | **45.6 tok/s** (128 tokens) · 45.2 (400 tokens) |
| decode, TP=1 × PP=10 | 21.0 tok/s |
| decode, eager (no CUDA graphs) | 3.5 tok/s |
| prefill at 2k / 8k / 32k | **1,491 / 2,126 / 2,357 tok/s** |
| aggregate, 16 concurrent | **84.1 tok/s** |
| decode with the DSpark drafter (N=2) | 53.9 tok/s (+20%, single stream only) |
| decode with vision enabled | 45.5 tok/s (no measurable cost) |

Prefill is bound by chunk latency and by power, both counter-intuitively.
`--max-num-batched-tokens 512` beat both the 2048 default and 1024, while 8192 was 2.6×
worse at 8k: with PP the chunk is one pipeline stage, so a bigger chunk lengthens every
stage and the bubble. Raising the power cap from 200 to 230 W bought another 21% at 32k
prefill (1,895 → 2,285 tok/s) at a 2.06 kW total draw; decode is insensitive to it.

## Context and KV cache

The 3.06 bpw pack is the interesting one for context: 17 GiB less of weights turns into
KV cache almost one for one.

| pack | KV dtype | util | max-num-seqs | KV pool | context |
|---|---|---|---|---|---|
| 3.5 bpw, no vision | fp16 | 0.92 | 256 | 126,720 | 65,536 |
| 3.5 bpw, no vision | fp8_e5m2 | 0.92 | 256 | 211,328 | 131,072 |
| 3.5 bpw, no vision | fp8_e5m2 | 0.95 | 8 | 281,856 | 262,144 |
| 3.5 bpw, vision + video | fp8_e5m2 | 0.96 | 8 | 223,488 | 131,072 |
| 3.5 bpw, + DSpark drafter | fp16 | 0.95 | 8 | 51,526 | ~51k |
| **3.06 bpw, vision + video** | **fp8_e5m2** | **0.96** | **8** | **570,880** | **131,072 (4.36×)** |

Two counter-intuitive facts came out of this table. First, at high `--gpu-memory-utilization`
what runs out is **activation**, not KV: 0.95 with the default `--max-num-seqs 256` OOMs
during profiling, while 0.95 with `--max-num-seqs 8` boots and gains 33% more pool. Second,
any vision encoder is replicated on **every** pipeline stage (+0.81 GiB/GPU here), so
"262k plus vision" does not fit no matter how the layers are partitioned.

The smaller pack costs nothing in speed — 44.9 tok/s vs 45.4 — because single-stream decode
at this geometry is bound by per-layer launch/all-reduce overhead, not by weight bandwidth.

## What had to be built

The model is a sparse-attention MoE with block-sparse top-k selection, EXL3 trellis weights
in attention/dense/MoE/`lm_head`, a bf16 vision tower, and an optional DSpark speculator.
Serving it needed:

- **EXL3 coverage** for every non-routed projection of the M3 (`VLLM_EXL3_MINIMAX_M3=1`) —
  without it the loader silently falls back to a routed-experts-only scope and attention
  weights are read as bf16 when they are trellis.
- **TP-aware EXL3 sharding**: the plugin's padded geometry, 128-aligned windows for
  `lm_head`, and a loader that fills the padding without materializing an anonymous copy of
  the trellis.
- **A correct MoE kernel choice**: the fused uniform-K kernel is wrong at K=3, so all
  uniform-K layers are routed through the mixed-K kernel (`EXL3_FORCE_MIXEDK=1`). Both packs
  contain K=3 layers, and the 3.06 pack is **uniformly K=3** — the flag is mandatory, not a
  tuning knob.
- **The DSpark drafter** (`vllm/models/minimax_m3/nvidia/dflash.py`): MXFP8 dequant, fused
  context-KV buffers, Gemma norms, speculator wiring, aux hidden states across PP.
- **Vision**: tower built with `quant_config=None`, modality limits restored (the pack's
  `quantization_config.json` has no vision key and `--language-model-only` zeroes every
  modality limit), and profiling dummies bounded by `vision_segment_max_frames`.
- **A reasoning parser** (`vllm/reasoning/minimax_m3_reasoning_parser.py`) for the
  `<mm:think>` delimiters.

`exllamav3` also has to be rebuilt for sm_86 with `ACT_SILU_OAI` (act 3) in the non-coop
path; without it the MoE returns SiLU instead of the intended OAI activation and the text
degrades into repetition.

## The six silent defects

Each of these booted and answered. None of them logged an error.

1. **MoE uniform-K kernel wrong at K=3.** `ffn_out` of the first uniform-K3 layer scored
   cos 0.32 against the oracle while attention, `ffn_in` and routing matched to 1e-6;
   routing margins were 1e-2, so it was not a tie. Symptom: `etim…` repetition. Fix:
   `EXL3_FORCE_MIXEDK=1`. Issue draft in
   [`BUG-exl3-moe-uniform-k3.md`](docs/exl3-mm31/BUG-exl3-moe-uniform-k3.md).
2. **`lm_head` in TP>1 cut a Hadamard block in half.** 200,064 rows / 2 = 100,032, and
   EXL3 applies its output Hadamard in blocks of 128 columns: rank 1 decoded against the
   wrong transform and produced a single spurious logit of 63.9 on token id 99968. Fix:
   each rank loads the whole 128-aligned window and drops the `lead` before concatenating.
3. **Host RAM OOM during the load in TP>1.** `.contiguous()` on the sharded trellis
   materialized an anonymous copy of every expert (≈150 GB over the load) and the
   OOM-killer took the workers without a traceback. Fix: return the view; watch
   `AnonPages`, not `free`.
4. **Drafter with 0% acceptance while everything looked healthy.** Three independent
   silent faults: the MXFP8 `weight_scale_inv` suffix with a stray dot matched no tensor,
   so raw `float8_e4m3` values (|w| up to 448) went into fp16 parameters; the drafter
   inherited the sparse attention backend although its 5 layers are dense; and the probe
   counted SSE chunks instead of tokens.
5. **Vision in three layers of failure.** `--language-model-only` zeroed the limits of every
   modality (`At most 0 image(s) may be provided in one prompt`); the EXL3 plugin claimed
   the bf16 tower and read it as trellis; and the profiling dummy video was sized from
   `max_model_len` (500 frames at 262k), blowing up inside the tower's first `layer_norm`.
6. **Reasoning leaked into `content` while streaming.** `is_reasoning_end(prompt)` scanned
   the whole prompt for the marker pair, but the chat template's own instruction block
   contains the literal `<mm:think></mm:think>`. Fix: only the end of the prompt counts.

---

# DeepSeek-V4.1-Flash-1017-EXL3

The port of the same EXL3 pack the Project Maya engine serves: **TP=2 × PP=5**, layer
partition `8,8,10,8,6`, `--dtype bfloat16`, `--kv-cache-dtype fp8_ds_mla`, the DSpark
drafter (`mtp.*`, N=5), Engram served from disk, vision on. The full record is in
[`docs/exl3-ds41/RUNBOOK.md`](docs/exl3-ds41/RUNBOOK.md).

## Results

Validated at `util 0.968`, `VLLM_MAX_MODEL_LEN_CAP=524288`, vision `image:4` + DSpark N=5:

| regime | measured |
|---|---|
| KV pool at 512k | **574,853 tokens** (1.10× concurrency), 49–53 MiB free VRAM floor |
| cold prefill, 514,936 tokens | 359.1 s = **1,434 tok/s**, 100% needle recall, zero OOM |
| decode C=1 | **74.1 tok/s** (70.7–76.6 across prompts, peak 92.5) |
| decode C=8, spec bench | **248.0 tok/s** median (168.4 cold / 253.2 warm) |
| prefix cache, multi-turn ~32k | 99.3% hit: TTFT 25.4 s → **1.68–1.92 s** |
| prefill at 201k, vision on | 1,565–1,683 tok/s (+38–49% over the baseline) |

Sequential prefill scales to the context edge; decode is where this differs from the Maya
engine, which trades prefill for a much shorter decode step (see the comparison table in
the runbook).

## What had to be built

- **Zero-workspace paged FP8 sparse prefill** (`VLLM_DS41_PAGED_PREFILL=1`): streams
  `fp8_ds_mla` (584 B/slot: 448 B FP8 NoPE + 8×`e8m0` scales + 64 B BF16 RoPE) straight
  from the paged MLA and SWA caches through `topk_indices` and `block_table`, replacing the
  `[chunk, M, 512]` BF16 gather workspace — 513.2 MiB/rank freed on PP2 at 512k, 1.68×
  faster prefill attention at 256k, `max_abs_diff = 0.0156`.
- **Pre-locked indexer buffers** (`VLLM_DS41_REUSE_INDEXER_BUFS=1`): the hoisted `k_bf16`
  and `logits` tensors live in `WorkspaceManager`, so ~490 chunked-prefill steps allocate
  **0 B** and the caching allocator does not fragment (120.6 MiB of churn removed) while
  the Tensor Core path stays intact.
- **Hybrid SWA/MLA prefix cache** (`VLLM_FIX_HYBRID_SWA_PREFIX=1`): dropping one aligned
  full-attention block backs a prefix hit up by `max_block_size` tokens, so SWA has to
  retain that many extra trailing tokens under `eagle_block_drop`.
- **SM86 prefill and indexer**: `BLOCK_K=16` sparse prefill (2 CTAs/SM instead of 1), SM86
  indexer KV grouping without materializing `k_bf16` over the whole context, and TP=2
  indexer query sharding for chunks ≥ 513 tokens.
- **PP host and collective overhead**: no drain sleep on idle stages, no empty all-gathers,
  decode `direct_out`. Host-staged PP transport was measured and **lost** (46.8 vs 60.6
  tok/s) — the bubble is not NCCL.
- **EXL3 MoE coop limits** raised to 512 slots / 88 rows, which removes the `slots = 288 >
  256` fallback cliff at C=8 (bit-exact against the reference).
- **Engram disk gather**: persistent C `pread` pool, open-addressing deduplication and a
  131,072-row striped cache — 3.74–4.90× faster at T=48, bit-exact.
- **512k specifics**: the vision tower is instantiated only on PP0; the `deepseek_v41`
  tool-call and reasoning parsers plus OpenAI `reasoning_effort` mapping; and
  `VLLM_DS41_STRICT_DSML=1`, which turns an unparsable tool-call block into `content`
  instead of a phantom tool call (`empty_args` 2→0, `wrong_tool` 9→0 in the parser audit).

---

# Qwen3.8-27B-NVFP4 on Ampere

NVFP4 (E2M1 weights with FP8-E4M3 scales per 16 values) has no native kernel on sm_86.
It still serves here, **weight-only**: MLP and `lm_head` dequantize through Marlin
**W4A16**, the FP8 attention/GDN projections through **MarlinFP8 W8A16**, and the vision
tower stays BF16. Runbook and tooling: [`docs/qwen38-nvfp4/RUNBOOK.md`](docs/qwen38-nvfp4/RUNBOOK.md),
[`tools/qwen38/`](tools/qwen38).

```bash
export VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel
tools/qwen38/serve.sh 2 3001 32768 4                       # TP=2, 32k
DRAFT=/path/Qwen3.8-27B-DFlash2 tools/qwen38/serve.sh 2 3001 32768 4   # + DFlash2
```

The env var is not optional: `CutlassFP8ScaledMMLinearKernel.is_supported` answers yes on
any CUDA arch and then fails at runtime on sm_86.

## Results

| config | prefill | decode C=1 | C=4 aggregate | draft acceptance |
|---|---|---|---|---|
| TP=1 | 576 tok/s | 17.9 | — (C=2: 33.6) | — |
| TP=2 | 1,083 | 36.8 | 120.9 | — |
| TP=4 | 1,385 | 67.4 | 162.9 | — |
| **TP=2 + DFlash2** | 1,171 | **105.7** | **230.3** | 48.6% |
| TP=4 + DFlash2 | 1,213 | 93.0 | 234.6 | 46.3% |

TP=2 + DFlash2 is the sweet spot, and it *beats* TP=4 + DFlash2 single-stream: a draft step
is latency-bound, so the 4-way all-reduce costs more than the extra bandwidth returns. At
C=4 the two are level — four sequences already saturate the GPU and speculation has less
room. TP=1 cannot host the drafter at all (21.7 GB target + 3.9 GB draft > 24 GB).

## The three fixes DFlash2 needed

`z-lab/Qwen3.8-27B-DFlash2` is a 5-layer block-diffusion drafter (block 8, one anchor token
plus 7 drafts, 16 candidates per step selected by a Triton walk). It could not run in this
fork until three defects were fixed:

1. **The V2-forcing list was inverted.** The #53183 refactor moved the versions that force
   Model Runner V2 and left `dflash2 drafts` on the *rejection* list, so the config refused
   the drafter. DFlash2 requires V2 — the candidate selector exists only in the V2
   speculator.
2. **The DSpark confidence hook assumed its own speculator.** `model_runner.py` read
   `speculator.draft_token_confidence_probs` unguarded; only DSpark defines it, so every
   other speculator died with `AttributeError`.
3. **The drafter inherited the target's batch descriptor.** `propose()` ran the draft with
   the target's CUDA-graph dispatch, but the draft needs exactly
   `num_reqs × num_query_per_req` rows: the eager dispatch rounds to the 16-token bucket
   while the draft's `query_start_loc` clamps at `num_reqs`, and FlashInfer rejects the
   mismatch (`q.shape[0] (16) != qo_indptr[-1] (8)`). The draft now falls back to eager with
   the exact token count when the shape does not match.

`VLLM_DF2_DEBUG=1` logs each draft step's shape.

---

# Measurement traps

- Counting **SSE chunks** instead of tokens inflates or deflates results by 2× depending on
  spec decode and the reasoning parser. Use `usage.completion_tokens`. Qwen3.8 streams its
  thinking in `delta.reasoning`, and each chunk carries ~3.4 tokens.
- Comparing runs with a different **number of generated tokens** produced a fake +50% once.
- `systemctl is-active` does not prove the server works; only a real generation does.
- `pgrep`/`pkill` of workers: nvidia-smi shows host PIDs, workers appear with comm
  `VLLM::Worker_PP*` (and `VLLM::Worker` for TP workers), and killing only the EngineCore
  leaves workers holding VRAM. Find them with `lsof /dev/nvidia*`.
- `${VAR:-{json}}` in bash eats the JSON's closing brace — the default needs its own
  assignment.
- With an encoder in the loop, high `--gpu-memory-utilization` is not the pool ceiling:
  activation from the profiling dummy is.
- Any block-transform quantization combined with TP must align the shard boundary to the
  transform block. Split on the tensor and one rank decodes garbage.
- An arch gate that answers "supported" on every CUDA version is not a gate. Both the
  CUTLASS FP8 kernel here and the NVFP4 path only fail once they execute.
- A speculator whose attention metadata clamps at `num_reqs` cannot be run with a dispatch
  that pads to a graph bucket. Either pad both or run neither.

# Repository layout

```
vllm/models/minimax_m3/                          MiniMax M3 model, DSpark drafter, vision
vllm/reasoning/minimax_m3_reasoning_parser.py    <mm:think> parser
vllm/models/deepseek_v4_1/                       DeepSeek V4.1, paged FP8 sparse prefill
vllm/tokenizers/deepseek_v41.py                  reasoning_effort mapping
vllm/v1/worker/gpu/spec_decode/dflash{,2}/       block-diffusion drafter (+ the padding fix)
tools/qwen38/serve.sh, bench.py                  NVFP4 launcher and throughput bench
docs/exl3-mm31/                                  MiniMax defect catalogue, patches, scripts
docs/exl3-ds41/RUNBOOK.md                        DeepSeek port: every wave, every number
docs/qwen38-nvfp4/RUNBOOK.md                     NVFP4 on SM86 + DFlash2
README-upstream.md                               the original vllm-backport README
```

# Reproducing

The engine changes are already in `master`, so a plain clone is enough:

```bash
git clone https://github.com/DoktorMincs/vllm-backport-exl3 && cd vllm-backport-exl3
```

The cumulative diff exists so the delta against upstream can be read — and applied to a
clean checkout of `wtdcode/vllm-backport` at `ced6985fd1`:

```bash
git clone https://github.com/wtdcode/vllm-backport && cd vllm-backport
git checkout ced6985fd1
git apply /path/to/vllm-backport-exl3/docs/exl3-mm31/patches/backport-exl3-mm31-cumulative.diff
```

Plugin and runtime for the EXL3 models (AGPL-3.0-only — patch, do not vendor). The diff is
generated against `vcruz305/vllm-exl3` at `053c634`; check that commit out first. The
scripts read `EXL3_PORT` (default `~/exl3-port`) as the environment root and expect the
plugin at `$EXL3_ROOT/pkgs/vllm-exl3`:

```bash
EXL3_ROOT="${EXL3_PORT:-$HOME/exl3-port}"
git clone https://github.com/vcruz305/vllm-exl3 "$EXL3_ROOT/pkgs/vllm-exl3"
cd "$EXL3_ROOT/pkgs/vllm-exl3"
git checkout 053c634
git apply /path/to/vllm-backport-exl3/docs/exl3-mm31/patches/plugin-exl3-local-cumulative.diff
# rebuild exllamav3 (vcruz305 fork, 329e051 or later, which provides
# exl3_moe_mixedk) for sm_86 with ACT_SILU_OAI (act 3) in the non-coop path
```

The measurement and runbook tooling is in `docs/exl3-mm31/scripts/` (sanitized; see its
`README.md` for the environment variables it expects).

Boot, MiniMax (TP=2 × PP=5 on 10 GPUs):

```bash
# 3.5 bpw: 12,12,14,12,10 | 3.06 bpw: 13,12,12,12,11 (max 16.16 GiB/GPU)
VLLM_PP_LAYER_PARTITION=12,12,14,12,10 \
EXL3_FORCE_MIXEDK=1 VLLM_EXL3_MINIMAX_M3=1 \
VLLM_EXL3_MOE_TP_ALIGN=128 EXL3_ALLOW_TP_INDEXER=1 \
vllm serve <pack> \
  --quantization exl3 --dtype float16 \
  --tensor-parallel-size 2 --pipeline-parallel-size 5 \
  --max-model-len 131072 --gpu-memory-utilization 0.96 \
  --max-num-seqs 8 --max-num-batched-tokens 512 \
  --kv-cache-dtype fp8_e5m2 \
  --reasoning-parser minimax_m3 \
  --limit-mm-per-prompt '{"image": 2, "video": 1}'
```

Neither small value is arbitrary. `--max-num-seqs 8` is what makes util 0.96 boot at all —
at the default 256 the profiling activation reservation OOMs before any KV exists — and
`512` is the measured prefill optimum on these cards.

`docs/exl3-mm31/scripts/serve_best.sh` boots the same geometry with the more conservative
0.92/256 setting and vision enabled by default. The 211,328-token figure in its header is
the text-only row above (`LANGUAGE_MODEL_ONLY=1`); with the vision tower the pool is
smaller. The 0.96/8 recipe above is the measured vision + video row (223,488 tokens).

Boot, DeepSeek V4.1 (same 10 GPUs, `docs/exl3-ds41/RUNBOOK.md` for the full flag set):

```bash
VLLM_PP_LAYER_PARTITION=8,8,10,8,6 \
EXL3_FORCE_MIXEDK=1 VLLM_EXL3_V41_NATIVE_MOE=1 VLLM_EXL3_MOE_TP_ALIGN=128 \
VLLM_MAX_MODEL_LEN_CAP=524288 \
vllm serve <ds4.1-pack> \
  --quantization exl3 --dtype bfloat16 \
  --tensor-parallel-size 2 --pipeline-parallel-size 5 \
  --max-model-len 524288 --gpu-memory-utilization 0.968 \
  --kv-cache-dtype fp8_ds_mla \
  --max-num-seqs 8 --max-num-batched-tokens 512 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5}' \
  --engram-config '{"storage": "disk", "disk_threads": 128}' \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --limit-mm-per-prompt '{"image": 4}'
```

Boot, Qwen3.8-27B-NVFP4 — one GPU up to four, no plugin needed:

```bash
VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel \
tools/qwen38/serve.sh 2 3001 32768 4
```

# Credits

Almost none of what runs here is ours. In rough order of how much this repo owes them:

| project / people | what it provides | license |
|---|---|---|
| [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) — MiniMax | the model itself (official MXFP8 checkpoint), the DSpark speculative draft and the vision tower, copied verbatim into the packs | follow MiniMax's model terms |
| **Mia-AiLab** — the EXL3 packs | `MiniMax-M3.1-Flash-EXL3-EAQ-v1` at 3.55 and 3.06 bpw, and the DeepSeek-V4.1 EXL3 pack: the trellis quantization of every body linear, the `lm_head`, and the packing this engine is written against | derivative of the base model; same terms |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — turboderp | the EXL3 format (`trellis`/`suh`/`svh`, `mul1` codebook, `out_scales`), the quantizer and the dequant/GEMM kernels everything decodes through | MIT |
| [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) — vcruz305 | the vLLM EXL3 plugin: weight loading, MoE and linear methods, TP geometry. Our work there exists only as the diffs in `patches/` | **AGPL-3.0-only** |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) and its contributors | the engine everything else is a fork of — including the MiniMax-M3, DeepSeek-V4 and Qwen3.8 models themselves: block-sparse attention, the top-k indexer, the fused MoE, NVFP4 quantization and the multimodal preprocessor this work builds on | Apache-2.0 |
| [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) — lazymio | the backport this tree is a fork of: sm_86 support and the model bring-ups (DeepSeek-V4, Qwen3.8, GLM-5.3, MiMo) | Apache-2.0 |
| [avtc/vllm-backport](https://github.com/avtc/vllm-backport) — avtc | author of the base commit `ced6985fd1`; their fork of `wtdcode/vllm-backport` is where that commit was written | Apache-2.0 |
| [nvidia/Qwen3.8-27B-NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD) — NVIDIA | the NVFP4 checkpoint and its QAD recipe | NVIDIA Open Model License |
| [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) — z-lab | the DFlash2 block-diffusion drafter | Apache-2.0 |
| [deepseek-ai](https://huggingface.co/deepseek-ai) | DeepSeek-V4.1-Flash-1017 and its MTP/DSpark draft heads | follow DeepSeek's model terms |

And to the usual stack underneath all of it: PyTorch, Triton, FlashInfer, NCCL and CUDA.

**What is ours** is narrow and specific: the six MiniMax fixes in `docs/exl3-mm31/`, the
DeepSeek port (zero-workspace paged FP8 prefill, pre-locked indexer buffers, hybrid SWA
prefix cache, SM86 indexer/sharding, Engram gather and coop-slot work, the 512k validation),
the NVFP4 weight-only path with the DFlash2 drafter fixes, the measurements, and the
runbooks. If you reuse any of it, credit the projects above first.

# Scope and licensing

- The engine changes here inherit **Apache-2.0** from vLLM, `wtdcode/vllm-backport` and
  `avtc/vllm-backport`.
- The diffs under `docs/exl3-mm31/patches/` that touch `vllm-exl3` are derivative works of
  an **AGPL-3.0-only** project and remain under that license; see
  [`patches/NOTICE-plugin.md`](docs/exl3-mm31/patches/NOTICE-plugin.md) and the full text in
  [`patches/LICENSE-AGPL-3.0`](docs/exl3-mm31/patches/LICENSE-AGPL-3.0). The plugin source
  itself is **not** vendored here — patch it in place.
- The tooling under `docs/exl3-mm31/scripts/`, `tools/qwen38/` and the NVFP4 launcher is part
  of this repository and follows the Apache-2.0 license of the tree.
- The MiniMax base model is distributed under the **MiniMax Community License** (commercial
  terms and use restrictions apply); the EXL3 packs are derivatives of it and carry the same
  terms. `Qwen3.8-27B-NVFP4-QAD` is under the **NVIDIA Open Model License** and the DFlash2
  drafter under **Apache-2.0**. The packs belong to their authors; **no model weights are
  distributed here**.
- Hardware-specific by design: this was tuned on 10× RTX 3090 with PCIe only. Numbers on
  other cards will differ, and the sm_86 `exllamav3` rebuild is required for the EXL3 models.
