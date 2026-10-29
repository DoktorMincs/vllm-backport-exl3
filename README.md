# vllm-backport-exl3

A fork of [`wtdcode/vllm-backport`](https://github.com/wtdcode/vllm-backport) carrying
the engine work needed to serve **MiniMax-M3.1-Flash-EXL3-EAQ** — an EXL3
trellis-quantized MiniMax-M3.1-Flash — on **10× RTX 3090 (sm_86, 24 GB, PCIe, no
NVLink)**. It branches from `ced6985fd1`, a commit authored in
[`avtc/vllm-backport`](https://github.com/avtc/vllm-backport) (itself a fork of
`wtdcode/vllm-backport`) and present in both.

Two packs boot and serve on this stack (images and video were verified end to end on the
3.5 bpw pack; the 3.06 bpw pack was validated for text, pool size and speed):

| pack | size | layers | bits |
|---|---|---|---|
| `MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw` | 184 GB | 177.4 GiB | 3.55 avg, 6-bit heads, mixed K3/K4 |
| `MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw` | 160 GB | 153.2 GiB | 3.06 avg, 8-bit heads, uniform K3 |

The engine is the upstream backport plus the commits on branch `exl3-mm31-flash`
(derived from `ced6985fd1`). Quantization comes from the
[`vllm-exl3`](https://github.com/vcruz305/vllm-exl3) plugin, which needed local
patches — those are distributed **as diffs only** (`docs/exl3-mm31/patches/`), since the
plugin is AGPL-3.0-only. The engineering record — every defect, measurement and dead end — is
in [`docs/exl3-mm31/`](docs/exl3-mm31/README.md).

This is not a general-purpose serving recipe. It is the record of making a 184 GB model
that was never meant for 24 GB cards run correctly and fast on ten of them, where
"correctly" was the hard part: most of what we hit loaded, booted and answered — with the
wrong number, the wrong text, or zero acceptance.

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

## Measurement traps

- Counting **SSE chunks** instead of tokens inflates or deflates results by 2× depending on
  spec decode and the reasoning parser. Use `usage.completion_tokens`.
- Comparing runs with a different **number of generated tokens** produced a fake +50% once.
- `systemctl is-active` does not prove the server works; only a real generation does.
- `pgrep`/`pkill` of workers: nvidia-smi shows host PIDs, workers appear with comm
  `VLLM::Worker_PP*`, and killing only the EngineCore leaves workers holding VRAM.
- `${VAR:-{json}}` in bash eats the JSON's closing brace — the default needs its own
  assignment.
- With an encoder in the loop, high `--gpu-memory-utilization` is not the pool ceiling:
  activation from the profiling dummy is.
- Any block-transform quantization combined with TP must align the shard boundary to the
  transform block. Split on the tensor and one rank decodes garbage.

## Reproducing

The engine changes are already in the branch, so a plain clone is enough. The cumulative
diff exists so the delta against upstream can be read — and applied to a clean checkout of
`wtdcode/vllm-backport` at `ced6985fd1`:

```bash
git clone https://github.com/DoktorMincs/vllm-backport-exl3 && cd vllm-backport-exl3
git checkout exl3-mm31-flash

# alternatively, apply the delta to a clean upstream checkout
git clone https://github.com/wtdcode/vllm-backport && cd vllm-backport
git checkout ced6985fd1
git apply /path/to/vllm-backport-exl3/docs/exl3-mm31/patches/backport-exl3-mm31-cumulative.diff
```

Plugin and runtime (AGPL-3.0-only — patch, do not vendor). The diff is generated against
`vcruz305/vllm-exl3` at `053c634`; check that commit out first. The scripts read
`EXL3_PORT` (default `~/exl3-port`) as the environment root and expect the plugin at
`$EXL3_ROOT/pkgs/vllm-exl3`:

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

Boot (the flags that matter, TP=2 × PP=5 on 10 GPUs):

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

## Repository layout

```
vllm/models/minimax_m3/nvidia/dflash.py          DSpark drafter (new)
vllm/reasoning/minimax_m3_reasoning_parser.py    <mm:think> parser
vllm/models/minimax_m3/nvidia/model.py           vision tower built without EXL3
vllm/models/minimax_m3/common/mm_preprocess.py   dummy video bounded by the segment cap
vllm/models/minimax_m3/common/sparse_attention.py
docs/exl3-mm31/README.md                         the defect catalogue (start here)
docs/exl3-mm31/STATUS.md                         chronological record with every number
docs/exl3-mm31/FINDINGS-20261027.md              component-by-component investigation
docs/exl3-mm31/BUG-exl3-moe-uniform-k3.md        issue draft for the plugin
docs/exl3-mm31/RUNBOOK.md                        how the environment is built and run
docs/exl3-mm31/patches/                          engine + plugin diffs, AGPL notice + text
docs/exl3-mm31/scripts/                          sanitized measurement/runbook tooling
README-upstream.md                               the original vllm-backport README
```

## Credits

Almost none of what runs here is ours. In rough order of how much this repo owes them:

| project / people | what it provides | license |
|---|---|---|
| [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) — MiniMax | the model itself (official MXFP8 checkpoint), the DSpark speculative draft and the vision tower, copied verbatim into the packs | follow MiniMax's model terms |
| **Mia-AiLab** — the EXL3 packs | `MiniMax-M3.1-Flash-EXL3-EAQ-v1` at 3.55 and 3.06 bpw: the trellis quantization of every body linear, the 8/6-bit `lm_head`, and the packing this engine is written against | derivative of the base model; same terms |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — turboderp | the EXL3 format (`trellis`/`suh`/`svh`, `mul1` codebook, `out_scales`), the quantizer and the dequant/GEMM kernels everything decodes through | MIT |
| [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) — vcruz305 | the vLLM EXL3 plugin: weight loading, MoE and linear methods, TP geometry. Our work there exists only as the diffs in `patches/` | **AGPL-3.0-only** |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) and its contributors | the engine everything else is a fork of — including the MiniMax-M3 model itself: block-sparse attention, the top-k indexer, the fused MoE and the multimodal preprocessor that this work builds on | Apache-2.0 |
| [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) — lazymio | the backport this tree is a fork of: sm_86 support and the model bring-ups (DeepSeek-V4, Qwen3.8, GLM-5.3, MiMo) | Apache-2.0 |
| [avtc/vllm-backport](https://github.com/avtc/vllm-backport) — avtc | author of the base commit `ced6985fd1`; their fork of `wtdcode/vllm-backport` is where that commit was written | Apache-2.0 |

And to the usual stack underneath all of it: PyTorch, Triton, FlashInfer, NCCL and CUDA.

**What is ours** is narrow and specific: the six fixes in `docs/exl3-mm31/` (MoE K=3 kernel
routing, the 128-aligned `lm_head` TP window, the view-returning TP shard loader, the DSpark
drafter port with its MXFP8 dequant, the vision-tower and dummy-video corrections, the
`<mm:think>` streaming parser), the measurements, and the runbook. If you reuse any of it,
credit the projects above first.

## Scope and licensing

- The engine changes here inherit **Apache-2.0** from vLLM, `wtdcode/vllm-backport` and
  `avtc/vllm-backport`.
- The diffs under `docs/exl3-mm31/patches/` that touch `vllm-exl3` are derivative works of
  an **AGPL-3.0-only** project and remain under that license; see
  [`patches/NOTICE-plugin.md`](docs/exl3-mm31/patches/NOTICE-plugin.md) and the full text in
  [`patches/LICENSE-AGPL-3.0`](docs/exl3-mm31/patches/LICENSE-AGPL-3.0). The plugin source
  itself is **not** vendored here — patch it in place.
- The tooling under `docs/exl3-mm31/scripts/` is part of this repository and follows the
  Apache-2.0 license of the tree.
- The base model is distributed under the **MiniMax Community License** (commercial terms
  and use restrictions apply); the EXL3 packs are derivatives of it and carry the same
  terms. The packs belong to their authors; **no model weights are distributed here**.
- Hardware-specific by design: this was tuned on 10× RTX 3090 with PCIe only. Numbers on
  other cards will differ, and the sm_86 `exllamav3` rebuild is required.
