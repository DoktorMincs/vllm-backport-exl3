# Aviso sobre os patches do plugin EXL3

`plugin-exl3-local-cumulative.diff` é um diff sobre
[`vcruz305/vllm-exl3`](https://github.com/vcruz305/vllm-exl3), distribuído sob
**GNU Affero General Public License v3.0 (AGPL-3.0-only)**.

Este diff é uma obra derivada daquele projeto e, como tal, permanece licenciado
sob **AGPL-3.0-only**. O texto completo está em
[`LICENSE-AGPL-3.0`](LICENSE-AGPL-3.0) neste diretório (cópia da licença do
plugin), no repositório original e em
<https://www.gnu.org/licenses/agpl-3.0.html>. O diff foi gerado contra o plugin
no commit `053c634`; aplique nesse commit. Esta versão (20/11/2026) cobre o
estado do branch `exl3-ds41-local` que roda o MiniMax-M3.1 e o DeepSeek-V4.1.

In English: `plugin-exl3-local-cumulative.diff` is a derivative work of
`vcruz305/vllm-exl3` and is distributed under **AGPL-3.0-only**. The full license
text is in `LICENSE-AGPL-3.0`. The patch applies at commit `053c634`. The plugin
source itself is not vendored in this repository.

Créditos: plugin original de `vcruz305`; os motores de quantização EXL3 vêm de
[`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3)
(MIT, conforme o `LICENSE` do projeto original).

## O que o diff muda

| mudança | motivo |
|---|---|
| `EXL3_FORCE_MIXEDK` | camadas de MoE de K uniforme passam pelo kernel mixed-K; o kernel de K uniforme erra em K=3 (`ffn_out` com cos 0,32 na L18) — ver `../BUG-exl3-moe-uniform-k3.md` |
| `_exl3_pad_col_window` | em TP>1, o `lm_head` (200.064 linhas) precisa ser fatiado em janelas alinhadas a 128 colunas, senão a Hadamard de saída é cortada no meio do bloco e um rank inteiro decodifica errado |
| `VLLM_EXL3_MINIMAX_M3` | cobertura das lineares non-routed do M3 |
| indexer fp16 | o cache lateral do indexer segue o dtype do qkv |
| `_narrow_tp` devolve view | evita a cópia anônima do trellis em TP>1 que estourava a RAM do host no load |
| `VLLM_EXL3_V41_NATIVE_MOE` | experts roteados do DeepSeek-V4.1 no caminho nativo (ABI 3), com o gate de geometria em `deepseek_v41.py` |
| K fracionário (2,5) | `Exl3Config.layer_bits` aceita 2,5 e as arrays por expert do `exl3_moe_mixedk` passam a trabalhar em meias unidades (`k2 = 2*bits + half`); pack do turboderp 2.68 bpw |
| `VLLM_EXL3_COOP_SLOTS_MAX=512`, `VLLM_EXL3_COOP_ROWS_MAX=88` | remove o cliff `slots = 288 > 256` no C=8 do DeepSeek (M=48, topk=6), bit-exato |
| `VLLM_EXL3_MOE_KERNEL=b12x` | backend MoE alternativo W4A16 trellis (opcional; experimento em `experiments/b12x_moe/`, exige o vendor `b12x` e `VLLM_EXL3_B12X_PATH`) |

Nenhum peso de modelo, e nenhum código de terceiros além do próprio plugin
modificado, é distribuído neste repositório.
