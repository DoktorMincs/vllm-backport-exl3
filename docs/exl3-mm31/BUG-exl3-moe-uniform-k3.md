# Rascunho de issue: `exl3_moe` (K uniforme) produz saída errada com K=3

Rascunho de issue para o repositório `vcruz305/vllm-exl3`. O plugin é
AGPL-3.0-only; este documento e o diff em `patches/` são publicados sob essa
licença (ver `patches/NOTICE-plugin.md` e `patches/LICENSE-AGPL-3.0`).

## Resumo

Num pack MiniMax-M3.1 EXL3 com camadas MoE de **K uniforme = 3**, o caminho
fundido `fused_moe=exl3_moe` (`apply_exl3_fused_moe`, stamp `"batched"`) devolve
um `ffn_out` que não tem relação com a referência, enquanto o mesmo layer pelo
`apply_exl3_python_loop` (LinearEXL3) e pelo `exl3_moe_mixedk` acerta.

Camadas de K uniforme 4 no mesmo modelo funcionam; a única camada mixed-K
(K = {3,4}) também funciona.

## Ambiente

- GPU: 10× RTX 3090 (sm_86), TP=1 × PP=10, `--enforce_eager`
- vLLM: backport wtdcode v0.13.2.dev4+gced6985fd
- plugin `vcruz305/vllm-exl3` @ 053c634
- exllamav3 `vcruz305/exllamav3` @ 329e051 (com o patch local de ACT_SILU_OAI)
- pack: `Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw`, hidden 6144,
  intermediate 3072, 128 experts, top-4, codebook **mul1**, `swiglu_limit` 7,0,
  `routed_scaling_factor` 2,0, act 3 (SwiGLU-OAI)
- censo de K por camada (do ledger): L3–16 K4, L17 mixed {3,4}, L18–45 **K3**,
  L46–59 K4

## Evidência

1. Prefill de 210 tokens; dump por camada do vLLM; oráculo F32 independente
   (`ref.py`, dequant próprio).
2. Até a L17: cos 1.000000 em todas as posições testadas (10/100/205/209).
3. A partir da L18 (primeira K3): cos cai a 0,987 e segue caindo.
4. Isolando a L18, partindo do estado L17 **do próprio vLLM**:

   | tensor | cos | relerr |
   |---|---|---|
   | `attn_out` | 1.000000 | 6,6e-04 |
   | `ffn_in` | 1.000000 | 4,4e-04 |
   | `ffn_out` | **0.319032** | **1,67e+00** |

   Controle na L17 (mixed-K), mesmo método: `ffn_out` cos 0.999997.
5. Roteamento na L18 (fórmula do oráculo sobre o `ffn_in` do vLLM): margem
   top-4/top-5 de 1,3e-2 a 7,8e-2 → o conjunto de experts é estável, não é
   desempate.
5. A/B:
   - `EXL3_FORCE_MIXEDK=1` (mixedk para todas as camadas uniformes) → o prefill
     volta a `<mm:think>` com logprob −0,015 (oráculo: −0,013);
   - `EXL3_UNIFORM_LOOP=1` (loop Python) → −0,014;
   - estoque (`exl3_moe`) → `'Você'` −2,101.
6. `parity_check.py` (LinearEXL3) nas amostras reais K3 w1/w3/w2 da L20 e K4 w1
   da L8: cos ≥ 0,9999988 → o dequant e o GEMM denso estão corretos.

## Repro mínimo sugerido

Um único layer MoE uniforme K3 do pack, `x` de 210 linhas (batch de prefill),
roteamento top-4 do gate BF16; comparar `apply_exl3_fused_moe` com
`apply_exl3_python_loop` (mesmos `inners`). Esperado: cos ~1; observado: cos
~0,3.

## Suspeitas (ordem)

1. Instância de kernel escolhida em `exl3_moe.cu` para K=3 com codebook mul1
   (`exl3_moe_kernel_k3_n256_cb2` para `m_tile<=16`;
   `exl3_moe_kernel_k3_n128_cb2_m32/m64` para tiles maiores) — o K=4 do mesmo
   caminho acerta.
2. `mul1` codebook em K3 nesse kernel (o dense K3 mul1 funciona).
3. Montagem de `token_sorted`/`expert_count`/`m_tile` para K3.

## Contorno usado neste branch

`EXL3_FORCE_MIXEDK=1` (patch de 8 linhas em `exl3.py`: força `mixed_k=True` em
layers uniformes e desliga `_exl3_coop_compat` nelas). Resultado correto em
prefill e decode.
