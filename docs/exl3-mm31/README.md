# MiniMax-M3.1-Flash EXL3 no vllm-backport — o que aprendemos

Registro de engenharia de rodar `MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw`
(184 GB, 3,55 bpw) no [`wtdcode/vllm-backport`](https://github.com/wtdcode/vllm-backport)
com o plugin [`vcruz305/vllm-exl3`](https://github.com/vcruz305/vllm-exl3),
em **10× RTX 3090 (24 GB, sm_86)**.

O que este diretório documenta é, quase todo, **defeito silencioso**: coisas que
carregam, sobem e respondem — com número errado, texto errado ou aceitação
zero — e que só apareceram porque cada uma foi medida contra um oráculo ou
contra outra configuração.

| arquivo | conteúdo |
|---|---|
| `STATUS.md` | estado cronológico completo, com os números de cada campanha |
| `FINDINGS-20261027.md` | investigação do texto corrompido, componente por componente |
| `BUG-exl3-moe-uniform-k3.md` | rascunho de issue para o autor do plugin (kernel MoE K uniforme) |
| `RUNBOOK.md` | como o ambiente está montado e como operá-lo |
| `patches/` | diffs aplicáveis sobre o upstream limpo |
| `scripts/` | ferramentas de medição/operação, versão sanitizada |

## Números

TP=2 × PP=5, CUDA graphs, KV `fp8_e5m2`, `--max-num-batched-tokens 512`.

| regime | valor |
|---|---|
| decode single-stream | **45,5 tok/s** (21,0 com TP=1×PP=10; 3,5 em eager) |
| prefill 2k / 8k / 32k | **1.491 / 2.126 / 2.357 tok/s** |
| agregado C=16 | **84,1 tok/s** |
| decode com drafter DSpark (N=2) | **53,9 tok/s** (+20%) |
| decode com o drafter, C=4 (agregado) | 48,5 (−23% vs sem drafter) |

O decode single-stream aparece como 45,3–45,6 tok/s conforme a execução (README
45,6; este documento 45,5; STATUS 45,3) — é a mesma configuração, medida em
momentos diferentes.

Contexto máximo, por configuração:

| configuração | pool de KV | contexto |
|---|---|---|
| sem drafter, KV fp16, util 0,92 | 126.720 | 65.536 |
| sem drafter, KV `fp8_e5m2`, util 0,92 | 211.328 | 131.072 |
| sem drafter, KV `fp8_e5m2`, util 0,95, `max-num-seqs 8` | **281.856** | 262.144 |
| com drafter, KV fp16, util 0,95 | 51.526 | ~51k |
| com visão+vídeo, util 0,94 | 176.512 | 131.072 |
| com visão+vídeo, util 0,96 | **223.488** | 131.072 (1,71×) |

## Catálogo de defeitos

Cada item aqui custou pelo menos um dia e **não** aparecia em log de erro.

### 1. Kernel MoE de K uniforme errado em K=3 (plugin EXL3)

- **Sintoma:** texto degradado (`etimetim…`), prefill devolvendo `'etim'` em vez
  de `<mm:think>`.
- **Localização:** `ffn_out` da **primeira** camada de K=3 uniforme (L18) com
  cos 0,32 contra o oráculo, enquanto atenção, `ffn_in` e roteamento batiam.
  As camadas mixed-K (L17) e K4 (L3–16, L46–59) estavam certas.
- **Causa:** o kernel MoE fundido de K uniforme em K=3.
- **Correção:** `EXL3_FORCE_MIXEDK=1` — manda as camadas de K uniforme pelo
  kernel mixed-K. Issue em `BUG-exl3-moe-uniform-k3.md`.

### 2. `lm_head` em TP=2: Hadamard cortada no meio do bloco

- **Sintoma:** um logit espúrio de 63,9 no id 99968 (`'etim'`) dominando todo
  prompt; rank 1 inteiro errado (cos −0,50).
- **Causa:** o `lm_head` tem 200.064 linhas e o EXL3 aplica Hadamard de **saída
  em blocos de 128 colunas**. A fatia do TP=2 (100.032 = 781,5 × 128) cai no meio
  de um bloco: rank 1 decodifica contra a transformada errada.
- **Correção:** cada rank carrega a **janela 128-alinhada inteira** (rank 0:
  `[0, 100096)`; rank 1: `[99968, 200064)`, `lead=64`) e o apply descarta o
  `lead` antes de concatenar (`_exl3_pad_col_window`).
- **Lição geral:** qualquer quantização com transformada por bloco + TP precisa
  alinhar o corte ao bloco, não à divisão do tensor.

### 3. OOM de RAM do host no load em TP>1

- **Sintoma:** boot enche 96 GB + swap e o OOM-killer mata os workers; o log
  corta no meio de um expert, **sem traceback**.
- **Causa:** `.contiguous()` no shard em TP>1 materializa uma cópia **anônima**
  do trellis de cada expert, viva até o fim do load (~1,3 GiB × 12 camadas ×
  10 ranks ≈ 150 GB). Em TP=1 a função devolve a view do mmap (page cache,
  reclamável) e nada acontece.
- **Correção:** devolver a view (`EXL3_TP_COPY_SHARD=1` restaura o antigo).
  Pico anônimo depois: 23,4 GB.
- **Diagnóstico:** olhar `AnonPages`, não `free`.

### 4. Drafter DSpark — aceitação 0% com tudo "funcionando"

Três defeitos independentes, todos silenciosos:

1. **Peso MXFP8 nunca desquantizado.** `_SCALE_SUFFIX` sem o ponto fazia
   `name[:-len(sufixo)]` sobrar `...q_proj.` (com ponto) contra a chave
   `...q_proj`: nenhum par casava e os valores `float8_e4m3` (|w| até 448) iam
   **crus** para parâmetros fp16. Sintoma: `Drafted: 154 / Accepted: 0`.
2. **Backend de atenção errado.** O drafter herdava o backend esparso do alvo;
   suas 5 camadas são densas e não têm pesos de indexer.
3. **Medição.** O probe contava **chunks SSE**, não tokens: com spec decode o
   servidor manda ~2 tokens por chunk e ele reportava 18 tok/s em vez de 52.

**Sweep de N** (o ótimo não é o maior N — o custo do passo é dominado pelo
fixo, então quem manda é `L`, tokens por passo):

| N | decode | L | aceitação por posição |
|---|---|---|---|
| 1 | 47,0 | 1,59 | 0,592 |
| **2** | **53,9** | 2,01 | 0,612 / 0,401 |
| 3 | 51,6 | 2,15 | 0,552 / 0,368 / 0,227 |
| 7 | 42,6 | 2,05 | 0,519 / 0,154 / 0,115 / … |

Teto: `max_num_seqs × (N+1) ≤ 16`. Acima disso o `lm_head` EXL3 cai em
`reconstruct` (1,23 GiB/rank) e, dentro do CUDA graph, esse temporário fica no
pool do graph → OOM no startup.

### 5. Vision: três defeitos em série

1. `--language-model-only` zera o limite de **todas** as modalidades
   (`get_limit_per_prompt` devolve 0) → o cliente vê `At most 0 image(s) may be
   provided in one prompt`.
2. **O EXL3 reivindicava as lineares da torre** e lia bf16 como trellis:
   `EXL3 linear shard 0: neither mcg nor mul1 marker is set`. O
   `quantization_config.json` do pack não tem nenhuma chave de visão.
   Correção: `quant_config=None` na construção da torre.
3. **O dummy de vídeo do profiling era dimensionado por `max_model_len`**, não
   pelo limite do encoder. Com 262k pedia até 500 frames (o `vision_config` é
   um **dict** em vLLM, então `getattr` não achava `vision_segment_max_frames=4`)
   e o profiling estourava no primeiro `layer_norm` da torre — 2,75 GiB num
   encoder de 300M. Correção: capar nos 4 frames do segmento.

### 6. Reasoning parser: `<mm:think>` em vez de `think`

O M3 delimita raciocínio com `<mm:think>…</mm:think>`. Ligar
`--reasoning-parser minimax_m3` resolve o caso simples, mas **o streaming
continuava vazando a tag**: `is_reasoning_end(prompt)` varria o prompt inteiro
procurando o par, e o bloco `<thinking_instructions>` do próprio template
escreve `<mm:think></mm:think>` literalmente. O prompt então "parecia" já ter
fechado o raciocínio e tudo ia como `content`. O não-streaming nunca sofreu (só
olha a saída). Correção: conta apenas o que o template **prefila no fim do
prompt**.

## Armadilhas de medição

- **Contar chunks SSE em vez de tokens** infla ou deprime o resultado conforme o
  modo (spec decode, reasoning parser). Usar `usage.completion_tokens`.
- **`N generated` diferente entre comparações** — 128 → 74 tokens inflou +50%
  falso num teste anterior.
- **`systemctl is-active` não prova função.** Só uma geração real fecha um gate.
- **`pgrep`/`pkill` de worker**: nvidia-smi mostra PIDs do host e o `/proc` do
  container tem outros; workers aparecem com comm `VLLM::Worker_PP`, não
  `python`. Matar só o EngineCore deixa workers segurando VRAM.
- **`${VAR:-{json}}` em bash** come o `}` do JSON — o default precisa de
  atribuição separada, senão chega `{"image": 2}}` no argparse.
- **Util alta não é o teto do pool quando há encoder**: quem reserva memória é a
  ativação do profiling (dummy multimodal, `max_num_seqs`), não o KV.

## Como reproduzir

O branch já contém o engine; para usar, basta clonar este repositório:

```bash
git clone https://github.com/DoktorMincs/vllm-backport-exl3 && cd vllm-backport-exl3
git checkout exl3-mm31-flash
```

Para aplicar o delta sobre um upstream limpo (`wtdcode/vllm-backport` em
`ced6985fd1`):

```bash
git clone https://github.com/wtdcode/vllm-backport && cd vllm-backport
git checkout ced6985fd1
git apply /caminho/para/vllm-backport-exl3/docs/exl3-mm31/patches/backport-exl3-mm31-cumulative.diff
```

O plugin EXL3 precisa dos patches correspondentes (AGPL-3.0-only — ver
`patches/NOTICE-plugin.md` e `patches/LICENSE-AGPL-3.0`), aplicados sobre o
commit `053c634`:

```bash
cd vllm-exl3 && git checkout 053c634
git apply .../patches/plugin-exl3-local-cumulative.diff
```

E do `exllamav3` reconstruído para sm_86 com `ACT_SILU_OAI` (act 3) no caminho
não-coop — sem isso o MoE devolve SILU no lugar do OAI e o texto degrada. Use o
fork `vcruz305/exllamav3` em `329e051` ou posterior, que fornece
`exl3_moe_mixedk`.

As ferramentas de medição e operação citadas no RUNBOOK estão em
[`scripts/`](scripts/README.md), na versão sanitizada (caminhos e unidades de
produção vêm do ambiente).

## Licenças

- O código deste fork herda **Apache-2.0** do vLLM/`wtdcode/vllm-backport`.
- Os arquivos em `patches/` que tocam `vllm-exl3` são derivados de um projeto
  **AGPL-3.0-only** e permanecem sob essa licença; ver `patches/NOTICE-plugin.md` e
  o texto completo em `patches/LICENSE-AGPL-3.0`.
- Os scripts em `scripts/` seguem a licença do repositório (Apache-2.0).
- Nenhum peso de modelo é distribuído aqui; o modelo base usa a **MiniMax
  Community License**.
