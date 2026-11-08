## This was made by GLM-5.3-flash, you may study, use or fork this repo but it must retain the MIT License

# QuantRunner

**A streaming quantization + token-inference engine in pure C# (.NET 8, zero NuGet dependencies) — built to run very large LLMs (up to and beyond 2.8T parameters) on ordinary hardware, faster than AirLLM.**

```
dotnet build QuantRunner.sln -c Release
dotnet run --project src/QuantRunner.Cli -c Release -- selftest
```

---

## Why QuantRunner exists

AirLLM made one thing possible: running huge models layer-by-layer from disk without loading
them into RAM. Its weakness is that every layer pays the **full disk latency synchronously on
the critical path**, materializes fp16 dequantized copies in torch, and allocates a tensor per
weight per layer in Python.

QuantRunner keeps the "never load the whole model" idea and fixes the three costs that hurt
AirLLM most:

| Cost | AirLLM | QuantRunner |
|---|---|---|
| Layer streaming | synchronous — I/O blocks every token | **async ping-pong prefetch pipeline** — layer N+1 streams from disk on a background task while layer N computes |
| Dequantization | fp16 round-trip through torch tensors | **fused dequantize+dot kernels** (Vector256/Vector128) read compressed blocks directly — weights never become floats |
| Allocation churn | torch tensor per weight per layer | **zero allocation forward** — everything is pointer arithmetic over mmap / pinned staging buffers |

Everything runs on the CPU (any AVX2/NEON-capable machine) with graceful scalar fallback, and
the engine is deliberately **hardware-agnostic at the format level**: GGML block formats are
byte-exact, so a llama.cpp GGUF converts to `.qw` with no requantization when formats match.

## Feature list

- **Quantization formats** (GGML byte-compatible): `Q4_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`,
  `Q5_K`, `Q6_K`, plus `F32/F16/BF16`, `I8_BLK`, a native 1.58-bit **ternary TRI** format
  (BitNet-style `{-s, 0, +s}`, 2.25 bits/weight incl. scales), and a **sub-TRI ladder** that
  breaks the ternary bit floor: `B1` (1.06 bpw), `B1X` (1.47 bpw, binary + exact outlier
  exceptions), `T1E`/`T1E_LO` (1.25 / 1.00 bpw, sparsity-controlled ternary + rANS entropy
  coding) — see "Going below TRI" below. Every format is selectable per weight group in
  `quantize` (`--fmt/--qattn/--qffn/--qemb/--qhead`).
- **Requantization** (`requant`) — quantize an existing `.qw` into a smaller `.qw`:
  one streaming dequant→requant pass, byte-exact passthrough for untouched tensors,
  optional per-tensor format picking from an `analyze` report — see "Requantization" below.
- **Any-to-any conversion** (`convert`) — GGUF ↔ `.qw` ↔ HuggingFace safetensors: import
  gguf/HF into `.qw`, export `.qw` back to a llama.cpp-loadable GGUF, or export either to a
  sharded HF directory with `config.json` / `model.safetensors.index.json` / tokenizer files
  ready for fine-tuning — see "Format conversion" below.
- **Fused SIMD kernels** — one pass over compressed bytes computes the dot product; no
  intermediate fp16/fp32 weights. Vector256 with Vector128/scalar fallback (ARM-friendly).
- **`.qw` streaming format** — `QWR1` magic + JSON header + 64-byte-aligned data; all offsets
  are `long` (multi-TB-safe); tensors are read zero-copy through `MemoryMappedFile` with
  `AcquirePointer`.
- **Async prefetch pipeline** — two pinned ping-pong staging buffers; per-layer byte budget;
  automatic per-layer fallback to mmap for huge MoE layers; I/O-stall and overlap telemetry.
- **Paged KV cache** — 256-token pages allocated on demand (RAM grows with the real sequence,
  not max context), fused GQA attention (scores → softmax → weighted V sum) over pages.
- **Architectures** — Llama-family dense (RMSNorm, RoPE, SwiGLU, GQA), Mixtral-style MoE
  (softmax router + top-k experts, packed 3-D GGUF expert tensors split per expert at convert
  time), and the GLM-5.x hybrid family end to end (mHC + KDA + MLA + GLM sigmoid-MoE — see the
  quickstart notes). Pluggable via `ModelConfig` + canonical tensor names.
- **Speculative decoding** (`run --draft`) — a small (or self-requantized) draft model proposes
  tokens, the target verifies; output stays EXACTLY the target's greedy trajectory — see
  "Speculative decoding" below.
- **HF tokenizer** — byte-level BPE (`tokenizer.json`): GPT-2 pre-tokenization regex,
  rank-based merges, added/special tokens, byte-map detokenization, incremental `DecodeToken`
  for streaming output.
- **Sampler** — temperature, top-k, top-p (nucleus), min-p, repeat penalty, seeded
  reproducibility, greedy (temperature 0).
- **Benchmark suite** — per-layer disk streaming MB/s, forward-pass tok/s with prefetch ON vs
  OFF, I/O stall %, working-set size, and an AirLLM comparison table.

## Quickstart

```bash
# 0. sanity check (no model needed)
dotnet run --project src/QuantRunner.Cli -c Release -- selftest

# 1a. convert a GGUF model (byte passthrough when formats match)
dotnet run --project src/QuantRunner.Cli -c Release -- \
    convert --gguf model-Q4_K_M.gguf -o model.qw --qattn q4_k --qffn q4_k

# 1b. or convert a HuggingFace safetensors directory
dotnet run --project src/QuantRunner.Cli -c Release -- \
    convert --hf ./Qwen2-7B-Instruct -o model.qw --qattn q4_k --qffn q4_k --qemb f16

# 1c. quantize a single .safetensors file, or assemble a sharded checkpoint shard by shard
dotnet run --project src/QuantRunner.Cli -c Release -- \
    quantize model-00001-of-00062.safetensors -o model.qw --layers 80 --qattn q4_k --qffn q4_k
for i in $(seq -w 2 62); do     # append the rest — data is appended at EOF, never copied
    dotnet run --project src/QuantRunner.Cli -c Release -- \
        quantize --append model-000$i-of-00062.safetensors -o model.qw
done
# no config.json next to the file? the architecture is inferred from tensor names/shapes
# (heads/kv-heads/ffn/vocab/layers, MoE expert tensors); pin a partial shard with
# --layers/--heads/--kv-heads/--dim/--ffn/--vocab/--experts, or force --config config.json.
# `info model.qw` reports completeness (layers present / tensors missing) while assembling.

# 1c+. what checkpoints are supported?
#   - Llama-family naming: model.layers.*, model.language_model.* (transformers-5 multimodal
#     containers), layers.* (original-Llama), plus GQA attention and Mixtral-style routed MoE
#   - fused projections are split automatically: qkv_proj / W_pack (contiguous) and GLM/BLOOM
#     query_key_value (per-head interleaved), gate_up_proj (dense + per-expert)
#   - config.json nested under text_config (GLM-5.x / GLM-4V / omni-style) is read automatically
#   - first-k-dense MoE models (GLM-4.5/GLM-5/DeepSeek-style) run: layers without a router
#     fall back to the dense FFN path
#   - GLM-5.x hybrids run END TO END (validated against the transformers reference math):
#       * KDA linear attention (Kimi delta rule: short causal conv, bounded forget gate,
#         fine-grained per-channel decay, gated output norm)
#       * MLA multi-latent attention (nope-only NoPE like GLM-5.3-Flash, rope variant for
#         DeepSeek-style checkpoints) — the KV cache stores the COMPRESSED latent
#         (kv_lora_rank [+ rope] per token), attention runs weight-absorbed
#       * mHC hyper-connections (hc_mult residual streams, Sinkhorn-normalized mixing)
#       * GLM sigmoid/noaux_tc MoE: e_score_correction_bias, routed_scaling_factor,
#         shared expert, limited SwiGLU (swiglu_limit)
#       * FP8 (float8_e4m3fn) source weights with 128x128 block scales (weight_scale_inv)
#         are dequantized on import
#     DSA note: the deepseek sparse-attention indexer is not needed for context <= index_topk
#     (2048 for GLM-5.3-Flash) — the engine attends over the full latent cache; beyond that
#     it runs full attention instead of sparse top-k (a superset — outputs diverge from the
#     reference DSA at very long context). MTP / vision-tower weights are skipped.
#   - K-quant formats (q2_k..q6_k) and the sub-TRI ladder (b1/b1x/t1e/t1e_lo) need every
#     mapped tensor's column count (input dim) to be divisible by 256 (the dot kernels walk
#     256-weight blocks along rows); q8_0/q4_0/f16/tri have no such constraint. When a
#     requested format cannot align, the converter falls back to q8_0 (f32 if not even
#     32-aligned) for those tensors and prints a note — the .qw is always runnable. All
#     real GLM-5.x dims comply, so this only fires on tiny test models.
#   - unrecognized tensor names are always reported (count + examples) — nothing is dropped
#     silently; if --append adds 0 tensors you will see exactly which names failed to map

# 1d. no model handy? build a synthetic one (random weights, real formats/dims)
dotnet run --project src/QuantRunner.Cli -c Release -- \
    synthesize demo.qw --dim 512 --layers 4 --heads 8 --kv-heads 4 --ffn 1408 \
                       --vocab 1024 --experts 4 --active 2 --quant q4_k

# 1d+. ...or the GLM-5.x hybrid SHAPE (mHC + KDA:MLA 3:1 + GLM sigmoid-MoE, NoPE MLA) —
#      this runs the real GLM-5 forward path end to end (random weights = machinery check):
dotnet run --project src/QuantRunner.Cli -c Release -- \
    synthesize glm5.qw --glm5 --dim 256 --layers 4 --ffn 512 --vocab 512 \
                       --experts 4 --active 2 --quant q4_k
#      the mini tokenizer is embedded, so this needs no --tokenizer:
dotnet run --project src/QuantRunner.Cli -c Release -- run glm5.qw -p "hi" --temp 0

# 1e. which format buys the most quality per bit on YOUR weights? measure it:
dotnet run --project src/QuantRunner.Cli -c Release -- \
    analyze model-00001-of-00062.safetensors --formats q4_k,q3_k,tri,b1x,t1e,t1e_lo,b1

# 1f. requantize an existing .qw into a smaller one (sub-TRI formats included)
dotnet run --project src/QuantRunner.Cli -c Release -- \
    requant model.qw -o model-small.qw --fmt tri --qemb q4_k --dry-run   # preview the plan
dotnet run --project src/QuantRunner.Cli -c Release -- \
    requant model.qw -o model-small.qw --fmt b1x                          # attention+ffn -> B1X
dotnet run --project src/QuantRunner.Cli -c Release -- \
    requant model.qw -o model-smart.qw --from-report report.json --max-rell2 0.35
#     ^-- per-tensor formats picked from `analyze -o report.json` of the ORIGINAL checkpoint:
#         smallest bpw whose relL2 fits the gate; tensors that cannot afford it stay byte-exact

# 2. inspect
dotnet run --project src/QuantRunner.Cli -c Release -- info model.qw

# 3. chat / generate — the tokenizer comes from INSIDE the model when it can:
#    a .gguf carries its tokenizer in the file, and a .qw imported from gguf keeps it
#    (embedded gguf_kvs) — no --tokenizer needed for either. --tokenizer tokenizer.json
#    is only required for .qw quantized from safetensors (weights carry no vocabulary).
dotnet run --project src/QuantRunner.Cli -c Release -- \
    run model.gguf -p "Why is the sky blue?" -n 256 --temp 0.8 --topk 40 --seed 42
dotnet run --project src/QuantRunner.Cli -c Release -- \
    run model.qw -p "Why is the sky blue?" -n 256 --temp 0.8 --topk 40 --seed 42

# 4. benchmark (streaming MB/s, tok/s, prefetch overlap, RAM)
dotnet run --project src/QuantRunner.Cli -c Release -- bench model.qw --steps 8
```

## Going below TRI: the sub-1.58-bit ladder

TRI stores every ternary symbol at a flat 2 bits (2.25 bpw with the block scale), even when
most weights are zero. Four formats push past that floor by attacking different error terms;
all take 256-weight blocks, all have fused dot kernels, and all round-trip through `.qw`:

```
        format   bpw    block bytes   y_i = ...
        B1       1.062  34            ±alpha          alpha = mean|w| (closed-form L2 optimum)
        B1X      1.469  47            ±alpha + k<=4 exact fp16 exceptions (greedy residual
                                        matching: pull the worst |w|-alpha weight out, refit
                                        alpha on the rest, repeat)
        T1E      1.250  40            {0, ±alpha} with a controllable zero fraction, symbols
                                        rANS-entropy-coded (3-symbol static rANS, <0.1 bit
                                        waste vs Shannon)
        T1E_LO   1.000  32            same, tighter stream budget (more zeros forced)
```

Measured on LLM-shaped tensors (relL2 = ||W-Q||_F/||W||_F; heavy-tail = mean over 5 tensors
shaped product-gaussian x 0.02 + 0.4% outliers x 12 — trained-weight statistics, the regime
BiLLM/OneBit target; gaussian = one pure-gaussian control tensor):

```
format    bpw     relL2 (heavy tail)   relL2 (gaussian)
Q4_K      4.500   0.101                0.078
Q3_K      3.438   0.215                0.181
TRI       2.250   0.571                0.430
B1X       1.469   0.521  <— beats TRI  0.550
T1E       1.250   0.638                0.640
T1E_LO    1.000   0.670                0.726
B1        1.062   0.805                0.602
```

**The headline: on realistic heavy-tailed weights, B1X at 1.469 bpw reconstructs better than
TRI at 2.25 bpw — 35% smaller AND more accurate.** On near-gaussian blocks plain B1/T1E are
the better picks; `analyze` measures per tensor so the choice can be data-driven. Caveats,
honestly: these are weight-only PTQ numbers — no calibration data, no global transforms
(QuIP#-style incoherence processing, GPTVQ codebooks) and no quantization-aware training.
Expect the same ranking the literature reports: Q4_K > sub-TRI formats on end-task quality,
with B1X the strongest ~1-bit point and a clear quality floor below which only QAT or global
transforms help.

Layouts (decoder contract, also in `BlockDequantizer.cs`):

```
B1/B1X : [0..1] f16 alpha, [2..33] 256-bit sign mask (bit set -> +alpha),
         B1X only: [34] exception count k, [35..34+3k) k x (u8 index, f16 exact value)
T1E(_LO): [0..1] u16 zero count, [2..3] u16 plus count, [4..5] f16 alpha,
          [6..] rANS stream (zero-padded to 34/26 bytes; minus = 256 - zeros - plus)
```

## Requantization: shrink an existing .qw

`requant` quantizes an already-quantized model: every selected tensor is dequantized block
by block and re-encoded with a smaller format in one streaming pass (the source is never
materialized, so RAM needs match a first-time quantize). The header config, tensor set and
names are preserved, so the output runs everywhere the original did.

```bash
requant model.qw -o model-small.qw --fmt tri --qemb q4_k          # explicit targets
requant model.qw -o model-small.qw --from-report report.json --max-rell2 0.35
requant model.qw --fmt b1x --dry-run                              # preview the plan only
```

Guarantees: tensors whose target equals their source are copied **byte-exact** (no second
round-trip error); norms/router/KDA-gate-beta/mHC groups are never touched; 1-d tensors and
sub-`--min-elems` tensors are never touched; a target whose 256-weight block does not divide
the row length is skipped with a note instead of being force-fallback to q8_0.

**Pros**
- Smaller file from a model you already have — no original checkpoint needed (the fp16
  safetensors may be long gone, or 62 shards you don't want to re-download).
- One streaming pass at disk speed; the output is immediately runnable (`info`, `run`,
  `bench` all work — arch/config are copied verbatim).
- Per-group control (`--fmt/--qattn/--qffn/--qemb/--qhead`) and per-name
  `--only`/`--exclude`, so you can keep attention sharp and shrink only the FFN.
- **Smart mode**: `analyze` measures per-tensor relL2 on the real weights; `--from-report`
  spends bits exactly where they help, picking the smallest format each tensor can afford
  under a quality gate — heavy-tail FFN tensors often take B1X/T1E while attention stays at
  Q4_K/TRI.
- Compounds with shard assembly: requantize a partially assembled `.qw` after the last
  `--append`, without touching the source shards.

**Cons (read before shipping)**
- **Double quantization error.** Requant is `W → Q1 → Q2`; the errors compound
  (`||W−Q2|| ≤ ||W−Q1|| + ||Q1−Q2||`, and in practice adds in quadrature at best). Requant
  from Q4_K to TRI is strictly worse than quantizing the original fp16 to TRI directly.
- **Irreversible.** Information thrown away in pass 1 (or 2) cannot be recovered — requant
  can never improve quality, only trade it for size.
- **Error already priced in.** A Q4_K block scale carries the Q4 rounding pattern; the
  second quantizer sees warped statistics (e.g. clipped outliers are gone), so measured
  relL2 on requantized weights is optimistic relative to the true error vs fp16.
- Emb/lm_head quality is expensive to lose — output-token distributions degrade faster
  from embedding quantization than from FFN quantization; keep them fp or high-bit.
- The result inherits every loss of both passes; always verify with `logits` + a short
  generation, and prefer `--dry-run` first to see exactly what changes.

Rule of thumb: quantize once, correctly, from the best source you have. Requant is for when
you need the smaller file *now* and the only source is an already-quantized model — and for
measured, per-tensor downgrades where the analyze report shows a tensor can afford less.

## Format conversion: GGUF <-> .qw <-> HuggingFace safetensors

`convert` moves models between the three formats. The source is a path (`.gguf`, `.qw`, or a
HuggingFace directory) and the destination kind is detected from `-o`:

```bash
convert model.gguf -o model.qw            # import: dequant ggml blocks -> quantize into .qw
convert hf-dir/    -o model.qw            # import: safetensors -> .qw (same as quantize)
convert model.qw   -o model.gguf          # export: .qw -> GGUF v3 llama.cpp can load
convert model.gguf -o hf-out/             # export: gguf -> split safetensors (HF-ready dir)
convert model.qw   -o hf-out/             # export: qw -> split safetensors (dequantized)
```

Import options are the usual quantize options (`--fmt/--qattn/--qffn/--qemb/--qhead/--min-elems`).
Export options:

```bash
# safetensors export
--shards N                 split into N shards (default 1 -> model.safetensors)
--layers-per-shard M       group M transformer layers per shard; embeddings first,
                           final norm + lm_head last (the usual HF split shape)
--dtype f16|f32|bf16       dequantized output precision (default f16)

# gguf export
--ggml-type auto|f16|f32|q8_0   auto (default) keeps every layout-identical format
                                byte-exact; f16/f32/q8_0 force a re-encode of all weights
```

What the safetensors export produces — a directory transformers/vLLM/axolotl can load for
fine-tuning or training:

```
hf-out/
  model-00001-of-000NN.safetensors    dequantized weights (__metadata__ format=pt)
  model.safetensors.index.json        weight_map + total_size (when > 1 shard)
  config.json                         architectures/model_type + all dims from the .qw/.gguf
  generation_config.json              bos/eos ids when known
  tokenizer.json                      BPE tokenizers rebuilt from GGUF metadata (when present)
  tokenizer_config.json               special tokens + chat template (when present)
  special_tokens_map.json
```

How faithful is each path?

- **gguf -> qw**: K-quants (Q2_K..Q6_K), Q8_0, F32/F16/BF16 are byte-exact passthrough when
  the target format matches (llama.cpp's own dequant formulas, verified block by block against
  `ggml-quants.c`). Legacy ggml Q4_0/Q4_1/Q5_0/Q5_1 (18/20/22/24-byte blocks, signed delta) are
  dequantized with the reference semantics and re-encoded. IQ*/TQ* types are rejected with a
  clear message.
- **qw -> gguf**: layout-identical tensors copy byte-exact (no quality change); our 20-byte
  Q4_0 is re-encoded as a *true* ggml 18-byte Q4_0; TRI/B1/B1X/T1E/T1E_LO/I8_BLK have no ggml
  counterpart and export as F16 (`--ggml-type q8_0` keeps them smaller). MoE experts are
  reassembled into the 3-D `blk.N.ffn_*_exps` layout. If the .qw was imported from a GGUF, the
  full original metadata (tokenizer, chat template, rope scaling, arch details) was embedded in
  the .qw header and is restored verbatim — the output tokenizes exactly like the source. A .qw
  quantized from safetensors has no tokenizer, so the GGUF is weights-only.
- **qw / gguf -> safetensors**: weights are dequantized to the requested dtype — this restores
  *portability*, not the original checkpoint (a Q4_K .qw stays Q4_K-quality in fp16 clothing).
  2-D tensors keep the HF [out, in] layout, so no transpose surprises. Standard Llama-family and
  MLA (DeepSeek-style) tensor names round-trip; KDA / mHC tensors have no HF counterpart and are
  skipped with a note. BPE (gpt2) tokenizers are rebuilt exactly; sentencepiece ("llama")
  tokenizers cannot be reconstructed from GGUF metadata alone — the config files are written and
  a note tells you to copy `tokenizer.model`/`tokenizer.json` from the original repo.

## Running GGUF models directly

`run` accepts a `.gguf` as well as a `.qw`:

```
dotnet run --project src/QuantRunner.Cli -c Release -- run Qwen3-4B-Q4_K_M.gguf -p "hi" -n 128
```

How it works, and what it does to your disk:

- the engine streams from `.qw` (mmap + our dot kernels), so the first `run` of a GGUF converts
  it **once** into a cached `.qw` beside the file (`model.gguf -> model.gguf.qw`) and every later
  run reuses that cache. The conversion is checked against file mtimes — replace the gguf and the
  cache rebuilds. Delete the cache any time; it is derived data.
- **formats are kept as-is** (`KeepSourceFormats`): layout-identical ggml types (F32/F16/BF16,
  Q8_0, Q2_K..Q6_K) copy **byte-exact** — running a Q4_K gguf runs the very same Q4_K bytes,
  no quality change. Legacy ggml layouts (Q4_0/Q4_1 18/20 B, Q5_0/Q5_1 22/24 B) have no
  byte-identical counterpart and requant at the same depth (Q4_0/Q4_1 -> Q4_0, Q5_0/Q5_1 ->
  Q5_K). IQ*/TQ* types are rejected with a clear message.
- **the tokenizer comes from inside the model.** GGUF always carries `tokenizer.ggml.*`
  metadata; the cache keeps it in the `.qw` header (gguf_kvs) and `run` builds a byte-level BPE
  tokenizer from it at load time. The same applies to a `.qw` that was imported from a GGUF —
  `--tokenizer` is not needed for either. Tokenizer resolution order:
  1. explicit `--tokenizer tokenizer.json`
  2. `tokenizer.json` beside the model file
  3. the model's embedded GGUF metadata
  Only `.qw` models quantized from safetensors have no vocabulary inside (weights carry no
  tokens) — those ask for `--tokenizer` with a message that says exactly that. Sentencepiece
  ("llama") GGUFs cannot have their binary SPM vocab rebuilt from metadata, so they also ask
  for `--tokenizer` instead of silently guessing.
- `synthesize` models now embed the same metadata, so even a synthetic `.qw` runs with zero
  external files (handy for smoke tests).

## Speculative decoding: `run --draft`

Draft-model decoding without changing a single output token:

```bash
# a small dense draft (must share the target's vocab so token ids are comparable)
dotnet run --project src/QuantRunner.Cli -c Release -- \
    synthesize draft.qw --dim 128 --layers 2 --heads 4 --ffn 256 --vocab 1024 --quant q8_0

dotnet run --project src/QuantRunner.Cli -c Release -- \
    run model.qw -p "prompt" --temp 0 --draft draft.qw --draft-tokens 6
```

- **How it works:** each round the draft greedily proposes up to `--draft-tokens` tokens; the
  target verifies them with its own forwards and accepts the longest agreeing prefix. On the
  first mismatch the target's own prediction corrects the draft (the target NEVER consumes a
  wrong token), both engines rewind — O(1) cache truncation plus KDA recurrent-state
  snapshot/restore — and a fully-accepted round yields one bonus token from the last verify
  forward for free.
- **Correctness contract:** the output is bit-for-bit the target's plain greedy trajectory
  (repeat penalty included). Selftest: speculative == plain greedy with an UNRELATED draft
  (worst case: every round misses and it degrades to plain decoding, never to different text).
- **When it wins:** acceptance rate × target-cost is the lever. Draft forwards are ~free
  relative to a streamed 2.8T target layer, so even 30–40% acceptance is a large real-world
  win; with a random synthetic draft acceptance is ~0% and it just costs the draft's time.
- **Self-drafting:** `requant model.qw -o model_b1.qw --fmt b1` gives a same-shape draft for
  ~1 bit/weight. Note on tiny synthetic models the greedy argmax flips easily under quantization
  noise, so measured acceptance is low there — real checkpoints are far smoother.
- Greedy only for now (`--temp` is ignored with `--draft`): sampling-with-distribution
  acceptance is on the roadmap.

## Parameter counts: `info` on any model file

`info` accepts all four source kinds — `.qw`, `.gguf`, a single `.safetensors` shard, or a whole
HuggingFace directory (`config.json` + shards) — and reports the full parameter budget, not just
a raw total:

```
# make a small synthetic MoE and inspect it (reproduce this output yourself)
dotnet run --project src/QuantRunner.Cli -c Release -- synthesize moe.qw \
    --dim 512 --layers 4 --ffn 1536 --vocab 1024 --experts 8 --active 2
dotnet run --project src/QuantRunner.Cli -c Release -- info moe.qw

arch        : moe
layers      : 4   dim: 512   heads: 8   kv-heads: 8
ffn         : 1536 (expert ffn 1536)
vocab       : 1024   rope theta: 10000   ctx: 2048
parameters  : 80,761,344  (80.76M, 127 tensors; 4.66 bits/weight effective)
    embedding                     524,288  (524.3K)
    lm_head (untied)              524,288  (524.3K)
    attention                   4,194,304  (4.19M)   ~1.05M/layer x 4
    experts (routed)           75,497,472  (75.5M)   ~18.87M/layer x 4
    router                         16,384  (16.38K)
    norms                           4,608  (4.61K)
    non-embedding               79,712,768  (79.71M)
    active/token                  ~24.14M  (2 of 8 routed experts + shared + dense)
    by format          Q4_K 79.69M (98.7%), F16 1.05M (1.3%), F32 20.99K (0.0%)
```

What you get, and why it is honest:

- **total parameters** (exact + human-readable) and a **per-component breakdown** — embedding,
  untied head, attention, KDA linear attention, dense FFN, routed experts (summed over *all*
  experts), shared experts, router, norms, other. Names are classified across all three schemes
  QuantRunner reads: canonical `.qw` (`l0.wq`, `l0.e3.w_gate`), GGUF (`blk.0.attn_q`,
  `blk.0.ffn_gate_exps`) and HuggingFace (`model.layers.0.self_attn.q_proj`, `mlp.experts.3.*`).
- **non-embedding parameters** — the number papers usually quote.
- **active/token for MoE** — routed experts are *stored* in full but only `n_active/n_experts`
  of the pool is touched per token, so `active/token = total - routed + routed x active/n + shared`.
  For a 2.8T-class MoE this is the difference between "671B model" and "37B per token".
- **bits/weight effective** (file bytes x 8 / parameters) and the **format mix** (how many
  parameters live in Q4_K vs F16 vs ...).
- a **partial model in progress counts what is present** and says so — during shard assembly
  the totals grow with each `quantize --append` instead of lying.
- a single `.safetensors` shard reports that shard's totals with a note to point `info` at the
  containing directory for whole-model numbers; a directory aggregates every shard and
  cross-checks `model.safetensors.index.json` when present.

## How streaming works (and why 2.8T is feasible)

```
token t:
  x = embed[tok]                              ← row of tok_emb, streamed from mmap
  BeginPrefetch(layer 0)
  for l in 0..L-1:
      wait for prefetch(l)                    ← usually already done (overlap!)
      RMSNorm → wq/wk/wv matvec (fused dequant+dot) → RoPE
      KV page append + GQA attention
      wo matvec + residual
      RMSNorm → dense FFN or top-k experts
      BeginPrefetch(l+1)                      ← disk I/O overlaps the math above
  final_norm → lm_head → logits
```

**Memory per token is independent of parameter count.** What lives in RAM:

| Component | Size |
|---|---|
| One layer of weights (prefetch staging, ×2 buffers) | `bytes(layer)` |
| Activations | a few × hidden_dim floats |
| Paged KV | `2 · kv_dim · 4 B · layers` per token |
| mmap pages (OS-managed, evictable) | bounded by page cache pressure |

A 2.8T-parameter MoE at Q4_K (~4.8 bits/weight) is ~1.68 TB on disk. With 8 experts / 2 active
per layer, **compute per token touches ~420 GB-equivalents of expert data per forward — but
only the 2 active experts' bytes are actually streamed per layer**; a layer of a 2.8T Mixtral
class model (~186 layers × 8 experts) holds ~9 GB of weights, of which ~2.5 GB (attention +
2 experts) is read per token. On a 3.5 GB/s NVMe that is ~0.7 ms of I/O per layer, fully
hidden behind ~2-6 ms of matmul for 2 experts + attention with the fused kernels — i.e. the
prefetch pipeline keeps the disk ahead of the GPU-less compute. Nothing in the format, the
offset math (`long`, 64-byte aligned), the KV pager, or the engine assumes a size limit: the
same binary streams a 125M toy and a 2.8T monster. The only true requirement is
**disk bandwidth ≥ dequantized-matmul appetite**, which is the same law AirLLM obeys —
QuantRunner just stops paying the latency serially.

## The `.qw` format

```
[0..3]   magic "QWR1"
[4..7]   int32 header length
[8..]    UTF-8 JSON: { "config": {...}, "data_offset"?, "tensors": [{name, dims, format, offset, size}] }
[...]    64-byte-aligned tensor data (offsets relative to data start)
```

When `data_offset` is present the header region is **reserved** (quantize does this by
default): the JSON can grow without moving the data region, so `quantize --append` adds
shard tensors in place — new bodies stream at EOF, existing bytes are never copied.

- `dims` are `[rows, cols]` (row-major, cols contiguous — GGUF `ne` order is normalized at convert time).
- `format` is the `QuantFormat` enum value; GGML types map 1:1 (Q4_0=2→Q4_0, Q8_0=8→Q8_0, Q2_K=10→Q2_K, Q3_K=11→Q3_K, Q4_K=12→Q4_K, Q5_K=13→Q5_K, Q6_K=14→Q6_K).
- All offsets/lengths are 64-bit; the header is tiny so it stays cached.
- Tensors are never copied at inference: `ModelStore.Open` mmaps the file and hands out raw pointers.

Canonical tensor names: `tok_emb`, `lm_head?`, `final_norm`, `l{i}.attn_norm`, `l{i}.wq/wk/wv/wo`,
`l{i}.ffn_norm`, `l{i}.w_gate/w_up/w_down` (dense) or `l{i}.router` + `l{i}.e{j}.w_gate/w_up/w_down` (MoE).

## Project layout

```
QuantRunner.sln
src/QuantRunner.Core/
  Dtypes.cs                     formats, block specs, f16/bf16 conversion
  Quantization/BlockQuantizer.cs    GGML byte-exact quantizers (Q2_K..Q6_K, Q4_0, Q8_0, TRI, I8_BLK)
  Quantization/BlockDequantizer.cs  inverse + get_scale_min_k4 + q3 kmask
  Kernels/SimdMath.cs           nibble/2-bit/tri unpack, fp32 dot, axpy
  Kernels/DotKernels.cs         fused dequant+dot per format, MatVec, MatVecParallel, GEMM
  Kernels/TransformerOps.cs     RMSNorm, RoPE, softmax, SwiGLU
  Memory/ModelStore.cs          .qw reader (zero-copy TensorView) + streaming writer
  Models/ModelConfig.cs         architecture config (in .qw header)
  Models/NameMapper.cs          GGUF/HF -> canonical names
  Importers/GgufReader.cs       GGUF v2/v3 (metadata KV section preserved verbatim)
  Importers/GgmlDequant.cs      ggml legacy block dequant (Q4_0/Q4_1/Q5_0/Q5_1) + ggml Q4_0 writer
  Importers/SafeTensorsReader.cs HF safetensors
  Importers/ModelConverter.cs   GGUF/ HF -> .qw (passthrough + requantization + KV embedding)
  Importers/Requantizer.cs      .qw -> smaller .qw (streaming dequant->requant, report-driven)
  Importers/GgufExporter.cs     .qw -> GGUF v3 (byte-exact types, MoE 3-D, tokenizer restore)
  Importers/HfExporter.cs       .qw/.gguf -> sharded HF safetensors + config/index/tokenizer
  Runtime/PagedKvCache.cs       paged KV + fused GQA attention
  Runtime/PrefetchPipeline.cs   async ping-pong layer staging (the AirLLM killer)
  Runtime/StreamingEngine.cs    layer-streaming forward + token generation
  Runtime/Sampler.cs            temp/top-k/top-p/repeat-penalty
  Tokenizers/BpeTokenizer.cs    HF byte-level BPE
src/QuantRunner.Cli/            quantize | requant | convert | synthesize | analyze | info | run | logits | bench | selftest
tests/QuantRunner.Tests/        17 correctness checks (formats, kernels, .qw, KV, BPE, engine,
                                gguf/ggml conversion)
```

## Correctness

`selftest` / `tests` verify, among others:

- f16/bf16 round trips incl. subnormals, max, inf;
- quantize→dequantize relative RMS per format (Q8_0 ≤ 2%, Q6_K ≤ 3%, Q4_K ≈ 9%, Q2_K ≤ 36%,
  TRI ≤ 55%, B1X ≤ 62%, T1E ≤ 72%, T1E_LO ≤ 80% of signal RMS — matching theory for these
  block schemes);
- sub-TRI ladder invariants: B1X error ≤ B1 error by construction and clearly better on
  outlier-heavy weights; T1E never loses to the tighter T1E_LO; every extreme format stays
  far above sign-flipping noise and runs end-to-end through the real engine;
- rANS entropy coder: 400-trial fuzz (random splits, degenerate 256/0/0 distributions,
  zero-padded payloads) round-trips bit-exactly;
- every fused dot kernel equals the dequantized fp32 reference to 2e-4;
- `.qw` write→read byte-exactness and 64-byte alignment;
- GQA attention invariants (uniform softmax ⇒ output equals mapped kv-head V);
- MoE router top-k weighting, paged KV growth (300 tokens ⇒ 2 pages/layer, not max-ctx);
- BPE merge ranks (`hello` → `he|ll|o`), unicode/emoji round trips, seeded sampling reproducibility;
- prefetch ON produces bit-identical logits to prefetch OFF;
- shard assembly (`--append`) is byte-identical to a single-shot conversion; idempotent re-append;
- the sub-TRI formats quantize through the real `quantize` path (safetensors → .qw), store in
  their exact format, and run end-to-end;
- requant: downgraded tensors re-encode in the target format, untouched tensors are byte-exact
  passthrough, `--dry-run` writes nothing, `--from-report` honors the quality gate (formats
  above the gate are rejected, never grows a tensor), and requantized models run on the engine;
- fused-projection splits (qkv_proj / gate_up_proj / GLM interleaved query_key_value) are
  byte-exact against the known sub-tensors, and the split model runs end-to-end;
- MLA/KDA/mHC checkpoints convert and run (fp32 pipeline matches the numpy reference to ~5e-7;
  nested `text_config` parses; per-layer attention kinds come from layer_types or tensor names);
- the GLM-5 hybrid path runs END TO END on a synthesized hybrid model: the engine selects the
  GLM-5 forward, logits stay finite across positions, and `ResetConversation` rewinds the KDA
  recurrence + MLA latent cache to the bit-identical greedy trajectory (also reproduced by a
  fresh engine instance);
- speculative decoding reproduces the target's plain greedy trajectory EXACTLY (same tokens,
  repeat penalty included) even with a completely unrelated draft model — corrections, EOS
  handling, KDA snapshot/restore and O(1) cache rollbacks all verified;
- `--append --config` heals a stored config that was written with zeroed fields;
- ggml legacy blocks (Q4_0/Q4_1/Q5_0/Q5_1), Q8_0 and one hand-built block per K-quant dequant
  to values computed from the llama.cpp reference (bit-plane order, kmask shuffle,
  get_scale_min_k4), and the ggml Q4_0 writer obeys the signed-delta rule;
- gguf -> qw -> gguf -> qw round trips are byte-stable: Q8_0/Q4_K passthrough stays byte-exact,
  experts reassemble to the 3-D `ffn_*_exps` layout byte-exactly, and the tokenizer/arch
  metadata embedded at import is restored verbatim on export;
- qw/gguf -> HF export: shard files + index.json cover every tensor, config.json fields match
  the source config, F16 exports are bit-equal to the .qw dequant, BPE tokenizer.json carries
  the exact vocab/merges/specials from the GGUF metadata.

## Honest benchmarking notes

The `bench` command prints measured numbers **from your machine** — layer stream MB/s,
tok/s with prefetch ON vs OFF, I/O stall %, and working set. Compare against AirLLM by
running both engines on the same model file and the same disk; the structural advantages
(no fp16 round-trip, no per-layer torch allocations, overlapped I/O) are constant, but the
magnitude depends on your storage. With a warm page cache, mmap-only mode can beat the
prefetch pipeline (no copy); the prefetch pipeline wins where it matters — cold and slow
storage (HDD, network volumes, cold NVMe) where synchronous streaming dominates AirLLM's
per-token time.

## Roadmap

- GPU offload hooks (`EngineOptions` already reserves the flag): cuBLAS/TensorRT-style matmul
  for resident layers, CPU streaming for the rest (hybrid CPU+GPU layer placement).
- Shared-expert fusion for DeepSeek-style MoE (currently skipped with a note at convert time).
- Multi-token chunked prefill through `DotKernels.GEMM` (already implemented, not yet wired
  into `StreamingEngine.Forward`).
- Sampling-with-distribution speculative acceptance (the current `--draft` verifies greedy
  only), and a chat-REPL runner (`run --chat`) on top of it.

## License

MIT. GGML block layout compatibility is to the llama.cpp open-source format specification.
