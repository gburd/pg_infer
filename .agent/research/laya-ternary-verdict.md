# Can the Laya models be converted to BitNet-style ternary weights and run on CPU?

Research report, 2026-11-18. Sources are linked inline; every number is attributed to the
page or paper it came from, and claims I could not verify are flagged as such.

**Short answer: no, and the experiment has already been run in public on Laya's own
network. It lost on both axes it was supposed to win.**

---

## 0. The finding that settles it

Before the theory, the measurement. Ollaya ([github.com/ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya),
1.4k stars) is the Jev-compatible server that serves the Laya checkpoints. It also publishes
its own model family, `coba`, described on its own model page as "**Laya's decision network**
on the Ettin encoders, distilled from `pplx-decider:flash`"
([ollaya.dev/library/coba](https://ollaya.dev/library/coba)). The `coba:400m-ternary` card
states the network explicitly: "Laya's (`laya` 0.3.7): the question, a `[MASK]` before every
option and the state in one sequence of up to 512 tokens; a two-layer transformer head scores
each option at its marker"
([huggingface.co/ollaya-dev/coba-400m-ternary](https://huggingface.co/ollaya-dev/coba-400m-ternary)).
That is Laya's architecture, at Laya's parameter count, on a ModernBERT-family encoder, with
matched full-precision and ternary tags. It is the exact experiment the question asks about.

The results, measured by Ollaya on its own hardware and published with raw data
([bench.ollaya.dev](https://bench.ollaya.dev), Edition 2026.10):

| tag | weights | params | typed-decisions acc | ECE | CPU, 5 questions | GPU, 5 questions |
|---|---|---|---|---|---|---|
| `coba:400m` | fp32 | 421M | 0.591 | 0.089 | 832 ms | 21 ms |
| `coba:400m-ternary` | ternary | 421M | 0.559 | 0.047 | **924 ms** | 20 ms |
| `coba:150m-ternary` | ternary | 164M | 0.535 | 0.053 | 461 ms | 15 ms |
| `coba:68m-ternary` | ternary | 75M | 0.482 | 0.076 | 314 ms | 14 ms |
| `coba:1b` | fp32 | 1.1B | 0.608 | 0.090 | 1479 ms | 20 ms |

(CPU figures from the Ollaya latency chart, [ollaya.dev/results#speed](https://ollaya.dev/results#speed),
Threadripper 3960X; accuracy and ECE from the leaderboard table.)

At matched parameter count, ternary is **3.2 accuracy points worse and 11% slower on CPU**.
The speed axis is the one ternarisation exists to win, and it lost.

Two reasons the Ollaya team gives, both on their own pages:

1. **The ternary weights are stored as float32 for CPU serving.** From the model page:
   "A ternary tag's weights are −1, 0 or +1 times a scale per output row (as BitNet b1.58),
   **stored as float32 for Ollaya**; the 2-bit graph the browser runs is in the same Hugging
   Face repository." The 2-bit packing exists for the WASM/WebGPU target (a 147 MB browser
   download), not for the server. On CPU the model runs through ONNX Runtime at fp32 over
   dequantised ternary values, which is strictly worse than fp32 dense: same FLOPs, same byte
   traffic, plus whatever the quantise/dequantise plumbing costs. The 11% regression is that
   plumbing.
2. **It required quantisation-aware training, not conversion.** "Ternary weights:
   quantization-aware training from the fp model of the same size (absmean per row, as BitNet
   b1.58, with a straight-through estimator), **distilled from it (logits, attention relations
   and every layer's hidden states)**, with the fp weights blended into the quantized ones over
   the first 30% of the steps."

So the team that owns this exact network, with the training data, the teacher model and the
runtime, needed QAT plus three-signal distillation plus a 30%-of-steps weight-blending schedule
to get within 3.2 points, and still shipped a CPU path that is slower than the dense one.

---

## 1. What BitNet actually requires

### The b1.58 paper trains from scratch, and says so

Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
([arXiv:2402.17764](https://arxiv.org/abs/2402.17764), 2024-02-27) is unambiguous. Section 2:

> "BitNet b1.58 is based on the BitNet architecture, which is a Transformer that replaces
> nn.Linear with BitLinear. **It is trained from scratch, with 1.58-bit weights and 8-bit
> activations.**"

And the paper positions itself explicitly *against* post-training quantisation, in the
introduction:

> "One approach to address these challenges is to use post-training quantization to create
> low-bit models for inference. [...] **However, post-training quantization is sub-optimal**,
> even though it is widely used in industry LLMs."

The paper's headline claim ("matches the full-precision Transformer LLM with the same model
size and training tokens") is a claim about a model trained from scratch at 1.58 bits, not a
claim about converting an existing one. Reading it as a conversion recipe inverts the paper.

This is reinforced everywhere downstream in the BitNet line:

- **BitNet-b1.58-2B-4T** ([arXiv:2504.12285](https://arxiv.org/abs/2504.12285)) is described
  as "the first open-source, **native** 1-bit LLM", trained on 4 trillion tokens.
- **BitNet a4.8** ([arXiv:2411.04965](https://arxiv.org/abs/2411.04965)) and **BitNet v2**
  ([arXiv:2504.18415](https://arxiv.org/abs/2504.18415)) both report results for models
  "trained from scratch" or "trained with" the respective activation scheme. BitNet v2's
  abstract: "BitNet v2 **trained from scratch** with 8-bit activations matches BitNet b1.58
  performance."
- **bitnet.cpp**'s own conversion guide says it in the imperative: the `--outtype i2_s` mode
  "**requires BitNet natively-trained models with ternary weights**"
  ([docs/bitnet-embeddings-i2s-guide.md](https://github.com/microsoft/BitNet/blob/main/docs/bitnet-embeddings-i2s-guide.md)).

I2_S is a *packing format*, not a quantiser. It takes weights that are already ternary and
stores them at 2 bits per weight (`scale = 1/mean(|w|)`, `q = round(w*scale).clamp(-1,1)`,
128 values per 32-byte block). Pointing it at a dense fp16 encoder produces a model whose
weights have been rounded to three values with no training to compensate, which is the
"Q2BERT" regime discussed below.

### The encoder counter-case: Microsoft's BitNet embedding models

This is the strongest evidence *for* ternary encoders, and it is worth stating fairly before
dismissing it. In July 2026 Microsoft released **BitNet-embedding-0.6B** and
**BitNet-embedding-270M** ([HF](https://huggingface.co/microsoft/BitNet-embedding-0.6B)),
the first ternary models targeting an encoder-ish workload, with measured x86 CPU prefill
speedups:

| | F16 GGUF (t/s) | I2_S GGUF (t/s) | speedup |
|---|---|---|---|
| pp128 | 382.15 | 870.90 | 2.28x |
| pp512 | 371.86 | 716.27 | 1.93x |
| pp1024 | 341.55 | 620.58 | 1.82x |
| pp4096 | 236.76 | 336.32 | 1.42x |

(0.6B model, Intel Xeon Platinum 8573C, 8 threads, from the I2_S guide.) Quality loss is
genuinely small: MMTEB (eng, v2) average 67.60 for the ternary model vs 67.95 for the FP16
teacher, a 0.35-point gap, and the I2_S *packing* step costs almost nothing on top of the
already-ternary safetensors (0.7180 vs 0.7188 average over eight MTEB tasks).

So prefill-dominated ternary inference on x86 does work, it is 1.4-2.3x, and the loss can be
held under half a point. Three things make it not transferable to Laya:

1. **Same from-scratch requirement.** The model card, verbatim: "Weights are quantized to
   ternary values {−1, 0, +1} using absmean quantization. [...] **Trained from scratch with
   this quantization scheme, not post-training quantized.**" The pipeline is BitNet
   conversion → continual contrastive pre-training on 1B text pairs → distillation-based
   supervised fine-tuning with similarity-distribution and attention-relation distillation
   from an FP16 teacher. That is a pre-training-scale budget, not a conversion.
2. **They are decoders.** "The models use **decoder-only architecture** with last-token pooling."
   Backbones are Qwen3-0.6B and Gemma3. They are embedding models, not bidirectional encoders,
   and their speedup curve is measured on *prefill*, which is the right analogue for Laya, but
   the architecture is not.
3. **The speedup decays with sequence length, and the guide says why.** From the results
   summary: "Speedup is highest at short prompt lengths where matmul (weight-bound) dominates,
   and **decreases at longer prompts where attention (compute-bound) takes over**." And on the
   270M: "the 270M model has fewer linear projection parameters relative to other operations,
   so the benefit of ternary weight quantization is proportionally smaller." Both trends point
   the wrong way for a 421M model at 512-1024 tokens. See §2.

### Post-training ternary methods, and what they cost

| method | regime | requires training? | reported retention |
|---|---|---|---|
| **TernaryBERT** ([arXiv:2009.12812](https://arxiv.org/abs/2009.12812), EMNLP 2020) | W2-E2-A8 BERT | yes: QAT + layer-wise distillation | GLUE dev 83.3 MNLI-m vs 84.5 FP, 14.9x smaller |
| **Q2BERT** (TernaryBERT's own baseline, min-max ternarisation of Q8BERT) | W2-E8-A8 | no (the PTQ-like control) | **MNLI-m 47.2 vs 84.5; CoLA 0.0; STS-B 4.4** |
| **BinaryBERT** ([arXiv:2012.15701](https://arxiv.org/abs/2012.15701)) | W1 | yes: ternary weight splitting + fine-tune | "slight drop", 24x smaller |
| **TernaryLLM** ([arXiv:2406.07177](https://arxiv.org/abs/2406.07177)) | W1.58A16 | yes: dual learnable ternarisation + OFF distillation | beats prior W2A16 SOTA by 5.8 ppl on C4 |
| **BitDistiller** ([arXiv:2402.10631](https://arxiv.org/abs/2402.10631)) | sub-4-bit | yes: QAT + self-distillation (CAKLD) | best-in-class at 3- and 2-bit |
| **GPTQ / AWQ / HQQ / AQLM at 2 bits** | PTQ | no | see §3; 2-bit is where PTQ breaks |

The pattern is complete and has held for six years: **every ternary method that retains
accuracy involves training.** The names say it: TernaryBERT is "Distillation-aware",
BitDistiller is a distiller, TernaryLLM's contribution is "Outlier-Friendly Feature Knowledge
Distillation". No method in this table ternarises weights without a gradient step.

The Q2BERT row is the most useful number in the table, because it is the closest published
analogue to "ternarise an existing encoder and run it". TernaryBERT's authors built it
themselves as a control, "by modifying the min-max 8-bit quantization to min-max ternarization
using the released code of Q8BERT". It scores **47.2 on MNLI-m against the FP model's 84.5**,
and **0.0 on CoLA and 4.4 on STS-B** -- not degraded, destroyed. That is what PTQ-to-ternary
on a BERT-class encoder looks like.

TernaryBERT's own ablation then prices the training that rescues it (Table 6, `TernaryBERT_TWN`):

| | MNLI-m/mm | CoLA | RTE | SQuAD v1.1 |
|---|---|---|---|---|
| TernaryBERT (full recipe) | 83.3/83.3 | 55.7 | 72.9 | 79.9/87.4 |
| − Transformer-layer distillation | 82.9/83.3 | 52.7 | 69.0 | 76.6/84.9 |
| − Transformer − logit distillation (plain CE on labels) | 80.8/81.1 | **45.4** | **56.3** | 74.3/83.2 |

The paper's own summary: "the accuracy for CoLA, RTE and SQuAD v1.1 drops by **over 5%** by
removing the distillation." And Table 7 shows initialising from a *pre-trained* rather than a
*fine-tuned* BERT costs another 9.7 points on CoLA (55.7 → 46.0). The distillation is not a
nice-to-have; it is most of the method.

On reproduction status: TernaryBERT is EMNLP 2020, peer-reviewed, with a widely-cited
Huawei-Noah implementation, so I treat its GLUE numbers as credible. The BitNet b1.58 line is
a different matter -- 2402.17764 is an arXiv preprint whose central "matches full-precision"
claim at 3B scale I have not found an independent from-scratch reproduction of (the training
budget makes that expensive, which is presumably why). What *has* been independently exercised
is the inference side: bitnet.cpp's kernels and the released 2B-4T checkpoint are open and
widely run. The pattern throughout is that the hard-to-verify claims are about training and
the easy-to-verify ones are about kernels.

---

## 2. Encoders vs decoders: where Laya's time actually goes

This is where the case fails on its own terms, independent of accuracy.

Ternarisation buys speed by shrinking weight *bytes*. It helps exactly when a kernel is
starved for bandwidth, which is the signature of autoregressive decode: one token, one pass
over every weight, arithmetic intensity around 2 FLOP/byte, nothing to amortise. bitnet.cpp's
paper states the premise directly: "mixed-precision matrix multiplication (mpGEMM) constitutes
the bulk of inference time in ternary LLMs"
([arXiv:2502.11880](https://arxiv.org/abs/2502.11880)).

Laya is the opposite regime. From its config
([encoder/config.json](https://huggingface.co/convaiinnovations/laya/raw/main/encoder/config.json)):
`hidden_size` 1024, 28 layers, `intermediate_size` 2624 (GeGLU, so `mlp.Wi` is [5248,1024]),
`num_attention_heads` 16, vocab 50368, and crucially `layer_types` alternating 10 `full_attention`
with 18 `sliding_attention` at `local_attention` 128. From the safetensors headers, the encoder's
linear weights total 343.2M parameters (`mlp.Wi` 150.5M, `attn.Wqkv` 88.1M, `mlp.Wo` 75.2M,
`attn.Wo` 29.4M), plus 51.6M of token embeddings, plus a 25.3M decision head.

Computing arithmetic intensity against the encoder's linear weights:

| seq len | GEMM GFLOP | attention GFLOP | AI vs f32 weights | AI vs f16 | AI vs 2-bit |
|---|---|---|---|---|---|
| 192 (`head_max_len`) | 131.8 | 3.3 | 98 | 197 | 1575 |
| 320 (state budget) | 219.6 | 7.2 | 165 | 330 | 2644 |
| 512 (`laya` max_len) | 351.4 | 15.6 | 267 | 535 | 4277 |
| 1024 (`typed-decisions` max_len) | 702.8 | 52.6 | 550 | 1101 | 8805 |

Even at fp32 and the shortest realistic sequence, arithmetic intensity is ~100 FLOP/byte.
Roofline crossover on a modern x86 core sits somewhere in the 10-50 FLOP/byte range depending
on how many AVX-512 FMA ports you can keep fed. **Laya's encoder forward pass is already
compute-bound by one to two orders of magnitude at fp16, before any quantisation.** Shrinking
the weights from 686 MB to 86 MB moves a number that is not the binding constraint. There is
no bandwidth wall to remove.

Two architectural details make it worse than the generic encoder case. First, `tie_word_embeddings`
is true and the vocab is 50368, so 51.6M parameters are embeddings -- and both bitnet.cpp's
I2_S convention and Ollaya's own ternary models leave embeddings out of ternarisation ("Embedding
weights: float16" in the I2_S tensor-type table; "the token embeddings have 4 bits" on the coba
card). Second, sliding-window attention at 128 tokens on 18 of 28 layers already keeps the
attention term small (15.6 of 367 GFLOP at S=512), so the compute that ternarisation *cannot*
touch is a smaller slice than usual -- which sounds good until you notice it means the GEMMs
ternarisation would accelerate are already the thing that is compute-bound, not bandwidth-bound.

The empirical check agrees. The BitNet embedding speedup decays monotonically from 2.28x at
pp128 to 1.42x at pp4096, and Microsoft attributes the decay to exactly this mechanism. It also
shrinks with model size at fixed sequence length (0.6B: 1.42-2.28x; 270M: 1.32-1.74x). Laya at
421M and 512-1024 tokens sits in the part of both curves where the measured gain is smallest --
and that is for a model trained from scratch at 1.58 bits with hand-written AVX2/NEON mpGEMM
kernels behind it.

And then the Ollaya A/B lands on the same spot from the other direction: 832 ms fp32 vs 924 ms
ternary. Theory says the gain should be small; measurement on this exact network says it is
negative.

One more structural point. Laya's whole batching story works against ternary. The card says
"Every question in a call is answered in one single forward pass", and the measured curve
(from [eval/results.json](https://huggingface.co/convaiinnovations/laya/raw/main/eval/results.json),
T4) is 38.4 ms for 1 question, 156.0 ms for 10 (15.6 ms/q), 721.4 ms for 50 (14.4 ms/q). Batching
raises the effective batch dimension of every GEMM, which moves the model *further* into the
compute-bound regime. The harder you use Laya the way it is designed to be used, the less a
weight-bandwidth optimisation can possibly buy.

---

## 3. The calibration problem

Laya's value proposition is calibrated probabilities. Its own card reports the typed-decisions
numbers (Brier 0.062, ECE 0.213) and its benchmark table is against TypeSafe Jev. So the
question is not whether ternarisation preserves argmax but whether it preserves the
distribution. The evidence says distributions are the first thing to go.

**Direct evidence at 2 bits.** "Reliability Scaling Laws for Quantized Large Language Models"
([arXiv:2607.10855](https://arxiv.org/abs/2607.10855)) evaluates four LLaMA models at 2, 3, 4
and 8 bits across six PTQ methods (BitsandBytes, AWQ, GPTQ, HQQ, Quanto, AQLM) measuring
accuracy, uncertainty *and* calibration (Brier score) separately -- 5,859 experiments. The
headline is that accuracy and reliability come apart: "while the performance scales
monotonically with the total number of bits, **the reliability scalings are nonlinear**. A
reliability peak occurs for 4-bit quantized models". And specifically on 2 bits:

> "2-bit quantized models exhibit a **substantial behavioral shift relative to their
> full-precision counterparts**, which contributes to their consistently poor performance. For
> 3-bit quantization, the KL divergence remains noticeably larger than that of 4-bit models.
> [...] In contrast, 4-bit quantization maintains a small KL divergence across model scales."

KL divergence from the full-precision model's output distribution is precisely the quantity
that matters for Laya: it is the distance between the probabilities you shipped and the
probabilities you now return. The paper finds it large at 2 bits and small at 4. It also notes
that their 2-bit arm is AQLM-**PV**, i.e. AQLM plus an additional fine-tuning step -- so even
the best 2-bit number in the study is not pure PTQ. Their own appendix adds that EfficientQAT
"achieves impressive results in 2-bit scenarios" but "a performance gap remains compared to
higher bitwidths".

The paper's own caveat, which I should pass on: they study decoders on QA tasks, with
token-level Brier, not encoders emitting a distribution over a closed label set. The mechanism
(output distributions shift more than argmax does, and disproportionately below 3 bits)
transfers cleanly; the magnitudes should not be read across.

**Corroborating evidence.** ZeroQuant-V2 ([arXiv:2303.08302](https://arxiv.org/abs/2303.08302))
found across RTN/GPTQ/ZeroQuant on 125M-176B models that "none of the current methods can
achieve the original model quality" even at INT4-weight -- two bits above ternary.
"Target-Aware Calibration Data Selection for Preserving Uncertainty in Quantized Language
Models" ([arXiv:2608.21019](https://arxiv.org/abs/2608.21019)) exists because the problem is
real: it frames calibration-data choice "as a target-dependent uncertainty-preservation
problem", noting that quantisation's "effect on uncertainty behavior, such as confidence,
margins, and abstention, is rarely treated as a primary objective", and shows via a
"mixture-mismatch argument" that **no single calibration recipe fits all targets**. If Laya
were ternarised, its calibration would have to be refit per deployment, which is work the
project already warns about at full precision.

**Where the evidence cuts the other way, honestly.** Two findings complicate the simple story:

- Ollaya's measured ECE for `coba:400m-ternary` is **0.047, better than `coba:400m`'s 0.089**.
  I do not think this rescues ternary. The ternary tags were trained in-domain on ten of the
  benchmark's train splits while the fp32 tags were not, so the comparison is confounded
  (Ollaya flags both tags "trained on split" and leaves those tasks out of the headline score,
  which is why the ternary tags' chance-corrected scores are 4.4 and 4.6 against the fp32
  models' 35.8 and 36.3). What the ECE column does establish is narrower but still useful: a
  ternary model of this architecture *can* be well calibrated **when QAT and distillation fit
  the calibration during training**. That is a point in favour of the Ollaya recipe, not in
  favour of post-training conversion.
- "When Calibration Depends on the Scoring Rule"
  ([arXiv:2608.03854](https://arxiv.org/abs/2608.03854)) finds that for 7B classification at
  FP16/INT8/INT4, implementation choices swamp the quantisation effect: switching from summed
  to mean-token log-likelihood **reverses which model looks better calibrated**, while prompt
  choice moves accuracy 2.9-17.8 points and "eight-bit quantization changes accuracy by at most
  1.1 percentage points". The honest reading is that at 8 bits the quantisation effect on
  calibration is small enough to be drowned out by scoring and prompting decisions. That is an
  argument for int8 (§5), not for ternary -- their lowest precision is INT4, two bits above.

There is also a specific, named mechanism by which ternarisation would likely hurt Laya more
than a generic classifier. Laya scores each option at its own `[MASK]` token and softmaxes
*within* the question, so the probabilities depend on small logit *differences* between
option markers that share a sequence. The card documents that this is already fragile: issue
[#156](https://github.com/NandhaKishorM/laya/issues/156) reports `noul` "can follow its option
labels instead of the state", returning "a confident 'no' for clearly positive input", and the
`choice:11+` fitted temperature is **0.1006** -- a tenth, meaning logits for large option sets
are scaled up tenfold to be calibrated, so the per-marker logit spread there is genuinely tiny.
Ternarisation injects per-row weight error that perturbs exactly those differences. I have not
found a paper measuring quantisation error on marker-scored encoders specifically, so I flag
this as mechanism-level reasoning, not a cited result -- but it is the kind of thing that makes
the published Brier of 0.062 unsafe to assume survives.

---

## 4. What larql can actually do today

I cloned `chrishayuk/larql` and read the code rather than the docs. The user's description is
accurate, and the constraints are tighter than stated.

**The conversion path refuses non-BitNet input.** In
`crates/larql-cli/src/commands/extraction/convert_cmd.rs:516-531`, `--keep-quant` is gated on
an architecture-string check:

```rust
let is_bitnet = gguf.metadata.get("general.architecture")
    .and_then(|v| v.as_str())
    .map(|s| s.starts_with("bitnet"))
    .unwrap_or(false);
let do_keep_quant = keep_quant && is_bitnet;
if keep_quant && !is_bitnet {
    eprintln!("  --keep-quant: ignored (architecture is not BitNet; no I2_S tensors to retain)");
}
```

It then loads only ggml type 36 (I2_S) tensors verbatim. So `--keep-quant` is a *retain* flag,
not a *quantise* flag: it preserves ternary bytes that are already in the GGUF. On any
non-BitNet input it prints a warning and silently does nothing. There is no code path in larql
that produces ternary weights from dense ones.

**The ternary kernel is matvec-only, which is the wrong shape for an encoder.** Every ternary
entry point in `crates/larql-compute/src/cpu/ops/ternary_matvec.rs` is matrix-**vector**:
`matvec_i2s_f32`, `matvec_i2s_q8_into`, `matvec_i2s_q8_neon_into`, dispatched through
`matvec_i2s_a8_into`. A grep for `matmul`/`gemm` in the ternary module finds only references to
the dequant-and-matmul *reference* used for parity tests. An encoder forward pass over 512
tokens needs a GEMM with M=512; running it as 512 independent matvecs forfeits the operand
reuse that makes a GEMM fast, which is the single biggest reason this path would not help even
with perfect SIMD.

**The x86 gap is confirmed, with a number.** The module header
(`ternary_matvec.rs:14-30`) and `ROADMAP.md:2902` agree:

> "NEON gives ~12–13× the f32 reference on BitNet shapes; **x86_64 has the scalar-A8 ~2.4× today.**"
> "Remaining: an AVX2 `_mm256_sign_epi8` twin so x86_64 gets the full SIMD win."

So the measured x86 position is 2.4x over larql's *own* f32 reference from int8 activations
alone, with no ternary SIMD, against 12-13x on aarch64. I could **not** verify the "~1 tok/s
for a 2B decoder on x86" figure -- there is no BitNet entry in `bench/baselines/` and no
tok/s number attached to any BitNet row in the ROADMAP. Treat that one as unconfirmed; the
2.4x-vs-12x ratio is what the repo actually documents.

**Does Graviton change the answer?** Partly, and not enough. aarch64 gets the real kernel, so
the 12-13x matvec speedup is available. But (a) it is 12-13x over larql's own scalar f32
reference, not over a tuned fp16 GEMM via oneDNN or ONNX Runtime, so it is not the comparison
that matters for a deployment decision; (b) it is still matvec, so the encoder's M=512 reuse is
still forfeited; and (c) none of it is reachable, because of the next point.

**larql cannot load an encoder at all.** The ternary forward pass is decoder-only by
construction: `crates/larql-inference/src/ternary/predict.rs:162,299` and `ffn.rs:171` all
implement "per-head **causal-masked** scaled-dot-product attention", with a `BitnetKvCache`
for autoregressive decode. Laya is bidirectional with sliding-window attention; there is no
bidirectional path to use. And the architecture registry
(`crates/larql-models/src/detect/registry/table.rs`) lists gemma/llama/qwen/olmo/mistral/
mamba2/mixtral/gpt2/gpt_oss/deepseek/glm/kimi/starcoder2/granite/bitnet -- **no `modernbert`,
no `bert`, no encoder family at all** (a repo-wide grep for `ModernBert` returns nothing).
Finally the BitNet stack is explicitly a parallel one: ROADMAP.md documents that "BitNet
bypasses every shared seam: `QuantFormat`/`FormatRoute` dispatch, the `KvEngine` trait,
`larql-kv::KvCache`, the models arch registry, and the vindex build pipeline", and that
`BitLinearWeight` "is reachable only by direct call".

The practical sequence to use larql for Laya would therefore be: add ModernBERT to the arch
registry, write a bidirectional encoder forward pass with sliding-window attention, write a
ternary GEMM (not matvec), write the AVX2 kernel for x86, write a GGUF exporter for Laya's
custom head (`scorer.*`, `act_head.*`, `type_emb`, which are not standard tensors), and
separately obtain ternary weights from somewhere -- because larql cannot make them. That last
item is the QAT run from §1, and it is the whole project.

---

## 5. The practical alternatives, ranked

The goal, restated: run a 421M encoder cheaply on CPU, preserving calibration.

**0. Check whether this is already solved, because it largely is.** The premise in the question
("the Laya ecosystem already ships ONNX, the Ollaya server is described as fast: ONNX,
memory-mapped") needs one correction and one confirmation.

*Correction:* none of the three `convaiinnovations/*` repos contain an ONNX file. I listed
every sibling in all three via the HF API: `model.safetensors`, `encoder/config.json`,
`rl_agent_config.json`, tokenizer files, and in the root repo the `multilingual/` and
`typed-decisions/` subfolders. No `.onnx`. The `laya` PyPI package has an `onnx` extra
(`onnx`, `onnxruntime`, `onnxscript`) and ships `laya/onnx_agent.py` which builds an
`ort.InferenceSession` over CUDA-then-CPU providers, so laya can *export and run* ONNX --
but the published checkpoints are PyTorch, and you do the export.

*Confirmation:* Ollaya serves exactly this, and publishes derived ONNX graphs at
[huggingface.co/ollaya-dev](https://huggingface.co/ollaya-dev) (including `ollaya-dev/laya`).
Its stated precision policy: "ONNX uses fp16 on GPU and **fp32 on CPU** when available", with
"fp32 exports give the same decision as the PyTorch reference on 100% of 2,383 questions per
checkpoint". So the shipped CPU path today is **fp32 ONNX**, and it costs 945 ms for five
questions on a Threadripper 3960X (`laya:en`), 944 ms (`laya:typed-decisions`), 516 ms
(`laya:multilingual`).

That last number matters: fp32, not int8, not bf16. The cheap wins are unclaimed.

**1. Pick the smaller checkpoint (free, measured, available now).**
`laya:multilingual` at 322M runs 516 ms against `laya:en`'s 945 ms on the same CPU -- **1.83x**
for a model whose chance-corrected bench score is *higher* (36.7 vs 30.1) and which covers
100+ languages. The card's own claim is "~2.2x faster". No conversion, no risk, available
today. If English-only is a hard requirement this does not apply, but it is the first thing to
measure.

**2. int8 dynamic quantisation via ONNX Runtime.** This is the right tool and it is one
`quantize_dynamic` call on an existing export. Weights to int8, activations quantised per-batch
at runtime, no calibration dataset needed, and it lands on AVX-512 VNNI / `vpdpbusd` on x86 or
i8mm on Graviton3+ -- hardware that *exists* and that oneDNN already targets, unlike ternary
where the kernel is the research project. The calibration case is strong from §3: the
scoring-rule paper measured "at most 1.1 percentage points" accuracy change at 8 bits for
adapted checkpoints, and the Reliability Scaling Laws paper puts 8-bit comfortably on the
good side of the reliability curve whose cliff is at 2-3 bits. Honest caveats: real-world int8
dynamic speedups on transformer encoders are typically 1.5-3x, not the 4x the bit ratio
suggests, because activation quantisation overhead and non-GEMM ops do not shrink; and I have
not measured it on Laya, so verify both latency and ECE against the fp32 reference.

**3. bf16 or fp16.** Halves weight bytes and gets AVX-512-BF16 / AMX on recent x86 and native
bf16 on Graviton3+. Smaller win than int8 but nearly zero accuracy risk, and Ollaya's own
measurement of its fp16 graphs ("agree on 99.6% to 99.8% of decisions; the others are
near-ties") suggests half precision is near-free for this network. A reasonable first step if
int8 turns out to move ECE.

**4. ONNX graph optimisation.** Operator fusion (attention, LayerNorm, GELU), constant folding,
`ORT_ENABLE_ALL`. Free, no numerical risk at fp32, and worth doing before any quantisation so
you are measuring against an optimised baseline rather than crediting quantisation for wins
that were graph overhead.

**5. Batch more questions per forward pass.** Already supported (`predict_batch`, and "Every
question in a call is answered in one single forward pass"), and the measured curve is steep:
38.4 → 15.6 → 14.4 ms/question at 1, 10, 50 questions. Per-question cost falls 2.7x from 1 to
50. Note §2: this pushes further into compute-bound territory, so it composes with int8 (which
helps compute via VNNI) better than it would with ternary (which helps bandwidth).

**6. Distil to a smaller encoder.** The highest ceiling and the highest cost, and it is the
option with the best evidence, because Ollaya published the curve for Laya's own network:

| tag | params | CPU (5 q) | typed-decisions | vs `coba:400m` |
|---|---|---|---|---|
| `coba:400m` | 421M | 832 ms | 0.591 | baseline |
| `coba:150m-ternary` | 164M | 461 ms | 0.535 | 1.81x faster, −0.056 |
| `coba:68m-ternary` | 75M | 314 ms | 0.482 | 2.65x faster, −0.109 |

Distilling to one third the parameters bought 1.81x on CPU. Ternarising at the *same*
parameter count bought −1.11x. If you are going to spend a training budget, this is where it
goes: parameter count is the lever that moves CPU latency for a compute-bound encoder, not bits
per weight. (The ternary tags are the only small ones Ollaya published, so these rows conflate
"smaller" with "ternary" -- but that cuts in favour of the argument, since the smaller ternary
models beat the larger fp32 one on speed by a margin far larger than ternary's own −11%.)

**Not recommended: ternary.** Needs a QAT run with three-signal distillation (§1), needs a
ternary GEMM and an AVX2 kernel that do not exist in larql (§4), threatens the calibration that
is the product (§3), and targets a bandwidth bottleneck the model does not have (§2). The one
published A/B on this exact network came out negative on both accuracy and CPU latency (§0).

---

## Verdict

Not worth attempting. The decisive fact is not theoretical: Ollaya has already built Laya's
decision network with ternary weights at Laya's own parameter count, with QAT plus
logits/attention-relation/hidden-state distillation plus a weight-blending schedule, and the
result is 3.2 accuracy points worse than the fp32 twin and **11% slower on CPU** (924 vs 832 ms
for five questions), because the CPU path stores ternary weights as float32 and only the
browser gets the 2-bit graph. That is the best case, from the team with the training data and
the runtime; a post-training conversion has no QAT and no distillation and would land nearer
TernaryBERT's own Q2BERT control, which scored 47.2 on MNLI-m against 84.5 full-precision and
0.0 on CoLA. The arithmetic explains why there is nothing to win even if accuracy held: Laya's
encoder runs at ~100-500 FLOP/byte at fp16 over 192-1024 tokens, one to two orders of magnitude
compute-bound, so shrinking weights from 686 MB to 86 MB moves a quantity that is not the
constraint -- which is the same mechanism Microsoft cites for its own ternary embedding speedup
decaying from 2.28x at 128 tokens to 1.42x at 4096, and shrinking further on the smaller model.
And larql could not execute it regardless: `--keep-quant` refuses any architecture whose name
does not start with `bitnet` and only *retains* existing I2_S tensors (it never creates them),
the ternary path is matrix-**vector** with no GEMM, the forward pass is causal-masked
decoder-only, and no BERT or ModernBERT family appears in the architecture registry at all.

The single highest-value cheap experiment is **int8 dynamic quantisation of an ONNX export of
`laya-multilingual`, measured for latency and ECE together against the fp32 baseline on your
own labelled set.** It is roughly a day: export with the `laya[onnx]` extra, run
`quantize_dynamic`, then score both arms on the same questions reporting p50 latency, accuracy,
Brier *and* ECE side by side. It is cheap because it needs no training and no kernel work;
it targets the right bottleneck because VNNI/i8mm accelerate compute rather than bandwidth; the
calibration risk is bounded by published evidence (≤1.1 points accuracy change at 8 bits, with
8-bit on the safe side of the reliability curve whose cliff sits at 2-3 bits); and it stacks
with the two free wins you have not taken -- the 322M checkpoint is already 1.83x faster than
the 421M one on the same CPU at a *higher* bench score, and the current shipped CPU path is
plain fp32. Measure ECE, not just accuracy: if int8 preserves argmax but moves ECE, that is the
signal to stop at bf16, and it is the same signal that would have killed the ternary plan on
day one.

---

## Verification status

Verified by reading the primary source:

- All three Laya `config.json`, `encoder/config.json`, `rl_agent_config.json`, and the
  safetensors tensor headers (parameter counts, shapes, dtypes) via the HF API.
- Absence of ONNX files in all three `convaiinnovations/*` repos (full sibling listing).
- `laya` PyPI metadata: the `onnx` extra and `laya/onnx_agent.py`'s CPU/CUDA provider logic.
- The b1.58 "trained from scratch" and "post-training quantization is sub-optimal" quotes,
  from the ar5iv rendering of 2402.17764.
- bitnet.cpp's I2_S guide: the "requires BitNet natively-trained models" line, the packing
  format, the tensor-type table, and the pp128-pp4096 F16-vs-I2_S tables.
- BitNet embedding model cards: "trained from scratch with this quantization scheme, not
  post-training quantized", decoder-only architecture, MTEB numbers.
- TernaryBERT Tables 1, 6 and 7 (GLUE dev, distillation ablation, initialisation ablation) and
  the Q2BERT baseline, from the ar5iv rendering of 2009.12812.
- Reliability Scaling Laws (2607.10855) full text: the 4-bit reliability peak, the 2-bit
  behavioural-shift/KL finding, the AQLM-PV caveat, the EfficientQAT appendix.
- larql at HEAD: the `--keep-quant` gate, matvec-only ternary ops, the NEON-vs-x86 2.4x/12-13x
  line, causal-masked decoder-only ternary forward, and the architecture registry's contents.
- Ollaya: README precision policy, the `coba` model page tables, the bench leaderboard rows
  (accuracy/ECE/params), the CPU latency chart, and the `coba-400m-ternary` card's QAT recipe.

Could not verify, flagged rather than filled in:

- **The "~1 tok/s for a 2B decoder on x86" figure for larql.** No BitNet baseline in
  `bench/baselines/`, no tok/s figure in any BitNet ROADMAP row. What the repo documents is
  "x86_64 has the scalar-A8 ~2.4× today" against NEON's 12-13x.
- **Independent reproduction of BitNet b1.58's "matches full-precision" training claim.** I
  found none; the inference-side kernels and the released 2B-4T checkpoint are open and widely
  run, but the from-scratch training claim at 3B scale appears unreplicated outside the
  authors' group.
- **The Jev comparison numbers** in Laya's own tables are third-party published, which the
  cards state plainly: "Jev figures are third-party published, never measured here (no
  TypeSafe API access); sample sizes and prompts differ."
- **Whether `coba:400m-ternary`'s better ECE (0.047 vs 0.089) survives a clean comparison.**
  The ternary tags trained in-domain on ten benchmark train splits and the fp32 tags did not,
  so the two are not matched on data. Ollaya itself excludes those tasks from its headline
  score for this reason.
- **Quantisation error on marker-scored encoder heads specifically.** The §3 argument that
  ternarisation would disproportionately damage Laya's within-question softmax over `[MASK]`
  logits is mechanism-level reasoning from the architecture plus the documented `choice:11+`
  temperature of 0.1006 and issue #156, not a cited measurement.
- **Any int8/bf16 measurement on Laya.** None published that I could find, by the Laya team or
  Ollaya; Ollaya's CPU path is fp32 and its fp16 graphs are GPU-only. This is precisely why the
  recommended experiment is cheap and worth running.
