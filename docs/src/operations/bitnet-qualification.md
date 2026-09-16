# Real-Model Qualification: BitNet b1.58-2B-4T

Results from qualifying pg_infer against a real `larql-server` serving
`microsoft/bitnet-b1.58-2B-4T`, rather than the synthetic fixture the unit
tests use. Recorded because the numbers are only meaningful with the setup
attached, and because one of the findings below could not have been found
any other way.

## Setup

| | |
|---|---|
| Instance | EC2 `m7i.8xlarge` (32 vCPU, 123 GB, x86_64) |
| pg_infer | 0.1.2-alpha |
| larql-server | 0.2.0 (`feat/bitnet-serve`) |
| PostgreSQL | 18.6 (pgrx-managed) |
| Rust | 1.98.0 (pinned) |
| Model | `microsoft/bitnet-b1.58-2B-4T-gguf` / `ggml-model-i2_s.gguf` (1.2 GB) |

Container built with:

```sh
larql convert gguf-to-vindex --keep-quant --dense-only --f16 \
  --level inference -o bitnet-dense.vindex ggml-model-i2_s.gguf
```

Producing 210 I2_S tensors, 30 layers, hidden 2560, vocab 128256,
`bitnet_layout` stamped, **0 gate layers** (so `is_dense_only()` is true).
`tokenizer.json` must be copied next to the GGUF first -- the converter
requires it and the GGUF repo does not ship it.

## Correctness

`/v1/infer` against the real model, three request shapes:

| request | top token | probability | mode |
|---|---|---|---|
| **no `mode` field** (what pg_infer sends) | `Paris` | 0.9494 | `bitnet` |
| `mode=dense` | `Paris` | 0.9494 | `bitnet` |
| `mode=walk` | `Paris` | 0.9494 | `bitnet` |

All three agreeing to 4dp is the result, not a formality: `mode` defaults
to `walk` when a client omits it, and this container has no gate vectors,
so without `is_dense_only()` coercion the first row would be garbage from
an empty KNN store. 0.9494 matches the 94.5% measured on the pre-rebase
tree.

Through SQL:

```
SELECT * FROM infer('The capital of France is', 3, 'bitnet');
  token  | probability | rank
--------+-------------+------
 ĠParis |      0.9494 |    1
```

(`Ġ` is GPT-2 byte-BPE for a leading space.)

## Is it fast enough to use in practice?

Yes for the graph surface, no for generation on x86 — and the first is what
pg_infer is for.

**The earlier decode numbers on this page measure the wrong thing.** They
measure `infer()`, which is 1 of pg_infer's 26 SQL functions, and the one
upstream explicitly says is *not* the point:

> Thesis: the differentiated functionality is the database, not the tok/s.
> [...] "query, edit, and interpret the model like a graph database" --
> `DESCRIBE`, `INSERT INTO EDGES`, `walk` -- is a genuine moat with **no
> competitor**. -- larql `ROADMAP.md`

The other 24 functions are graph queries. Measured against the same real
2B model, on a `--level browse` container (30 gate layers x 6912 features):

| query | median | rows |
|---|---|---|
| `infer_show_layers` | **28.5 ms** | 30 |
| `nearest_to` | **51.3 ms** | 5 |
| `infer_show_features` | **55.0 ms** | 10 |
| `describe` | **287.9 ms** | 20 |
| `infer_show_relations` | **444.3 ms** | 50 |
| `similar_to` | **638.6 ms** | 1 |
| `walk` | **676.6 ms** | 150 |
| `infer` (1 token) | 4757 ms | 5 |

Spreads under 2 ms on every graph query. Raw:
[`bench_graph.csv`](bench_graph.csv).

Every graph query is **28--677 ms**: usable inside a real query, not a
batch job. `describe()` at 288 ms is roughly 16x faster than a single
generated token on the same hardware, and `show_layers` at 28 ms is
interactive.

### Under concurrency

`describe()` from independent `psql` processes, A/B alternated, 3 reps.
Raw: [`bench_graph_conc.csv`](bench_graph_conc.csv).

| concurrency | qps | p50 | speedup |
|---|---|---|---|
| 1 | 3.43 | 0.291s | 1.00x |
| 2 | 7.37 | 0.270s | 2.15x |
| 4 | 15.18 | 0.260s | 4.43x |
| 8 | 29.86 | 0.263s | 8.71x |
| 16 | **41.39** | 0.268s | 12.07x |

**41 qps at c=16 with p50 flat at ~0.27s.** Super-linear through c=8
(8.71x on 8 workers) because the single-query path does not saturate a
core and the server's page cache warms; it falls off to 12.07x at c=16 as
32 vCPUs start to contend. Latency does not degrade -- p50 is *lower* at
c=16 than at c=1.

And the assertion that matters more than the throughput:

```
DISTINCT ANSWERS ACROSS 93 QUERIES: {'|Zone|332.80'}
```

One answer. A qps figure cannot tell you whether concurrency changed the
result; asserting a single answer across every level is what rules out a
race in the activation cache or the session overlay.

### So, practically

| use | verdict |
|---|---|
| Graph queries (`describe`, `walk`, `similar_to`, `nearest_to`, `show_*`) | **Yes.** 28--677 ms single, 41 qps at c=16, latency flat under load. |
| Generation (`infer`) on **arm64** | Plausible -- NEON kernels exist; unmeasured here. |
| Generation (`infer`) on **x86_64** | **No.** ~1 tok/s, scalar kernel. Use a served model. |

One caveat on *quality* rather than speed: `describe('France')` on this
model returns edges like `Zone`, `demographics`, `Barb` -- structurally
correct output (real gate scores, real layers) but weak semantics. BitNet
b1.58 is a heavily-quantised 2B; the feature-labelling heuristics were
tuned on Gemma-class models. **The mechanism works; do not read these
particular labels as a quality benchmark.** A Gemma 3 4B browse vindex
would be the fair test of output quality.

## Decode throughput

A/B alternated across cases, 3 reps, medians. Raw samples:
[`bench_bitnet.csv`](bench_bitnet.csv).

| case | median | spread | effective |
|---|---|---|---|
| `infer_short` (5-token prompt) | 4.757s | 0.008s | -- |
| `infer_long` (~40-token prompt) | 24.785s | 0.100s | -- |
| `gen_8tok` (SSE) | 11.629s | 2.690s | 0.69 tok/s |
| `gen_32tok` (SSE) | 32.661s | 0.773s | 0.98 tok/s |

**~1 tok/s is expected on x86_64, not a regression.**
`larql-compute`'s `ternary_matvec` has a NEON path under
`cfg(target_arch = "aarch64")` and no x86 SIMD equivalent, so x86_64 runs
the scalar kernel. The server's own log confirms it is not falling back
for some other reason (`q4k_matvec=avx2` is the *k-quant* kernel, a
different path). An AVX2/AVX-512 ternary kernel is the obvious follow-up.

Anyone comparing against `bitnet.cpp`'s published figures should note it
ships hand-written x86 kernels; this is a scalar-vs-SIMD gap, not an
algorithmic one.

## Concurrency qualification

45 SQL queries via independent `psql` processes (one backend and one
`RemoteBackend` runtime thread each -- a single session would serialise
and measure nothing). A/B alternated across concurrency levels. Raw:
[`bench_pg_infer.csv`](bench_pg_infer.csv).

| concurrency | wall span | p50 latency | qps | errors |
|---|---|---|---|---|
| 1 | 4.796s | 4.795s | 0.208 | 0 |
| 2 | 4.808s | 4.805s | 0.416 | 0 |
| 4 | 5.046s | 5.043s | 0.793 | 0 |
| 8 | 5.220s | 5.191s | 1.533 | 0 |

7.37x speedup at c=8 against an ideal 8x -- **92% scaling efficiency**,
with p50 rising only 4.80s -> 5.19s. Zero errors.

The load-bearing assertion is not the throughput, it is this:

```
DISTINCT ANSWERS ACROSS ALL 45 QUERIES: {('ĠParis', 0.9494)}
```

One distinct answer. A throughput number hides whether concurrency
changed the result; asserting a single answer across every level is what
rules out a shared-state race in the activation cache or the session
overlay. Had the lock discipline been wrong, this is where it would show.

## What real weights found that the fixture could not

Non-streaming `/v1/completions`, `/v1/chat/completions` and `/v1/responses`
returned:

```
503  "failed to load model weights: IO error: No such file or directory"
```

All three take `&mut ModelWeights` and so call `lock_weights_for_gen()`,
and a `--keep-quant` container has no dense weights to lock. The synthetic
fixture is a dense V2 container that *has* those files, so no unit test
could have surfaced it. Fixed by guarding `lock_weights_for_gen()` itself
-- one chokepoint every non-streaming path routes through -- with a message
naming the routes that do work.

The lesson generalises: a fixture that is a *simplified* version of the
real artifact cannot exercise code that branches on the difference.

## Other real-model confirmations

- eager ternary pre-load: `Pre-loaded BitNet model for 'bitnet2b' in 3.3s`
  (the ternary path; the dense path would have allocated ~5 GB)
- both SSE surfaces stream coherent text (`" Paris. Paris is a city that"`),
  exactly one `[DONE]`, `finish_reason: length`
- chat refuses `tools` with the intended message rather than answering
  with prose
- `/v1/runtime` reports `decode_tokens_per_second: 0.98` -- the
  `GenerationTally` wired during the port reaches the stats surface
  instead of reporting zero
- `/v1/capabilities`: schema 1, profile `single_model`, 42 routes, and it
  correctly reports `/v1/cache/stats` and `/v1/rank` as absent, so
  pg_infer skips those calls entirely
- `/v1/embed` on the real model: hidden_size 2560, non-zero L2
- `infer_warmup('bitnet')` -> "30 layers prefetched ... total 1607 ms"
  (real counters; this reported a hardcoded 0 before the wire fix)

## Reproducing

```sh
hf download microsoft/bitnet-b1.58-2B-4T-gguf --local-dir bitnet-gguf
hf download microsoft/bitnet-b1.58-2B-4T --include "tokenizer*" \
  --local-dir bitnet-base
cp bitnet-base/tokenizer.json bitnet-gguf/

larql convert gguf-to-vindex --keep-quant --dense-only --f16 \
  --level inference -o bitnet-dense.vindex bitnet-gguf/ggml-model-i2_s.gguf
larql-server bitnet-dense.vindex --port 28080

psql -c "SELECT infer_create_model_remote('bitnet','http://127.0.0.1:28080')"
psql -c "SELECT * FROM infer('The capital of France is', 3, 'bitnet')"
```

A `browse` container (needed for the graph queries) is a separate, slower
build -- it extracts gate vectors and runs feature/relation clustering:

```sh
larql convert gguf-to-vindex --level browse --f16 \
  -o bitnet-browse.vindex bitnet-gguf/ggml-model-i2_s.gguf   # ~45 min
```

`--keep-quant --dense-only` and `--level browse` are mutually exclusive in
practice: the first has no gate vectors (so no graph surface), the second
has no weights (so `infer()` returns 503 "vindex does not contain model
weights"). Serving both surfaces means two containers.

Scripts: `scripts/verify_bitnet.py`, `scripts/bench_bitnet.py`,
`scripts/bench_pg_infer.py`, `scripts/bench_graph.py`,
`scripts/bench_graph_conc.py`.
