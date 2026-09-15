# Upstream sync plan: larql → pg_infer

Status: plan only, nothing applied. Written 2026-09-14.

Fork: **`github.com/gburd/larql`** (fork of `chrishayuk/larql`, last push 2026-07-08).
Upstream: **`github.com/chrishayuk/larql`**.

## 0. Where the fork actually stands

| | |
|---|---|
| `fork/main` ahead of upstream | **41 commits** (26 non-merge) |
| `fork/main` behind upstream | **1255 commits** |
| Merge base | `ab7a08c1` (2026-06-22) |

`git cherry` reports all 26 as unique (`+`), but that is misleading — it compares
patch-ids, and squash-merges change them. Checking by *content* instead:

### Most of your work already landed upstream

Ten commits authored `Greg Burd <greg@burd.me>` are in `origin/main`:

```
b5b892e0 2026-06-17 feat(bitnet): native-ternary 1.58 inference end-to-end
81421773 2026-06-17 fix(ggml): correct I2_S decode to microsoft strided block layout
9467ae11 2026-05-25 feat(ggml): add I2_S decoder for Microsoft bitnet.cpp GGUFs
ebc50c34 2026-05-25 feat(ggml): TQ1_0 / TQ2_0 ternary quantisation for BitNet 1.58
5c097022 2026-05-25 fix(server): infer deadlock + concurrent-load OOM (P0)
a5a3fcc2 2026-05-25 feat(server): startup memory pre-flight (§5.5)
660e6afb 2026-05-25 feat(server): inference timeout + 504 response (S5.6)
1bf2ff02 2026-06-19 review: document BitNet engine-divergence, dual I2_S layout
60c7faff 2026-06-19 review: fix stale I2_S header doc, add full-block tests
f5d273f0 2026-06-19 review: skip memcheck for quantized vindexes
```

Upstream now carries the whole ternary stack under your authorship:
`crates/larql-inference/src/ternary/{predict,kv_cache,streaming,ffn,load}.rs`,
`larql-compute/src/cpu/ops/ternary_matvec.rs`,
`larql-vindex/src/extract/{bitnet_loader,bitnet_writer}.rs`,
`larql-models/src/architectures/bitnet.rs`, plus the test modules.

So ~20 of your 26 commits are **already upstream** and need no porting. The
BitNet work was the bulk of the fork and it is home.

### `fork/main` does not compile

Merge `918a3978 Merge remote-tracking branch 'upstream/main'` silently dropped
`is_bitnet()` / `get_or_load_bitnet()` from `crates/larql-server/src/state.rs`:

```
is_bitnet in 918a3978^1 (pre-merge):  1
is_bitnet in 918a3978  (post-merge):  0
```

Two callers survive the merge and reference the now-missing method:
- `crates/larql-server/src/routes/openai/chat.rs:461` — `if model.is_bitnet() {`
- `crates/larql-server/src/routes/openai/completions.rs:309` — same

`f11d8a16 fix(merge): drop duplicate ... mod decls` cleaned up the `E0428`s from
the same bad merge but missed this. The definition still exists on
`fork/feat/bitnet-streaming-walk` (`state.rs`, 1 hit).

**Consequence: `fork/main` is not a usable base and its state is not a reliable
record of what you built.** Recover from the feature branches, not from `main`.

### What is genuinely still yours

Exactly one thing survives as unique content — `git diff --diff-filter=A` from
the merge base gives 7 added files, all one crate, none present upstream:

**`crates/larql-cloud`** (~78 KB) — outbound LLM clients:

| File | Size |
|---|---|
| `src/bedrock.rs` | 24 KB |
| `src/openai.rs` | 18 KB |
| `src/bin/proxy.rs` | 15 KB |
| `tests/proxy_integration.rs` | 9 KB |
| `src/{lib,types}.rs`, `Cargo.toml` | 11 KB |

Its own doc comment says who it is for:

> The single trait `CloudClient` exposes the three operations **pg_infer** needs
> to delegate to a cloud LLM: `infer`, `embed`, `chat`. […] `larql-server`
> accepts a `--proxy <provider>` flag […] pg_infer hits `/v1/infer` /
> `/v1/embeddings` / `/v1/chat/completions` and never knows whether the back
> end was a local vindex or a remote API.

Providers: OpenAI, Exoscale, Together, Fireworks, vLLM, llama.cpp, ollama,
AWS Bedrock (incl. `c4bd1b20 fix(cloud/bedrock): recognise cross-region
inference profile model ids`).

Plus modifications to 11 files upstream still has (`session.rs`, `insert.rs`,
`openai/{chat,completions}.rs`, `patch/overlay.rs`, `nix/package.nix`, …) —
triaged in §1.

## 1. Triage of the modifications

### Superseded — drop, do not port

**`session.rs` lock-release wedge** (`bfa2f2b7 fix(server): release outer locks
across walk`). Your fix made `SessionState.patched` an
`Arc<RwLock<PatchedVindex>>` so callers snapshot the inner `Arc` and drop the
outer `sessions` lock across a multi-second walk.

Upstream refactored `session.rs` into a module
(`session/{mod,manager,state,lease,clock}.rs`) and solved the same problem
differently — `SessionState` now holds `overlay: Option<PatchedVindex>` +
`Arc<SessionLease>`, and `manager.rs::get_or_create` returns a *copy*:

> The returned overlay is a *copy* — the session's patches replayed onto a fresh
> clone of the model's base — so the caller can read it **without holding any
> session lock**.

`apply_patch` documents the same discipline you were fixing ("never holds the
`sessions` write guard while awaiting … an earlier implementation called
`model.patched.blocking_read()` from inside `or_insert_with`"). Different shape,
same invariant. **Upstream's is better** — it also handles the KV-cache
resurrection race via `SessionLease::is_alive`, which yours does not.

`insert.rs` and `patch/overlay.rs`'s `impl Clone for PatchedVindex` exist only
to serve your `Arc<RwLock<…>>` shape. Both go with it. (Upstream has no
`impl Clone for PatchedVindex`; if the copy path needs one, add it on a clean
base as its own change.)

### Rebase, don't port

`nix/package.nix`, `nix/patches/use-system-protoc.patch`, `flake.lock`,
`c3da3838 fix(nix): repair nix build .#larql`. Upstream's Nix tree moved a lot.
Re-derive against current `origin/main`; a 142-line patch-file diff will not
apply.

`990537cc deps: cargo update` — discard, 1255 commits stale.

### Salvage only if the branch tests prove it's needed

`openai/{chat,completions}.rs` BitNet SSE streaming (`f43c07ef`, `a29d7b26`)
and `af4aacc9 force dense mode for dense-only BitNet vindexes`. Upstream **has**
`ternary::{generate_streaming_bitnet, infer_bitnet_walk}` exported
(`ternary/mod.rs:134`) but **nothing in `larql-server` calls them** — only
`larql-cli`'s `convert_cmd.rs` / `run_cmd.rs` do:

```
$ git grep -ln "ternary" origin/main -- crates/larql-server/src
(no output)
```

So the server-side BitNet serving path is **the one real gap**: the engine
landed, the HTTP wiring did not. That is the piece worth re-proposing — but
`fork/main`'s version is exactly the broken code from §0, so port from
`fork/feat/bitnet-streaming-walk` (which still has `is_bitnet`), rebased.

## 2. Recommended fork procedure

Do not merge `fork/main` forward. It is 1255 behind, contains a broken merge, and
~20 of its commits are already upstream — a merge would re-litigate all of it.

```sh
git clone https://github.com/gburd/larql ~/ws/larql
cd ~/ws/larql
git remote add upstream https://github.com/chrishayuk/larql
git fetch upstream

# Keep the broken state for reference, then reset to upstream.
git branch archive/main-pre-resync main
git checkout -B main upstream/main

# Replay only what survives triage, each on clean upstream.
git checkout -b feat/larql-cloud   # cherry-pick 38ff8d50 2be34af7 c4bd1b20
git checkout -b feat/bitnet-serve  # from feat/bitnet-streaming-walk, WITH state.rs
```

Two focused branches, ~28 commits retired. Verify with `cargo build -p
larql-server` — `fork/main` fails this today.

Both are upstreamable: `larql-cloud` is additive and self-contained;
`bitnet-serve` completes work upstream already took. Getting `larql-cloud`
upstream matters most — pg_infer's cloud story otherwise depends on a private
fork forever.

## 3. The gap that affects pg_infer

pg_infer pins `c880fb7` (2026-05-12) in `docs/src/compatibility/upstream.md`.
Upstream HEAD is `23a56db1` (2026-09-13) — **1495 commits**, 6 new crates
(`larql-execution`, `larql-factory`, `larql-vindex-spec`, `larql-compute-metal`,
`larql-demos`, `vindex-cli`).

pg_infer's pin is older than the fork's merge base, so the fork question and the
pg_infer question are largely independent. **The bugs below are in pg_infer
today, regardless of what happens to the fork.**

### 3.1 `/v1/relations` — hard parse failure (P0)

```rust
// crates/infer-client/src/types.rs:95
pub struct RelationSummary {
    pub token: String,   // ← no serde(default) ⇒ REQUIRED
    pub count: usize,
    pub layers: Vec<usize>,
}
```

Server emits (`larql-server/src/routes/relations.rs:226`, **identical at the
pin**):

```json
{"name": "capital", "count": 42, "max_score": 44.1,
 "min_layer": 14, "max_layer": 17, "examples": [...]}
```

`token` vs `name` ⇒ missing required field ⇒ **every `show_relations()` against a
real server errors.** And `layers` silently defaults to `[]` because the server
sends `min_layer`/`max_layer`. Never worked.

Fix: `#[serde(rename = "name", alias = "token")]`; derive `layers` from the pair.

### 3.2 `/v1/models` — grid discovery cannot work (P0)

**Wrong envelope.** `src/backend/grid.rs:58` expects `{"models": [...]}` with
`#[serde(default)]`; server and router emit `{"object": "list", "data": [...]}`.
Defaults to empty ⇒ "no servers for this model" — a *misleading error, not a
parse error*, which is why it went unnoticed.

**Deeper: the router does not serve what pg_infer needs.** `ModelEntry` wants a
per-model `url`/`server` to round-robin. The router's `/v1/models`
(`larql-router/src/openai/mod.rs:157`) emits `{id, object, created, owned_by}` —
**no URL, deliberately.** The router is a *proxy*: it fans out `/v1/walk-ffn`
itself and exposes no shard directory.

Not a shape bug — an architecture mismatch. Fix: point one `RemoteBackend` at the
router and let it proxy. Deletes most of `grid.rs`; the router already
load-balances. Add a `ponytail:` note if per-shard affinity ever matters.

### 3.3 Endpoints pg_infer calls that upstream never had (P1)

| pg_infer calls | At pin | At HEAD |
|---|---|---|
| `/v1/cache/stats` | absent | absent |
| `/v1/rank` | absent | absent |
| `/v1/features` | absent | absent (comment only) |
| `/v1/layers` | absent | absent (comment only) |

The 404 fallbacks are fine; the docs are not — `upstream.md` implies these are
`/v1` contract. They are pg_infer inventions. Mark them local or propose them.

`/v1/cache/stats` now has a real replacement: `GET /v1/stats` carries a `server`
block (`v3_kv.{hits,misses,entries,capacity,ttl_secs}`, `uptime_secs`,
`requests_served`).

### 3.4 `/v1/warmup` — response shape diverged (P1)

pg_infer wants `{warmed, already_cached, latency_ms}`, all `serde(default)`, so
it parses and returns `(0, 0)`. Upstream emits `{model, weights_loaded,
weights_load_ms, layers_prefetched, prefetch_ms, experts_prefetched,
expert_prefetch_ms, hnsw_built, hnsw_warmup_ms, total_ms}`. **Warmup silently
reports zero work forever.** Request also changed: `{entities}` →
`{layers: Option<Vec<usize>>, skip_weights: bool}`.

### 3.5 Why CI passes anyway (P0 — the root cause)

`crates/infer-client/tests/mock_server_integration.rs` hand-writes the JSON
pg_infer *wishes for*, so it tests pg_infer against itself:

```rust
("GET", "/v1/relations") => serde_json::json!({
    "relations": [{"token": "capital", ...}]   // server says "name"
}),
```

The mock is why four wire bugs shipped. `/v1/models`, `/v1/warmup`,
`/v1/cache/stats`, `/v1/rank` have **no mock arm and no test at all**.

Fix: upstream serves `GET /v1/openapi.json` (utoipa). Fetch it in CI and assert
`infer-client`'s DTOs deserialize the documented schemas — contract testing
against the server's own description of itself.

## 4. What does not break: the vindex format

Bounds the whole effort. pg_infer's `VindexConfig`
(`crates/infer-vindex/src/config/types.rs`) still matches upstream's V2
`index.json` (`larql-vindex/src/config/index.rs`) field-for-field; required
fields **identical**: `version, model, family, hidden_size, intermediate_size,
vocab_size, embed_scale, down_top_k`.

Three upstream V2 fields pg_infer lacks. No `deny_unknown_fields`, so they're
ignored — but two change how bytes are *interpreted*:

| Field | Risk |
|---|---|
| `fp4: Option<Fp4Config>` | **Silent misread.** `block_elements`, `sub_block_elements`, scale dtypes, per-projection precision. Upstream: readers "MUST dispatch on this tag and MUST NOT sniff filenames." |
| `bitnet_layout: Option<BitnetLayout>` | Same class: `rms_eps`, `head_dim`, `n_q_heads`, scale packing. **Written by your own `bitnet_writer.rs`** — a `--keep-quant` vindex you produce is exactly what pg_infer would misread. |
| `ffn_layout: Option<FfnLayout>` | Only variant `PerLayer`. Cosmetic. |

Also: upstream accepts `#[serde(alias = "kquant")]` on `QuantFormat::Q4K`;
pg_infer does not, so a vindex with the new canonical tag fails to load.
One line.

**VINDEX3 is a separate generation, not a migration** — schema 3–4, own
`Vindex3Index`, LYRW v2, system graph, attestations. V2 stays first-class
(`V2_MIN_SCHEMA=1..=2`, `detect_generation()` dispatches). **Stay on V2.** Add a
guard so a V3 directory gets a clear refusal, not a confusing parse error.

## 5. New upstream features, triaged

~40 new endpoints. Ranked by value per line of work:

### Worth taking

| Feature | Why |
|---|---|
| **`GET /v1/capabilities`** | Best thing in 1495 commits *for pg_infer*. Server reports which routes it mounts, derived from the router's own mount ledger (cannot drift). Replaces all five 404-probe-and-guess paths with one query at registration. |
| **`GET /v1/stats` → `server` block** | Real `/v1/cache/stats` replacement (§3.3). |
| **`POST /v1/embed`** | `src/backend/remote.rs:551` returns `RemoteUnsupported("embed (not wired yet)")` — it has existed upstream all along. JSON `{token_ids}` or binary `application/x-larql-ffn`. |
| **`POST /v1/select`** | Server-side filtered edge selection (`entity, relation, layer, limit, min_confidence, order_by, order`). Predicate pushdown — natural for a SQL extension. |
| **`larql-cloud --proxy`** (yours) | The crate was written for this. Gives pg_infer cloud-LLM backends through the same `/v1` surface, no pg_infer changes. Blocked on §2 landing it. |

### Later

`/v1/explain-infer` (attribution → `infer_explain()` SRF), `/v1/patches/*`
(already a logged divergence; upstream now has full CRUD), `/v1/logits`,
`/v1/token/{encode,decode}`.

### Skip

`/v1/{chat/completions,completions,embeddings,responses}` (OpenAI surface — not
a database's job, and `larql-cloud` covers the outbound case),
`/v1/query` (LQL; pg_infer's thesis is native SQL), `/v1/experts/*` +
`/v1/walk-ffn*` (grid-internal), `/v1/{plan,components,representations,provenance,authority,runtime/*}`
(VINDEX3 control plane), `/v1/{sessions,stream,shard}`.

## 6. Sequenced plan

Fork and pg_infer tracks are independent; run them in parallel.

### Fork track

- **F1.** Clone, add `upstream`, archive `main`, reset to `upstream/main` (§2).
- **F2.** `feat/larql-cloud` — cherry-pick `38ff8d50 2be34af7 c4bd1b20` onto clean
  upstream. Verify `cargo test -p larql-cloud`. Open upstream PR.
- **F3.** `feat/bitnet-serve` — port server-side BitNet serving from
  `feat/bitnet-streaming-walk`, **including the `state.rs` `is_bitnet` /
  `get_or_load_bitnet` that `fork/main` lost.** Gate on `cargo build -p
  larql-server`.
- **F4.** Re-derive the Nix changes against current upstream. Drop
  `session.rs` / `insert.rs` / `overlay.rs` / `cargo update` entirely.

### pg_infer track

- **P1 — make the tests capable of failing (blocks all else).** CI job: build
  `larql-server`, fetch `/v1/openapi.json`, assert every `infer-client` DTO
  deserializes the documented schema. Add arms for the four untested endpoints.
  **Confirm the new test fails on `/v1/relations` before fixing it** — if it
  doesn't, the harness is still lying. *Deliverable: red CI.*

- **P2 — fix the wire bugs** (each now has a test):

  | Fix | File |
  |---|---|
  | `RelationSummary`: `rename="name"`, derive `layers` | `crates/infer-client/src/types.rs:95` |
  | `/v1/models`: OpenAI `{object, data}` envelope | `src/backend/grid.rs:58` |
  | Grid: router as one endpoint, delete discovery | `src/backend/grid.rs` |
  | `WarmupResponse` + request `{layers, skip_weights}` | `types.rs:139`, `remote.rs:157` |
  | `QuantFormat`: `serde(alias="kquant")` | `crates/infer-vindex/src/config/types.rs:145` |
  | Reject VINDEX3 dirs clearly | `crates/infer-vindex/src/format/load.rs:41` |

- **P3 — capabilities handshake.** Query `/v1/capabilities` at registration, cache
  on the backend. Replace the five `msg.contains("404")` string-matches —
  matching on error text is fragile independent of all this.

- **P4 — honest vindex handling.** Parse `fp4` / `bitnet_layout` and **refuse**
  vindexes carrying them (§4). Refusal is cheap; wrong dequantization is a silent
  wrong answer — and your own `bitnet_writer` emits `bitnet_layout`.

- **P5 — new capability**, gated on P3. `/v1/embed` → remote `embed()`.
  `/v1/stats.server` → `infer_cache_stats()`. `/v1/select` → predicate pushdown.

- **P6 — docs.** Re-pin to `23a56db1`. Mark `/v1/{cache/stats,rank,features,layers}`
  pg_infer-local. Note V2-only, VINDEX3 out of scope. Fix the fork URLs: docs say
  `codeberg.org/gregburd/larql` (`deployment/overview.md:19`,
  `introduction.md:49`) — that repo **does not exist**; it's `github.com/gburd/larql`.

## 7. Unrelated finding

`Cargo.toml` and `pg_infer.control` both say `1.0.0`; HEAD is *"Classify project
as experimental (v0.1.0-alpha)"*. Pick one — the docs currently promise a
stability guarantee (§"Breaking Change Policy") the code doesn't intend to keep.

## 8. Effort

| Step | Value |
|---|---|
| F1 + F3 | **Highest.** `fork/main` doesn't compile; the fix is one lost `state.rs` hunk. |
| P1 | **Highest.** Without real contract tests nothing else is verified. |
| P2 | **Highest.** Three live bugs, two never worked. |
| F2 | High. `larql-cloud` is the only unique thing left; upstream it or carry a fork forever. |
| P3 | High. Deletes fragile string-matching. |
| P4 | Medium-high. Prevents silent wrong answers on your own vindexes. |
| P5 | Medium. Real features, nothing broken without them. |
| F4, P6 | Low effort, stops misinforming readers. |

The fork is in better shape than it looks: most of it shipped upstream under your
name. What's left is one crate to upstream, one lost hunk to restore, and a set
of pg_infer wire bugs that were never about the fork at all.
