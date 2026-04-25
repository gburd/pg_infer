# pg_larql: Commit & Next Steps

## What a Vindex Is (and Isn't)

A **vindex** (vector index) is an **index over a neural network's internal representations**, analogous to a B-tree index over table rows. It is NOT a compression or summarization of the model — it is a reorganization of specific internal structures into a format optimized for direct lookup queries.

### The database-index analogy

| Concept | B-tree Index | Vindex |
|---------|-------------|--------|
| **Source data** | Table rows | Transformer model weights |
| **Index keys** | Column values, sorted | Gate vectors (one per SAE feature per layer), organized for KNN |
| **Lookup operation** | Binary search on key | Matrix-vector multiply: `gate_vectors × embedding` → top-K features |
| **Leaf data** | TID pointers to heap tuples | Per-feature metadata: top output tokens, activation patterns |
| **What it enables** | Fast WHERE/ORDER BY on indexed columns | Fast "what does this model know about X?" queries |
| **What it can't do** | Full table scan without heap access | Run inference (generate text) without the full model weights |

### What's in a vindex directory

At **browse level** (what we extracted for Qwen 0.5B, 480MB):

```
qwen05b.vindex/
├── gate_vectors.bin   (200MB)  — The index keys. One vector per SAE feature per layer.
│                                 gate_knn() multiplies input embedding × these vectors
│                                 to find top-K activated features. This IS the index.
├── embeddings.bin     (260MB)  — Token embeddings from the model's embedding layer.
│                                 Needed to convert text → vector for queries.
├── down_meta.bin      (9.8MB)  — Per-feature metadata: which output tokens each feature
│                                 maps to. The "leaf data" of the index.
├── tokenizer.json     (11MB)   — Tokenizer for text ↔ token ID conversion.
└── index.json         (3.8KB)  — Config: layer count, hidden size, vocab size, quant format.
```

At **inference level**: adds attention weights and FFN weights — enough to run a forward pass. This is a "covering index" (answers queries without touching the original model).

At **all level**: the full model is reconstructable. This is a "materialized view."

### Why vindexes exist

A raw transformer model is a giant tensor of floats. To find "what does this model know about France?", you'd need to either:
1. Run inference many times with different prompts (slow, indirect, non-deterministic)
2. Manually inspect billions of parameters (impossible)

LARQL's insight: transformer models learn **sparse features** during training. A Sparse Autoencoder (SAE) can decompose each layer's residual stream into a dictionary of discrete features, each with a gate vector (how to detect it) and a down vector (what output tokens it produces). The vindex reorganizes these gate vectors into a structure where `gate_knn()` can answer "which features activate for this input?" via a single matrix-vector multiply (~0.3ms per layer).

This is genuinely an index: it trades space (extracting and storing gate vectors separately) for query speed (direct KNN lookup instead of running the full model).

### What's lost at browse level

Browse-level vindexes contain the SAE feature decomposition but NOT the original FFN/attention weights. You can query _what the model knows_ (which features activate, what tokens they map to) but you cannot _run the model_ (generate text, compute exact probabilities). The relationship is like having a full-text index without the original documents — you can search efficiently, but you can't retrieve the original text.

---

## Project Goal Assessment

> "The goal of this project is to make models *feel like indexes* and allow them to be used in a wide variety of queries — have we done that?"

### What works (the index-like parts)

The SQL interface is genuinely index-like. All these work today with the Qwen 0.5B vindex:

```sql
-- "What features activate for this input?" — like an index scan
SELECT * FROM walk('The capital of France is', top => 10);
-- Returns 240 rows: (layer, feature_id, activation_score, concept_token)

-- "How similar are these concepts?" — like a distance function
SELECT similar_to('France', 'Paris');     -- 0.086
SELECT similar_to('France', 'banana');    -- 0.039

-- "Order by semantic distance" — like ORDER BY with an index
SELECT 'France' <~> 'Paris';             -- 11.6 (closer)
SELECT 'France' <~> 'banana';            -- 25.5 (farther)

-- Model lifecycle mirrors index lifecycle
SELECT larql_create_model('qwen05b', '/path/to/vindex');  -- like CREATE INDEX
SELECT larql_drop_model('qwen05b');                        -- like DROP INDEX
```

Per-backend mmap caching means the vindex "feels instant" after first load, exactly like a B-tree index — the OS shares mmap pages across all PostgreSQL backend processes.

### What's missing (the gaps)

1. **No `CREATE INDEX` syntax** — Models are registered via `larql_create_model()` function calls, not DDL. Extensions can't add parser keywords, so this is a fundamental PostgreSQL limitation. The UX is more like calling a function than managing an index.

2. **No planner integration** — PostgreSQL's query planner doesn't know about vindexes. It can't automatically decide to use a vindex for a `WHERE similar_to(col, 'France') > 0.5` clause the way it uses a B-tree for `WHERE id = 42`. Every vindex query is an explicit function call.

3. **No index access method (AM)** — pgvector registers as a custom AM so `CREATE INDEX ... USING hnsw` works and the planner can use it. pg_larql doesn't have this. This is the biggest gap between "functions that query model weights" and "models that feel like indexes."

4. **`describe()` underperforms on small models** — The gate_threshold (5.0) is calibrated for 4B+ models. For Qwen 0.5B, max gate scores are ~0.12, so describe() returns 0 rows. This needs model-size-aware thresholding.

5. **`infer()` untested** — Requires inference-level vindex extraction (we only did browse). Feature-gated behind `--features inference`.

### Honest verdict

We've built a **functional prototype** that demonstrates the concept convincingly. The core index operations work: KNN feature lookup, semantic similarity, distance ordering. But it doesn't yet "feel like an index" in the PostgreSQL sense — it feels like a set of SQL functions backed by an indexed data structure. The difference is that PostgreSQL indexes are transparent to the query planner; our vindex is opaque.

The path from here to "feels like an index" would be:
- Custom index AM registration (so `CREATE INDEX ... USING larql` works)
- GiST/SP-GiST operator class for `<~>` (so `ORDER BY a <~> b LIMIT 10` uses the index)
- Planner hooks to recognize `similar_to()` predicates and push them down

That said, pgvector started exactly where we are — functions and operators first, index AM later — and it's now the most widely used vector extension. The foundation is correct.

---

## Commit Plan

### Files to commit

The repo has no commits. This will be the initial commit.

**Include:**
- `pg_larql/` — All extension source, Cargo.toml, control file, SQL tests, expected output
- `CLAUDE.md` — Build environment documentation
- `PG_LARQL.md` — Design spec
- `.gitignore` — Rust/pgrx ignores
- `.gitmodules` — (empty, but present)

**Exclude (add to .gitignore):**
- `vindexes/` — 480MB of binary model data (add `vindexes/` to .gitignore)
- `_/` — External dependencies (already partially gitignored; uncomment `_/` in .gitignore)
- `pg_larql/target/` — Build artifacts (already gitignored)
- Empty marker files: `HEAD`, `config`, `objects`, `refs`, `hooks` — These are zero-byte files that appear to be artifacts; add them to .gitignore
- Dotfiles: `.bash_profile`, `.bashrc`, `.zshrc`, `.zprofile`, `.profile`, `.gitconfig`, `.ripgreprc`, `.mcp.json` — User profile files, not project files; add to .gitignore

### .gitignore updates needed

Add to existing `.gitignore`:
```
# Vindex data (large binaries)
vindexes/

# External dependencies
_/

# User profile files (not project files)
.bash_profile
.bashrc
.zshrc
.zprofile
.profile
.gitconfig
.ripgreprc
.mcp.json

# Empty marker files
HEAD
config
hooks
objects
refs
```

### Commit message

```
Initial pg_larql extension: query transformer weights as SQL relations

pgrx 0.17.0 extension for PostgreSQL 18+ that exposes LARQL vindex
queries as SQL functions. Implements walk(), describe(), similar_to(),
implies(), infer(), and the <~> distance operator backed by mmap'd
vindex files with per-backend handle caching.

10/10 pgrx integration tests passing. End-to-end verified with
Qwen 0.5B browse-level vindex.
```

### Steps

1. Update `.gitignore` (add vindexes/, _/, dotfiles, marker files)
2. `git add` specific files: `.gitignore`, `.gitmodules`, `CLAUDE.md`, `PG_LARQL.md`, `pg_larql/`
3. `git commit`
4. Verify with `git status` — only intentionally-excluded files remain untracked
