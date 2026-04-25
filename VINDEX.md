# VINDEX: Vector Index for Neural Network Knowledge

## What a Vindex Is

A **vindex** (vector index) is an index over a neural network's internal representations — specifically the Sparse Autoencoder (SAE) feature decomposition of each transformer layer. It is analogous to a B-tree index over table rows: where a B-tree reorganizes column values for efficient lookup, a vindex reorganizes gate vectors for efficient KNN queries against model knowledge.

A vindex is NOT a compression or summarization of the model. It is a reorganization of specific internal structures into a format optimized for direct lookup queries.

### The Database-Index Analogy

| Concept | B-tree Index | Vindex |
|---------|-------------|--------|
| **Source data** | Table rows | Transformer model weights |
| **Index keys** | Column values, sorted | Gate vectors (one per SAE feature per layer), organized for KNN |
| **Lookup operation** | Binary search on key | Matrix-vector multiply: `gate_vectors × embedding` → top-K features |
| **Leaf data** | TID pointers to heap tuples | Per-feature metadata: top output tokens, activation patterns |
| **What it enables** | Fast WHERE/ORDER BY on indexed columns | Fast "what does this model know about X?" queries |
| **What it can't do** | Full table scan without heap access | Run inference (generate text) without the full model weights |

## Directory Structure (File-Based Vindex)

At **browse level** (Qwen 0.5B example, ~480MB total):

```
qwen05b.vindex/
├── gate_vectors.bin   (200MB)  — Index keys. One f16 vector per SAE feature per layer.
│                                 gate_knn() multiplies input × these vectors → top-K features.
├── embeddings.bin     (260MB)  — Token embeddings from the model's embedding layer.
│                                 Converts text → vector for queries.
├── down_meta.bin      (9.8MB)  — Per-feature metadata: which output tokens each feature
│                                 maps to. The "leaf data" of the index.
├── tokenizer.json     (11MB)   — Tokenizer for text ↔ token ID conversion.
└── index.json         (3.8KB)  — Config: num_layers, hidden_size, vocab_size, quant format.
```

## Extract Levels

| Level | Contents | Enables | Size (Qwen 0.5B) |
|-------|----------|---------|-------------------|
| **browse** | SAE gate vectors + embeddings + down metadata + tokenizer | walk(), describe(), similar_to(), implies() | ~480 MB |
| **inference** | Browse + FFN weights + attention weights | All browse ops + infer() (forward pass) | ~1.5 GB |
| **all** | Full model reconstructable | Everything, including fine-tuning | ~2+ GB |

Browse is the "index-only scan" — you can query what the model knows without the full model. Inference is the "covering index" — answers queries without the original. All is the "materialized view."

## Why Vindexes Exist

A raw transformer model is a giant tensor of floats. To find "what does this model know about France?", you'd need to either:
1. Run inference many times with different prompts (slow, indirect, non-deterministic)
2. Manually inspect billions of parameters (impossible)

LARQL's insight: transformer models learn **sparse features** during training. A Sparse Autoencoder (SAE) decomposes each layer's residual stream into a dictionary of discrete features, each with:
- A **gate vector** (how to detect if this feature activates for a given input)
- A **down vector** (what output tokens this feature produces)

The vindex reorganizes gate vectors into a structure where `gate_knn()` can answer "which features activate for this input?" via a single matrix-vector multiply (~0.3ms per layer). This is the core index operation.

## PostgreSQL Page-Based Binary Format

When stored as a PostgreSQL index via `CREATE INDEX ... USING infer`, vindex data is reorganized into standard 8KB PostgreSQL pages. This enables WAL logging, buffer management, and crash recovery.

### Page Layout

Every page has a standard 24-byte `PageHeader` and a 16-byte special section (`InferPageOpaque`) at the tail:

```
[PageHeader 24B] [... usable data 8152B ...] [InferPageOpaque 16B]
```

Usable data per page: 8192 - 24 (header) - 16 (opaque) = **8,152 bytes**.

### InferPageOpaque (16 bytes, at pd_special)

```rust
#[repr(C)]
struct InferPageOpaque {
    page_type: u8,       // META=1, LAYER_DIR=2, GATE=3, EMBED=4, DOWN_META=5, BLOB=6
    flags: u8,           // compression, quantization indicators
    layer_id: u16,       // 0xFFFF for non-layer pages
    next_blkno: u32,     // chain for blob pages (InvalidBlockNumber if none)
    reserved: [u8; 8],
}
```

### Page Types

#### Metapage (block 0)

Contains model metadata: magic number (`0x494E4652` = "INFR"), format version, model dimensions, data type info, and block ranges for each section.

#### Layer Directory (block 1)

Maps layer IDs to their gate and down block ranges. 20 bytes per entry, supports 407 layers per page.

#### Gate Vector Pages

Store f16 gate vectors. 4 vectors per page for hidden_size=896 (Qwen 0.5B: 1,792 bytes/vector).

#### Embedding Pages

Same layout as gate pages. Token ID → page is O(1): `embed_start_blk + token_id / 4`.

#### Down Meta Pages

Per-feature metadata records. 92 records per page (88 bytes each: top_token_id + c_score + 10 × (token_id + logit)).

#### Tokenizer Blob Pages

Tokenizer JSON split into 8,128-byte chunks, chained via `next_blkno`.

### Size Estimate (Qwen 0.5B, f16 browse)

| Section | Pages | Size |
|---------|-------|------|
| Meta + layer dir | 2 | 16 KB |
| Gate vectors | 29,184 | 228 MB |
| Embeddings | 37,984 | 297 MB |
| Down metadata | 1,269 | 9.9 MB |
| Tokenizer | ~1,406 | 11 MB |
| **Total** | **~69,845** | **~546 MB** |

13.7% overhead vs raw files (480 MB). Comparable to normal PostgreSQL index overhead.

## Relationship to the Index Access Method

The `infer` index AM stores vindex data in PostgreSQL pages:

```sql
-- Build a vindex as a PostgreSQL index
CREATE INDEX qwen05b ON infer._models USING infer (name)
    WITH (source = '/data/qwen05b.vindex');

-- Or via the convenience function (internally creates the index)
SELECT infer_create_model('qwen05b', '/data/qwen05b.vindex');
```

Benefits of page-based storage:
- **WAL-logged**: crash recovery, replication
- **Buffer-managed**: shared_buffers caching, OS page cache
- **pg_basebackup compatible**: indexes travel with the cluster
- **VACUUM/ANALYZE aware**: standard maintenance
- **Per-backend caching**: decoded f32 gate layers cached in process memory after first read

### Gate KNN Access Strategy

For each layer query:
1. Allocate f32 buffer: `features_per_layer × hidden_size × 4` bytes
2. Read gate pages sequentially, decode f16 → f32 into contiguous buffer
3. Single BLAS GEMV on the contiguous buffer
4. Top-K selection

After first query, gate pages reside in shared_buffers and subsequent reads are fast. A per-backend LRU cache of decoded f32 layers avoids repeated page reads + decode.
