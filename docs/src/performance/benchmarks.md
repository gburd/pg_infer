# Benchmarks

Detailed performance analysis and benchmark data for pg_infer.

## Operation Breakdown

**Model: Qwen 0.5B (24 layers, hidden=896, 4864 features/layer)**
**Vindex size: 480 MB (gate_vectors=200MB, embeddings=260MB)**
**Hardware: x86_64 Linux**

### Category 1: Negligible (< 0.1 ms)

| Operation | Time | Notes |
|-----------|------|-------|
| tokenization | 3-27 us | Scales linearly with tokens |
| embedding average | 0.1-4 us | Trivial vector math |
| score_to_distance | < 1 ns | Single division |

### Category 2: Moderate Cost

| Operation | Time | Notes |
|-----------|------|-------|
| model loading | 2327 ms | One-time per backend, mmap + tokenizer |
| gate mmap | 1698 ms | Kernel page-in on first access |
| embeddings mmap | 358 ms | |
| tokenizer JSON | 272 ms | |

### Category 3: Dominant Bottleneck

| Operation | Time | Notes |
|-----------|------|-------|
| gate KNN per layer | 6.4 ms | Brute-force dot product over 4864 features |
| gate KNN all layers | 154 ms | 24 * 6.4 ms |
| describe() | 176 ms | 1x full gate KNN + metadata lookup |
| similar_to() | 354 ms | 2x full gate KNN + overlap detection |
| scan per row | 138 ms | 1x row embed + 2x gate KNN |
| scan 100 rows | 13780 ms | Sequential, no parallelism |

### Where the Time Goes (per similar_to call)

```
tokenize A:         0.003 ms  ( 0.0%)
embed A:            0.001 ms  ( 0.0%)
tokenize B:         0.003 ms  ( 0.0%)
embed B:            0.001 ms  ( 0.0%)
gate KNN A, 24L:    154   ms  (43.5%)
gate KNN B, 24L:    154   ms  (43.5%)
overlap detection:   46   ms  (13.0%)
TOTAL:             ~354   ms
```

## Scaling Analysis

The ONLY significant performance bottleneck is gate_knn:

- 6.4 ms/layer * 24 layers * 2 texts = 307 ms per similarity comparison
- This is a brute-force O(features * hidden_size) dot product per layer:
  4864 features * 896 dims * 2 bytes (f16) = 8.7 MB scanned per layer
  24 layers = 209 MB scanned per query

### ORDER BY Scan Cost

For `ORDER BY <~>` scans with N rows:

```
Total cost = N * (embed_row + 2 * gate_knn_all_layers)
           = N * (0.001 + 2 * 154) ms
           = N * 308 ms

10 rows:    3.1 seconds
100 rows:  30.8 seconds
1000 rows: 308  seconds (5+ minutes)
```

## Qwen 2.5 3B Performance (36 layers, 2048 hidden)

### HNSW Impact

| Call | Time | What Happens |
|------|------|--------------|
| First similar_to() | 98s | HNSW build for all 36 layers (30s) + query (2s) + overhead |
| Second similar_to() | 1.9s | HNSW cached, only query |
| Third similar_to() | 1.9s | HNSW cached, only query |

**Improvement after first call: 51x faster** (98s -> 1.9s)

### Why Cached similar_to() Is Still 1.9s

The algorithm queries ALL 36 layers * 2 embeddings = 72 gate_knn calls:

- Each gate_knn with HNSW: ~13ms
- 72 calls * 13ms = 936ms baseline
- Additional overhead: tokenization, embedding, deduplication

## Remote Backend Benchmarks

Benchmarked with pgbench using `benches/schema.sql` (10k products + 8 bench
phrases):

```sh
pgbench -n -c 1  -T 30 -f benches/pgbench_similar_to.sql
pgbench -n -c 32 -T 60 -f benches/pgbench_similar_to.sql
```

### Results (Indicative)

| Workload | `backend='local'` | `backend='remote'` (UDS) |
|----------|-------------------|--------------------------|
| `similar_to` cold | ~1.9 s | 8-15 ms |
| `similar_to` warm | ~1.9 s | < 1 ms (L2 hit) |
| 20-row re-rank | ~37 s | ~20-40 ms (one fan-out) |
| 32 concurrent | OOM after ~50 | ~10k req/s sustained |

The "cold vs warm" split measures the server's L2 activation cache: once a
query phrase has been walked for any client, subsequent walks for the same
phrase from any pg_infer backend hit the cache.

## Registration Benchmarks

| Model Size | Index Build | Total Registration |
|------------|-------------|-------------------|
| Qwen 0.5B (480MB) | ~19.7s -> 546MB PG index | ~22s |
| Qwen 2.5 3B (2.2GB) | ~90.4s -> 4.3GB PG index | ~93s |

Scaling: ~50s/GB (I/O bound with memmap2). Writes all vindex data through
PostgreSQL's WAL system via GenericXLog.

## Memory Profile

### Qwen 0.5B (Local Mode)

| Component | Size |
|-----------|------|
| Vindex mmap | 480 MB (shared via page cache) |
| Gate cache (4 layers) | ~28 MB per backend |
| HNSW indexes (if enabled) | ~50 MB total |
| Feature metadata | ~15 MB |

### Qwen 2.5 3B (Local Mode)

| Component | Size |
|-----------|------|
| Vindex mmap | 2.2 GB (shared via page cache) |
| PG index | 4.3 GB |
| Gate cache (4 layers) | ~450 MB per backend |
| HNSW indexes (if enabled) | ~200 MB total |
| Feature metadata | ~50 MB |

## Validation Queries

```sql
-- Test 1: Registration time
\timing on
SELECT infer_create_model('test', 'qwen2.5-3b.vindex');

-- Test 2: First similar_to() (HNSW build)
SELECT similar_to('France', 'Paris');

-- Test 3: Cached similar_to()
SELECT similar_to('France', 'Paris');
SELECT similar_to('Einstein', 'physics');

-- Test 4: ORDER BY performance (10 rows)
CREATE TABLE test (id int, txt text);
-- ... insert data
SELECT * FROM test ORDER BY txt <~> 'search term' LIMIT 10;
```
