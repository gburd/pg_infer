# Optimization

Strategies for improving pg_infer query performance.

## Performance Bottleneck

The dominant bottleneck in pg_infer is the **gate KNN** operation: a brute-force
dot product over all features in each layer. For a 3B-parameter model:

- 36 layers * 2 embeddings = 72 gate_knn calls per `similar_to()`
- Each call: ~4864 features * 2048 dimensions = ~10M FLOPs
- Total: ~720M FLOPs per similarity computation

## Solution 1: Eager HNSW Build (High Priority)

**Problem:** First `similar_to()` call triggers HNSW build for ALL layers,
taking 30+ seconds.

**Fix:** Build HNSW indexes during `infer_create_model()` instead of lazy init.

```sql
SET infer.build_hnsw_on_load = true;
SELECT infer_create_model('model', 'path.vindex');
```

**Impact:**

| Scenario | Before | After |
|----------|--------|-------|
| Registration | 90s | 120s (+30s) |
| First similar_to() | 98s | 1.9s |
| Cached similar_to() | 1.9s | 1.9s (unchanged) |

Moves the 30s cost from first query to registration where it belongs.

## Solution 2: Layer Sampling (Medium Priority)

**Problem:** Querying all 36 layers is expensive when only a subset contributes
meaningful semantic signal.

**Fix:** Sample evenly-spaced layers instead of querying all.

```sql
-- Query 8 evenly-spaced layers instead of all 36:
SET infer.similarity_max_layers = 8;
```

**Impact:**

| Scenario | All Layers | 12 Layers | 8 Layers |
|----------|-----------|-----------|----------|
| similar_to() | 1.9s | ~0.6s | ~0.4s |
| Accuracy loss | baseline | minimal | minimal |

## Solution 3: Parallel Layer Queries (Medium Priority)

**Problem:** Layer queries are sequential despite being embarrassingly parallel.

**Fix:** Use Rayon to process layers concurrently.

```sql
SET infer.parallel_similarity = true;
```

**Impact:**

| Cores | Speedup | similar_to() |
|-------|---------|-------------|
| 1 | 1x | 1.9s |
| 4 | ~3.5x | ~0.5s |
| 8 | ~6x | ~0.3s |

## Combined Results

With all optimizations applied:

| Scenario | Current | Optimized |
|----------|---------|-----------|
| Registration | 90s | 120s (+30s HNSW build) |
| First similar_to() | 98s | 0.2s |
| Cached similar_to() | 1.9s | 0.2s |

**Target: 490x improvement on first query, 10x on cached queries.**

## Future Optimizations

### Priority 1: Batched Gate KNN (10-50x potential)

Pre-compute query's gate activations once, store as bitvector.
For each row, only compute dot products for the top-K features
from the query (not all 4864 features per layer).

Expected: 50-100 features vs 4864 = 50x reduction.

### Priority 2: SIMD-Accelerated Dot Products (2-4x potential)

The f16 gate vectors require conversion before dot product.
Pre-convert to f32 on load, use AVX2/AVX-512 for dot products.

Already partially implemented in `infer-compute`:
- Q4_K and Q6_K dequantization (AVX2, NEON, scalar fallbacks)
- f16 to f32 conversion (batch)
- Dot product and cosine similarity
- SQ8 quantized distance for HNSW index scans

### Priority 3: Approximate KNN Methods

Replace brute-force with approximate methods:
- Product quantization (PQ)
- Locality-sensitive hashing
- HNSW graph on gate vectors (currently implemented)

Combined potential: 100-1000x speedup on scans, bringing
100-row scan from 14s to 14-140ms (interactive speed).

### Priority 4: Result Caching

Add an LRU cache for recent (text, text) -> score lookups:

- Near-instant lookups for repeated queries
- Memory overhead bounded by cache size
- Only helps with repeated queries
- Cache invalidation on model changes

## Remote Backend: The Pragmatic Solution

For production workloads, the [remote backend](../deployment/remote.md) sidesteps
most local-mode performance issues:

- Shared activation cache (95%+ hit rate on recurring queries)
- No per-backend memory pressure
- One copy of the model serves all connections
- Sub-millisecond latency for cached queries over UDS

The remote backend achieves < 1ms latency for warm queries without any of the
above optimizations, because the server maintains a three-tier activation cache
that amortizes the gate KNN cost across all clients.

## Model Quality Notes

Performance optimizations apply regardless of model choice, but model quality
affects the usefulness of results:

| Model Type | describe() Quality | similar_to() Quality |
|------------|-------------------|---------------------|
| Code-focused (Qwen) | Poor (random tokens) | Mixed (some correct, some wrong) |
| General-purpose (Gemma, Llama) | Expected: semantic relationships | Expected: reliable rankings |

For best results, use general-purpose instruction-tuned models (Gemma, Llama,
Mistral) rather than code-focused models.
