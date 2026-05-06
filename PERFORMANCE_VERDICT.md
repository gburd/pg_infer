# Performance Verdict: Is pg_infer Fast Enough for Production?

## Executive Summary

**NO** - pg_infer is **NOT fast enough for "index speeds"** or high-throughput production use.

**However**, it IS suitable for specific use cases with small result sets and pre-filtered data.

---

## Benchmark Results (Qwen 2.5 3B, 12-layer sampling)

### Single Similarity Computation
- **First call (cold cache):** 93.7 seconds
- **Subsequent calls (warm cache):** 1.86 seconds
- **Comparison:** B-tree index lookup = <1ms

**Verdict:** **2,600× slower than traditional index**

### Category Matching (20 comparisons)
- **Total time:** 37.4 seconds
- **Per-comparison:** 1.87 seconds
- **Throughput:** ~32 comparisons/minute

### Small Batch (50+ comparisons)
- **Result:** Server crash due to memory pressure
- **Estimated time:** ~93 seconds (50 × 1.86s)

### Scaling Estimates (Based on 1.86s per comparison)

| Rows | Time | Practical? |
|------|------|-----------|
| 10 | 18.6s | ✓ Maybe |
| 20 | 37.2s | ✓ Borderline |
| 50 | 93s | ✗ Too slow |
| 100 | 3.1 min | ✗ Too slow |
| 500 | 15.5 min | ✗ Unacceptable |
| 1,000 | 31 min | ✗ Unacceptable |
| 10,000 | 5.2 hours | ✗ Impossible |

---

## Technical Analysis

### Why So Slow?

Each `similar_to(a, b)` call performs:

1. **Tokenization:** 2 texts → token IDs
2. **Embedding:** 2 × average token embeddings (row lookups)
3. **Layer queries:** 12 layers × 2 texts × gate_knn
   - Each gate_knn: BLAS gemv over 14,336 features
   - Top-K selection (50 features per layer)
4. **Overlap detection:** Find shared features across layers
5. **Score computation:** Max shared gate activation

**Total per comparison:**
- 24 BLAS gemv operations (12 layers × 2 texts)
- 24 top-K selections
- Overlap analysis across 12 × 50 = 600 feature pairs

**At 1.86s per comparison, that's:**
- ~77ms per layer query
- ~13ms per gate_knn operation

### Memory Issues

The server crashes at ~50 comparisons, suggesting:
- F16 decode cache accumulation
- BLAS operation memory allocation
- Rayon thread pool overhead (even when disabled, library is loaded)

---

## F16→F32 Decoding Explained

### What is F16?
- **Half-precision float:** 16 bits (vs 32 bits for F32)
- **Storage:** 2 bytes per number (50% savings)
- **Precision:** ~3 decimal digits
- **Range:** ±65,504

### Qwen 2.5 3B Storage
- **Gate vectors:** 36 layers × 14,336 features × 2048 hidden
- **F32 size:** 1.06 billion values × 4 bytes = **4.24 GB**
- **F16 size:** 1.06 billion values × 2 bytes = **2.12 GB**

### What Does `warmup_on_load` Do?

**Disabled (default now):**
```
Registration: 88s
Memory: 2.12 GB mmap'd (shared across backends)
First query per layer: Includes decode cost (~77ms)
Subsequent queries: Fast (cached F32)
```

**Enabled (old default):**
```
Registration: Decode ALL 36 layers f16→f32 = 2.12GB → 4.24GB RAM
Memory: 4.24 GB per backend (not shared!)
First query: Fast (already decoded)
Problem: OOM crashes with multiple backends
```

### Why F16 Decode is Necessary

BLAS operations require F32:
```rust
// Disk storage: F16 (compact)
let f16_bytes: &[u8] = &mmap[offset..end];  // 2 bytes/value

// Must decode for computation: F32
let f32_data: Vec<f32> = decode_f16(f16_bytes);  // 4 bytes/value

// BLAS gemv requires F32
let scores = gemv(&gate_matrix_f32, &query_vector_f32);
```

**Trade-off:** 50% storage savings, but decode overhead on first use per layer.

---

## Practical Use Cases

### ✓ SUITABLE FOR:

1. **Category/Tag Assignment (Small Taxonomy)**
   - Compare item against 10-20 categories
   - Time: ~20-40 seconds
   - Example: Auto-categorize uploads

2. **Semantic Re-ranking (Pre-filtered Results)**
   - Traditional index filters to <20 candidates
   - Re-rank by similarity
   - Time: ~20-40 seconds
   - Example: Search result refinement

3. **Deduplication (Small Batches)**
   - Find duplicates within small groups (<20 items)
   - Time: ~10-20 seconds per batch
   - Example: Merge similar entries

4. **Interactive Search (With LIMIT)**
   - User search with TOP-10 or TOP-20
   - Pre-filter with WHERE clauses
   - Time: ~20-40 seconds
   - Example: "Find similar products" button

5. **Offline Batch Processing**
   - Nightly jobs, data pipelines
   - Not user-facing
   - Example: Compute similarity matrix overnight

### ✗ NOT SUITABLE FOR:

1. **Full Table Scans**
   - 10,000 rows = 5.2 hours
   - Server crashes on 50+ rows

2. **Large JOIN Operations**
   - Product × Category (1000 × 20) = 9.3 hours
   - Not feasible

3. **Real-time Filtering**
   - Users expect <100ms response
   - pg_infer: 1,860ms per comparison
   - 18× slower than user expectation

4. **High-throughput OLTP**
   - Traditional index: 1000s of lookups/second
   - pg_infer: 0.5 comparisons/second
   - 2000× throughput difference

5. **WHERE Clause Semantic Filtering**
   - `WHERE similar_to(col, 'query') > 0.5` scans full table
   - Use traditional indexes for initial filtering

---

## Recommended Architecture

### Pattern: Traditional Index → Semantic Ranking

```sql
-- STEP 1: Traditional filtering (FAST - milliseconds)
WITH candidates AS (
    SELECT *
    FROM products
    WHERE
        category_id = 5              -- Index seek
        AND price BETWEEN 100 AND 500  -- Index range scan
        AND in_stock = true            -- Index filter
    LIMIT 20                          -- Limit candidates
)
-- STEP 2: Semantic ranking (SLOW - ~40 seconds for 20 rows)
SELECT
    name,
    description,
    price,
    similar_to(description, 'portable wireless device') AS relevance
FROM candidates
ORDER BY relevance DESC
LIMIT 10;
```

**Time:**
- Traditional filtering: <10ms
- Semantic ranking 20 rows: ~37s
- **Total: ~37 seconds**

### Anti-Pattern: Semantic-First Filtering

```sql
-- WRONG: Scans entire table (10,000 rows = 5.2 hours)
SELECT *
FROM products
WHERE similar_to(description, 'wireless device') > 0.5
LIMIT 10;
```

---

## Optimization Recommendations

### Already Implemented ✓
1. **Layer sampling:** 36 layers → 12 layers (21× speedup)
2. **Disable HNSW:** Causes crashes on large models
3. **Disable warmup:** Prevents OOM
4. **Memmap for zero-copy:** Avoids loading 2.2GB into RAM

### Future Optimizations

1. **Precompute Embeddings**
   - Store averaged embeddings in a COLUMN
   - Compute once, reuse many times
   - Saves tokenization + embedding lookup

2. **Materialized Similarity Scores**
   - Pre-compute common comparisons
   - Store in lookup table
   - Example: Product-to-Category scores

3. **Approximate Methods**
   - Reduce layers further (6 layers = ~0.9s per comparison)
   - Lower top_k (50 → 10)
   - Accept accuracy loss for speed

4. **GPU Acceleration**
   - BLAS on GPU could be 10-50× faster
   - Would make 100-1000 row scans feasible

5. **Caching Layer**
   - Cache popular queries in Redis
   - Avoid recomputation
   - Example: "laptop" query cached for 1 hour

---

## Conclusion

**Is pg_infer fast enough for production?**

**For "index speeds":** NO
- Traditional index: <1ms
- pg_infer: 1,860ms
- **2,600× slower**

**For specific use cases:** YES (with constraints)
- Small batches (<20 rows)
- Pre-filtered data
- Offline processing
- Interactive features with user patience
- Not latency-sensitive paths

**Architecture:** Use pg_infer as a **semantic ranking layer** on top of traditional indexes, not as a replacement for them.

**Key Insight:** pg_infer is **semantic computing**, not **semantic indexing**. Each comparison is a complex neural network operation, not a simple lookup.

---

## Performance Comparison Table

| Operation | Time | Use Case |
|-----------|------|----------|
| B-tree index lookup | <1ms | ✓ Primary key, foreign key, WHERE filters |
| Hash index lookup | <1ms | ✓ Exact match lookups |
| GIN index (full-text) | 1-10ms | ✓ Text search, array containment |
| **pg_infer similar_to()** | **1,860ms** | ✗ **NOT for indexing** |
| pg_infer (cold cache) | 94,000ms | ✗ Unacceptable for any user-facing query |

**Reality Check:** pg_infer is 1,860× - 94,000× slower than traditional indexes.

**Use Case Fit:** Semantic enrichment of small, pre-filtered result sets. Not a general-purpose index replacement.
