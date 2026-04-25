# pg_infer: Neural Knowledge as a PostgreSQL Index

## Executive Summary

pg_infer integrates LARQL's neural network knowledge querying into PostgreSQL as a native extension, enabling SQL queries to directly access and reason with world knowledge embedded in transformer model weights. Unlike vector databases that store pre-computed embeddings, pg_infer queries the model weights themselves, providing access to factual knowledge, semantic relationships, and reasoning capabilities within standard SQL syntax.

The extension follows PostgreSQL's established patterns (like pg_vector, PostGIS, full-text search) by adding custom types, functions, and operators without modifying the SQL parser. This enables seamless composition with existing PostgreSQL features and extensions.  Conceptually models become indexes useful when querying within Postgres.

Imagine using this in concert with pg_vector/vectorscale, pg_textsearch, pg_trigrm, pg_tre (a Tre REGEX index with approximate matching), pg_mentat (a datalog system useful for storing facts and relationships available to inform reasoning).

Imagine that the "agent" is inside the database available to enrich the extent of queries possible in an RDBMS.

## Context and Motivation

### Current State: Knowledge Lives Outside the Database

Today's applications require two separate systems:
- **PostgreSQL**: Stores structured data you explicitly inserted
- **External LLMs**: Provide world knowledge and reasoning via API calls

This creates friction:
- Network latency for every knowledge lookup (~100ms per API call)
- Rate limits and per-token costs
- Data privacy concerns (sending data to external services)
- Complex orchestration between database queries and LLM calls
- No way to JOIN structured data with world knowledge

### LARQL's Innovation

LARQL decompiles transformer model weights into a queryable format called a "vindex":
- Gate vectors become a KNN index (0.008ms per layer lookup)
- Model knowledge accessible via DESCRIBE, WALK, INFER operations
- No GPU required for knowledge queries
- Full model reasoning available locally

### The Opportunity

What becomes possible when world knowledge is as natural to query as a table?

```sql
-- Impossible today: JOIN your data with model knowledge
SELECT c.name, d.target AS capital
FROM countries c
JOIN LATERAL describe(c.name) d ON d.relation = 'capital';

-- Impossible today: Semantic similarity without embedding pipelines
SELECT * FROM products
WHERE similar_to(name, 'machine learning tools') > 0.7;

-- Impossible today: Fill missing data using world knowledge
UPDATE products
SET category = (SELECT token FROM infer('category of ' || name) LIMIT 1)
WHERE category IS NULL;
```

## Design Principles

### 1. Work Within Existing SQL Syntax

Every successful PostgreSQL extension follows this pattern:
- **pg_vector**: Adds `vector` type and `<->` operator
- **PostGIS**: Adds `geometry` type and `ST_*` functions
- **pg_trgm**: Adds `%` operator and `similarity()` function
- **Full-text search**: Adds `tsvector` type and `@@` operator

**No extension modifies PostgreSQL's parser.** pg_infer follows this pattern exactly.

### 2. Make Model Access Transparent

Users shouldn't need to understand vindexes, extraction, or LARQL internals. Models are registered like indexes are created:

```sql
CREATE MODEL gemma3_4b FROM 'google/gemma-3-4b-it';
```

The system handles:
- Auto-downloading from HuggingFace
- Extracting to vindex format
- Caching in local storage
- Memory mapping for queries

### 3. Compose with Everything

Every function returns standard SQL types that compose with JOINs, CTEs, WHERE clauses, and other extensions:

```sql
-- Combines pg_infer + pg_vector + pg_trgm + BM25
WITH model_concepts AS (
    SELECT target FROM describe('coffee warmer') WHERE confidence > 10
),
vector_matches AS (
    SELECT * FROM products WHERE embedding <-> embed_text('coffee warmer') < 0.6
),
fuzzy_matches AS (
    SELECT * FROM products p, model_concepts c WHERE p.name % c.target
)
SELECT * FROM vector_matches UNION SELECT * FROM fuzzy_matches;
```

## API Design

### Model Registration

Models are registered once and referenced by name. Auto-downloads and extracts as needed.

```sql
-- From HuggingFace model ID (auto-downloads weights, extracts to vindex)
CREATE MODEL gemma3_4b FROM 'google/gemma-3-4b-it';

-- From pre-built vindex on HuggingFace (fastest, no extraction)
CREATE MODEL gemma3_4b FROM 'hf://chrishayuk/gemma-3-4b-it-vindex';

-- From local safetensors directory
CREATE MODEL llama3_8b FROM '/models/llama-3-8b/';

-- With options
CREATE MODEL gemma3_4b FROM 'google/gemma-3-4b-it'
    WITH (extract_level = 'inference', dtype = 'f16', max_memory = '8GB');

-- Set session default
SET infer.default_model = 'gemma3_4b';

-- Inspect registered models
SELECT model_name, vindex_path, extract_level, layers, features, memory_usage
FROM infer_models;

-- Remove model (unloads from memory, deletes cached vindex)
DROP MODEL gemma3_4b;
```

### Set-Returning Functions (SRFs)

These return table-valued results that compose with all SQL constructs.

#### `describe(entity text [, model text]) → TABLE`

Returns relationships the model knows about an entity.

```sql
-- Returns: (relation text, target text, confidence float8, layer int)
SELECT * FROM describe('France');
-- relation  | target  | confidence | layer
-- capital   | Paris   | 42.5       | 27
-- language  | French  | 35.2       | 24
-- continent | Europe  | 14.4       | 25

-- Explicit model name
SELECT * FROM describe('Einstein', model => 'llama3_8b');

-- Compose with JOINs
SELECT c.name, d.target AS capital
FROM countries c
JOIN LATERAL describe(c.name) d ON d.relation = 'capital';
```

**Implementation**: Calls LARQL's DESCRIBE operation via FFI. Returns averaged gate KNN scores across model layers, deduplicated by target token.

#### `walk(prompt text [, top int] [, model text]) → TABLE`

Traces model activation for a prompt, showing which features fire.

```sql
-- Returns: (layer int, feature int, activation float8, concept text)
SELECT * FROM walk('The capital of France is', top => 10);
-- layer | feature | activation | concept
-- 27    | 1436    | 45.2      | Paris
-- 24    | 892     | 35.1      | French

-- Find features that activate for product descriptions
SELECT p.name, w.concept, w.activation
FROM products p
JOIN LATERAL walk(p.description, top => 5) w ON true
ORDER BY w.activation DESC;
```

**Implementation**: Calls LARQL's WALK operation. Uses last token embedding as query vector for gate KNN across specified layers.

#### `infer(prompt text [, top int] [, model text]) → TABLE`

Runs full inference (forward pass with attention), returns token predictions.

```sql
-- Returns: (token text, probability float8, rank int)
SELECT * FROM infer('The capital of France is', top => 5);
-- token | probability | rank
-- Paris | 0.9791      | 1
-- the   | 0.0042      | 2

-- Fill missing categories
UPDATE products
SET category = sub.token
FROM (
    SELECT p.id, i.token
    FROM products p
    JOIN LATERAL infer('Product category: ' || p.name, top => 1) i ON true
    WHERE p.category IS NULL
) sub
WHERE products.id = sub.id;
```

**Implementation**: Requires `inference` extract level. Calls LARQL's full forward pass with attention. Returns softmax-normalized predictions.

### Scalar Functions

Return single values, work in WHERE clauses and expressions.

#### `similar_to(a text, b text [, model text]) → float8`

Semantic similarity between concepts using model's internal representation.

```sql
-- Direct comparison
SELECT similar_to('France', 'Paris');        -- high score (related)
SELECT similar_to('France', 'banana');       -- low score (unrelated)

-- In WHERE clauses
SELECT * FROM products
WHERE similar_to(category, 'artificial intelligence') > 15.0;

-- Cross-correlation analysis
SELECT c.name AS category, d.name AS department,
       similar_to(c.name, d.name) AS relevance
FROM categories c CROSS JOIN departments d
WHERE similar_to(c.name, d.name) > 20.0
ORDER BY relevance DESC;
```

**Implementation**: Averages token embeddings for each input, computes dot product with gate vectors across layers, returns maximum activation score. NOT cosine similarity on raw embeddings.

#### `implies(subject text, object text [, model text]) → bool`

Tests if model knowledge supports a directional relationship.

```sql
-- Does model think France implies Paris?
SELECT implies('France', 'Paris');  -- true

-- Validate referential integrity using world knowledge
SELECT c.name, c.headquarters
FROM companies c
WHERE NOT implies(c.name, c.headquarters);
-- Returns companies whose claimed HQ contradicts model knowledge
```

**Implementation**: Calls `describe(subject)`, checks if `object` appears as a target with confidence > threshold.

### Operators (Optional Convenience)

```sql
-- Semantic distance operator (wraps 1/similar_to for distance semantics)
CREATE OPERATOR <~> (
    LEFTARG = text, RIGHTARG = text,
    FUNCTION = infer_distance,
    COMMUTATOR = <~>
);

-- Usage
SELECT * FROM products
ORDER BY name <~> 'machine learning'
LIMIT 10;
```

## Integration with PostgreSQL Ecosystem

### Combined with pg_vector

pg_vector stores your embeddings; pg_infer provides world knowledge. They're complementary:

```sql
-- pg_vector: "find MY documents similar to this query"
SELECT * FROM docs ORDER BY embedding <-> query_vec LIMIT 10;

-- pg_infer: "what does the MODEL know about this topic?"
SELECT * FROM describe('quantum computing');

-- Combined: find docs about topics the model associates with a concept
WITH related_topics AS (
    SELECT target FROM describe('quantum computing') WHERE confidence > 10
)
SELECT d.title, d.content
FROM docs d, related_topics rt
WHERE d.embedding <-> embed(rt.target) < 0.5;
```

### Combined with pg_trgm (Fuzzy String Matching)

```sql
-- Entity resolution: string similarity + semantic equivalence
SELECT a.name AS system_a, b.name AS system_b,
       similarity(a.name, b.name) AS string_sim,      -- pg_trgm
       similar_to(a.name, b.name) AS semantic_sim     -- pg_infer
FROM suppliers_a a CROSS JOIN suppliers_b b
WHERE similarity(a.name, b.name) > 0.3                -- fuzzy string match
   OR similar_to(a.name, b.name) > 25.0;              -- semantic match
-- Catches: "IBM Corp" ↔ "International Business Machines"
```

### Combined with BM25/Full-Text Search

```sql
-- Semantic query expansion, then BM25 ranking
WITH expanded_terms AS (
    SELECT 'automobile' AS original
    UNION ALL
    SELECT d.target FROM describe('automobile') d
    WHERE d.confidence > 20 AND LENGTH(d.target) > 2
    -- → car, vehicle, sedan, SUV, transportation, ...
)
SELECT d.title,
       ts_rank(to_tsvector(d.content),
               to_tsquery(string_agg(et.original, ' | '))) AS relevance
FROM documents d, expanded_terms et
GROUP BY d.title
ORDER BY relevance DESC;
```

## Implementation Architecture

### Extension Structure (No PostgreSQL Core Modifications)

```
contrib/pg_infer/
├── pg_infer--1.0.sql           # CREATE FUNCTION/OPERATOR definitions
├── pg_infer.control            # Extension metadata
├── pg_infer.c                  # Extension init, GUC registration, hooks
├── model_registry.c            # CREATE MODEL, model loading lifecycle
├── fn_describe.c               # describe() SRF implementation
├── fn_walk.c                   # walk() SRF implementation
├── fn_infer.c                  # infer() SRF implementation
├── fn_similar_to.c             # similar_to() and <~> operator
├── fn_implies.c                # implies() scalar function
├── vindex_bridge.c             # FFI bridge to LARQL Rust crates
├── shared_memory.c             # Shared memory management for vindexes
├── error_handling.c            # Error code mapping, structured messages
└── Makefile                    # Build config, links against LARQL libs
```

### FFI Bridge to LARQL

The extension links against LARQL's existing Rust crates via C FFI. All vindex loading, mmap management, gate KNN, tokenizer, and inference code is reused.

Create a new crate `crates/larql-pg-ffi/` that exposes C-ABI functions:

```rust
// crates/larql-pg-ffi/src/lib.rs
use std::ffi::{CStr, CString, c_char, c_int};
use larql_vindex::{VectorIndex, PatchedVindex};

#[repr(C)]
pub struct VindexHandle {
    vindex: Box<VectorIndex>,
    patched: Option<Box<PatchedVindex>>,
}

#[repr(C)]
pub struct DescribeRow {
    relation: *mut c_char,    // PostgreSQL will pfree these
    target: *mut c_char,
    confidence: f64,
    layer: u32,
}

#[no_mangle]
pub extern "C" fn infer_load_model(
    source: *const c_char,      // Model ID, hf:// path, or local path
    extract_level: c_int,       // 1=browse, 2=inference, 3=all
    data_dir: *const c_char,    // Where to cache extracted vindexes
    error_msg: *mut *mut c_char // OUT: error message on failure
) -> *mut VindexHandle;

#[no_mangle]
pub extern "C" fn infer_describe(
    handle: *const VindexHandle,
    entity: *const c_char,
    results: *mut DescribeRow,
    max_results: usize,
    actual_count: *mut usize    // OUT: actual rows returned
) -> c_int;                     // 0=success, <0=error

#[no_mangle]
pub extern "C" fn infer_gate_knn(
    handle: *const VindexHandle,
    layer: usize,
    residual: *const f32,
    hidden_size: usize,
    top_k: usize,
    results: *mut KnnResult,
    actual_count: *mut usize
) -> c_int;

#[no_mangle]
pub extern "C" fn infer_similar_to(
    handle: *const VindexHandle,
    entity_a: *const c_char,
    entity_b: *const c_char,
    score: *mut f64             // OUT: similarity score
) -> c_int;

#[no_mangle]
pub extern "C" fn infer_free_handle(handle: *mut VindexHandle);

#[no_mangle]
pub extern "C" fn infer_free_cstring(s: *mut c_char);
```

### Memory Management Strategy

#### Shared Memory for Vindexes

Vindexes are mmap'd files shared across all PostgreSQL backends:

```c
// In pg_infer.c
typedef struct InferSharedState {
    LWLock     *lock;
    int         num_models;
    Size        total_memory;
    Size        memory_limit;      // From infer.max_memory GUC
    HTAB       *model_registry;    // Hash: model_name → VindexEntry
} InferSharedState;

typedef struct VindexEntry {
    char        model_name[64];
    char        vindex_path[MAXPGPATH];
    VindexHandle *handle;          // FFI handle to Rust VectorIndex
    int         ref_count;         // Number of backends using this
    TimestampTz last_accessed;     // For LRU eviction
    Size        memory_usage;      // Estimated RSS
} VindexEntry;

static InferSharedState *infer_shared = NULL;

void infer_shmem_startup(void) {
    bool found;
    infer_shared = ShmemInitStruct("pg_infer",
                                   sizeof(InferSharedState),
                                   &found);
    if (!found) {
        infer_shared->lock = &(GetNamedLWLockTranche("pg_infer"))->lock;
        infer_shared->num_models = 0;
        infer_shared->total_memory = 0;
        infer_shared->memory_limit = infer_max_memory * 1024L * 1024L;
        // Create hash table for model registry
    }
}
```

#### Per-Query Memory Context

SRF results use PostgreSQL's memory context system for automatic cleanup:

```c
// In fn_describe.c
typedef struct DescribeFuncState {
    MemoryContext   result_context;
    VindexHandle   *vindex;
    DescribeRow    *results;
    int             num_results;
    int             current_row;
} DescribeFuncState;

PG_FUNCTION_INFO_V1(infer_describe);
Datum infer_describe(PG_FUNCTION_ARGS) {
    FuncCallContext *funcctx;
    DescribeFuncState *state;

    if (SRF_IS_FIRSTCALL()) {
        funcctx = SRF_FIRSTCALL_INIT();

        // Switch to multi-call context for SRF state
        MemoryContext oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        state = palloc(sizeof(DescribeFuncState));
        state->result_context = AllocSetContextCreate(funcctx->multi_call_memory_ctx, ...);

        // Call LARQL FFI
        text *entity = PG_GETARG_TEXT_P(0);
        char *entity_str = text_to_cstring(entity);

        VindexHandle *vindex = get_model_handle(infer_default_model);
        state->results = palloc(sizeof(DescribeRow) * MAX_DESCRIBE_RESULTS);

        int success = infer_describe(vindex, entity_str, state->results,
                                    MAX_DESCRIBE_RESULTS, &state->num_results);
        if (success < 0) {
            ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                           errmsg("LARQL describe failed")));
        }

        funcctx->user_fctx = state;
        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    state = (DescribeFuncState *) funcctx->user_fctx;

    if (state->current_row < state->num_results) {
        // Build result tuple
        Datum values[4];
        bool nulls[4] = {false, false, false, false};

        values[0] = CStringGetTextDatum(state->results[state->current_row].relation);
        values[1] = CStringGetTextDatum(state->results[state->current_row].target);
        values[2] = Float8GetDatum(state->results[state->current_row].confidence);
        values[3] = Int32GetDatum(state->results[state->current_row].layer);

        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        Datum result = HeapTupleGetDatum(tuple);

        state->current_row++;
        SRF_RETURN_NEXT(funcctx, result);
    } else {
        SRF_RETURN_DONE(funcctx);
    }
}
```

### Model Loading and Caching

```c
// In model_registry.c
VindexHandle* get_model_handle(const char *model_name) {
    VindexEntry *entry;

    LWLockAcquire(infer_shared->lock, LW_SHARED);
    entry = hash_search(infer_shared->model_registry, model_name, HASH_FIND, NULL);

    if (entry) {
        entry->ref_count++;
        entry->last_accessed = GetCurrentTimestamp();
        LWLockRelease(infer_shared->lock);
        return entry->handle;
    }
    LWLockRelease(infer_shared->lock);

    // Model not loaded, need exclusive lock to load
    LWLockAcquire(infer_shared->lock, LW_EXCLUSIVE);

    // Check again in case another backend loaded it
    entry = hash_search(infer_shared->model_registry, model_name, HASH_FIND, NULL);
    if (entry) {
        entry->ref_count++;
        entry->last_accessed = GetCurrentTimestamp();
        LWLockRelease(infer_shared->lock);
        return entry->handle;
    }

    // Load model via FFI
    char error_msg[512];
    VindexHandle *handle = infer_load_model(model_source, extract_level,
                                           infer_data_directory, error_msg);
    if (!handle) {
        LWLockRelease(infer_shared->lock);
        ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                       errmsg("Failed to load model %s: %s", model_name, error_msg)));
    }

    // Add to registry
    entry = hash_search(infer_shared->model_registry, model_name, HASH_ENTER, NULL);
    strncpy(entry->model_name, model_name, sizeof(entry->model_name));
    entry->handle = handle;
    entry->ref_count = 1;
    entry->last_accessed = GetCurrentTimestamp();

    infer_shared->num_models++;
    infer_shared->total_memory += estimate_model_memory(handle);

    LWLockRelease(infer_shared->lock);
    return handle;
}
```

### Configuration (GUCs)

```c
// In pg_infer.c
char *infer_default_model = NULL;
char *infer_data_directory = NULL;
int infer_max_memory = 8192;        // MB
bool infer_auto_download = true;

void _PG_init(void) {
    DefineCustomStringVariable("infer.default_model",
                              "Default model name for LARQL functions",
                              "Model to use when not explicitly specified",
                              &infer_default_model,
                              NULL,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

    DefineCustomStringVariable("infer.data_directory",
                              "Directory for cached vindexes",
                              "Where extracted vindexes are stored",
                              &infer_data_directory,
                              "infer",  // Default: $PGDATA/infer/
                              PGC_SIGHUP,
                              0, NULL, NULL, NULL);

    DefineCustomIntVariable("infer.max_memory",
                           "Maximum memory for loaded models (MB)",
                           "Total memory budget for all loaded vindexes",
                           &infer_max_memory,
                           8192,     // 8GB default
                           512,      // Min 512MB
                           65536,    // Max 64GB
                           PGC_SIGHUP,
                           0, NULL, NULL, NULL);

    DefineCustomBoolVariable("infer.auto_download",
                            "Auto-download models from HuggingFace",
                            "Whether CREATE MODEL can download from HF",
                            &infer_auto_download,
                            true,
                            PGC_SUSET,
                            0, NULL, NULL, NULL);

    // Request shared memory
    RequestAddinShmemSpace(sizeof(InferSharedState) + hash_estimate_size(64, sizeof(VindexEntry)));
    RequestNamedLWLockTranche("pg_infer", 1);

    // Hook for shared memory initialization
    prev_shmem_startup_hook = shmem_startup_hook;
    shmem_startup_hook = infer_shmem_startup;
}
```

## Amazon-Scale Use Cases

### 1. Zero-Shot Product Categorization

Millions of products arrive daily with no category or wrong categories. Traditional ML requires training data, model training, retraining cycles.

```sql
-- Categorize uncategorized products using world knowledge
UPDATE products
SET category = sub.predicted_category,
    category_confidence = sub.confidence
FROM (
    SELECT p.id,
           i.token AS predicted_category,
           i.probability AS confidence
    FROM products p
    JOIN LATERAL infer('Product category for "' || p.name || '":') i ON true
    WHERE p.category IS NULL AND i.rank = 1
) sub
WHERE products.id = sub.id;

-- Validate existing categories against model knowledge
SELECT p.id, p.name, p.category AS current_category,
       d.target AS model_suggests,
       d.confidence AS model_confidence
FROM products p
JOIN LATERAL describe(p.name) d ON d.relation IN ('category', 'type', 'class')
WHERE NOT similar_to(p.category, d.target) > 15.0
ORDER BY d.confidence DESC;
```

### 2. Cross-Sell Without Purchase History

New products have zero co-purchase data. Collaborative filtering can't help. Model knowledge bridges this gap.

```sql
-- Customer bought "camping tent", what else might they need?
WITH tent_related AS (
    SELECT target AS item, confidence
    FROM describe('camping tent')
    WHERE relation IN ('used_with', 'requires', 'part_of', 'complements')
      AND confidence > 10
),
available_products AS (
    SELECT p.id, p.name, tr.item, tr.confidence,
           similar_to(p.name, tr.item) AS semantic_match
    FROM products p, tent_related tr
    WHERE similar_to(p.name, tr.item) > 15.0
      AND p.category != 'Tents'
)
SELECT name, semantic_match, confidence
FROM available_products
ORDER BY semantic_match DESC, confidence DESC
LIMIT 20;
-- → sleeping bag, camping stove, headlamp, first aid kit...
-- Even if never seen co-purchased
```

### 3. Supply Chain Intelligence

```sql
-- Which suppliers are in regions the model associates with semiconductor manufacturing?
WITH semiconductor_regions AS (
    SELECT d.target AS region, d.confidence
    FROM describe('semiconductor manufacturing') d
    WHERE d.relation IN ('location', 'country', 'region')
      AND d.confidence > 15
)
SELECT s.name AS supplier, s.country, sr.region, sr.confidence,
       similar_to(s.country, sr.region) AS region_relevance
FROM suppliers s
JOIN semiconductor_regions sr ON similar_to(s.country, sr.region) > 20.0
ORDER BY region_relevance DESC, sr.confidence DESC;
```

### 4. Intelligent Search Autocomplete

```sql
-- As user types "wire", suggest completions based on model + inventory
WITH model_expansions AS (
    SELECT d.target AS suggestion, d.confidence
    FROM describe('wire') d
    WHERE d.confidence > 10
      AND LENGTH(d.target) > 3
    UNION ALL
    SELECT w.concept, w.activation
    FROM walk('wire', top => 20) w
    WHERE w.activation > 15
),
inventory_matches AS (
    SELECT DISTINCT p.name, similar_to(p.name, 'wire') AS relevance
    FROM products p
    WHERE similar_to(p.name, 'wire') > 10.0
       OR p.name % 'wire'  -- pg_trgm for typos
)
SELECT name AS suggestion, relevance
FROM inventory_matches
ORDER BY relevance DESC
LIMIT 10;
-- → wireless mouse, wire stripper, wireless charger, ethernet cable...
```

## Impossible-Without-This Queries

### Semantic Foreign Keys

Traditional foreign keys match by value. Semantic foreign keys match by meaning.

```sql
-- Find inconsistent category naming across systems
-- "Electronics" vs "Tech Gadgets" — same meaning, different words
SELECT o.order_id, p.category AS product_category, o.category AS order_category,
       similar_to(p.category, o.category) AS semantic_similarity
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE p.category != o.category  -- String mismatch
  AND similar_to(p.category, o.category) > 20.0;  -- But semantically same
-- Traditional constraints can't catch "soft mismatches"
```

### Commonsense Data Validation

```sql
-- Flag products with implausible attributes using world knowledge
CREATE OR REPLACE FUNCTION validate_product_weight(name text, weight_kg float8)
RETURNS boolean AS $$
BEGIN
    -- Use model knowledge about typical weights
    RETURN EXISTS (
        SELECT 1 FROM describe(name || ' weight') d
        WHERE d.confidence > 5
          AND d.target::float8 BETWEEN weight_kg * 0.1 AND weight_kg * 10
    );
END;
$$ LANGUAGE plpgsql;

-- Find implausible listings
SELECT id, name, weight_kg
FROM products
WHERE weight_kg > 0
  AND NOT validate_product_weight(name, weight_kg);
-- → "USB cable weighing 50kg", "laptop at 0.01kg"
```

### Multi-Hop Reasoning

```sql
-- "Which customers work in industries that compete with our target markets?"
WITH customer_industries AS (
    SELECT c.id, c.name AS customer,
           d.target AS industry
    FROM customers c
    JOIN LATERAL describe(c.profession) d ON d.relation LIKE '%industry%'
    WHERE d.confidence > 15
),
target_markets AS (
    SELECT target AS market
    FROM describe((SELECT strategy FROM business_plan WHERE year = 2025))
    WHERE confidence > 10
),
competing_industries AS (
    SELECT tm.market, d.target AS competitor_industry
    FROM target_markets tm
    JOIN LATERAL describe(tm.market) d ON d.relation IN ('competitor', 'competes_with')
    WHERE d.confidence > 10
)
SELECT ci.customer, ci.industry, ci2.market, ci2.competitor_industry,
       similar_to(ci.industry, ci2.competitor_industry) AS competition_strength
FROM customer_industries ci
CROSS JOIN competing_industries ci2
WHERE similar_to(ci.industry, ci2.competitor_industry) > 25.0
ORDER BY competition_strength DESC;
```

### Semantic Aggregation

```sql
-- Group support tickets by model-understood topic, not keywords
SELECT d.target AS topic,
       count(*) AS ticket_count,
       avg(t.resolution_time_hours) AS avg_resolution_time,
       array_agg(t.id ORDER BY t.created_at DESC LIMIT 5) AS recent_tickets
FROM support_tickets t
JOIN LATERAL describe(t.subject, top => 1) d ON d.confidence > 8
WHERE t.created_at > now() - interval '30 days'
GROUP BY d.target
HAVING count(*) > 5
ORDER BY ticket_count DESC;
-- Groups: "can't log in", "password reset", "account locked" → "authentication"
--         "slow page", "timeout", "won't load" → "performance"
```

## Performance Considerations

### Target Performance (vs Standalone LARQL)

| Operation | Target | Standalone LARQL | Overhead |
|-----------|--------|-----------------|-----------|
| `describe()` | <50ms | 33ms | <20ms FFI + result formatting |
| `walk()` | <10ms | 0.3ms | <10ms FFI + SRF overhead |
| `similar_to()` | <5ms | ~1ms | <4ms FFI + scalar result |
| `infer()` | <600ms | 517ms | <100ms FFI + attention setup |

### Memory Usage

- **Vindex mmap**: Same as standalone (450MB RSS for Gemma-3-4B browse level)
- **Shared memory overhead**: ~1MB for registry + locks
- **Per-query overhead**: ~10KB for SRF state, freed at query end

### Optimization Strategies

1. **Connection pooling**: Reuse FFI handles across queries in same session
2. **Result caching**: Cache frequent `describe()` results in PostgreSQL shared memory
3. **Lazy model loading**: Only load models on first use, evict by LRU
4. **Batch operations**: Process multiple entities in single FFI call when possible

## Error Handling and Diagnostics

### Error Code Mapping

```c
// Map LARQL errors to PostgreSQL error codes
typedef struct {
    int infer_error;
    int pg_error_code;
    const char *pg_error_category;
} ErrorMapping;

static ErrorMapping error_mappings[] = {
    {INFER_ERROR_MODEL_NOT_FOUND, ERRCODE_INVALID_NAME, "model"},
    {INFER_ERROR_VINDEX_CORRUPT, ERRCODE_DATA_CORRUPTED, "vindex"},
    {INFER_ERROR_INSUFFICIENT_MEMORY, ERRCODE_OUT_OF_MEMORY, "memory"},
    {INFER_ERROR_EXTRACT_LEVEL, ERRCODE_FEATURE_NOT_SUPPORTED, "operation"},
    {0, 0, NULL}  // Sentinel
};

void report_infer_error(int infer_error, const char *detail) {
    ErrorMapping *mapping = find_error_mapping(infer_error);
    ereport(ERROR,
            (errcode(mapping->pg_error_code),
             errmsg("LARQL %s error: %s", mapping->pg_error_category, detail),
             errhint("Check model registration and extract level")));
}
```

### Diagnostic Views

```sql
-- Inspect loaded models and memory usage
CREATE VIEW infer_model_status AS
SELECT model_name, vindex_path, extract_level,
       memory_usage_mb, ref_count, last_accessed,
       age(now(), last_accessed) AS idle_time
FROM infer_models_internal();

-- Monitor query performance
CREATE VIEW infer_query_stats AS
SELECT function_name, total_calls, total_time_ms,
       avg_time_ms, max_time_ms, error_count
FROM infer_performance_counters();
```

## Installation and Configuration

### Build Requirements

```bash
# PostgreSQL development headers
sudo apt-get install postgresql-server-dev-14

# Rust toolchain (for LARQL FFI crate)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# BLAS library (for gate KNN)
sudo apt-get install libblas-dev liblapack-dev

# Build LARQL with FFI crate
cd larql/
cargo build --release -p larql-pg-ffi

# Build PostgreSQL extension
cd contrib/pg_infer/
make install USE_PGXS=1
```

### Extension Installation

```sql
-- Enable extension (requires superuser)
CREATE EXTENSION pg_infer;

-- Configure memory limit and data directory
ALTER SYSTEM SET infer.max_memory = '16GB';
ALTER SYSTEM SET infer.data_directory = '/var/lib/postgresql/infer';
SELECT pg_reload_conf();

-- Register first model
CREATE MODEL gemma3_4b FROM 'hf://chrishayuk/gemma-3-4b-it-vindex';
SET infer.default_model = 'gemma3_4b';

-- Verify installation
SELECT * FROM infer_models;
SELECT * FROM describe('France') LIMIT 3;
```

## Testing Strategy

### Smoke Tests

```sql
-- Basic functionality
SELECT count(*) FROM describe('France');                    -- Should return > 0
SELECT similar_to('France', 'Paris') > similar_to('France', 'banana');  -- Should be true
SELECT token FROM infer('The capital of France is') LIMIT 1;            -- Should be "Paris"

-- Composition with SQL
CREATE TEMP TABLE test_countries (name text);
INSERT INTO test_countries VALUES ('France'), ('Germany'), ('Japan');

SELECT c.name, d.target
FROM test_countries c
JOIN LATERAL describe(c.name) d ON d.relation = 'capital';
-- Should return 3 rows with capitals
```

### Performance Tests

```sql
-- Measure latency
\timing on
SELECT * FROM describe('artificial intelligence') LIMIT 10;
-- Should complete in <50ms

-- Memory usage
SELECT sum(memory_usage_mb) FROM infer_model_status;
-- Should be reasonable (< configured limit)
```

### Regression Tests

- Existing LARQL test suite (`cargo test`) should pass unchanged
- PostgreSQL regression test suite (`pg_regress`) with pg_infer-specific tests
- Concurrency tests (multiple backends querying same model simultaneously)
- Memory leak detection under repeated SRF calls

## Security Considerations

### Access Control

```sql
-- Function-level permissions (like other extensions)
GRANT EXECUTE ON FUNCTION describe(text) TO analysts;
REVOKE EXECUTE ON FUNCTION infer(text) FROM public;  -- Expensive operation

-- Model-level access control (future enhancement)
GRANT USAGE ON MODEL gemma3_4b TO data_science_team;
```

### Data Privacy

- Models run entirely local (no external API calls)
- Vindex files contain no user data (only pre-trained weights)
- Query text never leaves PostgreSQL server

### Resource Limits

```sql
-- Per-user resource limits
ALTER ROLE analyst SET infer.max_memory = '2GB';
ALTER ROLE analyst SET infer.default_model = 'small_model';
```

## Future Enhancements

### Phase 2: Model Mutations

Support for inserting knowledge into models via LARQL's patch system:

```sql
-- Insert new facts (creates transaction-scoped patch)
INSERT INTO model_knowledge (entity, relation, target, confidence)
VALUES ('Acme Corp', 'headquarters', 'Seattle', 0.9);

-- Patch persisted on COMMIT, discarded on ROLLBACK
COMMIT;  -- Saves patch to .vlp file

-- Query sees inserted knowledge immediately
SELECT * FROM describe('Acme Corp');  -- Returns "headquarters → Seattle"
```

### Phase 3: Multi-Model Queries

```sql
-- Compare knowledge across models
SELECT d1.target AS gemma_says, d2.target AS llama_says
FROM describe('artificial intelligence', model => 'gemma3_4b') d1
FULL JOIN describe('artificial intelligence', model => 'llama3_8b') d2
    ON d1.relation = d2.relation
WHERE d1.target != d2.target;  -- Model disagreements
```

### Phase 4: Query Plan Integration

Teach PostgreSQL's query planner about model operations for better optimization:

- Index scans guided by `similar_to()` predicates
- Parallel execution of independent `describe()` calls
- Materialized results caching for repeated model queries

## Conclusion

pg_infer transforms PostgreSQL into the first database to natively support neural network knowledge querying. By working within established PostgreSQL patterns, it enables seamless composition of world knowledge with relational data, opening entirely new classes of applications that blend structured data with AI reasoning.

The extension's success will be measured not by its technical sophistication, but by how naturally developers adopt it for problems they couldn't solve before — entity resolution, semantic search, automated data enrichment, and commonsense reasoning — all within familiar SQL syntax.
