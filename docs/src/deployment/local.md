# Local (mmap) Deployment

Best for: single-user exploration, development, small models (< 1B parameters).

## How It Works

```
psql --> PostgreSQL --> pg_infer --> mmap vindex file
```

Each PostgreSQL backend memory-maps the vindex directory directly. Simple to
deploy but memory-intensive: every connection holds its own f16-to-f32 decode
cache.

**Tradeoffs**: No shared cache. Each `similar_to` call redoes gate-KNN from
scratch. Tops out at 2-3 concurrent users on a 3B-parameter model before OOM.

## Step 1: Install the Extension

```sh
cargo pgrx install --release
```

## Step 2: Prepare a Vindex

Place the vindex directory where PostgreSQL can read it:

```sh
cp -r /path/to/qwen-0.5b.vindex /var/lib/postgresql/18/data/infer/
```

Or set an absolute path via the `infer.data_directory` GUC.

## Step 3: Register the Model

```sql
CREATE EXTENSION pg_infer;
SELECT infer_create_model('qwen05b', 'qwen-0.5b.vindex');
SET infer.default_model = 'qwen05b';
```

## Step 4: Query

```sql
SELECT * FROM describe('France');
SELECT similar_to('Paris', 'Berlin');
```

## Memory Considerations

Each loaded model in local mode consumes memory in several tiers:

| Component | Size (3B model) | Notes |
|-----------|-----------------|-------|
| Mmap'd vindex | ~2 GB | Shared via OS page cache |
| Decoded gate cache | ~112 MB/layer | Bounded by `gate_cache_max_layers` |
| HNSW indexes | ~200 MB/layer | Only if `use_hnsw = true` |
| Feature metadata | ~50 MB | Loaded once per model |

**Key insight**: The mmap'd vindex is shared across all PostgreSQL backends via
the OS page cache. The decoded gate cache is per-backend (per-connection). This
is why `gate_cache_max_layers` is critical for multi-connection deployments.

See [Tuning](../operations/tuning.md) for memory configuration details.
