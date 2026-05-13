# Introduction

pg_infer is a PostgreSQL extension that provides SQL-accessible neural network
inference. It exposes transformer model internals (gate vectors, sparse
features, activation patterns) as queryable relations within PostgreSQL.

## Quick Start

```sql
CREATE EXTENSION pg_infer;

-- Register a local model (mmap mode):
SELECT infer_create_model('qwen05b', '/data/qwen-0.5b.vindex');

-- Query knowledge edges:
SELECT * FROM describe('France');

-- Semantic similarity:
SELECT similar_to('Paris', 'France');

-- Walk activations:
SELECT * FROM walk('capital cities', top => 20);
```

## Deployment Modes

pg_infer supports three deployment topologies, selectable per model:

- **[Local (mmap)](deployment/local.md)** -- Each PostgreSQL backend memory-maps
  the vindex directly. Simple to deploy but memory-intensive.
- **[Remote (larql-server)](deployment/remote.md)** -- A dedicated server process
  owns the mmap and activation cache. All PostgreSQL backends are thin HTTP clients.
- **[Grid (larql-router)](deployment/grid.md)** -- Multiple servers, each hosting
  a shard of the model's layers. Enables models too large for a single host.

## Documentation Overview

| Chapter | Description |
|---------|-------------|
| [Architecture](architecture.md) | System design, crate structure, data flow |
| [Deployment](deployment/overview.md) | Production deployment across all topologies |
| [Operations](operations/tuning.md) | GUC reference, monitoring, troubleshooting |
| [Compatibility](compatibility/versioning.md) | Version strategy, upgrade procedures |
| [Performance](performance/benchmarks.md) | Benchmark data and optimization strategies |
| [Development](development/contributing.md) | Build from source, run tests |

## See Also

- [SECURITY.md](https://codeberg.org/gregburd/larql/src/branch/main/SECURITY.md) -- Security policy (project root)
