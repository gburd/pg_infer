# Deployment Overview

This chapter covers production deployment of pg_infer across all three
topologies: local, remote, and grid.

## Prerequisites

| Component | Source | Notes |
|-----------|--------|-------|
| PostgreSQL 18+ | postgresql.org | pg_infer uses pgrx 0.17 which targets PG18 |
| pg_infer extension | This repository | `cargo pgrx install --release` |
| larql-server | Upstream larql repository | Required for remote/grid topologies |
| Vindex directory | Pre-built from a model checkpoint | `.vindex` directory on disk |

**Important**: `larql-server` is an external binary built from the upstream
larql project, not from this repository. Build it separately:

```sh
git clone https://codeberg.org/gregburd/larql && cd larql
cargo build --release -p larql-server
cp target/release/larql-server /usr/local/bin/
```

## Choosing a Topology

| Topology | Best For | Concurrent Users | Memory Model |
|----------|----------|------------------|--------------|
| [Local (mmap)](local.md) | Single-user exploration, dev, small models | 1-3 | Per-backend gate cache |
| [Remote (larql-server)](remote.md) | Production, multi-user, models > 1B params | Unlimited | Shared activation cache |
| [Grid (larql-router)](grid.md) | Models too large for one host, HA | Unlimited | Distributed shards |

## Security Considerations

- larql-server has no built-in authentication. Use network-level controls
  (firewall rules, VPC, Unix socket permissions) to restrict access.
- UDS is preferred for same-host deployments: no TCP exposure, lower latency.
- Vindex files are read-only; larql-server never writes to them.
- The `infer.auto_download` GUC (default true) allows downloading from
  HuggingFace. Set to `false` in production environments where models should
  be pre-staged.
