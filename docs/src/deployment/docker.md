# Docker Deployment

Docker Compose examples for deploying pg_infer with larql-server.

## Remote Topology (Single Server)

A minimal `docker-compose.yml` for the remote topology:

```yaml
services:
  larql-server:
    image: your-registry/larql-server:latest
    command: ["/data/model.vindex", "--port", "8080"]
    volumes:
      - ./vindexes:/data:ro
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/v1/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  postgres:
    image: postgres:18
    environment:
      POSTGRES_DB: app
      POSTGRES_PASSWORD: secret
    volumes:
      - ./pg_infer.so:/usr/lib/postgresql/18/lib/pg_infer.so:ro
      - ./pg_infer.control:/usr/share/postgresql/18/extension/pg_infer.control:ro
      - ./pg_infer--1.0.0.sql:/usr/share/postgresql/18/extension/pg_infer--1.0.0.sql:ro
    ports:
      - "5432:5432"
    depends_on:
      larql-server:
        condition: service_healthy
```

After startup:

```sql
CREATE EXTENSION pg_infer;
SELECT infer_create_model_remote('model', 'http://larql-server:8080');
```

## Grid Topology (Multi-Server)

A `docker-compose.yml` for the grid topology with layer sharding:

```yaml
services:
  shard-0:
    image: your-registry/larql-server:latest
    command: ["/data/model-shard-0.vindex", "--port", "8080"]
    volumes:
      - ./vindexes:/data:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/v1/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  shard-1:
    image: your-registry/larql-server:latest
    command: ["/data/model-shard-1.vindex", "--port", "8080"]
    volumes:
      - ./vindexes:/data:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/v1/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  router:
    image: your-registry/larql-router:latest
    command: ["--servers", "http://shard-0:8080,http://shard-1:8080", "--port", "9090"]
    ports:
      - "9090:9090"
    depends_on:
      shard-0:
        condition: service_healthy
      shard-1:
        condition: service_healthy

  postgres:
    image: postgres:18
    environment:
      POSTGRES_DB: app
      POSTGRES_PASSWORD: secret
    volumes:
      - ./pg_infer.so:/usr/lib/postgresql/18/lib/pg_infer.so:ro
      - ./pg_infer.control:/usr/share/postgresql/18/extension/pg_infer.control:ro
      - ./pg_infer--1.0.0.sql:/usr/share/postgresql/18/extension/pg_infer--1.0.0.sql:ro
    ports:
      - "5432:5432"
    depends_on:
      router:
        condition: service_started
```

After startup:

```sql
CREATE EXTENSION pg_infer;
SELECT infer_create_model_grid('model', 'http://router:9090');
```

## Building the Images

### larql-server Image

```dockerfile
FROM rust:1.80-slim AS builder
WORKDIR /src
RUN git clone https://codeberg.org/gregburd/larql .
RUN cargo build --release -p larql-server

FROM debian:bookworm-slim
COPY --from=builder /src/target/release/larql-server /usr/local/bin/
ENTRYPOINT ["larql-server"]
```

### pg_infer Extension

The extension must be compiled against the exact PostgreSQL version used in the
target image. Use the pgrx Docker workflow:

```dockerfile
FROM postgres:18

# Copy pre-built extension files
COPY pg_infer.so /usr/lib/postgresql/18/lib/
COPY pg_infer.control /usr/share/postgresql/18/extension/
COPY pg_infer--1.0.0.sql /usr/share/postgresql/18/extension/
```

## Health Checks

Both services expose HTTP health endpoints suitable for Docker health checks
and orchestrator probes:

- **larql-server**: `GET /v1/health` returns `{"status":"ok"}` with HTTP 200
- **larql-router**: `GET /v1/health` returns aggregated health of all shards

See [Monitoring](../operations/monitoring.md) for detailed health check configuration.
