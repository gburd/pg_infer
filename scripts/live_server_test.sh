#!/usr/bin/env bash
# Live integration smoke test: pg_infer + a real larql-server.
#
# Prerequisites (the script exits early with a clear message if any
# are missing):
#   - larql-server binary on PATH, or LARQL_SERVER env var pointing at one
#   - a vindex directory (LARQL_VINDEX env var)
#   - a running PostgreSQL with pg_infer installed (PG* env vars honored)
#
# Flow:
#   1. Launch larql-server on a free HTTP port in the background.
#   2. Wait for /v1/health to respond.
#   3. Register the model via infer_create_model_remote().
#   4. Exercise describe(), walk(), similar_to(), similar_to_many(),
#      show_layers(), show_relations() through pg_infer.
#   5. Tear down.

set -euo pipefail

die()  { printf 'live-server-test: %s\n' "$*" >&2; exit 1; }
info() { printf '\033[1;34m[live]\033[0m %s\n' "$*" >&2; }

LARQL_SERVER=${LARQL_SERVER:-$(command -v larql-server || true)}
[[ -n "$LARQL_SERVER" ]] || die "larql-server binary not found; set LARQL_SERVER or add to PATH"

LARQL_VINDEX=${LARQL_VINDEX:?set LARQL_VINDEX to a vindex directory}
[[ -d "$LARQL_VINDEX" ]] || die "LARQL_VINDEX '$LARQL_VINDEX' is not a directory"

PSQL=${PSQL:-psql}
PGDATABASE=${PGDATABASE:-postgres}
export PGDATABASE

MODEL_NAME=${MODEL_NAME:-live_test_model}

# Pick a free port.
port=$(python3 - <<'PY'
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0))
print(s.getsockname()[1]); s.close()
PY
)
base_url="http://127.0.0.1:${port}"

info "launching larql-server on ${base_url} with vindex ${LARQL_VINDEX}"
log=$(mktemp)
trap 'info "tearing down server (log: $log)"; kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

"$LARQL_SERVER" "$LARQL_VINDEX" --port "$port" > "$log" 2>&1 &
server_pid=$!

# Wait up to 30s for /v1/health.
info "waiting for server readiness..."
for _ in $(seq 1 60); do
    if curl -sf "$base_url/v1/health" >/dev/null 2>&1; then
        info "server is up"
        break
    fi
    sleep 0.5
    if ! kill -0 "$server_pid" 2>/dev/null; then
        die "server process exited prematurely; see log: $log"
    fi
done
curl -sf "$base_url/v1/health" >/dev/null 2>&1 || die "server did not become ready; see log: $log"

run_sql() {
    info "SQL: $1"
    "$PSQL" -v ON_ERROR_STOP=1 -c "$1"
}

info "registering model '${MODEL_NAME}' → ${base_url}"
run_sql "CREATE EXTENSION IF NOT EXISTS pg_infer;"
run_sql "SELECT infer_drop_model('${MODEL_NAME}');" || true
run_sql "SELECT infer_create_model_remote('${MODEL_NAME}', '${base_url}');"

info "exercise describe()"
run_sql "SELECT * FROM describe('France', model => '${MODEL_NAME}') LIMIT 5;"

info "exercise walk()"
run_sql "SELECT * FROM walk('The capital of France is', top => 5, model => '${MODEL_NAME}');"

info "exercise similar_to()"
run_sql "SELECT similar_to('France', 'Paris', '${MODEL_NAME}');"

info "exercise similar_to_many() — batch fanout"
run_sql "SELECT unnest(similar_to_many(ARRAY['Paris', 'Lyon', 'Berlin', 'banana'], 'France', '${MODEL_NAME}'));"

info "exercise show_layers()"
run_sql "SELECT * FROM infer_show_layers(model => '${MODEL_NAME}') LIMIT 5;"

info "exercise show_relations()"
run_sql "SELECT * FROM infer_show_relations(model => '${MODEL_NAME}') LIMIT 5;"

info "cleanup"
run_sql "SELECT infer_drop_model('${MODEL_NAME}');"

info "✅ all live-server round trips succeeded"
