#!/usr/bin/env python3
"""Graph-query benchmark: what pg_infer is actually FOR.

The earlier benchmark measured infer() -- next-token generation -- which is
1 of pg_infer's 26 SQL functions and the one upstream explicitly says is
*not* the differentiator ("the differentiated functionality is the
database, not the tok/s"). The other 24 are graph queries: describe(),
walk(), similar_to(), nearest_to(), show_relations().

Those are the ones an operator would put in a query, so those are the ones
whose latency decides whether this is usable in practice. A 5-second
describe() is a batch tool; a 50 ms describe() is something you can join
against.

A/B alternated across query kinds, medians reported, raw CSV written.
"""
import csv
import statistics
import subprocess
import sys
import time

PSQL = "/data/pgrx/18.6/pgrx-install/bin/psql"
PORT = "28818"
M = "bitnet"
REPS = 5

# (label, sql) -- the graph surface, not generation.
QUERIES = [
    ("describe", f"SELECT * FROM describe('France', '{M}')"),
    ("walk", f"SELECT * FROM walk('The capital of France is', 5, '{M}')"),
    ("similar_to", f"SELECT similar_to('France', 'Germany', '{M}')"),
    ("nearest_to", f"SELECT * FROM nearest_to('France', 14, 5, '{M}')"),
    ("show_relations", f"SELECT * FROM infer_show_relations('{M}')"),
    ("show_layers", f"SELECT * FROM infer_show_layers('{M}')"),
    ("show_features", f"SELECT * FROM infer_show_features(14, NULL, NULL, 10, '{M}')"),
    # The generation call, for contrast -- expected to be ~1000x slower.
    ("infer_1tok", f"SELECT * FROM infer('The capital of France is', 1, '{M}')"),
]


def run(sql):
    t0 = time.perf_counter()
    p = subprocess.run(
        [PSQL, "-p", PORT, "-d", "postgres", "-tA", "-c", sql],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        return dt, None, p.stderr.strip().splitlines()[0][:160] if p.stderr else "?"
    return dt, len((p.stdout or "").strip().splitlines()), None


rows = []
first_err = {}

# Warm once so the first sample does not carry connection + mmap cost.
for _, sql in QUERIES:
    run(sql)

for rep in range(1, REPS + 1):
    for label, sql in QUERIES:
        dt, nrows, err = run(sql)
        rows.append(
            {
                "rep": rep,
                "query": label,
                "seconds": round(dt, 4),
                "rows": nrows if nrows is not None else -1,
                "error": err or "",
            }
        )
        if err and label not in first_err:
            first_err[label] = err

with open("/data/work/bench_graph.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{'query':16} {'median':>10} {'min':>9} {'max':>9} {'rows':>6}  note")
print("-" * 72)
for label, _ in QUERIES:
    sel = [r for r in rows if r["query"] == label]
    ok = [r for r in sel if not r["error"]]
    if not ok:
        print(f"{label:16} {'ERROR':>10}                          {first_err.get(label,'')[:30]}")
        continue
    secs = [r["seconds"] for r in ok]
    med = statistics.median(secs)
    unit = f"{med * 1000:.1f} ms" if med < 1 else f"{med:.3f} s"
    print(
        f"{label:16} {unit:>10} {min(secs) * 1000:>7.1f}ms {max(secs) * 1000:>7.1f}ms "
        f"{ok[0]['rows']:>6}"
    )

print("\nraw: /data/work/bench_graph.csv")

# The practical question, stated as a check rather than left to the reader.
gq = [
    statistics.median([r["seconds"] for r in rows if r["query"] == q and not r["error"]])
    for q, _ in QUERIES
    if q != "infer_1tok"
    and any(r["query"] == q and not r["error"] for r in rows)
]
if gq:
    worst = max(gq)
    print(f"\nslowest graph query median: {worst * 1000:.1f} ms")
    print(
        "  interactive (<100ms)" if worst < 0.1
        else "  usable in a query (<1s)" if worst < 1.0
        else "  batch-only (>1s)"
    )
