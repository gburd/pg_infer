#!/usr/bin/env python3
"""Does describe_many() actually amortise the per-call cost?

The claim being tested: describe() has a fixed per-call latency, PostgreSQL
evaluates a set-returning function once per row, so an N-row join pays that
latency N times sequentially. describe_many() issues the N requests
concurrently over one connection, so it should cost roughly one round trip
plus the server's own concurrency limit -- not N.

Compares, for the same N entities:
  A) N separate describe() calls        (what a join does today)
  B) one describe_many(ARRAY[...]) call (the batch path)

Also checks A and B return the SAME edges, because a faster wrong answer is
not an optimisation.
"""
import statistics
import subprocess
import sys
import time

PSQL = "/data/pgrx/18.6/pgrx-install/bin/psql"
PORT = "28818"
M = "bitnet"
REPS = 3
ENTITIES = [
    "France", "Germany", "Einstein", "Paris", "London",
    "physics", "capital", "river", "mountain", "language",
]


def q(sql, timeout=3600):
    t0 = time.perf_counter()
    p = subprocess.run(
        [PSQL, "-p", PORT, "-d", "postgres", "-tA", "-F", "|", "-c", sql],
        capture_output=True, text=True, timeout=timeout,
    )
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        return dt, None, (p.stderr or "").strip().splitlines()[0][:160]
    return dt, (p.stdout or "").strip().splitlines(), None


def sql_list(n):
    return ", ".join("'" + e.replace("'", "''") + "'" for e in ENTITIES[:n])


print(f"{'N':>4}  {'sequential':>12}  {'batched':>12}  {'speedup':>8}  {'rows match':>10}")
print("-" * 58)

for n in (1, 2, 5, 10):
    seq_t, bat_t = [], []
    seq_rows = bat_rows = None
    for _ in range(REPS):
        # A) N separate calls, as a per-row join would issue them.
        t0 = time.perf_counter()
        acc = []
        for e in ENTITIES[:n]:
            _, rows, err = q(
                f"SELECT relation,target FROM describe('{e}', '{M}') "
                f"ORDER BY confidence DESC"
            )
            if err:
                print(f"  seq error: {err}")
                sys.exit(1)
            acc.extend(rows or [])
        seq_t.append(time.perf_counter() - t0)
        seq_rows = len(acc)

        # B) one batched call.
        dt, rows, err = q(
            f"SELECT relation,target FROM describe_many(ARRAY[{sql_list(n)}], '{M}')"
        )
        if err:
            print(f"  batch error: {err}")
            sys.exit(1)
        bat_t.append(dt)
        bat_rows = len(rows or [])

    s, b = statistics.median(seq_t), statistics.median(bat_t)
    match = "yes" if seq_rows == bat_rows else f"NO {seq_rows}v{bat_rows}"
    print(f"{n:>4}  {s * 1000:>10.1f}ms  {b * 1000:>10.1f}ms  {s / b:>7.2f}x  {match:>10}")

# Same edges, not just the same count.
_, seq, _ = q(
    f"SELECT relation,target FROM describe('France','{M}') ORDER BY relation,target"
)
_, bat, _ = q(
    f"SELECT relation,target FROM describe_many(ARRAY['France'],'{M}') "
    f"ORDER BY relation,target"
)
print()
if seq == bat:
    print(f"CONTENT IDENTICAL for France ({len(seq)} edges)")
else:
    print(f"CONTENT DIFFERS: seq={len(seq or [])} batch={len(bat or [])}")
    sys.exit(1)
