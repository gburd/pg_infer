#!/usr/bin/env python3
"""pg_infer qualification at scale against a real larql-server.

Two things worth measuring, and they are different questions:

  1. throughput/latency of SQL-visible inference under concurrency, which
     is what an operator feels;
  2. whether concurrency changes the *answer*, which is a correctness
     question that a throughput number hides.

Concurrency is driven with separate psql processes so each gets its own
backend and its own RemoteBackend runtime thread -- a single session would
serialise and measure nothing.

A/B alternated across concurrency levels (not all of level 1, then all of
level 4) so drift over the run does not bias the higher level.
"""
import concurrent.futures as cf
import csv
import json
import re
import statistics
import subprocess
import sys
import time

PSQL = "/data/pgrx/18.6/pgrx-install/bin/psql"
PORT = "28818"
MODEL = "bitnet"
PROMPT = "The capital of France is"
LEVELS = [1, 2, 4, 8]
REPS = 3


def one_query():
    sql = f"SELECT token, probability FROM infer('{PROMPT}', 1, '{MODEL}');"
    t0 = time.perf_counter()
    p = subprocess.run(
        [PSQL, "-p", PORT, "-d", "postgres", "-tA", "-F", "|", "-c", sql],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        return dt, None, p.stderr.strip()[:200]
    line = (p.stdout or "").strip().splitlines()
    if not line:
        return dt, None, "empty result"
    tok, _, prob = line[0].partition("|")
    return dt, (tok.strip(), float(prob)), None


rows = []
errors = []
answers = set()

for rep in range(1, REPS + 1):
    for n in LEVELS:
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            out = list(ex.map(lambda _: one_query(), range(n)))
        span = time.perf_counter() - t0
        lat = [o[0] for o in out]
        for _, ans, err in out:
            if err:
                errors.append(err)
            if ans:
                answers.add((ans[0].strip(), round(ans[1], 4)))
        rows.append(
            {
                "rep": rep,
                "concurrency": n,
                "wall_span_s": round(span, 3),
                "p50_latency_s": round(statistics.median(lat), 3),
                "max_latency_s": round(max(lat), 3),
                "qps": round(n / span, 4),
                "errors": sum(1 for o in out if o[2]),
            }
        )
        print(
            f"  rep{rep} c={n:<2} span={span:7.3f}s p50={statistics.median(lat):6.3f}s "
            f"qps={n / span:5.3f} errors={sum(1 for o in out if o[2])}"
        )

with open("/data/work/bench_pg_infer.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\nMEDIANS BY CONCURRENCY")
for n in LEVELS:
    sel = [r for r in rows if r["concurrency"] == n]
    print(
        f"  c={n:<2} span={statistics.median([r['wall_span_s'] for r in sel]):7.3f}s "
        f"p50={statistics.median([r['p50_latency_s'] for r in sel]):6.3f}s "
        f"qps={statistics.median([r['qps'] for r in sel]):5.3f}"
    )

print(f"\nDISTINCT ANSWERS ACROSS ALL {sum(LEVELS) * REPS} QUERIES: {answers}")
ok = True
if len(answers) != 1:
    print("FAIL: concurrency changed the answer")
    ok = False
if errors:
    print(f"FAIL: {len(errors)} errors, first: {errors[0]}")
    ok = False
tok = next(iter(answers))[0] if answers else ""
if "Paris" not in tok:
    print(f"FAIL: top token {tok!r} is not Paris")
    ok = False

print("\nraw: /data/work/bench_pg_infer.csv")
print("QUALIFICATION PASS" if ok else "QUALIFICATION FAIL")
sys.exit(0 if ok else 1)
