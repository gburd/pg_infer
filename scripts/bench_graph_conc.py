#!/usr/bin/env python3
"""Concurrency behaviour of pg_infer's graph surface.

describe() rather than infer(): 24 of pg_infer's 26 SQL functions are graph
queries, and upstream's own thesis is that the database surface is the
differentiator, not tok/s. So the useful question is how the surface people
would actually query behaves under load.

Two things measured, and they are different questions:
  1. throughput and latency as concurrency rises;
  2. whether concurrency changes the ANSWER -- a correctness question a
     throughput number hides.

Separate psql processes, so each gets its own backend and its own
RemoteBackend runtime thread; one session would serialise and measure
nothing. A/B alternated across levels so drift does not bias the higher one.
"""
import concurrent.futures as cf
import csv
import statistics
import subprocess
import sys
import time

PSQL = "/data/pgrx/18.6/pgrx-install/bin/psql"
PORT = "28818"
MODEL = "bitnet"
LEVELS = [1, 2, 4, 8, 16]
REPS = 3
SQL = (
    f"SELECT relation, target, round(confidence::numeric,2) "
    f"FROM describe('France', '{MODEL}') ORDER BY confidence DESC LIMIT 1;"
)


def one():
    t0 = time.perf_counter()
    p = subprocess.run(
        [PSQL, "-p", PORT, "-d", "postgres", "-tA", "-c", SQL],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        return dt, None, (p.stderr or "?").strip().splitlines()[0][:150]
    out = (p.stdout or "").strip()
    return dt, out, None


rows, errors, answers = [], [], set()

for rep in range(1, REPS + 1):
    for n in LEVELS:
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            res = list(ex.map(lambda _: one(), range(n)))
        span = time.perf_counter() - t0
        lat = [r[0] for r in res]
        for _, ans, err in res:
            if err:
                errors.append(err)
            elif ans:
                answers.add(ans)
        rows.append(
            {
                "rep": rep,
                "concurrency": n,
                "wall_span_s": round(span, 4),
                "p50_s": round(statistics.median(lat), 4),
                "max_s": round(max(lat), 4),
                "qps": round(n / span, 3),
                "errors": sum(1 for r in res if r[2]),
            }
        )
        print(
            f"  rep{rep} c={n:<3} span={span:7.3f}s p50={statistics.median(lat):6.3f}s "
            f"qps={n / span:7.3f} errors={sum(1 for r in res if r[2])}"
        )

with open("/data/work/bench_graph_conc.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\nMEDIANS BY CONCURRENCY")
base = None
for n in LEVELS:
    sel = [r for r in rows if r["concurrency"] == n]
    q = statistics.median([r["qps"] for r in sel])
    p50 = statistics.median([r["p50_s"] for r in sel])
    if base is None:
        base = q
    print(f"  c={n:<3} qps={q:7.3f}  p50={p50:6.3f}s  speedup={q / base:5.2f}x")

print(f"\nDISTINCT ANSWERS ACROSS {sum(LEVELS) * REPS} QUERIES: {answers}")
ok = True
if len(answers) != 1:
    print("FAIL: concurrency changed the answer")
    ok = False
if errors:
    print(f"FAIL: {len(errors)} errors, first: {errors[0]}")
    ok = False
print("\nraw: /data/work/bench_graph_conc.csv")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
