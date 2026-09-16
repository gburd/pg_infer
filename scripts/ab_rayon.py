#!/usr/bin/env python3
"""A/B: does rayon-across-layers help, and at what?

Two distinct questions that a single number would conflate:

  LATENCY   one query at a time. Rayon should win here -- a 12-layer
            describe() becomes 12 concurrent gemv calls.

  THROUGHPUT many queries at once. Rayon should NOT win here if the scan is
            memory-bound: the same bytes are read either way, and at c=16
            the box already has 16 queries' worth of work in flight. If it
            *does* win, the bottleneck was scheduling, not bandwidth --
            which would be worth knowing.

Alternates baseline/rayon per rep rather than running all of one then all of
the other, so drift over the run cannot masquerade as a difference.
"""
import json
import statistics
import subprocess
import sys
import time
import urllib.request

PORT = 28080
REPS = 5
BANDS = [("knowledge", 12), ("all", 30)]
CONC = [1, 4, 16, 32]


def wait_up(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False


def one(entity="France", band="knowledge"):
    url = f"http://127.0.0.1:{PORT}/v1/describe?entity={entity}&limit=20&band={band}"
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=600) as r:
        body = r.read()
    return time.perf_counter() - t0, body


def start(binary):
    subprocess.run(["pkill", "-f", "larql-server /data/models"], capture_output=True)
    time.sleep(3)
    p = subprocess.Popen(
        [binary, "/data/models/bitnet-browse.vindex", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if not wait_up():
        print(f"FATAL: {binary} did not come up")
        sys.exit(1)
    for _ in range(3):          # warm the f16 decode cache for every band
        one(band="all")
    return p


import concurrent.futures as cf

BINS = {"baseline": "/data/work/srv_baseline", "rayon": "/data/work/srv_rayon"}
lat = {k: {b: [] for b, _ in BANDS} for k in BINS}
thr = {k: {c: [] for c in CONC} for k in BINS}
answers = {k: set() for k in BINS}

for rep in range(1, REPS + 1):
    for name, path in BINS.items():          # alternate within each rep
        proc = start(path)
        try:
            for band, _n in BANDS:
                dt, body = one(band=band)
                lat[name][band].append(dt)
                top = (json.loads(body).get("edges") or [{}])[0]
                answers[name].add((top.get("target"), round(top.get("gate_score", 0), 2)))
            for c in CONC:
                t0 = time.perf_counter()
                with cf.ThreadPoolExecutor(max_workers=c) as ex:
                    list(ex.map(lambda _: one(), range(c)))
                span = time.perf_counter() - t0
                thr[name][c].append(c / span)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    print(f"  rep {rep}/{REPS} done", flush=True)

print("\n=== SINGLE-QUERY LATENCY (medians) ===")
print(f"{'band':>10} {'layers':>7} {'baseline':>10} {'rayon':>10} {'speedup':>8}")
for band, n in BANDS:
    b = statistics.median(lat["baseline"][band])
    r = statistics.median(lat["rayon"][band])
    print(f"{band:>10} {n:>7} {b * 1000:>9.1f}ms {r * 1000:>9.1f}ms {b / r:>7.2f}x")

print("\n=== THROUGHPUT (qps, medians) ===")
print(f"{'conc':>6} {'baseline':>10} {'rayon':>10} {'speedup':>8}")
for c in CONC:
    b = statistics.median(thr["baseline"][c])
    r = statistics.median(thr["rayon"][c])
    print(f"{c:>6} {b:>10.1f} {r:>10.1f} {b and r / b:>7.2f}x")

print("\n=== CORRECTNESS ===")
print(f"  baseline top edge: {answers['baseline']}")
print(f"  rayon    top edge: {answers['rayon']}")
ok = answers["baseline"] == answers["rayon"] and len(answers["rayon"]) <= len(BANDS)
print("  IDENTICAL" if ok else "  *** DIVERGED ***")
sys.exit(0 if ok else 1)
