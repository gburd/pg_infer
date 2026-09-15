#!/usr/bin/env python3
"""Decode-throughput benchmark for the BitNet ternary path.

A/B alternated across prompt lengths rather than batched, so thermal or
neighbour drift over the run does not bias whichever shape ran second.
Reports medians; raw samples go to CSV.

Measures wall-clock end-to-end per request, and separately asks the
server what *it* thinks its decode rate was (/v1/runtime), because the
two disagreeing is itself information: a large gap is request-handling
overhead, not model speed.
"""
import csv
import json
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:28080"
REPS = 3

SHORT = "The capital of France is"
LONG = (
    "The following is a detailed encyclopedia entry about the history, "
    "geography, culture, and economy of the French Republic, a country "
    "in Western Europe whose capital city is"
)

CASES = [
    ("infer_short", {"prompt": SHORT, "top": 5}),
    ("infer_long", {"prompt": LONG, "top": 5}),
]
GEN_CASES = [
    ("gen_8tok", 8),
    ("gen_32tok", 32),
]


def post(path, body, timeout=1800):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = r.read()
    return time.perf_counter() - t0, payload


def runtime_perf():
    with urllib.request.urlopen(f"{BASE}/v1/runtime", timeout=60) as r:
        return json.load(r).get("performance", {})


rows = []

# A/B alternate: one rep of every case, then the next rep.
for rep in range(1, REPS + 1):
    for name, body in CASES:
        wall, payload = post("/v1/infer", body)
        d = json.loads(payload)
        rows.append(
            {
                "rep": rep,
                "case": name,
                "wall_s": round(wall, 3),
                "server_latency_ms": d.get("latency_ms"),
                "tokens_out": len(d.get("predictions") or []),
                "decode_tps": None,
            }
        )
        print(f"  rep{rep} {name:12} wall={wall:7.3f}s server={d.get('latency_ms')}ms")

    for name, ntok in GEN_CASES:
        # stream=True: a --keep-quant container has no dense weights, so
        # the non-streaming batch loop is refused by design. Streaming is
        # the ternary generation path.
        wall, payload = post(
            "/v1/completions",
            {"model": "bitnet2b", "prompt": SHORT, "max_tokens": ntok, "stream": True},
        )
        perf = runtime_perf()
        tps = perf.get("decode_tokens_per_second")
        rows.append(
            {
                "rep": rep,
                "case": name,
                "wall_s": round(wall, 3),
                "server_latency_ms": perf.get("last_request_latency_ms"),
                "tokens_out": ntok,
                "decode_tps": round(tps, 4) if tps else None,
            }
        )
        eff = ntok / wall if wall else 0
        print(
            f"  rep{rep} {name:12} wall={wall:7.3f}s "
            f"eff={eff:5.2f} tok/s server_tps={tps}"
        )

with open("/data/work/bench_bitnet.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\nMEDIANS")
for name in [c[0] for c in CASES] + [c[0] for c in GEN_CASES]:
    walls = [r["wall_s"] for r in rows if r["case"] == name]
    if not walls:
        continue
    med = statistics.median(walls)
    spread = max(walls) - min(walls)
    extra = ""
    if name.startswith("gen_"):
        ntok = int(name.split("_")[1].replace("tok", ""))
        extra = f"  ({ntok / med:.2f} tok/s effective)"
    print(f"  {name:12} median={med:7.3f}s  spread={spread:6.3f}s{extra}")

print("\nraw samples: /data/work/bench_bitnet.csv")
