#!/usr/bin/env python3
"""Real-model verification for the BitNet serving path.

Asserts the three claims PR #480 makes, against microsoft/bitnet-b1.58-2B-4T:

  1. a request with NO `mode` field returns correct predictions (this is
     what pg_infer's remote backend posts, and the bug this fixes is that
     it silently defaulted to walk-mode on a container with no gate
     vectors and returned garbage);
  2. explicit mode:dense agrees with it;
  3. explicit mode:walk *also* agrees, i.e. is_dense_only() coerces it,
     rather than answering from an empty KNN store.
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:28080"
PROMPT = "The capital of France is"


def infer(body):
    req = urllib.request.Request(
        f"{BASE}/v1/infer",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def show(label, d):
    print(f"{label}")
    print(f"   mode={d.get('mode')}  latency_ms={d.get('latency_ms')}")
    for p in d.get("predictions", [])[:5]:
        print("   {:14} {:.4f}".format(p["token"], p["probability"]))
    return d


cases = {
    "no mode (what pg_infer sends)": {"prompt": PROMPT, "top": 5},
    "explicit mode=dense": {"prompt": PROMPT, "top": 5, "mode": "dense"},
    "explicit mode=walk (must coerce)": {"prompt": PROMPT, "top": 5, "mode": "walk"},
}

results = {}
for label, body in cases.items():
    results[label] = show(label, infer(body))
    print()

failures = []
for label, d in results.items():
    top = (d.get("predictions") or [{}])[0]
    # GPT-2 byte-BPE marks a leading space as U+0120 (Ġ).
    tok = top.get("token", "").replace("\u0120", " ").strip()
    prob = top.get("probability", 0.0)
    if tok != "Paris":
        failures.append(f"{label}: top token is {tok!r}, expected 'Paris'")
    if prob < 0.90:
        failures.append(f"{label}: Paris at {prob:.4f}, expected >= 0.90")
    if d.get("mode") != "bitnet":
        failures.append(f"{label}: mode is {d.get('mode')!r}, expected 'bitnet'")

# All three must agree to 4dp: the coercion must not change the answer.
probs = [round((d["predictions"][0])["probability"], 4) for d in results.values()]
if len(set(probs)) != 1:
    failures.append(f"the three modes disagree: {probs}")

if failures:
    print("FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)

print(f"ALL PASS: Paris @ {probs[0]:.4f} on all three request shapes, mode=bitnet")
