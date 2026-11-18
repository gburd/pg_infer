#!/usr/bin/env python3
"""Measure Laya relation classification, using the model's REAL input format.

My first attempt scored each option as its own sequence and mean-pooled the
encoder. That was wrong twice over: it answered `located_in` for every input
(including "a dog is an example of a mammal"), and it cost 6 sequences per
edge. The reference implementation (github.com/aovestdipaperino/laya-rust)
shows the actual layout:

    [CLS] question [SEP] ([MASK] option)* [SEP] state [SEP]

ALL options live in ONE sequence, each preceded by a [MASK] marker, and the
head reads one logit per marker position. So it is one forward pass per
EDGE, not per option -- and that is the property that decides whether this
is affordable.

Pipeline: encoder -> head (2 transformer layers, with a question-type
embedding added) -> scorer (per-marker logit) -> softmax at the per-shape
temperature.
"""
import json
import os
import statistics
import time

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL = "/data/models/laya-td"
RELATIONS = ["related_to", "part_of", "instance_of",
             "located_in", "has_property", "similar_to"]
OPTIONS = [
    "the two things are loosely related",
    "the second thing is a part or region of the first",
    "the first thing is an example or instance of the second",
    "the first thing is located in the second",
    "the second thing is a property or attribute of the first",
    "the two things are similar kinds of thing",
]
QUESTION = "How are the two things related?"
QTYPE_CHOICE = 0   # type_emb row order: choice / score / noul


def band_phrase(layer, n):
    b = n // 3
    if layer < b:
        return "an early layer, where the model handles grammar and word form"
    if layer >= n - b:
        return "a late layer, where the model is choosing its next word"
    return "a middle layer, where the model stores facts about the world"


def state_for(entity, target, secondaries, layer, n):
    s = (f'A language model was asked about "{entity}". Inside '
         f'{band_phrase(layer, n)}, it strongly associated '
         f'"{entity}" with "{target}".')
    if secondaries:
        s += f' Words that came up alongside "{target}" were: {", ".join(secondaries)}.'
    return s


class Head(nn.Module):
    """Laya's decision head: 2 ModernBERT-style layers over encoder states."""

    def __init__(self, sd, hidden=1024, ff=4096, heads=16, nlayers=2):
        super().__init__()
        self.nlayers = nlayers
        self.q = nn.ModuleList()
        for i in range(nlayers):
            d = nn.ModuleDict({
                "norm1": nn.LayerNorm(hidden),
                "linear1": nn.Linear(hidden, ff),
                "linear2": nn.Linear(ff, hidden),
                "norm2": nn.LayerNorm(hidden),
                "attn_qkv": nn.Linear(hidden, 3 * hidden),
                "attn_out": nn.Linear(hidden, hidden),
            })
            self.q.append(d)
        self.heads = heads
        self.hidden = hidden
        self._load(sd)

    def _load(self, sd):
        with torch.no_grad():
            for i, d in enumerate(self.q):
                p = f"head.layers.{i}."
                for name, mod in (("norm1", d["norm1"]), ("norm2", d["norm2"]),
                                  ("linear1", d["linear1"]), ("linear2", d["linear2"])):
                    if p + name + ".weight" in sd:
                        mod.weight.copy_(sd[p + name + ".weight"])
                    if p + name + ".bias" in sd:
                        mod.bias.copy_(sd[p + name + ".bias"])
                for key, mod in (("attn.Wqkv", d["attn_qkv"]), ("attn.Wo", d["attn_out"])):
                    if p + key + ".weight" in sd:
                        mod.weight.copy_(sd[p + key + ".weight"])
                    if p + key + ".bias" in sd:
                        mod.bias.copy_(sd[p + key + ".bias"])

    def forward(self, h, mask):
        B, L, D = h.shape
        add = (1.0 - mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
        for d in self.q:
            x = d["norm1"](h)
            qkv = d["attn_qkv"](x).view(B, L, 3, self.heads, D // self.heads)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=add)
            h = h + d["attn_out"](a.transpose(1, 2).reshape(B, L, D))
            y = d["norm2"](h)
            h = h + d["linear2"](torch.nn.functional.gelu(d["linear1"](y)))
        return h


class Scorer(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.norm = nn.LayerNorm(1024)
        self.l1 = nn.Linear(1024, 1024)
        self.l2 = nn.Linear(1024, 1)
        with torch.no_grad():
            self.norm.weight.copy_(sd["scorer.0.weight"])
            self.norm.bias.copy_(sd["scorer.0.bias"])
            self.l1.weight.copy_(sd["scorer.1.weight"])
            self.l1.bias.copy_(sd["scorer.1.bias"])
            self.l2.weight.copy_(sd["scorer.3.weight"])
            self.l2.bias.copy_(sd["scorer.3.bias"])

    def forward(self, h):
        return self.l2(nn.functional.gelu(self.l1(self.norm(h)))).squeeze(-1)


def build_sequence(tok, state, options, question=QUESTION):
    """[CLS] question [SEP] ([MASK] option)* [SEP] state [SEP], one per edge."""
    cls_, sep, msk = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id
    ids = [cls_] + tok(question, add_special_tokens=False)["input_ids"] + [sep]
    markers = []
    for o in options:
        markers.append(len(ids))
        ids.append(msk)
        ids += tok(o, add_special_tokens=False)["input_ids"]
    ids.append(sep)
    ids += tok(state, add_special_tokens=False)["input_ids"]
    ids.append(sep)
    return ids, markers


def make_batch(n):
    seeds = [
        ("France", "Paris", ["city", "capital"], 15),
        ("France", "Europe", ["continent"], 15),
        ("France", "wine", ["bottle", "drink"], 18),
        ("France", "Macron", ["president"], 12),
        ("France", "French", ["language"], 4),
        ("France", "Lyon", ["city"], 16),
        ("France", "cheese", ["food"], 19),
        ("France", "Seine", ["river"], 14),
    ]
    out = []
    while len(out) < n:
        e, t, s, l = seeds[len(out) % len(seeds)]
        out.append((e, t, s, l, 30))
    return out[:n]


def main():
    sd = load_file(f"{MODEL}/model.safetensors")
    cfg = json.load(open(f"{MODEL}/rl_agent_config.json"))
    temp = cfg["temperature_by_options"]["choice:6-10"]
    tok = AutoTokenizer.from_pretrained(f"{MODEL}/tokenizer")
    enc_sd = {k[8:]: v for k, v in sd.items() if k.startswith("encoder.")}
    type_emb = sd["type_emb.weight"]
    print(f"temperature(choice:6-10)={temp:.4f}  "
          f"max_len={cfg['max_len']}  head_layers={cfg['head_layers']}")

    devices = [("cpu", torch.float32)]
    if torch.cuda.is_available():
        devices.append(("cuda", torch.bfloat16))
    results = {}

    for dev, dt in devices:
        ecfg = AutoConfig.from_pretrained(f"{MODEL}/encoder")
        enc = AutoModel.from_config(ecfg)
        enc.load_state_dict(enc_sd, strict=False)
        enc = enc.to(device=dev, dtype=dt).eval()
        head = Head(sd).to(device=dev, dtype=dt).eval()
        scorer = Scorer(sd).to(device=dev, dtype=dt).eval()
        te = type_emb.to(device=dev, dtype=dt)
        if dev == "cpu":
            torch.set_num_threads(os.cpu_count())

        def infer(batch):
            seqs = [build_sequence(tok, state_for(*ev), OPTIONS) for ev in batch]
            L = max(len(s[0]) for s in seqs)
            ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
            am = torch.zeros((len(seqs), L), dtype=torch.long)
            for i, (s, _) in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s)
                am[i, :len(s)] = 1
            ids, am = ids.to(dev), am.to(dev)
            with torch.no_grad():
                h = enc(input_ids=ids, attention_mask=am).last_hidden_state
                h = h + te[QTYPE_CHOICE]        # question-type embedding
                h = head(h, am)
                lg = scorer(h)                   # one logit per position
            out = []
            for i, (_, mk) in enumerate(seqs):
                out.append(lg[i, torch.tensor(mk, device=dev)].float())
            return torch.stack(out)

        label = "bf16" if dev == "cuda" else "fp32"
        thr = torch.get_num_threads() if dev == "cpu" else "n/a"
        print(f"\n=== {dev} ({label}, threads={thr}) — 1 SEQUENCE PER EDGE ===")
        print(f"{'edges':>6} {'median ms':>11} {'ms/edge':>9}  top answer")
        for n in (1, 5, 10, 20, 40):
            b = make_batch(n)
            for _ in range(2):
                infer(b)
            if dev == "cuda":
                torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                lg = infer(b)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            med = statistics.median(ts)
            results[f"{dev}/{n}"] = med
            p = torch.softmax(lg[0] / temp, 0)
            k = int(p.argmax())
            print(f"{n:>6} {med:>11.1f} {med/n:>9.2f}  "
                  f"{RELATIONS[k]} p={float(p[k]):.3f}")

        # Sanity: does it discriminate now?
        print("  --- discrimination check (was broken before) ---")
        probes = [
            ("Paris is located in France.", "located_in"),
            ("A dog is an example of a mammal.", "instance_of"),
            ("The wheel is a part of the car.", "part_of"),
            ("The sky is blue in colour.", "has_property"),
        ]
        hits = 0
        for st, want in probes:
            ids, mk = build_sequence(tok, st, OPTIONS)
            t = torch.tensor([ids], device=dev)
            a = torch.ones_like(t)
            with torch.no_grad():
                h = enc(input_ids=t, attention_mask=a).last_hidden_state
                h = head(h + te[QTYPE_CHOICE], a)
                lg = scorer(h)[0, torch.tensor(mk, device=dev)].float()
            p = torch.softmax(lg / temp, 0)
            k = int(p.argmax())
            ok = RELATIONS[k] == want
            hits += ok
            print(f"    {st[:32]:<34} -> {RELATIONS[k]:<13} "
                  f"p={float(p[k]):.3f} {'OK' if ok else 'want ' + want}")
        print(f"    {hits}/{len(probes)} correct")

        del enc, head, scorer
        if dev == "cuda":
            torch.cuda.empty_cache()

    print("\n=== cost against describe()'s 13-18 ms budget ===")
    for dev, _ in devices:
        t = results.get(f"{dev}/20")
        if t:
            print(f"  {dev}: 20 edges = +{t:.0f} ms on a 15 ms query "
                  f"({t/15:.1f}x the whole query)")
    json.dump(results, open("/data/work/laya_cost2.json", "w"), indent=2)


if __name__ == "__main__":
    main()
