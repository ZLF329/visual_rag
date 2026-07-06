#!/usr/bin/env python3
"""ViDoSeek accuracy by source_type (2d_layout/chart/table/text), and cross-tab with hop type."""
import json, os, collections

DATASET = "/root/autodl-tmp/vidoseek/eval/test_vidoseek.jsonl"
PREDS = os.environ["PREDS"]

def key_of(r):
    for k in ("qa_id", "sample_id", "id"):
        if r.get(k) is not None: return str(r[k])
    return None

ds = {}
for l in open(DATASET):
    d = json.loads(l)
    ds[key_of(d)] = d

preds = [json.loads(l) for l in open(PREDS)]
tot = collections.Counter(); ok = collections.Counter()
# cross-tab source_type x hop
cross_tot = collections.Counter(); cross_ok = collections.Counter()
n = 0
for r in preds:
    d = ds.get(key_of(r))
    if d is None: continue
    n += 1
    correct = isinstance(r.get("judge"), dict) and r["judge"].get("correct") is True
    st = d.get("source_type", "unknown")
    hop = d.get("vido_split", "unknown")
    tot[st] += 1; ok[st] += int(correct)
    cross_tot[(st, hop)] += 1; cross_ok[(st, hop)] += int(correct)

print(f"matched = {n}/{len(preds)}")
print()
print("=== by source_type ===")
for st in sorted(tot, key=lambda k: -tot[k]):
    a = 100 * ok[st] / tot[st]
    print(f"  {st:12s}: {ok[st]:4d}/{tot[st]:4d} = {a:.2f}%")

print()
print("=== cross-tab source_type x hop (Extraction/Logic) ===")
for st in sorted(set(k[0] for k in cross_tot), key=lambda s: -tot[s]):
    for hop in ("Extraction", "Logic"):
        t = cross_tot.get((st, hop), 0)
        if t == 0: continue
        a = 100 * cross_ok.get((st, hop), 0) / t
        print(f"  {st:12s} x {hop:11s}: {cross_ok.get((st,hop),0):3d}/{t:3d} = {a:.2f}%")
