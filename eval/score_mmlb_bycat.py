#!/usr/bin/env python3
"""Per-category accuracy for MMLongBench-Doc 847. Join predictions (judge.correct) to dataset (mmlb_sources)."""
import json, os, collections
DATASET = os.environ["DATASET"]
PREDS = os.environ["PREDS"]

def key_of(r):
    for k in ("qa_id", "sample_id", "id"):
        if r.get(k) is not None: return str(r[k])
    return None

ds = {}
for l in open(DATASET):
    d = json.loads(l); ds[key_of(d)] = d

preds = [json.loads(l) for l in open(PREDS)]
CATS = ["Text", "Table", "Chart", "Figure", "Layout"]
tot = collections.Counter(); ok = collections.Counter()
overall_ok = overall_n = matched = 0
for r in preds:
    k = key_of(r)
    d = ds.get(k)
    if d is None: continue
    matched += 1
    correct = isinstance(r.get("judge"), dict) and r["judge"].get("correct") is True
    overall_n += 1; overall_ok += int(correct)
    for c in set(d.get("mmlb_sources", [])):
        if c in CATS:
            tot[c] += 1; ok[c] += int(correct)

print(f"matched preds<->dataset = {matched}/{len(preds)}")
print(f"OVERALL(recompute) = {overall_ok}/{overall_n} = {100*overall_ok/max(1,overall_n):.2f}%")
print("PER-CATEGORY (VRAG-RL/VISOR column style):")
for c in CATS:
    a = 100*ok[c]/tot[c] if tot[c] else 0.0
    print(f"  {c:8s}: {ok[c]:3d}/{tot[c]:3d} = {a:.2f}%   (VISOR n={ {'Text':291,'Table':217,'Chart':178,'Figure':290,'Layout':118}[c] })")
