#!/usr/bin/env python3
"""ViDoSeek Extraction/Logic accuracy. Join predictions(judge.correct) to dataset(vido_split)."""
import json, os, collections
DATASET = os.environ["DATASET"]; PREDS = os.environ["PREDS"]
def key_of(r):
    for k in ("qa_id","sample_id","id"):
        if r.get(k) is not None: return str(r[k])
    return None
ds = {key_of(json.loads(l)): json.loads(l) for l in open(DATASET)}
preds = [json.loads(l) for l in open(PREDS)]
tot = collections.Counter(); ok = collections.Counter(); ov_ok = ov_n = 0
for r in preds:
    d = ds.get(key_of(r))
    if d is None: continue
    c = isinstance(r.get("judge"), dict) and r["judge"].get("correct") is True
    ov_n += 1; ov_ok += int(c)
    s = d.get("vido_split")
    if s: tot[s] += 1; ok[s] += int(c)
print(f"matched = {ov_n}/{len(preds)}")
print(f"OVERALL = {ov_ok}/{ov_n} = {100*ov_ok/max(1,ov_n):.2f}%")
for s, n in [("Extraction", 645), ("Logic", 497)]:
    a = 100*ok[s]/tot[s] if tot[s] else 0.0
    print(f"  {s:11s}: {ok[s]:3d}/{tot[s]:3d} = {a:.2f}%   (VISOR n={n})")
# count-weighted overall for cross-check vs paper style
if tot['Extraction'] and tot['Logic']:
    w = (ok['Extraction']+ok['Logic'])/(tot['Extraction']+tot['Logic'])
    print(f"  weighted-overall = {100*w:.2f}%")
