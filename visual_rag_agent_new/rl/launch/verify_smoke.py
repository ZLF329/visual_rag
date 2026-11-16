"""Require real sampled G continuations, reward agreement and five finite GSPO updates."""
import json
import math
from pathlib import Path
import re

run = Path(__file__).resolve().parents[1] / 'gpu_smoke'
assert (run / 'exit').read_text().strip() == '0'
log = (run / 'train.log').read_text()
assert 'PREFILL_REAL_TOKENIZER_CHECK_PASSED' in log
audit = re.findall(r'PREFILL_GPU_AUDIT (\{[^\n]+\})', log)
assert audit, 'No GPU rollout audit'
evidence = json.loads(audit[-1])
assert evidence['trajectories'] == 80
assert evidence['forced_turns'] > 0 and evidence['forced_answers'] > 0
assert evidence['forced_correct'] > 0
assert evidence['forced_gpu_rows_checked'] == evidence['forced_turns']
fallback = run / 'judge_fallbacks.jsonl'
assert not fallback.exists() or not fallback.stat().st_size, 'Online judge fell back on at least one rollout'

step = next(line for line in reversed(log.splitlines()) if 'step:1 ' in line and 'actor/grad_norm' in line)
def metric(name):
    match = re.search(re.escape(name) + r':([-+0-9.eE]+)(?:\s|$)', step)
    assert match, name
    value = float(match.group(1))
    assert math.isfinite(value), name
    return value
norms = [metric(f'trajectory/mini_{i:02d}/grad_norm') for i in range(1, 6)]
assert all(value > 0 for value in norms), norms
# Console logs round small LRs to zero, so verify the effective scheduler config.
import yaml
cfg = yaml.safe_load((run / 'effective_config.yaml').read_text())
optimizer = cfg['actor_rollout_ref']['actor']['optim']
assert optimizer['lr'] > 0 and optimizer['lr_warmup_steps_ratio'] == 0
report = dict(passed=True, **evidence, gsppo_optimizer_minibatches=5,
              grad_norms=norms, learning_rate=optimizer['lr'])
(run / 'verified.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
