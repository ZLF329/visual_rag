"""Run a genuine one-round smoke or the approved 75-round training."""
from pathlib import Path
import json
import os
import sys

from omegaconf import OmegaConf
from src.judge import judge_answer_equivalence

prep = Path(__file__).resolve().parents[1]
cfg = OmegaConf.load(prep / 'deployed_config.yaml')
assert Path(prep / 'deployment_verified.json').is_file()
assert cfg.env.slidevqa.forced_answer_prefill
assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == 'gspo_trajectory'
formal = Path(cfg.trainer.default_local_dir).parent
mode = sys.argv[1]
if mode == 'smoke':
    run = prep / 'gpu_smoke'
    cfg.data.train_files = str(Path(cfg.data.train_files).with_name('smoke.parquet'))
    cfg.env.max_steps = 2  # automatic search, one ordinary decision, then G
    cfg.trainer.total_training_steps = cfg.actor_rollout_ref.actor.optim.total_training_steps = 1
    cfg.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio = 0.0
    cfg.trainer.save_freq = cfg.trainer.test_freq = -1
    cfg.trainer.logger = ['console']
    cfg.trainer.experiment_name += '_gpu_smoke'
    cfg.trainer.default_local_dir = str(run / 'checkpoints')
    cfg.trainer.rollout_data_dir = str(run / 'rollouts')
    cfg.trainer.validation_data_dir = str(run / 'validation')
elif mode == 'formal':
    assert json.loads((prep / 'gpu_smoke/verified.json').read_text())['passed']
    assert cfg.trainer.total_training_steps == 75 and cfg.env.max_steps == 10
    run = formal
    assert not (run / 'effective_config.yaml').exists(), 'Formal run already started'
else:
    raise ValueError(mode)

run.mkdir(parents=True, exist_ok=True)
os.environ['TENSORBOARD_DIR'] = str(run / 'tb')
os.environ['ROLLOUT_DATA_DIR'] = str(run / 'rollouts')
os.environ['DEEPSEEK_FALLBACK_LOG'] = str(run / 'judge_fallbacks.jsonl')
OmegaConf.save(cfg, run / 'effective_config.yaml')

import ray
from verl.trainer.main_ppo import run_ppo
from transformers import AutoTokenizer
from agent_system.multi_turn_rollout.rollout_loop import _render_observation_prompt
from src.forced_answer import build_messages, PREFIX

tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
messages = build_messages(dict(question='What is 2 plus 2?', nodes=[], pages=[]))
assert _render_observation_prompt(tokenizer, messages, {}, PREFIX).endswith(PREFIX)
if mode == 'smoke':
    for prediction, expected in [('4', True), ('5', False)]:
        verdict = judge_answer_equivalence(question='What is 2 plus 2?', prediction=prediction, gold_answer='4',
                                           sample_id='prefill_gpu_smoke_judge_probe', timeout=30, max_retries=1)
        assert verdict['correct'] is expected
print('PREFILL_REAL_TOKENIZER_CHECK_PASSED', flush=True)
try:
    run_ppo(cfg)
finally:
    ray.shutdown()
