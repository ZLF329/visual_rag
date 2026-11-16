"""Answer reward for RL = the eval judge (src/judge.py: DeepSeek v4-flash, the same prompt and model as every eval),
so a rollout is rewarded exactly when an eval would count it correct (DESIGN.md §11.31). Point the env's
reward_fn_path here; it defines no judge_answer_and_support, so rl/envs.py calls compute_score.
compute_score(data_source, solution_str, ground_truth, extra_info) -> 1.0 / 0.0; a failed judge call -> 0.0 and a log
line. A failed online judgement must not interrupt an RL worker."""
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.judge import judge_answer_equivalence  # noqa: E402


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    extra = extra_info or {}
    try:
        judged = judge_answer_equivalence(
            question=str(extra.get("question", "")), prediction=str(solution_str), gold_answer=str(ground_truth),
            sample_id=str(extra.get("sample_id", "")), timeout=float(os.environ.get("DEEPSEEK_REWARD_TIMEOUT", "60")),
            max_retries=int(os.environ.get("DEEPSEEK_REWARD_MAX_RETRIES", "2")))
        return 1.0 if judged["correct"] is True else 0.0
    except Exception as exc:  # The online service is optional for completing this trajectory.
        cause = exc.__cause__ or exc
        event = {"time": time.time(), "sample_id": str(extra.get("sample_id", "")),
                 "cause": type(cause).__name__, "http_status": getattr(cause, "code", None), "answer_reward": 0}
        try:
            if path := os.getenv("DEEPSEEK_FALLBACK_LOG"):
                with open(path, "a", encoding="utf-8") as log:
                    log.write(json.dumps(event) + "\n")
        except OSError:
            pass
        try:
            print(f"[reward_judge_error] fallback_answer_reward=0 cause={type(cause).__name__}", flush=True)
        except OSError:
            pass
        return 0.0
