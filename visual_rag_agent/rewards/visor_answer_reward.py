from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"

_JUDGE_CACHE: dict[tuple[str, str, str], float] = {}
_JUDGE_ERROR_COUNT = 0


class JudgeError(RuntimeError):
    pass


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """DeepSeek LLM judge reward for SlideVQA final answers.

    Reward contract:
    - 1.0 if DeepSeek judges the prediction equivalent to the gold answer.
    - -1.0 if DeepSeek judges it incorrect, or if prediction/gold is empty.
    - 0.0 on judge/API/parser failures by default, because those are external
      failures rather than model trajectory mistakes. Set
      DEEPSEEK_REWARD_ON_ERROR=minus_one to return -1.0, or raise to fail fast.
    """
    _ = data_source
    extra_info = extra_info or {}
    prediction = extract_answer(solution_str)
    gold = str(ground_truth or "").strip()
    question = str(extra_info.get("question") or "").strip()
    sample_id = str(extra_info.get("sample_id") or extra_info.get("deck_name") or "")
    if not prediction or not gold:
        return -1.0

    cache_key = (question, gold, prediction)
    if cache_key in _JUDGE_CACHE:
        return _JUDGE_CACHE[cache_key]

    try:
        judgment = judge_answer_equivalence(
            question=question,
            prediction=prediction,
            gold_answer=gold,
            sample_id=sample_id,
        )
        reward = 1.0 if judgment.get("correct") is True and int(judgment.get("score", 0)) == 1 else -1.0
    except Exception as exc:
        global _JUDGE_ERROR_COUNT
        _JUDGE_ERROR_COUNT += 1
        on_error = os.environ.get("DEEPSEEK_REWARD_ON_ERROR", "zero").strip().lower()
        if on_error in {"minus_one", "-1"}:
            reward = -1.0
        elif on_error in {"raise", "error", "fail"}:
            raise
        else:
            reward = 0.0
        print(
            "[deepseek_judge_error] "
            f"fallback_reward={reward} count={_JUDGE_ERROR_COUNT} sample_id={sample_id} error={type(exc).__name__}: {exc}",
            flush=True,
        )
    _JUDGE_CACHE[cache_key] = reward
    return reward


def extract_answer(text: str) -> str:
    text = str(text or "").strip()
    for marker in ("Final answer:", "Answer:", "答案：", "答案:"):
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text.strip().strip("`")


def judge_answer_equivalence(
    *,
    question: str,
    prediction: str,
    gold_answer: str,
    sample_id: str = "",
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    env_file = os.environ.get("ACTIVE_GRAPH_RL_ENV_FILE")
    if env_file:
        load_dotenv(env_file)
    load_dotenv(Path("/scratch/punim0614/lifuzhang/active_graph_rl_workspace/.env"))
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise JudgeError("DEEPSEEK_API_KEY is missing for LLM answer reward.")

    model = model or os.environ.get("DEEPSEEK_REWARD_MODEL") or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
    base_url = (base_url or os.environ.get("DEEPSEEK_REWARD_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
    if timeout is None:
        timeout = float(os.environ.get("DEEPSEEK_REWARD_TIMEOUT", "60"))
    if max_retries is None:
        max_retries = int(os.environ.get("DEEPSEEK_REWARD_MAX_RETRIES", "2"))
    if max_tokens is None:
        max_tokens = int(os.environ.get("DEEPSEEK_REWARD_MAX_TOKENS", os.environ.get("DEEPSEEK_JUDGE_MAX_TOKENS", "512")))

    system = (
        "You are a strict but reasonable evaluator for SlideVQA answer equivalence. "
        "Judge whether the model prediction correctly answers the question according to the gold answer. "
        "Accept harmless formatting differences, extra explanatory text, currency/unit spelling differences, "
        "and equivalent numeric forms. Reject wrong numbers, wrong units that change meaning, missing answers, "
        "or answers that only discuss related context without giving the requested value. "
        "Return JSON only with exactly these keys: correct, score, rationale, normalized_prediction, normalized_gold. "
        "Use correct=true and score=1 for equivalent answers; otherwise correct=false and score=0."
    )
    user = json.dumps(
        {
            "sample_id": sample_id,
            "question": question,
            "gold_answer": gold_answer,
            "model_prediction": prediction,
        },
        ensure_ascii=False,
        indent=2,
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    parsed = _chat_completion_json(
        base_url=base_url,
        api_key=api_key,
        payload=payload,
        timeout=timeout,
        max_retries=max_retries,
    )
    correct = bool(parsed.get("correct"))
    score = int(parsed.get("score", 1 if correct else 0))
    return {
        "judge_provider": "deepseek",
        "judge_model": model,
        "correct": correct,
        "score": 1 if score else 0,
        "rationale": str(parsed.get("rationale", "")).strip(),
        "normalized_prediction": str(parsed.get("normalized_prediction", "")).strip(),
        "normalized_gold": str(parsed.get("normalized_gold", "")).strip(),
    }


_SUPPORT_CACHE: dict[tuple, dict[str, Any]] = {}


def judge_answer_and_support(
    *,
    question: str,
    prediction: str,
    gold_answer: str,
    facts=(),
    sample_id: str = "",
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """ONE DeepSeek call judging BOTH (a) answer correctness vs gold and
    (b) whether the model's COLLECTED FACTS entail/support the answer.
    Returns {"correct": bool, "score": 0/1, "support": bool, "rationale": str}.
    `support` is judged independently here; the caller gates it on correctness.
    """
    facts_list = [str(f).strip() for f in (facts or []) if str(f).strip()]
    cache_key = (question, gold_answer, prediction, tuple(facts_list))
    cached = _SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    env_file = os.environ.get("ACTIVE_GRAPH_RL_ENV_FILE")
    if env_file:
        load_dotenv(env_file)
    load_dotenv(Path("/scratch/punim0614/lifuzhang/active_graph_rl_workspace/.env"))
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise JudgeError("DEEPSEEK_API_KEY is missing for LLM answer reward.")

    model = model or os.environ.get("DEEPSEEK_REWARD_MODEL") or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
    base_url = (base_url or os.environ.get("DEEPSEEK_REWARD_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
    if timeout is None:
        timeout = float(os.environ.get("DEEPSEEK_REWARD_TIMEOUT", "60"))
    if max_retries is None:
        max_retries = int(os.environ.get("DEEPSEEK_REWARD_MAX_RETRIES", "2"))
    if max_tokens is None:
        max_tokens = int(os.environ.get("DEEPSEEK_REWARD_MAX_TOKENS", os.environ.get("DEEPSEEK_JUDGE_MAX_TOKENS", "512")))

    system = (
        "You are a strict but reasonable evaluator for a visual-RAG agent on SlideVQA. "
        "You must judge THREE independent things and return JSON only.\n"
        "(1) correct: does the model prediction correctly answer the question according to the gold answer? "
        "Accept harmless formatting differences, extra explanatory text, currency/unit spelling differences, "
        "and equivalent numeric forms. Reject wrong numbers, wrong units that change meaning, missing answers, "
        "or answers that only discuss related context without giving the requested value.\n"
        "(2) support: do the model's COLLECTED FACTS, by themselves, contain the specific evidence needed to "
        "derive the model prediction? support=true ONLY if the facts entail/justify the prediction. "
        "support=false if the facts are empty, irrelevant, merely topically related, or insufficient to derive "
        "the answer -- even if the prediction happens to be correct from outside knowledge. "
        "Judge support on the facts as given, NOT on the gold answer.\n"
        "(3) support_gold: do the model's COLLECTED FACTS, by themselves, contain the specific evidence "
        "needed to derive the GOLD answer? Same rule as (2) but judged against the gold answer: "
        "support_gold=true ONLY if the facts entail/justify the gold answer; support_gold=false if the facts "
        "are empty, irrelevant, merely topically related, or insufficient to derive the gold answer.\n"
        "Return JSON only with exactly these keys: correct, score, support, support_gold, rationale. "
        "Keep the rationale to ONE short sentence (only the boolean fields are used downstream; do not run long). "
        "Use correct=true and score=1 for equivalent answers, otherwise correct=false and score=0. "
        "Use support / support_gold = true or false per the rules above."
    )
    user = json.dumps(
        {
            "sample_id": sample_id,
            "question": question,
            "gold_answer": gold_answer,
            "model_prediction": prediction,
            "model_collected_facts": facts_list,
        },
        ensure_ascii=False,
        indent=2,
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    parsed = _chat_completion_json(
        base_url=base_url,
        api_key=api_key,
        payload=payload,
        timeout=timeout,
        max_retries=max_retries,
    )
    correct = bool(parsed.get("correct"))
    score = int(parsed.get("score", 1 if correct else 0))
    result = {
        "judge_provider": "deepseek",
        "judge_model": model,
        "correct": correct,
        "score": 1 if score else 0,
        "support": bool(parsed.get("support")),
        "support_gold": bool(parsed.get("support_gold")),
        "rationale": str(parsed.get("rationale", "")).strip(),
    }
    _SUPPORT_CACHE[cache_key] = result
    return result


def load_dotenv(path: str | Path, *, override: bool = False) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def _chat_completion_json(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            text = _chat_completion(base_url=base_url, api_key=api_key, payload=payload, timeout=timeout)
            return _extract_json_object(text)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise JudgeError(f"DeepSeek judge returned unparsable JSON after {attempt + 1} attempts: {last_error}") from exc


def _chat_completion(*, base_url: str, api_key: str, payload: dict[str, Any], timeout: float) -> str:
    endpoint = f"{base_url}/chat/completions"
    current_payload = dict(payload)
    for allow_response_format in (True, False):
        if not allow_response_format:
            current_payload = dict(current_payload)
            current_payload.pop("response_format", None)
        try:
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(current_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            return str(response_payload["choices"][0]["message"].get("content", ""))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 400 and "response_format" in current_payload and allow_response_format:
                continue
            raise JudgeError(f"DeepSeek HTTP {exc.code}: {body[:1000]}") from exc
    raise JudgeError("DeepSeek judge request failed")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise JudgeError(f"No JSON object found in judge output: {stripped[:300]}")
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeError(f"Invalid JSON from judge: {exc}: {stripped[:300]}") from exc
