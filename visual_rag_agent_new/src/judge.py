from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


class JudgeError(RuntimeError):
    pass


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> None:
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


def judge_prediction_row(
    row: dict[str, Any],
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    question = str(row.get("query") or row.get("question") or "").strip()
    prediction = str(row.get("answer") or "").strip()
    gold_answer = str(row.get("gold_answer") or "").strip()
    sample_id = str(row.get("sample_id", ""))
    return judge_answer_equivalence(
        question=question,
        prediction=prediction,
        gold_answer=gold_answer,
        sample_id=sample_id,
        model=model,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
    )


def judge_answer_equivalence(
    *,
    question: str,
    prediction: str,
    gold_answer: str,
    sample_id: str = "",
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise JudgeError("DEEPSEEK_API_KEY is missing. Put it in .env or export it before judging.")

    model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
    base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
    if max_tokens is None:
        max_tokens = int(os.environ.get("DEEPSEEK_JUDGE_MAX_TOKENS", "1024"))

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
            text = _chat_completion(
                base_url=base_url,
                api_key=api_key,
                payload=payload,
                timeout=timeout,
                max_retries=max_retries if attempt == 0 else 0,
            )
            return _extract_json_object(text)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise JudgeError(f"DeepSeek judge returned unparsable JSON after {attempt + 1} attempts: {last_error}") from exc



def _chat_completion(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    max_retries: int,
) -> str:
    endpoint = f"{base_url}/chat/completions"
    last_error: Exception | None = None
    current_payload = dict(payload)

    for attempt in range(max_retries + 1):
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
            last_error = JudgeError(f"DeepSeek HTTP {exc.code}: {body[:1000]}")
            if exc.code == 400 and "response_format" in current_payload:
                current_payload = dict(current_payload)
                current_payload.pop("response_format", None)
                continue
        except Exception as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))

    raise JudgeError(f"DeepSeek judge request failed: {last_error}")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise JudgeError(f"No JSON object found in judge output: {text[:300]}")
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeError(f"Invalid JSON from judge: {exc}: {text[:300]}") from exc


def add_judge_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    judged = [row.get("judge") for row in rows if isinstance(row.get("judge"), dict)]
    errors = [row.get("judge_error") for row in rows if row.get("judge_error")]
    if not judged and not errors:
        return
    correct = sum(1 for item in judged if item.get("correct") is True)
    summary["judge"] = {
        "provider": "deepseek",
        "model": judged[0].get("judge_model") if judged else os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        "judged": len(judged),
        "correct": correct,
        "errors": len(errors),
        "accuracy": correct / len(judged) if judged else 0,
    }
    summary["accuracy"] = summary["judge"]["accuracy"]
