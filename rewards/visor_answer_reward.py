from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Small rule reward for the first plain GRPO pass.

    This is intentionally conservative and lexical. It is a runnable baseline,
    not a replacement for a stronger SlideVQA judge.
    """
    prediction = extract_answer(solution_str)
    gold = str(ground_truth or "")
    if not prediction.strip() or not gold.strip():
        return 0.0

    pred_norm = normalize(prediction)
    gold_norm = normalize(gold)
    if pred_norm == gold_norm:
        return 1.0
    if numeric_equal(prediction, gold):
        return 1.0
    if gold_norm and gold_norm in pred_norm:
        return 0.8

    f1 = token_f1(pred_norm, gold_norm)
    if f1 >= 0.9:
        return 0.7
    if f1 >= 0.6:
        return 0.4
    return 0.0


def extract_answer(text: str) -> str:
    text = str(text or "").strip()
    for marker in ["Final answer:", "Answer:", "答案：", "答案:"]:
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text.strip().strip("`")


def normalize(text: str) -> str:
    text = str(text).lower()
    text = text.replace("%", " percent ")
    text = re.sub(r"[$£€¥,]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def numeric_equal(prediction: str, gold: str, *, rel_tol: float = 1e-3) -> bool:
    pred_nums = extract_numbers(prediction)
    gold_nums = extract_numbers(gold)
    if not pred_nums or not gold_nums:
        return False
    for pred in pred_nums:
        for target in gold_nums:
            if math.isclose(pred, target, rel_tol=rel_tol, abs_tol=rel_tol):
                return True
    return False


def extract_numbers(text: str) -> list[float]:
    values = []
    for match in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(text)):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = prediction.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
