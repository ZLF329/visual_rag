#!/usr/bin/env python
"""Concurrent SFT trajectory generator.

Mirrors the per-row pipeline in
notebooks/generate_gpt54mini_sft_trajectories.ipynb cell-by-cell, but replaces
the serial outer loop with a ThreadPoolExecutor sliding-window scheduler
modelled after multimodalrag's run_manai_pipeline.

Prompt steps are preserved exactly: each row runs the main agent's
decide -> (search) -> analyse-each-page -> memory_update -> answer loop using
src.prompts and src.schemas. Workers only generate; the main thread writes the
JSONL, updates stats, and checks the kept-so-far early-stop.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

PROJECT = Path(os.environ.get('SFT_PROJECT_DIR', '/users/is/lzhang/Desktop/visual_rag'))
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from man.ai.mangpt.types import (  # noqa: E402
    Chat,
    Content,
    ContentImageURL,
    Message,
    REASONING_MODELS,
)
from src.memory import Memory  # noqa: E402
from src.prompts import (  # noqa: E402
    build_analyse_prompt,
    build_decide_prompt,
    build_memory_update_prompt,
)
from src.retriever import Retriever, normalize_rows  # noqa: E402
from src.schemas import AnalyseResult, DecideResult, MemoryUpdateResult  # noqa: E402


def load_dotenv(path: Path = PROJECT / '.env', *, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


load_dotenv()

DATASET_FILE = Path(os.environ.get('SFT_DATASET_FILE', str(PROJECT / 'data/corpora/slidevqa/train.jsonl')))
INDEX_DIR = Path(os.environ.get('SFT_INDEX_DIR', str(PROJECT / 'data/indexes/slidevqa')))
GENERATOR_MODEL = os.environ.get('MAN_SFT_GENERATOR_MODEL', 'gpt-5-mini')
VALIDATOR_MODEL = os.environ.get('MAN_SFT_VALIDATOR_MODEL', 'gpt-5-mini')
RETRIEVER_MODEL = os.environ.get(
    'SFT_RETRIEVER_MODEL',
    '/data/team/ml/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-8B/snapshots/2c4565515e0f265c6511776e7193b22c0968ddc7',
)
RETRIEVER_DEVICE = os.environ.get('SFT_RETRIEVER_DEVICE', 'cuda')
RETRIEVER_DTYPE = os.environ.get('SFT_RETRIEVER_DTYPE', 'bfloat16')
RETRIEVER_ATTN = os.environ.get('SFT_RETRIEVER_ATTN', 'sdpa')
START_INDEX = int(os.environ.get('SFT_START_INDEX', '0'))
MAX_SAMPLES = int(os.environ.get('SFT_MAX_SAMPLES', '5000'))
TARGET_KEPT = int(os.environ.get('SFT_TARGET_KEPT', '1200'))
MAX_RETRIEVAL_STEPS = int(os.environ.get('SFT_MAX_RETRIEVAL_STEPS', '5'))
RETRIEVAL_TOP_K = int(os.environ.get('SFT_RETRIEVAL_TOP_K', '1'))
MAX_CONTEXT_IMAGES = int(os.environ.get('SFT_MAX_CONTEXT_IMAGES', '15'))
MAX_IMAGE_PIXELS = int(os.environ.get('SFT_IMAGE_MAX_PIXELS', '1250000'))
IMAGE_JPEG_QUALITY = int(os.environ.get('SFT_IMAGE_JPEG_QUALITY', '85'))
IMAGE_DETAIL = os.environ.get('SFT_IMAGE_DETAIL', 'high')
GENERATOR_TEMPERATURE = float(os.environ.get('SFT_GENERATOR_TEMPERATURE', '0.2'))
REASONING_EFFORT = os.environ.get('MAN_REASONING_EFFORT', 'low')
REQUEST_TIMEOUT = float(os.environ.get('MAN_REQUEST_TIMEOUT', '180'))
MAX_RETRIES = int(os.environ.get('MAN_MAX_RETRIES', '3'))
CONCURRENCY = int(os.environ.get('SFT_CONCURRENCY', '8'))
CHECKPOINT_EVERY = int(os.environ.get('SFT_CHECKPOINT_EVERY', '50'))
RUN_ID = os.environ.get('SFT_RUN_ID', time.strftime('%Y%m%d_%H%M%S'))
OUTPUT_DIR = Path(os.environ.get(
    'SFT_TRAJECTORY_OUTPUT_DIR',
    str(PROJECT / 'outputs/sft_trajectories' / f'gpt5mini_{RUN_ID}'),
))
RAW_PATH = OUTPUT_DIR / 'raw_trajectories.jsonl'
KEPT_PATH = OUTPUT_DIR / 'kept_sft_trajectories.jsonl'
REJECTED_PATH = OUTPUT_DIR / 'rejected_trajectories.jsonl'
SUMMARY_PATH = OUTPUT_DIR / 'summary.json'


def read_jsonl(path: Path, *, limit: int | None = None, start: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if seen < start:
                seen += 1
                continue
            rows.append(json.loads(line))
            seen += 1
            if limit is not None and len(rows) >= limit:
                break
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def first_present(row: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ''):
            return value
    return default


def extract_question(row: dict[str, Any]) -> str:
    return str(first_present(row, ('query', 'question', 'prompt', 'input', 'problem'), '')).strip()


def extract_answer(row: dict[str, Any]) -> str:
    value = first_present(row, ('answer', 'answers', 'target', 'label', 'response'), '')
    return ' | '.join(str(item) for item in value) if isinstance(value, list) else str(value).strip()


def extract_sample_id(row: dict[str, Any], row_index: int) -> str:
    return str(first_present(row, ('id', 'qid', 'question_id', 'uid', 'eval_id', 'qa_id'), row_index))


def extract_deck_name(row: dict[str, Any]) -> str:
    return str(first_present(row, ('deck_name', 'deck_id', 'doc_id', 'document_id', 'pdf_id'), '')).strip()


def page_num_to_label(deck_name: str, page_num: int) -> str:
    return f'{deck_name}/page_{page_num:02d}' if deck_name else f'page_{page_num:02d}'


def label_to_page_num(label: str) -> int | None:
    match = re.search(r'page[_-]?(\d+)', str(label), flags=re.I)
    return int(match.group(1)) if match else None


def extract_reference_pages(row: dict[str, Any]) -> set[int]:
    value = first_present(row, ('reference_pages', 'gold_pages', 'evidence_pages', 'page_ids', 'pages', 'answer_pages'), [])
    if isinstance(value, str):
        return {int(num) for num in re.findall(r'\d+', value)}
    if isinstance(value, (int, float)):
        return {int(value)}
    out: set[int] = set()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                item = first_present(item, ('page', 'page_num', 'page_id', 'index'), None)
            try:
                out.add(int(item))
            except (TypeError, ValueError):
                pass
    return out


def extract_reference_labels(row: dict[str, Any]) -> set[str]:
    deck_name = extract_deck_name(row)
    return {page_num_to_label(deck_name, page_num) for page_num in extract_reference_pages(row)}


class ManAIError(RuntimeError):
    pass


def _resize_jpeg_bytes(path: str | Path) -> bytes:
    with Image.open(path) as image:
        image = image.convert('RGB')
        w, h = image.size
        if MAX_IMAGE_PIXELS > 0 and w * h > MAX_IMAGE_PIXELS:
            scale = math.sqrt(MAX_IMAGE_PIXELS / float(w * h))
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = BytesIO()
        image.save(buf, format='JPEG', quality=IMAGE_JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def _user_content(text: str, image_paths: list[str] | None) -> list[Content]:
    blocks: list[Content] = [Content(type='text', text=text)]
    for image_path in image_paths or []:
        image_bytes = _resize_jpeg_bytes(image_path)
        blocks.append(Content(
            type='image_url',
            image_url=ContentImageURL.from_bytes(image_bytes, mimetype='image/jpeg', detail=IMAGE_DETAIL),
        ))
    return blocks


def _chat_kwargs(model: str, max_output_tokens: int, temperature: float | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {'model': model, 'timeout': REQUEST_TIMEOUT, 'max_tokens': max_output_tokens}
    if model in REASONING_MODELS:
        kwargs['reasoning_effort'] = REASONING_EFFORT
    elif temperature is not None:
        kwargs['temperature'] = temperature
    return kwargs


def call_json(*, model: str, system: str, user: str, response_model, image_paths: list[str] | None = None, max_output_tokens: int = 700, temperature: float | None = 0.0) -> tuple[dict[str, Any], str, Any]:
    messages = [
        Message(role='system', content=system),
        Message(role='user', content=_user_content(user, image_paths)),
    ]
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            chat = Chat(
                messages=messages,
                response_format=response_model,
                **_chat_kwargs(model, max_output_tokens, temperature),
            )
            response = chat.create()
            raw = (Message.from_chat_response(response).content or '').strip()
            parsed = response_model.model_validate_json(raw)
            return parsed.model_dump(), raw, response
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(min(20, 1.5 * (attempt + 1)))
    raise ManAIError(f'man.ai call failed after {MAX_RETRIES + 1} attempts: {type(last_error).__name__}: {last_error}')


def compact_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, 'usage', None)
    if usage is None:
        return {}
    if hasattr(usage, 'model_dump'):
        try:
            return usage.model_dump()
        except Exception:
            pass
    try:
        return dict(usage)
    except Exception:
        return {}


class DeckRestrictedSearcher:
    """Deck-restricted retriever wrapper. Thread-safe by guarding the embedder
    forward pass: shared GPU model gets serialised access, which matches what
    multimodalrag's pipeline does in practice."""

    def __init__(self) -> None:
        self.retriever = Retriever(
            model_path=RETRIEVER_MODEL,
            index_path=INDEX_DIR,
            device=RETRIEVER_DEVICE,
            dtype=RETRIEVER_DTYPE,
            attn_implementation=RETRIEVER_ATTN,
            load_model=True,
        )
        if self.retriever.embeddings is None or not self.retriever.entries:
            raise RuntimeError(f'Missing retriever index under {INDEX_DIR}')
        self._embedder_lock = Lock()

    def search(self, *, query: str, deck_name: str, top_k: int) -> list[dict[str, Any]]:
        if self.retriever.embedder is None:
            raise RuntimeError('retriever embedder not loaded')
        with self._embedder_lock:
            q = self.retriever.embedder.embed_texts([query])[0]
        q = normalize_rows(q[None, :])[0]
        scores = self.retriever.embeddings @ q
        ranked = np.argsort(-scores)
        out: list[dict[str, Any]] = []
        prefix = f'{deck_name}/' if deck_name else ''
        for raw_idx in ranked:
            entry = self.retriever.entries[int(raw_idx)]
            label = entry.page_label
            if prefix and not label.startswith(prefix):
                continue
            image_path = Path(entry.image_path)
            if not image_path.is_absolute():
                image_path = PROJECT / image_path
            if not image_path.exists():
                continue
            out.append({
                'page_label': label,
                'page_num': label_to_page_num(label),
                'image_path': str(image_path),
                'score': float(scores[int(raw_idx)]),
            })
            if len(out) >= top_k:
                break
        return out


JUDGE_SYSTEM = (
    'You are a strict but reasonable evaluator for SlideVQA answer equivalence. '
    'Judge whether the model prediction correctly answers the question according to the gold answer. '
    'Accept harmless formatting differences, extra explanatory text, currency/unit spelling differences, '
    'and equivalent numeric forms. Reject wrong numbers, wrong units that change meaning, missing answers, '
    'or answers that only discuss related context without giving the requested value. '
    'Return JSON only with exactly these keys: correct, score, rationale, normalized_prediction, normalized_gold. '
    'Use correct=true and score=1 for equivalent answers; otherwise correct=false and score=0.'
)


class JudgeOutput(BaseModel):
    correct: bool
    score: Literal[0, 1]
    rationale: str
    normalized_prediction: str
    normalized_gold: str


def decide_next(*, question: str, memory_context: str, retained_image_paths: list[str] | None):
    system, user = build_decide_prompt(original_query=question, memory_context=memory_context)
    return call_json(
        model=GENERATOR_MODEL,
        system=system,
        user=user,
        response_model=DecideResult,
        image_paths=retained_image_paths or None,
        max_output_tokens=500,
        temperature=GENERATOR_TEMPERATURE,
    )


def analyse_page(*, question: str, memory_context: str, image_path: str):
    system, user = build_analyse_prompt(original_query=question, memory_context=memory_context)
    return call_json(
        model=GENERATOR_MODEL,
        system=system,
        user=user,
        response_model=AnalyseResult,
        image_paths=[image_path],
        max_output_tokens=900,
        temperature=GENERATOR_TEMPERATURE,
    )


def update_memory_step(*, question: str, previous_key_facts: list[str], latest_judge: str, latest_key_facts: list[str]):
    system, user = build_memory_update_prompt(
        original_query=question,
        previous_key_facts=previous_key_facts,
        latest_judge=latest_judge,
        latest_key_facts=latest_key_facts,
    )
    return call_json(
        model=GENERATOR_MODEL,
        system=system,
        user=user,
        response_model=MemoryUpdateResult,
        max_output_tokens=600,
        temperature=GENERATOR_TEMPERATURE,
    )


def judge_answer(*, question: str, gold_answer: str, prediction: str, sample_id: str):
    user = json.dumps(
        {
            'sample_id': sample_id,
            'question': question,
            'gold_answer': gold_answer,
            'model_prediction': prediction,
        },
        ensure_ascii=False,
        indent=2,
    )
    return call_json(
        model=VALIDATOR_MODEL,
        system=JUDGE_SYSTEM,
        user=user,
        response_model=JudgeOutput,
        max_output_tokens=500,
        temperature=0.0,
    )


def build_sft_messages(record):
    messages = [
        {
            'role': 'system',
            'content': (
                'You are a visual-RAG agent. Each iteration: decide whether to search for one more page '
                'or to answer; if searching, output a short query; for every retrieved page emit an '
                'analyse JSON (think, summary, key_facts, judge); after each non-"no" analyse, emit a '
                'memory_update JSON (key_facts). Answer only when compact memory is sufficient.'
            ),
        },
        {'role': 'user', 'content': record['question']},
    ]
    for step in record.get('trace', []):
        kind = step.get('step')
        if kind == 'decide':
            messages.append({'role': 'assistant', 'content': json.dumps(step.get('result', {}), ensure_ascii=False)})
        elif kind == 'search':
            messages.append({
                'role': 'tool',
                'name': 'retrieve_pages',
                'content': json.dumps({'query': step.get('query'), 'pages': step.get('pages', [])}, ensure_ascii=False),
            })
        elif kind == 'analyse':
            payload = {'page': step.get('page'), **(step.get('result') or {})}
            messages.append({'role': 'assistant', 'content': json.dumps(payload, ensure_ascii=False)})
        elif kind == 'memory_update':
            messages.append({'role': 'assistant', 'content': json.dumps(step.get('result', {}), ensure_ascii=False)})
    if record.get('final_answer'):
        messages.append({'role': 'assistant', 'content': record['final_answer']})
    return messages


def run_one_example(searcher: DeckRestrictedSearcher, row: dict[str, Any], row_index: int) -> dict[str, Any]:
    """Run the main agent loop for a single row (decide -> analyse -> memory_update -> answer).

    Identical step-by-step to notebook cell 5 / src/agent.py.
    """
    sample_id = extract_sample_id(row, row_index)
    question = extract_question(row)
    reference_answer = extract_answer(row)
    deck = extract_deck_name(row)
    reference_labels = extract_reference_labels(row)

    sample_image_dir = OUTPUT_DIR / 'memory_images' / re.sub(r'[^A-Za-z0-9_.-]', '_', str(sample_id))
    memory = Memory(original_query=question, image_dir=sample_image_dir)

    trace: list[dict[str, Any]] = []
    usage: list[dict[str, Any]] = []
    retrieved_labels: list[str] = []
    pages_by_label: dict[str, dict[str, Any]] = {}
    decision_obj: DecideResult | None = None
    decide_payload: dict[str, Any] | None = None

    while memory.iter < MAX_RETRIEVAL_STEPS:
        retained_paths = [str(p) for p in memory.retained_image_paths()]
        decide_payload, decide_raw, decide_resp = decide_next(
            question=question,
            memory_context=memory.context_for_decide(),
            retained_image_paths=retained_paths,
        )
        decision_obj = DecideResult.model_validate(decide_payload)
        decision_obj.validate_branch()
        usage.append({'iter': memory.iter, 'call': 'decide', 'usage': compact_usage(decide_resp)})
        trace.append({
            'iter': memory.iter,
            'step': 'decide',
            'result': decide_payload,
            'retained_images': retained_paths,
            'raw_text': decide_raw,
            'response_id': getattr(decide_resp, 'id', None),
        })

        if decision_obj.action == 'answer':
            break

        search_query = (decision_obj.content or '').strip() or question
        pages = searcher.search(query=search_query, deck_name=deck, top_k=RETRIEVAL_TOP_K)
        retrieved_labels.extend(p['page_label'] for p in pages)
        for page in pages:
            pages_by_label[page['page_label']] = page
        trace.append({'iter': memory.iter, 'step': 'search', 'query': search_query, 'pages': pages})

        if not pages:
            memory.add_empty_search_warning(search_query)
            memory.iter += 1
            continue

        for page in pages:
            pil_image = Image.open(page['image_path']).convert('RGB')
            analyse_payload, analyse_raw, analyse_resp = analyse_page(
                question=question,
                memory_context=memory.context_for_analyse(),
                image_path=page['image_path'],
            )
            analyse_obj = AnalyseResult.model_validate(analyse_payload)
            analyse_obj.validate_branch()
            usage.append({'iter': memory.iter, 'call': 'analyse', 'usage': compact_usage(analyse_resp)})
            memory.write(analyse_obj, pil_image, search_query, page_label=page['page_label'])
            memory.append_consolidated_summary(analyse_obj.summary, search_query=search_query)
            trace.append({
                'iter': memory.iter,
                'step': 'analyse',
                'page': page['page_label'],
                'query': search_query,
                'result': analyse_payload,
                'raw_text': analyse_raw,
                'response_id': getattr(analyse_resp, 'id', None),
            })

            if analyse_obj.judge == 'no':
                trace.append({'iter': memory.iter, 'step': 'memory_update_skipped', 'page': page['page_label'], 'reason': 'judge_no'})
                continue

            mu_payload, mu_raw, mu_resp = update_memory_step(
                question=question,
                previous_key_facts=list(memory.consolidated_key_facts),
                latest_judge=analyse_obj.decision,
                latest_key_facts=list(analyse_obj.key_facts),
            )
            mu_obj = MemoryUpdateResult.model_validate(mu_payload)
            memory.update_consolidated(mu_obj)
            usage.append({'iter': memory.iter, 'call': 'memory_update', 'usage': compact_usage(mu_resp)})
            trace.append({
                'iter': memory.iter,
                'step': 'memory_update',
                'page': page['page_label'],
                'result': mu_payload,
                'raw_text': mu_raw,
                'response_id': getattr(mu_resp, 'id', None),
            })

        memory.iter += 1

    answered = decision_obj is not None and decision_obj.action == 'answer'
    final = (decision_obj.content or '').strip() if answered and decision_obj.content else ''
    stop_reason = 'decide_answer' if answered else 'max_iters'

    judge_payload, judge_raw, judge_resp = judge_answer(
        question=question,
        gold_answer=reference_answer,
        prediction=final,
        sample_id=sample_id,
    )
    usage.append({'iter': None, 'call': 'validator', 'usage': compact_usage(judge_resp)})

    retrieved_set = set(retrieved_labels)
    missing = sorted(reference_labels - retrieved_set)
    retrieval_steps = sum(1 for x in trace if x.get('step') == 'search')
    checks = {
        'all_reference_pages_retrieved': bool(reference_labels) and not missing,
        'retrieval_steps_ok': retrieval_steps <= MAX_RETRIEVAL_STEPS,
        'answer_matches_reference': bool(judge_payload.get('correct')) and int(judge_payload.get('score', 0)) == 1,
    }
    reasons: list[str] = []
    if not reference_labels:
        reasons.append('no_reference_pages')
    if not checks['all_reference_pages_retrieved']:
        reasons.append('missing_reference_pages')
    if not checks['retrieval_steps_ok']:
        reasons.append('too_many_retrieval_steps')
    if not checks['answer_matches_reference']:
        reasons.append('answer_mismatch')

    record: dict[str, Any] = {
        'sample_id': sample_id,
        'row_index': row_index,
        'deck_name': deck,
        'question': question,
        'reference_answer': reference_answer,
        'reference_page_labels': sorted(reference_labels),
        'retrieved_page_labels': retrieved_labels,
        'found_reference_page_labels': sorted(reference_labels & retrieved_set),
        'missing_reference_page_labels': missing,
        'retrieval_steps': retrieval_steps,
        'stop_reason': stop_reason,
        'final_answer': final,
        'final_decision': decide_payload if answered else None,
        'validator': {'provider': 'man.ai', 'model': VALIDATOR_MODEL, **judge_payload},
        'validator_raw_text': judge_raw,
        'validator_response_id': getattr(judge_resp, 'id', None),
        'keep_checks': checks,
        'keep': all(checks.values()),
        'reject_reasons': reasons,
        'memory': memory.as_serializable(),
        'trace': trace,
        'api': {
            'provider': 'man.ai',
            'generator_model': GENERATOR_MODEL,
            'validator_model': VALIDATOR_MODEL,
            'retriever_model': RETRIEVER_MODEL,
            'reasoning_effort': REASONING_EFFORT,
            'usage': usage,
        },
    }
    record['sft_messages'] = build_sft_messages(record)
    return record


def process_one_safe(searcher: DeckRestrictedSearcher, row: dict[str, Any], row_index: int) -> dict[str, Any]:
    """Worker entry-point: never raises. Returns a record (possibly an error stub)."""
    started = time.time()
    sample_id = extract_sample_id(row, row_index)
    try:
        record = run_one_example(searcher, row, row_index)
    except Exception as exc:  # noqa: BLE001
        record = {
            'sample_id': sample_id,
            'row_index': row_index,
            'question': extract_question(row),
            'reference_answer': extract_answer(row),
            'reference_page_labels': sorted(extract_reference_labels(row)),
            'keep': False,
            'reject_reasons': ['error'],
            'error': f'{type(exc).__name__}: {exc}',
        }
    record['elapsed_sec'] = round(time.time() - started, 2)
    return record


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [x for x in records if not x.get('error')]
    kept = [x for x in generated if x.get('keep')]
    rejected = [x for x in generated if not x.get('keep')]
    reasons = Counter(r for x in rejected for r in x.get('reject_reasons', []))
    return {
        'num_records': len(records),
        'num_generated': len(generated),
        'num_errors': sum(1 for x in records if x.get('error')),
        'num_kept': len(kept),
        'target_kept': TARGET_KEPT,
        'keep_rate': len(kept) / len(generated) if generated else 0,
        'reject_reason_counts': dict(reasons),
        'mean_retrieval_steps_generated': statistics.mean([x.get('retrieval_steps', 0) for x in generated]) if generated else 0,
        'provider': 'man.ai',
        'generator_model': GENERATOR_MODEL,
        'validator_model': VALIDATOR_MODEL,
        'retriever_model': RETRIEVER_MODEL,
        'max_retrieval_steps': MAX_RETRIEVAL_STEPS,
        'retrieval_top_k': RETRIEVAL_TOP_K,
        'concurrency': CONCURRENCY,
        'validation_mechanism': f'man.ai {VALIDATOR_MODEL} judge using src/judge.py SlideVQA prompt (gold_answer/normalized_gold schema)',
    }


def main() -> None:
    print(json.dumps({
        'generator': GENERATOR_MODEL,
        'validator': VALIDATOR_MODEL,
        'retriever': RETRIEVER_MODEL,
        'concurrency': CONCURRENCY,
        'target_kept': TARGET_KEPT,
        'max_samples_window': MAX_SAMPLES,
        'max_retrieval_steps': MAX_RETRIEVAL_STEPS,
        'output_dir': str(OUTPUT_DIR),
    }, ensure_ascii=False), flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(DATASET_FILE, limit=MAX_SAMPLES, start=START_INDEX)

    existing = read_jsonl(RAW_PATH) if RAW_PATH.exists() else []
    done_ids = {str(x.get('sample_id')) for x in existing}
    kept_so_far = sum(1 for x in existing if x.get('keep'))
    print(json.dumps({
        'resume': True,
        'existing': len(existing),
        'kept_so_far': kept_so_far,
        'target_kept': TARGET_KEPT,
    }, ensure_ascii=False), flush=True)

    if kept_so_far >= TARGET_KEPT:
        print(f'Already have {kept_so_far} kept records (>= TARGET_KEPT={TARGET_KEPT}); skipping generation.', flush=True)
    else:
        searcher = DeckRestrictedSearcher()
        pending: list[tuple[int, dict[str, Any]]] = []
        for local_idx, row in enumerate(rows):
            row_index = START_INDEX + local_idx
            sample_id = extract_sample_id(row, row_index)
            if sample_id in done_ids:
                continue
            pending.append((row_index, row))
        print(json.dumps({'plan': True, 'pending': len(pending), 'concurrency': CONCURRENCY}, ensure_ascii=False), flush=True)

        attempts_made = 0
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures: dict = {}
            pending_iter = iter(pending)

            def submit_next() -> bool:
                try:
                    row_index, row = next(pending_iter)
                except StopIteration:
                    return False
                fut = pool.submit(process_one_safe, searcher, row, row_index)
                futures[fut] = row_index
                return True

            for _ in range(CONCURRENCY):
                if not submit_next():
                    break

            target_hit = False
            while futures:
                done_futs, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    row_index = futures.pop(fut)
                    record = fut.result()
                    append_jsonl(RAW_PATH, record)
                    done_ids.add(record.get('sample_id'))
                    attempts_made += 1
                    if record.get('keep'):
                        kept_so_far += 1
                    print(json.dumps({
                        'done': len(done_ids),
                        'attempts': attempts_made,
                        'kept_so_far': kept_so_far,
                        'target_kept': TARGET_KEPT,
                        'sample_id': record.get('sample_id'),
                        'row_index': row_index,
                        'keep': record.get('keep'),
                        'reject_reasons': record.get('reject_reasons'),
                        'retrieval_steps': record.get('retrieval_steps'),
                        'elapsed_sec': record.get('elapsed_sec'),
                        'error': record.get('error'),
                    }, ensure_ascii=False), flush=True)

                    if kept_so_far >= TARGET_KEPT and not target_hit:
                        target_hit = True
                        print(f'[stop] reached TARGET_KEPT={TARGET_KEPT} after {attempts_made} attempts; '
                              f'draining {len(futures)} in-flight workers, no new submissions.', flush=True)

                    if not target_hit:
                        submit_next()

    all_records = read_jsonl(RAW_PATH)
    write_jsonl(KEPT_PATH, [x for x in all_records if x.get('keep')])
    write_jsonl(REJECTED_PATH, [x for x in all_records if not x.get('keep')])
    summary = summarize_records(all_records)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output_dir': str(OUTPUT_DIR), 'summary': summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
