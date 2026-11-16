"""Dataset rows -> agent inputs, and the page-label helpers shared by evaluation and SFT generation.

A page label is "<deck>/page_NN", the layout of the retriever index. Gold evidence is compared as
(deck, page) keys, so in a flat corpus a same-numbered page of another deck never counts as gold.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# flat: one corpus over every deck of the split (SCORE setting); deck: search only the question's deck.
RETRIEVAL_SCOPES = ("flat", "deck")


def read_jsonl(path: Path, *, start: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if limit is not None and len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def repair_jsonl_tail(path: Path) -> None:
    """Drop a torn last line (a crash mid-append) so resumed appends start on a fresh line."""
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])


def done_sample_ids(path: Path) -> set[str]:
    """sample_ids already committed to a run's jsonl (resume); a torn last line (crash mid-write) is ignored."""
    path = Path(path)
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(str(json.loads(line)["sample_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def first_present(row: dict[str, Any], names: tuple[str, ...], default: Any = "") -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def extract_question(row: dict[str, Any]) -> str:
    return str(first_present(row, ("query", "question", "prompt", "input", "problem"))).strip()


def extract_answer(row: dict[str, Any]) -> str:
    value = first_present(row, ("answer", "answers", "target", "label", "response"))
    return " | ".join(str(v) for v in value) if isinstance(value, list) else str(value)


def extract_deck_name(row: dict[str, Any]) -> str:
    return str(first_present(row, ("deck_name", "deck_id", "doc_id", "document_id", "pdf_id"))).strip()


def extract_sample_id(row: dict[str, Any], idx: int) -> str:
    return str(first_present(row, ("id", "qid", "question_id", "uid", "eval_id", "sample_id"), idx))


def extract_reference_pages(row: dict[str, Any]) -> set[int]:
    """1-based gold page numbers of the question's deck."""
    value = first_present(row, ("evidence_pages", "reference_pages", "gold_pages", "page_ids", "pages", "answer_pages"), [])
    if isinstance(value, str):
        return {int(v) for v in re.findall(r"\d+", value)}
    if isinstance(value, (int, float)):
        return {int(value)}
    out: set[int] = set()
    for item in value or []:
        if isinstance(item, dict):
            item = first_present(item, ("page", "page_num", "page_id", "index"), None)
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            continue
    return out


def page_label(deck_name: str, page: int) -> str:
    return f"{deck_name}/page_{int(page):02d}"


def reference_labels(row: dict[str, Any]) -> list[str]:
    deck = extract_deck_name(row)
    return [page_label(deck, p) for p in sorted(extract_reference_pages(row))]


def page_number(label: Any) -> int | None:
    m = re.search(r"page[_-]?(\d+)", str(label or ""), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def page_key(label: Any) -> tuple[str, int] | None:
    """"deck/page_05" -> ("deck", 5); insensitive to zero padding."""
    deck, _, leaf = str(label or "").rpartition("/")
    n = page_number(leaf)
    return None if n is None else (deck, n)


def retrieval_deck(row: dict[str, Any], scope: str) -> str | None:
    """deck_name to pass to retriever.search: None searches the whole (flat) corpus."""
    if scope not in RETRIEVAL_SCOPES:
        raise ValueError(f"retrieval_scope must be one of {RETRIEVAL_SCOPES}, got {scope!r}")
    return (extract_deck_name(row) or None) if scope == "deck" else None


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))[:120] or "sample"
