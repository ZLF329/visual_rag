import importlib.util
import tempfile
from pathlib import Path

from src.dataset import done_sample_ids, page_key, reference_labels, repair_jsonl_tail, retrieval_deck

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("evaluate_script", ROOT / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def test_reference_labels_and_keys():
    assert reference_labels({"deck_name": "d1", "evidence_pages": [12, 3]}) == ["d1/page_03", "d1/page_12"]
    assert page_key("d1/page_03") == ("d1", 3) and page_key("d1/page_3") == ("d1", 3)
    assert page_key("page_03") == ("", 3) and page_key("d1/cover") is None


def test_retrieval_deck():
    row = {"deck_name": "d1"}
    assert retrieval_deck(row, "flat") is None and retrieval_deck(row, "deck") == "d1"
    try:
        retrieval_deck(row, "global")
    except ValueError:
        return
    raise AssertionError("unknown scope accepted")


def test_eval_summary_matches_deck_and_page():
    rows = [
        {"reference_pages": ["d1/page_01"], "final_pages": ["d1/page_01", "d2/page_04"], "counts": {}, "turns": 2},
        {"reference_pages": ["d1/page_02"], "final_pages": ["d9/page_02"], "counts": {}, "turns": 2},
    ]
    summary = evaluate.summarize(rows)
    assert summary["final_coverage"] == 0.5 and summary["avg_non_gold_final_pages"] == 1.0


def test_done_sample_ids_ignores_partial_line():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "predictions.jsonl"
        assert done_sample_ids(path) == set()
        path.write_text('{"sample_id": "a"}\n{"sample_id": "b"}\n{"sample_id": "c", "tra', encoding="utf-8")
        assert done_sample_ids(path) == {"a", "b"}


def test_eval_summary_accuracy_by_hop():
    rows = [
        {"hop": "single", "judge": {"correct": True}, "counts": {}, "turns": 2},
        {"hop": "single", "judge": {"correct": False}, "counts": {}, "turns": 2},
        {"hop": "multi", "judge": {"correct": True}, "counts": {}, "turns": 4},
        {"hop": "multi", "judge_error": "timeout", "counts": {}, "turns": 4},
    ]
    by_hop = evaluate.summarize(rows)["accuracy_by_hop"]
    assert by_hop == {"multi": {"judged": 1, "correct": 1, "accuracy": 1.0},
                      "single": {"judged": 2, "correct": 1, "accuracy": 0.5}}


def test_repair_jsonl_tail_drops_a_torn_line():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "predictions.jsonl"
        repair_jsonl_tail(path)                                            # missing file: no-op
        path.write_text('{"sample_id": "a"}\n{"sample_id": "b", "tr', encoding="utf-8")
        repair_jsonl_tail(path)
        assert path.read_text(encoding="utf-8") == '{"sample_id": "a"}\n'
        repair_jsonl_tail(path)                                            # intact file: unchanged
        assert path.read_text(encoding="utf-8") == '{"sample_id": "a"}\n'


