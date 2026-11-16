import importlib.util
import tempfile
from pathlib import Path

from src.agent import Agent
from tests.fakes import ADD, ANSWER, CROP, EXPAND, RESOLVE, REVISIT, SEARCH, UPD_FACT, FakeRetriever, FakeVLM, act, page

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("gen_sft", ROOT / "scripts" / "generate_sft_trajectories.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

TABLE = {"gold": ["deck/page_01", "deck/page_77"], "second": ["deck/page_02", "deck/page_78"]}


def _run():
    tmp = tempfile.mkdtemp(prefix="new_sft_")
    vlm = FakeVLM([
        act(None, SEARCH("find gold")),
        act(page("A chart", [ADD("A=3", [0, 0, 100, 100])]), SEARCH("second page"), tree=EXPAND("A?", "3", "B?", ["P1"])),
        act(page("B table"), CROP(bbox=[100, 100, 300, 200])),
        act(page(facts=[ADD("B=5", [0, 0, 50, 50])]), REVISIT("P1"), tree=RESOLVE("5", ["P2"])),
        act(page(facts=[UPD_FACT("F1", "A=3 exactly", [0, 0, 100, 100])]), ANSWER("B is larger")),
    ])
    agent = Agent(vlm=vlm, retriever=FakeRetriever(table=TABLE), max_turns=8, retrieve_k=2)
    result = agent.run("Compare A and B?", output_dir=tmp)
    result.update({"sample_id": "t1", "gold_answer": "B", "evidence_pages": [1, 2],
                   "reference_pages": ["deck/page_01", "deck/page_02"], "judge": {"correct": True, "score": 1}})
    result["tags"] = gen.trajectory_tags(result)
    return result


def test_build_sft_calls_shapes():
    result = _run()
    calls = gen.build_sft_calls(result)
    assert [c["action"] for c in calls] == ["search", "search", "crop", "revisit", "answer"]
    assert [c["view"] for c in calls] == ["none", "page", "page", "zoom", "page"]
    for c in calls:
        markers = sum(str(m["content"]).count("<image>") for m in c["messages"])
        assert markers == len(c["image_paths"]), (c["id"], markers, c["image_paths"])
        assert all(Path(p).exists() for p in c["image_paths"])
        assert c["conversations"][-1]["from"] == "gpt" and c["conversations"][-1]["value"] == c["target"]
    assert len(calls[3]["image_paths"]) == 2                    # zoom turn: page + zoom
    assert "F1:" not in calls[0]["messages"][-1]["content"]     # a turn only sees memory committed before it
    assert "├─ F1: A=3" not in calls[1]["messages"][-1]["content"] and "F1: A=3" in calls[2]["messages"][-1]["content"]


def test_tags_and_filter():
    result = _run()
    assert result["tags"] == {"revisit": 1, "crop": 1, "update_fact": 1, "expand": 1}
    assert result["final_pages"] == ["deck/page_01", "deck/page_02"]
    keep, reasons = gen.should_keep_trajectory(result, require_judge_correct=True, require_all_reference_pages=True)
    assert keep, reasons
    bad = dict(result, trace=list(result["trace"]) + [{"step": "edit_error", "error": "x"}])
    assert "error_steps:edit_error" in gen.should_keep_trajectory(bad, require_judge_correct=True, require_all_reference_pages=True)[1]
    bad = dict(result, terminated_by="budget_exhausted")
    assert "terminated_by:budget_exhausted" in gen.should_keep_trajectory(bad, require_judge_correct=False, require_all_reference_pages=False)[1]
    bad = dict(result, final_pages=["deck/page_01"])
    assert "missing_reference_pages:deck/page_02" in gen.should_keep_trajectory(bad, require_judge_correct=False, require_all_reference_pages=True)[1]
    # flat corpus: the same page number in another deck is not gold
    bad = dict(result, final_pages=["other/page_01", "deck/page_02"])
    assert "missing_reference_pages:deck/page_01" in gen.should_keep_trajectory(bad, require_judge_correct=False, require_all_reference_pages=True)[1]
    bad = dict(result, judge={"correct": False})
    assert "judge_incorrect" in gen.should_keep_trajectory(bad, require_judge_correct=True, require_all_reference_pages=False)[1]


def test_run_totals_from_trajectories():
    rows = [
        {"sample_id": "a", "sft_keep": True, "sft_reject_reasons": [], "hop": "multi", "tags": {"revisit": 1, "crop": 0}},
        {"sample_id": "b", "sft_keep": False, "sft_reject_reasons": ["judge_incorrect", "missing_reference_pages:d/page_02"],
         "hop": "single", "tags": {"revisit": 0, "crop": 2}},
        {"sample_id": "c", "sft_keep": True, "sft_reject_reasons": [], "hop": "single", "tags": {"revisit": 0, "crop": 1}},
    ]
    totals = gen.run_totals(rows)
    assert totals["processed"] == 3 and totals["kept"] == 2
    assert totals["kept_by_hop"] == {"multi": 1, "single": 1} and totals["processed_by_hop"] == {"multi": 1, "single": 2}
    assert totals["reject_reasons"] == {"judge_incorrect": 1, "missing_reference_pages": 1}       # grouped by kind
    assert totals["kept_with"] == {"revisit": 1, "crop": 1}

