from src.memory import Memory
from src.schemas import AnalyseResult, MemoryUpdateResult
from src.vlm import VLM


def test_update_memory_drops_key_facts_when_latest_judge_is_no() -> None:
    vlm = VLM(load_model=False)
    memory = Memory(original_query="question")
    captured = {}

    def fake_generate_structured(**kwargs):
        captured["user"] = kwargs["user"]
        return MemoryUpdateResult(key_facts=[])

    vlm._generate_structured = fake_generate_structured  # type: ignore[method-assign]
    vlm.update_memory(
        original_query="question",
        memory=memory,
        latest_analysis=AnalyseResult(
            think="not relevant",
            summary="This page is about a different section.",
            key_facts=["should not be passed"],
            judge="no",
        ),
    )

    assert "Latest page summary" not in captured["user"]
    assert "Latest page key_facts:\n[]" in captured["user"]
    assert "should not be passed" not in captured["user"]
