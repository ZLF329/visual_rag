from PIL import Image
import pytest

from src.agent import Agent
from src.schemas import AnalyseResult, DecideResult, MemoryUpdateResult


class FakeRetriever:
    def search(self, query: str, top_k: int = 1):
        return [(Image.new("RGB", (100, 100), "white"), "doc/page1")]


class FakeVLM:
    def __init__(self) -> None:
        self.decide_calls = 0

    def decide_next(self, original_query, memory, warnings_summary=None):
        self.decide_calls += 1
        if self.decide_calls == 1:
            assert memory.retained_image_paths() == []
            return DecideResult(think="need evidence", action="search", content=original_query)
        assert len(memory.retained_image_paths()) == 1
        return DecideResult(think="enough", action="answer", content="final fact")

    def analyse(self, image, original_query, memory_context):
        return AnalyseResult(
            think="visible",
            summary="page summary final",
            key_facts=["final fact"],
            decision="yes",
            answer_text="final fact",
            cited_statements=["visible fact"],
        )

    def update_memory(self, *, original_query, memory, latest_analysis, max_new_tokens=1000):
        assert max_new_tokens == 1000
        assert latest_analysis.decision == "yes"
        assert latest_analysis.summary == "page summary final"
        assert latest_analysis.key_facts == ["final fact"]
        return MemoryUpdateResult(key_facts=["compact fact"])


def test_agent_smoke(tmp_path) -> None:
    agent = Agent(vlm=FakeVLM(), retriever=FakeRetriever(), top_k=1, max_iters=5)
    result = agent.run("question", output_dir=tmp_path / "run")
    assert result["answer"] == "final fact"
    assert result["terminated_by"] == "decide"
    assert len(result["memory"]["confirmed"]) == 1
    assert result["memory"]["consolidated_summary"] == "[query: question] page summary final"
    assert result["memory"]["consolidated_key_facts"] == ["compact fact"]
    assert any(step["step"] == "memory_update" for step in result["trace"])
    assert (tmp_path / "run" / "trace.jsonl").exists()


class OomVLM:
    def decide_next(self, original_query, memory, warnings_summary=None):
        raise RuntimeError("CUDA out of memory. Tried to allocate 96.00 MiB")


def test_agent_reraises_cuda_oom(tmp_path) -> None:
    agent = Agent(vlm=OomVLM(), retriever=FakeRetriever(), top_k=1, max_iters=5)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        agent.run("question", output_dir=tmp_path / "oom")


class NoJudgeVLM:
    def __init__(self) -> None:
        self.decide_calls = 0

    def decide_next(self, original_query, memory, warnings_summary=None):
        self.decide_calls += 1
        if self.decide_calls == 1:
            return DecideResult(think="need page", action="search", content=original_query)
        return DecideResult(think="stop", action="answer", content="final from decide")

    def analyse(self, image, original_query, memory_context):
        return AnalyseResult(
            think="not useful",
            summary="irrelevant page summary",
            key_facts=["should be ignored"],
            judge="no",
        )

    def update_memory(self, **kwargs):
        raise AssertionError("memory_update should not be called for judge=no")


def test_agent_appends_no_summary_without_memory_update(tmp_path) -> None:
    agent = Agent(vlm=NoJudgeVLM(), retriever=FakeRetriever(), top_k=1, max_iters=5)
    result = agent.run("question", output_dir=tmp_path / "no_run")
    assert result["answer"] == "final from decide"
    assert result["memory"]["consolidated_summary"] == "[query: question] irrelevant page summary"
    assert result["memory"]["consolidated_key_facts"] == []
    assert any(step["step"] == "memory_update_skipped" for step in result["trace"])
