from PIL import Image
import pytest

from src.schemas import DecideResult, ImageSummaryResult
from src.summary_baseline import AgenticSummaryBaseline


class FakeRetriever:
    def search(self, query: str, top_k: int = 1):
        return [(Image.new("RGB", (100, 100), "white"), f"page/{query}")]


class FakeVLM:
    def __init__(self) -> None:
        self.decide_calls = 0

    def summary_baseline_decide(
        self,
        *,
        original_query,
        summaries_context,
        recent_images,
        recent_image_labels,
    ):
        self.decide_calls += 1
        if self.decide_calls == 1:
            assert summaries_context == "None yet."
            assert recent_images == []
        else:
            assert "summary:" in summaries_context
            assert len(recent_images) >= 1
        if self.decide_calls == 3:
            return DecideResult(think="enough summaries", action="answer", content="baseline answer")
        return DecideResult(
            think="need another page",
            action="search",
            content=f"q{self.decide_calls}",
        )

    def summary_baseline_summarise_image(
        self,
        *,
        image,
        original_query,
        search_query,
        summaries_context,
        page_label,
    ):
        assert search_query.startswith("q")
        return ImageSummaryResult(
            summary=f"summary for {page_label}",
            key_facts=["fact"],
            visible_text=["text"],
        )


def test_agentic_summary_baseline(tmp_path) -> None:
    baseline = AgenticSummaryBaseline(
        vlm=FakeVLM(),
        retriever=FakeRetriever(),
        top_k=1,
        max_iters=3,
        recent_rounds=2,
    )
    result = baseline.run("question", output_dir=tmp_path / "run")
    assert result["answer"] == "baseline answer"
    assert result["baseline"] == "agentic_summary"
    assert result["terminated_by"] == "decide"
    assert len(result["summary_memory"]["summaries"]) == 2
    assert len(result["summary_memory"]["recent_images"]) == 2
    assert (tmp_path / "run" / "summary_memory.json").exists()


class OomBaselineVLM:
    def summary_baseline_decide(
        self,
        *,
        original_query,
        summaries_context,
        recent_images,
        recent_image_labels,
    ):
        raise RuntimeError("CUDA out of memory. Tried to allocate 96.00 MiB")


def test_agentic_summary_baseline_reraises_cuda_oom(tmp_path) -> None:
    baseline = AgenticSummaryBaseline(
        vlm=OomBaselineVLM(),
        retriever=FakeRetriever(),
        top_k=1,
        max_iters=3,
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        baseline.run("question", output_dir=tmp_path / "oom")
