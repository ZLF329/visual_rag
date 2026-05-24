from pathlib import Path

from PIL import Image

from src.memory import Memory
from src.schemas import AnalyseResult, DecideResult
from src.vlm import VLM


def test_decide_receives_retained_yes_and_partial_images(tmp_path: Path) -> None:
    vlm = VLM(load_model=False)
    memory = Memory(original_query="question", image_dir=tmp_path / "images")
    image = Image.new("RGB", (100, 100), "white")
    memory.write(
        AnalyseResult(
            think="yes",
            summary="yes summary",
            key_facts=["yes fact"],
            judge="yes",
        ),
        image,
        "q1",
        page_label="p1",
    )
    memory.write(
        AnalyseResult(
            think="partial",
            summary="partial summary",
            key_facts=["partial fact"],
            judge="partial",
        ),
        image,
        "q2",
        page_label="p2",
    )
    memory.write(
        AnalyseResult(
            think="no",
            summary="no summary",
            key_facts=["should not matter"],
            judge="no",
        ),
        image,
        "q3",
        page_label="p3",
    )
    captured = {}

    def fake_generate_structured(**kwargs):
        captured["images"] = kwargs["images"]
        return DecideResult(think="answer", action="answer", content="done")

    vlm._generate_structured = fake_generate_structured  # type: ignore[method-assign]
    vlm.decide_next("question", memory)

    assert len(captured["images"]) == 2
    assert len(memory.retained_image_paths()) == 2
