from PIL import Image

from src.memory import Memory
from src.schemas import AnalyseResult, MemoryUpdateResult


def test_memory_write_yes_partial_no(tmp_path) -> None:
    image = Image.new("RGB", (1000, 1000), "white")
    memory = Memory(original_query="question", image_dir=tmp_path / "images")

    memory.write(
        AnalyseResult(
            think="saw it",
            summary="page summary yes",
            key_facts=["fact yes"],
            decision="yes",
            answer_text="answer",
            cited_statements=["evidence"],
        ),
        image,
        "query one",
        page_label="doc/page1",
    )
    assert len(memory.confirmed) == 1
    assert memory.confirmed[0].summary == "page summary yes"
    assert memory.confirmed[0].key_facts == ["fact yes"]
    assert memory.confirmed[0].image_ref.pixel_budget == 400_000
    assert memory.confirmed[0].image_ref.path is not None

    memory.write(
        AnalyseResult(
            think="some evidence",
            summary="page summary partial",
            key_facts=["fact partial"],
            decision="partial",
            partial_answer="partial 1",
            cited_statements=["evidence"],
            remaining_gap="gap",
        ),
        image,
        "query two",
        page_label="doc/page2",
    )
    assert len(memory.partials) == 1
    assert memory.partials[0].summary == "page summary partial"
    assert memory.partials[0].key_facts == ["fact partial"]
    assert memory.partials[0].image_ref.pixel_budget == 200_000

    memory.write(
        AnalyseResult(
            think="bad page",
            summary="irrelevant page summary",
            decision="no",
            failure_type="wrong_doc_section",
            suggestion="search another section",
        ),
        image,
        "query three",
        page_label="doc/page3",
    )
    assert len(memory.warnings) == 1
    assert memory.warnings[0].summary == "irrelevant page summary"
    assert memory.warnings[0].failure_type == "wrong_doc_section"
    assert len(memory.observations) == 3
    assert memory.observations[0]["decision"] == "yes"
    assert memory.observations[1]["key_facts"] == ["fact partial"]
    assert "key_facts" not in memory.observations[2]


def test_partial_fifo_cap_is_two() -> None:
    image = Image.new("RGB", (100, 100), "white")
    memory = Memory(original_query="question")

    for idx in range(3):
        memory.iter = idx
        memory.write(
            AnalyseResult(
                think="some evidence",
                summary=f"page summary partial {idx}",
                key_facts=[f"fact partial {idx}"],
                decision="partial",
                partial_answer=f"partial {idx}",
                cited_statements=[],
                remaining_gap="gap",
            ),
            image,
            f"query {idx}",
        )

    assert len(memory.partials) == 2
    assert [item.partial_answer for item in memory.partials] == ["partial 1", "partial 2"]


def test_memory_update_consolidated_context() -> None:
    memory = Memory(original_query="question")
    image = Image.new("RGB", (100, 100), "white")
    memory.write(
        AnalyseResult(
            think="saw it",
            summary="raw yes page summary",
            key_facts=["raw yes fact"],
            decision="yes",
            answer_text="raw answer",
            cited_statements=["raw citation"],
        ),
        image,
        "raw query",
        page_label="doc/raw",
    )
    memory.append_consolidated_summary("combined memory")
    memory.update_consolidated(
        MemoryUpdateResult(key_facts=["fact one", "fact two"])
    )
    context = memory.context_for_decide()
    assert "combined memory" in context
    assert "fact one" in context
    assert "raw yes page summary" not in context
    assert "raw yes fact" not in context
    assert "raw answer" not in context
    assert "decision" not in context
    assert memory.consolidated_key_facts == ["fact one", "fact two"]
