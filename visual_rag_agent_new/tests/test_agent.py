import json
import re
import tempfile
from pathlib import Path

from PIL import Image

from src.agent import NO_NEW_PAGE_NOTE, Agent
from src.prompts import question_text
from src.protocol import get_frame
from tests.fakes import (ADD, ANSWER, CROP, EXPAND, RESOLVE, REVISIT, SEARCH, UPD_FACT, FakeRetriever, FakeVLM, act,
                         page)

TABLE = {"gold": ["deck/page_01", "deck/page_77"], "second": ["deck/page_02", "deck/page_78"]}


def _agent(responses, **kw):
    vlm = FakeVLM(responses)
    return Agent(vlm=vlm, retriever=FakeRetriever(table=TABLE), max_turns=kw.pop("max_turns", 8), retrieve_k=2, **kw), vlm


def _node(out, nid):
    return next(n for n in out["memory"]["nodes"] if n["id"] == nid)


def _target(sft):
    """The SFT target's page / tree / act blocks as {"page", "tree", "action"} (DESIGN.md §11.24)."""
    found = dict(re.findall(r"<(page|tree|act)>(.*?)</\1>", sft["target"], re.DOTALL))
    return {"page": json.loads(found["page"]), "tree": json.loads(found["tree"]), "action": json.loads(found["act"])}


def test_single_hop():
    agent, vlm = _agent([
        act(None, SEARCH("find gold")),
        act(page("results slide", [ADD("X is 42", [10, 10, 200, 50])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])),
    ])
    out = agent.run("What is X?")
    assert out["answer"] == "42" and out["terminated_by"] == "answer" and out["final_pages"] == ["deck/page_01"]
    assert out["counts"] == {"search": 1} and out["turns"] == 2
    assert (_node(out, "N0")["status"], _node(out, "N0")["pages"]) == ("resolved", ["P1"])
    assert [m["role"] for m in vlm.calls[0][0]] == ["system", "user"] and vlm.calls[0][1] == []
    assert "Answer only from what the retrieved pages show" in vlm.calls[0][0][0]["content"]
    assert "the answer only: a word, number or short phrase, no explanation" in vlm.calls[0][0][0]["content"]
    msgs, imgs = vlm.calls[1]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"] and len(imgs) == 1
    assert "page P1 (new; retrieved for: find gold; 728x532 pixels)" in msgs[3]["content"]
    assert "deck/page_01" not in json.dumps(msgs)                      # real page names stay hidden (§2, §11.12)


def test_multi_hop_with_crop_and_zoom_frame_fact():
    agent, vlm = _agent([
        act(None, SEARCH("find gold")),
        act(page("A chart", [ADD("A=3", [0, 0, 100, 100])]), SEARCH("second page"),
            tree=EXPAND("What is A?", "3", "What is B?", ["P1"])),
        act(page("B table"), CROP(bbox=[100, 100, 300, 200])),
        act(page(facts=[ADD("B=5", [0, 0, 50, 50])]), ANSWER("B (5) is larger than A (3)"), tree=RESOLVE("5", ["P2"])),
    ])
    out = agent.run("Compare A and B?")
    assert out["answer"] == "B (5) is larger than A (3)" and out["final_pages"] == ["deck/page_01", "deck/page_02"]
    msgs, imgs = vlm.calls[3]
    assert len(imgs) == 2 and msgs[3]["content"].count("<image>") == 2 and "ZOOM image" in msgs[3]["content"]
    zoom_fact = out["memory"]["pages"][1]["facts"][0]
    assert zoom_fact["src"] == "zoom" and 60 <= zoom_fact["bbox"][0] <= 120, zoom_fact
    assert out["counts"] == {"search": 2, "crop": 1}
    assert _node(out, "N1")["pages"] == ["P1"] and _node(out, "N2")["pages"] == ["P2"]
    assert (_node(out, "N0")["status"], _node(out, "N0")["answer"]) == ("resolved", "B (5) is larger than A (3)")


def test_crop_by_fact_id():
    agent, vlm = _agent([
        act(None, SEARCH("find gold")),
        act(page(facts=[ADD("small value", [100, 100, 300, 200])]), CROP(fact_id="F1")),
        act(None, ANSWER("v"), tree=RESOLVE("v", ["P1"])),
    ])
    out = agent.run("q")
    assert out["answer"] == "v" and len(vlm.calls[2][1]) == 2
    assert out["trace"][1]["act"]["action"] == {"type": "crop", "fact_id": "F1"}


def test_revisit_corrects_a_fact_and_an_earlier_answer():
    agent, vlm = _agent([
        act(None, SEARCH("find gold")),
        act(page(facts=[ADD("growth 2011 7.5% (third)", [0, 0, 100, 100])]), SEARCH("second page"),
            tree=EXPAND("Which year?", "2011", "Profit in 2011?", ["P1"])),
        act(page(facts=[ADD("profit 2011 12.5", [0, 0, 100, 100])]), REVISIT("P1"), tree=RESOLVE("12.5", ["P2"])),
        act(page(facts=[UPD_FACT("F1", "growth 2011 7.5%, 2007 7.4%", [0, 0, 120, 100])]), ANSWER("12.5"),
            tree=RESOLVE("2011 (7.5% > 7.4%)", ["P1"], node_id="N1")),
    ])
    out = agent.run("Profit in the third-growth year?")
    assert out["answer"] == "12.5" and out["counts"] == {"search": 2, "revisit": 1}
    last = out["trace"][-1]
    assert last["view"] == "page" and last["page"] == "P1" and last["events"]["resolved"] == ["N1"]
    assert "page P1 (revisit; 728x532 pixels)" in vlm.calls[3][0][3]["content"]
    assert (_node(out, "N1")["answer"], _node(out, "N1")["pages"]) == ("2011 (7.5% > 7.4%)", ["P1"])
    assert out["memory"]["pages"][0]["facts"][0]["revision"] == 1 and len(out["memory"]["pages"][0]["visits"]) == 2
    assert out["final_pages"] == ["deck/page_01", "deck/page_02"]


def test_tree_concludes_from_memory_without_a_revisit():
    """DESIGN.md §11.23: a page read before its question existed is attached by listing it in the tree's "pages";
    no revisit is needed to conclude from facts already in memory."""
    agent, vlm = _agent([
        act(None, SEARCH("find gold")),
        act(page("KPI slide", [ADD("2011 profit 12.5", [0, 0, 100, 100])]), SEARCH("second page")),
        act(page("growth chart", [ADD("2011 is third", [0, 0, 100, 100])]), ANSWER("12.5"),
            tree=RESOLVE("12.5", ["P1", "P2"])),
    ])
    out = agent.run("Profit in the third-growth year?")
    assert out["answer"] == "12.5" and out["counts"] == {"search": 2}
    assert _node(out, "N0")["pages"] == ["P1", "P2"] and out["final_pages"] == ["deck/page_01", "deck/page_02"]
    assert '"pages": ["P2"]' in vlm.calls[0][0][0]["content"]


def test_tree_edit_with_no_page_in_view():
    """After a search finds no new page the tree can still conclude from page memory (page must be null)."""
    agent, vlm = _agent([
        act(None, SEARCH("find gold")),
        act(page("KPI slide", [ADD("2011 profit 12.5", [0, 0, 100, 100])]), SEARCH("find gold")),     # -> page_77
        act(page("noise"), SEARCH("find gold")),                                                    # no new page
        act(None, ANSWER("12.5"), tree=RESOLVE("12.5", ["P1"])),
    ])
    out = agent.run("Profit in 2011?")
    assert out["trace"][-1]["view"] == "none" and out["answer"] == "12.5"
    assert _node(out, "N0")["pages"] == ["P1"] and out["final_pages"] == ["deck/page_01"]


def test_teacher_hint_reaches_the_teacher_but_not_the_sft_rows():
    """DESIGN.md §11.15: a teacher-only note is appended to the question the teacher sees; SFT rows are
    rendered without it (also when the SFT frame equals the teacher frame)."""
    hint = "This question needs evidence from two different pages."
    for sft_frame in (None, "qwen3"):
        agent, vlm = _agent([act(None, SEARCH("find gold")),
                             act(page(facts=[ADD("X is 42", [0, 0, 10, 10])]), ANSWER("42"), tree=RESOLVE("42", ["P1"]))],
                            frame="qwen3", sft_frame=sft_frame, teacher_hint=hint)
        out = agent.run("What is X?")
        assert all(hint in call[0][1]["content"] for call in vlm.calls)                     # every teacher call
        for row in out["trace"]:
            if row.get("sft"):
                assert hint not in json.dumps(row["sft"]["messages"])
                assert row["sft"]["messages"][1]["content"].startswith(question_text("What is X?"))
    agent, vlm = _agent([act(None, SEARCH("find gold")), act(None, ANSWER("x"))])
    agent.run("q")
    assert "Note:" not in vlm.calls[0][0][1]["content"]                                        # default: no note


def test_edit_error_is_soft_and_retried():
    agent, vlm = _agent([
        act(None, SEARCH("find gold")),
        act(None, REVISIT("P9")),                                                  # unknown page -> soft
        act(page(facts=[ADD("f", [900, 900, 950, 950])]), SEARCH("x")),            # off-page box -> soft
        act(None, SEARCH("x"), tree=RESOLVE("x", ["P1"], node_id="N7")),           # unknown question -> soft
        act(None, SEARCH("x"), tree=RESOLVE("x", ["P1"])),                         # P1 has no facts yet -> soft
        act(page(facts=[ADD("X is 42", [0, 0, 10, 10])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])),
    ])
    out = agent.run("q")
    assert out["answer"] == "42" and [r["step"] for r in out["trace"]] == ["turn"] + ["edit_error"] * 4 + ["turn"]
    assert "NOT applied" in vlm.calls[2][0][-1]["content"] and "Page in view: P1" in vlm.calls[2][0][-1]["content"]
    assert "has no facts" in vlm.calls[5][0][-1]["content"]
    assert len(out["memory"]["pages"][0]["facts"]) == 1


def test_format_error_after_retry():
    agent, vlm = _agent([act(None, SEARCH("find gold")), "<think>oops</think>", "still invalid"])
    out = agent.run("q")
    assert out["terminated_by"] == "format_error" and out["answer"] == "" and len(vlm.calls) == 3
    retry = vlm.calls[2][0][-1]["content"]
    assert "Your previous response was invalid" in retry
    assert "<page>{...} or null</page>" in retry and "<tree>{...} or null</tree>" in retry and '<act>{"type": ...}</act>' in retry


def test_budget_forces_one_answer_turn():
    agent, vlm = _agent([
        act(None, SEARCH("find gold")),
        act(page(facts=[ADD("A=3", [0, 0, 10, 10])]), SEARCH("second page"), tree=EXPAND("A?", "3", "B?", ["P1"])),
        act(None, ANSWER("3, B unknown")),
    ], max_turns=2)
    out = agent.run("A and B?")
    assert out["terminated_by"] == "budget_exhausted" and out["answer"] == "3, B unknown"
    forced = vlm.calls[2][0][-1]["content"]
    assert "Budget exhausted" in forced and "only what your page memory supports" in forced and out["trace"][-1]["forced"]
    assert "Turn 3 of 3:" in forced                                                    # turns 0..2: three model turns
    assert out["final_pages"] == ["deck/page_01"]


def test_turn_counter_is_shown_every_turn():
    """DESIGN.md §11.24: every turn states "Turn k of n", counting the model's own turns (the fixed first search with
    the question is not one); the answer-only turn after a failed budget turn shows only its own hint."""
    agent, vlm = _agent([act(None, SEARCH("find gold")), act(page("p"), SEARCH("more")), act(None, ANSWER("x"))],
                        max_turns=2)
    agent.run("q")
    hints = [call[0][-1]["content"] for call in vlm.calls]
    assert ["Turn 1 of 3:" in hints[0], "Turn 2 of 3:" in hints[1], "Turn 3 of 3:" in hints[2]] == [True] * 3
    assert "This turn:" not in "".join(hints)
    agent, vlm = _agent([act(None, SEARCH("more")), act(None, ANSWER("x"))], first_search="question", max_turns=2)
    agent.run("find gold")
    assert "Turn 1 of 2:" in vlm.calls[0][0][-1]["content"] and "Turn 2 of 2:" in vlm.calls[1][0][-1]["content"]
    agent, vlm = _agent([act(None, SEARCH("find gold")), act(None, SEARCH("more")), act(None, SEARCH("more")),
                         act(None, ANSWER("x"))], max_turns=1)                     # budget turn, its retry, answer-only
    out = agent.run("q")
    assert out["trace"][-1]["answer_only"] and "Turn 2 of 2:" in vlm.calls[1][0][-1]["content"]
    assert "Answer now" in vlm.calls[3][0][-1]["content"] and "Turn " not in vlm.calls[3][0][-1]["content"]


def test_failed_budget_turn_gets_one_answer_only_turn():
    """DESIGN.md §11.11: a budget turn that cannot be used is followed by one answer-only turn."""
    agent, vlm = _agent([act(None, SEARCH("find gold")), act(None, SEARCH("more")), act(None, SEARCH("more")),
                         act(None, ANSWER("x"))], max_turns=1)
    out = agent.run("q")
    assert out["terminated_by"] == "budget_exhausted" and out["answer"] == "x"
    assert [s["step"] for s in out["trace"]] == ["turn", "format_error", "turn"] and out["trace"][-1]["answer_only"]
    hint = vlm.calls[-1][0][-1]["content"]
    assert "Answer now" in hint and "<page>null</page>" in hint and "Fix it" not in hint
    # the answer-only turn keeps the answer and drops any page or tree edit written with it (§11.26): the memory is
    # untouched, and a tree listing a page without facts no longer loses the answer
    agent, vlm = _agent([act(None, SEARCH("find gold")), act(None, SEARCH("more")), act(None, SEARCH("more")),
                         act(page("p"), ANSWER("x"), tree=RESOLVE("x", ["P1"]))], max_turns=1)
    out = agent.run("q")
    assert out["terminated_by"] == "budget_exhausted" and out["answer"] == "x"
    last = out["trace"][-1]
    assert last["answer_only"] and last["act"]["page"] is None and last["act"]["tree"] is None
    assert all(p["descriptor"] != "p" for p in last["memory"]["pages"]) and last["memory"]["nodes"][0]["pages"] == []
    # any other action on the answer-only turn (and on its format retry) ends without an answer
    agent, vlm = _agent([act(None, SEARCH("find gold")), act(None, SEARCH("more")), act(None, SEARCH("more")),
                         act(None, SEARCH("again")), act(None, SEARCH("again"))], max_turns=1)
    out = agent.run("q")
    assert out["terminated_by"] == "format_error" and out["answer"] == ""
    # an edit error on the budget turn is handled the same way
    agent, vlm = _agent([act(None, SEARCH("find gold")), act(None, ANSWER("v"), tree=RESOLVE("v", ["P1"], node_id="N9")),
                         act(None, ANSWER("v2"))], max_turns=1)
    out = agent.run("q")
    assert out["terminated_by"] == "budget_exhausted" and out["answer"] == "v2"
    assert [s["step"] for s in out["trace"]] == ["turn", "edit_error", "turn"]


def test_turn_messages_strictly_alternate():
    """DESIGN.md §11.17: system = the policy alone; user = the question; then assistant = the previous response and
    one user message = observation + memory block. Training frameworks need this alternation."""
    agent, vlm = _agent([
        act(None, SEARCH("find gold")),
        act(page("A chart"), CROP(bbox=[10, 10, 300, 200])),
        act(page(facts=[ADD("X is 42", [10, 10, 200, 50])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])),
    ])
    out = agent.run("What is X?")
    assert out["answer"] == "42" and len(vlm.calls) == 3
    for i, (msgs, imgs) in enumerate(vlm.calls):
        roles = [m["role"] for m in msgs]
        assert roles == (["system", "user"] if i == 0 else ["system", "user", "assistant", "user"]), roles
        assert "=== QUESTION ===" not in msgs[0]["content"] and "What is X?" in msgs[1]["content"]
        assert msgs[-1]["content"].count("<image>") == len(imgs)
        assert "Question tree:" in msgs[-1]["content"]
    obs, state = vlm.calls[2][0][-1]["content"].split("\n\nQuestion tree:")
    assert obs.startswith("Observation: page P1") and "Zoom of page P1" in obs and len(vlm.calls[2][1]) == 2
    for row in out["trace"]:
        if row.get("sft"):
            roles = [m["role"] for m in row["sft"]["messages"]]
            assert roles in (["system", "user"], ["system", "user", "assistant", "user"]), roles


def test_sft_target_is_the_canonical_rendering():
    """DESIGN.md §11.11 / §11.24: lenient parsing accepts Qwen3.5's stray tags and an extra brace in a block;
    SFT targets are re-rendered canonically (all four blocks, compact JSON)."""
    raw = ('<think></think>\n\n<think>the slide gives X</think>\n<page>{"desc": "s", "facts": '
           '[{"op": "add_fact", "text": "X is 42", "bbox": [0, 0, 10, 10]}]}}</page>\n'
           '<tree>{"op": "resolve", "answer": "42", "pages": ["P1"]}</tree>\n<act>{"type": "answer", "text": "42"}</act>\n</think>')
    agent, vlm = _agent([act(None, SEARCH("find gold")), raw])
    out = agent.run("q")
    assert out["answer"] == "42" and out["terminated_by"] == "answer" and out["final_pages"] == ["deck/page_01"]
    target = out["trace"][-1]["sft"]["target"]
    assert target == ('<think>the slide gives X</think>\n'
                      '<page>{"desc":"s","facts":[{"op":"add_fact","text":"X is 42","bbox":[0,0,10,10]}]}</page>\n'
                      '<tree>{"op":"resolve","answer":"42","pages":["P1"]}</tree>\n<act>{"type":"answer","text":"42"}</act>')


def test_no_new_page_note():
    agent, vlm = _agent([
        act(None, SEARCH("find gold")),
        act(page("p"), SEARCH("find gold")),
        act(page("p"), SEARCH("find gold")),
        act(None, ANSWER("unknown")),
    ])
    out = agent.run("q")
    assert NO_NEW_PAGE_NOTE in vlm.calls[3][0][-1]["content"] and "Page in view: none" in vlm.calls[3][0][-1]["content"]
    assert out["answer"] == "unknown" and out["final_pages"] == []


def test_teacher_image_budget_leaves_sft_images_at_the_student_size():
    """DESIGN.md §11.25: image_max_pixels lowers only what the running model sees (page and zoom); boxes keep their
    original-pixel meaning and SFT rows keep the student's images."""
    budget = 128 * 28 * 28
    with tempfile.TemporaryDirectory() as tmp:
        agent, vlm = _agent([
            act(None, SEARCH("find gold")),
            act(page(facts=[ADD("title", [0, 0, 1000, 100])]), CROP(bbox=[250, 250, 500, 500])),
            act(page(facts=[ADD("v", [0, 0, 1000, 1000])]), ANSWER("v"), tree=RESOLVE("v", ["P1"])),
        ], frame="qwen3", sft_frame="qwen2.5-vl", image_max_pixels=budget)
        out = agent.run("q", output_dir=tmp)
        assert vlm.calls[1][1][0].size == (352, 256)                                   # the 800x600 page at ~0.1 MP
        assert all(w * h <= 1.2 * budget for w, h in (im.size for im in vlm.calls[2][1])) and len(vlm.calls[2][1]) == 2
        facts = out["memory"]["pages"][0]["facts"]
        assert facts[0]["bbox"] == [0.0, 0.0, 800.0, 60.0] and facts[1]["bbox"] == [172.0, 122.0, 428.0, 328.0]
        row2 = out["trace"][2]["sft"]
        assert [Image.open(q).size for q in row2["image_paths"]] == [(728, 532), (504, 420)]   # the student's sizes


def test_teacher_frame_rewritten_to_student_frame():
    with tempfile.TemporaryDirectory() as tmp:
        agent, vlm = _agent([
            act(None, SEARCH("find gold")),
            act(page(facts=[ADD("title", [0, 0, 1000, 100])]), CROP(bbox=[250, 250, 500, 500])),
            act(page(facts=[ADD("v", [0, 0, 1000, 1000])]), ANSWER("v"), tree=RESOLVE("v", ["P1"])),
        ], frame="qwen3", sft_frame="qwen2.5-vl")
        out = agent.run("q", output_dir=tmp)
        # the teacher works in its native frame: 0-1000, no pixel sizes, images on its 32-px grid
        t_msgs, t_imgs = vlm.calls[1]
        assert "0-1000" in t_msgs[0]["content"] and "0-1000 RELATIVE" in t_msgs[-1]["content"]
        assert "pixels" not in t_msgs[3]["content"].split("\n\nQuestion tree:")[0] and t_imgs[0].size == (704, 544)
        # boxes are stored in original page pixels (800x600)
        facts = out["memory"]["pages"][0]["facts"]
        assert facts[0]["bbox"] == [0.0, 0.0, 800.0, 60.0]
        assert facts[1]["src"] == "zoom" and facts[1]["bbox"] == [172.0, 122.0, 428.0, 328.0]   # crop [200,150,400,300] + 28 px
        # SFT rows are re-rendered in the student's frame
        row1, row2 = out["trace"][1]["sft"], out["trace"][2]["sft"]
        t1 = _target(row1)
        assert t1["page"]["facts"][0]["bbox"] == [0, 0, 728, 53] and t1["action"]["bbox"] == [182, 133, 364, 266]
        assert "ABSOLUTE PIXELS on the PAGE image (728 wide x 532 high)" in row1["messages"][-1]["content"]
        assert "0-1000" not in row1["messages"][0]["content"] + row1["messages"][-1]["content"]
        assert row2["messages"][2]["content"] == row1["target"]                      # previous turn, student frame
        assert [Image.open(q).size for q in row2["image_paths"]] == [(728, 532), (504, 420)]   # zoom 256x206 raw
        assert _target(row2)["page"]["facts"][0]["bbox"] == [0, 0, 504, 420] and _target(row2)["tree"]["pages"] == ["P1"]


def test_run_artifacts_and_image_paths():
    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = _agent([
            act(None, SEARCH("find gold")),
            act(None, CROP(bbox=[100, 100, 300, 200])),
            act(None, SEARCH("second page")),
            act(None, REVISIT("P1")),
            act(page(facts=[ADD("X is 42", [0, 0, 10, 10])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])),
        ])
        out = agent.run("q", output_dir=tmp)
        paths = out["trace"][2]["sft"]["image_paths"]
        assert len(paths) == 2 and all(Path(p).exists() for p in paths)
        revisit_paths = out["trace"][4]["sft"]["image_paths"]
        assert revisit_paths == [out["trace"][1]["sft"]["image_paths"][0]]   # revisit reuses the stored page
        assert (Path(tmp) / "trace.jsonl").exists() and (Path(tmp) / "memory_final.json").exists()
        assert (Path(tmp) / "answer.txt").read_text() == "42"


def test_first_search_is_the_question():
    agent, vlm = _agent([
        act(page(facts=[ADD("X is 42", [10, 10, 200, 50])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])),
    ], first_search="question")
    out = agent.run("find gold?")
    assert out["answer"] == "42" and out["counts"] == {"search": 1} and out["final_pages"] == ["deck/page_01"]
    assert out["trace"][0]["step"] == "auto_search" and "sft" not in out["trace"][0]       # never an SFT row
    msgs, imgs = vlm.calls[0]                                                              # the model starts on P1
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"] and len(imgs) == 1
    assert msgs[2]["content"] == ('<think>First, search with the question itself.</think>\n<page>null</page>\n'
                                  '<tree>null</tree>\n<act>{"type":"search","query":"find gold?"}</act>')
    assert "retrieved for: find gold?" in msgs[3]["content"]
    assert agent.retriever.deck_names == [None] and out["trace"][1]["turn"] == 1


def test_first_search_counts_against_the_turn_budget():
    agent, vlm = _agent([act(page(facts=[ADD("v", [0, 0, 10, 10])]), ANSWER("v"), tree=RESOLVE("v", ["P1"]))],
                        first_search="question", max_turns=1)
    out = agent.run("find gold")
    assert "Budget exhausted" in vlm.calls[0][0][-1]["content"] and out["terminated_by"] == "budget_exhausted"
    assert "Turn 1 of 1:" in vlm.calls[0][0][-1]["content"]


def test_search_guidance_prompt_variant():
    plain, _ = _agent([])
    guided, _ = _agent([], search_guidance=True)
    from src.prompts import policy_system
    base = policy_system(plain.frame)
    rich = policy_system(guided.frame, search_guidance=True)
    assert "The same query again shows the" in rich and "next-ranked page" in rich and "next-ranked" not in base
    assert "sure" in rich and "sure" in base and "next-ranked" not in base
    # The example of an unsure page repeats the query only where the cursor is explained.
    assert '<act>{"type":"search","query":"Nestle 2011 trading operating profit"}</act>' in rich.split("Unilever")[-1]
    assert "Unilever" not in base


def test_system_prompt_is_compact_and_leaves_per_turn_rules_to_the_turn_hint():
    from src.prompts import policy_system
    for frame in ("qwen3", "qwen2.5-vl"):
        for guided in (False, True):
            text = policy_system(get_frame(frame), search_guidance=guided)
            assert len(text) < 7000, (frame, guided, len(text))  # ... -> 6.4k (§11.21) -> 6.2k (§11.22) -> 6.6k (§11.23 tree) -> four blocks (§11.24)
            assert "first resolved from a page (expand)" in text                     # bridge rule (§11.12)
            assert "file name" not in text
            examples = text.split("Examples")[1]
            for think in re.findall(r"<think>(.*?)</think>", examples, re.DOTALL):
                assert len(think) <= 170, think                          # one short clause per example
            rules = text.split("Rules:")[1].split("Examples")[0]
            assert rules.count("\n- ") == 3, rules
            # The turn hint states these every turn and the protocol rejects violations.
            assert "No page in view:" not in text and "A zoom is in view" not in text


def test_system_prompt_crop_by_answer_type():
    """DESIGN.md §11.26 (user: crop share >= 30%): the crop trigger is the objective one of the 83.97 run - zoom before
    resolving whenever the answer or an intermediate value is a number, date, name, label or table cell - with the
    "read, then zoom to confirm" examples. Subjective triggers ("small or dense", "hard to read") gave 3-8% (§11.24-25)."""
    from src.prompts import policy_system
    for frame in ("qwen3", "qwen2.5-vl"):
        for guided in (False, True):
            text = policy_system(get_frame(frame), search_guidance=guided)
            crop = text.split('{"type": "crop"', 1)[1].split('{"type": "answer"')[0]
            assert "zoom into the page in view, once per visit. Do this before resolving whenever the answer or an" in crop
            assert "intermediate value is a number, date, name, label or table cell: read it from the zoom, not from the" in crop
            assert "full page. Titles, headings and page-level content can be read directly." in crop
            for gone in ("small or dense", "too small to read", "needs no zoom", "before leaving a page", "hides answer-critical"):
                assert gone not in text, gone
            examples = text.split("Examples")[1]
            chart, expand, kpi, zoom = examples.split("</act>")[:4]
            assert "zoom to confirm the labels" in chart and '<act>{"type":"crop"' in chart and '"op":"add_fact"' in chart
            assert '<tree>{"op":"expand"' in expand and '"pages":["P1"]' in expand and '<page>{"desc":null,"facts":[]}</page>' in expand
            assert "zoom to confirm the cell" in kpi and '"op":"add_fact","text":"Trading operating profit 2011: 12.5 bn"' in kpi
            assert '"op":"update_fact","fact_id":"F1"' in zoom and '"pages":["P2"]' in zoom and '"type":"answer"' in zoom
            assert examples.count('"type":"crop"') == 3                              # the chart, the KPI table, the footnote
            assert "2796" not in examples                                    # SCORE Fig. 4: never read from the full page


def test_system_prompt_is_the_83_97_prompt_in_four_blocks():
    """DESIGN.md §11.26 (user decision: accuracy at the 83.9 level, four blocks, crop >= 30%): the §11.22 prompt of the
    83.97 run (outputs/new_eval/prompt_fix_qwen3.txt) with only the four blocks, the tree's "pages" and the examples'
    block layout changed. Its SCORE layout (Part 1 / Part 2, the observe sentence, the query advice), rules, revisit
    and answer wording are verbatim; the 84.97 prompt's wording (§11.24 final) is gone."""
    from src.prompts import policy_system
    for frame in ("qwen3", "qwen2.5-vl"):
        for guided in (False, True):
            text = policy_system(get_frame(frame), search_guidance=guided)
            assert text.startswith("You are a visual-RAG agent. You answer a question from a corpus of page images by searching pages, "
                                   "reading\nthem and recording what they give in a memory that is shown to you every turn.")
            order = ["MEMORY (shown every turn):",
                     "EVERY TURN output exactly these four blocks, in order, with no text outside them:",
                     "Part 1 - <think>your reasoning</think>",
                     "derive the intermediate conclusion yourself (rank,\n  compare or compute the values; name the year, "
                     "item or page you need next), stated explicitly.",
                     'Part 2 - <page>{"desc": ..., "facts": [...]}</page> <tree>{...}</tree> <act>{...}</act>   one JSON value each',
                     "  <page>: what the page in view gives (null with no page in view; null / [] for parts you do not change):",
                     "  <tree>: what you conclude for the ACTIVE question, or null:", "  <act>, exactly one:",
                     "Rules:", "Examples (each block is one full response):"]
            assert [text.index(x) for x in order] == sorted(text.index(x) for x in order), order
            assert "Search with the" in text and "question itself; refine the query only for a specific intermediate target" in text
            assert ("The same query again shows the next-ranked page." in text) == guided
            revisit = text.split('{"type": "revisit", "page_id": "P1"}', 1)[1].split('{"type": "crop"', 1)[0]
            assert "look again at a page seen before (not the one in view), e.g. to" in revisit
            assert "re-read a value or correct a fact or an answer." in revisit
            assert '"pages": the pages whose facts the answer rests on (the page in view and earlier pages).' in text
            rules = text.split("Rules:")[1].split("Examples")[0]
            assert rules.count("\n- ") == 3 and "first resolved from a page (expand)" in rules
            assert ('Every page whose facts an answer uses must be listed in its "pages" (the page in view and earlier pages '
                    'alike);\n  a listed page must have facts.') in rules
            assert "only when you are sure it contains that answer" in rules and "If not sure, " in rules
            answer_spec = text.split('{"type": "answer", "text": "<final answer>"}', 1)[1].split("Rules:", 1)[0]
            assert answer_spec.lstrip().startswith('finish. "text" is the answer only')
            for gone in ("Prefer another search", "every part of the question", '"update"', '"uses"',
                         "Each turn you see at most one", "what the page covers", "once per page per question"):
                assert gone not in text, gone
            examples = text.split("Examples")[1]
            assert ('Resolve with pages P1 and P2.</think>\n<page>{"desc":null,"facts":[]}</page>\n'
                    '<tree>{"op":"resolve","answer":"CHF 12.5 bn","pages":["P1","P2"]}</tree>') in examples
            assert '"type":"revisit"' not in examples
    # the teacher's prompt (qwen3 frame, guided) is the archived 83.97 prompt plus the four-block edits, nothing else
    assert len(policy_system(get_frame("qwen3"), search_guidance=True)) == 6312


def test_system_prompt_describes_the_four_blocks():
    """DESIGN.md §11.24: <think>, <page>, <tree>, <act> - one JSON value per block instead of one nested object."""
    from src.prompts import policy_system
    text = policy_system(get_frame("qwen3"), search_guidance=True)
    assert "EVERY TURN output exactly these four blocks, in order, with no text outside them:" in text
    assert '{"op": "resolve", "answer": "<its answer>", "pages": ["P2"]}' in text and '"node_id": "N1"' in text
    assert '"next": "<the rest>",' in text and '"pages": ["P1"]}' in text
    assert '{"type": "revisit", "page_id": "P1"}' in text and '{"type": "answer", "text": "<final answer>"}' in text
    for gone in ("validate", "support_nodes", "task_ops", "page_ops", "memory_edit", "partial", '"update"', '"uses"',
                 '"page":', '"tree":', '"action":', "all three keys"):
        assert gone not in text, gone
    for block in text.split("Examples (each block is one full response):")[1].strip().split("\n\n"):
        assert re.findall(r"<(think|page|tree|act)>", block) == ["think", "page", "tree", "act"], block
