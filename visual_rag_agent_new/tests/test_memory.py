from PIL import Image

from src.memory import Memory, View, Zoom, apply_edit, render_pages, render_state, render_tree
from src.protocol import Act, EditError, PageEdit, crop_region, get_frame
from tests.fakes import ADD, EXPAND, RESOLVE, UPD_FACT

F, F3 = get_frame("qwen2.5-vl"), get_frame("qwen3")


def _mem(question="Q", pages=("deck/page_01",), size=(728, 532)):          # on the 28-px grid: shown == original
    m = Memory(question)
    for label in pages:
        m.add_page(label, Image.new("RGB", size, "white"), f"query for {label}")
    return m


def _edit(desc=None, facts=()):
    return PageEdit.model_validate({"desc": desc, "facts": list(facts)})


def _tree(op):
    return Act.model_validate({"tree": op, "action": {"type": "search", "query": "q"}}).tree


def _apply(m, edit=None, tree=None, view=None, turn=1, frame=F):
    return apply_edit(m, edit, None if tree is None else _tree(tree), view, turn=turn, frame=frame)


def _read(m, alias, text="fact", turn=1):
    """Record one fact on page `alias` (a page needs facts before a conclusion lists it)."""
    return _apply(m, _edit(facts=[ADD(text, [0, 0, 10, 10])]), view=View(alias, "search"), turn=turn)[0]


def _zoom(m, alias, box_raw):
    region = crop_region(box_raw, m.pages[alias].image.size)
    return Zoom(image=m.pages[alias].image.crop(region), region=region, turn=0)


def _raises(fn, exc=EditError):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_aliases_and_initial_state():
    m = _mem(pages=("deck/page_07", "deck/page_02"))
    assert list(m.pages) == ["P1", "P2"] and m.aliases["deck/page_07"] == "P1"
    assert m.add_page("deck/page_07", None, "x").id == "P1"
    assert m.active_id == "N0" and m.root.parent_id is None and m.children == [] and m.evidence_pages() == []


def test_add_fact_page_frame_and_desc():
    m = _mem()
    m2, ev = _apply(m, _edit("chart", [ADD("f1", [10, 10, 100, 50]), ADD("f2", [-5, 0, 900, 20])]), view=View("P1", "search"))
    p = m2.pages["P1"]
    assert [f.id for f in p.facts] == ["P1.F1", "P1.F2"] and [f.src for f in p.facts] == ["page", "page"]
    assert p.facts[1].bbox == [0.0, 0.0, 728.0, 20.0] and p.descriptor == "chart"
    assert ev["added_facts"] == ["P1.F1", "P1.F2"] and ev["page_desc"] == "P1"
    assert m.pages["P1"].facts == [] and m.pages["P1"].descriptor is None            # original untouched


def test_no_edit_is_a_no_op():
    m = _mem()
    m2, ev = _apply(m)
    assert m2.pages["P1"].facts == [] and ev["added_facts"] == [] and ev["resolved"] == []
    m3, _ = _apply(m, _edit())                                                       # an empty page object too
    assert m3.root.status == "open"


def test_zoom_frame_remap():
    m = _mem()
    z = _zoom(m, "P1", [100, 100, 300, 200])                       # region (72, 72, 328, 228), 256x156 raw
    assert F.shown_size(z.image.size) == (588, 364)
    view = View("P1", "search", zoom=z)
    m2, _ = _apply(m, _edit(facts=[ADD("z", [0, 0, 588, 364])]), view=view)
    f = m2.pages["P1"].facts[0]
    assert f.src == "zoom" and f.bbox == [72.0, 72.0, 328.0, 228.0]
    m3, _ = _apply(m, _edit(facts=[ADD("z", [0, 0, 500, 500])]), view=view, frame=F3)   # 0-1000 frame
    assert m3.pages["P1"].facts[0].bbox == [72.0, 72.0, 200.0, 150.0]


def test_page_frame_scales_to_original_pixels():
    m = _mem(size=(1024, 768))                                       # shown 728x532 (qwen2.5) / 0-1000 (qwen3)
    m2, _ = _apply(m, _edit(facts=[ADD("a", [364, 0, 728, 266])]), view=View("P1", "search"))
    assert m2.pages["P1"].facts[0].bbox == [512.0, 0.0, 1024.0, 384.0]
    m3, _ = _apply(m, _edit(facts=[ADD("a", [500, 0, 1000, 500])]), view=View("P1", "search"), frame=F3)
    assert m3.pages["P1"].facts[0].bbox == [512.0, 0.0, 1024.0, 384.0]


def test_soft_errors_leave_memory_untouched():
    m = _read(_mem(pages=("deck/page_01", "deck/page_02")), "P1", "a")
    view2 = View("P2", "search")
    _raises(lambda: _apply(m, _edit(facts=[UPD_FACT("P1.F1", "x", [0, 0, 10, 10])]), view=view2))   # other page
    _raises(lambda: _apply(m, _edit(facts=[UPD_FACT("P2.F9", "x", [0, 0, 10, 10])]), view=view2))   # unknown fact
    _raises(lambda: _apply(m, _edit(facts=[ADD("ok", [0, 0, 10, 10]), ADD("bad", [900, 900, 950, 950])]), view=view2))
    _raises(lambda: _apply(m, tree=RESOLVE("a", ["P1"], node_id="N7"), view=view2))                # unknown question
    _raises(lambda: _apply(m, tree=RESOLVE("a", ["P9"]), view=view2))                               # unknown page
    _raises(lambda: _apply(m, tree=RESOLVE("a", ["P2"]), view=view2))                               # page without facts
    _raises(lambda: _apply(m, tree=EXPAND("s", "a", "t", ["P1", "P2"]), view=view2))                # one bad page spoils it
    _raises(lambda: _apply(m, _edit(facts=[ADD("x", [0, 0, 10, 10])])))                             # no page in view
    _raises(lambda: _apply(m, _edit("d"), tree=RESOLVE("a", ["P1"])))                                # page edit, no page
    assert m.pages["P2"].facts == [] and m.root.status == "open" and len(m.nodes) == 1


def test_resolve_attaches_the_listed_pages_only():
    """DESIGN.md §11.23: a conclusion attaches exactly the pages it lists; the page in view is not implied."""
    m = _read(_read(_mem("Q", pages=("p1", "p2")), "P1", "X is 42"), "P2", "unrelated")
    m2, ev = _apply(m, tree=RESOLVE("42", ["P1"]), view=View("P2", "search"))
    assert (m2.root.status, m2.root.answer, m2.root.pages) == ("resolved", "42", ["P1"]) and ev["resolved"] == ["N0"]
    # resolving again adds the newly listed page; the original question stays fixed
    m3, _ = _apply(m2, tree=RESOLVE("43", ["P2", "P1"]), view=View("P2", "search"), turn=2)
    assert m3.root.pages == ["P1", "P2"] and m3.root.answer == "43" and m3.root.question == "Q"
    assert m3.evidence_pages() == ["P1", "P2"]


def test_tree_edit_from_memory_with_no_page_in_view():
    m = _read(_mem("Q", pages=("p1",)), "P1", "X is 42")
    m2, ev = _apply(m, tree=RESOLVE("42", ["P1"]), view=None)
    assert m2.root.pages == ["P1"] and ev["resolved"] == ["N0"]


def test_page_edit_is_applied_before_the_tree_edit():
    """The page in view can be listed in the same turn its first fact is recorded."""
    m = _mem("Q", pages=("p1",))
    _raises(lambda: _apply(m, tree=RESOLVE("42", ["P1"]), view=View("P1", "search")))              # no facts yet
    m2, ev = _apply(m, _edit(facts=[ADD("X is 42", [0, 0, 10, 10])]), RESOLVE("42", ["P1"]), View("P1", "search"))
    assert ev["added_facts"] == ["P1.F1"] and ev["resolved"] == ["N0"] and m2.root.pages == ["P1"]


def test_expand_from_root_then_from_child():
    m = _read(_read(_read(_mem("Q", pages=("p1", "p2", "p3")), "P1"), "P2"), "P3")
    m, ev = _apply(m, tree=EXPAND("s1", "a1", "s2", ["P1"]), view=View("P1", "search"))
    assert [n.id for n in m.nodes] == ["N0", "N1", "N2"] and ev["expanded"] == ["N1", "N2"]
    assert (m.node("N1").status, m.node("N1").answer, m.node("N1").pages) == ("resolved", "a1", ["P1"])
    assert m.node("N2").status == "open" and m.node("N2").pages == [] and m.active_id == "N2" and m.root.question == "Q"
    m, _ = _apply(m, tree=EXPAND("s2 part", "a2", "s3", ["P2"]), view=View("P2", "search"), turn=2)
    assert [n.id for n in m.nodes] == ["N0", "N1", "N2", "N3"]
    assert (m.node("N2").question, m.node("N2").status, m.node("N2").pages) == ("s2 part", "resolved", ["P2"])
    assert m.active_id == "N3" and all(n.parent_id == "N0" for n in m.children)
    assert m.evidence_pages() == ["P1", "P2"]


def test_resolve_with_node_id_corrects_an_earlier_question():
    m = _read(_read(_mem("Q", pages=("p1", "p2")), "P1"), "P2")
    m, _ = _apply(m, tree=EXPAND("s1", "a1", "s2", ["P1"]), view=View("P1", "search"))
    m, _ = _apply(m, tree=RESOLVE("a2", ["P2"]), view=View("P2", "search"), turn=2)                # the active N2
    assert m.node("N2").pages == ["P2"] and m.active_id == "N0"
    m2, ev = _apply(m, tree=RESOLVE("a1 fixed", ["P1"], node_id="N1"), view=View("P1", "revisit"), turn=3)
    assert (m2.node("N1").answer, m2.node("N1").pages) == ("a1 fixed", ["P1"]) and ev["resolved"] == ["N1"]
    assert m2.node("N2").answer == "a2"
    m3, _ = _apply(m, tree=RESOLVE("a1", ["P2"], node_id="N1"), view=View("P2", "search"), turn=3)
    assert m3.node("N1").pages == ["P1", "P2"]                                                       # page added


def test_tree_pages_attach_earlier_pages():
    """An earlier page is attached by listing it, with no revisit (replaces "uses", DESIGN.md §11.10 -> §11.23)."""
    m = _read(_read(_mem("Q", pages=("p1", "p2")), "P1", "profit 12.5"), "P2", "2011 is third")
    m2, ev = _apply(m, tree=RESOLVE("12.5", ["P2", "P1"]), view=View("P2", "search"), turn=2)
    assert m2.root.pages == ["P2", "P1"] and m2.evidence_pages() == ["P2", "P1"] and ev["resolved"] == ["N0"]
    m3, _ = _apply(m, tree=EXPAND("year?", "2011", "profit 2011?", ["P2"]), view=View("P2", "search"), turn=2)
    m3, _ = _apply(m3, tree=RESOLVE("12.5", ["P1"]), view=View("P2", "search"), turn=3)             # from memory
    assert m3.node("N1").pages == ["P2"] and m3.node("N2").pages == ["P1"] and m3.active_id == "N0"


def test_fact_correction_keeps_the_conclusions():
    m = _mem("Q", pages=("p1",))
    m, _ = _apply(m, _edit(facts=[ADD("g", [0, 0, 10, 10])]), RESOLVE("final", ["P1"]), View("P1", "search"))
    m2, ev = _apply(m, _edit(facts=[UPD_FACT("F1", "g fixed", [0, 0, 12, 12])]), view=View("P1", "revisit"), turn=2)
    assert ev["updated_facts"] == ["P1.F1"] and m2.root.status == "resolved" and m2.root.answer == "final"
    f = m2.pages["P1"].facts[0]
    assert f.text == "g fixed" and f.revision == 1 and f.history[0]["text"] == "g"


def test_short_fact_ids_refer_to_the_page_in_view():
    m = _read(_read(_mem(pages=("p1", "p2")), "P1", "a"), "P2", "b", turn=2)
    m2, ev = _apply(m, _edit(facts=[UPD_FACT("F1", "b fixed", [0, 0, 12, 12])]), view=View("P2", "search"), turn=3)
    assert ev["updated_facts"] == ["P2.F1"] and m2.pages["P2"].facts[0].text == "b fixed" and m2.pages["P1"].facts[0].text == "a"
    _raises(lambda: _apply(m, _edit(facts=[UPD_FACT("F2", "x", [0, 0, 10, 10])]), view=View("P2", "search"), turn=3))


def test_finish_resolves_the_root():
    m = _read(_read(_mem("Q", pages=("p1", "p2")), "P1"), "P2")
    m, _ = _apply(m, tree=EXPAND("s1", "a1", "s2", ["P1"]), view=View("P1", "search"))
    m, _ = _apply(m, tree=RESOLVE("a2", ["P2"]), view=View("P2", "search"), turn=2)
    m.finish("B")
    assert (m.root.status, m.root.answer, m.root.pages) == ("resolved", "B", []) and m.evidence_pages() == ["P1", "P2"]


def test_render_tree_and_page_memory():
    m = _mem("Compare A and B?", pages=("deck/page_01", "deck/page_02"))
    m, _ = _apply(m, _edit("chart of A", [ADD("A=3", [1, 2, 30, 40]), ADD("unit: %", [5, 5, 60, 20])]),
                  EXPAND("What is A?", "3", "What is B?", ["P1"]), View("P1", "search"))
    tree = render_tree(m)
    assert tree == "\n".join([
        "Question tree:",
        "N0 [open]: Compare A and B?",
        "│",
        "├─ N1 [resolved]: What is A?",
        "│  answer: 3",
        "│  pages: [P1]",
        "│",
        "└─ N2 [open, ACTIVE]: What is B?",
    ]), tree
    pages = render_pages(m)
    assert pages == "\n".join([
        "Page memory:",
        'P1: chart of A   (found by: "query for deck/page_01")   -> answers N1',
        "├─ F1: A=3",
        "└─ F2: unit: %",
        "",
        'P2   (found by: "query for deck/page_02")',
    ]), pages
    m, _ = _apply(m, _edit(facts=[ADD("B=5", [1, 2, 30, 40])]), RESOLVE("5", ["P2"]), View("P2", "search"), turn=2)
    tree = render_tree(m)
    assert tree.splitlines()[1] == "N0 [open, ACTIVE]: Compare A and B?"
    assert tree.splitlines()[-3:] == ["└─ N2 [resolved]: What is B?", "   answer: 5", "   pages: [P2]"]
    assert 'P2   (found by: "query for deck/page_02")   -> answers N2\n└─ F1: B=5' in render_pages(m)
    for text in (tree, pages):
        assert "bbox" not in text and "[1, 2, 30, 40]" not in text and "deck/page_0" not in text.replace('"query for deck/page_01"', "").replace('"query for deck/page_02"', "")


def test_render_state_hints():
    m = _mem("Q")
    s = render_state(m, None, frame=F)
    assert "Page in view: none" in s and "<page>null</page>" in s and "\nPage memory:\nP1   (found by" in s and s.startswith("Question tree:")
    assert "This turn: no page in view" in s
    t = render_state(m, None, frame=F, turn_of=(3, 10))                                  # DESIGN.md §11.24
    assert "Turn 3 of 10: no page in view" in t and "This turn" not in t
    assert "Turn 10 of 10: page P1 is in view" in render_state(m, View("P1", "search"), frame=F, force=True, turn_of=(10, 10))
    s = render_state(m, View("P1", "search"), frame=F, note="hello")
    assert "Page in view: P1\n" in s and "ABSOLUTE PIXELS on the PAGE image (728 wide x 532 high)" in s
    assert "crop this page once" in s and "Note: hello" in s and "0-1000" not in s
    s3 = render_state(m, View("P1", "search"), frame=F3)
    assert "0-1000 RELATIVE coordinates on the PAGE image" in s3 and "wide x" not in s3 and "PIXELS" not in s3
    z = _zoom(m, "P1", [100, 100, 300, 200])
    s = render_state(m, View("P1", "search", zoom=z), frame=F, force=True)
    assert "Page in view: P1 + zoom" in s and "ABSOLUTE PIXELS on the ZOOM image (588 wide x 364 high)" in s
    assert "No further crop" in s and "Budget exhausted" in s and "must be answer" in s
    assert "0-1000 RELATIVE coordinates on the ZOOM image" in render_state(m, View("P1", "search", zoom=z), frame=F3)
