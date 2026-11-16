from PIL import Image

from src.protocol import (
    IMAGE_MAX_PIXELS,
    IMAGE_MIN_PIXELS,
    BoxError,
    ProtocolError,
    crop_region,
    get_frame,
    parse_turn,
    render_turn,
    unit_to_region,
)
from tests.fakes import ADD, ANSWER, CROP, EXPAND, RESOLVE, REVISIT, SEARCH, UPD_FACT, act, blocks, page


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return
    except Exception as other:  # noqa: BLE001
        raise AssertionError(f"wrong error type {type(other).__name__}: {other}")
    raise AssertionError(f"expected {exc.__name__}")


# ------------------------------------------------------------------ happy paths

def test_first_turn_search():
    t = parse_turn(act(None, SEARCH("revenue 2019")), view="none")
    assert t.act.action.type == "search" and t.act.page is None and t.act.tree is None


def test_page_turn_full_edit():
    t = parse_turn(act(page("growth chart", [ADD("A=3", [1, 2, 300, 400])]), SEARCH("B"),
                       tree=EXPAND("What is A?", "3", "What is B?", ["p1"])), view="page", current_page="P1")
    p, tree = t.act.page, t.act.tree
    assert p.desc == "growth chart" and p.facts[0].bbox == [1.0, 2.0, 300.0, 400.0]
    assert tree.op == "expand" and tree.answered == "What is A?" and tree.next == "What is B?" and tree.pages == ["P1"]


def test_resolve_with_and_without_node_id():
    t = parse_turn(act(None, ANSWER("42"), tree=RESOLVE("42", ["P1"])), view="page", current_page="P1")
    assert t.act.tree.op == "resolve" and t.act.tree.node_id is None and t.act.action.text == "42"
    t = parse_turn(act(None, SEARCH("q"), tree=RESOLVE("2011", ["P1"], node_id="n1")), view="page")
    assert t.act.tree.node_id == "N1"                                                     # ids are upper-cased


def test_tree_pages_are_required_ids():
    """DESIGN.md §11.23: a conclusion lists the pages it rests on (replaces the implicit page in view + "uses")."""
    t = parse_turn(act(None, ANSWER("12.5"), tree=RESOLVE("12.5", ["p1", "P2", "p1"])), view="page", current_page="P2")
    assert t.act.tree.pages == ["P1", "P2"]                                                # upper-cased, deduplicated
    assert t.act.to_payload()["tree"] == {"op": "resolve", "answer": "12.5", "pages": ["P1", "P2"]}
    t = parse_turn(act(None, SEARCH("q"), tree=EXPAND("a?", "b", "c?", ["P1", "P3"])), view="page")
    assert t.act.tree.pages == ["P1", "P3"]
    for bad in ([], "P1", None, [""]):
        _raises(lambda b=bad: parse_turn(act(None, ANSWER("x"), tree=RESOLVE("x", b)), view="page"), ProtocolError)


def test_tree_edit_needs_no_page_in_view():
    """The tree concludes from page memory; with no page in view only "page" must be null."""
    t = parse_turn(act(None, ANSWER("x"), tree=RESOLVE("x", ["P1"])), view="none")
    assert t.act.tree.pages == ["P1"] and t.act.page is None


def test_crop_by_box_or_fact():
    assert parse_turn(act(None, CROP(bbox=[10, 20, 300, 400])), view="page").act.action.bbox == [10.0, 20.0, 300.0, 400.0]
    assert parse_turn(act(None, CROP(fact_id="p1.f2")), view="page").act.action.fact_id == "P1.F2"


def test_zoom_turn_update_fact_and_answer():
    t = parse_turn(act(page(facts=[UPD_FACT("P1.F1", "x = 42", [0, 0, 50, 50])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])),
                   view="zoom", current_page="P1")
    assert t.act.page.facts[0].fact_id == "P1.F1" and t.act.tree.answer == "42"


def test_revisit_and_answer_without_page():
    assert parse_turn(act(None, REVISIT("p2")), view="none").act.action.page_id == "P2"
    assert parse_turn(act(None, REVISIT("P2")), view="zoom", current_page="P1").act.action.page_id == "P2"
    assert parse_turn(act(None, ANSWER("a")), view="none").act.action.text == "a"


def test_missing_page_and_tree_blocks_are_null():
    """DESIGN.md §11.24: <page> and <tree> may be left out or written null; an empty page object is an empty edit."""
    t = parse_turn(blocks(act=SEARCH("q")), view="none")
    assert t.act.page is None and t.act.tree is None and t.act.action.query == "q"
    t = parse_turn(blocks(page={}, act=SEARCH("q")), view="page")
    assert t.act.page.desc is None and t.act.page.facts == [] and t.act.tree is None
    t = parse_turn(blocks(page="null", tree="null", act=SEARCH("q")), view="page")
    assert t.act.page is None and t.act.tree is None


def test_a_slip_in_one_block_keeps_the_others():
    """DESIGN.md §11.24: each block is its own JSON value, decoded up to its end. The §11.23 slip (one brace too many
    after the facts) now only closes the page value; the tree and the action sit in their own blocks."""
    slip = '{"desc": "d", "facts": [{"op": "add_fact", "text": "x", "bbox": [1, 2, 30, 40]}]}}'
    t = parse_turn(blocks(page=slip, tree=RESOLVE("a", ["P1"]), act=ANSWER("a")), view="page", current_page="P1")
    assert (t.act.page.facts[0].text, t.act.tree.answer, t.act.action.text) == ("x", "a", "a")
    t = parse_turn(blocks(page="null", act='{"type": "answer", "text": "x"}}'), view="none")
    assert t.act.action.text == "x"


# ------------------------------------------------------------------ hard errors

def test_structure_errors():
    _raises(lambda: parse_turn('<think>t</think>', view="none"), ProtocolError)                  # no act block
    _raises(lambda: parse_turn(blocks(page="null", tree="null"), view="none"), ProtocolError)    # still no act block
    _raises(lambda: parse_turn(blocks(act="{oops}"), view="none"), ProtocolError)
    _raises(lambda: parse_turn(blocks(act='{"type": "answer", "text": "x"'), view="none"), ProtocolError)  # truncated
    _raises(lambda: parse_turn(blocks(act="[1, 2]"), view="none"), ProtocolError)              # not an object
    _raises(lambda: parse_turn(blocks(act="null"), view="none"), ProtocolError)                # the action is required
    _raises(lambda: parse_turn(blocks(act={}), view="none"), ProtocolError)                    # schema
    _raises(lambda: parse_turn(blocks(page="[1]", act=SEARCH("q")), view="page"), ProtocolError)    # object or null
    _raises(lambda: parse_turn(blocks(tree='"N1"', act=SEARCH("q")), view="page"), ProtocolError)   # object or null
    _raises(lambda: parse_turn(blocks(page="{oops}", act=SEARCH("q")), view="page"), ProtocolError)
    unclosed = '<think>t</think>\n<page>{"desc": "d", "facts": []}\n<tree>null</tree>\n<act>{"type": "search", "query": "q"}</act>'
    _raises(lambda: parse_turn(unclosed, view="page"), ProtocolError)                          # never silently dropped
    one_object = '<think>t</think><act>{"page": null, "tree": null, "action": {"type": "answer", "text": "x"}}</act>'
    _raises(lambda: parse_turn(one_object, view="none"), ProtocolError)                        # the §11.23 format


def test_errors_name_the_block():
    cases = ((dict(page={"facts": [ADD("f", [1, 2, 3])]}, act=SEARCH("q")), "<page>"),
             (dict(tree={"op": "resolve", "answer": "a"}, act=SEARCH("q")), "<tree>"),
             (dict(act={"type": "zoom"}), "<act>"),
             (dict(page="{oops}", act=SEARCH("q")), "<page>"))
    for parts, tag in cases:
        try:
            parse_turn(blocks(**parts), view="page", current_page="P1")
        except ProtocolError as exc:
            assert str(exc).startswith(tag), (tag, str(exc))
        else:
            raise AssertionError(parts)


def test_lenient_parsing_accepts_stray_tags_and_trailing_junk():
    """DESIGN.md §11.11: all think blocks merged, the last block of each kind, JSON decoded up to its end, stray tags
    and text outside the blocks ignored."""
    ans = blocks(think=None, page="null", tree="null", act=ANSWER("x"))
    t = parse_turn("<think>a</think>\n" + ans + "\n</think>", view="none")                   # stray closing tag
    assert (t.think, t.act.action.text) == ("a", "x")
    t = parse_turn("<think>a</think>\n<think>b</think>\n" + ans, view="none")                 # two think blocks
    assert t.think == "a\nb"
    assert parse_turn("<think></think>\n\n" + ans, view="none").think == ""                   # empty think allowed
    assert parse_turn("<think></think>\n\n</think>\n" + ans, view="none").think == ""
    assert parse_turn("<think></think>\n<think>real</think>\n" + ans, view="none").think == "real"
    t = parse_turn("<think>a</think>" + act(None, SEARCH("first")) + act(None, SEARCH("last")), view="none")
    assert t.act.action.query == "last"                                                          # last block wins
    assert parse_turn(act() + " trailing text", view="none").act.action.type == "search"
    assert parse_turn(ans, view="none").think == ""                                              # no think at all
    quoted = '<think>I write <page>{"desc": "no"}</page> and <act>{}</act> here</think>'
    t = parse_turn(quoted + blocks(think=None, act=ANSWER("y")), view="none")
    assert t.act.action.text == "y" and t.act.page is None                                       # blocks in a think are not read


def test_answer_only_rules():
    """DESIGN.md §11.11 / §11.26: on the answer-only turn only <act> counts and it must be an answer; a <page> or
    <tree> block written anyway (even unclosed or malformed) is dropped, so a slip there never loses the answer."""
    ans = act(None, ANSWER("x"))
    assert parse_turn(ans, view="page", answer_only=True).act.action.text == "x"
    t = parse_turn(act(page("p"), ANSWER("x"), tree=RESOLVE("x", ["P1"])), view="page", answer_only=True)
    assert t.act.page is None and t.act.tree is None and t.act.action.text == "x"
    t = parse_turn('<think>t</think>\n<page>{"desc": "p"\n<tree>{"op": "resolve", "pages": [P9]}</tree>\n'
                   '<act>{"type": "answer", "text": "x"}</act>', view="page", answer_only=True)
    assert t.act.page is None and t.act.tree is None and t.act.action.text == "x" and t.think == "t"
    _raises(lambda: parse_turn(act(None, SEARCH("q")), view="page", answer_only=True), ProtocolError)
    _raises(lambda: parse_turn(act(page("p"), ANSWER("x"), tree=RESOLVE("x", ["P1"])), view="none"), ProtocolError)


def test_schema_errors():
    bad = [
        dict(act={"type": "zoom"}),                                                       # unknown action
        dict(act={"type": "validate"}),                                                   # no validate action
        dict(act={"type": "search", "query": "q", "extra": 1}),                           # extra key
        dict(act={"memory_edit": {}, "action": SEARCH("q")}),                             # an old format
        dict(page={"task_ops": []}, act=SEARCH("q")),
        dict(page={"update": RESOLVE("a", ["P1"])}, act=SEARCH("q")),                     # the §11.8 page-centric update
        dict(page={"desc": "d", "tree": RESOLVE("a", ["P1"])}, act=SEARCH("q")),          # a tree edit goes in <tree>
        dict(page={"facts": [{"op": "delete_fact", "fact_id": "P1.F1"}]}, act=SEARCH("q")),
        dict(page={"facts": [ADD("f", [1, 2, 3])]}, act=SEARCH("q")),                     # 3 numbers
        dict(page={"facts": [ADD("", [1, 2, 3, 4])]}, act=SEARCH("q")),                   # empty text
        dict(tree={"op": "expand", "answered": "a", "answer": "b", "pages": ["P1"]}, act=SEARCH("q")),   # no next
        dict(tree={"op": "resolve", "pages": ["P1"]}, act=SEARCH("q")),                   # no answer
        dict(tree={"op": "resolve", "answer": "a"}, act=SEARCH("q")),                     # no pages
        dict(tree={"op": "resolve", "answer": "a", "pages": ["P1"], "uses": ["P2"]}, act=SEARCH("q")),  # no uses any more
        dict(tree={"op": "expand", "answered": "a", "answer": "b", "next": "c", "pages": ["P1"], "node_id": "N1"},
             act=SEARCH("q")),                                                            # expand works on the active question
        dict(tree={"op": "update", "node_id": "N1"}, act=SEARCH("q")),                    # no update op
        dict(tree=[RESOLVE("a", ["P1"])], act=SEARCH("q")),                               # one tree edit, not a list
        dict(act={"type": "crop"}),                                                       # crop without target
        dict(act={"type": "crop", "bbox": [1, 2, 3, 4], "fact_id": "P1.F1"}),
        dict(act={"type": "crop", "bbox": [1, 2, 3, float("nan")]}),
        dict(act={"type": "answer", "text": " "}),
        dict(act={"type": "answer", "text": "a", "support_nodes": ["N0"]}),               # no support_nodes any more
    ]
    for parts in bad:
        _raises(lambda p=parts: parse_turn(blocks(**p), view="page", current_page="P1"), ProtocolError)


def test_rule_errors():
    _raises(lambda: parse_turn(act(page("desc")), view="none"), ProtocolError)              # page edit without a page
    _raises(lambda: parse_turn(act(page(facts=[ADD("f", [0, 0, 10, 10])])), view="none"), ProtocolError)
    assert parse_turn(act(page()), view="none").act.page.is_empty                           # an empty page object is fine
    assert parse_turn(act(None, SEARCH("q"), tree=RESOLVE("a", ["P1"])), view="none").act.tree.answer == "a"
    _raises(lambda: parse_turn(act(None, CROP(bbox=[0, 0, 10, 10])), view="none"), ProtocolError)
    _raises(lambda: parse_turn(act(None, CROP(bbox=[0, 0, 10, 10])), view="zoom", current_page="P1"), ProtocolError)
    _raises(lambda: parse_turn(act(None, REVISIT("p1")), view="page", current_page="P1"), ProtocolError)
    _raises(lambda: parse_turn(act(None, REVISIT("p1")), view="zoom", current_page="P1"), ProtocolError)  # zoomed or not
    _raises(lambda: parse_turn(act(None, SEARCH("q")), view="page", force_answer=True), ProtocolError)
    assert parse_turn(act(None, ANSWER("a")), view="page", force_answer=True)


# ------------------------------------------------------------------ canonical form

def test_canonical_render():
    raw = '<think>I mention <act> here</think>\n<act>{ "type":"crop","fact_id":"p1.f1" }</act>'
    t = parse_turn(raw, view="page")
    assert t.act.to_payload() == {"page": None, "tree": None, "action": {"type": "crop", "fact_id": "P1.F1"}}
    out = render_turn(t.think, t.act)
    assert out == ('<think>I mention <act> here</think>\n<page>null</page>\n<tree>null</tree>\n'
                   '<act>{"type":"crop","fact_id":"P1.F1"}</act>')                                 # all four blocks, in order
    again = parse_turn(out, view="page")
    assert (again.think, again.act.to_payload()) == (t.think, t.act.to_payload())                  # round trip
    full = parse_turn(blocks(page={}, tree=RESOLVE("a", ["P1"]), act=ANSWER("a")), view="page").act
    assert full.to_payload() == {"page": {"desc": None, "facts": []}, "tree": {"op": "resolve", "answer": "a", "pages": ["P1"]},
                                 "action": {"type": "answer", "text": "a"}}                     # absent node_id not written
    assert full.to_text() == ('<page>{"desc":null,"facts":[]}</page>\n<tree>{"op":"resolve","answer":"a","pages":["P1"]}'
                              '</tree>\n<act>{"type":"answer","text":"a"}</act>')
    with_id = parse_turn(act(None, tree=RESOLVE("a", ["P1"], node_id="n2")), view="page").act
    assert '<tree>{"op":"resolve","answer":"a","node_id":"N2","pages":["P1"]}</tree>' in with_id.to_text()
    expand = parse_turn(act(None, tree=EXPAND("a?", "b", "c?", ["P1"])), view="page").act
    assert '<tree>{"op":"expand","answered":"a?","answer":"b","next":"c?","pages":["P1"]}</tree>' in expand.to_text()
    boxes = parse_turn(act(page(facts=[ADD("x", [10, 20.5, 30, 40])]), CROP(bbox=[1, 2, 300, 400])), view="page").act
    assert '"bbox":[10,20.5,30,40]' in boxes.to_text() and '"bbox":[1,2,300,400]' in boxes.to_text()   # no 1.0 floats


QWEN25, QWEN3 = get_frame("qwen2.5-vl"), get_frame("qwen3")


# ------------------------------------------------------------------ frames & geometry

def test_frame_shown_size_follows_score_budget():
    # every shown image has 256*28*28..512*28*28 pixels, sides on the model's patch grid
    assert QWEN25.shown_size((1024, 768)) == (728, 532) and QWEN25.shown_size((800, 600)) == (728, 532)
    assert QWEN3.shown_size((800, 600)) == (704, 544)
    assert QWEN25.shown_size((300, 120)) == (728, 308)                     # small crops are upscaled
    assert QWEN25.shown_size((728, 532)) == (728, 532)                     # already on grid: unchanged
    for frame in (QWEN25, QWEN3):
        for size in [(1024, 768), (300, 120), (4000, 3000), (800, 600), (1000, 250)]:
            w, h = frame.shown_size(size)
            assert w % frame.factor == 0 and h % frame.factor == 0
            assert IMAGE_MIN_PIXELS <= w * h <= IMAGE_MAX_PIXELS, (frame.name, size, w, h)
    assert QWEN25.show(Image.new("RGB", (1024, 768))).size == (728, 532)


def test_frame_image_budget_can_be_lowered():
    """DESIGN.md §11.25: a smaller image budget for the running model only (the teacher sees ~0.1 MP pages)."""
    budget = 128 * 28 * 28
    low = QWEN3.with_max_pixels(budget)
    assert (low.name, low.coord, low.factor, low.max_pixels) == ("qwen3", "norm1000", 32, budget) and low != QWEN3
    for size in [(1024, 768), (800, 600), (300, 120), (4000, 3000)]:
        w, h = low.shown_size(size)
        assert w % 32 == 0 and h % 32 == 0 and w * h <= 1.2 * budget, (size, w, h)
    assert low.shown_size((800, 600)) == (352, 256) and QWEN3.shown_size((800, 600)) == (704, 544)   # default unchanged
    assert low.to_unit([0, 0, 500, 500], low.shown_size((800, 600))) == [0.0, 0.0, 0.5, 0.5]       # 0-1000: size-free
    assert low.show(Image.new("RGB", (800, 600))).size == (352, 256)


def test_frame_unit_conversion_and_errors():
    shown = (728, 532)
    assert QWEN25.to_unit([-10, 0, 364, 532], shown) == [0.0, 0.0, 0.5, 1.0]
    assert QWEN3.to_unit([0, 0, 500, 1000], shown) == [0.0, 0.0, 0.5, 1.0]
    assert QWEN3.to_unit([0, 0, 1200, 1000], shown) == [0.0, 0.0, 1.0, 1.0]
    assert QWEN25.from_unit([0.0, 0.0, 0.5, 1.0], shown) == [0, 0, 364, 532]
    assert QWEN3.from_unit([0.25, 0.5, 0.5, 1.0], shown) == [250, 500, 500, 1000]
    _raises(lambda: QWEN25.to_unit([100, 100, 50, 200], shown), BoxError)        # inverted
    _raises(lambda: QWEN25.to_unit([800, 0, 900, 10], shown), BoxError)          # off the image
    _raises(lambda: QWEN25.to_unit([10, 10, 13, 50], shown), BoxError)           # < 4 shown px wide
    _raises(lambda: QWEN3.to_unit([10, 10, 15, 50], shown), BoxError)            # 5/1000*728 < 4 px
    assert QWEN3.to_unit([10, 10, 16, 50], shown)[2] == 0.016


def test_crop_region_pads_original_page_pixels():
    # SCORE App. B: box mapped to the original page, padded by 28 px, clamped to the page
    assert crop_region([100, 100, 200, 200], (1000, 500)) == (72, 72, 228, 228)
    assert crop_region([0.5, 480.2, 50, 500], (1000, 500)) == (0, 452, 78, 500)
    _raises(lambda: crop_region([0, 0, 8000, 20], (8000, 20)), BoxError)         # extreme aspect ratio
    assert unit_to_region([0, 0, 1, 1], (72, 72, 228, 228)) == [72.0, 72.0, 228.0, 228.0]
    assert unit_to_region([0.5, 0.25, 1, 1], (0, 0, 1000, 500)) == [500.0, 125.0, 1000.0, 500.0]
