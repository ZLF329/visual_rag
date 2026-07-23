#!/usr/bin/env python3
"""AGv2 refactor smoke tests: parser matrix, graph transitions, crop semantics,
violation handling, and a scripted fake-VLM eval episode. Run from refactor root:
    python3 test_agv2.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("VRA_PATH", str(Path(__file__).parent)))

from PIL import Image

from src.protocol import (
    BoxFormatError,
    CropContext,
    ProtocolError,
    commit_crop_decision,
    commit_page_decision,
    crop_displayed_box,
    finalize_root,
    finish_crop_chain,
    open_nonroot_nodes,
    parse_turn,
    pending_hint,
)
from src.active_clue_graph import ClueGraph, is_sufficient
from src.schemas import GraphDecisionResult

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def expect_error(name, fn, exc_type):
    try:
        fn()
    except exc_type:
        check(name, True)
    except Exception as e:
        check(name, False, f"wrong error type {type(e).__name__}: {e}")
    else:
        check(name, False, "no error raised")


print("== parser: happy paths ==")
t = parse_turn("<think>go</think>\n<search>revenue 2019</search>", observation_pending=False)
check("first-turn search", t.action == "search" and t.action_payload == "revenue 2019" and t.update_payload is None)

t = parse_turn(
    '<think>page answers it</think>\n'
    '<update_graph>{"type":"accept","answer":"$4.2B","supporting_facts":["rev $4.2B"]}</update_graph>\n'
    '<answer>$4.2B</answer>',
    observation_pending=True,
)
check("commit+answer", t.action == "answer" and t.update_payload["type"] == "accept")

t = parse_turn(
    '<think>partial</think>\n'
    '<update_graph>{"type":"expand","answered_subquestion":"a?","answer":"x",'
    '"supporting_facts":["f"],"remaining_subquestion":"b?"}</update_graph>\n'
    '<search>next thing</search>',
    observation_pending=True,
)
check("commit expand+search", t.action == "search" and t.update_payload["type"] == "expand")

t = parse_turn(
    '<think>zoom</think>\n'
    '<update_graph>{"type":"accept","answer":"40%","supporting_facts":["about 40%"]}</update_graph>\n'
    '<bbox>[10, 20, 300, 400]</bbox>',
    observation_pending=True,
)
check("commit+bbox", t.action == "bbox" and t.box == [10.0, 20.0, 300.0, 400.0])

t = parse_turn(
    '<think>commit the last leaf; answer next turn with the full graph</think>\n'
    '<update_graph>{"type":"accept","answer":"$5.1B","supporting_facts":["2020 rev $5.1B"]}</update_graph>',
    observation_pending=True,
)
check("commit-only turn", t.action is None and t.update_payload["type"] == "accept")

print("== parser: violations ==")
expect_error("no think", lambda: parse_turn("<search>q</search>", observation_pending=False), ProtocolError)
expect_error("empty think", lambda: parse_turn("<think>  </think><search>q</search>", observation_pending=False), ProtocolError)
expect_error("missing commit when pending", lambda: parse_turn("<think>t</think><search>q</search>", observation_pending=True), ProtocolError)
expect_error("commit when not pending", lambda: parse_turn('<think>t</think><update_graph>{"type":"reject","summary":"s","reason":"r"}</update_graph><search>q</search>', observation_pending=False), ProtocolError)
expect_error("two actions", lambda: parse_turn("<think>t</think><search>a</search><answer>b</answer>", observation_pending=False), ProtocolError)
expect_error("no action", lambda: parse_turn("<think>t</think>", observation_pending=False), ProtocolError)
expect_error("commit-only with trailing text", lambda: parse_turn('<think>t</think><update_graph>{"type":"reject","summary":"s","reason":"r"}</update_graph>oops', observation_pending=True), ProtocolError)
expect_error("stray text", lambda: parse_turn("<think>t</think>hello<search>q</search>", observation_pending=False), ProtocolError)
expect_error("bad update json", lambda: parse_turn("<think>t</think><update_graph>{oops}</update_graph><search>q</search>", observation_pending=True), ProtocolError)
expect_error("empty search", lambda: parse_turn("<think>t</think><search></search>", observation_pending=False), ProtocolError)
expect_error("empty answer", lambda: parse_turn("<think>t</think><answer> </answer>", observation_pending=False), ProtocolError)
expect_error("bbox 3 numbers", lambda: parse_turn("<think>t</think><bbox>[1,2,3]</bbox>", observation_pending=False), BoxFormatError)
expect_error("bbox inverted", lambda: parse_turn("<think>t</think><bbox>[100,100,50,200]</bbox>", observation_pending=False), BoxFormatError)

print("== graph: single-hop accept + implicit root finalize ==")
g = ClueGraph.from_root_question("What is X?")
d = GraphDecisionResult.model_validate({"type": "accept", "answer": "42", "supporting_facts": ["X is 42"]})
dtype, facts_node, resume = commit_page_decision(g, d, "obs")
check("accept type", dtype == "accept" and facts_node == "N0" and resume is None)
check("root sufficient after accept", is_sufficient(g.root()))
check("facts plain (no bbox key)", g.root().known_facts and "bbox_2d" not in g.root().known_facts[0])
finalize_root(g, "42")
check("finalize keeps answer", g.root().answer == "42")

print("== graph: multi-hop expand -> accept leaf -> implicit finalize ==")
g = ClueGraph.from_root_question("Compare A and B?")
d = GraphDecisionResult.model_validate({
    "type": "expand", "answered_subquestion": "What is A?", "answer": "3",
    "supporting_facts": ["A=3"], "remaining_subquestion": "What is B?",
})
dtype, facts_node, _ = commit_page_decision(g, d, "obs")
answered_id = facts_node
check("expand facts node is answered child", g.nodes[answered_id].answer == "3")
check("active moved to remaining child", g.active().question == "What is B?")
check("open nonroot exists (gate would fire on premature answer)", bool(open_nonroot_nodes(g)))
d2 = GraphDecisionResult.model_validate({"type": "accept", "answer": "5", "supporting_facts": ["B=5"]})
dtype2, node2, _ = commit_page_decision(g, d2, "obs")
check("leaf accepted", dtype2 == "accept" and g.nodes[node2].answer == "5")
check("no open nonroot after final accept", not open_nonroot_nodes(g))
finalize_root(g, "B is larger")
check("root finalized via answer", is_sufficient(g.root()) and g.root().answer == "B is larger")

print("== graph: reject keeps node open ==")
g = ClueGraph.from_root_question("What is Y?")
d = GraphDecisionResult.model_validate({"type": "reject", "summary": "wrong page", "reason": "try other terms"})
dtype, facts_node, _ = commit_page_decision(g, d, "obs")
check("reject", dtype == "reject" and facts_node is None)
check("root still open", g.root().status == "open")

print("== crop pin/resume: expand + bbox keeps active on answered child ==")
g = ClueGraph.from_root_question("Compare A and B?")
d = GraphDecisionResult.model_validate({
    "type": "expand", "answered_subquestion": "What is A?", "answer": "3",
    "supporting_facts": ["A=3"], "remaining_subquestion": "What is B?",
})
dtype, facts_node, resume = commit_page_decision(g, d, "obs", defer_active_shift=True)
check("expand+bbox pins active on answered child", g.active_node_id == facts_node, g.active_node_id)
check("resume points at remaining child", g.nodes[resume].question == "What is B?")
ctx = CropContext(source_image="img", displayed_size=(10, 10), source_page_label="p",
                  target_node_id=facts_node, resume_active_node_id=resume)
finish_crop_chain(g, ctx)
check("chain end resumes remaining child", g.active().question == "What is B?")
check("ctx cleared", not ctx.ready and ctx.resume_active_node_id is None)

g2 = ClueGraph.from_root_question("Root has child?")
d = GraphDecisionResult.model_validate({
    "type": "expand", "answered_subquestion": "sub?", "answer": "x",
    "supporting_facts": ["f"], "remaining_subquestion": "rem?",
})
_, fn2, _ = commit_page_decision(g2, d, "obs")
d2 = GraphDecisionResult.model_validate({"type": "accept", "answer": "y", "supporting_facts": ["g"]})
dtype2, fn3, resume2 = commit_page_decision(g2, d2, "obs", defer_active_shift=True)
check("accept+bbox pins active on accepted node", g2.active_node_id == fn3)
check("accept resume -> parent", resume2 == g2.nodes[fn3].parent_id)

print("== crop semantics ==")
g = ClueGraph.from_root_question("What pct?")
d = GraphDecisionResult.model_validate({"type": "accept", "answer": "~40%", "supporting_facts": ["seg near 40%"]})
dtype, facts_node, _ = commit_page_decision(g, d, "obs")
nfacts_before = len(g.nodes[facts_node].known_facts)
dz = GraphDecisionResult.model_validate({"type": "accept", "answer": "42%", "supporting_facts": ["label reads 42%"]})
commit_crop_decision(g, facts_node, dz)
check("crop facts appended to same node", len(g.nodes[facts_node].known_facts) == nfacts_before + 1)
check("crop accept refreshes node answer", g.nodes[facts_node].answer == "42%")
dz2 = GraphDecisionResult.model_validate({"type": "reject", "summary": "zoom useless", "reason": "nothing new"})
r = commit_crop_decision(g, facts_node, dz2)
check("crop reject discards", r == "reject" and len(g.nodes[facts_node].known_facts) == nfacts_before + 1)
check("crop reject keeps node answer", g.nodes[facts_node].answer == "42%")

print("== crop box mapping ==")
raw = Image.new("RGB", (2000, 1000), "white")
crop = crop_displayed_box(raw, (1000, 500), [100, 100, 200, 200], pad=28, min_pixels=1000)
# displayed(1000x500) -> raw(2000x1000): box [200,200,400,400] +-28 => 256x256 region, 32-aligned resize
check("crop mapping size sane", abs(crop.size[0] - 256) <= 32 and abs(crop.size[1] - 256) <= 32, str(crop.size))
crop2 = crop_displayed_box(raw, (2000, 1000), [0, 0, 64, 64], pad=0, min_pixels=500_000)
check("min_pixels upscale", crop2.size[0] * crop2.size[1] >= 500_000, str(crop2.size))
expect_error("degenerate box after mapping", lambda: crop_displayed_box(raw, (1000, 500), [1999, 999, 2000, 1000], pad=0, min_pixels=1000), BoxFormatError)

print("== schemas: fact coercion ==")
d = GraphDecisionResult.model_validate({"type": "accept", "answer": "a", "supporting_facts": [{"fact": "f1", "bbox_2d": [1, 2, 3, 4]}, "f2"]})
check("dict+str facts coerce, bbox dropped", [f.fact for f in d.supporting_facts] == ["f1", "f2"])

print("== fake-VLM eval episode ==")
from src.agent import Agent


class FakeVLM:
    """Scripted multi-hop episode: search -> expand+search -> accept+bbox -> crop commit+answer."""
    def __init__(self):
        self.turns = [
            "<think>need A</think>\n<search>find A</search>",
            '<think>got A, need B</think>\n<update_graph>{"type":"expand","answered_subquestion":"What is A?","answer":"3","supporting_facts":["A=3"],"remaining_subquestion":"What is B?"}</update_graph>\n<search>find B</search>',
            '<think>B page, zoom to confirm</think>\n<update_graph>{"type":"accept","answer":"5","supporting_facts":["B=5"]}</update_graph>\n<bbox>[10,10,200,200]</bbox>',
            '<think>zoom confirms; finish</think>\n<update_graph>{"type":"accept","answer":"5 exactly","supporting_facts":["B=5.0 exact"]}</update_graph>\n<answer>B (5) is larger than A (3)</answer>',
        ]
        self.i = 0

    def generate_turn(self, *, messages, images, observation_pending):
        from src.protocol import parse_turn as pt
        text = self.turns[self.i]
        self.i += 1
        return text, pt(text, observation_pending=observation_pending)


class FakeRetriever:
    def __init__(self):
        self.calls = 0

    def search(self, query, top_k=1, deck_name=None):
        self.calls += 1
        return [(Image.new("RGB", (800, 600), "gray"), f"deck/page_{self.calls:02d}")]


agent = Agent(vlm=FakeVLM(), retriever=FakeRetriever(), top_k=1, max_iters=8)
result = agent.run("Compare A and B?", output_dir=None, deck_name=None)
check("episode answers", result["terminated_by"] == "answer", result["terminated_by"])
check("final answer text", result["answer"] == "B (5) is larger than A (3)", result["answer"])
gnodes = result["graph"]["nodes"]
check("graph has 3 nodes (root+2 children)", len(gnodes) == 3, str(len(gnodes)))
root = gnodes["N0"]
check("root sufficient+answered", root["status"] == "sufficient" and root["answer"].startswith("B (5)"))
crop_facts = [f for n in gnodes.values() for f in n["known_facts"] if "exact" in f.get("fact", "")]
check("crop fact landed in a node", len(crop_facts) == 1)
steps = [row.get("action") for row in result["trace"] if row.get("step") == "turn"]
check("trace actions", steps == ["search", "search", "bbox", "answer"], str(steps))
# active pinned during the crop-commit turn: the LAST turn (crop commit + answer) must show
# the accepted child (crop target) as active, not the root.
crop_commit_turn = [row for row in result["trace"] if row.get("step") == "turn"][-1]
tgt = next(row["commit"]["facts_node"] for row in result["trace"]
           if row.get("step") == "turn" and row.get("commit", {}).get("kind") == "page"
           and row["commit"]["type"] == "accept")
check("crop-commit turn active pinned on target", crop_commit_turn["active_node_id"] == tgt,
      f"{crop_commit_turn['active_node_id']} vs {tgt}")

print("== fake-VLM: zoom-reject then bbox retry ==")


class ZoomRetryVLM:
    """search -> accept+bbox -> zoom-reject + bbox RETRY -> zoom-accept + answer."""
    def __init__(self):
        self.turns = [
            "<think>find the chart</think>\n<search>market share chart</search>",
            '<think>chart page; value small, zoom</think>\n<update_graph>{"type":"accept","supporting_facts":["segment near 40%"],"answer":"~40%"}</update_graph>\n<bbox>[10,10,120,120]</bbox>',
            '<think>crop hit the wrong region; retry lower</think>\n<update_graph>{"type":"reject","summary":"crop shows the title only","reason":"zoom lower onto the segment label"}</update_graph>\n<bbox>[200,300,500,560]</bbox>',
            '<think>label legible now</think>\n<update_graph>{"type":"accept","supporting_facts":["segment label reads 42%"],"answer":"42%"}</update_graph>\n<answer>42%</answer>',
        ]
        self.i = 0

    def generate_turn(self, *, messages, images, observation_pending):
        from src.protocol import parse_turn as pt
        text = self.turns[self.i]
        self.i += 1
        return text, pt(text, observation_pending=observation_pending)


agent_zr = Agent(vlm=ZoomRetryVLM(), retriever=FakeRetriever(), top_k=1, max_iters=8)
res_zr = agent_zr.run("What pct?", output_dir=None, deck_name=None)
check("zoom-retry episode answers", res_zr["terminated_by"] == "answer", res_zr["terminated_by"])
zr_steps = [row.get("action") for row in res_zr["trace"] if row.get("step") == "turn"]
check("zoom-retry trace actions", zr_steps == ["search", "bbox", "bbox", "answer"], str(zr_steps))
check("zoom-retry no error steps", all(row.get("step") in {"turn"} for row in res_zr["trace"]),
      str([row.get("step") for row in res_zr["trace"]]))
zr_commits = [row["commit"] for row in res_zr["trace"]
              if row.get("step") == "turn" and row.get("commit")]
check("both zoom commits hit same target",
      len({c["target_node"] for c in zr_commits if c["kind"] == "crop"}) == 1, str(zr_commits))
zr_node = res_zr["graph"]["nodes"][zr_commits[-1]["target_node"]]
check("zoom accept refreshed node answer", zr_node["answer"] == "42%", zr_node["answer"])

print("== teacher norm1000 bbox frame conversion ==")
from src.agent import _rewrite_box_to_displayed_px


class NormFrameVLM:
    """Teacher emitting Qwen3-VL-style 0-1000 normalized boxes."""
    def __init__(self):
        self.turns = [
            "<think>find the chart</think>\n<search>market share chart</search>",
            '<think>value small, zoom</think>\n<update_graph>{"type":"accept","supporting_facts":["segment near 40%"],"answer":"~40%"}</update_graph>\n<bbox>[250,500,750,900]</bbox>',
            '<think>legible</think>\n<update_graph>{"type":"accept","supporting_facts":["label reads 42%"],"answer":"42%"}</update_graph>\n<answer>42%</answer>',
        ]
        self.i = 0

    def generate_turn(self, *, messages, images, observation_pending):
        from src.protocol import parse_turn as pt
        text = self.turns[self.i]
        self.i += 1
        return text, pt(text, observation_pending=observation_pending)


agent_nf = Agent(vlm=NormFrameVLM(), retriever=FakeRetriever(), top_k=1, max_iters=8,
                 bbox_frame="norm1000")
res_nf = agent_nf.run("What pct?", output_dir=None, deck_name=None)
check("norm1000 episode answers", res_nf["terminated_by"] == "answer", res_nf["terminated_by"])
nf_bbox_turn = next(r for r in res_nf["trace"] if r.get("step") == "turn" and r.get("action") == "bbox")
# retriever pages are 800x600 -> [250,500,750,900]/1000 * (800,600) = [200,300,600,540]
check("bbox rewritten to displayed px in target",
      "<bbox>[200,300,600,540]</bbox>" in nf_bbox_turn["sft"]["target"], nf_bbox_turn["sft"]["target"][-90:])
check("original norm payload gone", "[250,500,750,900]" not in nf_bbox_turn["sft"]["target"])


class _TurnStub:
    box = [1200.0, 50.0, 1400.0, 300.0]


stub = _TurnStub()
raw_px = "<think>t</think>\n<bbox>[1200,50,1400,300]</bbox>"
check("boxes already in px (>1000) untouched",
      _rewrite_box_to_displayed_px(raw_px, stub, (800, 600)) == raw_px and stub.box == [1200.0, 50.0, 1400.0, 300.0])


class _TurnStub2:
    box = [250.0, 500.0, 750.0, 900.0]


stub2 = _TurnStub2()
raw_decoy = ('<think>earlier I tried <bbox>[9,9,99,99]</bbox> and it was too dark</think>\n'
             '<update_graph>{"type":"reject","summary":"s","reason":"r"}</update_graph>\n'
             '<bbox>[250,500,750,900]</bbox>')
out_decoy = _rewrite_box_to_displayed_px(raw_decoy, stub2, (800, 600))
check("decoy bbox inside think untouched, action bbox rewritten",
      "<bbox>[9,9,99,99]</bbox>" in out_decoy and "<bbox>[200,300,600,540]</bbox>" in out_decoy
      and "[250,500,750,900]" not in out_decoy, out_decoy[-80:])

print("== fake-VLM: multi-hop with commit-only final ==")


class CommitOnlyVLM:
    """expand+search -> accept commit-only -> answer with full graph rendered."""
    def __init__(self):
        self.turns = [
            "<think>need A</think>\n<search>find A</search>",
            '<think>got A, need B</think>\n<update_graph>{"type":"expand","answered_subquestion":"What is A?","answer":"3","supporting_facts":["A=3"],"remaining_subquestion":"What is B?"}</update_graph>\n<search>find B</search>',
            '<think>last leaf; commit only, answer next turn</think>\n<update_graph>{"type":"accept","answer":"5","supporting_facts":["B=5"]}</update_graph>',
            '<think>graph shows A=3 and B=5; B larger.</think>\n<answer>B (5) is larger</answer>',
        ]
        self.i = 0
        self.answer_turn_messages = None

    def generate_turn(self, *, messages, images, observation_pending):
        from src.protocol import parse_turn as pt
        text = self.turns[self.i]
        if self.i == 3:
            self.answer_turn_messages = messages
        self.i += 1
        return text, pt(text, observation_pending=observation_pending)


vlmc = CommitOnlyVLM()
agent_c = Agent(vlm=vlmc, retriever=FakeRetriever(), top_k=1, max_iters=8)
rc = agent_c.run("Compare A and B?", output_dir=None)
check("commit-only episode answers", rc["terminated_by"] == "answer" and rc["answer"] == "B (5) is larger", rc["terminated_by"])
graph_block_at_answer = vlmc.answer_turn_messages[-1]["content"]
check("answer turn graph shows BOTH facts", "A=3" in graph_block_at_answer and "B=5" in graph_block_at_answer,
      graph_block_at_answer[-300:])
check("answer turn hint = no pending", "do NOT emit" in graph_block_at_answer)
prev_pair_user = vlmc.answer_turn_messages[-2]["content"]
check("answer turn sees commit echo", "Graph update:" in prev_pair_user, prev_pair_user[:80])

print("== fake-VLM: violation terminates ==")


class BadVLM:
    def generate_turn(self, *, messages, images, observation_pending):
        raise __import__("src.vlm", fromlist=["StructuredOutputError"]).StructuredOutputError(
            "ParsedTurn", ["<garbage>"], ProtocolError("missing tags"))


agent2 = Agent(vlm=BadVLM(), retriever=FakeRetriever(), top_k=1, max_iters=4)
r2 = agent2.run("q?", output_dir=None)
check("policy_error termination", r2["terminated_by"] == "policy_error", r2["terminated_by"])

print("== fake-VLM: bbox without target terminates ==")


class NoTargetVLM:
    def __init__(self):
        self.turns = [
            "<think>zoom immediately</think>\n<bbox>[1,1,50,50]</bbox>",
        ]
        self.i = 0

    def generate_turn(self, *, messages, images, observation_pending):
        from src.protocol import parse_turn as pt
        text = self.turns[self.i]
        self.i += 1
        return text, pt(text, observation_pending=observation_pending)


agent3 = Agent(vlm=NoTargetVLM(), retriever=FakeRetriever(), top_k=1, max_iters=4)
r3 = agent3.run("q?", output_dir=None)
check("bbox-no-target terminates", r3["terminated_by"] == "policy_error", r3["terminated_by"])

print("== pending hint ==")
check("hint pending", "MUST commit" in pending_hint(True))
check("hint not pending", "do NOT emit" in pending_hint(False))
h = pending_hint(True, crop_page_label="deck/page_14", crop_target_node="N2")
check("hint crop names target", "ZOOM" in h and "deck/page_14" in h and "[N2]" in h, h)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
