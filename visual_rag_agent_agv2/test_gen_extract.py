#!/usr/bin/env python3
"""Mechanical test of the AGv2 SFT extraction layer: run a scripted fake-VLM episode through
the REAL Agent (with persisted observation images), then verify
generate_active_graph_sft_trajectories.build_dynamic_chat_sft_calls / should_keep_trajectory.
Run: VRA_PATH=/root/autodl-tmp/visual_rag_agent python3 test_gen_extract.py
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

VRA = os.environ.get("VRA_PATH", "/root/autodl-tmp/visual_rag_agent")
sys.path.insert(0, VRA)

from PIL import Image
from src.agent import Agent
from src.protocol import parse_turn

spec = importlib.util.spec_from_file_location(
    "gen_sft", Path(VRA) / "scripts" / "generate_active_graph_sft_trajectories.py"
)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

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


class FakeVLM:
    """Multi-hop with a zoom: search -> expand+search -> accept+bbox -> crop commit+answer."""
    def __init__(self):
        self.turns = [
            "<think>need A</think>\n<search>find A</search>",
            '<think>got A, need B</think>\n<update_graph>{"type":"expand","answered_subquestion":"What is A?","supporting_facts":["A=3"],"answer":"3","remaining_subquestion":"What is B?"}</update_graph>\n<search>find B</search>',
            '<think>B page, zoom to confirm</think>\n<update_graph>{"type":"accept","supporting_facts":["B=5"],"answer":"5"}</update_graph>\n<bbox>[10,10,200,200]</bbox>',
            '<think>zoom confirms; graph has A=3 and B=5, B larger</think>\n<update_graph>{"type":"accept","supporting_facts":["B=5.0 exact"],"answer":"5 exactly"}</update_graph>\n<answer>B (5) is larger than A (3)</answer>',
        ]
        self.i = 0

    def generate_turn(self, *, messages, images, observation_pending):
        text = self.turns[self.i]
        self.i += 1
        return text, parse_turn(text, observation_pending=observation_pending)


class FakeRetriever:
    def __init__(self):
        self.calls = 0

    def search(self, query, top_k=1, deck_name=None):
        self.calls += 1
        return [(Image.new("RGB", (800, 600), "gray"), f"deck/page_{self.calls:02d}")]


tmp = tempfile.mkdtemp(prefix="agv2_gen_test_")
agent = Agent(vlm=FakeVLM(), retriever=FakeRetriever(), top_k=1, max_iters=8)
result = agent.run("Compare A and B?", output_dir=tmp, deck_name=None)
result["sample_id"] = "t1"
result["gold_answer"] = "B"
result["evidence_pages"] = [1, 2]
result["judge"] = {"correct": True, "score": 1}
result["retrieved_pages"] = sorted(gen.extract_retrieved_pages(result))

print("== extraction ==")
calls = gen.build_dynamic_chat_sft_calls(result, sample_id="t1")
check("4 sft calls (one per turn)", len(calls) == 4, str(len(calls)))
check("targets are verbatim responses", calls[1]["target"].startswith("<think>got A") and "expand" in calls[1]["target"])
check("actions recorded", [c["action"] for c in calls] == ["search", "search", "bbox", "answer"], str([c["action"] for c in calls]))
for i, c in enumerate(calls):
    n_markers = sum(str(m.get("content", "")).count("<image>") for m in c["messages"])
    check(f"call{i} image markers == image_paths ({n_markers})", n_markers == len(c["image_paths"]),
          f"{n_markers} vs {len(c['image_paths'])}")
    if c["image_paths"]:
        check(f"call{i} image files exist", all(Path(p).exists() for p in c["image_paths"]))
check("crop-commit turn context mentions ZOOM", any("ZOOM" in str(m.get("content", "")) for m in calls[3]["messages"]))
check("conversations format", calls[0]["conversations"][-1]["from"] == "gpt" and calls[0]["conversations"][-1]["value"] == calls[0]["target"])

print("== keep/reject ==")
keep, reasons = gen.should_keep_trajectory(result, require_judge_correct=True, require_all_reference_pages=True)
check("clean episode kept", keep, str(reasons))

bad = dict(result)
bad["trace"] = list(result["trace"]) + [{"step": "box_error", "error": "x"}]
keep2, reasons2 = gen.should_keep_trajectory(bad, require_judge_correct=True, require_all_reference_pages=True)
check("error-step episode rejected", not keep2 and any("error_steps" in r for r in reasons2), str(reasons2))

bad3 = dict(result)
bad3["terminated_by"] = "max_iters"
keep3, reasons3 = gen.should_keep_trajectory(bad3, require_judge_correct=True, require_all_reference_pages=True)
check("non-answer termination rejected", not keep3 and any("terminated_by" in r for r in reasons3), str(reasons3))

bad4 = dict(result)
bad4["retrieved_pages"] = [1]
keep4, reasons4 = gen.should_keep_trajectory(bad4, require_judge_correct=True, require_all_reference_pages=True)
check("missing reference page rejected", not keep4 and any("missing_reference_pages" in r for r in reasons4), str(reasons4))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
