#!/usr/bin/env python3
"""AGv2 RL-env smoke tests: drive SlideVQAMultiProcessEnv._step_active_graph directly with
scripted actions and a fake retriever (agent_system stubbed). Run from refactor root:
    python3 test_agv2_env.py
"""
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, os.environ.get("VRA_PATH", str(ROOT)))              # src/ (visual_rag_agent)
sys.path.insert(0, os.environ.get("ENVS_DIR", str(ROOT / "rl")))       # envs.py

# ---- stub the verl-agent package bits envs.py imports -----------------------------------
agent_system = types.ModuleType("agent_system")
environments = types.ModuleType("agent_system.environments")
prompts_mod = types.ModuleType("agent_system.environments.prompts")
prompts_mod.SLIDEVQA_ACTIVE_GRAPH_TEMPLATE = (
    "Root question: {question}\nActive: {active_question}\n{graph_state}\n{query_history}\n{valid_actions}"
)
prompts_mod.SLIDEVQA_TEMPLATE = "Q: {question}\nMemory: {memory}"
sys.modules["agent_system"] = agent_system
sys.modules["agent_system.environments"] = environments
sys.modules["agent_system.environments.prompts"] = prompts_mod

from PIL import Image
import envs as E

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


class FakeRetriever:
    """Deterministic: query 'find gold' -> gold page rank-1; anything else -> filler pages."""
    def __init__(self):
        self.n = 0

    def search(self, query, top_k=1, deck_name=None):
        self.n += 1
        if "second" in query:
            pages = [f"deck/page_02", f"deck/page_78"]
        elif "gold" in query:
            pages = [f"deck/page_01", f"deck/page_77"]
        else:
            pages = [f"deck/page_{50 + self.n:02d}", f"deck/page_{60 + self.n:02d}"]
        return [(Image.new("RGB", (800, 600), "gray"), p) for p in pages[:max(2, top_k)]]


def make_env(max_steps=8):
    cfg = {
        "max_steps": max_steps,
        "slidevqa": {
            "mode": "active_graph",
            "observation_mode": "multi_turn",
            "top_k": 1,
        },
    }
    env = E.SlideVQAMultiProcessEnv(seed=0, env_num=1, group_n=1, env_config=cfg)
    env.retriever = FakeRetriever()
    env._ensure_retriever = lambda: None  # keep the fake
    # no-op judge: correct iff exact match (avoid DeepSeek in tests)
    env.reward_fn = lambda **kw: 1.0 if kw["solution_str"].strip() == kw["ground_truth"].strip() else 0.0
    env.support_fn = None
    return env


def reset_one(env, question="What is X?", answer="42", refs=("deck/page_01",)):
    obs, infos = env.reset([{
        "question": question, "answer": answer,
        "deck_name": "deck", "reference_pages": list(refs),
        "sample_id": "t1",
    }])
    return obs[0], infos[0]


UP_ACCEPT = '{"type":"accept","answer":"42","supporting_facts":["X is 42"]}'
UP_REJECT = '{"type":"reject","summary":"wrong page","reason":"different terms"}'
UP_EXPAND = ('{"type":"expand","answered_subquestion":"What is A?","answer":"3",'
             '"supporting_facts":["A=3"],"remaining_subquestion":"What is B?"}')


print("== reset ==")
env = make_env()
obs, info = reset_one(env)
check("reset hint no-pending", "do NOT emit" in obs, obs[-120:])
check("no format error at reset", not info.get("format_error"))

print("== single-hop: search -> commit accept + answer ==")
env = make_env()
reset_one(env)
o, r, d, i = env.step(["<think>go</think><search>find gold page</search>"])
check("search not done", not d[0])
check("search step reward=1 (rank1 new gt)", i[0]["ra_search_reward"] == 1.0)
check("obs pending hint", "MUST commit" in o[0], o[0][-150:])
o, r, d, i = env.step([f'<think>page answers</think><update_graph>{UP_ACCEPT}</update_graph><answer>42</answer>'])
check("answer done", d[0])
check("episode reward = 1(ans) + 0.5*hit", abs(r[0] - 1.5) < 1e-6, str(r[0]))
check("commit type-match reward", i[0]["ra_step_reward"] == 1.0 and i[0]["ra_step_type_match"])
check("won flag", i[0]["won"])

print("== reject path ==")
env = make_env()
reset_one(env)
env.step(["<think>go</think><search>irrelevant stuff</search>"])
o, r, d, i = env.step([f'<think>useless</think><update_graph>{UP_REJECT}</update_graph><search>find gold page</search>'])
check("reject+search continues", not d[0])
check("reject on non-gt type-match=1", i[0]["ra_analyse_reward"] == 1.0, str(i[0].get("ra_analyse_reward")))
o, r, d, i = env.step([f'<think>gold</think><update_graph>{UP_ACCEPT}</update_graph><answer>42</answer>'])
check("finish after reject path", d[0] and r[0] >= 1.0, str(r[0]))

print("== crop flow: accept + bbox -> crop commit + answer ==")
env = make_env()
reset_one(env)
env.step(["<think>go</think><search>find gold page</search>"])
o, r, d, i = env.step([f'<think>zoom</think><update_graph>{UP_ACCEPT}</update_graph><bbox>[100,100,400,300]</bbox>'])
check("bbox continues", not d[0])
check("bbox step kind", i[0]["ra_step_kind"] in ("bbox",), i[0]["ra_step_kind"])
check("crop obs hint names ZOOM + target node", "ZOOM" in o[0] and "[N0]" in o[0], o[0][-260:])
o, r, d, i = env.step(['<think>crop shows detail</think><update_graph>{"type":"accept","answer":"42.0","supporting_facts":["exactly 42.0"]}</update_graph><answer>42</answer>'])
check("crop commit + answer done", d[0] and r[0] >= 1.0, str(r[0]))
check("crop commit neutral step", i[0]["ra_step_kind"] == "crop_commit" and i[0]["ra_step_reward"] == 0.0)

print("== crop retry: zoom-reject then bbox again ==")
env = make_env()
reset_one(env)
env.step(["<think>go</think><search>find gold page</search>"])
env.step([f'<think>zoom</think><update_graph>{UP_ACCEPT}</update_graph><bbox>[100,100,400,300]</bbox>'])
o, r, d, i = env.step(['<think>bad crop, retry lower</think><update_graph>{"type":"reject","summary":"crop off-target","reason":"zoom lower"}</update_graph><bbox>[120,150,420,360]</bbox>'])
check("zoom-reject + bbox retry legal (no format_error)", (not d[0]) and not i[0].get("format_error"), str(i[0].get("ra_step_kind")))
check("retry step kind bbox", i[0]["ra_step_kind"] == "bbox", i[0]["ra_step_kind"])
check("crop ctx retained on retry", env.states[0].crop_target_node_id is not None)
o, r, d, i = env.step(['<think>legible now</think><update_graph>{"type":"accept","answer":"42.0","supporting_facts":["exactly 42.0"]}</update_graph><answer>42</answer>'])
check("retry chain completes", d[0] and r[0] >= 1.0, str(r[0]))

print("== crop pin/resume: expand + bbox keeps active question on answered part ==")
env = make_env()
reset_one(env, question="Compare A and B?", answer="B", refs=("deck/page_01", "deck/page_02"))
env.step(["<think>go</think><search>find gold page</search>"])
o, r, d, i = env.step([f'<think>zoom A</think><update_graph>{UP_EXPAND}</update_graph><bbox>[100,100,400,300]</bbox>'])
check("expand+bbox continues", not d[0])
check("active question pinned on ANSWERED subq during crop", "What is A?" in o[0], o[0][-400:])
o, r, d, i = env.step(['<think>zoom adds detail</think><update_graph>{"type":"accept","answer":"3.0","supporting_facts":["A=3.0 exact"]}</update_graph><search>find second gold</search>'])
check("crop commit + search continues", not d[0])
check("active RESUMED to remaining subq", "What is B?" in o[0], o[0][-400:])
env2_state = env.states[0]
check("crop ctx cleared after resume", env2_state.crop_target_node_id is None)
o, r, d, i = env.step(['<think>final</think><update_graph>{"type":"accept","answer":"5","supporting_facts":["B=5"]}</update_graph><answer>B</answer>'])
check("multi-hop with mid-chain crop completes", d[0] and r[0] >= 1.0, str(r[0]))

print("== violations ==")
env = make_env()
reset_one(env)
o, r, d, i = env.step(["<think>t</think><bbox>[1,1,50,50]</bbox>"])
check("bbox without target -> format_error done", d[0] and r[0] == -1.0 and i[0].get("format_error"))

env = make_env()
reset_one(env)
env.step(["<think>go</think><search>find gold page</search>"])
o, r, d, i = env.step(["<think>skip commit</think><search>another query</search>"])
check("missing commit -> format_error", d[0] and r[0] == -1.0 and i[0].get("format_error"))

env = make_env()
reset_one(env)
o, r, d, i = env.step([f'<think>t</think><update_graph>{UP_ACCEPT}</update_graph><search>q</search>'])
check("commit without obs -> format_error", d[0] and r[0] == -1.0)

env = make_env()
reset_one(env)
env.step(["<think>go</think><search>find gold page</search>"])
o, r, d, i = env.step(['<think>bad</think><update_graph>{"type":"accept"}</update_graph><answer>42</answer>'])
check("accept-without-answer payload -> format_error", d[0] and r[0] == -1.0, f"r={r[0]}")

env = make_env()
reset_one(env)
env.step(["<think>go</think><search>find gold page</search>"])
o, r, d, i = env.step([f'<think>zoom</think><update_graph>{UP_ACCEPT}</update_graph><bbox>[9,9,8,20]</bbox>'])
check("inverted bbox payload -> soft step -1, not done", (not d[0]) and i[0]["ra_step_reward"] == -1.0 and i[0]["ra_step_kind"] == "box_error")

print("== multi-hop + gate ==")
env = make_env()
reset_one(env, question="Compare A and B?", answer="B", refs=("deck/page_01", "deck/page_02"))
env.step(["<think>go</think><search>find gold page</search>"])
o, r, d, i = env.step([f'<think>partial</think><update_graph>{UP_EXPAND}</update_graph><search>find second gold</search>'])
check("expand+search continues", not d[0])
check("expand on gt type-match", i[0]["ra_analyse_reward"] == 1.0)
# premature answer while remaining child open -> gate -1
o, r, d, i = env.step(['<think>rush</think><update_graph>{"type":"accept","answer":"B","supporting_facts":["B big"]}</update_graph><answer>B</answer>'])
# note: this accept CLOSES the remaining child (it is the active node) -> gate should NOT fire
check("full multi-hop completes (no gate)", d[0] and r[0] >= 1.0, str(r[0]))

env = make_env(max_steps=3)
reset_one(env, question="Compare A and B?", answer="B", refs=("deck/page_01", "deck/page_02"))
env.step(["<think>go</think><search>find gold page</search>"])
env.step([f'<think>partial</think><update_graph>{UP_EXPAND}</update_graph><search>find second gold</search>'])
o, r, d, i = env.step([f'<think>reject second</think><update_graph>{UP_REJECT}</update_graph><answer>B</answer>'])
check("answer with open child -> gate -1", d[0] and r[0] == -1.0, str(r[0]))

print("== multi-hop with commit-only final ==")
env = make_env()
reset_one(env, question="Compare A and B?", answer="B", refs=("deck/page_01", "deck/page_02"))
env.step(["<think>go</think><search>find gold page</search>"])
env.step([f'<think>partial</think><update_graph>{UP_EXPAND}</update_graph><search>find second gold</search>'])
o, r, d, i = env.step(['<think>last leaf, commit only</think><update_graph>{"type":"accept","answer":"5","supporting_facts":["B=5"]}</update_graph>'])
check("commit-only continues", not d[0])
check("commit-only action label", i[0]["ra_action"] == "commit", i[0]["ra_action"])
check("commit-only earns type-match step reward", i[0]["ra_step_reward"] == 1.0)
check("next obs shows updated graph, no pending", "B=5" in o[0] and "do NOT emit" in o[0], o[0][-300:])
o, r, d, i = env.step(["<think>A=3, B=5, B larger</think><answer>B</answer>"])
check("answer after commit-only completes", d[0] and r[0] >= 1.0, str(r[0]))

print("== max_steps timeout ==")
env = make_env(max_steps=2)
reset_one(env)
env.step(["<think>go</think><search>filler one</search>"])
o, r, d, i = env.step([f'<think>meh</think><update_graph>{UP_REJECT}</update_graph><search>filler two</search>'])
check("timeout done at max_steps", d[0])
check("timeout no-answer reward 0", abs(r[0]) < 1e-6, str(r[0]))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
