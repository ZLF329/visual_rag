import envs as E
from tests.fakes import ADD, ANSWER, CROP, EXPAND, RESOLVE, REVISIT, SEARCH, UPD_FACT, FakeRetriever, act, page

TABLE = {"gold": ["deck/page_01", "deck/page_77"], "second": ["deck/page_02", "deck/page_78"]}
MAX = 8
STEP_COST = 0.1 / MAX     # ACCESS_COST_W * one access / max_steps


def make_env(max_steps=MAX, scope=None, slidevqa=None):
    slidevqa = dict(slidevqa or {})
    if scope:
        slidevqa["retrieval_scope"] = scope
    env = E.SlideVQAMultiProcessEnv(seed=0, env_num=1, group_n=1, env_config={"max_steps": max_steps, "slidevqa": slidevqa})
    env.retriever = FakeRetriever(size=(728, 532), table=TABLE)       # on the 28-px grid: shown == original
    env.reward_fn = lambda **kw: 1.0 if kw["solution_str"].strip() == kw["ground_truth"].strip() else 0.0
    env.support_fn = None
    env._step_workers = 1
    return env


def reset_one(env, question="What is X?", answer="42", refs=("deck/page_01",), keyfacts=None):
    item = {"question": question, "answer": answer, "deck_name": "deck", "reference_pages": list(refs), "sample_id": "t1"}
    if keyfacts is not None:
        item["keyfacts"] = keyfacts
    obs, infos = env.reset([item])
    return obs[0], infos[0]


def step(env, text):
    o, r, d, i = env.step([text])
    return o[0], r[0], d[0], i[0]


def close(a, b):
    return abs(a - b) < 1e-6


def test_reset_observation():
    env = make_env()
    obs, info = reset_one(env)
    assert "Question tree:" in obs and "Page in view: none" in obs and "Page memory: none" in obs
    assert f"Turn 1 of {MAX}:" in obs
    assert info["observation_chat"][0]["role"] == "system" and info["observation_images"] == []
    roles = [m["role"] for m in info["observation_chat"]]                        # strict alternation (§11.17)
    assert roles in (["system", "user"], ["system", "user", "assistant", "user"]), roles


def test_single_hop_reward():
    env = make_env()
    reset_one(env)
    o, r, d, i = step(env, act(None, SEARCH("find gold")))
    assert not d and r == 0.0 and i["ra_step_reward"] == 0.0 and "Page in view: P1" in o and "deck/page_01" not in o
    assert len(i["observation_images"]) == 1
    o, r, d, i = step(env, act(page(facts=[ADD("X is 42", [0, 0, 10, 10])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])))
    assert d and i["won"] and i["ra_final_pages"] == ("deck/page_01",)
    # 1 + 0.5*cov(1) + 0.2*cmp(1) + 0.5*grounding(0) - 0.1*1/8
    assert close(r, 1.7 - STEP_COST), r


def test_multi_hop_coverage_and_cmp_on_final_pages():
    env = make_env()
    reset_one(env, question="Compare A and B?", answer="B", refs=("deck/page_01", "deck/page_02"))
    step(env, act(None, SEARCH("find gold")))
    step(env, act(page(facts=[ADD("A=3", [0, 0, 10, 10])]), SEARCH("second page"), tree=EXPAND("A?", "3", "B?", ["P1"])))
    o, r, d, i = step(env, act(page(facts=[ADD("B=5", [0, 0, 10, 10])]), ANSWER("B"), tree=RESOLVE("5", ["P2"])))
    assert d and i["ra_coverage"] == 1.0 and i["ra_cmp_reward"] == 1.0 and close(r, 1.7 - 2 * STEP_COST), r
    assert i["ra_final_pages"] == ("deck/page_01", "deck/page_02") and i["ra_counts"] == {"search": 2}


def test_partial_coverage_and_noise_page():
    env = make_env()
    reset_one(env, question="Q", answer="42", refs=("deck/page_01", "deck/page_02"))
    step(env, act(None, SEARCH("nothing")))                        # noise page P1
    step(env, act(page(facts=[ADD("n", [0, 0, 10, 10])]), SEARCH("find gold"), tree=EXPAND("noise part", "n", "rest", ["P1"])))
    o, r, d, i = step(env, act(page(facts=[ADD("X is 42", [0, 0, 10, 10])]), ANSWER("42"), tree=RESOLVE("42", ["P2"])))
    # cov = 1/2 (page_01 found, page_02 not), cmp gated off
    assert d and close(i["ra_coverage"], 0.5) and close(r, 1 + 0.25 - 2 * STEP_COST), r


def test_final_pages_are_the_pages_the_tree_cites():
    env = make_env()
    reset_one(env, refs=("deck/page_01",))
    step(env, act(None, SEARCH("find gold")))
    step(env, act(page(facts=[ADD("X is 42", [0, 0, 10, 10])]), SEARCH("other"), tree=RESOLVE("42", ["P1"])))   # P1 answers N0
    o, r, d, i = step(env, act(page("unrelated"), ANSWER("42")))                             # P2 (page_51) only visited
    assert i["ra_final_pages"] == ("deck/page_01",) and i["ra_cmp_reward"] == 1.0          # visited noise not counted


def test_format_error_terminates():
    env = make_env()
    reset_one(env)
    o, r, d, i = step(env, "<think>t</think>")
    assert d and r == -1.0 and i["format_error"]
    env = make_env()
    reset_one(env)
    step(env, act(None, SEARCH("find gold")))
    step(env, act(None, CROP(bbox=[0, 0, 100, 100])))
    o, r, d, i = step(env, act(None, CROP(bbox=[0, 0, 100, 100])))                         # second crop in a visit
    assert d and r == -1.0


def test_edit_error_is_soft():
    env = make_env()
    reset_one(env)
    step(env, act(None, SEARCH("find gold")))
    o, r, d, i = step(env, act(page(facts=[UPD_FACT("P1.F7", "x", [0, 0, 10, 10])]), SEARCH("q")))
    assert not d and r == 0.0 and i["ra_step_kind"] == "edit_error" and i["ra_step_reward"] == -1.0
    assert "NOT applied" in o and "Page in view: P1" in o
    o, r, d, i = step(env, act(page(facts=[ADD("X is 42", [0, 0, 10, 10])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])))
    assert d and i["won"]


def test_revisit_and_crop_show_images_and_cost():
    env = make_env()
    reset_one(env)
    step(env, act(None, SEARCH("find gold")))
    step(env, act(page("p", [ADD("v", [100, 100, 300, 200])]), SEARCH("other")))
    o, r, d, i = step(env, act(None, REVISIT("P1")))
    assert "page P1 (revisit" in i["observation_chat"][-1]["content"] and len(i["observation_images"]) == 1
    assert i["valid_actions"] == ("search", "revisit", "crop", "answer")
    o, r, d, i = step(env, act(None, CROP(fact_id="P1.F1")))
    assert len(i["observation_images"]) == 2 and "ZOOM image" in o and i["valid_actions"] == ("search", "revisit", "answer")
    o, r, d, i = step(env, act(None, ANSWER("42"), tree=RESOLVE("42", ["P1"])))
    assert i["ra_counts"] == {"search": 2, "revisit": 1, "crop": 1} and close(r, 1.7 - 4 * STEP_COST)


def test_grounding_uses_page_frame_facts_only():
    env = make_env()
    reset_one(env, keyfacts=[{"page": 1, "bbox_2d": [0, 0, 100, 100]}])
    step(env, act(None, SEARCH("find gold")))
    step(env, act(page(facts=[ADD("page fact", [0, 0, 100, 100])]), CROP(bbox=[0, 0, 200, 200])))
    o, r, d, i = step(env, act(page(facts=[ADD("zoom fact", [0, 0, 10, 10])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])))
    assert close(i["ra_grounding_reward"], 1.0)
    env = make_env()
    reset_one(env, keyfacts=[{"page": 1, "bbox_2d": [0, 0, 100, 100]}])
    step(env, act(None, SEARCH("find gold")))
    o, r, d, i = step(env, act(page(facts=[ADD("half", [0, 0, 50, 100])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])))
    assert close(i["ra_grounding_reward"], 0.5)


def test_budget_is_a_hard_timeout():
    env = make_env(max_steps=2)
    reset_one(env)
    step(env, act(None, SEARCH("find gold")))
    o, r, d, i = step(env, act(page("p", [ADD("X is 42", [0, 0, 10, 10])]), SEARCH("other")))
    assert d and r == -1 and i["ra_terminated_by"] == "budget_timeout"
    assert i["ra_budget_exhausted"] and not i["format_error"] and not i["won"]
    assert i["valid_actions"] == () and env.retriever.n == 1
    assert "Budget exhausted" not in o
    o, r, d, i = step(env, act(None, ANSWER("42")))
    assert d and r == 0 and not i["ra_final_answer"]
    for invalid in ("<think>missing action</think>",):
        env = make_env(max_steps=1)
        reset_one(env)
        _, reward, done, info = step(env, invalid)
        assert done and reward == -1.0 and info["format_error"] and info["valid_actions"] == ()
        assert not info["won"] and not info["ra_final_answer"]


def test_no_new_page_then_answer():
    env = make_env()
    reset_one(env)
    step(env, act(None, SEARCH("find gold")))
    step(env, act(None, SEARCH("find gold")))
    o, r, d, i = step(env, act(None, SEARCH("find gold")))
    assert not d and "no new page" in o and i["ra_retrieved_page_ids"] == ()
    o, r, d, i = step(env, act(None, ANSWER("x")))
    assert d and i["ra_final_pages"] == () and close(r, -3 * STEP_COST)


def test_retrieval_scope():
    env = make_env()
    reset_one(env)
    step(env, act(None, SEARCH("find gold")))
    assert env.retriever.deck_names == [None]                       # flat corpus by default
    env = make_env(scope="deck")
    reset_one(env)
    step(env, act(None, SEARCH("find gold")))
    assert env.retriever.deck_names == ["deck"]


def test_grounding_in_original_pixels_when_pages_are_scaled():
    env = make_env()
    env.retriever = FakeRetriever(size=(800, 600), table=TABLE)       # shown 728x532
    reset_one(env, keyfacts=[{"page": 1, "bbox_2d": [0, 0, 400, 300]}])
    o, r, d, i = step(env, act(None, SEARCH("find gold")))
    assert i["observation_images"][0].shape[:2] == (532, 728) and "728x532 pixels" in i["observation_chat"][-1]["content"]
    o, r, d, i = step(env, act(page(facts=[ADD("quarter", [0, 0, 364, 266])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])))
    assert close(i["ra_grounding_reward"], 1.0)


def test_first_search_is_the_question_at_reset():
    env = make_env()
    env.slidevqa_config = {"first_search": "question"}
    env.first_search = "question"
    obs, info = reset_one(env, question="find gold?")
    assert "Page in view: P1" in obs and len(info["observation_images"]) == 1 and f"Turn 1 of {MAX - 1}:" in obs
    assert env.retriever.deck_names == [None] and env.states[0].turn == 1 and env.states[0].counts["search"] == 1
    o, r, d, i = step(env, act(page(facts=[ADD("X is 42", [0, 0, 10, 10])]), ANSWER("42"), tree=RESOLVE("42", ["P1"])))
    assert d and i["won"]



def test_tree_pages_attach_earlier_pages():
    """DESIGN.md §11.23: the tree lists the pages a conclusion rests on, earlier ones included (no revisit)."""
    env = make_env()
    obs, info = reset_one(env, question="Profit?", answer="12.5", refs=("deck/page_01", "deck/page_02"))
    assert "<tree>{...}</tree>" in info["observation_chat"][0]["content"]                 # four blocks (§11.24)
    step(env, act(None, SEARCH("find gold")))
    step(env, act(page(facts=[ADD("profit 12.5", [0, 0, 10, 10])]), SEARCH("second page")))
    o, r, d, i = step(env, act(page(facts=[ADD("year 2011", [0, 0, 10, 10])]), ANSWER("12.5"), tree=RESOLVE("12.5", ["P1", "P2"])))
    assert d and set(i["ra_final_pages"]) == {"deck/page_01", "deck/page_02"}


def test_coverage_all_mode_is_all_or_nothing():
    """COVERAGE_MODE=all (DESIGN.md §11.31, the old v9 R_hit_all_gt): coverage counts only when every gold page is in
    E_final; a partial hit scores 0, a full hit 1."""
    import os
    os.environ["COVERAGE_MODE"] = "all"
    try:
        env = make_env()
        reset_one(env, question="Q", answer="42", refs=("deck/page_01", "deck/page_02"))
        step(env, act(None, SEARCH("nothing")))
        step(env, act(page(facts=[ADD("n", [0, 0, 10, 10])]), SEARCH("find gold"), tree=EXPAND("noise part", "n", "rest", ["P1"])))
        o, r, d, i = step(env, act(page(facts=[ADD("X is 42", [0, 0, 10, 10])]), ANSWER("42"), tree=RESOLVE("42", ["P2"])))
        assert d and i["ra_coverage"] == 0.0 and close(r, 1 - 2 * STEP_COST), (i["ra_coverage"], r)
        env = make_env()
        reset_one(env, question="Compare A and B?", answer="B", refs=("deck/page_01", "deck/page_02"))
        step(env, act(None, SEARCH("find gold")))
        step(env, act(page(facts=[ADD("A=3", [0, 0, 10, 10])]), SEARCH("second page"), tree=EXPAND("A?", "3", "B?", ["P1"])))
        o, r, d, i = step(env, act(page(facts=[ADD("B=5", [0, 0, 10, 10])]), ANSWER("B"), tree=RESOLVE("5", ["P2"])))
        assert d and i["ra_coverage"] == 1.0 and close(r, 1.7 - 2 * STEP_COST), (i["ra_coverage"], r)
    finally:
        del os.environ["COVERAGE_MODE"]


def test_corpus_routes_the_search():
    """A question searches the corpus it comes from (kwargs "corpus", default "train"): val questions retrieve from
    the val index, as when their pass@5 was measured (DESIGN.md §11.31)."""
    env = make_env()
    train, val = env.retriever, FakeRetriever(size=(728, 532), table=TABLE)
    env.retrievers = {"train": train, "val": val}
    env.retriever = None
    reset_one(env)
    step(env, act(None, SEARCH("find gold")))
    assert train.n == 1 and val.n == 0
    item = {"question": "Q", "answer": "42", "deck_name": "deck", "reference_pages": ["deck/page_01"], "sample_id": "v1",
            "corpus": "val"}
    env.reset([item])
    step(env, act(None, SEARCH("find gold")))
    assert train.n == 1 and val.n == 1
    item["corpus"] = "test"
    env.reset([item])
    try:
        step(env, act(None, SEARCH("find gold")))
        raise AssertionError("an unknown corpus must fail loudly")
    except KeyError:
        pass


def test_previous_response_is_the_canonical_render():
    """As in eval (src/agent.py): the next turn's context holds the canonical render of the applied turn, not the raw
    text (stray whitespace or text after the blocks never reaches the next prompt, DESIGN.md §11.31)."""
    from src.protocol import parse_turn, render_turn
    env = make_env()
    reset_one(env)
    raw = act(None, SEARCH("find gold")) + "\n\n trailing words"
    o, r, d, i = step(env, raw)
    prev = [m for m in i["observation_chat"] if m["role"] == "assistant"][-1]["content"]
    t = parse_turn(raw, view="page", current_page="P1")
    assert prev == render_turn(t.think, t.act) and "trailing" not in prev, prev


def test_reward_judge_is_the_eval_judge():
    """rl/reward_judge.compute_score (the env's reward_fn_path) is src/judge.judge_answer_equivalence: 1.0 when it judges
    correct, 0.0 when not, 0.0 (logged) when the judge call fails (DESIGN.md §11.31)."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("reward_judge", Path(E.__file__).with_name("reward_judge.py"))
    rj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rj)
    seen = []
    rj.judge_answer_equivalence = lambda **kw: seen.append(kw) or {"correct": kw["prediction"] == "42"}
    assert rj.compute_score("slidevqa", "42", "42", {"question": "What is X?", "sample_id": "t1"}) == 1.0
    assert seen[-1]["question"] == "What is X?" and seen[-1]["gold_answer"] == "42"
    assert rj.compute_score("slidevqa", "41", "42", {"question": "What is X?"}) == 0.0

    def boom(**kw):
        raise RuntimeError("api down")
    rj.judge_answer_equivalence = boom
    assert rj.compute_score("slidevqa", "42", "42", {}) == 0.0


def test_every_info_has_the_same_keys():
    """verl-agent's rollout loop copies the ra_* fields of every step's infos into the batch and collates the steps;
    a key that only some steps carry (e.g. ra_retrieved_page_ids, set by a search) breaks the collation. Reset, search,
    crop, answer and already-done infos all carry the same keys (DESIGN.md §11.31)."""
    env = make_env()
    _, info0 = reset_one(env)
    keys = set(info0)
    o, r, d, i1 = step(env, act(None, SEARCH("find gold")))
    o, r, d, i2 = step(env, act(page(facts=[ADD("X is 42", [0, 0, 10, 10])]), CROP([0, 0, 100, 100])))
    o, r, d, i3 = step(env, act(page(desc=None), ANSWER("42"), tree=RESOLVE("42", ["P1"])))
    o, r, d, i4 = step(env, act(None, ANSWER("42")))                   # after done
    for info in (i1, i2, i3, i4):
        assert set(info) == keys, (set(info) ^ keys)
    for k in ("ra_step_kind", "ra_retrieved_page_ids", "ra_hit_pages", "ra_hit_all_gt", "ra_retrieval_reward"):
        assert k in keys, k


def test_envs_with_the_same_config_share_one_retriever():
    """verl-agent builds a train and a val env; each used to load its own ColQwen encoder and page banks, and the second
    load ran the GPU out of memory at the final validation (DESIGN.md §11.31). Envs with the same retriever config now
    share one set of retrievers."""
    built = []
    orig = E._build_retrievers
    E._RETRIEVERS.clear()
    E._build_retrievers = lambda cfg: built.append(1) or ({"train": FakeRetriever(size=(728, 532), table=TABLE)}, None)
    try:
        cfg = {"max_steps": MAX, "slidevqa": {"index_paths": "train=/a", "retriever_path": "/m/colqwen"}}
        a = E.SlideVQAMultiProcessEnv(env_config=cfg)
        b = E.SlideVQAMultiProcessEnv(env_config=cfg)
        a._ensure_retriever(); b._ensure_retriever()
        assert len(built) == 1 and a.retrievers["train"] is b.retrievers["train"]
    finally:
        E._build_retrievers = orig
        E._RETRIEVERS.clear()
