"""Behavioral tests for the agreed search-only decay and hard-budget protocol."""
from pathlib import Path
import yaml

from src.agent import Agent
from tests.fakes import ADD, ANSWER, CROP, RESOLVE, REVISIT, SEARCH, UPD_FACT, FakeVLM, act, page
from tests.test_env import close, make_env, reset_one, step


def config():
    path = Path(__file__).resolve().parents[1] / "rl/experiments/post_gold_search_decay.args"
    return {key.removeprefix("env.slidevqa."): yaml.safe_load(value)
            for key, value in (line.lstrip("+").split("=", 1) for line in path.read_text().splitlines())}


def test_decay_counts_each_extra_search_including_empty_results():
    for extra, expected in [(0, .5), (1, .25), (2, .125), (3, .0625)]:
        env = make_env(max_steps=10, slidevqa=config())
        reset_one(env)
        _, _, _, info = step(env, act(None, SEARCH("gold")))
        assert info["ra_post_gold_searches"] == 0 and info["ra_first_all_gold_seen_turn"] == 1
        if extra:
            step(env, act(page(facts=[ADD("42", [0, 0, 10, 10])]), SEARCH("gold"), tree=RESOLVE("42", ["P1"])))
            for _ in range(extra - 1):
                step(env, act(None, SEARCH("gold")))
            final = act(None, ANSWER("42"))
        else:
            final = act(page(facts=[ADD("42", [0, 0, 10, 10])]), ANSWER("42"), tree=RESOLVE("42", ["P1"]))
        _, reward, done, info = step(env, final)
        assert done and close(reward, 1.0 + expected)
        assert info["ra_post_gold_searches"] == extra
        assert close(.5 * info["ra_post_gold_search_reward"], expected)
        assert close(info["ra_post_gold_search_bonus"], expected)


def test_crop_and_revisit_preserve_bonus_and_do_not_reset_search_count():
    env = make_env(max_steps=10, slidevqa=config()); reset_one(env)
    step(env, act(None, SEARCH("gold")))
    step(env, act(page(facts=[ADD("42", [0, 0, 100, 100])]), CROP([0, 0, 100, 100]), tree=RESOLVE("42", ["P1"])))
    _, reward, done, info = step(env, act(None, ANSWER("42")))
    assert done and close(reward, 1.5) and info["ra_post_gold_searches"] == 0
    env = make_env(max_steps=10, slidevqa=config()); reset_one(env)
    step(env, act(None, SEARCH("gold")))
    step(env, act(page(facts=[ADD("42", [0, 0, 100, 100])]), SEARCH("second"), tree=RESOLVE("42", ["P1"])))
    step(env, act(None, REVISIT("P1")))
    step(env, act(None, CROP([0, 0, 100, 100])))
    _, reward, _, info = step(env, act(None, ANSWER("42")))
    assert close(reward, 1.25) and info["ra_post_gold_searches"] == 1
    assert info["ra_first_all_gold_seen_turn"] == 1


def test_bonus_requires_correct_answer_and_final_gold_evidence():
    for answer, cite, refs, expected in [("wrong", True, ("deck/page_01",), 0),
                                         ("42", False, ("deck/page_01",), 1),
                                         ("42", True, ("deck/page_01", "deck/page_02"), 1),
                                         ("42", True, (), 1)]:
        env = make_env(slidevqa=config()); reset_one(env, refs=refs)
        step(env, act(None, SEARCH("gold")))
        _, reward, done, info = step(env, act(page(facts=[ADD("42", [0, 0, 10, 10])]), ANSWER(answer),
                                            tree=RESOLVE("42", ["P1"]) if cite else None))
        assert done and close(reward, expected) and info["ra_post_gold_search_reward"] == 0


def test_grounding_remains_independent_but_cannot_offset_timeout_or_format_failure():
    for answer, expected in [("42", 2.0), ("wrong", .5)]:
        env = make_env(slidevqa=config())
        reset_one(env, keyfacts=[{"page": 1, "bbox_2d": [0, 0, 100, 100]}])
        step(env, act(None, SEARCH("gold")))
        _, reward, done, info = step(env, act(page(facts=[ADD("42", [0, 0, 100, 100])]),
                                            ANSWER(answer), tree=RESOLVE("42", ["P1"])))
        assert done and close(info["ra_grounding_reward"], 1.0) and close(reward, expected)
        assert close(info["ra_post_gold_search_bonus"], .5 if answer == "42" else 0.0)
    for final, end in [(act(None, SEARCH("more")), "budget_timeout"), ("bad", "format_error")]:
        env = make_env(max_steps=3, slidevqa=config())
        reset_one(env, keyfacts=[{"page": 1, "bbox_2d": [0, 0, 100, 100]}])
        step(env, act(None, SEARCH("gold")))
        step(env, act(page(facts=[ADD("42", [0, 0, 100, 100])]), CROP([0, 0, 100, 100]),
                      tree=RESOLVE("42", ["P1"])))
        _, reward, done, info = step(env, final)
        assert done and info["ra_terminated_by"] == end and reward == -1.0
        assert info["ra_post_gold_search_bonus"] == 0.0


def test_last_gold_search_is_not_charged_and_initial_question_search_is_observed():
    env = make_env(slidevqa=config()); reset_one(env, refs=("deck/page_01", "deck/page_02"))
    step(env, act(None, SEARCH("gold")))
    step(env, act(page(facts=[ADD("a", [0, 0, 10, 10])]), SEARCH("second")))
    _, reward, done, info = step(env, act(page(facts=[ADD("b", [0, 0, 10, 10])]), ANSWER("42"),
                                        tree=RESOLVE("42", ["P1", "P2"])))
    assert done and close(reward, 1.5) and info["ra_first_all_gold_seen_turn"] == 2
    assert info["ra_post_gold_searches"] == 0
    env = make_env(slidevqa=dict(config(), first_search="question"))
    _, info = reset_one(env, question="gold")
    assert info["ra_seen_pages"] == ("deck/page_01",) and info["ra_first_all_gold_seen_turn"] == 1


def test_hard_budget_alignment_for_answer_search_crop_and_edit_error():
    for final, end in [(act(None, ANSWER("42")), "answer"),
                       (act(None, SEARCH("more")), "budget_timeout"),
                       (act(None, CROP([0, 0, 100, 100])), "budget_timeout"),
                       (act(page(facts=[UPD_FACT("P1.F99", "x", [0, 0, 10, 10])]), ANSWER("42")), "budget_timeout"),
                       ("<think>invalid</think>", "format_error")]:
        env = make_env(max_steps=3, slidevqa=dict(config(), first_search="question"))
        _, initial = reset_one(env, question="gold")
        first = act(page(facts=[ADD("42", [0, 0, 10, 10])]), SEARCH("second"), tree=RESOLVE("42", ["P1"]))
        _, _, _, second = step(env, first)
        _, reward, done, info = step(env, final)
        vlm = FakeVLM([first, final])
        agent = Agent(vlm, make_env().retriever, first_search="question", max_turns=3, format_retries=0)
        out = agent.run("gold")
        assert done and info["ra_terminated_by"] == out["terminated_by"] == end
        assert out["turns"] == 3 and len(vlm.calls) == 2
        assert initial["observation_chat"] == vlm.calls[0][0]
        assert second["observation_chat"] == vlm.calls[1][0]
        assert not any("Budget exhausted" in m["content"] for calls in vlm.calls for m in calls[0])
        assert (reward == -1 if end != "answer" else close(reward, 1.25))
        assert env.retriever.n == agent.retriever.n == 2


def test_default_eval_has_no_hidden_format_retry():
    vlm = FakeVLM(["bad", act(None, ANSWER("42"))])
    out = Agent(vlm, make_env().retriever, max_turns=2).run("q")
    assert out["terminated_by"] == "format_error" and len(vlm.calls) == 1


def test_invalid_budget_and_decay_fail_early():
    for cfg in [dict(config(), post_gold_search_decay=-.5), dict(config(), post_gold_search_decay=1.1)]:
        try:
            make_env(slidevqa=cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid decay accepted")
    for create in [lambda: make_env(max_steps=1, slidevqa={"first_search": "question"}),
                   lambda: Agent(FakeVLM([]), make_env().retriever, max_turns=1, first_search="question")]:
        try:
            create()
        except ValueError:
            pass
        else:
            raise AssertionError("budget leaves no model turn")
