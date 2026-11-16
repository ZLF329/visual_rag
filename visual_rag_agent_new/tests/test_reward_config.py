"""Exercise the next experiment's real override file through terminal rollouts."""

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from tests.fakes import ADD, ANSWER, EXPAND, RESOLVE, SEARCH, act, page
from tests.test_env import close, make_env, reset_one, step


def experiment_config():
    path = Path(__file__).resolve().parents[1] / "rl/experiments/hybrid_coverage_gspo.args"
    overrides = dict(line.lstrip("+").split("=", 1) for line in path.read_text().splitlines())
    prefix = "env.slidevqa."
    return {key[len(prefix):]: yaml.safe_load(value) for key, value in overrides.items() if key.startswith(prefix)}


def test_hybrid_reward_partial_and_full_with_correct_and_wrong_answers():
    for prediction, answer_score in (("42", 1.0), ("wrong", 0.0)):
        env = make_env(slidevqa=experiment_config())
        reset_one(env, refs=("deck/page_01", "deck/page_02"))
        step(env, act(None, SEARCH("find gold")))
        _, reward, done, info = step(env, act(page(facts=[ADD("X=42", [0, 0, 10, 10])]),
                                            ANSWER(prediction), tree=RESOLVE("42", ["P1"])))
        assert done and info["ra_coverage"] == 0.5 and not info["ra_hit_all_gt"]
        assert close(reward, answer_score + 0.1), reward

        env = make_env(slidevqa=experiment_config())
        reset_one(env, refs=("deck/page_01", "deck/page_02"))
        step(env, act(None, SEARCH("find gold")))
        step(env, act(page(facts=[ADD("A=3", [0, 0, 10, 10])]), SEARCH("second page"),
                      tree=EXPAND("A?", "3", "B?", ["P1"])))
        _, reward, done, info = step(env, act(page(facts=[ADD("B=5", [0, 0, 10, 10])]),
                                            ANSWER(prediction), tree=RESOLVE("5", ["P2"])))
        assert done and info["ra_coverage"] == 1.0 and info["ra_hit_all_gt"]
        assert close(reward, answer_score + 0.5), reward


def test_hybrid_full_coverage_is_not_gated_by_precision():
    env = make_env(slidevqa=experiment_config())
    reset_one(env)
    step(env, act(None, SEARCH("nothing")))
    step(env, act(page(facts=[ADD("noise", [0, 0, 10, 10])]), SEARCH("find gold"),
                  tree=EXPAND("noise part", "n", "rest", ["P1"])))
    _, reward, done, info = step(env, act(page(facts=[ADD("X=42", [0, 0, 10, 10])]),
                                        ANSWER("42"), tree=RESOLVE("42", ["P2"])))
    assert done and info["ra_cmp_reward"] == 0.5 and info["ra_hit_all_gt"]
    assert close(reward, 1.5), reward


def test_hybrid_visited_but_uncited_gold_does_not_earn_coverage():
    env = make_env(slidevqa=experiment_config())
    reset_one(env)
    step(env, act(None, SEARCH("find gold")))
    step(env, act(None, SEARCH("nothing")))
    _, reward, done, info = step(env, act(page(facts=[ADD("noise", [0, 0, 10, 10])]),
                                        ANSWER("42"), tree=RESOLVE("42", ["P2"])))
    assert done and "deck/page_01" in info["ra_hit_pages"]
    assert info["ra_coverage"] == 0.0 and not info["ra_hit_all_gt"]
    assert close(reward, 1.0), reward


def test_hybrid_empty_gold_has_no_full_coverage_bonus():
    env = make_env(slidevqa=experiment_config())
    reset_one(env, refs=())
    _, reward, done, info = step(env, act(None, ANSWER("42")))
    assert done and info["ra_coverage"] == 0.0 and not info["ra_hit_all_gt"]
    assert close(reward, 1.0), reward


def test_hybrid_grounding_retains_half_weight_and_format_error_penalty():
    env = make_env(slidevqa=experiment_config())
    reset_one(env, keyfacts=[{"page": 1, "bbox_2d": [0, 0, 100, 100]}])
    step(env, act(None, SEARCH("find gold")))
    _, reward, done, info = step(env, act(page(facts=[ADD("half", [0, 0, 50, 100])]),
                                        ANSWER("42"), tree=RESOLVE("42", ["P1"])))
    assert done and close(info["ra_grounding_reward"], 0.5)
    assert close(reward, 1.75), reward
    reset_one(env)
    _, reward, done, info = step(env, "<think>incomplete</think>")
    assert done and info["format_error"] and reward == -1.0


def test_hybrid_explicit_config_overrides_launcher_exports():
    with patch.dict(os.environ, {"COVERAGE_MODE": "all", "COVERAGE_W": "0.5",
                                 "FULL_COVERAGE_W": "0", "GROUNDING_W": "0.9",
                                 "CMP_W": "0.2", "ACCESS_COST_W": "0.1"}):
        env = make_env(slidevqa=experiment_config())
        reset_one(env, refs=("deck/page_01", "deck/page_02"),
                  keyfacts=[{"page": 1, "bbox_2d": [0, 0, 100, 100]}])
        step(env, act(None, SEARCH("find gold")))
        _, reward, done, info = step(env, act(page(facts=[ADD("half", [0, 0, 50, 100])]),
                                            ANSWER("42"), tree=RESOLVE("42", ["P1"])))
        assert done and info["ra_coverage"] == 0.5
        assert close(reward, 1.35), reward
        reset_one(env)
        step(env, act(None, SEARCH("find gold")))
        _, reward, done, _ = step(env, act(page(facts=[ADD("X=42", [0, 0, 10, 10])]),
                                         ANSWER("42"), tree=RESOLVE("42", ["P1"])))
        assert done and close(reward, 1.5), reward
