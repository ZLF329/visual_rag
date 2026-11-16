"""Rollouts and judging on separate thread pools, shared by evaluation and SFT generation.

A rollout worker hands its result to the judge pool and immediately starts the next sample, so the
judge's API latency never holds a vLLM rollout slot.
"""
from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from typing import Any, Callable, Iterable


def rollout_then_judge(
    items: Iterable[Any],
    rollout: Callable[[Any], dict[str, Any]],
    judge: Callable[[dict[str, Any]], dict[str, Any]],
    finish: Callable[[dict[str, Any]], None],
    *,
    workers: int,
    judge_workers: int,
    stop: threading.Event | None = None,
) -> None:
    """rollout(item) on `workers` threads -> judge(result) on `judge_workers` threads -> finish(result).
    At most 2 x workers rollouts are in flight; no new item is submitted once `stop` is set. An exception
    in rollout, judge or finish is raised here after the pools drain."""
    with ThreadPoolExecutor(max_workers=judge_workers) as judge_pool, ThreadPoolExecutor(max_workers=workers) as pool:
        judged: list[Future] = []

        def hand_off(done: Future) -> None:
            result = done.result()
            judged.append(judge_pool.submit(lambda: finish(judge(result))))

        pending: set[Future] = set()
        for item in items:
            if stop is not None and stop.is_set():
                break
            pending.add(pool.submit(rollout, item))
            if len(pending) >= 2 * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    hand_off(fut)
        for fut in as_completed(pending):
            hand_off(fut)
        for fut in judged:
            fut.result()
