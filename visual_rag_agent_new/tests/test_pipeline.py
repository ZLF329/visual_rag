import threading
import time

from src.pipeline import rollout_then_judge


def test_judging_never_holds_a_rollout_slot():
    rolled, finished = [], []
    lock = threading.Lock()

    def rollout(i):
        time.sleep(0.01)
        with lock:
            rolled.append((i, time.time()))
        return {"id": i}

    def judge(r):
        time.sleep(0.2)                                   # slow judge
        return dict(r, judged=True)

    def finish(r):
        with lock:
            finished.append(r)

    t0 = time.time()
    rollout_then_judge(range(8), rollout, judge, finish, workers=1, judge_workers=8)
    assert sorted(r["id"] for r in finished) == list(range(8)) and all(r["judged"] for r in finished)
    assert max(t for _, t in rolled) - t0 < 0.15          # 8 rollouts done long before 8 x 0.2 s of judging
    assert time.time() - t0 < 0.8


def test_stop_halts_submission_and_errors_propagate():
    stop = threading.Event()
    finished = []

    def finish(r):
        finished.append(r)
        if len(finished) >= 2:
            stop.set()

    rollout_then_judge(range(100), lambda i: (time.sleep(0.01), i)[1], lambda r: r, finish,
                       workers=1, judge_workers=1, stop=stop)
    assert 2 <= len(finished) < 100

    def boom(i):
        raise RuntimeError("cuda out of memory")
    try:
        rollout_then_judge(range(3), boom, lambda r: r, finish, workers=2, judge_workers=2)
    except RuntimeError as exc:
        assert "out of memory" in str(exc)
    else:
        raise AssertionError("rollout error was swallowed")
