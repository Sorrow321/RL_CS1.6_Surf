"""--save-best-eval (pbr4b4, 2026-10-25): train_fast.eval_score is the autoresearch grader's
number for one greedy eval, so the snapshot the trainer keeps is the one the handoff picks.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/python/test_save_best_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


class _LineField:
    """d = x (u), honest below reach_max: enough of GoalField for eval_score."""
    reach_max = 1e6
    cell = 32.0

    def sample(self, p):
        return np.asarray(p, np.float64)[:, 0].astype(np.float32)


def _write(path, eps):
    """eps: [(episode index or None, [x...], end, tick_ms)] -> a record.py-style jsonl."""
    with open(path, "w", encoding="utf-8") as fh:
        for idx, xs, end, tick in eps:
            h = {"map": "m", "tick_ms": tick}
            if idx is not None:
                h["episode"] = idx
            fh.write(json.dumps(h) + "\n")
            for t, x in enumerate(xs):
                fh.write(json.dumps([t, float(x), 0.0, 0.0, 0, 0, 0, 0]) + "\n")
            fh.write(json.dumps({"end": end, "ticks": len(xs)}) + "\n")


def test_eval_score_rule(tmp_path):
    from train_fast import eval_score
    f = _LineField()
    p = tmp_path / "a.jsonl"
    # three non-finishers from d0 = 1000: best 40% (dmin 600); an episode index >= n is ignored
    _write(p, [(0, [1000, 900, 800], "fail", 10.0), (1, [1000, 700, 600, 650], "fail", 10.0),
               (2, [1000, 990], "trunc", 10.0), (9, [1000, 0], "fail", 10.0)])
    assert eval_score(p, f, 3) == pytest.approx(40.0)
    assert eval_score(p, f, 0) == pytest.approx(100.0)          # 0 = every episode
    # a finish counts 100, + 10 x 1/2 finished, + 10 x (1 - 30 s / 300)
    q = tmp_path / "b.jsonl"
    _write(q, [(1, [1000] * 3000, "done", 10.0), (0, [1000, 500], "fail", 10.0)])
    assert eval_score(q, f, 9) == pytest.approx(100.0 + 5.0 + 9.0)
    # no index: the first n in file order
    r = tmp_path / "c.jsonl"
    _write(r, [(None, [1000, 800], "fail", 10.0), (None, [1000, 100], "fail", 10.0)])
    assert eval_score(r, f, 1) == pytest.approx(20.0)
    e = tmp_path / "e.jsonl"
    e.write_text("")
    assert eval_score(e, f, 9) == float("-inf")


@pytest.mark.skipif(not (ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz").exists(),
                    reason="needs the princessburglar no-corner-cut field")
def test_eval_score_matches_the_grader(tmp_path):
    """On a real eval file the trainer's number is the grader's (score.py) to the rounding."""
    from train_fast import eval_score
    sys.path.insert(0, str(ROOT / "autoresearch"))
    import score
    src = sorted((Path.home() / "ar_pb" / "runs" / "pbr4b4" / "pbr4b4_c2").glob("traj_*.jsonl"))
    if not src:
        pytest.skip("no pulled pbr4b4_c2 eval on this host")
    for fn in src[-3:]:
        want = score.score_file(fn)["score"]
        got = eval_score(fn, score.field(), score.SCORE_EPISODES)
        assert got == pytest.approx(want, abs=1.5e-3), fn.name
