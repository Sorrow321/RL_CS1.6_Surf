"""--race-arc-self M (zlr1b2, 2026-10-31): after a greedy eval the --race-arc route becomes the
eval's deepest episode's own line, from the true start to its deepest tick.

(a) arc_self_candidate picks the episode with the smallest field distance and cuts it at its
    deepest tick; off-field rows never count; an empty recording gives None.
(b) resample_polyline lays the line at a uniform arc spacing, from its first point.
(c) the recorder lists the key as training-only; the trainer refuses it without --race-arc.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


class _XField:
    """d = 10,000 - x; x >= 1e6 is off the field."""
    reach_max = 1e9
    cell = 32.0

    def sample(self, pts):
        x = np.atleast_2d(np.asarray(pts, np.float64))[:, 0]
        return np.where(x >= 1e6, 1e9, 10_000.0 - x)


def _write(path, episodes):
    with open(path, "w", encoding="utf-8") as f:
        for k, xs in enumerate(episodes):
            f.write(json.dumps({"map": "m", "tick_ms": 7, "episode": k}) + "\n")
            for t, x in enumerate(xs):
                f.write(json.dumps([t, float(x), 0.0, 0.0, 0, 0, 0, 0, 0, 0]) + "\n")
            f.write(json.dumps({"end": "fail"}) + "\n")


def test_the_deepest_episode_cut_at_its_deepest_tick(tmp_path):
    import train_fast as tf
    p = tmp_path / "traj.jsonl"
    _write(p, [[0, 100, 300, 200], [0, 50, 500, 900, 400, 100], [0, 2e6, 10]])
    pts, d = tf.arc_self_candidate(p, _XField())
    assert d == pytest.approx(10_000.0 - 900.0)
    assert pts[:, 0].tolist() == [0.0, 50.0, 500.0, 900.0]     # cut at the deepest tick


def test_off_field_rows_never_count_and_empty_is_none(tmp_path):
    import train_fast as tf
    p = tmp_path / "traj.jsonl"
    _write(p, [[0, 2e6, 3e6]])
    pts, d = tf.arc_self_candidate(p, _XField())
    assert d == pytest.approx(10_000.0) and len(pts) == 1        # the spawn row only
    q = tmp_path / "empty.jsonl"
    q.write_text("", encoding="utf-8")
    assert tf.arc_self_candidate(q, _XField()) is None


def test_resample_polyline_is_uniform_from_the_first_point():
    import train_fast as tf
    pts = np.array([[0, 0, 0], [100, 0, 0], [100, 0, 0], [100, 300, 0]], np.float64)
    r = tf.resample_polyline(pts, 128.0)
    assert np.allclose(r[0], [0, 0, 0])
    seg = np.linalg.norm(np.diff(r, axis=0), axis=1)
    assert len(r) == 4 and np.all(seg <= 128.0 + 1e-6)
    assert np.allclose(r[1], [100, 28, 0])                       # around the corner by arc


def test_the_recorder_lists_it_as_training_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt as rc
    assert "race_arc_self" in rc.TRAIN_ONLY
