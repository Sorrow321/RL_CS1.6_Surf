"""The finish test the multi-map aggregate is built on must actually FIRE.

``race/maps_finished`` counts eval episodes whose recorded path crosses the
finish AABB. If that test silently never fires, the headline metric reads
0.00% forever and is indistinguishable from a genuine null - which on this
project is exactly the failure mode that has cost whole rounds. So it is
pinned in both directions, including the case it exists for: a 1 u thin
trigger curtain crossed at 3,500 u/s, where a point-in-box check tunnels
straight through and reports nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import _seg_hits_box, race_coverage       # noqa: E402


class _Field:
    """Distance-to-goal = distance to a point, which is all race_coverage
    reads (it only ever takes d[0] and d.min())."""

    def __init__(self, goal):
        self.goal = np.asarray(goal, np.float64)

    def sample(self, pts):
        pts = np.asarray(pts, np.float64)
        return np.linalg.norm(pts - self.goal, axis=-1)


def _row(t, p):
    # the recorder's row layout: [tick, x, y, z, vx, vy, vz, yaw, ...]
    return [t, p[0], p[1], p[2], 0.0, 0.0, 0.0, 0.0, 0, 1, 0.0, 0.0, 0.0, 1, 1]


def _write(tmp, episodes, map_name="surf_test"):
    p = Path(tmp) / "traj.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for k, pts in enumerate(episodes):
            f.write(json.dumps({"map": map_name, "tick_ms": 10,
                                "episode": k}) + "\n")
            for t, q in enumerate(pts):
                f.write(json.dumps(_row(t, q)) + "\n")
            f.write(json.dumps({"end": "done", "ticks": len(pts)}) + "\n")
    return p


# ------------------------------------------------------------ the slab test
CURTAIN = {"mins": [-100.0, -0.5, -100.0], "maxs": [100.0, 0.5, 100.0]}


def test_thin_curtain_at_speed_is_not_tunnelled():
    """35 u of travel per tick against a 1 u curtain. This is the whole
    reason src/env.c uses a swept slab test and not a point-in-box test."""
    pts = np.array([[0.0, -70.0, 0.0], [0.0, -35.0, 0.0],
                    [0.0, 35.0, 0.0], [0.0, 70.0, 0.0]])
    assert not any(abs(p[1]) <= 0.5 for p in pts), "no sample is inside"
    assert _seg_hits_box(pts, CURTAIN) is True


def test_path_that_misses_the_curtain_does_not_hit():
    # crosses the y=0 plane but 500 u outside the curtain in x
    pts = np.array([[500.0, -70.0, 0.0], [500.0, 70.0, 0.0]])
    assert _seg_hits_box(pts, CURTAIN) is False


def test_path_that_stops_short_does_not_hit():
    pts = np.array([[0.0, -70.0, 0.0], [0.0, -35.0, 0.0], [0.0, -10.0, 0.0]])
    assert _seg_hits_box(pts, CURTAIN) is False


def test_a_single_frame_cannot_hit():
    assert _seg_hits_box(np.array([[0.0, 0.0, 0.0]]), CURTAIN) is False


def test_axis_parallel_travel_inside_the_slab_still_hits():
    """Moving along x at y=0: dy is 0, so the y slab is degenerate and the
    'already inside' branch is the only thing that can register the hit."""
    pts = np.array([[-500.0, 0.0, 0.0], [500.0, 0.0, 0.0]])
    assert _seg_hits_box(pts, CURTAIN) is True


def test_axis_parallel_travel_outside_the_slab_does_not_hit():
    pts = np.array([[-500.0, 40.0, 0.0], [500.0, 40.0, 0.0]])
    assert _seg_hits_box(pts, CURTAIN) is False


# ------------------------------------------------------- the coverage metric
def test_coverage_counts_a_finish_and_a_null_separately(tmp_path):
    field = _Field([0.0, 0.0, 0.0])
    finisher = [[0.0, -1000.0, 0.0], [0.0, -500.0, 0.0],
                [0.0, -35.0, 0.0], [0.0, 35.0, 0.0]]
    faller = [[0.0, -1000.0, 0.0], [0.0, -900.0, 0.0]]
    p = _write(tmp_path, [finisher, faller])
    pct_sum, n, fin = race_coverage(p, field, CURTAIN)
    assert n == 2
    assert fin == 1, "the finishing episode must be counted"
    # finisher: d0 1000 -> min 35 = 96.5%; faller: 1000 -> 900 = 10%
    assert pct_sum / n == pytest.approx((96.5 + 10.0) / 2, abs=0.1)


def test_coverage_is_per_episode_not_against_a_shared_d0(tmp_path):
    """Two episodes spawning at different distances each score against their
    OWN start. A shared denominator would make a short spawn look like a
    poor run."""
    field = _Field([0.0, 0.0, 0.0])
    near = [[0.0, -100.0, 0.0], [0.0, -50.0, 0.0]]      # 50% of 100
    far = [[0.0, -1000.0, 0.0], [0.0, -500.0, 0.0]]     # 50% of 1000
    p = _write(tmp_path, [near, far])
    pct_sum, n, _ = race_coverage(p, field, None)
    assert n == 2
    assert pct_sum / n == pytest.approx(50.0, abs=0.1)


def test_coverage_clips_and_never_reports_negative(tmp_path):
    """An episode that only ever moves AWAY scores 0, not a negative that
    would drag a fleet mean below zero."""
    field = _Field([0.0, 0.0, 0.0])
    away = [[0.0, -100.0, 0.0], [0.0, -300.0, 0.0]]
    p = _write(tmp_path, [away], )
    pct_sum, n, _ = race_coverage(p, field, None)
    assert n == 1
    assert pct_sum == pytest.approx(0.0)


def test_a_death_dive_past_the_goal_is_not_a_finish(tmp_path):
    """CLAUDE.md's warning, as a test: an agent that falls PAST the finish
    into goal-adjacent airspace scores well on distance and has finished
    nothing. The box test must say 0 where the distance proxy says yes.

    The dive crosses the finish PLANE (y=0) at z = -167, outside the
    curtain's z extent - the geometry of falling under the finish rather
    than through it, which is what 5 of 9 cannonball episodes ending near
    z = -4,200 actually were."""
    field = _Field([0.0, 0.0, 0.0])
    dive = [[0.0, -1000.0, -300.0], [0.0, 0.0, -166.7], [0.0, 200.0, -140.0]]
    p = _write(tmp_path, [dive])
    pct_sum, n, fin = race_coverage(p, field, CURTAIN)
    assert n == 1
    d = np.linalg.norm(np.asarray(dive, float), axis=1)
    want = 100.0 * (d[0] - d.min()) / d[0]
    assert pct_sum == pytest.approx(want, abs=0.05)
    assert want > 80.0, "the distance metric is flattered by the dive"
    assert fin == 0, "and the box test is not"


def test_a_hit_and_a_miss_differ_only_in_the_crossing_point(tmp_path):
    """Same fall, shifted so it goes THROUGH the curtain instead of under
    it. If this does not flip to 1 the test above proves nothing."""
    field = _Field([0.0, 0.0, 0.0])
    through = [[0.0, -1000.0, -30.0], [0.0, 0.0, -16.7], [0.0, 200.0, -14.0]]
    p = _write(tmp_path, [through])
    _, n, fin = race_coverage(p, field, CURTAIN)
    assert n == 1
    assert fin == 1
