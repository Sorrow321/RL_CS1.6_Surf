"""explore_phase1 --cell-contact (zlr2b1, 2026-11-01): contact_rows marks the states the map holds -
walkable ground, or a tick whose vz change departs UP from free fall (a surfable face) - and never
free fall, a vertical wall (vz follows gravity), a ceiling (pushed down), a fall pinned at the
-sv_maxvelocity clamp, or a tick whose successor belongs to another run."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE          # noqa: E402
from explore_phase1 import contact_rows       # noqa: E402

G = 5.6                                       # 800 u/s^2 x 7 ms


def _hist(vz, og=None):
    h = np.zeros((len(vz), 1), STATE_DTYPE)
    h["velocity"][:, 0, 2] = vz
    h["onground"][:, 0] = -1 if og is None else og
    return h


def test_free_fall_wall_ceiling_and_clamp_are_not_contacts():
    vz = [100.0 - G * k for k in range(6)]                  # exact free fall
    assert not contact_rows(_hist(vz), np.zeros((6, 1), np.int32), G, 3500.0).any()
    vz = [300.0, 0.0, -G]                                    # a ceiling bonk pushes down
    assert not contact_rows(_hist(vz), np.zeros((3, 1), np.int32), G, 3500.0)[0, 0]
    vz = [-3500.0, -3500.0, -3500.0]                         # pinned at the clamp
    assert not contact_rows(_hist(vz), np.zeros((3, 1), np.int32), G, 3500.0)[:2].any()


def test_a_ramp_push_and_the_ground_are_contacts():
    vz = [-200.0, -200.0 - G + 40.0, -150.0]                 # the map gives vz back
    c = contact_rows(_hist(vz), np.zeros((3, 1), np.int32), G, 3500.0)
    assert c[0, 0] and not c[2, 0]                           # the last tick has no successor
    c = contact_rows(_hist([0.0, 0.0], og=[3, -1]), np.zeros((2, 1), np.int32), G, 3500.0)
    assert c[0, 0] and not c[1, 0]


def test_a_tick_whose_successor_is_another_run_reads_its_ground_flag_alone():
    vz = [-200.0, 100.0]                                     # a respawn, not a push
    rid = np.array([[0], [1]], np.int32)
    assert not contact_rows(_hist(vz), rid, G, 3500.0)[0, 0]
    rid = np.array([[0], [-1]], np.int32)
    assert not contact_rows(_hist(vz), rid, G, 3500.0)[0, 0]
