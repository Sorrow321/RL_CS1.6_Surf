"""explore_phase1 --no-brake: which held keys brake at a view offset from the velocity (CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import explore_phase1 as ep  # noqa: E402


def test_key_angle_matches_the_core_encoding():
    # src/env.c: fwd 0 = back, 2 = forward; side 0 = left (smove -400), 2 = right (+400);
    # pm.c's right vector is the view yaw - 90 deg
    assert ep.key_angle(1, 1) is None
    assert ep.key_angle(2, 1) == 0.0
    assert abs(abs(ep.key_angle(0, 1)) - 180.0) < 1e-9
    assert ep.key_angle(1, 0) == 90.0
    assert ep.key_angle(1, 2) == -90.0
    assert ep.key_angle(2, 0) == 45.0
    assert ep.key_angle(0, 2) == -135.0


def test_no_brake_table():
    ok = ep.no_brake_table(np.array([-90.0, 0.0, 90.0]))
    names = {(f, s): (("S", "", "W")[f] + ("A", "", "D")[s]) or "-" for f in range(3)
             for s in range(3)}
    allowed = [sorted(names[(f, s)] for f in range(3) for s in range(3) if ok[i, f, s])
               for i in range(3)]
    # view along the velocity: anything without a backward component (incl. the A/D strafe)
    assert allowed[1] == sorted(["-", "W", "WA", "WD", "A", "D"])
    # view 90 deg left of the velocity: S and W are now the perpendicular strafes, D is forward
    assert allowed[2] == sorted(["-", "W", "S", "D", "WD", "SD"])
    assert allowed[0] == sorted(["-", "W", "S", "A", "WA", "SA"])
    assert ok.reshape(3, 9).sum(1).tolist() == [6, 6, 6]
