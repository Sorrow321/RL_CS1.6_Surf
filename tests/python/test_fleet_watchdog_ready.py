"""The readiness kill must only ever fire on a box that NEVER came up.

2026-08-23: instance 48446220 had been training for 16 minutes when the
vast API reported it "offline" for a single poll. The sweep's readiness
rule keyed on the CURRENT status plus age-since-create, so it destroyed the
box as "never came up (status offline, age 28.1m)" and took the arm's
checkpoint, progress.csv and trajectories with it. A destroy is
unrecoverable; waiting for the deadline is merely expensive.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import fleet_watchdog as fw


@pytest.fixture
def sim(monkeypatch, tmp_path):
    """sweep() against an in-memory registry and a scripted instance list."""
    state = {"live": [], "reg": {}, "destroyed": [], "logs": []}
    monkeypatch.setattr(fw, "instances", lambda: state["live"])
    monkeypatch.setattr(fw, "load_reg", lambda: state["reg"])
    monkeypatch.setattr(fw, "save_reg", lambda d: state.__setitem__("reg", d))
    monkeypatch.setattr(fw, "log", lambda m: state["logs"].append(m))

    def fake_destroy(iid, why):
        state["destroyed"].append((iid, why))
        return 1

    monkeypatch.setattr(fw, "destroy", fake_destroy)
    return state


def _inst(iid, status, age_s, now):
    return {"id": iid, "actual_status": status, "start_date": now - age_s}


def test_a_box_that_came_up_survives_a_status_blip(sim, monkeypatch):
    now = 1.9e9
    monkeypatch.setattr(fw, "now", lambda: now)
    sim["reg"] = {"7": {"deadline": now + 3600, "label": "xVTGT"}}

    sim["live"] = [_inst(7, "running", 300.0, now)]
    assert fw.sweep() == 0
    assert sim["reg"]["7"]["ready"] is True          # latched

    # ...and now the API blips, well past READY_S
    sim["live"] = [_inst(7, "offline", 1686.0, now)]
    assert fw.sweep() == 0
    assert sim["destroyed"] == []
    assert any("leaving it to its deadline" in m for m in sim["logs"])


def test_a_box_that_never_came_up_is_still_destroyed(sim, monkeypatch):
    now = 1.9e9
    monkeypatch.setattr(fw, "now", lambda: now)
    sim["reg"] = {"9": {"deadline": now + 3600, "label": "xDEAD"}}
    sim["live"] = [_inst(9, "loading", fw.READY_S + 60.0, now)]
    assert fw.sweep() == 1
    assert sim["destroyed"][0][0] == "9"
    assert "never came up" in sim["destroyed"][0][1]


def test_the_deadline_still_wins_over_ready(sim, monkeypatch):
    now = 1.9e9
    monkeypatch.setattr(fw, "now", lambda: now)
    sim["reg"] = {"8": {"deadline": now - 1.0, "label": "xOLD", "ready": True}}
    sim["live"] = [_inst(8, "running", 5000.0, now)]
    assert fw.sweep() == 1
    assert "deadline passed" in sim["destroyed"][0][1]
