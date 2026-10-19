"""RLSURF_POLICY: the hard rental policy the autoresearch fleet runs under (user, 2026-10-20).

One GPU, a card named in config.max_dph, at or below its $/h cap, and no more than MAX_BOXES
live - enforced on EVERY instance on the account, whoever rented it. A breach is destroyed on
its second consecutive sighting, never the first (a single observation is not evidence), and
the policy is off unless the variable is set.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import fleet_watchdog as fw  # noqa: E402

POLICY = {"max_dph": {"RTX_3090": 0.30, "RTX_4090": 0.45, "RTX_5090": 0.60}}


@pytest.fixture
def sim(monkeypatch):
    state = {"live": [], "reg": {}, "destroyed": [], "logs": []}
    monkeypatch.setattr(fw, "instances", lambda: state["live"])
    monkeypatch.setattr(fw, "load_reg", lambda: state["reg"])
    monkeypatch.setattr(fw, "save_reg", lambda d: state.__setitem__("reg", d))
    monkeypatch.setattr(fw, "_merge_reg", lambda *a, **k: None)
    monkeypatch.setattr(fw, "log", lambda m: state["logs"].append(m))
    monkeypatch.setattr(fw, "_harvest_tick", lambda *a, **k: None)
    monkeypatch.setattr(fw, "destroy", lambda iid, why: state["destroyed"].append((iid, why)) or 1)
    monkeypatch.setattr(fw, "POLICY", POLICY)
    monkeypatch.setattr(fw, "_POLICY_SEEN", {})
    now = 1.9e9
    monkeypatch.setattr(fw, "now", lambda: now)
    state["now"] = now
    return state


def _inst(iid, gpu, dph, n=1, age=600.0, now=1.9e9):
    return {"id": iid, "gpu_name": gpu, "num_gpus": n, "dph_total": dph,
            "actual_status": "running", "start_date": now - age}


def _reg(sim, *ids):
    for i in ids:
        sim["reg"][str(i)] = {"deadline": sim["now"] + 3600, "label": f"a{i}", "ready": True}


def test_the_policy_reads_card_count_and_price():
    fw_policy = fw.POLICY
    try:
        fw.POLICY = POLICY
        assert fw.policy_breach(_inst(1, "RTX 3090", 0.25)) is None
        assert "not an allowed card" in fw.policy_breach(_inst(2, "H100 SXM", 0.25))
        assert "2 GPUs" in fw.policy_breach(_inst(3, "RTX 3090", 0.40, n=2))
        assert "above its 0.30 cap" in fw.policy_breach(_inst(4, "RTX 3090", 0.36))
        assert fw.policy_breach(_inst(5, "RTX 4090", 0.452)) is None, "a half-cent of slack"
        fw.POLICY = None
        assert fw.policy_breach(_inst(6, "H100 SXM", 9.0)) is None, "off without the variable"
    finally:
        fw.POLICY = fw_policy


def test_a_breach_dies_on_the_second_sighting_not_the_first(sim):
    sim["live"] = [_inst(1, "RTX 3090", 0.25), _inst(2, "A100 SXM4", 0.80)]
    _reg(sim, 1, 2)
    fw.sweep()
    assert sim["destroyed"] == [], "one observation is not evidence"
    fw.sweep()
    assert [d[0] for d in sim["destroyed"]] == ["2"]
    assert "not an allowed card" in sim["destroyed"][0][1]


def test_the_box_cap_takes_the_youngest(sim, monkeypatch):
    monkeypatch.setattr(fw, "MAX_BOXES", 2)
    sim["live"] = [_inst(1, "RTX 3090", 0.2, age=3000), _inst(2, "RTX 3090", 0.2, age=2000),
                   _inst(3, "RTX 3090", 0.2, age=100)]
    _reg(sim, 1, 2, 3)
    fw.sweep()
    fw.sweep()
    assert [d[0] for d in sim["destroyed"]] == ["3"]
    assert "cap" in sim["destroyed"][0][1]


def test_a_clean_fleet_is_untouched(sim):
    sim["live"] = [_inst(1, "RTX 3090", 0.29), _inst(2, "RTX 5090", 0.60)]
    _reg(sim, 1, 2)
    for _ in range(3):
        fw.sweep()
    assert sim["destroyed"] == []
