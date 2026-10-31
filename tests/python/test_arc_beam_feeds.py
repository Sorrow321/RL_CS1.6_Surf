"""A --race-arc checkpoint WITHOUT --obs-reward has no arc state in its inference (the arc
reaches the policy only through --obs-reward's slot 12), so beam_tas / plan_to_bc may load it
(zlr1b2, 2026-10-31); with --obs-reward both still refuse."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def test_make_eval_feeds_allows_arc_without_obs_reward():
    from surfgym.bc import make_eval_feeds
    slot, feed, latch = make_eval_feeds({"race_arc": "x.npz"}, None, 6000.0, 4)
    assert slot == -1 and feed is None and latch is None
    with pytest.raises(SystemExit):
        make_eval_feeds({"race_arc": "x.npz", "obs_reward": 1}, None, 6000.0, 4)


def test_beam_tas_refuses_arc_only_with_obs_reward():
    src = (ROOT / "tools" / "beam_tas.py").read_text(encoding="utf-8")
    assert 'not (k == "race_arc" and not cfg.get("obs_reward"))' in src
