"""mm/health (2026-11-22): --race-pay-d0 (plain-unit progress pay) and --priv-critic 2.

* --race-pay-d0 D: one unit of progress pays 100/D on every map, and the ratchet column is
  (d - b)/D; the reward object is today's RaceReward handed that scale and that normaliser, so
  the test is that the two numbers it reports are D-based and not d0-based;
* --priv-critic 2: the privileged block keeps its width, column 6 is d/D and the other nine
  are 0 whatever the state; mode "full" is today's block, column for column;
* the recorder and the search tools mention the key (record_ckpt's static audit).
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.privfeat import PRIV_DIM, PRIV_FEATURES, PRIV_FEATURES_DIST, PrivFeat  # noqa: E402


def _rows(n=5, seed=0):
    g = np.random.default_rng(seed)
    pos = g.uniform(-4000, 4000, (n, 3))
    vel = g.uniform(-3000, 3000, (n, 3))
    d = g.uniform(0, 200000, n)
    tick = g.integers(0, 30000, n)
    return pos, vel, d, tick


def test_full_mode_is_todays_block():
    pos, vel, d, tick = _rows()
    a = PrivFeat([10.0, -20.0, 30.0], 5000.0, 99700.0, 30000)
    b = PrivFeat([10.0, -20.0, 30.0], 5000.0, 99700.0, 30000, mode="full")
    oa = np.zeros((5, PRIV_DIM), np.float32)
    ob = np.full((5, PRIV_DIM), 7.0, np.float32)
    a.fill(oa, pos, vel, d, tick)
    b.fill(ob, pos, vel, d, tick)
    assert np.array_equal(oa, ob)
    assert a.names == PRIV_FEATURES == b.names
    assert np.allclose(oa[:, 6], d / 99700.0)


def test_dist_mode_one_live_column():
    pos, vel, d, tick = _rows()
    p = PrivFeat([10.0, -20.0, 30.0], 5000.0, 1e5, 30000, mode="dist")
    out = np.full((5, PRIV_DIM), 7.0, np.float32)       # stale values must not survive
    p.fill(out, pos, vel, d, tick, latch=np.ones(5, bool))
    assert np.allclose(out[:, 6], (d / 1e5).astype(np.float32))
    assert np.all(np.delete(out, 6, axis=1) == 0.0)
    assert p.names == PRIV_FEATURES_DIST and len(p.names) == PRIV_DIM
    assert p.names[6] == "d_abs" and set(p.names) - {"d_abs"} == {"zero"}
    # the same state on another map (another centre, scale, ep length) reads the same block
    q = PrivFeat([-900.0, 4.0, 0.0], 12000.0, 1e5, 15652, mode="dist")
    out2 = np.zeros((5, PRIV_DIM), np.float32)
    q.fill(out2, pos, vel, d, tick)
    assert np.array_equal(out, out2)
    assert "d / 100,000u" in p.describe()


def test_bad_mode_refused():
    with pytest.raises(ValueError):
        PrivFeat([0, 0, 0], 1.0, 1.0, 1, mode="pos")


def test_race_reward_scale_and_ratchet_column():
    from surfgym.rewards import RaceReward

    class _Field:
        def sample(self, p):
            return np.asarray(p, np.float64)[:, 0]

    for d0_map in (2673.0, 198991.0):
        # what train_fast hands RaceReward for this map: today vs --race-pay-d0 1e5
        for pay_d0 in (0.0, 1e5):
            D = pay_d0 or d0_map
            r = RaceReward(_Field(), scale=100.0 / D, time_pen=0.0, success_bonus=0.0,
                           ratchet=True, ratchet_d0=D)
            assert r.scale == pytest.approx(100.0 / D)
            assert r.ratchet_d0 == pytest.approx(D)
            # a gap of 500 u behind the record reads 500 / D
            got = r.ratchet_gap_of(np.array([1500.0]), np.array([1000.0]))
            assert got[0] == pytest.approx(500.0 / D, rel=1e-6)


def _literals(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


@pytest.mark.parametrize("rel", ["tools/record_ckpt.py", "tools/beam_tas.py",
                                 "tools/az_mcts.py"])
def test_tools_mirror_the_key(rel):
    assert "race_pay_d0" in _literals(ROOT / rel)


def test_trainer_flag_and_config_key():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--race-pay-d0"' in src
    assert 'meta["config"]["race_pay_d0"]' in src
    # every pay / ratchet site goes through the one helper; no 100/rf_d0 scale is left
    assert "100.0 / _s.rf_d0" not in src
    assert "ratchet_d0=_s.rf_d0" not in src
    assert src.count("_pay_d0(_s)") >= 5
