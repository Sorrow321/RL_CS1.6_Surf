"""The search scores a simulated primitive with the planner's OWN --plan-shaping rule
(goalsearch.edge_reward), the rule the trainer pays (PrimLearnedPlanner.on_tick).

2026-09-26: every rule used to be scored as refund's -max(bank, 0) with a bank that carried no
interest, so under refund_i a detour that drove the bank negative and then failed looked like a
loss in the tree - the value trough refund_i exists to remove, re-created by the search."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from surfgym.goalsearch import SHAPINGS, edge_reward  # noqa: E402

G = 0.95
FB, PP = 10.0, 1.0


def one(shaping, prog, bank, how):
    fnd, died = how == "fin", how == "dead"
    r, nb = edge_reward(shaping, FB, PP, np.array([prog]), np.array([bank]),
                        np.array([fnd]), np.array([died]), G)
    return float(r[0]), float(nb[0])


def test_alive_pays_progress_and_grows_the_bank():
    for sh in ("plain", "refund"):
        assert one(sh, 0.4, -1.0, "alive") == pytest.approx((0.4, -0.6))
    r, nb = one("refund_i", 0.4, -1.0, "alive")
    assert r == pytest.approx(0.4) and nb == pytest.approx(-0.6 / G)
    r, nb = one("pbrs", 0.4, -1.0, "alive")
    assert r == pytest.approx(G * (-0.6) - (-1.0)) and nb == pytest.approx(-0.6)


def test_death_after_a_detour():
    # the bank is NEGATIVE (the primitives so far went away from the finish)
    assert one("plain", -0.3, -1.2, "dead")[0] == pytest.approx(-0.3)
    assert one("refund", -0.3, -1.2, "dead")[0] == pytest.approx(0.0)    # asymmetric
    assert one("refund_i", -0.3, -1.2, "dead")[0] == pytest.approx(1.2)  # symmetric refund
    assert one("pbrs", -0.3, -1.2, "dead")[0] == pytest.approx(1.2)


def test_death_after_progress():
    assert one("refund", 0.3, 0.8, "dead")[0] == pytest.approx(-0.8)
    assert one("refund_i", 0.3, 0.8, "dead")[0] == pytest.approx(-0.8)


def test_finish_keeps_its_progress_except_pbrs():
    for sh in ("plain", "refund", "refund_i"):
        assert one(sh, 0.5, 2.0, "fin")[0] == pytest.approx(FB + 0.5)
    assert one("pbrs", 0.5, 2.0, "fin")[0] == pytest.approx(FB - 2.0)


def test_refund_i_failed_episode_nets_zero_discounted():
    """A chain of primitives that ends in a death sums to exactly 0 when each reward is discounted
    by the primitives before it - forward, back or mixed."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        progs = rng.normal(0.0, 1.0, size=int(rng.integers(1, 8)))
        bank, ret, disc = 0.0, 0.0, 1.0
        for k, p in enumerate(progs):
            last = k == len(progs) - 1
            r, bank = one("refund_i", float(p), bank, "dead" if last else "alive")
            ret += disc * r
            disc *= G
        assert ret == pytest.approx(0.0, abs=1e-9)


def test_unknown_rule_refused():
    with pytest.raises(ValueError):
        one("nope", 0.1, 0.0, "alive")
    assert set(SHAPINGS) == {"plain", "refund", "refund_i", "pbrs"}
