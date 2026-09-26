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


@pytest.mark.parametrize("shaping", ["refund_i", "pbrs"])
@pytest.mark.parametrize("durations", [None, (1.0, 0.5, 1.7, 0.8)])
def test_fail_leaf_makes_an_unfinished_path_neutral(shaping, durations):
    """--plan-mcts-leaf fail (Codex's review): live edges then a fail-now leaf back up to exactly
    the root's own failed-end accounting (0 from a fresh root) - for a per-primitive and a
    duration-based discount alike."""
    from surfgym.goalsearch import _Edge, fail_now_value
    rng = np.random.default_rng(3)
    for _ in range(20):
        progs = rng.normal(0.0, 0.8, size=4)
        bank0 = 0.0
        edges, bank = [], bank0
        for k, p in enumerate(progs):
            g = G if durations is None else G ** durations[k]
            r, nb = edge_reward(shaping, FB, PP, np.array([p]), np.array([bank]), np.array([False]),
                                np.array([False]), g)
            e = _Edge(np.zeros(6), float(r[0]), 0.0, False, False, {"origin": np.zeros(3)},
                      None, None, float(nb[0]), 100)
            e.disc = g
            edges.append(e)
            bank = float(nb[0])
        edges[-1].v = float(fail_now_value(shaping, PP, np.array([bank]))[0])
        for parent, child in zip(edges[:-1], edges[1:]):
            parent.child = [child]
        assert edges[0].q(G) == pytest.approx(-PP * bank0, abs=1e-9)


def test_fail_leaf_under_refund_and_plain():
    from surfgym.goalsearch import fail_now_value
    assert fail_now_value("refund", 1.0, np.array([-2.0, 3.0])).tolist() == [0.0, -3.0]
    assert fail_now_value("plain", 1.0, np.array([-2.0, 3.0])).tolist() == [0.0, 0.0]


def test_path_cover_pays_a_cell_once_along_a_path():
    """--plan-mcts-cover: a cell already on the path earns nothing (a loop through one rare cell
    cannot be farmed by a max backup); a new cell earns c / sqrt(1 + N)."""
    from surfgym.goalsearch import path_cover
    counts = np.zeros((4, 4, 4))
    counts[1, 1, 1] = 3.0
    path = frozenset({(0, 0, 0)})
    cv, sup = path_cover(path, [(1, 1, 1), (0, 0, 0), (2, 2, 2)], counts, 0.5)
    assert cv.tolist() == pytest.approx([0.25, 0.0, 0.5]) and sup == 1
    loop = path | {(1, 1, 1)}                 # the child that went to (1,1,1) carries it on
    cv2, sup2 = path_cover(loop, [(1, 1, 1), (0, 0, 0)], counts, 0.5)
    assert cv2.tolist() == [0.0, 0.0] and sup2 == 2
