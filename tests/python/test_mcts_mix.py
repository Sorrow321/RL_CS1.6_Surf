"""tools/mcts_search.py's child-mix rule (--mcts-mix-rule, azr1b2): the legacy fixed knob kept
draw for draw (flag off = the same random stream as runs 4/5), the sibling tournament, the
probability-matching shares with their floor, and the tree's per-node kind. No simulator, no
torch."""
import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import mcts_search  # noqa: E402
from mcts_search import (MODE_GREEDY, MODE_PERTURB, MODE_POLICY, MODE_RANDOM,  # noqa: E402
                         MixRule, Tree, kinds_from_masks, legacy_child_kinds, mix_shares,
                         sibling_credit)


def _old_inline(rng, rank, pf, rf):
    """The knob exactly as runs 4/5 ran it (mcts_search.py at f5421ec9, run_mcts)."""
    N = len(rank)
    explore = (rank > 0) & (rng.random(N) < float(pf))
    rnd = (rank > 0) & (rng.random(N) < float(rf))
    explore &= ~rnd
    return explore, rnd


def test_legacy_draw_is_the_old_draw_and_consumes_the_same_stream():
    for seed, pf, rf in ((0, 0.5, 0.5), (7, 0.0, 1.0), (11, 0.5, 0.0), (3, 0.25, 0.75)):
        rank = np.arange(2048) // 128
        r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)
        e1, n1 = legacy_child_kinds(r1, rank, pf, rf)
        e2, n2 = _old_inline(r2, rank, pf, rf)
        assert (e1 == e2).all() and (n1 == n2).all()
        # the stream after the draw is the same (the rest of the wave draws the same numbers)
        assert np.array_equal(r1.random(16), r2.random(16))


def test_log_mode_draws_the_legacy_knob():
    rank = np.arange(2048) // 128
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    mr = MixRule("log", perturb_frac=0.0, random_frac=1.0)
    k = mr.draw(r1, rank)
    e, n = _old_inline(r2, rank, 0.0, 1.0)
    assert (k == kinds_from_masks(rank, e, n)).all()
    assert (k[rank == 0] == MODE_GREEDY).all() and (k[rank > 0] == MODE_RANDOM).all()
    assert np.array_equal(r1.random(4), r2.random(4))
    # 'log' never moves the shares (the knob stays fixed); it only keeps the books
    mr.update(k, np.where(k == MODE_RANDOM, 1.0, 0.0), np.ones(len(k), np.int8),
              np.ones(len(k), bool), np.zeros(len(k)))
    assert np.allclose(mr.share, 1 / 3)
    assert mr.tot_wins[MODE_RANDOM] > 0


def test_off_path_in_run_mcts_is_the_legacy_call():
    src = inspect.getsource(mcts_search.run_mcts)
    assert "if mix is None:" in src
    assert ("legacy_child_kinds(rng, rank, float(a.mcts_perturb_frac),\n"
            "                                              float(a.mcts_random_frac))") in src
    # no random draw was added anywhere else in the search: the draws outside the knob are the
    # steer mixture (uniform + normal, or normal) and the hold coin, as at f5421ec9
    body = src.replace("legacy_child_kinds(rng", "").replace("mix.draw(rng", "")
    assert body.count("rng.") == 4
    assert body.count("rng.random(N)") == 1 and body.count("rng.uniform(") == 1
    assert body.count("rng.normal(") == 2


def test_mix_shares_probability_matching_and_floor():
    fl = 0.05
    s = mix_shares([10, 10, 10], [100, 100, 100], fl)
    assert np.allclose(s, 1 / 3)
    s = mix_shares([0, 0, 30], [100, 100, 100], fl)        # one kind wins everything
    assert np.allclose(s, [fl, fl, 1 - 2 * fl])
    s = mix_shares([10, 20, 0], [100, 50, 100], fl)        # yields 0.1, 0.4, 0
    assert np.isclose(s.sum(), 1.0) and (s >= fl - 1e-12).all()
    assert np.isclose(s[1] - fl, 4 * (s[0] - fl))
    prev = np.array([0.2, 0.3, 0.5])
    assert np.allclose(mix_shares([0, 0, 0], [5, 5, 5], fl, prev=prev), prev)
    assert np.allclose(mix_shares([0, 0, 0], [0, 0, 0], fl), 1 / 3)
    # per CHILD, not per kind: a kind with 9x the children and 9x the wins is not favoured
    s = mix_shares([9, 1, 1], [90, 10, 10], 0.0)
    assert np.allclose(s, 1 / 3)


def test_sibling_credit_value_then_cells():
    # leaf 7: three children, child 2 beats the parent's value -> it alone wins (on value)
    # leaf 9: nobody beats the parent; the live child with the most new cells wins (on cells)
    # leaf 4: nobody beats the parent and only a DEAD child opened cells -> no win
    own = np.array([7, 7, 7, 9, 9, 9, 4, 4])
    sc = np.array([0.5, 0.5, 0.6, 0.3, 0.3, 0.3, 0.2, 0.2])
    par = np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.2, 0.2])
    alive = np.array([1, 1, 1, 1, 1, 0, 1, 0], bool)
    nn = np.array([5, 0, 0, 1, 3, 9, 0, 4], float)
    cr, how = sibling_credit(own, sc, par, alive, nn)
    assert cr.tolist() == [0, 0, 1, 0, 1, 0, 0, 0]
    assert how.tolist() == [0, 0, 1, 0, 2, 0, 0, 0]


def test_sibling_credit_ties_split_and_dead_bankers_count():
    own = np.array([1, 1, 1, 1])
    sc = np.array([0.4, 0.7, 0.7, 0.1])
    par = np.full(4, 0.4)
    alive = np.array([1, 0, 1, 1], bool)          # child 1 died AFTER banking 0.7: it counts
    cr, how = sibling_credit(own, sc, par, alive, np.zeros(4))
    assert np.allclose(cr, [0, 0.5, 0.5, 0]) and how.tolist() == [0, 1, 1, 0]
    # a finish (score 2.0) beats everything
    cr, _ = sibling_credit(own, np.array([0.4, 2.0, 0.9, 0.1]), par, alive, np.zeros(4))
    assert cr.tolist() == [0, 1, 0, 0]
    # tiny float noise is not an improvement
    cr, how = sibling_credit(own, par + 1e-13, par, alive, np.array([0, 0, 2, 2.0]))
    assert np.allclose(cr, [0, 0, 0.5, 0.5]) and (how[2:] == 2).all()


def test_rule_moves_toward_the_winning_kind_and_keeps_the_floor():
    rng = np.random.default_rng(0)
    rank = np.arange(2048) // 128
    mr = MixRule("sib", floor=0.05, horizon=4, warm=1)
    for _ in range(40):
        k = mr.draw(rng, rank)
        assert (k[rank == 0] == MODE_GREEDY).all() and (k[rank > 0] > 0).all()
        # every R child wins a point, nobody else
        cr = np.where(k == MODE_RANDOM, 1.0, 0.0)
        mr.update(k, cr, np.where(cr > 0, 1, 0).astype(np.int8), np.ones(len(k), bool),
                  np.zeros(len(k)))
    assert np.allclose(mr.share, [0.05, 0.05, 0.9])
    # then the policy samples win: the share follows within a few horizons, R keeps its floor
    for _ in range(60):
        k = mr.draw(rng, rank)
        cr = np.where(k == MODE_POLICY, 1.0, 0.0)
        mr.update(k, cr, np.where(cr > 0, 1, 0).astype(np.int8), np.ones(len(k), bool),
                  np.zeros(len(k)))
    assert mr.share[0] > 0.89 and np.isclose(mr.share[2], 0.05, atol=1e-3)
    # the empirical draw follows the shares
    k = mr.draw(np.random.default_rng(1), np.ones(200000, np.int64))
    frac = np.array([(k == m).mean() for m in (MODE_POLICY, MODE_PERTURB, MODE_RANDOM)])
    assert np.allclose(frac, mr.share, atol=0.01)
    s = mr.summary("GRRP")
    assert s["best_path_kinds"] == "GRRP" and s["mix_rule"] == "sib"
    assert set(s["mix_share_mean"]) == {"P", "E", "R"} and len(s["mix_share_by_wave"]) <= 61


def test_warm_waves_hold_equal_shares():
    mr = MixRule("sib", warm=3)
    k = np.array([0, 1, 2, 3, 3, 3], np.int8)
    for i in range(2):
        mr.update(k, np.array([0, 0, 0, 1, 0, 0.0]), np.array([0, 0, 0, 1, 0, 0], np.int8),
                  np.ones(6, bool), np.zeros(6))
        assert np.allclose(mr.share, 1 / 3)
    mr.update(k, np.array([0, 0, 0, 1, 0, 0.0]), np.array([0, 0, 0, 1, 0, 0], np.int8),
              np.ones(6, bool), np.zeros(6))
    assert mr.share[2] > 0.8


def test_tree_kind_field():
    t = Tree(cap=4)
    r = t.add_children(0, [0.1, 0.2], [True, True], slot=[0, 1])
    assert (t.kind[r] == -1).all()                     # roots: no kind
    kids = t.add_children(int(r[0]), [0.3, 0.4, 0.5], [True, True, True],
                          kind=np.array([MODE_GREEDY, MODE_PERTURB, MODE_RANDOM], np.int8))
    assert t.kind[kids].tolist() == [0, 2, 3]
    for _ in range(4):                                   # growth keeps the field
        leaf = int(np.flatnonzero(t.alive[:t.n] & (t.ncld[:t.n] == 0))[0])
        t.add_children(leaf, [0.1], [True], kind=np.array([MODE_POLICY], np.int8))
    assert t.kind[kids].tolist() == [0, 2, 3] and t._cap >= t.n
