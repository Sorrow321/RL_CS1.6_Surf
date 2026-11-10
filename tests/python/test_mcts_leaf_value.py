"""--mcts-leaf-value (azr1b4): the network's value at the leaves guides the tree's SELECTION
(mcts_search.Tree.qv), while q - the real banked max that every label, value target and the best
line read - is untouched. Off (use_qv False, the default) a qv passed by the caller is never read.
No simulator, no torch."""
import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import mcts_search  # noqa: E402
from mcts_search import Tree  # noqa: E402


def _grow(t, qv_roots=None, qv_kids=None):
    """Two roots, the first expanded into three children; ``qv_*`` passed only when given."""
    kw = {} if qv_roots is None else {"qv": np.asarray(qv_roots, np.float64)}
    r = t.add_children(0, [0.10, 0.10], [True, True], tick=[0, 0], **kw)
    kw = {} if qv_kids is None else {"qv": np.asarray(qv_kids, np.float64)}
    k = t.add_children(int(r[0]), [0.10, 0.12, 0.11], [True, True, True], tick=[1, 1, 1], **kw)
    return r, k


def test_off_a_passed_qv_is_never_read():
    a, b = Tree(), Tree()
    _grow(a)
    _grow(b, qv_roots=[0.9, 0.0], qv_kids=[0.0, 0.0, 0.95])
    for name in ("q", "score", "visits", "nov", "exhausted"):
        assert np.array_equal(getattr(a, name)[:a.n], getattr(b, name)[:b.n]), name
    assert (a.qlo, a.qhi) == (b.qlo, b.qhi)
    for _ in range(3):
        assert a.select(2, c_puct=0.25) == b.select(2, c_puct=0.25)
        a.end_wave()
        b.end_wave()


def test_on_an_optimistic_leaf_is_chosen_then_falls_to_what_its_children_show():
    t = Tree()
    t.use_qv = True
    r = t.add_children(0, [0.10, 0.10], [True, True], tick=[0, 0], qv=np.array([0.10, 0.50]))
    assert t.qv[0] == 0.50
    # the second root's guess wins the descent although both banked the same
    assert t.select(1, c_puct=0.25) == [int(r[1])]
    t.end_wave()
    # its children show nothing better: its selection value falls back, its real q never moved
    k = t.add_children(int(r[1]), [0.10, 0.10], [True, True], tick=[1, 1],
                       qv=np.array([0.10, 0.10]))
    assert t.qv[r[1]] == 0.10 and t.qv[0] == 0.10
    assert t.q[r[1]] == 0.10 and t.q[0] == 0.10
    assert len(k) == 2
    # with the guess gone, the less visited root gets the next descent again
    assert t.select(1, c_puct=0.25) == [int(r[0])]


def test_on_a_guess_never_lowers_a_banked_score_and_deep_recompute_walks_up():
    t = Tree()
    t.use_qv = True
    r = t.add_children(0, [0.2, 0.1], [True, True], tick=[0, 0], qv=np.array([0.0, 0.1]))
    assert t.qv[r[0]] == 0.2                     # max(score, guess)
    k = t.add_children(int(r[0]), [0.2, 0.2], [True, True], tick=[1, 1],
                       qv=np.array([0.6, 0.3]))
    assert t.qv[r[0]] == 0.6 and t.qv[0] == 0.6
    assert t.q[r[0]] == 0.2                      # the real max is the banked one
    # the optimistic child expands into children that bank a little more than it had
    g = t.add_children(int(k[0]), [0.25, 0.22], [True, True], tick=[2, 2],
                       qv=np.array([0.25, 0.22]))
    assert t.qv[k[0]] == 0.25
    assert t.qv[r[0]] == 0.3                     # its sibling's guess is now the best below
    assert t.qv[0] == 0.3
    assert t.q[k[0]] == 0.25 and t.q[0] == 0.25  # q only ever rises, on real banking
    assert len(g) == 2
    # the range the selection normalises with covers the guesses
    assert t.qhi >= 0.6


def test_on_dead_children_keep_their_score():
    t = Tree()
    t.use_qv = True
    r = t.add_children(0, [0.1], [True], tick=[0])
    t.add_children(int(r[0]), [0.1, 0.1], [False, True], tick=[1, 1],
                   qv=np.array([0.1, 0.4]))
    assert t.qv[r[0]] == 0.4


def test_run_mcts_reads_the_flag_and_off_passes_no_qv():
    src = inspect.getsource(mcts_search.run_mcts)
    assert 'LEAFV = bool(int(getattr(a, "mcts_leaf_value", 0) or 0))' in src
    assert "tree.use_qv = LEAFV" in src
    # off: the value read never runs and no qv reaches add_children
    assert "        _kv = {}\n        if LEAFV:" in src


# ---- the labels of sampled and perturbed children (used once --mcts-mix-rule draws them) -------
def test_exec_view_z_is_the_flown_view_for_the_masked_rows_only():
    from surfgym.view import view_from_z_abs
    rng = np.random.default_rng(0)
    pz = rng.normal(0.0, 0.5, (6, 2)).astype(np.float32)
    view = view_from_z_abs(pz, "velocity")
    mask = np.array([False, True, False, True, True, False])
    delta = np.array([0.0, 12.0, 0.0, -7.5, 3.0, 0.0])
    flown = view.copy()
    flown[:, 0] = ((flown[:, 0] + delta + 180.0) % 360.0) - 180.0
    z = mcts_search.exec_view_z(pz, flown, mask, "velocity", 1.33)
    assert np.array_equal(z[~mask], pz[~mask])               # every other child: its own draw
    back = view_from_z_abs(z, "velocity")
    assert np.allclose(back[mask, 0], flown[mask, 0], atol=1e-3)   # the yaw it FLEW
    assert np.allclose(back[mask, 1], flown[mask, 1], atol=1e-3)   # pitch untouched
    # nothing masked: the draw itself, unchanged
    assert mcts_search.exec_view_z(pz, flown, np.zeros(6, bool), "velocity", 1.33) is not None
    assert np.array_equal(mcts_search.exec_view_z(pz, flown, np.zeros(6, bool), "velocity",
                                                  1.33), pz)


def test_line_decisions_walks_the_prefix_then_each_macro():
    ndec = np.zeros(10, np.int64)
    kinds = np.zeros(10, np.int8)
    ndec[5], kinds[5] = 3, 1            # a random-held macro
    ndec[7], kinds[7] = 2, 2            # a sampled one (cut short: a finish or the replay's end)
    node_d, kind_d, m_d = mcts_search.line_decisions([5, 7], ndec, kinds, root=3, rdec=4,
                                                     r_first=2, n_dec=8)
    assert node_d == [3, 3, 3, 3, 5, 5, 5, 7]
    assert kind_d == [-1, -1, -2, -2, 1, 1, 1, 2]
    assert m_d == [0, 1, 2, 3, 0, 1, 2, 0]


def test_run_mcts_keeps_per_decision_draws_only_with_labels():
    src = inspect.getsource(mcts_search.run_mcts)
    assert "E_pact = np.zeros((tree._cap if LAB else 0, M, 6), np.int16)" in src
    assert "            if LAB:\n                # every env's policy-space draw" in src
    lab = inspect.getsource(mcts_search._tree_labels)
    assert "elif kind_d[d] == 2:" in lab and "E_pact[node_d[d], m_d[d]]" in lab
