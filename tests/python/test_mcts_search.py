"""tools/mcts_search.py's pure parts: the PUCT tree (batched selection under virtual visits,
max backup, exhaustion), the cell archive and the contact rule. No simulator, no torch."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

from mcts_search import CellArchive, RowStore, Tree, contact_mask  # noqa: E402


def _roots(t, n=3, scores=None):
    sc = np.asarray(scores if scores is not None else np.linspace(0.1, 0.3, n))
    return t.add_children(0, sc, np.ones(len(sc), bool), slot=np.arange(len(sc)),
                          tick=np.arange(len(sc)) * 10, bank_d=1000 - 100 * sc,
                          bank_t=np.arange(len(sc)) * 10)


def test_backup_is_max_and_counts_visits():
    t = Tree(cap=8)
    r = _roots(t, 3)
    assert t.q[0] == np.max(t.score[r])
    kids = t.add_children(int(r[0]), [0.5, 0.2], [True, True], tick=[20, 20])
    assert t.q[r[0]] == 0.5 and t.q[0] == 0.5
    assert t.visits[r[0]] == 1 + 2 and t.visits[0] == 3 + 2
    assert all(t.parent[k] == r[0] for k in kids)
    assert t.depth[kids[0]] == 2
    # growth past the initial capacity keeps every array consistent
    for _ in range(10):
        leaf = int(np.flatnonzero(t.alive[:t.n] & (t.ncld[:t.n] == 0))[0])
        t.add_children(leaf, [0.1, 0.1, 0.1], [True, True, True], tick=[1, 1, 1])
    assert t.n > 8 and t._cap >= t.n
    assert t.q[0] == 0.5


def test_select_distinct_pending_leaves_and_end_wave():
    t = Tree()
    _roots(t, 4)
    leaves = t.select(10, c_puct=1.0)
    assert len(leaves) == 4                     # only 4 expandable leaves exist
    assert len(set(leaves)) == 4
    assert t.pending[leaves].all()
    assert t.select(1) == []                    # everything pending
    t.end_wave()
    assert not t.pending[:t.n].any() and not t.vloss[:t.n].any()
    assert len(t.select(2)) == 2


def test_select_prefers_high_q_but_explores():
    t = Tree()
    r = _roots(t, 2, scores=[0.0, 1.0])
    for leaf in (int(r[0]), int(r[1])):
        t.add_children(leaf, [t.score[leaf]] * 4, [True] * 4)
    picks = t.select(6, c_puct=1.0)
    par = [int(t.parent[p]) for p in picks]
    assert par.count(int(r[1])) >= par.count(int(r[0]))   # the deeper subtree gets more
    assert len(set(picks)) == len(picks)


def test_dead_children_exhaust_the_subtree():
    t = Tree()
    r = _roots(t, 2)
    t.add_children(int(r[0]), [0.9, 0.9], [False, False])     # both died
    assert t.exhausted[r[0]]
    assert not t.exhausted[0]
    leaves = t.select(5)
    assert leaves == [int(r[1])]
    t.end_wave()
    t.add_children(int(r[1]), [0.1], [False])
    assert t.exhausted[0]
    assert t.select(3) == []


def test_finished_node_wins_best_node():
    t = Tree()
    r = _roots(t, 2)
    kids = t.add_children(int(r[1]), [0.95, 2.0], [True, True], finished=[False, True],
                          tick=[400, 380], bank_t=[390, 380])
    assert t.best_node() == int(kids[1])
    assert t.finished[kids[1]] and not t.alive[kids[1]] and t.exhausted[kids[1]]
    assert t.path_to(int(kids[1])) == [int(r[1]), int(kids[1])]


def test_best_node_ties_take_the_earliest_bank():
    t = Tree()
    r = _roots(t, 1, scores=[0.2])
    k = t.add_children(int(r[0]), [0.6, 0.6], [True, True], tick=[50, 50], bank_t=[45, 30])
    assert t.best_node() == int(k[1])


def test_cell_weight_steers_exploration():
    t = Tree()
    r = _roots(t, 2, scores=[0.5, 0.5])
    t.cell[r] = [0, 1]
    seen = np.array([100.0, 0.0])
    picks = t.select(1, c_puct=10.0, cell_w=lambda ids: 1.0 / np.sqrt(1.0 + seen[ids]))
    assert picks == [int(r[1])]


def test_cell_archive_counts_and_new():
    a = CellArchive(cell=100.0, ecell=1000.0)
    o = np.array([[0, 0, 0], [50, 50, 50], [150, 0, 0]], np.float64)
    v = np.zeros((3, 3))
    ids, new = a.visit(a.keys(o, v))
    assert new == 2 and ids[0] == ids[1] and ids[2] != ids[0]
    assert a.seen[ids[0]] == 2
    w = a.weight(ids)
    assert np.isclose(w[0], 1 / np.sqrt(3)) and np.isclose(w[2], 1 / np.sqrt(2))
    # energy splits a position cell
    ids2, new2 = a.visit(a.keys(o[:1], np.array([[2000.0, 0, 0]])))
    assert new2 == 1 and ids2[0] not in ids


def test_contact_rule():
    g = -8.0
    vz0 = np.array([0.0, 0.0, -3992.0, -100.0])
    vz1 = np.array([-8.0, 0.0, -4000.0, -4000.0])
    c = contact_mask(vz0, vz1, g, 1.0, 4000.0)
    # free fall: no contact; vz held at 0 (a floor/ramp): contact; capped fall (-3992 -> -4000 is
    # -8 = free fall); a clamp from -100 to -4000 is not free fall: contact
    assert c.tolist() == [False, True, False, True]
    c2 = contact_mask(np.array([-4000.0]), np.array([-4000.0]), g, 1.0, 4000.0)
    assert c2.tolist() == [False]            # pinned at the cap: free fall


def test_row_store_grows():
    from surfgym.core import STATE_DTYPE
    st = RowStore(STATE_DTYPE, 5, cap=2)
    s = np.zeros(3, STATE_DTYPE)
    s["tick"] = [1, 2, 3]
    sl = st.add(s, np.ones((3, 5), np.float32), {"k": np.arange(3)[:, None] * np.ones((1, 2))})
    assert list(sl) == [0, 1, 2]
    S, O, SD = st.rows([2, 0])
    assert list(S["tick"]) == [3, 1] and SD["k"].shape == (2, 2) and SD["k"][0, 0] == 2


def test_a_pending_subtree_is_blocked_and_the_batch_spreads():
    t = Tree()
    r = _roots(t, 2, scores=[0.0, 1.0])
    # the deep root has 2 children, the shallow one 6: a batch of 8 must take all 8 leaves
    t.add_children(int(r[1]), [1.0, 1.0], [True, True])
    t.add_children(int(r[0]), [0.0] * 6, [True] * 6)
    picks = t.select(8, c_puct=0.01, max_fail=4)
    assert len(picks) == 8 and len(set(picks)) == 8
    assert t.blocked[r[1]]
    t.end_wave()
    assert not t.blocked[:t.n].any()


def test_novelty_value_is_new_cells_per_visit_and_steers():
    t = Tree()
    r = _roots(t, 2, scores=[0.5, 0.5])
    t.add_children(int(r[0]), [0.5, 0.5], [True, True], nov=[0.0, 0.0])
    t.add_children(int(r[1]), [0.5, 0.5], [True, True], nov=[1.0, 0.5])
    assert np.isclose(t.nov[r[1]], 1.5) and np.isclose(t.nov[0], 1.5)
    picks = t.select(1, c_puct=0.0, novelty=1.0)
    assert int(t.parent[picks[0]]) == int(r[1])
    t.end_wave()
    picks0 = t.select(1, c_puct=0.0, novelty=0.0)
    assert len(picks0) == 1


def test_banked_min_d_ignores_a_fatal_dive():
    sys.path.insert(0, str(ROOT / "python"))
    from train_fast import banked_min_d
    # a ramp ride (vz held at 0: contact) to d 500, then a free fall (vz -8 per tick) to d 100
    d = np.array([1000.0, 800.0, 600.0, 500.0, 400.0, 250.0, 100.0])
    vz = np.array([0.0, 0.0, 0.0, 0.0, -8.0, -16.0, -24.0])
    assert banked_min_d(d, vz, -8.0, 1.0, 4000.0) == 500.0
    # a fall pinned at the speed cap is still free flight
    vz2 = np.array([0.0, 0.0, 0.0, 0.0, -4000.0, -4000.0, -4000.0])
    assert banked_min_d(d, vz2, -8.0, 1.0, 4000.0) == 400.0   # the clamp tick itself is contact
    # no contact at all: nothing banked beyond the first row
    assert banked_min_d(d[:3], np.array([0.0, -8.0, -16.0]), -8.0, 1.0, 4000.0) == 1000.0
