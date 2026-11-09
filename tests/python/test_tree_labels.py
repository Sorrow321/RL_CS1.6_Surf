"""--tree-labels / --mcts-labels (azr1b3): the tree's labels (surfgym/treelabels.py), their file,
the trainer's buffer, the actor-only forward the label loss uses, and the flags' defaults.
No simulator, no GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

from mcts_search import Tree  # noqa: E402
from surfgym.treelabels import (TreeLabelBuffer, build_labels, improve_weights,  # noqa: E402
                                load_labels, save_labels, sibling_weights, value_ranks)


def test_sibling_weights_rank_by_q_then_survival():
    # one parent, four children: q 1, 3, 2, 3; the two q=3 children differ by survival
    w = sibling_weights([7, 7, 7, 7], [1.0, 3.0, 2.0, 3.0], [True, True, True, False])
    # ranks (0 = worst): q1 -> 0, q2 -> 1, q3 dead -> 2, q3 alive -> 3; r = rank / 3;
    # max(0, r - 1/2) = 0, 1/2, 0, 1/6 -> normalised 0, 3/4, 0, 1/4
    assert np.allclose(w, [0.0, 0.75, 0.0, 0.25])


def test_sibling_weights_ties_teach_nothing_and_groups_are_separate():
    par = [2, 5, 2, 5, 2, 5, 9]
    q = [0.4, 0.1, 0.4, 0.3, 0.4, 0.2, 0.7]
    surv = [True] * 7
    w = sibling_weights(par, q, surv)
    assert np.allclose(w[[0, 2, 4]], 0.0)            # parent 2: all tie
    assert w[6] == 0.0                                # an only child
    # parent 5: q 0.1, 0.3, 0.2 -> ranks 0, 2, 1 -> r 0, 1, 1/2 -> only the best pulls
    assert np.allclose(w[[1, 3, 5]], [0.0, 1.0, 0.0])
    # a tie inside a group shares its average rank
    w2 = sibling_weights([1] * 4, [0.0, 1.0, 1.0, 0.5], [True] * 4)
    # ranks 0, 2.5, 2.5, 1 -> r 0, 5/6, 5/6, 1/3 -> 0, 1/3, 1/3, 0 -> 0, .5, .5, 0
    assert np.allclose(w2, [0.0, 0.5, 0.5, 0.0])


def test_value_ranks_are_percentiles_with_shared_ties():
    v = value_ranks([5.0, 1.0, 5.0, 3.0])
    assert np.allclose(v, [2.5 / 3, 0.0, 2.5 / 3, 1.0 / 3])
    assert np.allclose(value_ranks([2.0]), [0.5])
    assert np.allclose(value_ranks([-np.inf, 0.0]), [0.0, 1.0])


def test_improve_weights_pull_only_past_the_policys_own_child():
    # one parent: own child (first) q 0.5; siblings 0.2, 0.9, 0.7, 0.5 (dead)
    w = improve_weights([4] * 5, [0.5, 0.2, 0.9, 0.7, 0.5], [True, True, True, True, False],
                        [True, False, False, False, False])
    # ranks: 0.2 -> 0, 0.5 dead -> 1, own 0.5 alive -> 2, 0.7 -> 3, 0.9 -> 4; r = rank / 4;
    # r - r_own = -, -, 0, 1/4, 1/2 -> 0.9 gets 2/3, 0.7 gets 1/3
    assert np.allclose(w, [0.0, 0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0])
    # the own child is the best: nothing teaches
    w = improve_weights([1, 1, 1], [0.9, 0.2, 0.5], [True] * 3, [True, False, False])
    assert np.allclose(w, 0.0)
    # a tie with the own child is not an improvement; two parents stay separate
    w = improve_weights([1, 1, 2, 2], [0.5, 0.5, 0.1, 0.3], [True] * 4,
                        [True, False, True, False])
    assert np.allclose(w, [0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        improve_weights([1, 1], [0.1, 0.2], [True, True], [False, False])


def _small_tree():
    """roots 1, 2 (children of the virtual root); node 1 expanded into 3 children (one dead),
    node 2 into 2 tied children, node 3 (child of 1) into 2 children."""
    t = Tree(cap=8)
    r = t.add_children(0, [0.1, 0.2], [True, True], tick=[0, 0])
    a = t.add_children(int(r[0]), [0.3, 0.5, 0.5], [True, True, False], tick=[1, 1, 1])
    b = t.add_children(int(r[1]), [0.2, 0.2], [True, True], tick=[1, 1])
    c = t.add_children(int(a[1]), [0.35, 0.9], [True, True], tick=[2, 2])
    return t, r, a, b, c


def test_build_labels_on_a_tree():
    t, r, a, b, c = _small_tree()
    n = t.n
    D = 5
    lab_node = np.array([r[0], r[1], a[1]])
    lab_scal = np.arange(3 * D, dtype=np.float32).reshape(3, D)
    lab_pose = np.arange(3 * 6, dtype=np.float32).reshape(3, 6)
    c_act = np.zeros((n, 6), np.int8)
    c_act[:, 2] = np.arange(n) % 4
    c_u = np.linspace(-0.9, 0.9, n).astype(np.float32)
    lab = build_labels(t.parent[:n], t.ncld[:n], t.cstart[:n], t.q[:n], t.alive[:n],
                       t.finished[:n], lab_node, lab_scal, lab_pose, c_act, c_u)
    assert lab is not None
    # every expanded node stays in the node table (the value term reads them all)
    assert lab["scal"].shape == (3, D) and lab["pose"].shape == (3, 6)
    assert np.array_equal(lab["scal"], lab_scal)
    st = lab["stats"]
    assert st["expanded"] == 3 and st["tied_nodes"] == 1
    # node r[0]: own child a0 q 0.3; a1 q 0.9 (its subtree), a2 q 0.5 dead -> a1 r 1, a2 r 1/2,
    # own r 0 -> weights 2/3, 1/3. node r[1]: tie -> no row. node a[1]: own c0 0.35 < c1 0.9 -> c1
    assert np.array_equal(lab["first"], [0, -1, 2])
    assert np.array_equal(lab["node"], [0, 0, 2])
    assert np.allclose(lab["w"], [2.0 / 3.0, 1.0 / 3.0, 1.0])
    assert np.allclose(lab["u"], [c_u[a[1]], c_u[a[2]], c_u[c[1]]])
    assert np.array_equal(lab["act"][:, 2], c_act[[a[1], a[2], c[1]], 2])
    assert st["taught"] == 2 and st["rows"] == 3
    # values: percentile of q over the expanded nodes r0 (0.9), r1 (0.2), a1 (0.9)
    assert np.allclose(lab["value"], [0.75, 0.0, 0.75])
    # the own child is a best child at r[1] only (the tie); beaten at r[0] and a[1]
    assert abs(st["own_best"] - 1.0 / 3.0) < 1e-3 and abs(st["own_beaten"] - 2.0 / 3.0) < 1e-3


def test_build_labels_none_without_captured_nodes():
    t = Tree(cap=8)
    r = t.add_children(0, [0.1], [True], tick=[0])
    n = t.n
    assert build_labels(t.parent[:n], t.ncld[:n], t.cstart[:n], t.q[:n], t.alive[:n],
                        t.finished[:n], [r[0]], np.zeros((1, 3), np.float32),
                        np.zeros((1, 6), np.float32), np.zeros((n, 6), np.int8),
                        np.zeros(n, np.float32)) is None


def _write(tmp_path, name, nn, D, seed=0, taught_every=2, per=3):
    rng = np.random.default_rng(seed)
    taught = np.arange(nn) % taught_every == 0
    node = np.repeat(np.flatnonzero(taught), per).astype(np.int32)
    w = rng.random(len(node)).astype(np.float32) + 0.1
    for k in np.flatnonzero(taught):
        w[node == k] /= w[node == k].sum()
    first = np.full(nn, -1, np.int64)
    first[taught] = np.arange(taught.sum()) * per
    lab = {"scal": rng.random((nn, D)).astype(np.float32),
           "pose": rng.random((nn, 6)).astype(np.float32),
           "value": np.linspace(0, 1, nn).astype(np.float32), "first": first, "node": node,
           "act": rng.integers(0, 3, (len(node), 6)).astype(np.int8),
           "u": rng.uniform(-1, 1, len(node)).astype(np.float32), "w": w,
           "stats": {"expanded": nn}}
    p = tmp_path / name
    save_labels(p, lab, {"scal_dim": D})
    return p, lab


def test_file_roundtrip_and_buffer(tmp_path):
    D = 7
    p1, lab1 = _write(tmp_path, "a.npz", 10, D, seed=1)
    z = load_labels(p1)
    for k in ("scal", "pose", "value", "first", "node", "act", "u", "w"):
        assert np.array_equal(z[k], lab1[k])
    assert z["meta"]["stats"] == {"expanded": 10}
    buf = TreeLabelBuffer(torch.device("cpu"), scal_dim=D, keep=2, seed=3)
    assert len(buf) == 0
    buf.add(p1, "k01")
    assert buf.n_nodes == 10 and buf.n_taught == 5 and buf.n_rows == 15
    s, po, a, u, v, m = buf.sample(20000)
    assert s.shape == (20000, D) and po.shape == (20000, 6) and a.shape == (20000, 6)
    assert u.shape == (20000,) and v.shape == (20000,) and m.shape == (20000,)
    # nodes come out uniformly, the mask is exactly the taught ones
    k = np.rint(v.numpy() * 9).astype(int)
    counts = np.bincount(k, minlength=10)
    assert counts.min() > 1600 and counts.max() < 2400
    assert np.array_equal(m.numpy() > 0, k % 2 == 0)
    # a taught node's rows are its own, drawn by weight; an untaught node carries zeros
    for node in range(0, 10, 2):
        sel = k == node
        rows = np.flatnonzero(lab1["node"] == node)
        got = u.numpy()[sel]
        assert set(np.round(got, 5)) <= set(np.round(lab1["u"][rows], 5))
        frac = [(np.isclose(got, lab1["u"][rr])).mean() for rr in rows]
        assert np.allclose(frac, lab1["w"][rows], atol=0.06)
        assert np.allclose(s.numpy()[sel][0], lab1["scal"][node])
    assert (u.numpy()[k % 2 == 1] == 0).all() and (a.numpy()[k % 2 == 1] == 0).all()
    p2, _ = _write(tmp_path, "b.npz", 5, D, seed=2)
    p3, _ = _write(tmp_path, "c.npz", 4, D, seed=3, taught_every=4)
    buf.add(p2, "k02")
    buf.add(p3, "k03")                             # keep 2: the first tree drops out
    assert [t["tag"] for t in buf.trees] == ["k02", "k03"]
    assert buf.n_nodes == 9 and buf.n_taught == 3 + 1 and buf.n_rows == 12
    s, _, _, u, v, m = buf.sample(4000)
    assert s.shape == (4000, D) and abs(float(m.mean()) - 4.0 / 9.0) < 0.05
    with pytest.raises(ValueError):
        TreeLabelBuffer(torch.device("cpu"), scal_dim=D + 1, keep=1).add(p1)


def test_buffer_with_no_policy_rows(tmp_path):
    D = 3
    p, _ = _write(tmp_path, "z.npz", 4, D, taught_every=1000, per=1)
    buf = TreeLabelBuffer(torch.device("cpu"), scal_dim=D, keep=1)
    buf.add(p, "k01")
    assert buf.n_nodes == 4 and buf.n_taught == 1 and buf.n_rows == 1
    p2, lab = _write(tmp_path, "y.npz", 4, D, taught_every=1000, per=1)
    lab["node"] = lab["node"][:0]
    lab["act"], lab["u"], lab["w"] = lab["act"][:0], lab["u"][:0], lab["w"][:0]
    lab["first"] = np.full(4, -1, np.int64)
    save_labels(p2, lab, {})
    buf.add(p2, "k02")
    s, _, a, u, v, m = buf.sample(64)
    assert s.shape == (64, D) and float(m.sum()) == 0.0 and (u == 0).all()


def test_load_refuses_a_row_outside_the_node_table(tmp_path):
    p, lab = _write(tmp_path, "a.npz", 4, 4)
    lab["node"] = lab["node"].copy()
    lab["node"][0] = 4
    save_labels(p, lab, {})
    with pytest.raises(ValueError):
        load_labels(p)


def test_actor_out_is_the_forward_split_actor():
    import train_fast as tf
    torch.manual_seed(0)
    route = 12                                    # a scalar block past the core 15 (keys etc.)
    p = tf.Policy(tf.N_SCALAR + route + 64 * 32 * 2, 64, 32, emb=64, hidden=32, in_ch=2,
                  simba=True, view_continuous=True, view_absolute="velocity",
                  route_dim=route)
    B = 6
    scal = torch.randn(B, tf.N_SCALAR + route)
    img = torch.rand(B, 64 * 32 * 2)
    with torch.no_grad():
        lg, _v = p.forward_split(scal, img)
        lg2, t = p.actor_out(scal, img)
    assert torch.equal(lg, lg2)
    assert t.shape == (B, 32) and p.action_head.in_features == 32


def test_flags_default_off():
    import train_fast as tf
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--tree-labels", type=float, default=None' in src
    bt = (ROOT / "tools" / "beam_tas.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--mcts-labels", action="store_true"' in bt
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    for k in ("tree_labels", "tree_labels_batch", "tree_labels_keep", "tree_value_coef"):
        assert f'"{k}"' in rc
    assert hasattr(tf.Policy, "actor_out")


def test_train_only_in_record_ckpt():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    for k in ("tree_labels", "tree_labels_batch", "tree_labels_keep", "tree_value_coef"):
        assert k in record_ckpt.TRAIN_ONLY
