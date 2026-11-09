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
from surfgym.treelabels import (TreeLabelBuffer, build_labels, load_labels,  # noqa: E402
                                save_labels, sibling_weights, value_ranks)


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


def _small_tree():
    """roots 1, 2 (children of the virtual root); node 1 expanded into 3 children (one dead),
    node 2 into 2 tied children, node 3 (child of 1) into 2 children."""
    t = Tree(cap=8)
    r = t.add_children(0, [0.1, 0.2], [True, True], tick=[0, 0])
    a = t.add_children(int(r[0]), [0.3, 0.5, 0.5], [True, True, False], tick=[1, 1, 1])
    b = t.add_children(int(r[1]), [0.2, 0.2], [True, True], tick=[1, 1])
    c = t.add_children(int(a[0]), [0.9, 0.35], [True, True], tick=[2, 2])
    return t, r, a, b, c


def test_build_labels_on_a_tree():
    t, r, a, b, c = _small_tree()
    n = t.n
    D = 5
    lab_node = np.array([r[0], r[1], a[0]])
    lab_scal = np.arange(3 * D, dtype=np.float32).reshape(3, D)
    lab_pose = np.arange(3 * 6, dtype=np.float32).reshape(3, 6)
    c_act = np.zeros((n, 6), np.int8)
    c_act[:, 2] = np.arange(n) % 4
    c_u = np.linspace(-0.9, 0.9, n).astype(np.float32)
    lab = build_labels(t.parent[:n], t.ncld[:n], t.cstart[:n], t.q[:n], t.alive[:n],
                       t.finished[:n], lab_node, lab_scal, lab_pose, c_act, c_u)
    assert lab is not None
    # node r[1]'s children tie -> it teaches nothing; r[0] and a[0] do
    assert lab["stats"]["expanded"] == 3 and lab["stats"]["nodes"] == 2
    assert lab["scal"].shape == (2, D) and lab["pose"].shape == (2, 6)
    assert np.array_equal(lab["scal"][0], lab_scal[0]) and np.array_equal(lab["scal"][1],
                                                                           lab_scal[2])
    # per taught node the weights sum to 1, and every row points at a taught node
    for k in range(2):
        assert np.isclose(lab["w"][lab["node"] == k].sum(), 1.0)
    # node r[0]: q of its children after backup: a0 0.9 (its subtree), a1 0.5 alive, a2 0.5 dead
    # -> a1 ranks over a2 at equal q: ranks a0 2, a1 1, a2 0 -> only a0 pulls, w = 1... r - 1/2:
    # a0 r = 1 -> 1/2, a1 r = 1/2 -> 0 -> a0 alone
    rows0 = np.flatnonzero(lab["node"] == 0)
    assert len(rows0) == 1 and lab["act"][rows0[0], 2] == c_act[a[0], 2]
    assert np.isclose(lab["u"][rows0[0]], c_u[a[0]])
    # node a[0]: children 0.9 and 0.35 -> the first
    rows1 = np.flatnonzero(lab["node"] == 1)
    assert len(rows1) == 1 and np.isclose(lab["u"][rows1[0]], c_u[c[0]])
    # values: percentile of q over the expanded nodes r0 (0.9), r1 (0.2), a0 (0.9)
    assert np.allclose(lab["value"], [0.75, 0.75])
    # the policy's own child (child 0 of every block) is the best at both taught nodes
    assert lab["stats"]["child0_best"] == 1.0


def test_build_labels_none_when_nothing_teaches():
    t = Tree(cap=8)
    r = t.add_children(0, [0.1], [True], tick=[0])
    t.add_children(int(r[0]), [0.2, 0.2], [True, True], tick=[1, 1])
    n = t.n
    assert build_labels(t.parent[:n], t.ncld[:n], t.cstart[:n], t.q[:n], t.alive[:n],
                        t.finished[:n], [r[0]], np.zeros((1, 3), np.float32),
                        np.zeros((1, 6), np.float32), np.zeros((n, 6), np.int8),
                        np.zeros(n, np.float32)) is None


def _write(tmp_path, name, nn, rows, D, seed=0):
    rng = np.random.default_rng(seed)
    node = np.repeat(np.arange(nn), rows // nn).astype(np.int32)
    w = rng.random(len(node)).astype(np.float32)
    for k in range(nn):
        w[node == k] /= w[node == k].sum()
    lab = {"scal": rng.random((nn, D)).astype(np.float32),
           "pose": rng.random((nn, 6)).astype(np.float32),
           "value": rng.random(nn).astype(np.float32), "node": node,
           "act": rng.integers(0, 3, (len(node), 6)).astype(np.int8),
           "u": rng.uniform(-1, 1, len(node)).astype(np.float32), "w": w,
           "stats": {"nodes": nn}}
    p = tmp_path / name
    save_labels(p, lab, {"scal_dim": D})
    return p, lab


def test_file_roundtrip_and_buffer(tmp_path):
    D = 7
    p1, lab1 = _write(tmp_path, "a.npz", 10, 40, D, seed=1)
    z = load_labels(p1)
    for k in ("scal", "pose", "value", "node", "act", "u", "w"):
        assert np.array_equal(z[k], lab1[k])
    assert z["meta"]["stats"] == {"nodes": 10}
    buf = TreeLabelBuffer(torch.device("cpu"), scal_dim=D, keep=2, seed=3)
    assert len(buf) == 0
    buf.add(p1, "k01")
    assert buf.n_nodes == 10 and buf.n_rows == 40
    s, po, a, u, v = buf.sample(4096)
    assert s.shape == (4096, D) and po.shape == (4096, 6) and a.shape == (4096, 6)
    assert u.shape == (4096,) and v.shape == (4096,)
    # rows come back with their own node's scalars and value
    rows = {tuple(np.round(lab1["scal"][k], 6)): lab1["value"][k] for k in range(10)}
    for i in range(0, 4096, 97):
        assert np.isclose(rows[tuple(np.round(s[i].numpy(), 6))], v[i].item())
    # sampling follows the weights: every node carries mass 1, so nodes come out ~uniformly
    _, _, _, _, v2 = buf.sample(20000)
    counts = np.array([(np.isclose(v2.numpy(), lab1["value"][k])).sum() for k in range(10)])
    assert counts.min() > 1500 and counts.max() < 2500
    p2, _ = _write(tmp_path, "b.npz", 5, 15, D, seed=2)
    p3, _ = _write(tmp_path, "c.npz", 4, 8, D, seed=3)
    buf.add(p2, "k02")
    buf.add(p3, "k03")                             # keep 2: the first tree drops out
    assert [t["tag"] for t in buf.trees] == ["k02", "k03"]
    assert buf.n_nodes == 9 and buf.n_rows == 23
    s, _, _, _, _ = buf.sample(1000)
    assert s.shape == (1000, D)
    with pytest.raises(ValueError):
        TreeLabelBuffer(torch.device("cpu"), scal_dim=D + 1, keep=1).add(p1)


def test_load_refuses_a_row_outside_the_node_table(tmp_path):
    p, lab = _write(tmp_path, "a.npz", 3, 6, 4)
    lab["node"] = lab["node"].copy()
    lab["node"][0] = 3
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
