"""azr1b4's tree-only learner: the tree's labels (tools/mcts_search.node_targets / onehot_rows),
the supervised loss (tools/az_loop.distil_loss) and the label buffer. CPU, no simulator."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import train_fast as tf  # noqa: E402
from mcts_search import Tree, node_targets, onehot_rows  # noqa: E402


def _tree():
    """root r (score 0.1) with 4 children: q 0.1 (dead, ties the parent), 0.3, 0.3 (a tie at
    the best), 0.2; child 1 gets its own children (0.5, 0.3) so its backed-up Q is 0.5."""
    t = Tree(cap=16)
    r = t.add_children(0, [0.1], [True], tick=[0])
    kids = t.add_children(int(r[0]), [0.1, 0.3, 0.3, 0.2], [False, True, True, True],
                          tick=[48] * 4)
    g = t.add_children(int(kids[1]), [0.5, 0.3], [True, True], tick=[96, 96])
    return t, int(r[0]), kids, g


def test_node_targets_picks_the_children_that_beat_their_siblings():
    t, r, kids, g = _tree()
    nvec = (15, 7, 4, 4, 2, 3)
    pact = np.zeros((t.n, 6), np.int16)
    pact[kids, 2] = [0, 1, 2, 3]            # fwd head: each child a different first move
    pact[kids, 4] = [1, 0, 1, 0]
    pz = np.zeros((t.n, 2), np.float32)
    pz[kids, 0] = [0.0, 1.0, 3.0, 9.0]
    pact[g, 2] = [3, 1]
    nt = node_targets(t, [r, int(kids[1]), int(kids[3])], pact, pz, nvec, 15)
    # the root: child 1's subtree reached 0.5 -> the ONLY best child (child 2 stays at 0.3)
    assert nt["keep"][0] and nt["n_best"][0] == 1
    assert nt["probs"][0, 2, 1] == 1.0 and nt["probs"][0, 2].sum() == 1.0
    assert nt["probs"][0, 4, 0] == 1.0
    assert np.isclose(nt["vmu"][0, 0], 1.0) and nt["vsd"][0, 0] == 0.0
    assert np.isclose(nt["value"][0], 0.5)
    assert not nt["probs"][0, :2].any()       # the view heads carry no categorical target
    # child 1: its own children 0.5 > 0.3, best = the first; 0.5 > its own score 0.3 -> kept
    assert nt["keep"][1] and nt["probs"][1, 2, 3] == 1.0
    # child 3 was never expanded -> no target
    assert not nt["keep"][2] and nt["n_best"][2] == 0


def test_node_targets_tie_spreads_the_mass_and_flat_nodes_are_dropped():
    t = Tree(cap=16)
    r = t.add_children(0, [0.2], [True], tick=[0])
    kids = t.add_children(int(r[0]), [0.4, 0.4, 0.1], [True, True, True], tick=[48] * 3)
    pact = np.zeros((t.n, 6), np.int16)
    pact[kids, 3] = [2, 0, 1]
    pz = np.zeros((t.n, 2), np.float32)
    pz[kids, 0] = [1.0, 3.0, 0.0]
    nt = node_targets(t, [int(r[0])], pact, pz, (15, 7, 4, 4, 2, 3), 15, prefer_first=False)
    assert nt["keep"][0] and nt["n_best"][0] == 2
    assert np.allclose(nt["probs"][0, 3, [2, 0]], 0.5)
    assert np.isclose(nt["vmu"][0, 0], 2.0) and np.isclose(nt["vsd"][0, 0], 1.0)
    # the default: the policy's own child (the first) ties for best -> it alone is the target
    nt1 = node_targets(t, [int(r[0])], pact, pz, (15, 7, 4, 4, 2, 3), 15)
    assert nt1["keep"][0] and nt1["n_best"][0] == 1
    assert nt1["probs"][0, 3, 2] == 1.0 and np.isclose(nt1["vmu"][0, 0], 1.0)
    assert nt1["vsd"][0, 0] == 0.0
    # ...and when a random child is strictly better, the first child gets nothing
    t3 = Tree(cap=16)
    r3 = t3.add_children(0, [0.2], [True], tick=[0])
    k3 = t3.add_children(int(r3[0]), [0.3, 0.4, 0.4], [True, True, True], tick=[48] * 3)
    p3 = np.zeros((t3.n, 6), np.int16)
    p3[k3, 3] = [2, 0, 1]
    nt3 = node_targets(t3, [int(r3[0])], p3, np.zeros((t3.n, 2), np.float32),
                       (15, 7, 4, 4, 2, 3), 15)
    assert nt3["n_best"][0] == 2 and np.allclose(nt3["probs"][0, 3, [0, 1]], 0.5)
    assert nt3["probs"][0, 3, 2] == 0.0
    # a node whose children all tie at its own score says nothing: dropped
    t2 = Tree(cap=8)
    r2 = t2.add_children(0, [0.2], [True], tick=[0])
    t2.add_children(int(r2[0]), [0.2, 0.2], [False, False], tick=[48, 48])
    nt2 = node_targets(t2, [int(r2[0])], np.zeros((t2.n, 6), np.int16),
                       np.zeros((t2.n, 2), np.float32), (15, 7, 4, 4, 2, 3), 15)
    assert not nt2["keep"][0]


def test_node_targets_refuses_an_action_outside_its_head():
    t, r, kids, _g = _tree()
    pact = np.zeros((t.n, 6), np.int16)
    pact[kids[1], 5] = 7                      # duck head has 3 bins
    with pytest.raises(ValueError):
        node_targets(t, [r], pact, np.zeros((t.n, 2), np.float32), (15, 7, 4, 4, 2, 3), 15)


def test_onehot_rows():
    oh = onehot_rows(np.array([[7, 3, 1, 2, 0, 2]]), (15, 7, 4, 4, 2, 3), 15)
    assert oh.shape == (1, 6, 15)
    assert not oh[0, :2].any()
    assert oh[0, 2, 1] == 1 and oh[0, 3, 2] == 1 and oh[0, 4, 0] == 1 and oh[0, 5, 2] == 1
    assert oh.sum() == 4


@pytest.fixture
def keys_hold():
    tf.set_keys_hold(True)
    yield
    tf.set_keys_hold(False)


def _policy(W=8, H=4, extra=11):
    torch.manual_seed(0)
    return tf.Policy(tf.N_SCALAR + extra + W * H * 2, W, H, emb=16, hidden=16, in_ch=2,
                     route_dim=extra, view_continuous=True, view_absolute="velocity",
                     simba=True)


def test_distil_loss_fits_a_fixed_target(keys_hold):
    import az_loop
    W, H, extra = 8, 4, 11
    pol = _policy(W, H, extra)
    packer = tf.HeadPacker(torch.device("cpu"))
    n = 64
    g = torch.Generator().manual_seed(1)
    scal = torch.randn(n, tf.N_SCALAR + extra, generator=g)
    img = torch.rand(n, W * H * 2, generator=g)
    pact = np.zeros((n, 6), np.int64)
    pact[:, 2] = np.arange(n) % 4
    pact[:, 3] = 3
    pact[:, 4] = 1
    pact[:, 5] = (np.arange(n) // 4) % 3
    probs = torch.as_tensor(onehot_rows(pact, tf.NVEC, tf.NPAD))
    vmu = torch.zeros(n, 2)
    vmu[:, 0] = torch.linspace(-1.0, 1.0, n)
    vsd = torch.zeros(n, 2)
    val = torch.linspace(0.0, 1.0, n)
    w = torch.ones(n)
    vw = torch.tensor([1.0, 0.0])
    opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        loss, st = az_loop.distil_loss(pol, packer, scal, img, probs, vmu, vsd, val, w, vw,
                                       0.25, tf.BC_VIEW_SIGMA)
        if first is None:
            first = st.tolist()
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = st.tolist()
    names = az_loop.STAT_NAMES
    f, l_ = dict(zip(names, first)), dict(zip(names, last))
    assert l_["loss"] < f["loss"]
    assert l_["acc"] > 0.95 and l_["acc"] > f["acc"]
    assert l_["yaw_mse"] < 0.05 and l_["value_mse"] < 0.01
    # the pitch head carried no weight: the loss does not depend on its target
    vmu2 = vmu.clone()
    vmu2[:, 1] = 5.0
    l1, _ = az_loop.distil_loss(pol, packer, scal, img, probs, vmu, vsd, val, w, vw, 0.25,
                                tf.BC_VIEW_SIGMA)
    l2, _ = az_loop.distil_loss(pol, packer, scal, img, probs, vmu2, vsd, val, w, vw, 0.25,
                                tf.BC_VIEW_SIGMA)
    assert torch.isclose(l1, l2)


def test_label_buffer_keeps_the_last_trees_and_mixes_kinds():
    import az_loop
    buf = az_loop.LabelBuffer(2, torch.device("cpu"), z_clip=3.0)

    def lab(n_node, n_line, n_pre, v):
        n = n_node + n_line + n_pre
        return {"scal": np.full((n, 3), v, np.float32), "pose": np.zeros((n, 6), np.float32),
                "probs": np.zeros((n, 6, 15), np.float32),
                "vmu": np.full((n, 2), 5.0, np.float32), "vsd": np.zeros((n, 2), np.float32),
                "value": np.full(n, v, np.float32),
                "kind": np.r_[np.zeros(n_node), np.ones(n_line),
                              np.full(n_pre, 2)].astype(np.int8),
                "w": np.ones(n, np.float32)}
    buf.add(0, lab(10, 2, 3, 0.0))
    buf.add(1, lab(5, 5, 2, 1.0))
    buf.add(2, lab(7, 1, 0, 2.0))
    assert buf.rounds() == [1, 2] and buf.rows() == 20
    draw, n_kind = buf.sampler(40, {0: 0.5, 1: 0.25, 2: 0.25}, torch.Generator().manual_seed(0))
    assert n_kind == {0: 12, 1: 6, 2: 2}
    b = draw()
    assert b["scal"].shape == (40, 3)
    assert int((b["kind"] == 0).sum()) == 20 and int((b["kind"] == 1).sum()) == 10
    assert int((b["kind"] == 2).sum()) == 10
    assert set(np.unique(b["value"].numpy())) <= {1.0, 2.0}
    assert float(b["vmu"].max()) == 3.0          # view targets clipped at |z| 3
    # recency 0: only the newest tree is drawn (its weight 0**0 = 1, older ones 0); its empty
    # prefix pool hands its share to the other two
    draw0, _nk = buf.sampler(40, {0: 0.5, 1: 0.25, 2: 0.25}, torch.Generator().manual_seed(0),
                             recency=0.0)
    b0 = draw0()
    assert set(np.unique(b0["value"].numpy())) == {2.0}
    assert int((b0["kind"] == 2).sum()) == 0 and len(b0["kind"]) == 40
    assert int((b0["kind"] == 0).sum()) == 27
    buf.add(3, None)                          # an empty tree is skipped
    assert buf.rounds() == [1, 2]


def test_node_targets_advantage_is_the_targets_margin_over_the_siblings_mean():
    t, r, kids, g = _tree()
    nvec = (15, 7, 4, 4, 2, 3)
    pact = np.zeros((t.n, 6), np.int16)
    nt = node_targets(t, [r, int(kids[1])], pact, np.zeros((t.n, 2), np.float32), nvec, 15)
    span = t.qhi - t.qlo                      # the tree's own Q range: 0.5 - 0.1
    # root: children Q 0.1, 0.5, 0.3, 0.2 -> target 0.5, mean 0.275
    assert np.isclose(nt["adv"][0], (0.5 - 0.275) / span)
    # kids[1]: children Q 0.5 (its first child, the target) and 0.3 -> mean 0.4
    assert np.isclose(nt["adv"][1], (0.5 - 0.4) / span)


def test_label_buffer_draws_node_rows_by_advantage():
    import az_loop
    buf = az_loop.LabelBuffer(2, torch.device("cpu"))
    n = 6
    lab = {"scal": np.arange(n, dtype=np.float32)[:, None], "pose": np.zeros((n, 6), np.float32),
           "probs": np.zeros((n, 6, 15), np.float32), "vmu": np.zeros((n, 2), np.float32),
           "vsd": np.zeros((n, 2), np.float32), "value": np.zeros(n, np.float32),
           "kind": np.array([0, 0, 0, 0, 1, 1], np.int8), "w": np.ones(n, np.float32),
           "adv": np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0], np.float32)}
    buf.add(0, lab)
    draw, _nk = buf.sampler(64, {0: 0.5, 1: 0.5}, torch.Generator().manual_seed(0),
                            adv_kinds=(0,))
    b = draw()
    nodes = b["scal"][b["kind"] == 0].reshape(-1).numpy()
    assert set(nodes.tolist()) == {2.0}           # only the node row with advantage is drawn
    # off: every node row alike; a file without the column loads with adv 1
    buf2 = az_loop.LabelBuffer(2, torch.device("cpu"))
    buf2.add(0, {k: v for k, v in lab.items() if k != "adv"})
    draw2, _nk2 = buf2.sampler(256, {0: 0.5, 1: 0.5}, torch.Generator().manual_seed(0))
    nodes2 = draw2()["scal"].reshape(-1).numpy()
    assert {0.0, 1.0, 2.0, 3.0} <= set(nodes2.tolist())


def test_sigma_fit_narrows_the_yaw_sigma_to_the_targets_spread_and_never_widens(keys_hold):
    import az_loop
    from surfgym.view import LOG_STD_INIT
    W, H, extra = 8, 4, 11
    pol = _policy(W, H, extra)
    packer = tf.HeadPacker(torch.device("cpu"))
    n = 256
    g = torch.Generator().manual_seed(3)
    scal = torch.randn(n, tf.N_SCALAR + extra, generator=g)
    img = torch.rand(n, W * H * 2, generator=g)
    probs = torch.as_tensor(onehot_rows(np.tile([[7, 3, 1, 1, 0, 0]], (n, 1)), tf.NVEC, tf.NPAD))
    # targets: a constant steer plus noise of std 0.1 that no state explains
    vmu = torch.zeros(n, 2)
    vmu[:, 0] = 0.5 + 0.1 * torch.randn(n, generator=g)
    vsd = torch.zeros(n, 2)
    val, w, vw = torch.zeros(n), torch.ones(n), torch.tensor([1.0, 0.0])
    opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
    for _ in range(400):
        loss, st = az_loop.distil_loss(pol, packer, scal, img, probs, vmu, vsd, val, w, vw,
                                       0.25, tf.BC_VIEW_SIGMA, sigma_fit=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            pol.view_std.log_std[0].clamp_(max=LOG_STD_INIT)
    sig = float(pol.log_std()[0].detach().exp())
    assert 0.05 < sig < 0.2                   # from 0.3 toward the unexplained spread ~0.1
    assert float(st[-1]) == pytest.approx(sig, rel=2e-2)   # logged before the last step
    # targets far wider than the init: sigma stays capped at its init
    vmu[:, 0] = 3.0 * torch.randn(n, generator=g)
    for _ in range(200):
        loss, st = az_loop.distil_loss(pol, packer, scal, img, probs, vmu, vsd, val, w, vw,
                                       0.25, tf.BC_VIEW_SIGMA, sigma_fit=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            pol.view_std.log_std[0].clamp_(max=LOG_STD_INIT)
    assert float(pol.view_std.log_std[0]) <= LOG_STD_INIT + 1e-6
    # off: the yaw log sigma gets no gradient at all
    pol2 = _policy(W, H, extra)
    l2, _ = az_loop.distil_loss(pol2, packer, scal, img, probs, vmu, vsd, val, w, vw, 0.25,
                                tf.BC_VIEW_SIGMA)
    l2.backward()
    gr = pol2.view_std.log_std.grad
    assert gr is None or float(gr.abs().sum()) == 0.0
