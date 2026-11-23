"""--dae (Direct Advantage Estimation, Pan et al., NeurIPS 2022; the user, 2026-11-23): the
policy's advantage is the advantage head's output, centred under the behaviour policy, fitted
with V by the paper's Eq. 13 over each env's segment.

(a) dae_advantage is EXACTLY centred under the behaviour policy: over the joint key bins the
    mu-weighted mean of the table part is zero (enumerated), and with the view drawn from the
    behaviour Gaussian the whole advantage averages to zero (Monte Carlo, within 5 standard
    errors), masked bins included.
(b) dae_targets is the recursion G_t = d_t + g (1 - done_t) G_{t+1}, G_T = vlast, on random
    segments with episode ends.
(c) the loss identifies the advantage: on a synthetic problem whose reward is a joint-bin table
    plus view terms, minimising Eq. 13 over a free head and a free V recovers the true centred
    advantage of the taken actions and V = E_mu r.
(d) Policy: adv_head is registered LAST (every other tensor is the flag-off model's under the
    same seed), the state_dict gains exactly its two keys, the plain forward is the flag-off
    model's, heads(adv=True) adds f of width nj (1 + 2 n_z); a discrete view is refused.
(e) the tools mention the key (record_ckpt's and beam_tas's strict loads build adv_head).
(f) trainer smokes (CPU, recipe v14 at smoke scale on two maps, MM_TEST_MAPS): the flag ON trains
    (config key, the dae/* columns, a checkpoint with adv_head, record_ckpt's audit), a resume
    restores it, a resume with --dae 0 is refused, --critic-pay-d0 with it is refused; the flag
    OFF is bit-identical to the PARENT commit (config, every eval trajectory, progress.csv minus
    fps, the final weights).
"""
from __future__ import annotations

import ast
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(HERE))

import torch                                                   # noqa: E402

import train_fast as tf                                        # noqa: E402
from train_fast import N_SCALAR, N_VIEW, NACT, NPAD, Policy    # noqa: E402

PARENT = "b5e3fa62"            # mm/split head the flag was built on
WIDTHS = (4, 4, 2, 3)          # --keys-hold's fwd, side, jump, duck


@pytest.fixture(autouse=True)
def _keys_hold():
    tf.set_keys_hold(True)
    yield
    tf.set_keys_hold(False)


def _behaviour(B, n_z, seed, mask=True):
    """a random behaviour policy: per-head log-probabilities padded to NPAD (pads and, with
    ``mask``, one live bin per row of the side head at probability zero), a Gaussian view"""
    g = torch.Generator().manual_seed(seed)
    lsm = torch.full((B, NACT - N_VIEW, NPAD), tf.NEG)
    for h, w in enumerate(WIDTHS):
        lg = torch.randn(B, w, generator=g) * 1.5
        if mask and h == 1:
            lg[:, 0] = tf.NEG
        lsm[:, h, :w] = torch.log_softmax(lg, -1)
    m = torch.randn(B, n_z, generator=g)
    ls = torch.randn(B, n_z, generator=g) * 0.3 - 1.0
    return lsm, m, ls


def _draw(lsm, m, ls, K, seed):
    """K draws per row from the behaviour policy -> (act (B K, NACT), z (B K, n_z))"""
    g = torch.Generator().manual_seed(seed)
    B, n_z = m.shape
    act = torch.zeros(B, K, NACT, dtype=torch.long)
    for h, w in enumerate(WIDTHS):
        p = lsm[:, h, :w].exp()
        act[:, :, N_VIEW + h] = torch.multinomial(p, K, replacement=True, generator=g)
    z = m[:, None, :] + ls.exp()[:, None, :] * torch.randn(B, K, n_z, generator=g)
    return act.reshape(B * K, NACT), z.reshape(B * K, n_z)


# ==========================================================================
# (a) exact centring
# ==========================================================================
@pytest.mark.parametrize("n_z", [2, 3])
def test_the_advantage_is_centred_under_the_behaviour_policy(n_z):
    B = 6
    widths, nj, width = tf.dae_shape(n_z)
    assert widths == WIDTHS and nj == 96 and width == 96 * (1 + 2 * n_z)
    lsm, m, ls = _behaviour(B, n_z, seed=1)
    gadv = torch.randn(B, width, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    # the table part alone (the view coefficients zeroed: at u = 0 the He2 term is -P, mean
    # zero only over the Gaussian), enumerated over every joint bin: sum_j mu(j) A(j) = 0
    gtab = gadv.clone()
    gtab[:, nj:] = 0.0
    grid = torch.stack(torch.meshgrid(*[torch.arange(w) for w in WIDTHS], indexing="ij"),
                       -1).reshape(-1, len(WIDTHS))
    assert grid.shape[0] == nj
    for b in range(B):
        act = torch.zeros(nj, NACT, dtype=torch.long)
        act[:, N_VIEW:] = grid
        a = tf.dae_advantage(gtab[b:b + 1].expand(nj, -1), lsm[b:b + 1].expand(nj, -1, -1),
                             act, torch.zeros(nj, n_z, dtype=torch.float64))
        p = torch.ones(nj, dtype=torch.float64)
        for h, w in enumerate(WIDTHS):
            p = p * lsm[b, h, :w].double().exp()[grid[:, h]]
        assert abs(float((p * a).sum())) < 1e-6     # float32 probabilities
        assert torch.isfinite(a).all()
    # the whole advantage, the view drawn from the behaviour Gaussian: mean zero
    K = 40000
    act, z = _draw(lsm, m, ls, K, seed=3)
    rep = lambda x: x.repeat_interleave(K, 0)
    u = (z - rep(m)) * torch.exp(-rep(ls))
    a = tf.dae_advantage(rep(gadv.float()), rep(lsm), act, u).reshape(B, K).double()
    se = a.std(1) / np.sqrt(K)
    assert (a.mean(1).abs() < 5 * se).all(), (a.mean(1), se)
    assert (a.std(1) > 0.1).all()


# ==========================================================================
# (b) the targets
# ==========================================================================
def test_the_targets_are_the_recursion():
    g = torch.Generator().manual_seed(4)
    T, B = 17, 9
    d = torch.randn(T, B, generator=g, dtype=torch.float64)
    done = (torch.rand(T, B, generator=g) < 0.15).double()
    done[:, 0] = 0.0
    done[-1, 1] = 1.0
    done[0, 2] = 1.0
    vlast = torch.randn(B, generator=g, dtype=torch.float64)
    gam = torch.tensor(0.987, dtype=torch.float64)
    got = tf.dae_targets(d, done, vlast, gam)
    want = torch.zeros(T, B, dtype=torch.float64)
    nxt = vlast.clone()
    for t in reversed(range(T)):
        nxt = d[t] + gam * (1.0 - done[t]) * nxt
        want[t] = nxt
    assert torch.allclose(got, want, atol=1e-12)
    # differentiable in d (G is linear in the advantage)
    d.requires_grad_(True)
    tf.dae_targets(d, done, vlast, gam).sum().backward()
    assert d.grad is not None and torch.isfinite(d.grad).all()


# ==========================================================================
# (c) the loss identifies the advantage
# ==========================================================================
def test_the_loss_recovers_the_true_advantage():
    torch.manual_seed(5)
    n_z, T, B = 2, 4, 512
    widths, nj, width = tf.dae_shape(n_z)
    lsm, m, ls = _behaviour(1, n_z, seed=6)
    true = torch.randn(width) * torch.cat([torch.full((nj,), 2.0), torch.full((2 * nj * n_z,),
                                                                             0.5)])
    vhat = torch.zeros((), requires_grad=True)
    head = torch.zeros(width, requires_grad=True)
    opt = torch.optim.Adam([vhat, head], lr=0.05)
    done = torch.ones(T, B)                     # every step its own episode: G_t = r_t - A_t

    def batch(seed):
        act, z = _draw(lsm, m, ls, T * B, seed)
        u = (z - m) * torch.exp(-ls)
        L = lsm.expand(T * B, -1, -1)
        r = tf.dae_advantage(true.expand(T * B, -1), L, act, u) + 3.0
        return act, u, L, r

    for it in range(600):
        act, u, L, r = batch(100 + it)
        A = tf.dae_advantage(head.expand(T * B, -1), L, act, u)
        G = tf.dae_targets((r - A).view(T, B), done, torch.zeros(B), torch.tensor(0.99))
        loss = 0.5 * (G - vhat).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    act, u, L, r = batch(7)
    with torch.no_grad():
        A = tf.dae_advantage(head.expand(T * B, -1), L, act, u)
        At = tf.dae_advantage(true.expand(T * B, -1), L, act, u)
    assert float(vhat.detach()) == pytest.approx(3.0, abs=0.05)
    assert float(torch.corrcoef(torch.stack([A, At]))[0, 1]) > 0.99
    assert float((A - At).abs().mean()) < 0.1 * float(At.abs().mean())


def test_the_step_is_scaled_by_the_gae_spread():
    """the policy step: A centred per map, divided by the per-map std of the GAE advantage of the
    same rows (ue_3m collapsed with A standardised by its own, 5-6x smaller, std); without
    scale_of the normaliser is the shipped one op for op"""
    g = torch.Generator().manual_seed(8)
    n, k = 400, 3
    slot = torch.randint(0, k, (n,), generator=g)
    slot[:2] = 2                                     # map 2 also has rows below...
    a = torch.randn(n, generator=g) * 0.15
    gae = torch.randn(n, generator=g) * torch.tensor([0.5, 0.8, 1.3])[slot]
    a64 = a.double()
    oh = torch.nn.functional.one_hot(slot, k).double()
    cnt = oh.sum(0)
    mean = (oh * a64[:, None]).sum(0) / cnt
    dev = a64[:, None] - mean[None, :]
    std = ((oh * dev * dev).sum(0) / (cnt - 1.0)).sqrt()
    old = ((a64 - mean[slot]) / (std[slot] + 1e-8)).float()
    assert torch.equal(tf.adv_norm_per_map(a, slot, k), old)
    got = tf.adv_norm_per_map(a, slot, k, scale_of=gae)
    for m in range(k):
        r = slot == m
        want = (a[r].double() - a[r].double().mean()) / (gae[r].double().std() + 1e-8)
        assert torch.allclose(got[r].double(), want, atol=1e-6)
    assert got.std() < 0.5                           # A's own scale, not unit
    # a map below min_rows falls back to the POOLED moments: a's mean, the GAE std
    few = tf.adv_norm_per_map(a, slot, k, min_rows=10 ** 6, scale_of=gae)
    assert torch.allclose(few.double(), (a64 - a64.mean()) / (gae.double().std() + 1e-8),
                          atol=1e-6)


# ==========================================================================
# (d) the model
# ==========================================================================
def _pol(seed, **kw):
    torch.manual_seed(seed)
    return Policy(N_SCALAR + 16 * 8, 16, 8, emb=64, hidden=64, view_continuous=True,
                  view_absolute="velocity", **kw).eval()


def test_adv_head_is_last_and_the_rest_is_the_flag_off_model():
    a, b = _pol(5), _pol(5, dae=True)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) - set(sa) == {"adv_head.weight", "adv_head.bias"}
    for k, v in sa.items():
        assert torch.equal(v, sb[k]), k
    assert list(sb)[-2:] == ["adv_head.weight", "adv_head.bias"]
    assert [n for n, _ in b.named_parameters()][-2:] == ["adv_head.weight", "adv_head.bias"]
    assert tuple(sb["adv_head.weight"].shape) == (96 * 5, 64)
    torch.manual_seed(11)
    x = torch.randn(5, N_SCALAR + 16 * 8)
    with torch.no_grad():
        la, va = a(x)
        lb, vb = b(x)
        l3, v3, g3 = b.forward_split(x[:, :b.scal_dim], x[:, b.scal_dim:], adv=True)
    assert torch.equal(la, lb) and torch.equal(va, vb) and va.shape == (5,)
    assert torch.equal(l3, lb) and torch.equal(v3, vb) and g3.shape == (5, 96 * 5)
    with pytest.raises(SystemExit, match="view-continuous"):
        torch.manual_seed(5)
        Policy(N_SCALAR + 16 * 8, 16, 8, emb=64, hidden=64, dae=True)
    with pytest.raises(SystemExit, match="dae"):
        _pol(5, dae=True, pay_split=True)


# ==========================================================================
# (e) the tools
# ==========================================================================
def _literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and isinstance(n.value, str)}


@pytest.mark.parametrize("rel", ["tools/record_ckpt.py", "tools/beam_tas.py"])
def test_tools_mention_the_key(rel):
    assert "dae" in _literals(ROOT / rel)


# ==========================================================================
# (f) trainer smokes
# ==========================================================================
import v14_smoke as V                                          # noqa: E402

KINDS5 = ("goal", "goalc", "sdf", "occ", "slabocc")            # no face-id: --obs-normal 0
MAPS = V.MM_MAPS[:2]
needs_two = pytest.mark.skipif(
    not (V.DLL.exists() and len(MAPS) == 2 and all(
        m.exists() and all(m.with_suffix(f".{k}_32.npz").exists() for k in KINDS5)
        for m in MAPS)), reason="MM_TEST_MAPS: two maps with their cell-32 caches")
SM = ["--obs-normal", "0", "--envs", "64", "--maxvel", "4000"]
ARCH = ["--wd", "0.01", "--wd-skip-norms", "1", "--conv-norm", "1", "--final-ln-affine", "0",
        "--mlp-act", "leaky"]


def _maps():
    return ["--maps", ",".join(str(m) for m in MAPS)]


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


@needs_two
def test_flag_on_trains_resumes_and_refuses():
    r, d = V.run_line(V.ROOT, "dae_on", _maps(), steps=8 * 64 * 12,
                      extra=SM + ARCH + ["--dae", "1.5"], timeout=5400)
    out = r.stdout + r.stderr
    try:
        assert r.returncode == 0, out[-5000:]
        assert "--dae 1.5: Direct Advantage Estimation" in out
        cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
        assert cfg["dae"] == 1.5
        sys.path.insert(0, str(V.ROOT / "tools"))
        import record_ckpt
        record_ckpt.audit_cfg(cfg)
        ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
        assert "adv_head.weight" in ck["policy"] and ck["config"]["dae"] == 1.5
        head, rows = _csv(d / "progress.csv")
        tags = [h.split(".", 1)[1] for h in head if h.startswith("dae/adv_std.")]
        assert len(tags) == 2 and len(rows) >= 3
        for c in ["dae/adv_std", "dae/adv_abs", "dae/ret", "dae/ev", "dae/adv_vs_gae"]:
            vals = [row[head.index(c)] for row in rows]
            assert all(v != "" and np.isfinite(float(v)) for v in vals), (c, vals)
        assert float(rows[-1][head.index("dae/adv_std")]) > 0.0
        # the resumes run WITHOUT round 2's --wd-skip-norms (that optimiser does not resume on
        # this branch, flag on or off - pre-existing); the run's own search spine is restored
        # from its config: policy-derived, SELF_STATES=1 declares it
        r1, d1 = V.run_line(V.ROOT, "dae_adam", _maps(), steps=8 * 64 * 12,
                            extra=SM + ["--dae", "1.5"], timeout=5400)
        assert r1.returncode == 0, (r1.stdout + r1.stderr)[-5000:]
        os.environ["SELF_STATES"] = "1"
        r2, d2 = V.run_line(V.ROOT, "dae_re", _maps(), steps=8 * 64 * 16,
                            extra=SM + ["--ckpt", str(d1 / "ckpt_final.pt")], timeout=5400)
        assert r2.returncode == 0, (r2.stdout + r2.stderr)[-5000:]
        assert "dae=1.5" in r2.stdout + r2.stderr
        ck2 = torch.load(d2 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        assert "adv_head.weight" in ck2["policy"] and ck2["config"]["dae"] == 1.5
        r3, d3 = V.run_line(V.ROOT, "dae_off", _maps(), steps=8 * 64 * 16,
                            extra=SM + ["--ckpt", str(d1 / "ckpt_final.pt"), "--dae", "0"],
                            timeout=5400)
        assert r3.returncode != 0 and "trained with --dae" in r3.stdout + r3.stderr
        r4, d4 = V.run_line(V.ROOT, "dae_pd", _maps(), steps=2048,
                            extra=SM + ["--dae", "1.5", "--critic-pay-d0", "100000"])
        assert r4.returncode != 0 and "--critic-pay-d0" in r4.stdout + r4.stderr
    finally:
        os.environ.pop("SELF_STATES", None)
        for name in ("dae_on", "dae_adam", "dae_re", "dae_off", "dae_pd"):
            shutil.rmtree(V.ROOT / "runs" / name, ignore_errors=True)


def _parent_tree(dst):
    pre = os.environ.get("DAE_PARENT_TREE")
    if pre:
        return Path(pre)
    dst.mkdir(parents=True, exist_ok=True)
    tar = dst / "base.tar"
    subprocess.run(["git", "-C", str(V.ROOT), "archive", "--format=tar", "-o", str(tar), PARENT,
                    "python"], check=True, capture_output=True)
    with tarfile.open(tar) as t:
        t.extractall(dst)
    return dst


@needs_two
def test_flag_off_is_bit_identical_to_the_parent_commit(tmp_path):
    base = _parent_tree(tmp_path / "base")
    r0, d0 = V.run_line(base, "dae_id", _maps(), extra=SM + ARCH)
    assert r0.returncode == 0, (r0.stdout + r0.stderr)[-4000:]
    r1, d1 = V.run_line(V.ROOT, "dae_id", _maps(), extra=SM + ARCH)
    assert r1.returncode == 0, (r1.stdout + r1.stderr)[-4000:]
    try:
        ca = json.loads((d0 / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((d1 / "run.json").read_text(encoding="utf-8"))["config"]
        for c in (ca, cb):
            c.pop("demo_file", None)                 # the spine path names the checkout
        assert ca == cb, sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
        assert "dae" not in cb
        ta, tb = sorted(d0.glob("traj_*.jsonl")), sorted(d1.glob("traj_*.jsonl"))
        assert len(ta) >= 4 and [p.name for p in ta] == [p.name for p in tb]
        for x, y in zip(ta, tb):
            assert x.read_bytes() == y.read_bytes(), x.name
        ha, ra = _csv(d0 / "progress.csv")
        hb, rb = _csv(d1 / "progress.csv")
        assert ha == hb
        fps = ha.index("time/fps")
        assert len(ra) == len(rb) >= 5
        for x, y in zip(ra, rb):
            assert x[:fps] + x[fps + 1:] == y[:fps] + y[fps + 1:]
        sa = torch.load(d0 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(d1 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        assert sa["policy"].keys() == sb["policy"].keys()
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        if not os.environ.get("DAE_PARENT_TREE"):
            shutil.rmtree(d0, ignore_errors=True)
