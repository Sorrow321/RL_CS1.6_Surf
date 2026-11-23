"""--vmpo (V-MPO, Song et al. 2019; the user, 2026-11-23: replace PPO): the policy loss is the
E-step / M-step pair on the top half of each minibatch's advantages, with a learned temperature
and a KL(pi_old || pi) trust region per part (key heads, view mean, view sigma), pi_old being the
stored behaviour distribution.

(a) vmpo_kls: the categorical part is KL(p_old || p_new) per head summed (pads and masked bins add
    0); the Gaussian's decoupled pair - with the old sigma kept the mean part is the full KL, with
    the old mean kept the sigma part is - against torch.distributions.
(b) trainer smokes (CPU, recipe v14 at smoke scale on two maps, MM_TEST_MAPS): the flag ON trains
    (config keys, the vmpo/* columns finite, the duals moved off their init, record_ckpt's audit),
    a resume restores the duals, --vmpo with --dae is refused; the flag OFF is bit-identical to the
    PARENT commit (config, every eval trajectory, progress.csv minus fps, the final weights).
"""
from __future__ import annotations

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
from torch.distributions import Categorical, Normal, kl_divergence   # noqa: E402

import train_fast as tf                                        # noqa: E402

PARENT = "e4f6d79b"            # mm/dae head the flag was built on (mm/algo's base)


def test_the_kls_are_the_papers():
    g = torch.Generator().manual_seed(1)
    B, H, P, n_z = 7, 4, tf.NPAD, 2
    widths = (4, 4, 2, 3)
    lo = torch.full((B, H, P), tf.NEG)
    ln = torch.full((B, H, P), tf.NEG)
    want = torch.zeros(B, dtype=torch.float64)
    for h, w in enumerate(widths):
        a = torch.randn(B, w, generator=g)
        b = torch.randn(B, w, generator=g)
        if h == 1:
            a[:, 0] = tf.NEG                     # a masked bin: probability 0 under both
            b[:, 0] = tf.NEG
        lo[:, h, :w] = torch.log_softmax(a, -1)
        ln[:, h, :w] = torch.log_softmax(b, -1)
        po, pn = lo[:, h, :w].exp().double(), ln[:, h, :w].exp().double()
        want += kl_divergence(Categorical(probs=po), Categorical(probs=pn))
    mu_o, mu_n = torch.randn(B, n_z, generator=g), torch.randn(B, n_z, generator=g)
    ls_o, ls_n = torch.randn(B, n_z, generator=g) * 0.3 - 1.0, torch.randn(B, n_z, generator=g) * 0.3 - 1.0
    kc, km, ks = tf.vmpo_kls(lo, ln, mu_o, mu_n, ls_o, ls_n)
    assert torch.allclose(kc.double(), want, atol=1e-5)
    assert (kc >= -1e-6).all()
    # the decoupled pair: sigma kept -> the mean part is the whole KL; mean kept -> the sigma part
    _, km2, ks2 = tf.vmpo_kls(lo, ln, mu_o, mu_n, ls_o, ls_o)
    full_m = kl_divergence(Normal(mu_o, ls_o.exp()), Normal(mu_n, ls_o.exp())).sum(-1)
    assert torch.allclose(km2, full_m, atol=1e-5) and torch.allclose(ks2, torch.zeros(B), atol=1e-6)
    _, km3, ks3 = tf.vmpo_kls(lo, ln, mu_o, mu_o, ls_o, ls_n)
    full_s = kl_divergence(Normal(mu_o, ls_o.exp()), Normal(mu_o, ls_n.exp())).sum(-1)
    assert torch.allclose(ks3, full_s, atol=1e-5) and torch.allclose(km3, torch.zeros(B), atol=1e-6)
    # in general: KL_full = KL_sigma + (mu - mu_old)^2 / (2 s_new^2)
    full = kl_divergence(Normal(mu_o, ls_o.exp()), Normal(mu_n, ls_n.exp())).sum(-1)
    assert torch.allclose(full, ks + (0.5 * (mu_n - mu_o).pow(2) / (2 * ls_n).exp()).sum(-1),
                          atol=1e-5)


# ==========================================================================
# (b) trainer smokes
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
COLS = ["vmpo/eta", "vmpo/alpha", "vmpo/alpha_mu", "vmpo/alpha_sigma", "vmpo/kl", "vmpo/kl_mu",
        "vmpo/kl_sigma", "vmpo/psi_max"]


def _maps():
    return ["--maps", ",".join(str(m) for m in MAPS)]


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


@needs_two
def test_flag_on_trains_resumes_and_refuses():
    r, d = V.run_line(V.ROOT, "vmpo_on", _maps(), steps=8 * 64 * 12,
                      extra=SM + ARCH + ["--vmpo", "1"], timeout=5400)
    out = r.stdout + r.stderr
    try:
        assert r.returncode == 0, out[-5000:]
        assert "--vmpo: V-MPO's policy loss in place of PPO's" in out
        cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
        assert cfg["vmpo"] == 1 and cfg["vmpo_eps_eta"] == 0.02 and cfg["vmpo_eps_sigma"] == 1e-5
        sys.path.insert(0, str(V.ROOT / "tools"))
        import record_ckpt
        record_ckpt.audit_cfg(cfg)
        ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
        assert ck["vmpo"]["raw"].shape == (4,)
        duals = (torch.nn.functional.softplus(ck["vmpo"]["raw"]) + 1e-8).tolist()
        assert all(np.isfinite(duals)) and not np.allclose(duals, [1.0, 5.0, 1.0, 1.0], atol=1e-4)
        head, rows = _csv(d / "progress.csv")
        assert len(rows) >= 3
        for c in COLS:
            vals = [row[head.index(c)] for row in rows]
            assert all(v != "" and np.isfinite(float(v)) for v in vals), (c, vals)
        assert all(float(row[head.index("vmpo/kl")]) >= -1e-6 for row in rows)
        # a resume restores the duals (without round 2's --wd-skip-norms: that optimiser does not
        # resume on this line, pre-existing); the run's own spine is policy-derived
        r1, d1 = V.run_line(V.ROOT, "vmpo_adam", _maps(), steps=8 * 64 * 12,
                            extra=SM + ["--vmpo", "1"], timeout=5400)
        assert r1.returncode == 0, (r1.stdout + r1.stderr)[-5000:]
        os.environ["SELF_STATES"] = "1"
        r2, d2 = V.run_line(V.ROOT, "vmpo_re", _maps(), steps=8 * 64 * 16,
                            extra=SM + ["--ckpt", str(d1 / "ckpt_final.pt")], timeout=5400)
        o2 = r2.stdout + r2.stderr
        assert r2.returncode == 0, o2[-5000:]
        assert "vmpo=1" in o2 and "--vmpo: restored the temperature and the multipliers" in o2
        r3, d3 = V.run_line(V.ROOT, "vmpo_dae", _maps(), steps=2048,
                            extra=SM + ["--vmpo", "1", "--dae", "1.5"])
        assert r3.returncode != 0 and "--vmpo is not implemented with --dae" in r3.stdout + r3.stderr
    finally:
        os.environ.pop("SELF_STATES", None)
        for name in ("vmpo_on", "vmpo_adam", "vmpo_re", "vmpo_dae"):
            shutil.rmtree(V.ROOT / "runs" / name, ignore_errors=True)


def _parent_tree(dst):
    pre = os.environ.get("VMPO_PARENT_TREE")
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
    r0, d0 = V.run_line(base, "vmpo_id", _maps(), extra=SM + ARCH)
    assert r0.returncode == 0, (r0.stdout + r0.stderr)[-4000:]
    r1, d1 = V.run_line(V.ROOT, "vmpo_id", _maps(), extra=SM + ARCH)
    assert r1.returncode == 0, (r1.stdout + r1.stderr)[-4000:]
    try:
        ca = json.loads((d0 / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((d1 / "run.json").read_text(encoding="utf-8"))["config"]
        for c in (ca, cb):
            c.pop("demo_file", None)
        assert ca == cb, sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
        assert "vmpo" not in cb
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
        if not os.environ.get("VMPO_PARENT_TREE"):
            shutil.rmtree(d0, ignore_errors=True)
