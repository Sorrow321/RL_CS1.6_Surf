"""--offpolicy sac|td3 (the user, 2026-11-23: replace PPO): a hybrid off-policy actor-critic - the
Policy as actor, twin Q heads per joint key bin over u = tanh(z) on their own encoder, a replay
ring of decisions, n-step targets.

(a) OffReplay: the n-step return stops at the first end and drops the bootstrap there; the
    successor of slot k is k + 1 across a rollout boundary; nothing samples a slot whose k + n is
    not written; the ring overwrites the oldest.
(b) joint_logp is the C-order joint of independent heads (dae_advantage's index), normalised.
(c) trainer smokes (CPU, recipe v14 at smoke scale on two maps, MM_TEST_MAPS): sac and td3 train
    (config keys, the op/* columns, gradient steps taken, a checkpoint with the critic, the actor
    the Policy, record_ckpt's audit), a resume is refused; the flag OFF is bit-identical to the
    PARENT commit.
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

import train_fast as tf                                        # noqa: E402

PARENT = "b6034dea"            # mm/algo with --vmpo, the commit the flag was built on


def _ring(C=6, N=3):
    rp = tf.OffReplay(C, N, 2, 4, torch.float32, 4, 2, torch.device("cpu"))
    return rp


def test_replay_n_step_targets_and_the_ring():
    rp = _ring()
    T = 4
    scal = torch.arange(T, dtype=torch.float32)[:, None, None].expand(T, 3, 2).clone()
    rew = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [8.0, 8.0, 8.0]])
    done = torch.zeros(T, 3)
    done[1, 1] = 1.0                              # env 1 ends after its decision 1
    rp.add(scal, torch.zeros(T, 3, 4), torch.zeros(T, 3, 4, dtype=torch.long),
           torch.zeros(T, 3, 2), rew, done)
    assert rp.sample(16, 4, 0.5) is None           # k + 4 never written yet
    torch.manual_seed(0)
    for _ in range(20):
        sc, im, keys, z, R, alive, gn, sc2, im2 = rp.sample(64, 2, 0.5)
        k = sc[:, 0].long()
        assert (k <= T - 1 - 2).all()
        assert torch.allclose(sc2[:, 0], (k + 2).float())   # the successor two decisions on
        assert gn == 0.25
    # hand-check R and alive per env and start
    torch.manual_seed(1)
    seen = set()
    for _ in range(200):
        sc, im, keys, z, R, alive, gn, sc2, im2 = rp.sample(8, 2, 0.5)
        # recover the env from the reward pattern is impossible; check the formula instead
        for b in range(8):
            k = int(sc[b, 0])
            seen.add(k)
            assert float(R[b]) in (rew[k, 0] + 0.5 * rew[k + 1, 0], rew[k, 1],
                                   rew[k, 1] + 0.5 * rew[k + 1, 1])
    assert seen == {0, 1}
    # env 1 starting at 0: r0 + 0.5 r1 then the end at 1 -> no bootstrap; at 1: r1, ended
    rp2 = _ring(C=6, N=1)
    d2 = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    r2 = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
    rp2.add(scal[:, :1], torch.zeros(T, 1, 4), torch.zeros(T, 1, 4, dtype=torch.long),
            torch.zeros(T, 1, 2), r2, d2)
    got = {}
    for _ in range(100):
        sc, im, keys, z, R, alive, gn, sc2, im2 = rp2.sample(4, 2, 0.5)
        for b in range(4):
            got[int(sc[b, 0])] = (float(R[b]), float(alive[b]))
    assert got[0] == (1.0 + 0.5 * 2.0, 0.0)        # ended inside: no bootstrap
    assert got[1] == (2.0, 0.0)                    # ended at once: the rest masked
    # the ring: 6 slots, write 4 more -> the oldest 2 overwritten, the cursor wraps
    rp2.add(scal[:, :1] + 10, torch.zeros(T, 1, 4), torch.zeros(T, 1, 4, dtype=torch.long),
            torch.zeros(T, 1, 2), r2, torch.zeros(T, 1))
    assert rp2.w == 8
    ks = set()
    for _ in range(200):
        sc, *_ = rp2.sample(4, 1, 0.5)
        ks.update(int(v) for v in sc[:, 0])
    assert ks == {2, 3, 10, 11, 12}               # slots 2..6 (decisions w-C .. w-1-n)


def test_joint_logp_is_the_c_order_joint():
    g = torch.Generator().manual_seed(3)
    W = (4, 4, 2, 3)
    lsm = torch.full((5, 4, tf.NPAD), tf.NEG)
    for h, w in enumerate(W):
        lsm[:, h, :w] = torch.log_softmax(torch.randn(5, w, generator=g), -1)
    lj = tf.joint_logp(lsm, W)
    assert lj.shape == (5, 96)
    assert torch.allclose(lj.exp().sum(-1), torch.ones(5), atol=1e-5)
    a = (2, 3, 1, 2)
    j = ((a[0] * W[1] + a[1]) * W[2] + a[2]) * W[3] + a[3]
    want = sum(lsm[:, h, a[h]] for h in range(4))
    assert torch.allclose(lj[:, j], want, atol=1e-6)


# ==========================================================================
# (c) trainer smokes
# ==========================================================================
import v14_smoke as V                                          # noqa: E402

KINDS5 = ("goal", "goalc", "sdf", "occ", "slabocc")
MAPS = V.MM_MAPS[:2]
needs_two = pytest.mark.skipif(
    not (V.DLL.exists() and len(MAPS) == 2 and all(
        m.exists() and all(m.with_suffix(f".{k}_32.npz").exists() for k in KINDS5)
        for m in MAPS)), reason="MM_TEST_MAPS: two maps with their cell-32 caches")
SM = ["--obs-normal", "0", "--envs", "64", "--maxvel", "4000"]
ARCH = ["--wd", "0.01", "--wd-skip-norms", "1", "--conv-norm", "1", "--final-ln-affine", "0",
        "--mlp-act", "leaky"]
OP = ["--op-buffer", "32", "--op-batch", "128", "--op-updates", "3", "--op-nstep", "2"]
COLS = ["op/q", "op/critic_loss", "op/actor_loss", "op/alpha_keys", "op/alpha_view",
        "op/ent_keys", "op/ent_view", "op/value_loss", "op/updates", "op/replay"]


def _maps():
    return ["--maps", ",".join(str(m) for m in MAPS)]


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


@needs_two
@pytest.mark.parametrize("kind", ["sac", "td3"])
def test_offpolicy_trains(kind):
    name = f"op_{kind}"
    r, d = V.run_line(V.ROOT, name, _maps(), steps=8 * 64 * 12,
                      extra=SM + ARCH + ["--offpolicy", kind] + OP, timeout=5400)
    out = r.stdout + r.stderr
    try:
        assert r.returncode == 0, out[-6000:]
        assert f"--offpolicy {kind}: a hybrid off-policy actor-critic in place of PPO" in out
        cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
        assert cfg["offpolicy"] == kind and cfg["op_batch"] == 128
        sys.path.insert(0, str(V.ROOT / "tools"))
        import record_ckpt
        record_ckpt.audit_cfg(cfg)
        ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
        assert "q" in ck["offp"] and ck["offp"]["n_upd"] > 0
        assert "view_head.weight" in ck["policy"]           # the actor is the Policy
        head, rows = _csv(d / "progress.csv")
        assert len(rows) >= 3
        last = rows[-1]
        for c in COLS:
            v = last[head.index(c)]
            assert v != "" and np.isfinite(float(v)), (c, v)
        assert float(last[head.index("op/updates")]) > 0
        if kind == "td3":
            assert abs(float(ck["policy"]["view_std.log_std"][0]) - np.log(0.15)) < 1e-6
        # an off-policy arm does not resume
        os.environ["SELF_STATES"] = "1"
        r2, _ = V.run_line(V.ROOT, name + "_re", _maps(), steps=2048,
                           extra=SM + ["--offpolicy", kind, "--ckpt", str(d / "ckpt_final.pt")])
        assert r2.returncode != 0 and "a resume (--ckpt)" in r2.stdout + r2.stderr
    finally:
        os.environ.pop("SELF_STATES", None)
        for nm in (name, name + "_re"):
            shutil.rmtree(V.ROOT / "runs" / nm, ignore_errors=True)


def _parent_tree(dst):
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
    r0, d0 = V.run_line(base, "op_id", _maps(), extra=SM + ARCH)
    assert r0.returncode == 0, (r0.stdout + r0.stderr)[-4000:]
    r1, d1 = V.run_line(V.ROOT, "op_id", _maps(), extra=SM + ARCH)
    assert r1.returncode == 0, (r1.stdout + r1.stderr)[-4000:]
    try:
        ca = json.loads((d0 / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((d1 / "run.json").read_text(encoding="utf-8"))["config"]
        for c in (ca, cb):
            c.pop("demo_file", None)
        assert ca == cb, sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
        ta, tb = sorted(d0.glob("traj_*.jsonl")), sorted(d1.glob("traj_*.jsonl"))
        assert len(ta) >= 4 and [p.name for p in ta] == [p.name for p in tb]
        for x, y in zip(ta, tb):
            assert x.read_bytes() == y.read_bytes(), x.name
        ha, ra = _csv(d0 / "progress.csv")
        hb, rb = _csv(d1 / "progress.csv")
        assert ha == hb
        fps = ha.index("time/fps")
        for x, y in zip(ra, rb):
            assert x[:fps] + x[fps + 1:] == y[:fps] + y[fps + 1:]
        sa = torch.load(d0 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(d1 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d0, ignore_errors=True)
