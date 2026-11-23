"""--critic-pay-d0 (the user, 2026-11-23): the PAY stays 100/d0 per unit on every map, and the
critic predicts in one unit on every map - two value heads, V_P (the progress part of the return
at 100/D per unit, the same everywhere) and V_R (the rest: the finish bonus, the time cost, the
energy term, the novelty), V(s) = k V_P + V_R with k = the map's own pay rate over 100/D.

(a) RaceReward(prog_split=True): r is the control's bit for bit, and prog_r is r's coefficient of
    `scale` - the same reward at twice the scale differs by exactly scale x prog_r, and prog_r does
    not move with the scale - on the v14 terms (ratchet, novelty, energy) and on every other term
    the pay rate multiplies (ng tax and charges, the death charge, speed-equiv, curiosity-cond).
(b) the two GAE streams: on a hand case with an episode end, the combined advantage is exactly
    k adv_P + adv_R and R = k R_P + R_R; the trainer's source carries the progress recursion WITH
    the nonterm mask.
(c) Policy: prog_head is registered LAST (every other tensor is the flag-off model's under the
    same seed), the state_dict gains exactly its two keys, the value output is (B, 2) with column
    0 the flag-off value bit for bit; --int-split with it is refused.
(d) the tools: record_ckpt and beam_tas mention the key (the strict loads build prog_head);
    az_mcts.value_k is D / d0 under the default pay and None without the flag.
(e) trainer smokes (CPU, recipe v14 at smoke scale on two maps, MM_TEST_MAPS): the flag ON trains
    (config key, the pay/* columns, a checkpoint with prog_head, record_ckpt's audit), a resume
    restores it, a resume with --critic-pay-d0 0 is refused, --race-pay-d0 with it is refused; the
    flag OFF is bit-identical to the PARENT commit (config, every eval trajectory, progress.csv
    minus fps).
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

from surfgym.rewards import RaceReward                         # noqa: E402
from surfgym.goalfield import EuclidField                      # noqa: E402
from train_fast import N_SCALAR, Policy                        # noqa: E402
from test_curiosity_cond import FakeCore, GOAL, _step          # noqa: E402

PARENT = "97653833"            # mm/health head the flag was built on
D = 1e5


# ==========================================================================
# (a) the reward: r untouched, prog_r its exact coefficient of scale
# ==========================================================================
def _script(n, k, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        d = rng.integers(-3, 4, size=(n, 3)) * 24.0
        spd = rng.uniform(0.0, 3000.0, size=n)
        vz = rng.uniform(-800.0, 400.0, size=n)
        done = rng.random(n) < 0.08
        goal = done & (rng.random(n) < 0.3)
        out.append((d, spd, vz, done, goal))
    return out


def _drive(rw, core, script, energy=0.0):
    if energy:
        rw.energy_pot = energy
    for i in range(core.num_envs):
        core.states_view["origin"][i] = (-11000.0 + 300.0 * i, 7487.0 + 9000.0, -1000.0)
    rs, ps = [], []
    for d, spd, vz, done, goal in script:
        o = core.states_view["origin"]
        o[:, 0] += d[:, 0]
        o[:, 1] += d[:, 1]
        o[:, 2] += d[:, 2]
        core.states_view["velocity"][:, 0] = spd
        core.states_view["velocity"][:, 2] = vz
        core.goal_hits[:] = goal.astype(np.uint8)
        r = np.asarray(_step(rw, core, done=done.astype(np.uint8)), np.float32)
        rs.append(r.copy())
        ps.append(np.zeros(len(r)) if rw.prog_r is None else rw.prog_r.copy())
    return np.stack(rs), np.stack(ps)


CASES = [
    dict(ratchet=True, int_coef=2.5, int_view=0, int_speed=8),             # v14's terms
    dict(ratchet=True, death_charge=0.5, ng_d0=20000.0),
    dict(ng=2, ng_d0=20000.0, ng_gamma=0.999),
    dict(ng=1, ng_d0=20000.0, ng_gamma=0.999),
    dict(speed_equiv=0.5, fail_pen=1.0),
    dict(ratchet=True, cc_tmax=2.0, int_coef=1.0),
]


@pytest.mark.parametrize("kw", CASES)
def test_prog_r_is_the_coefficient_of_scale_and_r_is_untouched(kw):
    n, s = 6, 100.0 / 198380.0
    script = _script(n, 60, seed=3)
    ra, _ = _drive(RaceReward(EuclidField(GOAL), scale=s, **kw), FakeCore(n), script, 3.0)
    rb, pb = _drive(RaceReward(EuclidField(GOAL), scale=s, prog_split=True, **kw),
                    FakeCore(n), script, 3.0)
    rc, pc = _drive(RaceReward(EuclidField(GOAL), scale=2.0 * s, prog_split=True, **kw),
                    FakeCore(n), script, 3.0)
    assert np.array_equal(ra, rb)                       # the flag moves nothing in r
    assert np.array_equal(pb, pc)                       # prog_r is in map units, not pay
    assert np.abs(pb).sum() > 0.0
    np.testing.assert_allclose(rc.astype(np.float64) - rb, s * pb, rtol=0, atol=2e-5)
    # the rest of r does not see the pay rate
    np.testing.assert_allclose(rb - s * pb, rc - 2.0 * s * pc, rtol=0, atol=2e-5)


def test_prog_r_is_rebuilt_every_call_and_arc_is_refused():
    rw = RaceReward(EuclidField(GOAL), scale=1e-3, prog_split=True, ratchet=True)
    core = FakeCore(2)
    core.states_view["origin"][:] = (-11000.0, 7487.0 + 5000.0, -1000.0)
    _step(rw, core)                                      # on_reset call: zeros
    assert rw.prog_r is not None and not rw.prog_r.any()
    core.states_view["origin"][:, 1] -= 50.0
    _step(rw, core)
    assert np.allclose(rw.prog_r, 50.0)
    _step(rw, core)                                      # no move: the coefficient is rebuilt
    assert not rw.prog_r.any()
    with pytest.raises(ValueError, match="arc"):
        RaceReward(EuclidField(GOAL), scale=1e-3, prog_split=True, arc=object(), arc_scale=1.0)


# ==========================================================================
# (b) the two streams
# ==========================================================================
def _gae(rew, val, last, g, lam, done):
    adv = np.zeros_like(rew)
    lg = np.zeros(rew.shape[1])
    for t in reversed(range(len(rew))):
        nv = last if t == len(rew) - 1 else val[t + 1]
        nt = 1.0 - done[t]
        delta = rew[t] + g * nv * nt - val[t]
        lg = delta + g * lam * nt * lg
        adv[t] = lg
    return adv


def test_the_combined_advantage_is_k_adv_p_plus_adv_r():
    rng = np.random.default_rng(0)
    T, N = 9, 4
    k = np.array([0.5, 2.78, 37.4, 1.0])
    q, o = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    vq, vr = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    lq, lr = rng.normal(size=N), rng.normal(size=N)
    done = np.zeros((T, N))
    done[3, 1] = done[6, 2] = 1.0
    g, lam = 0.998, 0.95
    adv = _gae(k * q + o, k * vq + vr, k * lq + lr, g, lam, done)
    aq, ar = _gae(q, vq, lq, g, lam, done), _gae(o, vr, lr, g, lam, done)
    np.testing.assert_allclose(adv, k * aq + ar, atol=1e-12)
    ret, rq = adv + k * vq + vr, aq + vq
    np.testing.assert_allclose(ret - k * rq, ar + vr, atol=1e-12)     # R_R, V_R's target


def test_trainer_source_runs_the_progress_recursion_with_the_cut():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    i = src.index("---- --critic-pay-d0: the progress stream's GAE")
    block = src[i:i + 2500]
    assert "delta_q = b_q[t] + g_eff * nextv_q * nonterm - b_vq[t]" in block
    assert "lastgae_q = delta_q + g_eff * args.gae * nonterm * lastgae_q" in block
    assert "ret_r = ret - K_ENV[None, :] * ret_q" in block
    # the rollout writes the combined value where V always went, and V_P to its own buffer
    assert "static_val.copy_(value[:, 0].float() + K_ENV * value[:, 1].float())" in src
    assert "q[ti] += (GAMMA_T * _tv[:, 1]).astype(np.float32)" in src


# ==========================================================================
# (c) the model
# ==========================================================================
def _pol(seed, **kw):
    torch.manual_seed(seed)
    return Policy(N_SCALAR + 16 * 8, 16, 8, emb=64, hidden=64, **kw).eval()


def test_prog_head_is_last_and_the_rest_is_the_flag_off_model():
    a, b = _pol(5), _pol(5, pay_split=True)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) - set(sa) == {"prog_head.weight", "prog_head.bias"}
    for k, v in sa.items():
        assert torch.equal(v, sb[k]), k
    assert list(sb)[-2:] == ["prog_head.weight", "prog_head.bias"]
    assert [n for n, _ in b.named_parameters()][-2:] == ["prog_head.weight", "prog_head.bias"]
    torch.manual_seed(11)
    x = torch.randn(5, N_SCALAR + 16 * 8)
    with torch.no_grad():
        _, va = a(x)
        _, vb = b(x)
    assert va.shape == (5,) and vb.shape == (5, 2)
    assert torch.equal(vb[:, 0], va)
    with pytest.raises(SystemExit, match="int-split"):
        _pol(5, pay_split=True, int_split=True)


# ==========================================================================
# (d) the tools
# ==========================================================================
def _literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and isinstance(n.value, str)}


@pytest.mark.parametrize("rel", ["tools/record_ckpt.py", "tools/beam_tas.py", "tools/az_mcts.py"])
def test_tools_mention_the_key(rel):
    assert "critic_pay_d0" in _literals(ROOT / rel)


def test_value_k_is_the_exchange_rate():
    sys.path.insert(0, str(ROOT / "tools"))
    import az_mcts
    assert az_mcts.value_k({}, 2506.0) is None
    assert az_mcts.value_k({"critic_pay_d0": D}, 2506.0) == pytest.approx(D / 2506.0)
    assert az_mcts.value_k({"critic_pay_d0": D}, 198991.8) == pytest.approx(0.50253, rel=1e-4)
    with pytest.raises(SystemExit):
        az_mcts.value_k({"critic_pay_d0": D}, float("nan"))


# ==========================================================================
# (e) trainer smokes
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
    r, d = V.run_line(V.ROOT, "paysplit_on", _maps(), steps=8 * 64 * 12,
                      extra=SM + ARCH + ["--critic-pay-d0", "100000"], timeout=5400)
    out = r.stdout + r.stderr
    try:
        assert r.returncode == 0, out[-5000:]
        assert "--critic-pay-d0 100,000: the PAY is unchanged" in out
        assert "--critic-pay-d0: k = map pay rate / critic rate:" in out
        cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
        assert cfg["critic_pay_d0"] == D and not cfg.get("race_pay_d0")
        sys.path.insert(0, str(V.ROOT / "tools"))
        import record_ckpt
        record_ckpt.audit_cfg(cfg)
        ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
        assert "prog_head.weight" in ck["policy"] and ck["config"]["critic_pay_d0"] == D
        head, rows = _csv(d / "progress.csv")
        tags = [h.split(".", 1)[1] for h in head if h.startswith("pay/ev_q.")]
        assert len(tags) == 2 and len(rows) >= 3
        for c in ["pay/ret_q", "pay/v_q", "pay/ev_q", "pay/ev_r"]:
            vals = [row[head.index(c)] for row in rows]
            assert all(v != "" and np.isfinite(float(v)) for v in vals), (c, vals)
        # the resumes run WITHOUT round 2's --wd-skip-norms: that optimiser's two parameter
        # groups do not resume on this branch at all, flag on or off (widen_for_route reads the
        # Adam state in parameters() order), which is not this flag's to fix. The run's own
        # search spine is restored from its config: policy-derived, SELF_STATES=1 declares it.
        r1, d1 = V.run_line(V.ROOT, "paysplit_adam", _maps(), steps=8 * 64 * 12,
                            extra=SM + ["--critic-pay-d0", "100000"], timeout=5400)
        assert r1.returncode == 0, (r1.stdout + r1.stderr)[-5000:]
        os.environ["SELF_STATES"] = "1"
        # a bare resume restores the flag; --critic-pay-d0 0 on it is refused
        r2, d2 = V.run_line(V.ROOT, "paysplit_re", _maps(), steps=8 * 64 * 16,
                            extra=SM + ["--ckpt", str(d1 / "ckpt_final.pt")], timeout=5400)
        assert r2.returncode == 0, (r2.stdout + r2.stderr)[-5000:]
        assert "critic_pay_d0=100000" in r2.stdout + r2.stderr
        ck2 = torch.load(d2 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        assert "prog_head.weight" in ck2["policy"] and ck2["config"]["critic_pay_d0"] == D
        r3, d3 = V.run_line(V.ROOT, "paysplit_off", _maps(), steps=8 * 64 * 16,
                            extra=SM + ["--ckpt", str(d1 / "ckpt_final.pt"),
                                        "--critic-pay-d0", "0"], timeout=5400)
        assert r3.returncode != 0 and "trained with --critic-pay-d0" in r3.stdout + r3.stderr
        r4, d4 = V.run_line(V.ROOT, "paysplit_pd", _maps(), steps=2048,
                            extra=SM + ["--critic-pay-d0", "100000", "--race-pay-d0", "100000"])
        assert r4.returncode != 0 and "--race-pay-d0" in r4.stdout + r4.stderr
    finally:
        os.environ.pop("SELF_STATES", None)
        for name in ("paysplit_on", "paysplit_adam", "paysplit_re", "paysplit_off",
                     "paysplit_pd"):
            shutil.rmtree(V.ROOT / "runs" / name, ignore_errors=True)


def _parent_tree(dst):
    pre = os.environ.get("PAYSPLIT_PARENT_TREE")
    if pre:
        return Path(pre)
    dst.mkdir(parents=True, exist_ok=True)
    tar = dst / "base.tar"
    subprocess.run(["git", "-C", str(V.ROOT), "archive", "--format=tar", "-o", str(tar), PARENT,
                    "python"], check=True, capture_output=True)
    with tarfile.open(tar) as tf:
        tf.extractall(dst)
    return dst


@needs_two
def test_flag_off_is_bit_identical_to_the_parent_commit(tmp_path):
    base = _parent_tree(tmp_path / "base")
    r0, d0 = V.run_line(base, "paysplit_id", _maps(), extra=SM + ARCH)
    assert r0.returncode == 0, (r0.stdout + r0.stderr)[-4000:]
    r1, d1 = V.run_line(V.ROOT, "paysplit_id", _maps(), extra=SM + ARCH)
    assert r1.returncode == 0, (r1.stdout + r1.stderr)[-4000:]
    try:
        ca = json.loads((d0 / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((d1 / "run.json").read_text(encoding="utf-8"))["config"]
        for c in (ca, cb):
            c.pop("demo_file", None)                 # the spine path names the checkout
        assert ca == cb, sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
        assert "critic_pay_d0" not in cb
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
        if not os.environ.get("PAYSPLIT_PARENT_TREE"):
            shutil.rmtree(d0, ignore_errors=True)
