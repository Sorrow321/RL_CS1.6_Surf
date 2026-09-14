"""--int-split (docs/int_split.md): a two-head critic with a NON-EPISODIC
intrinsic return (Burda, Edwards, Storkey, Klimov, "Exploration by random
network distillation", ICLR 2019, sec. 2.3) - mechanism 3 of
docs/litsurvey-labyrinth.md, for gap 2 (a dive that dies forfeits the
novelty it would have collected, so the agent is risk-averse for the
exploration reason too).

(a) RaceReward: with the flag the count bonus is MOVED, bit for bit, from
    r into int_r (r_split + int_r == r_control on every call, both novelty
    modes, the --curiosity-cond T scaling included); int_r is zero on ended
    rows and rebuilt every call; int_paid still counts it; the flag needs
    int_coef > 0.
(b) the intrinsic GAE: the documented recursion on a hand case where an
    episode ends inside the rollout - the extrinsic stream is cut at the
    done, the intrinsic one is not, and both agree when nothing ends; the
    trainer's source carries exactly that recursion, with no nonterm in it.
(c) Policy: int_head is registered LAST and its draw comes after every
    other parameter's (the shared tensors are bit-identical to the flag-off
    model under the same seed); the state_dict gains exactly its two keys;
    the value output is (B, 2) with column 0 the flag-off value bit for bit;
    --priv-critic keeps working (priv None -> a NaN pair).
(d) trainer smokes (CPU, the toy scratch set): the flag ON trains with
    48-tick episodes (the config keys, the int/* columns, a 2-wide
    checkpoint whose int_head is last in the Adam state); record_ckpt loads
    the 2-wide checkpoint (mirrored, constants TRAIN_ONLY); resuming it WITH
    the flag continues and restores the three constants; resuming it
    WITHOUT the flag is refused; resuming a one-head checkpoint WITH the
    flag is refused; --int-coef 0 and --rnn are refused; and the flag OFF
    is bit-identical to the PARENT commit's train_fast.py + rewards.py
    (config, progress.csv minus fps, the eval trajectory, weights, Adam
    moments) on the bins and on the absolute view.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                                   # noqa: E402

from surfgym.privfeat import PRIV_DIM                          # noqa: E402
from surfgym.rewards import RaceReward                         # noqa: E402
from train_fast import N_SCALAR, Policy                        # noqa: E402
from test_curiosity_cond import FakeCore, _race, _step         # noqa: E402
from test_unstuck import ABS, TRAIN, _csv, _run, _train        # noqa: E402
from test_view_continuous import (CANNONBALL, SMOKE_FLAGS,     # noqa: E402
                                  needs_run)

RECORD = ROOT / "tools" / "record_ckpt.py"
CELL = 256.0
START = (-11000.0, 7487.0 + 3000.0, -1000.0)
INT = ["--int-split", "--int-coef", "1.0", "--ep-ticks", "48"]
INT_KEYS = ("int_split", "int_gamma", "int_vf", "int_adv_coef")
INT_COLS = ["int/ret_mean", "int/v_mean", "int/adv_abs", "int/ev"]


# ==========================================================================
# (a) the reward: the bonus moves, nothing else does
# ==========================================================================
def _script(n, k, seed=0):
    """A scripted drive: per call, every env moves (dx, dy) on a lattice
    of quarter cells and turns; some calls mark some envs done."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        d = rng.integers(-3, 4, size=(n, 2)) * (CELL / 4.0)
        yaw = rng.uniform(-180.0, 180.0, size=n)
        spd = rng.uniform(0.0, 3000.0, size=n)
        done = rng.random(n) < 0.15
        out.append((d, yaw, spd, done))
    return out


def _drive(rw, core, script):
    rs, ints = [], []
    for d, yaw, spd, done in script:
        for i in range(core.num_envs):
            o = core.states_view["origin"][i]
            core.states_view["origin"][i] = (o[0] + d[i, 0], o[1] + d[i, 1], o[2])
            core.states_view["yaw"][i] = yaw[i]
            core.states_view["velocity"][i] = (spd[i], 0.0, 0.0)
        r = np.asarray(_step(rw, core, done=done.astype(np.uint8)), np.float32)
        rs.append(r.copy())
        ints.append(np.zeros_like(r) if rw.int_r is None else rw.int_r.copy())
    return np.stack(rs), np.stack(ints)


@pytest.mark.parametrize("kw", [
    dict(int_mode="cell", int_view=8, int_speed=3),
    dict(int_mode="edge"),
    dict(int_mode="cell", int_view=8, int_speed=3, cc_tmax=2.0),
])
def test_split_moves_the_bonus_bit_for_bit_and_nothing_else(kw):
    n, k = 6, 40
    ctl, spl = _race(int_coef=0.25, max_step=1000.0, **kw), \
        _race(int_coef=0.25, max_step=1000.0, int_split=True, **kw)
    cores = FakeCore(n), FakeCore(n)
    for c in cores:
        for i in range(n):
            c.states_view["origin"][i] = START
    for rw, c in zip((ctl, spl), cores):
        _step(rw, c)                                    # on_reset
        if kw.get("cc_tmax"):
            rw.set_cc_T([0.0, 0.5, 1.0, 2.0, 1.5, 0.25])
    assert ctl.int_r is None and spl.int_r is not None
    script = _script(n, k)
    r_ctl, i_ctl = _drive(ctl, cores[0], script)
    r_spl, i_spl = _drive(spl, cores[1], script)
    assert not i_ctl.any()
    assert r_ctl.dtype == r_spl.dtype == i_spl.dtype == np.float32
    # the split reward plus its intrinsic stream IS the control, bit for bit
    assert np.array_equal(r_ctl, r_spl + i_spl)
    # and the bonus really moved: the streams differ wherever novelty paid
    paid = i_spl != 0.0
    assert paid.any()
    assert np.all(r_ctl[paid] != r_spl[paid])
    assert i_spl.min() >= 0.0
    # ended rows are never paid novelty (their state is the next spawn)
    for (_, _, _, done), row in zip(script, i_spl):
        assert not row[done].any()
    # the bookkeeping the ledger reads is unchanged
    assert spl.int_paid == pytest.approx(ctl.int_paid, rel=1e-6)
    assert spl.int_paid == pytest.approx(float(i_spl.astype(np.float64).sum()), rel=1e-5)
    assert np.array_equal(ctl.counts_state(), spl.counts_state())


def test_int_r_is_rebuilt_every_call_and_the_flag_needs_a_bonus():
    rw = _race(int_coef=0.25, int_split=True, max_step=1000.0)
    core = FakeCore(2)
    for i in range(2):
        core.states_view["origin"][i] = START
    _step(rw, core)
    core.states_view["origin"][0] = (START[0] + CELL, START[1], START[2])
    _step(rw, core)
    assert rw.int_r[0] > 0.0 and rw.int_r[1] == 0.0
    _step(rw, core)                                      # nobody moved
    assert not rw.int_r.any()
    with pytest.raises(ValueError, match="int-coef"):
        _race(int_coef=0.0, int_split=True)


# ==========================================================================
# (b) the intrinsic GAE: non-episodic
# ==========================================================================
def _gae(rew, val, last, gamma, lam, done=None):
    """The trainer's recursion; `done` None = the intrinsic stream (no cut)."""
    T, N = rew.shape
    adv = np.zeros_like(rew)
    last_g = np.zeros(N)
    for t in reversed(range(T)):
        nextv = last if t == T - 1 else val[t + 1]
        nonterm = 1.0 if done is None else 1.0 - done[t]
        delta = rew[t] + gamma * nextv * nonterm - val[t]
        last_g = delta + gamma * lam * nonterm * last_g
        adv[t] = last_g
    return adv


def test_intrinsic_gae_crosses_the_episode_end_and_matches_the_hand_case():
    rew = np.array([[1.0], [0.0], [2.0]])
    val = np.array([[0.5], [0.25], [1.0]])
    last = np.array([0.75])
    done = np.array([[0.0], [1.0], [0.0]])       # the episode ends at t = 1
    a_int = _gae(rew, val, last, 0.9, 0.5)
    a_epi = _gae(rew, val, last, 0.9, 0.5, done)
    # by hand: t=2 1.675; t=1 0.65 + 0.45 x 1.675; t=0 0.725 + 0.45 x that
    assert a_int[:, 0] == pytest.approx([1.3566875, 1.40375, 1.675])
    assert a_epi[:, 0] == pytest.approx([0.6125, -0.25, 1.675])
    # the intrinsic stream carries the post-death novelty (t = 2) back over
    # the death; the episodic one cannot
    assert a_int[0, 0] > a_epi[0, 0] and a_int[1, 0] > a_epi[1, 0]
    # with no episode end the two are the same recursion
    assert np.allclose(_gae(rew, val, last, 0.9, 0.5, np.zeros_like(done)), a_int)


def test_trainer_source_runs_the_intrinsic_recursion_without_a_cut():
    src = TRAIN.read_text(encoding="utf-8")
    i0 = src.index("the intrinsic stream's GAE")
    i1 = src.index("ret_i = adv_i + b_vint", i0)
    # the CODE of the block (its comment says why there is no mask)
    block = "\n".join(ln for ln in src[i0:i1].splitlines()
                      if not ln.strip().startswith("#"))
    assert "delta_i = b_rint[t] + INT_GAMMA * nextv_i - b_vint[t]" in block
    assert "lastgae_i = delta_i + INT_GAMMA * args.gae * lastgae_i" in block
    assert "nonterm" not in block and "b_done" not in block
    # the extrinsic recursion is still the cut one, and stays V_E's target
    assert "delta = b_rew[t] + g_eff * nextval * nonterm - b_val[t]" in src
    assert "ret = adv + b_val\n" in src
    # the combined advantage enters the SAME normalisation path
    assert "adv = adv + INT_ADV_COEF * adv_i" in src
    assert "f_adv = adv.reshape(-1)" in src
    # V_I is fitted raw: the extrinsic target alone is normalised
    assert "f_reti = ret_i.reshape(-1) if INT_SPLIT else None" in src


# ==========================================================================
# (c) the policy
# ==========================================================================
def _pol(seed, **kw):
    torch.manual_seed(seed)
    return Policy(N_SCALAR + 16 * 8, 16, 8, emb=64, hidden=64, **kw).eval()


def test_int_head_is_last_and_the_rest_of_the_model_is_the_flag_off_one():
    p0, p1 = _pol(3), _pol(3, int_split=True)
    sd0, sd1 = p0.state_dict(), p1.state_dict()
    assert set(sd1) - set(sd0) == {"int_head.weight", "int_head.bias"}
    assert set(sd0) - set(sd1) == set()
    for k in sd0:
        assert torch.equal(sd0[k], sd1[k]), k
    names = [n for n, _ in p1.named_parameters()]
    assert names[-2:] == ["int_head.weight", "int_head.bias"]
    assert [n for n, _ in p0.named_parameters()] == names[:-2]
    assert tuple(sd1["int_head.weight"].shape) == (1, 64)
    assert p0.int_head is None
    torch.manual_seed(11)
    obs = torch.randn(5, N_SCALAR + 16 * 8)
    with torch.no_grad():
        l0, v0 = p0(obs)
        l1, v1 = p1(obs)
    assert torch.equal(l0, l1)
    assert tuple(v0.shape) == (5,) and tuple(v1.shape) == (5, 2)
    assert torch.equal(v1[:, 0], v0)
    assert torch.isfinite(v1).all()


def test_int_head_rides_the_privileged_block_too():
    p = _pol(5, int_split=True, priv_dim=PRIV_DIM, priv_hidden=8)
    assert tuple(p.int_head.weight.shape) == (1, 64 + 8)
    torch.manual_seed(1)
    obs = torch.randn(4, N_SCALAR + 16 * 8)
    with torch.no_grad():
        v = p(obs, priv=torch.randn(4, PRIV_DIM))[1]
        vn = p(obs, priv=None)[1]
    assert tuple(v.shape) == (4, 2) and torch.isfinite(v).all()
    assert tuple(vn.shape) == (4, 2) and torch.isnan(vn).all()


# ==========================================================================
# (d) trainer smokes
# ==========================================================================
def _rows(d: Path):
    rows = (d / "progress.csv").read_text(encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


def _cfg(d: Path):
    return json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_flag_on_trains_records_resumes_and_refuses():
    run = "int_split_on"
    d = ROOT / "runs" / run
    on = INT + ["--int-gamma", "0.9", "--int-vf", "0.25", "--int-adv-coef", "0.5"]
    r = _train(run, on)
    assert "--int-split: two-head critic with a NON-EPISODIC intrinsic" in r.stdout
    cfg = _cfg(d)
    assert {k: cfg[k] for k in INT_KEYS} == {"int_split": 1, "int_gamma": 0.9,
                                             "int_vf": 0.25, "int_adv_coef": 0.5}
    rows = _csv(run)
    assert len(rows) == 3
    head = list(rows[0])
    for c in INT_COLS:
        assert c in head
    assert head.index("int/ret_mean") > head.index("train/value_loss")
    for x in rows:
        for c in INT_COLS:
            assert x[c] != "" and np.isfinite(float(x[c])), (c, x[c])
        assert float(x["int/adv_abs"]) > 0.0          # novelty was paid
        assert float(x["int/ev"]) <= 1.0
        assert np.isfinite(float(x["train/value_loss"]))
        assert np.isfinite(float(x["train/approx_kl"]))
    # the 2-wide checkpoint: int_head last, with Adam moments of its own
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sd = ck["policy"]
    assert list(sd)[-2:] == ["int_head.weight", "int_head.bias"]
    assert tuple(sd["int_head.weight"].shape) == (1, 64)
    assert ck["config"]["int_split"] == 1
    pg = ck["optimizer"]["param_groups"][0]["params"]
    st = ck["optimizer"]["state"]
    assert tuple(st[pg[-2]]["exp_avg"].shape) == (1, 64)
    assert tuple(st[pg[-1]]["exp_avg"].shape) == (1,)
    assert float(st[pg[-2]]["exp_avg"].abs().sum()) > 0.0

    # record_ckpt: the flag is mirrored (a STRICT load of the 2-wide critic)
    # and the constants are TRAIN_ONLY, so the record gate passes
    rr = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
               "--map", str(CANNONBALL), "--episodes", "1", "--ep-ticks", "200",
               "--out", str(d / "rec.jsonl")], timeout=900)
    assert rr.returncode == 0, rr.stdout[-3000:] + rr.stderr[-3000:]
    assert (d / "rec.jsonl").exists()

    # resume WITH the flag: continues, the three constants come back from
    # the config when not given
    re = ROOT / "runs" / "int_split_re"
    shutil.rmtree(re, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", re.name,
               "--ckpt", str(d / "ckpt_final.pt")] + SMOKE_FLAGS
              + ["--steps", "8192"] + INT)
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert {k: _cfg(re)[k] for k in INT_KEYS} == {k: cfg[k] for k in INT_KEYS}
    rows2 = _csv(re.name)
    assert len(rows2) == 1 and all(np.isfinite(float(rows2[0][c])) for c in INT_COLS)
    ck2 = torch.load(re / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert "int_head.weight" in ck2["policy"] and ck2["global_step"] > ck["global_step"]

    # resume WITHOUT the flag: refused, both the head and the objective
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "int_split_noflag",
               "--ckpt", str(d / "ckpt_final.pt")] + SMOKE_FLAGS
              + ["--steps", "8192", "--int-coef", "1.0", "--ep-ticks", "48"])
    assert r3.returncode != 0
    assert "this checkpoint was trained with --int-split" in r3.stdout + r3.stderr

    # a ONE-HEAD checkpoint with the flag: refused
    off = ROOT / "runs" / "int_split_off"
    _train(off.name, ["--int-coef", "1.0", "--ep-ticks", "48"])
    assert not any(k in _cfg(off) for k in INT_KEYS)
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", "int_split_onehead",
               "--ckpt", str(off / "ckpt_final.pt")] + SMOKE_FLAGS
              + ["--steps", "8192"] + INT)
    assert r4.returncode != 0
    assert "cannot warm-start a ONE-HEAD checkpoint" in r4.stdout + r4.stderr

    # the refusals
    r5 = _run([sys.executable, "-u", str(TRAIN), "--run", "int_split_bad"]
              + SMOKE_FLAGS + ["--steps", "2048", "--int-split", "--int-coef", "0"])
    assert r5.returncode != 0
    assert "--int-split needs --int-coef > 0" in r5.stdout + r5.stderr
    r6 = _run([sys.executable, "-u", str(TRAIN), "--run", "int_split_bad"]
              + SMOKE_FLAGS + ["--steps", "2048", "--int-split", "--int-coef", "1.0",
                               "--rnn", "gru"])
    assert r6.returncode != 0
    assert "--int-split is not implemented with --rnn" in r6.stdout + r6.stderr
    for p in (d, re, off) + tuple(ROOT / "runs" / n for n in
                                  ("int_split_noflag", "int_split_onehead",
                                   "int_split_bad")):
        shutil.rmtree(p, ignore_errors=True)


def _parent_tree(dst: Path):
    """The python/ tree of the last first-parent commit WITHOUT --int-split
    (HEAD itself while the flag is uncommitted), extracted under `dst`;
    returns the ref, or None when the history is too old to compare."""
    try:
        r = subprocess.run(["git", "rev-list", "--first-parent",
                            "--max-count=200", "HEAD"],
                           capture_output=True, text=True, cwd=str(ROOT),
                           timeout=60)
        refs = r.stdout.split() if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        refs = []
    for ref in refs:
        r = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if r.returncode != 0:
            continue
        if b"--int-split" in r.stdout:
            continue
        if b"--int-rare-speed" not in r.stdout:
            return None                # predates this change's own anchors
        z = dst / "parent_python.zip"
        subprocess.run(["git", "archive", "--format=zip", "-o", str(z), ref,
                        "python"], check=True, cwd=str(ROOT), timeout=120)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dst)
        z.unlink()
        return ref
    return None


def _assert_identical(a: Path, b: Path, rows: int = 3):
    ca, cb = _cfg(a), _cfg(b)
    assert not any(k in ca for k in INT_KEYS)
    assert ca == cb
    ra, rb = _rows(a), _rows(b)
    assert len(ra) == len(rb) == rows
    assert list(ra[0]) == list(rb[0])            # the same header
    assert not any(c.startswith("int/") for c in ra[0])
    for x, y in zip(ra, rb):
        for k in x:
            if k != "time/fps":
                assert x[k] == y[k], (k, x[k], y[k])
    ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
    assert ta and [p.name for p in ta] == [p.name for p in tb]
    for p, q in zip(ta, tb):
        assert p.read_bytes() == q.read_bytes(), p.name
    sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert set(sa["policy"]) == set(sb["policy"])
    assert "int_head.weight" not in sa["policy"]
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    oa, ob = sa["optimizer"]["state"], sb["optimizer"]["state"]
    assert set(oa) == set(ob)
    for i in oa:
        for k in oa[i]:
            if torch.is_tensor(oa[i][k]):
                assert torch.equal(oa[i][k], ob[i][k]), (i, k)


@needs_run
@pytest.mark.parametrize("mode", ["bins", "abs"])
def test_flag_off_is_bit_identical_to_the_parent_commit(mode, tmp_path):
    """The flag OFF against the trainer AND reward of the parent commit
    (both files changed; the parent's whole python/ tree runs, so its
    train_fast imports its own surfgym.rewards): the config gains no key,
    and every number - progress.csv minus fps, the eval trajectory, the
    weights, the Adam moments - is the same, bins and absolute view."""
    ref = _parent_tree(tmp_path)
    if ref is None:
        pytest.skip("no pre-int-split python/ tree in the first-parent history")
    old = tmp_path / "python" / "train_fast.py"
    assert old.exists() and (tmp_path / "python" / "surfgym" / "rewards.py").exists()
    flags = ABS if mode == "abs" else []
    new_run = ROOT / "runs" / f"int_split_ctl_new_{mode}"
    old_run = tmp_path / "runs" / f"int_split_ctl_old_{mode}"
    _train(new_run.name, flags)
    _train(old_run.name, flags, script=old)
    assert old_run.exists(), "the parent trainer writes under its own tree"
    _assert_identical(new_run, old_run)
    shutil.rmtree(new_run, ignore_errors=True)
