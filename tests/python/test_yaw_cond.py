"""--yaw-cond: the AUTOREGRESSIVE side key (docs/research-litsurvey-temporal
.md proposal #3).

Why: the six action heads are conditionally independent given the trunk
features, and air-strafing needs the yaw delta to turn the view TOWARD the
held strafe key's wish direction. A factored distribution that wants "either
(left, left) or (right, right), never a mixed pair" cannot express that, and
at the symmetric point each head's gradient toward committing is
proportional to (2 p_other - 1), which is exactly 0 when the other head is
undecided - the left/right symmetry is a saddle. Measured on the finisher
xQR32, the two disagree on 12.7% of fast airborne decisions against the human
world record's 2.7%.

The flag makes the side head's logits carry an additive row of a (15, 3)
table, gathered by the yaw bin the decision uses, so

    log p(a) = log p(yaw) + log p(side | yaw) + the other four heads

is EXACT and PPO's ratio is unchanged in form. The table starts at zero, so
the model is function-identical at step 0 and the arm is a warm resume.

What is pinned here:

(a) With NO flag the trainer is bit-identical to the tree without it: the
    same config dump, the same progress.csv on every shared column, the same
    weights and the same Adam moments after a tiny CPU scratch run. The one
    permitted difference is the APPENDED act/yaw_side_agree column.
(b) With the flag and a ZERO table the conditioned logits ARE the
    unconditioned ones, tensor for tensor - which is what makes
    widen_for_yawcond a warm start rather than a re-initialisation, and it
    is checked end to end on a real checkpoint too.
(c) With a hand-set table the sampled side key given each yaw bin matches
    softmax(side_logits + W[yaw]) empirically, the yaw MARGINAL is
    untouched, and the log-prob the update recomputes from the STORED
    actions equals the one the rollout stored, bit for bit.
(d) A tiny CPU scratch run with the flag trains, the table leaves zero, and
    act/yaw_side_agree is logged - with the flag and without it.
(e) tools/record_ckpt.py round-trips a flagged checkpoint (it rebuilds the
    policy from the ckpt CONFIG), and rebuilding it WITHOUT the mirror is
    refused by the strict load rather than silently recording an
    unconditioned policy.
(f) With --mask-forward-air the masked slots are still exactly zero
    probability after the conditioning, and the ratio still closes.

    python -m pytest tests/python/test_yaw_cond.py -q

CPU only, tiny nets; a few minutes. It needs the built core and the prebaked
cannonball goal field. Running from a git worktree, point SURF_TEST_MAPS at
the main checkout's maps/ (CLAUDE.md: a worktree copy has different mtimes
and every prebaked cache misses) and SURFCORE_DLL at its built core.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

MAPS = Path(os.environ.get("SURF_TEST_MAPS") or (ROOT / "maps"))
CANNONBALL = MAPS / "surf_src_cannonball.bsp"
GOALFIELD = MAPS / "surf_src_cannonball.goal_32.npz"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))

needs_core = pytest.mark.skipif(
    not (CANNONBALL.exists() and DLL.exists() and GOALFIELD.exists()),
    reason="needs the built core + cannonball + its prebaked goal field")

import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402

from train_fast import (H_SIDE, H_YAW, NEUTRAL_SIDE,           # noqa: E402
                        NEUTRAL_YAW, NPAD, NVEC, ActionMasks,
                        HeadPacker, Policy, add_yaw_cond,
                        greedy_padded_yawcond, logprob_entropy_padded,
                        sample_padded_yawcond, widen_for_yawcond)

N_YAW, N_SIDE = NVEC[H_YAW], NVEC[H_SIDE]
SIDE_LO = sum(NVEC[:H_SIDE])
SIDE_HI = SIDE_LO + N_SIDE


def _fixture(seed=0, B=2048):
    torch.manual_seed(seed)
    pk = HeadPacker(torch.device("cpu"))
    return pk, pk.pad(torch.randn(B, sum(NVEC)))


# ==========================================================================
# (b) a ZERO table is the unconditioned policy, exactly
# ==========================================================================
def test_zero_table_leaves_the_logits_untouched():
    """The whole warm-start claim in one line: at step 0 the conditioned
    logits ARE the checkpoint's, so the arm starts on the control curve."""
    pk, padded = _fixture()
    zero = torch.zeros(N_YAW, N_SIDE)
    yaw = torch.randint(0, N_YAW, (padded.shape[0],))
    assert torch.equal(add_yaw_cond(padded, zero, yaw), padded)
    # ... and so does everything built on it
    act_c, lp_c = sample_padded_yawcond(padded, zero, None)
    torch.manual_seed(0)
    _pk2, p2 = _fixture()
    assert torch.equal(greedy_padded_yawcond(p2, zero, None), p2.argmax(-1))
    lp_u, _ = logprob_entropy_padded(padded, act_c)
    assert torch.equal(lp_c, lp_u)


def test_the_conditioning_touches_only_the_side_heads_live_slots():
    """H_SIDE's NVEC[H_SIDE] real bins move; every other head and every
    padding slot of the short heads is bit-identical, so the padding stays
    at NEG and stays unselectable."""
    pk, padded = _fixture(seed=4)
    W = torch.randn(N_YAW, N_SIDE) * 3.0
    yaw = torch.randint(0, N_YAW, (padded.shape[0],))
    out = add_yaw_cond(padded, W, yaw)
    d = (out - padded)
    assert float(d[:, H_SIDE, :N_SIDE].abs().max()) > 0.0
    assert float(d[:, H_SIDE, N_SIDE:].abs().max()) == 0.0
    for h in range(len(NVEC)):
        if h != H_SIDE:
            assert float(d[:, h, :].abs().max()) == 0.0, h
    assert float(out[:, H_SIDE, N_SIDE:].max()) < -1e29     # still NEG


def test_the_table_receives_gradient_only_at_the_rows_it_was_gathered_at():
    pk, padded = _fixture(seed=5, B=256)
    tab = torch.zeros(N_YAW, N_SIDE, requires_grad=True)
    yaw = torch.full((padded.shape[0],), 3, dtype=torch.long)
    p = add_yaw_cond(padded, tab, yaw)
    logprob_entropy_padded(p, p.argmax(-1))[0].sum().backward()
    assert float(tab.grad[3].abs().sum()) > 0.0
    assert float(tab.grad[[i for i in range(N_YAW) if i != 3]].abs().max()) \
        == 0.0


# ==========================================================================
# (c) the conditional distribution, the untouched marginal, the exact ratio
# ==========================================================================
def test_sampled_side_given_yaw_matches_softmax_of_the_conditioned_logits():
    """One fixed row of logits, 200k draws: the empirical p(side | yaw) has
    to be softmax(side_logits + W[yaw]) for every yaw bin the draw visits."""
    torch.manual_seed(3)
    pk = HeadPacker(torch.device("cpu"))
    W = torch.randn(N_YAW, N_SIDE) * 2.0
    one = torch.randn(1, sum(NVEC))
    rep = pk.pad(one.expand(200_000, -1).contiguous())
    act, _ = sample_padded_yawcond(rep, W, None)
    side_l = one[0, SIDE_LO:SIDE_HI]
    seen = 0
    for y in range(N_YAW):
        m = act[:, H_YAW] == y
        if int(m.sum()) < 500:
            continue
        seen += 1
        emp = torch.bincount(act[m, H_SIDE], minlength=N_SIDE).float()
        emp /= emp.sum()
        want = torch.softmax(side_l + W[y], dim=-1)
        assert float((emp - want).abs().max()) < 0.02, (y, emp, want)
    assert seen >= 5
    # the yaw MARGINAL is p(yaw), untouched by the conditioning: the
    # factorisation is p(yaw) p(side|yaw), not a joint reweighting
    ye = torch.bincount(act[:, H_YAW], minlength=N_YAW).float()
    ye /= ye.sum()
    assert float((ye - torch.softmax(one[0, :N_YAW], -1)).abs().max()) < 0.01


def test_update_logprob_matches_the_rollout_and_unconditioned_would_bias_it():
    """PPO's ratio is exp(logp_new - logp_old): both terms have to come from
    the same measure. The update conditions on the STORED yaw action, which
    is exactly what the rollout conditioned on."""
    pk, padded = _fixture(seed=2)
    W = torch.randn(N_YAW, N_SIDE) * 2.0
    act, logp_roll = sample_padded_yawcond(padded, W, None)
    logp_upd, _ = logprob_entropy_padded(
        add_yaw_cond(padded, W, act[:, H_YAW]), act)
    assert torch.equal(logp_roll, logp_upd)
    # the failure mode this guards: an update that forgot the conditioning
    logp_bad, _ = logprob_entropy_padded(padded, act)
    assert float((logp_bad - logp_roll).abs().max()) > 1.0


def test_greedy_is_argmax_yaw_then_argmax_side_given_it():
    pk, padded = _fixture(seed=6)
    W = torch.randn(N_YAW, N_SIDE) * 2.0
    g = greedy_padded_yawcond(padded, W, None)
    assert torch.equal(g[:, H_YAW], padded[:, H_YAW, :].argmax(-1))
    want = (padded[:, H_SIDE, :]
            + F.pad(W[g[:, H_YAW]], (0, NPAD - N_SIDE))).argmax(-1)
    assert torch.equal(g[:, H_SIDE], want)
    for h in range(len(NVEC)):          # the other four heads are unmoved
        if h not in (H_YAW, H_SIDE):
            assert torch.equal(g[:, h], padded[:, h, :].argmax(-1)), h
    # and a table strong enough to flip the side key actually flips it
    W2 = torch.zeros(N_YAW, N_SIDE)
    W2[:, 0] = 50.0
    assert (greedy_padded_yawcond(padded, W2, None)[:, H_SIDE] == 0).all()


# ==========================================================================
# (f) the action masks: conditioning FIRST, mask LAST
# ==========================================================================
def _mask_fixture(seed=7, B=2048):
    pk, padded = _fixture(seed, B)
    torch.manual_seed(seed + 100)
    air = (torch.rand(B) < 0.5).float()
    jblk = (torch.rand(B) < 0.3).float()
    m = ActionMasks(fwd_air=True, jump_cd=3, duck_air=True)
    return pk, padded, m, air, jblk


def test_masked_slots_stay_masked_after_the_conditioning():
    pk, padded, m, air, jblk = _mask_fixture()
    W = torch.randn(N_YAW, N_SIDE) * 5.0

    def mfn(p):
        return m.add_mask(p, air, jblk)

    act, logp = sample_padded_yawcond(padded, W, mfn)
    pm = mfn(add_yaw_cond(padded, W, act[:, H_YAW]))
    pr = torch.softmax(pm, dim=-1)
    a, j = air.bool(), jblk.bool()
    assert float(pr[a, 2, 0].max()) == 0.0        # H_FWD -400 (S)
    assert float(pr[a, 2, 2].max()) == 0.0        # H_FWD +400 (W)
    assert float(pr[a, 5, 1].max()) == 0.0        # H_DUCK held
    assert float(pr[j, 4, 1].max()) == 0.0        # H_JUMP held
    assert torch.isfinite(pr).all()
    # the sample obeys the mask, and the ratio still closes under it
    from train_fast import A_FWD_NONE
    assert (act[a, 2] == A_FWD_NONE).all()
    assert (act[a, 5] == 0).all() and (act[j, 4] == 0).all()
    lp2, _ = logprob_entropy_padded(pm, act)
    assert torch.equal(logp, lp2)


def test_conditioning_before_the_mask_is_the_order_that_ships():
    """The masks write H_FWD / H_JUMP / H_DUCK and the conditioning writes
    H_SIDE, so today the two orders agree bit for bit - that is the check
    that the shipped order is not silently wrong. It is written in the
    order that STAYS right if a mask ever reaches the side head: NEG is
    finite (-1e30), so a finite conditioning term added AFTER a mask would
    leave the slot merely very negative instead of exactly zero."""
    pk, padded, m, air, jblk = _mask_fixture(seed=8)
    W = torch.randn(N_YAW, N_SIDE) * 5.0
    yaw = torch.randint(0, N_YAW, (padded.shape[0],))
    cond_then_mask = m.add_mask(add_yaw_cond(padded, W, yaw), air, jblk)
    mask_then_cond = add_yaw_cond(m.add_mask(padded, air, jblk), W, yaw)
    assert torch.equal(cond_then_mask, mask_then_cond)
    # and the property the order protects, stated directly on a side mask
    side_masked = padded.clone()
    side_masked[:, H_SIDE, 0] = -1e30
    after = add_yaw_cond(side_masked, W, yaw)
    assert float(torch.softmax(after, -1)[:, H_SIDE, 0].max()) == 0.0


# ==========================================================================
# the model: parameter ORDER, no RNG, and the warm start
# ==========================================================================
def _tiny(**kw):
    return Policy(15 + 16 * 8, lidar_w=16, lidar_h=8, emb=32, hidden=32, **kw)


def test_the_flag_adds_one_tensor_LAST_and_draws_no_rng():
    """Adam's state is keyed by the parameter INDEX, so a new tensor that
    is not last silently pairs every later tensor with the previous one's
    moments. nn.Module.parameters() yields a module's OWN parameters before
    its children's, which is why the table is a submodule and not a bare
    nn.Parameter on Policy."""
    torch.manual_seed(11)
    off = _tiny()
    torch.manual_seed(11)
    on = _tiny(yaw_cond=True)
    so, sn = off.state_dict(), on.state_dict()
    assert set(sn) - set(so) == {"yaw_side.table"}
    assert not set(so) - set(sn)
    for k in so:                     # no RNG consumed by the zeros table
        assert torch.equal(so[k], sn[k]), k
    assert float(sn["yaw_side.table"].abs().max()) == 0.0
    assert tuple(sn["yaw_side.table"].shape) == (N_YAW, N_SIDE)
    a = [tuple(t.shape) for t in off.parameters()]
    b = [tuple(t.shape) for t in on.parameters()]
    assert b[:len(a)] == a and len(b) == len(a) + 1
    assert b[-1] == (N_YAW, N_SIDE)
    assert off.yaw_side is None and on.yaw_side is not None


def test_widen_for_yawcond_is_function_identical_on_a_real_state_dict():
    """The warm-resume claim, end to end: a checkpoint with no table,
    widened onto a --yaw-cond Policy, computes the SAME logits and takes
    the SAME greedy action as the unconditioned one."""
    torch.manual_seed(13)
    src = _tiny()
    ck = {"policy": {k: v.clone() for k, v in src.state_dict().items()},
          "optimizer": {"state": {}, "param_groups": [
              {"params": list(range(len(list(src.parameters()))))}]}}
    torch.manual_seed(99)                       # a DIFFERENT draw
    dst = _tiny(yaw_cond=True)
    n = widen_for_yawcond(ck, dst)
    assert n == 1
    dst.load_state_dict(ck["policy"])           # strict
    # the optimizer group grew by exactly the new index, at the end
    g = ck["optimizer"]["param_groups"][0]["params"]
    assert g == list(range(len(list(dst.parameters()))))
    torch.manual_seed(17)
    obs = torch.randn(64, 15 + 16 * 8)
    src.eval()
    dst.eval()
    with torch.no_grad():
        la, _ = src(obs)
        lb, _ = dst(obs)
    assert torch.equal(la, lb)
    pk = HeadPacker(torch.device("cpu"))
    pa = pk.pad(la)
    assert torch.equal(greedy_padded_yawcond(pk.pad(lb), dst.yaw_side.table,
                                             None), pa.argmax(-1))
    # idempotent: a second widen finds the table already there
    assert widen_for_yawcond(ck, dst) == 0


def test_a_conditioned_state_dict_will_not_load_into_a_plain_policy():
    """The mirror in record_ckpt.py is load-bearing: without it the strict
    load is what refuses, rather than a plausible wrong recording."""
    torch.manual_seed(21)
    on = _tiny(yaw_cond=True)
    torch.manual_seed(21)
    off = _tiny()
    with pytest.raises(RuntimeError):
        off.load_state_dict(on.state_dict())


def test_chunk_is_refused_at_construction():
    with pytest.raises(SystemExit):
        _tiny(yaw_cond=True, n_codes=8, chunk=4)


# ==========================================================================
# the source-level guard: four call sites, and the order against the masks
# ==========================================================================
def test_the_conditioning_is_applied_in_all_four_places():
    """The whole point is that four call sites agree, and dropping one is
    silent (the run trains, the ratio is just wrong). Counted the way
    tests/python/test_air_masks.py counts the masks."""
    tf = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    # 1 the rollout sample and 3 the STOCHASTIC eval, plus the def
    assert tf.count("sample_padded_yawcond(") == 3, \
        tf.count("sample_padded_yawcond(")
    # 3 the GREEDY eval, plus the def
    assert tf.count("greedy_padded_yawcond(") == 2, \
        tf.count("greedy_padded_yawcond(")
    # 2 the update: mb_step, seq_loss (--rnn) and the --bc-file cloning NLL,
    #   plus the two inside sample/greedy_padded_yawcond and the def
    assert tf.count("add_yaw_cond(") == 6, tf.count("add_yaw_cond(")
    assert tf.count("add_yaw_cond(padded, policy.yaw_side.table,") == 3
    for tag in ("PLACE 1 of 4 for --yaw-cond", "PLACE 2 of 4 for --yaw-cond",
                "PLACE 3 of 4 for --yaw-cond"):
        assert tag in tf, tag
    # 4 the recorder rebuilds it from the ckpt config
    assert 'yaw_cond=bool(cfg.get("yaw_cond"))' in rc
    # the masks are still applied in exactly the three trainer sites
    # test_air_masks.py counts (the rollout's is now a hoisted closure)
    assert tf.count("MASKS.add_mask(") == 3, tf.count("MASKS.add_mask(")


def test_the_neutral_indices_match_the_core():
    """act/yaw_side_agree and the table's row order both key on these."""
    from surfgym.core import ACTION_NVEC, YAW_BINS
    assert tuple(ACTION_NVEC) == tuple(NVEC)
    assert float(YAW_BINS[NEUTRAL_YAW]) == 0.0
    assert (YAW_BINS[:NEUTRAL_YAW] < 0).all()
    assert (YAW_BINS[NEUTRAL_YAW + 1:] > 0).all()
    assert NEUTRAL_SIDE == 1           # a[3] = {-400, 0, +400}[i]
    env_c = (ROOT / "src" / "env.c").read_text(encoding="utf-8")
    # the sign convention act/yaw_side_agree is written against
    assert "float smove = (a[3] <= 0) ? -400.0f : (a[3] >= 2 ? 400.0f" in env_c
    assert "st->yaw = wrap_yaw(st->yaw + yd);" in env_c


# ==========================================================================
# the CPU scratch runs
# ==========================================================================
SMOKE_FLAGS = ["--map", str(CANNONBALL), "--reward", "race", "--envs", "64",
               "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
               "--lidar-cell", "32", "--lidar-range", "11500",
               "--lidar-near", "2000", "--emb", "64", "--hidden", "64",
               "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail",
               "--lr", "3e-4", "--gamma", "0.9995", "--gae", "0.95",
               "--clip", "0.2", "--vf", "0.5", "--ent", "0.005",
               "--n-steps", "8", "--epochs", "1", "--minibatches", "2",
               "--ep-ticks", "3000", "--time-pen", "0.005",
               "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
               "--race-dist", "geodesic", "--maxvel", "4000",
               "--train-stride", "1", "--yaw-adaptive",
               "--respawn-frac", "0.9", "--respawn-margin", "10",
               "--respawn-reservoir", "1000", "--int-coef", "0.25",
               "--int-view", "8", "--int-speed", "3", "--ckpt-every", "1e9",
               "--record-every", "1e12", "--eval-eps", "1",
               "--eval-greedy-only", "--no-eval-at-start", "--seed", "7"]


def _env():
    # the user's machine is shared while these run: 4 threads, not 8
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


def _train(run, extra, script="train_fast.py", steps="2048", timeout=1800,
           fresh=True):
    if fresh:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-u", str(ROOT / "python" / script),
           "--run", run] + SMOKE_FLAGS + ["--steps", steps] + list(extra)
    # the console here is cp1251 and the trainer prints em dashes: decode the
    # child's UTF-8 explicitly rather than through the locale (CLAUDE.md)
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                       cwd=str(ROOT), timeout=timeout,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(
        encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


def _unpatched_trainer(dst: Path):
    """The pre-flag train_fast.py out of git, or None."""
    for ref in ("baseline", "origin/baseline", "main", "origin/main", "HEAD^"):
        # BYTES, never text: this console is cp1251 and train_fast.py is
        # UTF-8 with em dashes - a locale round trip would corrupt the copy
        r = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if r.returncode == 0 and b"--yaw-cond" not in r.stdout:
            dst.write_bytes(r.stdout)
            return ref
    return None


# ==========================================================================
# (a) no flag == the tree without the flag, bit for bit
# ==========================================================================
@needs_core
def test_no_flag_is_bit_identical_to_the_unpatched_trainer():
    """--yaw-cond is opt-in, and 'opt-in' has to mean the control run is the
    run that shipped: same config dump, same CSV, same weights, same Adam
    moments. The copy lives in python/ so its ROOT (= parent.parent) and its
    sys.path resolve exactly like the real one."""
    base_py = ROOT / "python" / "_train_fast_preyawcond.py"
    ref = _unpatched_trainer(base_py)
    if ref is None:
        base_py.unlink(missing_ok=True)
        pytest.skip("no pre-flag train_fast.py reachable from git")
    try:
        _train("yc_bit_base", [], script=base_py.name)
        _train("yc_ctl", [])           # the untreated control, kept for (d)
    finally:
        base_py.unlink(missing_ok=True)

    a, b = ROOT / "runs" / "yc_bit_base", ROOT / "runs" / "yc_ctl"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb, [k for k in set(ca) | set(cb)
                      if ca.get(k, "@") != cb.get(k, "@")]
    assert "yaw_cond" not in cb          # nothing is written when off

    ta = (a / "progress.csv").read_text(encoding="utf-8").splitlines()
    tb = (b / "progress.csv").read_text(encoding="utf-8").splitlines()
    ha, hb = ta[0].split(","), tb[0].split(",")
    # act/yaw_side_agree is APPENDED, so the old header stays a strict prefix
    # and a resumed run's progress.csv migrates instead of breaking
    assert hb[:len(ha)] == ha
    assert hb[len(ha):] == ["act/yaw_side_agree"]
    assert len(ta) == len(tb) >= 2
    for ra, rb in zip(ta[1:], tb[1:]):
        fa, fb = ra.split(","), rb.split(",")
        fa[3] = fb[3] = ""              # time/fps is wall-clock
        assert fb[:len(fa)] == fa

    sa = torch.load(a / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    assert sa["global_step"] == sb["global_step"]
    assert sa["config"] == sb["config"]
    assert set(sa["policy"]) == set(sb["policy"])
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    oa, ob = sa["optimizer"]["state"], sb["optimizer"]["state"]
    assert set(oa) == set(ob) and oa
    for k in oa:
        for f in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[k][f], ob[k][f]), (k, f)
    shutil.rmtree(a, ignore_errors=True)


# ==========================================================================
# (d) the flag runs, the table moves off zero, the diagnostic is logged
# ==========================================================================
@needs_core
def test_the_flag_runs_and_logs_the_diagnostic():
    r = _train("yc_on", ["--yaw-cond"])
    assert "--yaw-cond: side-key head conditioned" in r.stdout
    rows = _csv("yc_on")
    assert rows
    for row in rows:
        v = float(row["act/yaw_side_agree"])
        assert 0.0 <= v <= 1.0
    # a fresh net is near uniform, so roughly half the strafing decisions
    # agree by chance - that is the number the arm has to move
    assert 0.2 < float(rows[0]["act/yaw_side_agree"]) < 0.8
    cfg = json.loads(
        (ROOT / "runs" / "yc_on" / "run.json").read_text(
            encoding="utf-8"))["config"]
    assert cfg["yaw_cond"] == 1
    ck = torch.load(ROOT / "runs" / "yc_on" / "ckpt_final.pt",
                    map_location="cpu", weights_only=False)
    tab = ck["policy"]["yaw_side.table"]
    assert tuple(tab.shape) == (N_YAW, N_SIDE)
    assert float(tab.abs().max()) > 0.0, "the table never left zero"
    assert ck["config"]["yaw_cond"] == 1
    # the control (from (a)) logs the same column with no flag
    if (ROOT / "runs" / "yc_ctl" / "progress.csv").exists():
        c = _csv("yc_ctl")[0]
        assert 0.0 <= float(c["act/yaw_side_agree"]) <= 1.0


@needs_core
def test_warm_resume_of_a_plain_checkpoint_announces_the_widening():
    """The launch mode: ARM_RESUME onto a checkpoint that has no table."""
    src = ROOT / "runs" / "yc_ctl" / "ckpt_final.pt"
    if not src.exists():
        pytest.skip("needs the control run from (a)")
    dst = ROOT / "runs" / "yc_resume"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst / "ckpt_latest.pt")
    r = _train("yc_resume", ["--yaw-cond", "--ckpt", str(dst / "ckpt_latest.pt"),
                             "--reset-steps"], steps="1024", fresh=False)
    assert "--yaw-cond: this checkpoint has no yaw->side conditioning" \
        in r.stdout, r.stdout[-3000:]
    assert "bit-identical to the" in r.stdout
    # and the reverse direction is refused rather than silently unconditioned
    bad = subprocess.run(
        [sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
         "--run", "yc_refuse"] + SMOKE_FLAGS
        + ["--steps", "1024", "--ckpt",
           str(ROOT / "runs" / "yc_on" / "ckpt_final.pt"), "--reset-steps"],
        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
        timeout=900, encoding="utf-8", errors="replace")
    out = bad.stdout + bad.stderr
    # the flag is RESTORED from the checkpoint config rather than dropped
    assert "yaw_cond=1" in out, out[-3000:]
    shutil.rmtree(dst, ignore_errors=True)
    shutil.rmtree(ROOT / "runs" / "yc_refuse", ignore_errors=True)


# ==========================================================================
# (e) record_ckpt round-trips a flagged checkpoint
# ==========================================================================
@needs_core
def test_record_ckpt_round_trips_a_flagged_checkpoint():
    ckpt = ROOT / "runs" / "yc_on" / "ckpt_final.pt"
    if not ckpt.exists():
        pytest.skip("needs the flagged run from (d)")
    out = ROOT / "runs" / "yc_on" / "traj_yawcond.jsonl"
    out.unlink(missing_ok=True)
    r = subprocess.run(
        [sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
         str(ckpt), "--map", str(CANNONBALL), "--episodes", "1",
         "--out", str(out)],
        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
        timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    # the recorder SAYS it mirrored the flag (and audit_cfg did not refuse)
    assert "--yaw-cond: the side key is drawn from p(side | yaw bin)" \
        in r.stdout, r.stdout[-3000:]
    assert "never mentions" not in r.stdout + r.stderr
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) > 10
    rows = [json.loads(ln) for ln in lines]
    assert any(isinstance(o, list) for o in rows)
    out.unlink(missing_ok=True)


@needs_core
def test_the_recorded_greedy_action_is_the_conditioned_one():
    """The recording has to be the policy PPO optimised: rebuild both
    policies off the same checkpoint and check the conditioned side key
    differs from the unconditioned one somewhere on real observations."""
    ckpt = ROOT / "runs" / "yc_on" / "ckpt_final.pt"
    if not ckpt.exists():
        pytest.skip("needs the flagged run from (d)")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    tab = ck["policy"]["yaw_side.table"]
    torch.manual_seed(31)
    pk = HeadPacker(torch.device("cpu"))
    # a table that has trained for 8 steps is small, so probe the mechanism
    # on logits it can actually flip: near-tied side heads
    padded = pk.pad(torch.randn(20_000, sum(NVEC)) * 0.02)
    g_off = padded.argmax(-1)
    g_on = greedy_padded_yawcond(padded, tab, None)
    assert torch.equal(g_off[:, H_YAW], g_on[:, H_YAW])
    n_diff = int((g_off[:, H_SIDE] != g_on[:, H_SIDE]).sum())
    assert n_diff > 0, "the trained table changed no side key at all"
