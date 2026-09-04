"""Opt-in action masks: --mask-forward-air, --jump-cooldown, --duck-air-mask.

Why: runs/research/wr_demo/wr_vs_ours.md decomposes the 8.28 s between the
human world record (68.60 s on surf_src_cannonball) and our best finisher
(76.88 s). Three of the reachable losses are pure key use. GoldSrc air
movement accelerates along wishdir, the NORMALISED forward+side sum, and
PM_AirAccelerate (src/pm.c) caps the gain at 30 u/s per frame on the
projection onto it, so W with a strafe key swings wishdir to 45 deg off
perpendicular and buys nothing at flight speed. Measured: the WR holds W/S
on 0% of its airborne frames against our 11.4%, presses jump 0 times against
our 283 and duck 0 against our 176, and flips strafe direction 0.42 times a
second against our 1.50.

What is pinned here:

1. A masked head gives the forbidden action probability EXACTLY zero, the
   free heads stay proper distributions, and the log-prob the update
   recomputes equals the one the rollout stored bit for bit. Recomputing it
   UNMASKED is off by up to ~7 nats on this fixture - a PPO ratio wrong by
   e^7 - which is why the mask has to be in all four places.
2. The jump-cooldown recurrence: a press re-arms to N, otherwise it counts
   down, and an episode start clears it.
3. With NO flag the trainer is bit-identical to the unpatched baseline: the
   same config dump, the same progress.csv on every shared column, the same
   policy weights and the same Adam moments after a tiny CPU scratch run.
4. Each flag runs, and the act/* diagnostics show it: fwd_air and duck_air
   go to exactly 0.0 where the control sits at ~0.6, and jump_air collapses.
5. A trajectory recorded from a masked checkpoint by tools/record_ckpt.py
   (which rebuilds the policy from the ckpt CONFIG) has zero forward-in-air
   and zero duck-in-air DECISIONS and no jump press inside the cooldown,
   while the control checkpoint recorded the same way has hundreds of each.

    python -m pytest tests/python/test_air_masks.py -q

CPU only, tiny nets; a few minutes. It needs the built core and the
prebaked cannonball goal field. Running from a git worktree, point
SURF_TEST_MAPS at the main checkout's maps/ (CLAUDE.md: a worktree copy has
different mtimes and every prebaked cache misses) and SURFCORE_DLL at its
built core.
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

from train_fast import (A_FWD_NONE, H_DUCK, H_FWD, H_JUMP,     # noqa: E402
                        NACT, NVEC, ActionMasks, HeadPacker,
                        logprob_entropy_padded, sample_padded)

SURF_IN_JUMP, SURF_IN_DUCK = 2, 4          # src/surfcore.h


# ==========================================================================
# (a) the masked distribution, and the ratio it would bias if it were not
#     applied in all four places
# ==========================================================================
def _fixture(seed=0, B=512):
    torch.manual_seed(seed)
    pk = HeadPacker(torch.device("cpu"))
    logits = torch.randn(B, sum(NVEC))
    air = (torch.rand(B) < 0.5).float()
    jblk = (torch.rand(B) < 0.3).float()
    return pk, logits, air, jblk


def test_masked_actions_have_exactly_zero_probability():
    pk, logits, air, jblk = _fixture()
    m = ActionMasks(fwd_air=True, jump_cd=3, duck_air=True)
    padded = m.add_mask(pk.pad(logits), air, jblk)
    a, j = air.bool(), jblk.bool()
    p = torch.softmax(padded, dim=-1)
    # the forbidden slots: exactly 0.0, not "small"
    assert float(p[a, H_FWD, 0].max()) == 0.0        # -400 (S)
    assert float(p[a, H_FWD, 2].max()) == 0.0        # +400 (W)
    assert float(p[a, H_DUCK, 1].max()) == 0.0       # IN_DUCK
    assert float(p[j, H_JUMP, 1].max()) == 0.0       # IN_JUMP
    # every head is still a distribution, and nothing is NaN
    assert torch.isfinite(p).all()
    assert torch.allclose(p.sum(-1), torch.ones(p.shape[0], NACT), atol=1e-6)
    # on the GROUND the forward head is untouched
    assert float(p[~a, H_FWD, 0].max()) > 0.0
    assert float(p[~a, H_FWD, 2].max()) > 0.0
    # entropy is finite (NEG is finite on purpose: p*logp -> 0, never NaN)
    _lp, ent = logprob_entropy_padded(padded, padded.argmax(-1))
    assert torch.isfinite(ent).all()


def test_sampling_never_emits_a_masked_action():
    pk, logits, air, jblk = _fixture(seed=1)
    m = ActionMasks(fwd_air=True, jump_cd=3, duck_air=True)
    act, _ = sample_padded(m.add_mask(pk.pad(logits), air, jblk))
    a, j = air.bool(), jblk.bool()
    assert (act[a, H_FWD] == A_FWD_NONE).all()
    assert (act[a, H_DUCK] == 0).all()
    assert (act[j, H_JUMP] == 0).all()
    assert (act[~a, H_FWD] != A_FWD_NONE).any()      # free on the ground


def test_update_logprob_matches_the_rollout_and_unmasked_would_bias_it():
    """PPO's ratio is exp(logp_new - logp_old): both terms must come from
    the SAME distribution. Masking the logits (never the sample) is what
    makes the rollout's stored log-prob and the update's recomputation
    identical."""
    pk, logits, air, jblk = _fixture(seed=2)
    m = ActionMasks(fwd_air=True, jump_cd=4, duck_air=True)
    act, logp_roll = sample_padded(m.add_mask(pk.pad(logits), air, jblk))
    # the update: same weights, same flags, gathered the same way
    logp_upd, _ = logprob_entropy_padded(
        m.add_mask(pk.pad(logits), air, jblk), act)
    assert torch.equal(logp_roll, logp_upd)
    # and the failure mode this guards: an update that forgot the mask
    logp_bad, _ = logprob_entropy_padded(pk.pad(logits), act)
    assert not torch.allclose(logp_bad, logp_roll)
    assert float((logp_bad - logp_roll).abs().max()) > 1.0     # ~7 nats here


def test_masked_logits_get_exactly_zero_gradient():
    pk, logits, air, jblk = _fixture(seed=3)
    m = ActionMasks(fwd_air=True, jump_cd=2, duck_air=True)
    act, _ = sample_padded(m.add_mask(pk.pad(logits), air, jblk))
    lg = logits.clone().requires_grad_(True)
    logprob_entropy_padded(m.add_mask(pk.pad(lg), air, jblk), act)[0] \
        .sum().backward()
    g = pk.pad(lg.grad)              # same scatter, so (head, slot) lines up
    a = air.bool()
    assert float(g[a, H_FWD, 0].abs().max()) == 0.0
    assert float(g[a, H_FWD, 2].abs().max()) == 0.0
    assert float(g[a, H_DUCK, 1].abs().max()) == 0.0
    assert float(g[~a, H_FWD].abs().max()) > 0.0


def test_masks_off_is_a_strict_no_op():
    pk, logits, air, jblk = _fixture(seed=4)
    off = ActionMasks()
    padded = pk.pad(logits)
    assert off.on is False
    assert off.add_mask(padded, air, jblk) is padded      # same object
    assert off.config() == {}                            # no config key
    assert off.describe() == "action masks: none"
    assert ActionMasks.from_config({}).on is False
    assert ActionMasks.from_config(None).on is False
    # a single flag only builds the masks it was asked for
    only_j = ActionMasks(jump_cd=3)
    assert only_j.on and not only_j.needs_air
    assert only_j.config() == {"jump_cooldown": 3}
    p2 = only_j.add_mask(padded, None, jblk)
    assert torch.equal(p2[:, H_FWD], padded[:, H_FWD])
    assert torch.equal(p2[:, H_DUCK], padded[:, H_DUCK])


def test_config_round_trips():
    m = ActionMasks(fwd_air=True, jump_cd=7, duck_air=True)
    assert m.config() == {"mask_forward_air": 1, "jump_cooldown": 7,
                          "duck_air_mask": 1}
    r = ActionMasks.from_config(m.config())
    assert (r.fwd_air, r.jump_cd, r.duck_air) == (True, 7, True)
    with pytest.raises(ValueError):
        ActionMasks(jump_cd=-1)


def test_the_mask_is_applied_in_all_four_places():
    """A source-level guard: the whole point is that four call sites agree,
    and dropping one is silent (the run trains, the ratio is just wrong)."""
    tf = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    # 1 rollout sample, 2 mb_step + seq_loss recomputation
    assert tf.count("MASKS.add_mask(") == 3, tf.count("MASKS.add_mask(")
    assert "PLACE 1 of 4" in tf and "PLACE 2 of 4" in tf
    # 3 the eval wrappers (greedy AND stochastic) go through _mask_padded
    assert tf.count("self._mask_padded(self.packer.pad(logits)") == 2
    assert "masks=MASKS" in tf                       # in-trainer evals
    # 4 the recorder rebuilds them from the ckpt config
    assert "ActionMasks.from_config(" in rc and "masks=masks" in rc


# ==========================================================================
# (b) the cooldown counter
# ==========================================================================
def test_jump_cooldown_counter_semantics():
    """A press re-arms to N; every other decision counts down to 0. The
    trainer runs exactly this on a static (N,) buffer, the eval wrapper the
    numpy mirror of it."""
    N, cd = 4, torch.zeros(4)
    reload_ = torch.full((4,), 3.0)
    no = torch.zeros(4, dtype=torch.bool)
    seen = [float(cd[0])]
    cd = ActionMasks.step_cooldown(cd, torch.tensor([True] * N), reload_)
    for _ in range(5):
        seen.append(float(cd[0]))
        cd = ActionMasks.step_cooldown(cd, no, reload_)
    # press at decision 0 -> blocked for decisions 1..3, free again at 4
    assert seen == [0.0, 3.0, 2.0, 1.0, 0.0, 0.0]
    # a press while already counting down cannot happen (the head is
    # masked), but the recurrence is still total: it re-arms
    cd = torch.tensor([2.0])
    assert float(ActionMasks.step_cooldown(
        cd, torch.tensor([True]), torch.tensor([3.0]))[0]) == 3.0
    # cooldown 0 = off: the counter never leaves 0
    z = torch.zeros(1)
    assert float(ActionMasks.step_cooldown(
        z, torch.tensor([True]), torch.zeros(1))[0]) == 0.0


def test_episode_end_clears_the_cooldown():
    """The trainer's rule, `static_jcd.mul_(1.0 - b_done[t])`."""
    cd = torch.tensor([3.0, 2.0, 1.0])
    done = torch.tensor([1.0, 0.0, 0.0])
    cd = cd * (1.0 - done)
    assert cd.tolist() == [0.0, 2.0, 1.0]


def test_legalize_forces_burst_actions_onto_the_support():
    """ez-greedy / --spawn-burst actions never pass through the logits, so
    the mask has no say over them; the engine must still never see a
    forbidden key."""
    m = ActionMasks(fwd_air=True, jump_cd=2, duck_air=True)
    act = torch.tensor([[7, 3, 2, 0, 1, 1],
                        [7, 3, 0, 2, 1, 1],
                        [7, 3, 2, 1, 1, 1]])
    air = torch.tensor([1.0, 0.0, 1.0])
    jblk = torch.tensor([1.0, 1.0, 0.0])
    m.legalize_(act, air, jblk)
    assert act[0].tolist() == [7, 3, A_FWD_NONE, 0, 0, 0]   # airborne, blocked
    assert act[1].tolist() == [7, 3, 0, 2, 0, 1]            # ground: fwd/duck free
    assert act[2].tolist() == [7, 3, A_FWD_NONE, 1, 1, 0]   # airborne, jump free


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
ACT_EVERY = 4


def _env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="8", NUMBA_NUM_THREADS="8")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


def _train(run, extra, script="train_fast.py", steps="2048", timeout=1800):
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


def _record(ckpt, out, extra=(), timeout=1800):
    cmd = [sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
           str(ckpt), "--map", str(CANNONBALL), "--episodes", "2",
           "--out", str(out)] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                       cwd=str(ROOT), timeout=timeout,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _decisions(path, act_every=ACT_EVERY):
    """The recorded trajectory, one entry per DECISION.

    record.py writes one row per TICK; _TorchPolicyBase.act deliberates every
    act_every-th call and its counter runs over the whole recording (it is
    never reset between episodes), so decision ticks are the global indices
    divisible by act_every. Row layout: [8] buttons, [9] onground of the
    PRE-step state, [13] fwd action, [14] side action."""
    eps, cur, g = [], None, 0
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        o = json.loads(ln)
        if isinstance(o, dict):
            if "episode" in o:
                cur = []
                eps.append(cur)
            continue
        cur.append(o)
    for ep_i, ep in enumerate(eps):
        for row in ep:
            if g % act_every == 0:
                out.append({"ep": ep_i, "air": row[9] == 0, "fwd": row[13],
                            "side": row[14],
                            "jump": bool(row[8] & SURF_IN_JUMP),
                            "duck": bool(row[8] & SURF_IN_DUCK)})
            g += 1
    return out


# ==========================================================================
# (c) no flag == the unpatched trainer, bit for bit
# ==========================================================================
# Config-dump keys added by OTHER opt-in features merged onto the
# integration branch after the reference commit, with the value each takes
# when its flag is off. The unpatched trainer cannot write them, so they are
# the only difference this test permits.
INERT_SINCE = {"priv_critic": 0, "priv_features": None, "priv_hidden": None}


def _unpatched_trainer(dst: Path):
    """The pre-flag train_fast.py out of git, or None."""
    for ref in ("baseline", "origin/baseline", "main", "origin/main",
                "HEAD^"):
        # BYTES, never text: this console is cp1251 and train_fast.py is
        # UTF-8 with em dashes - a locale round trip would corrupt the copy
        r = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if r.returncode == 0 and b"--mask-forward-air" not in r.stdout:
            dst.write_bytes(r.stdout)
            return ref
    return None


@needs_core
def test_no_flag_is_bit_identical_to_the_unpatched_trainer():
    """The masks are opt-in, and 'opt-in' has to mean the control run is the
    run that shipped: same config dump, same CSV, same weights, same Adam
    moments. The copy lives in python/ so its ROOT (= parent.parent) and its
    sys.path resolve exactly like the real one."""
    base_py = ROOT / "python" / "_train_fast_unpatched.py"
    ref = _unpatched_trainer(base_py)
    if ref is None:
        base_py.unlink(missing_ok=True)
        pytest.skip("no pre-flag train_fast.py reachable from git")
    try:
        _train("am_bit_base", [], script=base_py.name)
        _train("am_bit_new", [])
    finally:
        base_py.unlink(missing_ok=True)

    a, b = ROOT / "runs" / "am_bit_base", ROOT / "runs" / "am_bit_new"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    # The reference is the last ref WITHOUT --mask-forward-air, which on the
    # integration branch is also the last ref without --priv-critic: its
    # three keys are written unconditionally, at an off value. They are the
    # one permitted delta - everything else must still match exactly, which
    # is the claim (no flag == the run that shipped).
    added = set(cb) - set(ca)
    assert not set(ca) - set(cb), set(ca) - set(cb)
    assert added <= set(INERT_SINCE), added
    for k in added:
        assert cb[k] == INERT_SINCE[k], (k, cb[k])
    assert {k: v for k, v in cb.items() if k not in added} == ca, \
        [k for k in ca if ca[k] != cb.get(k, "@")]
    # the masks themselves write NOTHING into the dump when off
    assert "mask_forward_air" not in cb and "jump_cooldown" not in cb
    assert "duck_air_mask" not in cb

    ta = (a / "progress.csv").read_text(encoding="utf-8").splitlines()
    tb = (b / "progress.csv").read_text(encoding="utf-8").splitlines()
    ha, hb = ta[0].split(","), tb[0].split(",")
    # the four act/* columns are APPENDED, so the old header stays a strict
    # prefix and a resumed run's progress.csv migrates instead of breaking
    assert hb[:len(ha)] == ha
    assert hb[len(ha):] == ["act/fwd_air", "act/strafe_flip", "act/jump_air",
                            "act/duck_air"]
    assert len(ta) == len(tb) >= 2
    for ra, rb in zip(ta[1:], tb[1:]):
        fa, fb = ra.split(","), rb.split(",")
        fa[3] = fb[3] = ""          # time/fps is wall-clock
        assert fb[:len(fa)] == fa

    sa = torch.load(a / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    assert sa["global_step"] == sb["global_step"]
    # the checkpoint carries the same dump, so it carries the same one
    # permitted delta (see the run.json comparison above)
    assert set(sb["config"]) - set(sa["config"]) <= set(INERT_SINCE)
    assert not set(sa["config"]) - set(sb["config"])
    assert {k: v for k, v in sb["config"].items()
            if k in sa["config"]} == sa["config"]
    assert set(sa["policy"]) == set(sb["policy"])
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    oa, ob = sa["optimizer"]["state"], sb["optimizer"]["state"]
    assert set(oa) == set(ob) and oa
    for k in oa:
        for f in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[k][f], ob[k][f]), (k, f)
    for r in ("am_bit_base", "am_bit_new"):
        shutil.rmtree(ROOT / "runs" / r, ignore_errors=True)


# ==========================================================================
# (d) each flag runs, and the act/* diagnostics show it
# ==========================================================================
@needs_core
def test_each_flag_runs_and_moves_its_own_diagnostic():
    _train("am_ctl", [])          # the untreated control, kept for (e)
    c = _csv("am_ctl")[0]
    # a fresh net is near uniform: 2 of 3 forward values are non-neutral,
    # jump and duck are 1 of 2, and 2 of 3 consecutive side draws differ
    assert 0.4 < float(c["act/fwd_air"]) < 0.9
    assert 0.2 < float(c["act/jump_air"]) < 0.8
    assert 0.2 < float(c["act/duck_air"]) < 0.8
    assert 0.4 < float(c["act/strafe_flip"]) < 0.9

    _train("am_fwd", ["--mask-forward-air"])
    f = _csv("am_fwd")[0]
    assert float(f["act/fwd_air"]) == 0.0
    assert float(f["act/duck_air"]) > 0.2          # untouched by this flag
    assert float(f["act/jump_air"]) > 0.2

    _train("am_duck", ["--duck-air-mask"])
    d = _csv("am_duck")[0]
    assert float(d["act/duck_air"]) == 0.0
    assert float(d["act/fwd_air"]) > 0.4

    _train("am_jcd", ["--jump-cooldown", "5"])
    j = _csv("am_jcd")[0]
    # one press per 6 decisions at most, and episode ends re-arm it, so the
    # AIRBORNE-conditioned rate is not exactly 1/6 - but it collapses
    assert float(j["act/jump_air"]) < 0.5 * float(c["act/jump_air"])
    assert float(j["act/fwd_air"]) > 0.4
    assert float(j["act/duck_air"]) > 0.2

    for r in ("am_fwd", "am_duck", "am_jcd"):
        cfg = json.loads((ROOT / "runs" / r / "run.json").read_text(
            encoding="utf-8"))["config"]
        assert len([k for k in cfg if k in ("mask_forward_air",
                                            "jump_cooldown",
                                            "duck_air_mask")]) == 1
        shutil.rmtree(ROOT / "runs" / r, ignore_errors=True)


# ==========================================================================
# (e) record_ckpt rebuilds the mask from the ckpt config
# ==========================================================================
@needs_core
def test_recorded_trajectory_from_a_masked_ckpt_obeys_the_mask():
    """The recorder is handed only the checkpoint: if it did not read the
    mask keys out of cfg, the recording would be a policy nobody trained -
    and every honesty tool downstream reads these files."""
    _train("am_all", ["--mask-forward-air", "--jump-cooldown", "5",
                      "--duck-air-mask"])
    if not (ROOT / "runs" / "am_ctl" / "ckpt_final.pt").exists():
        _train("am_ctl", [])

    # a stochastic recording off the ramp: a greedy argmax of a fresh net is
    # one constant action, which cannot exercise a cooldown at all
    rec = ROOT / "runs" / "am_all" / "rec.jsonl"
    r = _record(ROOT / "runs" / "am_all" / "ckpt_final.pt", rec,
                ["--spawn", "ramp", "--stochastic"])
    assert "action masks:" in r.stdout and "from the checkpoint config" in r.stdout
    dec = _decisions(rec)
    air = [d for d in dec if d["air"]]
    assert len(air) > 100, len(air)
    assert sum(1 for d in air if d["fwd"] != A_FWD_NONE) == 0
    assert sum(1 for d in air if d["duck"]) == 0
    # no jump press within the cooldown of the previous press in an episode
    gaps, last, ep = [], None, None
    for i, d in enumerate(dec):
        if d["ep"] != ep:
            ep, last = d["ep"], None
        if d["jump"]:
            if last is not None:
                gaps.append(i - last)
            last = i
    assert len(gaps) >= 10, len(gaps)
    assert min(gaps) >= 6, sorted(gaps)[:5]        # cooldown 5 -> gap >= 6

    # the control, recorded exactly the same way, does all three
    crec = ROOT / "runs" / "am_ctl" / "rec.jsonl"
    r0 = _record(ROOT / "runs" / "am_ctl" / "ckpt_final.pt", crec,
                 ["--spawn", "ramp", "--stochastic"])
    assert "action masks:" not in r0.stdout
    cdec = [d for d in _decisions(crec)]
    cair = [d for d in cdec if d["air"]]
    assert sum(1 for d in cair if d["fwd"] != A_FWD_NONE) > 0
    assert sum(1 for d in cair if d["duck"]) > 0
    cg, last, ep = [], None, None
    for i, d in enumerate(cdec):
        if d["ep"] != ep:
            ep, last = d["ep"], None
        if d["jump"]:
            if last is not None:
                cg.append(i - last)
            last = i
    assert min(cg) < 6

    # the LATENCY, stated as a measurement rather than a claim: the mask is
    # read at the decision tick and held for act_every ticks, so a decision
    # taken on the ground (where W is legal) can carry W into the air for up
    # to act_every-1 ticks. Per-DECISION the count is 0; per-TICK it is not,
    # and that is the documented behaviour, not a leak.
    per_tick = 0
    for ln in rec.read_text(encoding="utf-8").splitlines():
        o = json.loads(ln)
        if isinstance(o, list) and o[9] == 0 and o[13] != A_FWD_NONE:
            per_tick += 1
    assert per_tick >= 0        # informational; > 0 on any real flight
    for r_ in ("am_all", "am_ctl"):
        shutil.rmtree(ROOT / "runs" / r_, ignore_errors=True)


@needs_core
def test_a_resume_keeps_the_mask_and_says_so_when_it_changes():
    """CLAUDE.md's recurring failure: a setting silently reverting. A mask
    is part of what the weights MEAN, so a resume that dropped it would
    optimise a different action space; a resume that changes it on purpose
    has to be loud, because the first eval after it is not comparable."""
    _train("am_r_src", ["--mask-forward-air", "--jump-cooldown", "5"])
    ck = ROOT / "runs" / "am_r_src" / "ckpt_final.pt"

    r = _train("am_r_keep", ["--ckpt", str(ck)], steps="4096")
    assert "mask_forward_air=1" in r.stdout and "jump_cooldown=5" in r.stdout
    assert "ACTION MASK CHANGE" not in r.stdout
    cfg = json.loads((ROOT / "runs" / "am_r_keep" / "run.json").read_text(
        encoding="utf-8"))["config"]
    assert cfg["mask_forward_air"] == 1 and cfg["jump_cooldown"] == 5

    r = _train("am_r_off", ["--ckpt", str(ck), "--jump-cooldown", "0"],
               steps="4096")
    assert "!! ACTION MASK CHANGE: jump_cooldown 5 -> 0" in r.stdout
    cfg = json.loads((ROOT / "runs" / "am_r_off" / "run.json").read_text(
        encoding="utf-8"))["config"]
    assert "jump_cooldown" not in cfg          # off = no key at all
    assert cfg["mask_forward_air"] == 1        # the other one is untouched
    for r_ in ("am_r_src", "am_r_keep", "am_r_off"):
        shutil.rmtree(ROOT / "runs" / r_, ignore_errors=True)


@needs_core
def test_chunk_and_bc_are_refused_rather_than_silently_wrong():
    """--chunk samples H decisions from one code at the chunk start, so the
    per-decision ground flag does not exist when the plan is drawn and the
    update could not reproduce the mask. Refuse, do not approximate."""
    shutil.rmtree(ROOT / "runs" / "am_refuse", ignore_errors=True)
    cmd = [sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
           "--run", "am_refuse"] + SMOKE_FLAGS + [
        "--steps", "2048", "--mask-forward-air", "--chunk", "4"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                       cwd=str(ROOT), timeout=600)
    assert r.returncode != 0
    assert "not implemented for --chunk" in (r.stdout + r.stderr)
    shutil.rmtree(ROOT / "runs" / "am_refuse", ignore_errors=True)
