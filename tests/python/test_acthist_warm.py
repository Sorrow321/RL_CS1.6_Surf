"""``--act-hist`` / ``--obs-compass`` as a WARM RESUME.

Both blocks were implemented (``surfgym/obsaux.py``) and neither has ever
been run in an arm, because the arms that matter are warm resumes of a
FINISHER and adding an observation column to a trained checkpoint changes
the width of the towers' first Linear. ``--priv-critic`` solved the same
problem for the value head (``widen_for_priv``); this is the same trick on
the scalar side, and the thing it has to get right that the value head did
not is WHERE the new columns go.

The scalar-side block is ``[fan | latch | 6*K act-hist | 5 compass]`` and it
enters both towers as ``scal[:, N_SCALAR:]``, concatenated BETWEEN the fused
trunk output and the GRU state (``Policy.heads``). So growing it is an
insert at ``feat_dim + old_width``, which is the tensor's tail only when
there is no recurrence - ``widen_for_route``'s unconditional trailing pad
silently feeds a recurrent checkpoint permuted columns. What is pinned:

1. ``widen_for_obs`` + a strict ``load_state_dict`` makes the resumed policy
   compute the checkpoint's own function on its first forward: the new
   inputs multiply ZERO weights, so the logits and the value are the
   checkpoint's to ~1 ulp of fp32 (the wider GEMM's summation order, the
   same bound ``widen_for_priv`` documents). Whatever the new columns
   contain.
2. The recurrent case, which is the one the old trailing pad got wrong:
   with a GRU the insert is a MIDDLE insert, and ``widen_for_route`` moves
   the logits by 1.4e-2 where ``widen_for_obs`` does not move them at all.
3. Adam's exp_avg/exp_avg_sq get the same insert, zeros in the new columns.
4. The trained run: a plain checkpoint resumed with ``--act-hist 4``
   announces the widening, trains, and the new columns LEAVE zero - i.e.
   PPO is actually using them.
5. The refusals: the block only grows, and it cannot grow the history in
   front of an existing compass. Growing the COMPASS onto a checkpoint at
   an unchanged K is allowed (it is the trailing block).
6. ``tools/record_ckpt.py`` rebuilds the widened shape from the config and
   loads it STRICTLY.
7. No flag = the trainer that shipped, bit for bit: same config dump, same
   progress.csv, same weights, same Adam moments.

    python -m pytest tests/python/test_acthist_warm.py -q

CPU only (``CUDA_VISIBLE_DEVICES=-1``); set ``RL_SURF_MAPS`` to the main
checkout's maps/ when running from a worktree, or the goal-field tests skip
rather than start a 30-minute bake.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.obsaux import ACT_FEAT, CMP_FEAT                    # noqa: E402
from train_fast import (N_SCALAR, Policy, ck_obs_block,          # noqa: E402
                        widen_for_obs, widen_for_route)

_DLL_NAME = "surfcore.dll" if os.name == "nt" else "libsurfcore.so"
DLL = Path(os.environ.get("SURFCORE_DLL") or (ROOT / "build" / _DLL_NAME))


def _maps_dir():
    """cannonball AND its prebaked goal field, from the SAME directory.

    The caches key on the bsp's size + mtime_ns, so a worktree's copy of the
    map is a cache MISS and the trainer silently starts a 30-minute bake
    (CLAUDE.md). RL_SURF_MAPS points at the main checkout.
    """
    env = os.environ.get("RL_SURF_MAPS")
    for d in ([Path(env)] if env else []) + [ROOT / "maps"]:
        if ((d / "surf_src_cannonball.bsp").exists()
                and (d / "surf_src_cannonball.goal_32.npz").exists()):
            return d
    return None


MAPS = _maps_dir()
CANNONBALL = (MAPS or (ROOT / "maps")) / "surf_src_cannonball.bsp"
ROUTE = (MAPS or (ROOT / "maps")) / "surf_src_cannonball.route.npz"

needs_core = pytest.mark.skipif(not (CANNONBALL.exists() and DLL.exists()),
                                reason="needs the built core + cannonball")
needs_field = pytest.mark.skipif(
    MAPS is None,
    reason="needs the prebaked cannonball goal field (set RL_SURF_MAPS to "
           "the main checkout's maps/ when running from a worktree)")

W, H = 8, 4
FAN = 27                    # what maps/surf_src_cannonball.route.npz emits
K = 4
HIST = ACT_FEAT * K         # 24


def _pol(route_dim, seed, **kw):
    torch.manual_seed(seed)
    return Policy(N_SCALAR + route_dim + W * H, W, H, emb=32, hidden=24,
                  route_dim=route_dim, **kw)


def _ck(policy, moments=False):
    """A checkpoint payload shaped like the trainer's."""
    params = list(policy.parameters())
    st = {}
    if moments:
        for i, (name, p) in enumerate(policy.named_parameters()):
            if name in ("pi.0.weight", "vf.0.weight"):
                st[i] = {"exp_avg": torch.randn(*p.shape) + 1.0,
                         "exp_avg_sq": torch.rand(*p.shape) + 1.0,
                         "step": torch.tensor(7.0)}
    return {"policy": {k: v.clone() for k, v in policy.state_dict().items()},
            "optimizer": {"state": st,
                          "param_groups": [{"params": list(range(len(params)))}]}}


def _rows(n, route_dim, hist=None, seed=0):
    """(n, N_SCALAR + route_dim + W*H) observation rows; `hist` is the block
    to splice in after the fan, or None for a row without it."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, N_SCALAR + route_dim + W * H, generator=g)
    if hist is None:
        return x
    at = N_SCALAR + route_dim
    return torch.cat([x[:, :at], hist, x[:, at:]], dim=1)


# ==========================================================================
# (a) step 0 is the checkpoint's own function
# ==========================================================================
def test_widen_gives_the_checkpoints_own_logits_and_value():
    plain = _pol(FAN, 0).eval()
    ck = _ck(plain)
    assert ck_obs_block(ck, plain) == FAN
    wide = _pol(FAN + HIST, 99)               # a DIFFERENT draw
    assert widen_for_obs(ck, wide, FAN + HIST) == 2   # pi.0.w and vf.0.w
    wide.load_state_dict(ck["policy"])        # strict
    wide.eval()

    x = _rows(8, FAN, seed=3)
    for fill in (torch.zeros(8, HIST), torch.randn(8, HIST),
                 torch.full((8, HIST), -1.0)):
        la, va = plain(x)
        lb, vb = wide(_rows(8, FAN, hist=fill, seed=3))
        # ~1 ulp of fp32: the zero columns contribute 0 but move where the
        # accumulation happens (the bound widen_for_priv documents)
        assert torch.allclose(la, lb, atol=1e-6, rtol=0), \
            (la - lb).abs().max().item()
        assert torch.allclose(va, vb, atol=1e-6, rtol=0), \
            (va - vb).abs().max().item()


def test_the_new_columns_are_exactly_zero_and_the_old_ones_untouched():
    plain = _pol(FAN, 1)
    ck = _ck(plain)
    widen_for_obs(ck, _pol(FAN + HIST, 99), FAN + HIST)
    for name in ("pi.0.weight", "vf.0.weight"):
        got, was = ck["policy"][name], plain.state_dict()[name]
        assert tuple(got.shape) == (24, was.shape[1] + HIST)
        assert (got[:, -HIST:] == 0).all()          # the history block
        assert torch.equal(got[:, :was.shape[1]], was)   # everything before


def test_widen_is_idempotent_and_a_matching_checkpoint_is_untouched():
    wide = _pol(FAN + HIST, 2)
    ck = _ck(wide)
    assert widen_for_obs(ck, wide, FAN + HIST) == 0
    plain = _pol(FAN, 3)
    ck2 = _ck(plain)
    assert widen_for_obs(ck2, _pol(FAN + HIST, 99), FAN + HIST) == 2
    assert widen_for_obs(ck2, _pol(FAN + HIST, 99), FAN + HIST) == 0


def test_the_block_only_grows():
    ck = _ck(_pol(FAN + HIST, 4))
    with pytest.raises(SystemExit) as e:
        widen_for_obs(ck, _pol(FAN, 99), FAN)
    assert "only ever GROWS" in str(e.value)


def test_route_critic_only_leaves_the_actor_alone():
    """The flag routes the whole block to the value tower, so pi.0 never
    carries it - and the actor is then bit-identical, not 1-ulp identical.
    (The trainer refuses --route-critic-only WITH --act-hist for a different
    reason: a history the actor cannot read changes nothing.)"""
    plain = _pol(FAN, 5, route_critic_only=True).eval()
    ck = _ck(plain)
    wide = _pol(FAN + HIST, 99, route_critic_only=True)
    assert widen_for_obs(ck, wide, FAN + HIST) == 1        # vf.0.weight only
    wide.load_state_dict(ck["policy"])
    wide.eval()
    la, va = plain(_rows(6, FAN, seed=8))
    lb, vb = wide(_rows(6, FAN, hist=torch.randn(6, HIST), seed=8))
    assert torch.equal(la, lb)
    assert torch.allclose(va, vb, atol=1e-6, rtol=0)


def test_the_compass_grows_the_same_way_on_top_of_a_history():
    """[history | compass]: the compass is the TRAILING block, so adding it
    at an unchanged K leaves every history column where it was."""
    plain = _pol(FAN + HIST, 6).eval()
    ck = _ck(plain)
    wide = _pol(FAN + HIST + CMP_FEAT, 99)
    assert widen_for_obs(ck, wide, FAN + HIST + CMP_FEAT) == 2
    wide.load_state_dict(ck["policy"])
    wide.eval()
    la, va = plain(_rows(5, FAN + HIST, seed=9))
    lb, vb = wide(_rows(5, FAN + HIST, hist=torch.randn(5, CMP_FEAT), seed=9))
    assert torch.allclose(la, lb, atol=1e-6, rtol=0)
    assert torch.allclose(va, vb, atol=1e-6, rtol=0)


# ==========================================================================
# (a2) the recurrent case - the one a trailing pad gets WRONG
# ==========================================================================
def test_the_insert_is_positional_under_rnn_and_a_trailing_pad_is_not():
    """With a GRU the tower input is [f | route | h], so the new columns go
    in the MIDDLE. widen_for_route pushes them past the recurrent block and
    hands the checkpoint's weights permuted inputs."""
    rnn = dict(rnn="gru", rnn_size=8)
    plain = _pol(FAN, 7, **rnn).eval()
    x = _rows(5, FAN, seed=11)
    fill = torch.randn(5, HIST)
    x2 = _rows(5, FAN, hist=fill, seed=11)
    h = torch.randn(5, 8)
    la, va, ha = plain(x, h)

    ck = _ck(plain)
    assert ck_obs_block(ck, plain) == FAN        # the GRU width is excluded
    good = _pol(FAN + HIST, 99, **rnn)
    assert widen_for_obs(ck, good, FAN + HIST) == 2
    good.load_state_dict(ck["policy"])
    good.eval()
    lb, vb, hb = good(x2, h)
    assert torch.allclose(la, lb, atol=1e-6, rtol=0), \
        (la - lb).abs().max().item()
    assert torch.allclose(va, vb, atol=1e-6, rtol=0)
    assert torch.equal(ha, hb)          # the GRU reads `f` alone

    # the old trailing pad on the same checkpoint: a different policy
    bad_ck = _ck(plain)
    bad = _pol(FAN + HIST, 99, **rnn)
    assert widen_for_route(bad_ck, bad) == 2
    bad.load_state_dict(bad_ck["policy"])
    bad.eval()
    lc, vc, _ = bad(x2, h)
    assert (la - lc).abs().max() > 1e-4, (la - lc).abs().max().item()


# ==========================================================================
# (a3) Adam's moments
# ==========================================================================
def test_adam_moments_get_the_same_insert():
    plain = _pol(FAN, 8)
    ck = _ck(plain, moments=True)
    was = {i: {k: v.clone() for k, v in st.items() if torch.is_tensor(v)}
           for i, st in ck["optimizer"]["state"].items()}
    n = widen_for_obs(ck, _pol(FAN + HIST, 99), FAN + HIST)
    assert n == 2 + 4                       # two tensors + two moments each
    assert len(was) == 2
    for i, st in ck["optimizer"]["state"].items():
        for key in ("exp_avg", "exp_avg_sq"):
            t = st[key]
            assert tuple(t.shape) == (24, was[i][key].shape[1] + HIST)
            assert (t[:, -HIST:] == 0).all()
            assert torch.equal(t[:, :was[i][key].shape[1]], was[i][key])
        assert torch.equal(st["step"], torch.tensor(7.0))   # untouched


def test_string_keyed_optimizer_state_is_padded_too():
    """torch.load of a JSON-round-tripped state dict hands back str keys."""
    plain = _pol(FAN, 9)
    names = [n for n, _ in plain.named_parameters()]
    i = names.index("vf.0.weight")
    ck = _ck(plain)
    ck["optimizer"]["state"] = {
        str(i): {"exp_avg": torch.randn(24, N_SCALAR + FAN)}}
    # the row's scalar mask hides some columns, so use the real width
    ck["optimizer"]["state"][str(i)]["exp_avg"] = torch.randn(
        *plain.state_dict()["vf.0.weight"].shape)
    widen_for_obs(ck, _pol(FAN + HIST, 99), FAN + HIST)
    t = ck["optimizer"]["state"][str(i)]["exp_avg"]
    assert tuple(t.shape)[1] == plain.state_dict()["vf.0.weight"].shape[1] + HIST
    assert (t[:, -HIST:] == 0).all()


# ==========================================================================
# the trainer, end to end on the CPU (tiny nets, prebaked field)
# ==========================================================================
SMOKE = ["--reward", "race", "--envs", "64", "--spawn", "platform",
         "--lidar-w", "16", "--lidar-h", "8", "--lidar-cell", "32",
         "--lidar-range", "11500", "--lidar-near", "2000",
         "--emb", "64", "--hidden", "64",
         "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail",
         "--lr", "3e-4", "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2",
         "--vf", "0.5", "--ent", "0.005",
         "--n-steps", "8", "--epochs", "1", "--minibatches", "2",
         "--ep-ticks", "3000", "--time-pen", "0.005",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--maxvel", "4000",
         "--train-stride", "1", "--yaw-adaptive",
         "--respawn-frac", "0.9", "--respawn-margin", "10",
         "--respawn-reservoir", "1000", "--int-coef", "0.25",
         "--int-view", "8", "--int-speed", "3", "--ckpt-every", "1e9",
         "--eval-eps", "1", "--eval-greedy-only", "--no-eval-at-start"]


def _env():
    # CUDA_VISIBLE_DEVICES="-1", never "": Windows deletes an empty var.
    # The thread counts are INHERITED when the caller set them - these runs
    # are CPU-bound and the machine may be busy - and default to 8.
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "8"),
             NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS", "8"),
             SURFCORE_DLL=str(DLL))
    if MAPS is not None:
        e["RL_SURF_MAPS"] = str(MAPS)
    return e


def _train(run, extra, root=ROOT, script="train_fast.py", timeout=2400,
           check=True):
    shutil.rmtree(root / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-u", str(root / "python" / script),
           "--map", str(CANNONBALL), "--run", run] + SMOKE + extra
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                       cwd=str(root), timeout=timeout, encoding="utf-8",
                       errors="replace")
    if check:
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r, root / "runs" / run


@needs_core
@needs_field
def test_warm_resume_announces_the_widening_trains_and_leaves_zero(tmp_path):
    """(4) + (6): the whole arm, on a CPU-sized version of it."""
    r, ctl = _train("ah_ctl", ["--route", str(ROUTE), "--steps", "4096",
                              "--record-every", "1e12"])
    ck0 = torch.load(ctl / "ckpt_latest.pt", map_location="cpu",
                     weights_only=False)
    assert int(ck0["config"].get("act_hist") or 0) == 0
    w0 = ck0["policy"]["pi.0.weight"].shape[1]

    # --record-every 2048 so the trainer's OWN eval path runs with the block
    # (it builds a SECOND ObsAux for the one-env eval core, train_fast.py
    # `eval_aux`) - that path is what every number in the ledger comes from,
    # and --steps is the ABSOLUTE resumed counter, not a budget
    r, warm = _train("ah_warm", ["--act-hist", "4",
                                 "--ckpt", str(ctl / "ckpt_latest.pt"),
                                 "--steps", "8192", "--record-every", "2048"])
    out = r.stdout
    assert "greedy:" in out                    # the eval ran with the block
    assert len(list(warm.glob("traj_*.jsonl"))) >= 1
    assert "--act-hist 4:" in out and "ZERO columns" in out
    assert "grows from zero" in out
    assert "act-hist 4 decisions (24 cols" in out
    assert "resumed" in out
    # the route came back off the checkpoint's own config, so the arm is the
    # control plus exactly one block
    assert "route=surf_src_cannonball.route.npz" in out

    ck1 = torch.load(warm / "ckpt_latest.pt", map_location="cpu",
                     weights_only=False)
    assert ck1["config"]["act_hist"] == 4
    assert ck1["config"]["obs_compass"] == 0
    for name in ("pi.0.weight", "vf.0.weight"):
        assert ck1["policy"][name].shape[1] == w0 + HIST
        # (b) PPO actually used them: after a few hundred updates' worth of
        # steps the columns that started at zero are no longer zero
        assert (ck1["policy"][name][:, -HIST:] != 0).any(), name
    # and everything the checkpoint already had is still trainable, i.e. the
    # optimizer state came back at the widened shape rather than being reset
    st = ck1["optimizer"]["state"]
    idx = {n: i for i, (n, _) in enumerate(
        Policy(N_SCALAR + FAN + HIST + 16 * 8, 16, 8, emb=64, hidden=64,
               route_dim=FAN + HIST).named_parameters())}
    for name in ("pi.0.weight", "vf.0.weight"):
        m = st[idx[name]]["exp_avg"]
        assert tuple(m.shape) == tuple(ck1["policy"][name].shape)

    # (6) the recorder rebuilds the widened shape from the config and loads
    # it STRICTLY - the check that has caught every unmirrored flag
    traj = tmp_path / "rec.jsonl"
    rr = subprocess.run(
        [sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
         str(warm / "ckpt_latest.pt"), "--map", str(CANNONBALL),
         "--episodes", "1", "--out", str(traj)],
        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
        timeout=2400, encoding="utf-8", errors="replace")
    assert rr.returncode == 0, rr.stdout[-4000:] + rr.stderr[-4000:]
    assert "act-hist 4 decisions (24 cols" in rr.stdout
    assert traj.exists() and traj.stat().st_size > 0

    # a bare resume of the WIDENED checkpoint restores act_hist and needs no
    # widening of its own
    r, again = _train("ah_again", ["--ckpt", str(warm / "ckpt_latest.pt"),
                                   "--steps", "4096",
                                   "--record-every", "1e12"])
    assert "act_hist=4" in r.stdout
    assert "ZERO columns" not in r.stdout
    for d in (ctl, warm, again):
        shutil.rmtree(d, ignore_errors=True)


@needs_core
@needs_field
def test_the_compass_can_be_added_and_the_two_mismatches_are_refused():
    """(5). One trained checkpoint, three resumes: the allowed growth and
    the two shapes of refusal."""
    r, base = _train("ah_cmp_base", ["--act-hist", "2", "--steps", "4096",
                                     "--record-every", "1e12"])
    ck = base / "ckpt_latest.pt"
    cfg = torch.load(ck, map_location="cpu", weights_only=False)["config"]
    assert cfg["act_hist"] == 2 and cfg["obs_compass"] == 0

    # ALLOWED: the compass is the trailing block, so it grows at K=2
    r, cmp_run = _train("ah_cmp_on", ["--obs-compass", "1", "--ckpt", str(ck),
                                      "--steps", "4096",
                                      "--record-every", "1e12"])
    assert "--obs-compass 1:" in r.stdout and "ZERO columns" in r.stdout
    assert "act_hist=2" in r.stdout          # restored, not re-specified
    ck2 = torch.load(cmp_run / "ckpt_latest.pt", map_location="cpu",
                     weights_only=False)
    assert ck2["config"]["obs_compass"] == 1 and ck2["config"]["act_hist"] == 2

    # REFUSED: shrinking the history
    r, _ = _train("ah_cmp_small", ["--act-hist", "1", "--ckpt", str(ck),
                                   "--steps", "4096"], check=False)
    assert r.returncode != 0
    assert "only ever GROWS" in (r.stdout + r.stderr)

    # REFUSED: growing the history IN FRONT of an existing compass
    r, _ = _train("ah_cmp_shift",
                  ["--act-hist", "4", "--obs-compass", "1",
                   "--ckpt", str(cmp_run / "ckpt_latest.pt"),
                   "--steps", "4096"], check=False)
    assert r.returncode != 0
    msg = r.stdout + r.stderr
    assert "[history | compass]" in msg and "permuted" in msg

    for d in (base, cmp_run, ROOT / "runs" / "ah_cmp_small",
              ROOT / "runs" / "ah_cmp_shift"):
        shutil.rmtree(d, ignore_errors=True)


# ==========================================================================
# (7) no flag == the trainer that shipped, bit for bit
# ==========================================================================
def _unpatched_trainer(dst: Path):
    """The pre-patch train_fast.py out of git, or None."""
    for ref in ("baseline", "origin/baseline", "main", "origin/main", "HEAD^"):
        # BYTES, never text: this console is cp1251 and train_fast.py is
        # UTF-8 with em dashes - a locale round trip would corrupt the copy
        r = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if r.returncode == 0 and b"widen_for_obs" not in r.stdout:
            dst.write_bytes(r.stdout)
            return ref
    return None


@needs_core
@needs_field
def test_no_flag_is_bit_identical_to_the_unpatched_trainer():
    """The widening is opt-in, and 'opt-in' has to mean the CONTROL arm is
    the run that shipped: same config dump, same CSV, same weights, same
    Adam moments. The copy lives in python/ so its ROOT (= parent.parent)
    and its sys.path resolve exactly like the real one."""
    base_py = ROOT / "python" / "_train_fast_unpatched.py"
    ref = _unpatched_trainer(base_py)
    if ref is None:
        base_py.unlink(missing_ok=True)
        pytest.skip("no pre-patch train_fast.py reachable from git")
    try:
        _train("ah_bit_base", ["--steps", "8192", "--record-every", "1e12"],
               script=base_py.name)
        _train("ah_bit_new", ["--steps", "8192", "--record-every", "1e12"])
    finally:
        base_py.unlink(missing_ok=True)

    a, b = ROOT / "runs" / "ah_bit_base", ROOT / "runs" / "ah_bit_new"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb, [k for k in set(ca) | set(cb)
                      if ca.get(k, "@") != cb.get(k, "@")]

    ta = (a / "progress.csv").read_text(encoding="utf-8").splitlines()
    tb = (b / "progress.csv").read_text(encoding="utf-8").splitlines()
    assert ta[0] == tb[0]
    assert len(ta) == len(tb) >= 2
    for ra, rb in zip(ta[1:], tb[1:]):
        fa, fb = ra.split(","), rb.split(",")
        fa[3] = fb[3] = ""                  # time/fps is wall-clock
        assert fa == fb

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
    for i in oa:
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[i][k], ob[i][k]), (i, k)
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
