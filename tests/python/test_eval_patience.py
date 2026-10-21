"""--eval-patience SECS (2026-10-21, autoresearch r4b3): end a greedy EVAL episode once its
best field progress has not grown by 32u (cumulative, any rate) for SECS seconds.

(a) The hook (make_eval_patience_hook) on a fake core and a line field:
    * a line that keeps gaining - however slowly, or after a setback - is never cut;
    * a line that stops gaining is cut after exactly the window, once per window, and the
      episode index is recorded;
    * an episode boundary re-arms it; off-field samples are no progress; 0 = off.
(b) The greedy-line pool reads a cut episode as the truncated one it would have been
    (only its best-progress anchor), never as a death (surfgym.greedy_spawn.as_truncated).
(c) The recorder lists the flag as training-only.
(d) The trainer (CPU, sidistic): the eval episodes are cut, the trajectory header and the
    config say so, a bare resume restores it; a negative window is refused.
(e) Flag off: the trainer before this change and after it give the same config,
    progress.csv (fps aside), eval trajectory bytes and weights on the same smoke.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.core import STATE_DTYPE                               # noqa: E402
from surfgym.greedy_spawn import GreedyLinePool, as_truncated      # noqa: E402

from test_ez_viewc import SID, SMOKE, TRAIN, _csv, _run, _train, needs_run  # noqa: E402


# ==========================================================================
# (a) the hook
# ==========================================================================
class _FakeCore:
    def __init__(self, n=1):
        self.num_envs = n
        self.failed = np.zeros(n, np.uint8)
        self.states_view = np.zeros(n, dtype=[("origin", np.float64, (3,))])

    def force_fail(self, mask):
        self.failed |= np.asarray(mask, np.uint8)


class _LineField:
    """d = x (moving -x is progress); x >= 1e6 is off the field (the sentinel)."""
    reach_max = 1e6 + 16.0
    cell = 32.0

    def sample(self, pts):
        x = np.asarray(pts, np.float64)[:, 0].copy()
        return np.where(x >= 1e6, 1e6 + 20.0, x)


def _hook(window, eps=32.0):
    from train_fast import make_eval_patience_hook
    core = _FakeCore(1)
    return core, make_eval_patience_hook(core, _LineField(), window, eps)


def _drive(core, hook, xs, ends=()):
    """One tick per x; ``ends`` = ticks at which the episode ends (done). -> kill ticks."""
    kills = []
    for t, x in enumerate(xs):
        core.states_view["origin"][0, 0] = float(x)
        core.failed[:] = 0
        end = t in ends
        hook(t, None, None, np.array([end]), np.array([False]))
        if core.failed[0]:
            kills.append(t)
    return kills


def test_a_line_that_keeps_gaining_is_never_cut_however_slowly():
    core, hook = _hook(100)
    # 1/3 u per tick: 32u every 97 ticks, just inside a 100-tick window
    assert _drive(core, hook, 5000.0 - np.arange(3000) / 3.0) == []
    assert hook.state["n"] == 0 and hook.state["killed"] == []
    # gaining, a 50-tick setback 400u back up the field, then gaining past the old best
    xs = list(5000.0 - np.arange(200)) + [5200.0] * 50 + list(4761.0 - np.arange(200))
    core, hook = _hook(100)
    assert _drive(core, hook, xs) == []


def test_a_line_that_stops_gaining_is_cut_after_the_window_once_per_window():
    core, hook = _hook(100)
    xs = [1000.0] + [1000.0 - (i % 7) * 4.0 for i in range(350)]    # wobbles, never -32u
    kills = _drive(core, hook, xs)
    # armed on tick 0; `since` reaches 100 on tick 100, then re-arms (one kill per window)
    assert kills == [100, 200, 300]
    assert hook.state["n"] == 3 and hook.state["killed"] == [0]


def test_the_rule_is_cumulative_not_a_per_tick_step():
    """Training's stall rule needs ONE call that beats the running best by 32u; this one
    only needs the best to have moved 32u since the last re-arm, at any rate."""
    core, hook = _hook(100)
    xs = 1000.0 - 0.5 * np.arange(1000)          # 0.5u per tick: 32u every 64 ticks
    assert _drive(core, hook, xs) == []
    from train_fast import make_eval_stall_hook
    core2 = _FakeCore(1)
    stall = make_eval_stall_hook(core2, _LineField(), 100, 32.0, 1)
    assert _drive(core2, stall, xs) != []        # the mirror of training cuts the same line


def test_an_episode_boundary_rearms_and_indexes_the_kills():
    core, hook = _hook(50)
    xs = [1000.0] * 80 + [2000.0] * 30 + [3000.0] * 80
    kills = _drive(core, hook, xs, ends={79, 109})
    # ep 0 cut at 50; ep 1 (ticks 80..109) only 29 ticks; ep 2 starts at 110, cut at 160
    assert kills == [50, 160]
    assert hook.state["killed"] == [0, 2]


def test_off_field_samples_are_no_progress():
    core, hook = _hook(50)
    xs = [1000.0] + [2e6] * 40 + [990.0] * 20     # off the field, then a 10u gain only
    assert _drive(core, hook, xs) == [50]


def test_a_zero_window_is_off():
    core, hook = _hook(0)
    assert _drive(core, hook, np.full(500, 1000.0)) == []


# ==========================================================================
# (b) the greedy-line pool
# ==========================================================================
def _ep(n, tag, best_at):
    rows = np.zeros(n, STATE_DTYPE)
    rows["origin"][:, 0] = tag * 1e5 + 5000.0
    rows["origin"][best_at, 0] = tag * 1e5           # the episode's best (minimum d)
    rows["tick"] = np.arange(n)
    return rows


def _build(pool, eps, sums):
    hook = pool.hook()
    t = 0
    for ep in eps:
        for i in range(len(ep)):
            hook(t, ep[i:i + 1], np.zeros(1), np.array([i == len(ep) - 1]), np.array([False]))
            t += 1
    return pool.build(sums, 10.0, d_fn=lambda o: np.asarray(o, np.float64)[:, 0])


def test_the_pool_reads_a_cut_episode_as_a_stuck_line_not_a_death():
    eps = [_ep(3000, 1, 1000), _ep(3000, 2, 1000)]
    sums = [{"end": "fail"}, {"end": "fail"}]
    pool = GreedyLinePool(0.3, window=(0.25, 2.0))
    st = _build(pool, eps, as_truncated(sums, [1]))
    # episode 0 (a death): the 2.0..0.25 s before its end AND the 2 s up to its best;
    # episode 1 (cut): only the 2 s up to its best tick (1000)
    o = pool.rows["origin"][:, 0]
    ep1 = np.sort(o[o >= 2e5] - 2e5)
    assert len(ep1) == 201 and ep1.min() == 0.0      # ticks 800..1000, one of them the best
    assert len(o[o < 2e5]) == 201 + 175
    assert st["fail"] == 2 and pool.info["deaths_s"] == [30.0]
    assert as_truncated(sums, []) == sums and sums[1]["end"] == "fail"   # never in place


# ==========================================================================
# (c) the recorder
# ==========================================================================
def test_the_recorder_treats_the_flag_as_training_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    assert "eval_patience" in record_ckpt.TRAIN_ONLY


# ==========================================================================
# (d) the trainer
# ==========================================================================
def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


def _headers(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{") and '"map"' in line:
            out.append(json.loads(line))
    return out


# evals every 2,048 steps, 2 episodes, a 3 s patience on the smoke's 30 s episodes
PAT_SMOKE = ["--record-every", "2048", "--eval-eps", "2"]


@needs_run
def test_the_trainer_cuts_stalled_eval_episodes_and_a_resume_restores_it():
    run = "evpat_on"
    r = _train(run, PAT_SMOKE + ["--eval-patience", "3"], steps="4096")
    out = r.stdout
    assert "--eval-patience: a greedy eval episode ends" in out
    cfg = _cfg(run)
    assert cfg["eval_patience"] == 3.0
    trajs = sorted((ROOT / "runs" / run).glob("traj_*.jsonl"))
    assert trajs
    for p in trajs:
        for h in _headers(p):
            assert h["eval_patience"] == 3.0 and h["eval_stall"] == 0
    # an untrained greedy policy on sidistic stops gaining long before the 30 s cap
    ticks = [json.loads(line)["ticks"] for p in trajs
             for line in p.read_text(encoding="utf-8").splitlines()
             if line.startswith('{"end"')]
    assert ticks and max(ticks) < 3000, ticks
    assert re.search(r"patience-cut \d", out), out[-3000:]
    # a bare resume restores it
    run2 = "evpat_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + PAT_SMOKE
              + ["--steps", "6144"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "eval_patience=3" in r2.stdout
    assert _cfg(run2)["eval_patience"] == 3.0
    for dd in (run, run2):
        shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)


@needs_run
def test_a_negative_window_is_refused():
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "evpat_bad"] + SMOKE
             + ["--steps", "2048", "--eval-patience", "-1"])
    shutil.rmtree(ROOT / "runs" / "evpat_bad", ignore_errors=True)
    assert r.returncode != 0 and "--eval-patience SECS needs SECS >= 0" in r.stdout + r.stderr


# ==========================================================================
# (e) flag off
# ==========================================================================
def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this change (it imports only
    GreedyLinePool from surfgym.greedy_spawn, which this change leaves as it was)."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--eval-patience" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    pre = ROOT / "python" / "_train_fast_pre_patience.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("evpat_off_new", PAT_SMOKE)
        _train("evpat_off_old", PAT_SMOKE, script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "evpat_off_new", ROOT / "runs" / "evpat_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "eval_patience" not in ca
    ra, rb = _csv(a.name), _csv(b.name)
    assert ra and len(ra) == len(rb) and list(ra[0]) == list(rb[0])
    for x, y in zip(ra, rb):
        for k in x:
            if k != "time/fps":
                assert x[k] == y[k], k
    ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
    assert ta and [p.name for p in ta] == [p.name for p in tb]
    for p, q in zip(ta, tb):
        assert p.read_bytes() == q.read_bytes(), p.name
    sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    for dd in (a, b):
        shutil.rmtree(dd, ignore_errors=True)
