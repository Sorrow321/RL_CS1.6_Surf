"""--tick-ms-schedule: the physics tick as a live RAMP (surfgym.tick,
train_fast's per-second conversions, the checkpoint's ramp state).

Why the flag exists: a policy has MEMORISED its line against the tick it
trained at. The frozen finisher xQR32 finishes 9/9 at 10 ms and 0/9 at
7.63 ms - 30% more air-accelerate impulses per second of held strafe puts
it somewhere else at every ramp - so a warm resume AT the target tick
starts from a non-finisher and has nothing left to improve. A gradual
change is the alternative, and this file pins what "gradual" means:

1. The SCHEDULE itself: linear in MS from FROM to TO over STEPS
   environment steps counted from the ramp's origin, then HOLDING TO
   exactly; the realised tick is always an integer-ms pattern, re-derived
   only when the request has moved more than 0.05 ms.
2. A tiny CPU scratch run with a short ramp actually runs, logs one line
   per pattern change, drives the cores through those patterns, and ends
   at 7.6667 ms in run.json with a tick/tick_ms column in progress.csv.
3. WITHOUT the flag the run is BIT-IDENTICAL to the code before the flag
   existed: the same config dump, the same eval trajectory (rows are
   position/velocity states) and the same per-iteration training numbers.
   The reference is the parent of the commit that introduced the flag, so
   this cannot rot into comparing the change against itself.
4. A checkpoint carries the ramp WITH ITS ORIGIN, so a bare resume
   continues it and --tick-ms-schedule on the resume replaces it.

    python -m pytest tests/python/test_tick_schedule.py -q

CPU only (CUDA_VISIBLE_DEVICES=-1 is set for every child), tiny nets, a
few thousand steps per run; the whole file is ~minutes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.tick import (PATTERN_TOL_MS, TickClock, TickSchedule,  # noqa: E402
                          tick_pattern)

CANNONBALL = ROOT / "maps" / "surf_src_cannonball.bsp"
GOALFIELD = ROOT / "maps" / "surf_src_cannonball.goal_32.npz"
DLL = ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so")

needs_run = pytest.mark.skipif(
    not (CANNONBALL.exists() and DLL.exists() and GOALFIELD.exists()),
    reason="needs the built core + cannonball + its prebaked goal field")


# ==========================================================================
# 1. the schedule function
# ==========================================================================
def test_parse_and_the_linear_ramp():
    s = TickSchedule.parse("10:7.63:600e6")
    assert (s.from_ms, s.to_ms, s.steps, s.origin) == (10.0, 7.63, 600_000_000, 0)
    assert s.end_step == 600_000_000
    # the endpoints are EXACT, not interpolated (10 + (7.63-10)*1.0 is
    # 7.630000000000001 in IEEE and would re-derive a pattern forever)
    assert s.ms_at(0) == 10.0
    assert s.ms_at(600_000_000) == 7.63
    assert s.ms_at(9e8) == 7.63           # holds TO past the end
    assert s.ms_at(-1e9) == 10.0          # and FROM before the origin
    assert s.ms_at(300_000_000) == pytest.approx(8.815, abs=1e-12)
    assert s.ms_at(150_000_000) == pytest.approx(9.4075, abs=1e-12)
    # linear in MS, so NOT linear in Hz - the point of ramping the tick
    assert 1000.0 / s.ms_at(300e6) == pytest.approx(113.44, abs=0.01)


def test_the_origin_is_where_the_run_starts():
    s = TickSchedule.parse("10:7.63:1000", origin=5000)
    assert s.ms_at(4999) == 10.0 and s.ms_at(5000) == 10.0
    assert s.ms_at(5500) == pytest.approx(8.815)
    assert s.ms_at(6000) == 7.63 and s.ms_at(60_000) == 7.63
    d = s.to_dict()
    assert d == {"from_ms": 10.0, "to_ms": 7.63, "steps": 1000,
                 "origin_step": 5000}
    back = TickSchedule.from_dict(d)
    assert (back.from_ms, back.to_ms, back.steps, back.origin) == \
           (10.0, 7.63, 1000, 5000)
    assert back.spec() == "10:7.63:1000"


def test_bad_specs_are_refused():
    for bad in ("10:7.63", "10:7.63:600e6:2", "a:b:c", "10-7.63-600"):
        with pytest.raises(ValueError):
            TickSchedule.parse(bad)
    with pytest.raises(ValueError):        # STEPS must be positive
        TickSchedule.parse("10:7.63:0")
    with pytest.raises(ValueError):        # 0.5 ms is not a realisable tick
        TickSchedule.parse("10:0.5:100")
    with pytest.raises(ValueError):
        TickSchedule.parse("60:10:100")


def test_pattern_re_derivation_points():
    """The realised tick must be an integer-ms pattern, and the trainer
    re-derives one only when the REQUEST has moved more than 0.05 ms."""
    s = TickSchedule.parse("10:7.63:600e6")
    ch = s.pattern_changes()
    steps = [c[0] for c in ch]
    mss = [c[1] for c in ch]
    pats = [c[2] for c in ch]
    assert steps == sorted(steps) and steps[0] == 0
    assert ch[0][1] == 10.0 and ch[0][2] == [10]
    assert ch[-1][1] == 7.63 and ch[-1][2] == [8, 8, 7]
    assert len(ch) == 39                     # a 2.37 ms span at 0.05 ms
    # every consecutive pair really is more than the tolerance apart, and
    # every realised pattern is within the tolerance of its request
    for a, b in zip(mss, mss[1:]):
        assert abs(b - a) > PATTERN_TOL_MS
    for ms, pat in zip(mss, pats):
        assert pat == tick_pattern(ms)
        assert all(1 <= v <= 50 and int(v) == v for v in pat)
        assert abs(sum(pat) / len(pat) - ms) <= PATTERN_TOL_MS
    # monotone down in ms, and the whole span is covered
    assert mss == sorted(mss, reverse=True)
    assert 10.0 - mss[-1] == pytest.approx(2.37, abs=1e-9)


def test_retune_moves_a_clock_in_place():
    """The trainer mutates ONE clock: printers, closures and the config
    dump hold the object, so a rebind would strand them at the old tick."""
    c = TickClock(10.0)
    assert c.is_reference and c.pattern == [10] and c.ms == 10.0
    assert c.gamma(0.9995) == 0.9995 and c.per_tick(0.005) == 0.005
    assert c.retune(9.5) is True                     # the pattern changed
    assert c.pattern == [10, 9] and c.ms == 9.5 and not c.is_reference
    assert c.retune(9.49) is False                   # same pattern, no churn
    assert c.pattern == [10, 9] and c.requested_ms == 9.49
    assert c.retune(7.63) is True
    assert c.pattern == [8, 8, 7] and c.ms == pytest.approx(23.0 / 3.0)
    # the conversions follow the retune, in SECONDS
    assert c.secs_to_ticks(15.0) == 1956
    assert c.gamma(0.9995) == pytest.approx(0.9995 ** (c.ms / 10.0))
    assert c.per_tick(0.005) == pytest.approx(0.005 * c.ms / 10.0)
    # and back to the reference is bit-for-bit the legacy arithmetic again
    assert c.retune(10.0) is True
    assert c.is_reference and c.secs_to_ticks(15.0) == int(15.0 * 100.0)
    assert c.gamma(0.9995) == 0.9995


def test_clock_at_matches_ms_at():
    s = TickSchedule.parse("10:7.63:1000")
    for step in (0, 250, 500, 750, 1000, 5000):
        assert s.clock_at(step).requested_ms == s.ms_at(step)
        assert s.clock_at(step).pattern == tick_pattern(s.ms_at(step))


# ==========================================================================
# the trainer: one launcher for every child run
# ==========================================================================
SMOKE = ["--map", str(CANNONBALL), "--reward", "race", "--envs", "64",
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
         "--eval-eps", "1", "--eval-greedy-only"]


def _env():
    # CLAUDE.md: no GPU. -1 rather than "" because Windows deletes an empty
    # env var and the child would then see every card.
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1",
                PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="8",
                NUMBA_NUM_THREADS="8", SURFCORE_DLL=str(DLL))


def _train(root, run, extra, timeout=1800, drop=()):
    """Run <root>/python/train_fast.py; artifacts land in <root>/runs/<run>.

    `drop` removes a flag AND its value from the shared set - the episode
    cap has to come out to test the schedule's own sizing of it."""
    shutil.rmtree(Path(root) / "runs" / run, ignore_errors=True)
    flags = list(SMOKE)
    for d in drop:
        i = flags.index(d)
        del flags[i:i + 2]
    cmd = [sys.executable, "-u", str(Path(root) / "python" / "train_fast.py"),
           "--run", run] + flags + extra
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                          cwd=str(root), timeout=timeout)


# ==========================================================================
# 2. a tiny CPU scratch run with a short ramp
# ==========================================================================
@needs_run
def test_short_ramp_runs_logs_every_pattern_change_and_ends_at_7_6667():
    # 8 iterations of 8*64*4 = 2,048 steps: the ramp spans the first 8 and
    # the last 2 hold, so both halves of the schedule are exercised
    # --ep-secs, not --ep-ticks: the cap is a DURATION, which is what the
    # schedule has to keep across a ramp it cannot follow
    r = _train(ROOT, "tsched_ramp",
               ["--tick-ms-schedule", "10:7.63:16384", "--steps", "20480",
                "--ep-secs", "30", "--record-every", "1e12",
                "--no-eval-at-start"], drop=("--ep-ticks",))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    out = r.stdout
    assert "tick: tick 10 ms (100 Hz, the reference" in out   # STARTS at FROM
    assert "tick ramps 10 -> 7.63 ms (100.0 -> 130.4 Hz)" in out
    assert "39 pattern re-derivations over the ramp" in out
    # the three quantities that cannot follow the ramp are announced, and
    # the one frozen tick COUNT is sized at the ramp's SHORTEST tick so the
    # cap keeps its duration (30 s = 3000 ticks at 10 ms -> 3913 at 7.667)
    assert "FROZEN, no C setter" in out
    assert ("episode cap 3000 ticks = 30 s at the launch tick -> 3913 ticks"
            in out)
    assert "39.1 s now, 30.0 s at the end of the ramp" in out

    lines = [ln for ln in out.splitlines()
             if ln.startswith("tick schedule @ ")]
    assert len(lines) == 8, lines            # one per re-derivation, once
    assert "requested 9.7037 ms -> pattern [10,10,9] ms = 9.6667 ms" in lines[0]
    assert "requested 7.6300 ms -> pattern [8,8,7] ms = 7.6667 ms" in lines[-1]
    assert "(130.43 Hz)" in lines[-1]
    # every per-second constant moved WITH the tick, in the same line
    assert "gamma 0.99961664/tick" in lines[-1]          # 0.9995**(23/30)
    assert "time_pen 0.00383333/tick" in lines[-1]       # 0.005*23/30
    assert "stall_eps 24.53 u/call" in lines[-1]         # 32*23/30
    assert "stall 1956 ticks" in lines[-1]               # 15 s
    assert "respawn margin 1304 ticks" in lines[-1]      # 10 s
    assert "decision 30.7 ms" in lines[-1]               # act_every 4
    assert "episode cap 30.0 s" in lines[-1]            # the frozen count
    # ... and every step of the ramp is a real integer-ms pattern
    steps = [int(ln.split("@ ")[1].split(":")[0].replace(",", ""))
             for ln in lines]
    assert steps == [2048 * i for i in range(1, 9)]

    d = ROOT / "runs" / "tsched_ramp"
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    c = meta["config"]
    # the config dump is LIVE: a checkpoint states the tick its weights saw
    assert c["tick_ms"] == 7.63
    assert c["tick_ms_eff"] == pytest.approx(23.0 / 3.0)
    assert c["tick_pattern_ms"] == [8, 8, 7]
    assert c["gamma_tick"] == pytest.approx(0.9995 ** (23.0 / 30.0))
    assert c["time_pen_tick"] == pytest.approx(0.005 * 23.0 / 30.0)
    assert c["stall_eps_tick"] == pytest.approx(32.0 * 23.0 / 30.0)
    # FROZEN as one tick count, sized at the ramp's shortest tick, so the
    # cap is exactly the 30 s it stood for once the ramp lands
    assert c["ep_ticks"] == 3913
    assert c["ep_secs"] == pytest.approx(30.0, abs=0.01)
    assert c["tick_schedule"] == {"from_ms": 10.0, "to_ms": 7.63,
                                  "steps": 16384, "origin_step": 0}
    assert meta["tick_ms_final"] == 7.63
    assert meta["tick_ms_eff_final"] == pytest.approx(23.0 / 3.0)
    assert meta["tick_pattern_ms_final"] == [8, 8, 7]
    assert [ch[2] for ch in meta["tick_changes"]][0] == [10, 10, 9]
    assert [ch[2] for ch in meta["tick_changes"]][-1] == [8, 8, 7]
    assert len(meta["tick_changes"]) == 8

    # progress.csv carries the realised tick per iteration. By NAME: the
    # column is appended, and a later change may append after it.
    rows = (d / "progress.csv").read_text(encoding="utf-8").splitlines()
    head = rows[0].split(",")
    assert "tick/tick_ms" in head
    _ti = head.index("tick/tick_ms")
    ticks = [float(r.split(",")[_ti]) for r in rows[1:]]
    assert len(ticks) == 10
    assert ticks[0] == 10.0                       # iteration 1 ran at FROM
    assert ticks[-1] == pytest.approx(23.0 / 3.0, abs=1e-5)
    assert ticks == sorted(ticks, reverse=True)   # monotone down
    shutil.rmtree(d, ignore_errors=True)


# ==========================================================================
# 3. no flag == the code before the flag existed
# ==========================================================================
# Config-dump keys added by OTHER opt-in features that landed on the
# integration branch after the reference commit, with the value each takes
# when its flag is off. The pre-flag tree cannot know them, so they are the
# one permitted difference; every other key still has to match exactly.
# (--mask-forward-air / --jump-cooldown / --duck-air-mask write NO key when
# off, so the air-key merge contributes nothing here - only its four
# appended CSV columns, which _same_shared_csv and the strict-prefix
# assertion below already tolerate.)
INERT_SINCE = {"priv_critic": 0, "priv_features": None, "priv_hidden": None}


def _baseline_tree(tmp_path):
    """<tmp>/base holding python/ as it was BEFORE --tick-ms-schedule.

    The reference is the parent of the commit that first mentions the flag
    in train_fast.py, so this compares against real unpatched code rather
    than against a copy of the change. TICK_SCHED_BASE_REF overrides.
    """
    ref = os.environ.get("TICK_SCHED_BASE_REF")
    if not ref:
        r = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H",
                            "-S", "tick-ms-schedule", "--",
                            "python/train_fast.py"],
                           capture_output=True, text=True)
        shas = r.stdout.split()
        if r.returncode != 0 or not shas:
            pytest.skip("cannot resolve the pre-flag commit (not committed "
                        "yet, or no git); set TICK_SCHED_BASE_REF")
        ref = shas[-1] + "^"
    base = tmp_path / "base"
    base.mkdir(parents=True, exist_ok=True)
    tar = tmp_path / "base.tar"
    r = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar",
                        "-o", str(tar), ref, "python"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git archive {ref} failed: {r.stderr[-400:]}")
    with tarfile.open(tar) as tf:
        tf.extractall(base)
    src = (base / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert "--tick-ms-schedule" not in src, \
        "the resolved reference already has the flag - wrong commit"
    return base


# time/fps is wall-clock and never reproducible; tick/tick_ms is the column
# this change adds. Everything else in a row is a function of the rollout.
def _shared_csv(path, drop=("tick/tick_ms", "time/fps")):
    """progress.csv as a list of {column name: value} rows, minus `drop`.

    Keyed by NAME, not by position: the file's rule is that a new column is
    APPENDED and an older header stays a strict prefix, so a comparison
    against pre-flag code has to survive the other side simply not having a
    column at all. _same_shared_csv below compares the intersection."""
    rows = path.read_text(encoding="utf-8").splitlines()
    head = rows[0].split(",")
    keep = [i for i, h in enumerate(head) if h not in drop]
    return [{head[i]: r.split(",")[i] for i in keep} for r in rows[1:]]


def _same_shared_csv(a, b, drop=("tick/tick_ms", "time/fps")):
    """Every column the two files SHARE, row for row. A column only one side
    writes is a later append and is not part of what this test claims."""
    ra, rb = _shared_csv(a, drop), _shared_csv(b, drop)
    assert ra and len(ra) == len(rb), (len(ra), len(rb))
    for i, (x, y) in enumerate(zip(ra, rb)):
        shared = sorted(set(x) & set(y))
        assert len(shared) > 10, shared
        assert [x[k] for k in shared] == [y[k] for k in shared], \
            (i, [k for k in shared if x[k] != y[k]])
    return True


@needs_run
def test_no_flag_is_bit_identical_to_the_pre_flag_code(tmp_path):
    base = _baseline_tree(tmp_path)
    # eval at start + a record, so the comparison covers a real rollout:
    # every traj row is a position / velocity / view state per physics tick
    extra = ["--steps", "6144", "--record-every", "1e12"]
    r_old = _train(base, "tsched_ctl", extra)
    assert r_old.returncode == 0, r_old.stdout[-4000:] + r_old.stderr[-4000:]
    r_new = _train(ROOT, "tsched_ctl", extra)
    assert r_new.returncode == 0, r_new.stdout[-4000:] + r_new.stderr[-4000:]
    d_old, d_new = base / "runs" / "tsched_ctl", ROOT / "runs" / "tsched_ctl"

    # (a) the config dump: byte-identical, and the schedule key is ABSENT
    c_old = json.loads((d_old / "run.json").read_text(encoding="utf-8"))["config"]
    c_new = json.loads((d_new / "run.json").read_text(encoding="utf-8"))["config"]
    assert "tick_schedule" not in c_new
    # The reference predates this flag, so on the integration branch it also
    # predates every OTHER opt-in feature merged since. Those add config keys
    # of their own, always at an off value; the claim here is unchanged - the
    # tick schedule adds nothing and moves nothing - so it is stated as
    # "every key the two SHARE is identical, and the extras are exactly the
    # known inert ones". Anything else in the delta still fails.
    _added = set(c_new) - set(c_old)
    assert not set(c_old) - set(c_new), set(c_old) - set(c_new)
    assert _added <= set(INERT_SINCE), _added
    for _k in _added:
        assert c_new[_k] == INERT_SINCE[_k], (_k, c_new[_k])
    assert {k: v for k, v in c_new.items() if k not in _added} == c_old
    assert c_new["tick_ms"] == 10.0 and c_new["tick_pattern_ms"] == [10]

    # (b) the eval rollout: the whole trajectory file, byte for byte
    t_old = sorted(d_old.glob("traj_*.jsonl"))
    t_new = sorted(d_new.glob("traj_*.jsonl"))
    assert t_old and [p.name for p in t_old] == [p.name for p in t_new]
    for a, b in zip(t_old, t_new):
        ra = a.read_text(encoding="utf-8").splitlines()
        rb = b.read_text(encoding="utf-8").splitlines()
        assert len(ra) > 200, f"{a.name}: too few rows to mean anything"
        assert ra == rb, f"{a.name} differs"

    # (c) the TRAINING rollout, through every per-iteration number the
    #     trainer derives from it (reward, length, losses, kl, explained
    #     variance): identical rows, minus the one added column
    _same_shared_csv(d_new / "progress.csv", d_old / "progress.csv")
    _h_new = (d_new / "progress.csv").read_text(
        encoding="utf-8").splitlines()[0].split(",")
    _h_old = (d_old / "progress.csv").read_text(
        encoding="utf-8").splitlines()[0].split(",")
    # the added column, and the strict-prefix rule that makes a resumed
    # pre-flag progress.csv migrate rather than break
    assert "tick/tick_ms" in _h_new and "tick/tick_ms" not in _h_old
    assert _h_new[:len(_h_old)] == _h_old
    shutil.rmtree(d_new, ignore_errors=True)


@needs_run
def test_a_flat_schedule_is_the_unscheduled_run(tmp_path):
    """A ramp from 10 to 10 must change nothing but the bookkeeping - the
    machinery itself has to be inert at the reference tick."""
    extra = ["--steps", "6144", "--record-every", "1e12"]
    r0 = _train(ROOT, "tsched_flat0", extra)
    assert r0.returncode == 0, r0.stdout[-4000:] + r0.stderr[-4000:]
    r1 = _train(ROOT, "tsched_flat1",
                extra + ["--tick-ms-schedule", "10:10:4096"])
    assert r1.returncode == 0, r1.stdout[-4000:] + r1.stderr[-4000:]
    assert "tick schedule @ " not in r1.stdout      # nothing to re-derive
    d0, d1 = ROOT / "runs" / "tsched_flat0", ROOT / "runs" / "tsched_flat1"
    c0 = json.loads((d0 / "run.json").read_text(encoding="utf-8"))["config"]
    c1 = json.loads((d1 / "run.json").read_text(encoding="utf-8"))["config"]
    assert c1.pop("tick_schedule") == {"from_ms": 10.0, "to_ms": 10.0,
                                       "steps": 4096, "origin_step": 0}
    assert c1 == c0
    for a, b in zip(sorted(d0.glob("traj_*.jsonl")),
                    sorted(d1.glob("traj_*.jsonl"))):
        assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    _same_shared_csv(d1 / "progress.csv", d0 / "progress.csv")
    for d in (d0, d1):
        shutil.rmtree(d, ignore_errors=True)


@needs_run
def test_an_explicit_ep_ticks_still_names_a_tick_count():
    """The re-sizing is for a DURATION - the default, --ep-secs, or a cap
    carried over from a checkpoint. A caller who names a tick count gets
    exactly that count, at every point of the ramp."""
    r = _train(ROOT, "tsched_epfix",
               ["--tick-ms-schedule", "10:7.63:16384", "--steps", "4096",
                "--record-every", "1e12", "--no-eval-at-start"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    # SMOKE names --ep-ticks 3000, so nothing is re-sized
    assert "at the launch tick ->" not in r.stdout
    c = json.loads((ROOT / "runs" / "tsched_epfix" / "run.json")
                   .read_text(encoding="utf-8"))["config"]
    assert c["ep_ticks"] == 3000
    shutil.rmtree(ROOT / "runs" / "tsched_epfix", ignore_errors=True)


def test_tick_ms_and_a_schedule_together_are_refused():
    r = subprocess.run(
        [sys.executable, str(ROOT / "python" / "train_fast.py"),
         "--run", "tsched_conflict", "--tick-ms", "9",
         "--tick-ms-schedule", "10:7.63:1000", "--steps", "1"],
        capture_output=True, text=True, env=_env(), cwd=str(ROOT))
    assert r.returncode != 0
    assert "both set the physics tick" in (r.stdout + r.stderr)


# ==========================================================================
# 4. a checkpoint carries the ramp
# ==========================================================================
@needs_run
def test_a_resume_continues_the_ramp_and_the_flag_replaces_it():
    """The ramp is RUN STATE. The tick is re-derived at the TOP of an
    iteration, so the tick a run ends on is the one its LAST iteration ran
    at - 6,144 steps is 3 iterations of 2,048, the last starting at 4,096,
    so the ramp's value there (10 - 2.37*4096/40960 = 9.763) is what the
    checkpoint states."""
    r = _train(ROOT, "tsched_res_a",
               ["--tick-ms-schedule", "10:7.63:40960", "--steps", "6144",
                "--record-every", "1e12", "--no-eval-at-start"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    ck = ROOT / "runs" / "tsched_res_a" / "ckpt_final.pt"
    assert ck.exists()
    cfg = json.loads((ROOT / "runs" / "tsched_res_a" / "run.json")
                     .read_text(encoding="utf-8"))["config"]
    assert cfg["tick_schedule"] == {"from_ms": 10.0, "to_ms": 7.63,
                                    "steps": 40960, "origin_step": 0}
    assert cfg["tick_ms"] == pytest.approx(9.763, abs=1e-3)

    # (a) a BARE resume continues the same ramp from the same origin
    r = _train(ROOT, "tsched_res_b",
               ["--ckpt", str(ck), "--steps", "10240",
                "--record-every", "1e12", "--no-eval-at-start"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "tick_schedule=10:7.63:40960@0" in r.stdout
    assert "resumed" in r.stdout
    # it starts at the CHECKPOINT's own tick and the ramp moves it on at the
    # first iteration, so a bare resume must not fire a TICK TRANSFER
    assert "TICK TRANSFER" not in r.stdout
    assert "tick 9.763 ms requested" in r.stdout
    m = json.loads((ROOT / "runs" / "tsched_res_b" / "run.json")
                   .read_text(encoding="utf-8"))
    assert m["config"]["tick_schedule"]["origin_step"] == 0
    # iterations start at 6,144 and 8,192 -> 9.6444 then 9.5259
    assert [c[0] for c in m["tick_changes"]] == [6144, 8192]
    assert m["tick_changes"][0][1] == pytest.approx(9.6444, abs=1e-3)
    assert m["tick_ms_final"] == pytest.approx(9.5259, abs=1e-3)
    assert m["config"]["tick_pattern_ms"] == tick_pattern(m["tick_ms_final"])

    # (b) the flag on a resume REPLACES the ramp and re-origins at this step
    r = _train(ROOT, "tsched_res_c",
               ["--ckpt", str(ck), "--steps", "12288",
                "--tick-ms-schedule", "10:9:2048",
                "--record-every", "1e12", "--no-eval-at-start"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    m = json.loads((ROOT / "runs" / "tsched_res_c" / "run.json")
                   .read_text(encoding="utf-8"))
    assert m["config"]["tick_schedule"] == {"from_ms": 10.0, "to_ms": 9.0,
                                            "steps": 2048,
                                            "origin_step": 6144}
    # the new ramp STARTS at 10 ms, so this IS a tick transfer off a
    # 9.763 ms checkpoint and the trainer has to say so
    assert "TICK TRANSFER" in r.stdout
    assert m["tick_ms_final"] == 9.0
    assert m["config"]["tick_pattern_ms"] == [9]
    assert m["config"]["tick_ms_ckpt"] == pytest.approx(9.763, abs=1e-3)
    for n in ("tsched_res_a", "tsched_res_b", "tsched_res_c"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
