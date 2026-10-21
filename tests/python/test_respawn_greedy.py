"""--respawn-greedy F (2026-10-21, autoresearch r3b3): a share of the training spawn pool
drawn from the policy's OWN greedy-eval states in the seconds before each death
(surfgym.greedy_spawn.GreedyLinePool).

(a) The pool, on synthetic episodes:
    * build keeps HI..LO seconds before the last tick of every FAILED episode (and
      nothing of a done / truncated one), clocks zeroed;
    * mix overwrites only the TAIL of the non-start rows (the map-start rows and the
      novel rows after them are kept), with rows of the pool; an empty pool is a no-op;
    * the state survives a checkpoint; bad F / windows are refused; the recorder lists
      the flags as training-only.
(b) The capture hook on the real core: episode i of the capture is exactly the
    trajectory record_rollout writes for episode i (row for row, the trailer's ticks).
(c) The trainer (CPU, sidistic): a resume onto a filled reservoir builds the pool at
    every eval, mixes it into the next pools, saves and restores it, and record_ckpt
    reads the checkpoint; the refusals.
(d) Flag off: the trainer before this change and after it give the same config,
    progress.csv (fps aside), eval trajectory bytes and weights on the same smoke.
"""
from __future__ import annotations

import json
import os
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
from surfgym.greedy_spawn import GreedyLinePool                    # noqa: E402

from test_ez_viewc import (DLL, SID, SID_FIELD, SMOKE, TRAIN, _csv, _run,  # noqa: E402
                           _train, needs_run)


def _ep(n, tag):
    """n distinct states: origin x = tag * 1e5 + tick, tick / stuck_ticks non-zero."""
    rows = np.zeros(n, STATE_DTYPE)
    rows["origin"][:, 0] = tag * 1e5 + np.arange(n)
    rows["tick"] = np.arange(n) + 7
    rows["stuck_ticks"] = 3
    return rows


def _feed(pool, eps):
    """Drive the capture hook like record_rollout: one call per tick, done on the last."""
    hook = pool.hook()
    t = 0
    for ep in eps:
        for i in range(len(ep)):
            last = i == len(ep) - 1
            hook(t, ep[i:i + 1], np.zeros(1), np.array([last]), np.array([False]))
            t += 1


# ==========================================================================
# (a) the pool
# ==========================================================================
def test_build_keeps_hi_to_lo_seconds_before_each_failed_end():
    p = GreedyLinePool(0.2, window=(0.5, 5.0))
    eps = [_ep(900, 1), _ep(300, 2), _ep(40, 3), _ep(700, 4)]
    _feed(p, eps)
    st = p.build([{"end": "fail"}, {"end": "fail"}, {"end": "fail"}, {"end": "done"}],
                 tick_ms=10.0)
    # 900 ticks: [400, 850); 300: [0, 250); 40: inside the last 0.5 s -> nothing; done: nothing
    want = np.concatenate([eps[0][400:850], eps[1][0:250]])
    assert p.size == st["rows"] == 700
    assert np.array_equal(p.rows["origin"], want["origin"])
    assert (p.rows["tick"] == 0).all() and (p.rows["stuck_ticks"] == 0).all()
    assert st["eps"] == 4 and st["fail"] == 2 and st["deaths_s"] == [9.0, 3.0]
    assert "700 own states from 2/4 failed greedy episodes" in p.line()
    # the window is in SECONDS: at a 5 ms tick the same ticks span half the time
    p2 = GreedyLinePool(0.2, window=(0.5, 5.0))
    _feed(p2, [_ep(900, 1)])
    p2.build([{"end": "fail"}], tick_ms=5.0)
    assert np.array_equal(p2.rows["origin"], eps[0][0:800]["origin"])


def test_the_best_progress_anchor_catches_a_stuck_line_and_an_early_loss():
    """With the field: a truncated episode that stopped progressing at tick 1,700 gives
    the 5 s up to that tick; a fail whose best came at tick 500 of 2,000 gives both the
    last 5..0.5 s and the 5 s up to its best (a 0-tick best adds nothing); a fail whose
    best is its death adds only the end window."""
    best_at = {1: 1700, 2: 500, 3: 10**9}

    def d_fn(o):
        tag = np.floor(o[:, 0] / 1e5)
        t = o[:, 0] - tag * 1e5
        tb = np.array([best_at[int(g)] for g in tag], np.float64)
        d = np.abs(t - tb) + 1000.0             # decreasing to the best tick, then rising
        d[tag == 1] = np.where(t[tag == 1] > 1700, 1000.0 + 1.0, d[tag == 1])
        return d
    p = GreedyLinePool(0.2, window=(0.5, 5.0))
    stuck, early, plain = _ep(3000, 1), _ep(2000, 2), _ep(900, 3)
    _feed(p, [stuck, early, plain])
    st = p.build([{"end": "trunc"}, {"end": "fail"}, {"end": "fail"}], tick_ms=10.0,
                 d_fn=d_fn)
    want = np.concatenate([stuck[1200:1701], early[0:501], early[1500:1950], plain[400:850]])
    assert p.size == len(want) == 501 + 501 + 450 + 450
    assert np.array_equal(p.rows["origin"], want["origin"])
    assert st["fail"] == 3 and st["deaths_s"] == [20.0, 9.0] and st["stops_s"] == [17.0, 5.0]
    assert "best-progress stops at 17.0, 5.0 s" in p.line()
    # off the field everywhere (non-finite d): no best anchor, the end anchor stands
    q = GreedyLinePool(0.2, window=(0.5, 5.0))
    _feed(q, [_ep(2000, 2)])
    q.build([{"end": "fail"}], tick_ms=10.0, d_fn=lambda o: np.full(len(o), np.inf))
    assert q.size == 450


def test_done_and_truncated_episodes_add_nothing_and_an_empty_pool_mixes_nothing():
    p = GreedyLinePool(0.3)
    _feed(p, [_ep(900, 1), _ep(900, 2)])
    p.build([{"end": "done"}, {"end": "trunc"}], tick_ms=10.0)
    assert p.size == 0
    pool = np.zeros(4096, STATE_DTYPE)
    assert p.mix(pool, 410) is pool and p.mixed == 0


def test_a_new_eval_replaces_the_pool():
    p = GreedyLinePool(0.3)
    _feed(p, [_ep(900, 1)])
    p.build([{"end": "fail"}], tick_ms=10.0)
    assert p.size == 450
    _feed(p, [_ep(600, 2)])
    p.build([{"end": "fail"}], tick_ms=10.0)
    assert p.size == 450 and (p.rows["origin"][:, 0] >= 2e5).all()
    _feed(p, [_ep(600, 3)])
    p.build([{"end": "done"}], tick_ms=10.0)
    assert p.size == 0


def test_mix_overwrites_only_the_tail_of_the_non_start_rows():
    p = GreedyLinePool(0.15, seed=3)
    _feed(p, [_ep(900, 9)])
    p.build([{"end": "fail"}], tick_ms=10.0)
    pool = np.zeros(4096, STATE_DTYPE)
    pool["origin"][:, 1] = np.arange(4096)          # distinct, and y != 0 unlike the rows
    out = p.mix(pool, 410)
    n = int(round((4096 - 410) * 0.15))
    assert p.mixed == n == 553
    assert out[:4096 - n].tobytes() == pool[:4096 - n].tobytes()
    tail = out[4096 - n:]
    assert (tail["origin"][:, 1] == 0).all()
    assert np.isin(tail["origin"][:, 0], p.rows["origin"][:, 0]).all()
    assert len(np.unique(tail["origin"][:, 0])) > 200   # uniform draws, not one row
    assert pool["origin"][:, 1].tolist() == list(range(4096))   # the input is not touched


def test_the_pool_survives_a_checkpoint():
    p = GreedyLinePool(0.2)
    _feed(p, [_ep(900, 1), _ep(800, 2)])
    p.build([{"end": "fail"}, {"end": "fail"}], tick_ms=10.0)
    q = GreedyLinePool(0.2)
    assert q.load_state_dict(p.state_dict())
    assert q.rows.tobytes() == p.rows.tobytes() and q.info == p.info
    assert not q.load_state_dict({"rows": np.zeros(3, np.float32)})
    assert not q.load_state_dict(None)


@pytest.mark.parametrize("frac,window", [(0.0, (0.5, 5.0)), (1.0, (0.5, 5.0)),
                                         (-0.1, (0.5, 5.0)), (0.2, (5.0, 0.5)),
                                         (0.2, (-1.0, 2.0))])
def test_bad_fractions_and_windows_are_refused(frac, window):
    with pytest.raises(ValueError):
        GreedyLinePool(frac, window=window)


def test_the_recorder_treats_the_flags_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"respawn_greedy"' in src and '"respawn_greedy_window"' in src


# ==========================================================================
# (b) the capture hook against record_rollout on the real core
# ==========================================================================
@pytest.mark.skipif(not (SID.exists() and DLL.exists()), reason="needs the core + sidistic")
def test_the_capture_is_the_recorded_trajectory_row_for_row(tmp_path):
    from surfgym import SurfCore, default_config
    from surfgym.record import record_rollout

    core = SurfCore(str(SID), default_config(num_envs=1, max_episode_ticks=150,
                                             sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))

    class _Pol:
        def __init__(self):
            self.rng = np.random.default_rng(0)

        def act(self, obs):
            a = np.zeros((obs.shape[0], 6), np.int32)
            a[:, 0] = self.rng.integers(0, 15, obs.shape[0])
            a[:, 2] = 2
            return a

    p = GreedyLinePool(0.2, window=(0.0, 1.0))
    path = tmp_path / "t.jsonl"
    summ = record_rollout(core, _Pol(), path, episodes=2, max_ticks=500, seed=3,
                          on_tick=p.hook())
    eps = list(p._eps)
    assert len(eps) == len(summ) == 2
    rows, cur = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("{"):
            d = json.loads(line)
            if "end" in d:
                rows.append(np.asarray(cur))
                cur = []
            continue
        cur.append(json.loads(line))
    for ep, tr, js in zip(eps, summ, rows):
        assert len(ep) == tr["ticks"] == len(js)
        np.testing.assert_allclose(ep["origin"], js[:, 1:4], atol=0.006)
        np.testing.assert_allclose(ep["velocity"], js[:, 4:7], atol=0.006)


# ==========================================================================
# (c) the trainer
# ==========================================================================
def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


def _reachable_rows(core, field, n, seed=0):
    """``n`` states at reachable, hull-clear points of the field (test_respawn_novel's
    helper; that module cannot be imported where /root is unreadable)."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = field.grid.shape
    lim = field.reach_max - 0.5 * field.cell
    out = []
    while len(out) < n:
        ix, iy, iz = (rng.integers(0, nx, 20000), rng.integers(0, ny, 20000),
                      rng.integers(0, nz, 20000))
        ok = field.grid[iz, iy, ix] < lim
        for q in np.asarray(field.mins) + (np.stack([ix, iy, iz], 1)[ok] + 0.5) * field.cell:
            if not core.trace(q, q, hull=0).startsolid:
                out.append(q)
            if len(out) >= n:
                break
    rows = np.zeros(n, STATE_DTYPE)
    rows["origin"] = np.asarray(out, np.float32)
    rows["velocity"] = rng.normal(0.0, 900.0, (n, 3)).astype(np.float32)
    rows["yaw"] = rng.uniform(-180.0, 180.0, n).astype(np.float32)
    rows["pitch"] = rng.uniform(-30.0, 30.0, n).astype(np.float32)
    rows["onground"] = 3
    return rows


def _seeded_ckpt(run0):
    """A smoke checkpoint whose reservoir already holds 3,000 reachable states, so the
    trainer's reservoir branch (>= 2,000 states) builds the pools from iteration 1."""
    import torch
    from surfgym import SurfCore, default_config
    from surfgym.goalfield import GoalField
    _train(run0, [], steps="2048")
    d0 = ROOT / "runs" / run0
    ck = torch.load(d0 / "ckpt_final.pt", map_location="cpu", weights_only=False)
    z = np.load(SID_FIELD, allow_pickle=False)
    fld = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                    float(z["cell"]), float(z["reach_max"]))
    sc = SurfCore(str(SID), default_config(num_envs=1, spawn_mode=2, max_episode_ticks=200,
                                           sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))
    ck["respawn"] = {"states": _reachable_rows(sc, fld, 3000, seed=1),
                     "map_id": ck["respawn"]["map_id"]}
    seeded = d0 / "ckpt_seeded.pt"
    torch.save(ck, seeded)
    return d0, seeded


# the evals stall-kill (--eval-stall 1 --stall-secs 6) so the smoke's untrained greedy
# policy ends its eval episodes inside a 30 s cap, and evals run every 4,096 steps. A stall
# kill is a STOP to the pool (its best-progress window, test_greedy_stall.py), so the window
# is longer than the default 5 s HI: the stop anchor needs the best > HI s before the end
GREEDY_SMOKE = ["--respawn-reservoir", "5000", "--record-every", "4096", "--eval-eps", "2",
                "--eval-stall", "1", "--stall-secs", "6"]


@needs_run
def test_a_resume_builds_the_pool_at_every_eval_mixes_it_saves_and_records():
    import torch
    d0, seeded = _seeded_ckpt("greedy_base")
    run = "greedy_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + ["--steps", "18432", "--respawn-greedy", "0.25"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    out = r.stdout
    assert "respawn GREEDY: 25% of the non-start pool" in out
    built = re.findall(r"greedy-line pool: ([0-9,]+) own states from (\d+)/(\d+) failed", out)
    assert len(built) >= 3, out[-3000:]
    assert any(int(n.replace(",", "")) > 0 for n, _, _ in built), built
    mixed = [int(m) for m in re.findall(r"  greedy [0-9,]+x(\d+)", out)]
    assert mixed and max(mixed) == int(round((4096 - 410) * 0.25)), mixed
    cfg = _cfg(run)
    assert cfg["respawn_greedy"] == 0.25 and cfg["respawn_greedy_window"] == [0.5, 5.0]
    ck = torch.load(ROOT / "runs" / run / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    assert ck["respawn_greedy"]["rows"].dtype == STATE_DTYPE
    # a bare resume restores the flag and the pool
    run2 = "greedy_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "22528"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_greedy=0.25" in r2.stdout
    assert "restored --respawn-greedy state" in r2.stdout
    assert _cfg(run2)["respawn_greedy"] == 0.25
    # the recorder reads the checkpoint (the record gate's own backend)
    rec = ROOT / "runs" / run / "rec_greedy.jsonl"
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(rec),
               "--episodes", "1", "--ep-ticks", "400"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--respawn-greedy", "0.2", "--respawn-frac", "0"], "--respawn-greedy needs the reservoir"),
    (["--respawn-greedy", "1.5"], "--respawn-greedy F needs 0 < F < 1"),
    (["--respawn-greedy", "0.2", "--respawn-greedy-window", "3", "1"],
     "--respawn-greedy-window LO HI needs 0 <= LO < HI"),
])
def test_bad_settings_are_refused(extra, msg):
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "greedy_bad"] + SMOKE
             + ["--steps", "2048"] + extra)
    shutil.rmtree(ROOT / "runs" / "greedy_bad", ignore_errors=True)
    assert r.returncode != 0 and msg in r.stdout + r.stderr, r.stdout[-2000:]


# ==========================================================================
# (d) flag off
# ==========================================================================
def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this change."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--respawn-greedy" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    pre = ROOT / "python" / "_train_fast_pre_greedy.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("greedy_off_new", [])
        _train("greedy_off_old", [], script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "greedy_off_new", ROOT / "runs" / "greedy_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "respawn_greedy" not in ca
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
    assert "respawn_greedy" not in sa
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    for dd in (a, b):
        shutil.rmtree(dd, ignore_errors=True)
