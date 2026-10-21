"""--stall-cum: the race stall timer re-arms on a CUMULATIVE improvement of the episode's
running best (more than stall_eps since the last re-arm), at any call cadence.

Why: the shipped rule re-arms only when ONE call improves the running best by > stall_eps.
Called every tick (every=1, the default without --reward-per-decision) a call moves <= ~23 u at
full speed, so on the trilinear field only near-wall corner switches ever clear 32 u; replayed
on the incumbent's best eval it "stalls" every episode flying the greedy line at ~42.5 s
whatever its progress (NOTES, round 4).

(a) RaceReward: creeping progress (10 u per call) is killed by the shipped rule and NOT by
    --stall-cum; standing still is killed by both; an ended row re-anchors on its new spawn;
(b) the eval hook mirrors it (make_eval_stall_hook(cum=True));
(c) the trainer (CPU smoke): the config carries stall_cum, a bare resume restores it, the
    eval header says so, and record_ckpt mirrors it on an --eval-stall recording.
Flags off: covered by test_pitch_std_min.py's bit-identity test.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/python/test_stall_cum.py -p no:cacheprovider
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.core import STATE_DTYPE            # noqa: E402
from surfgym.rewards import RaceReward          # noqa: E402

from test_ez_viewc import SMOKE, TRAIN, _run, _train, needs_run  # noqa: E402

STALL = 20          # calls


class _LineField:
    """d = 10,000 - x: moving +x by s improves d by s."""

    def sample(self, pos):
        return 10000.0 - np.atleast_2d(np.asarray(pos, np.float64))[:, 0]


class _FakeCore:
    def __init__(self, n=1):
        from types import SimpleNamespace
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self.config = SimpleNamespace(phys=SimpleNamespace(sv_gravity=800.0))
        self.fails = np.zeros(n, int)

    def at(self, x):
        self.states_view["origin"][:, 0] = np.asarray(x, np.float32)
        return self

    def force_fail(self, m):
        self.fails += np.asarray(m, int)


def _rr(core, cum):
    rr = RaceReward(_LineField(), scale=1.0, time_pen=0.0, stall_ticks=STALL, stall_eps=32.0,
                    stall_cum=cum)
    rr.on_reset(core)
    return rr


def _step(rr, core, done=0):
    n = core.num_envs
    d = np.full(n, done, np.uint8)
    return rr(None, None, None, np.zeros(n, np.float32), d, np.zeros(n, np.uint8), core)


def _kills(cum, per_call, calls):
    core = _FakeCore().at(0.0)
    rr = _rr(core, cum)
    kills = 0
    for i in range(calls):
        core.at((i + 1) * per_call)
        _step(rr, core)
        if rr.pop_stall_mask() is not None:
            kills += 1
    return kills


def test_creeping_progress_is_killed_by_the_shipped_rule_and_not_by_stall_cum():
    assert _kills(False, 10.0, 5 * STALL) == 5          # 10 u per call never clears 32 u
    assert _kills(True, 10.0, 5 * STALL) == 0           # 40 u in 4 calls does


def test_standing_still_is_killed_by_both():
    assert _kills(False, 0.0, 5 * STALL) == 5
    assert _kills(True, 0.0, 5 * STALL) == 5


def test_slower_than_eps_per_window_is_still_killed():
    # 1 u per call: 20 u per 20-call window < 32 u -> one kill per window... the timer counts
    # from the last re-arm, and 32 u takes 33 calls, so the first window kills
    assert _kills(True, 1.0, STALL) == 1


def test_an_ended_row_reanchors_on_its_new_spawn():
    core = _FakeCore().at(0.0)
    rr = _rr(core, True)
    for i in range(10):
        core.at(3000.0)                                  # far down the field, then still
        _step(rr, core)
    core.at(0.0)                                         # a respawn far back
    _step(rr, core, done=1)
    assert rr._anchor[0] == pytest.approx(10000.0) and rr._best[0] == pytest.approx(10000.0)
    assert rr._since[0] == 0
    for k in range(4):                                   # 40 u from the NEW spawn re-arms
        core.at(10.0 * (k + 1))
        _step(rr, core)
    assert rr._since[0] == 0 and rr._anchor[0] == pytest.approx(9960.0)


def test_off_keeps_no_anchor():
    core = _FakeCore().at(0.0)
    rr = _rr(core, False)
    assert rr._anchor is None and rr.stall_cum is False


def test_the_eval_hook_mirrors_it():
    import train_fast as tf
    for cum, want in ((False, True), (True, False)):
        core = _FakeCore().at(0.0)
        hk = tf.make_eval_stall_hook(core, _LineField(), STALL, 32.0, 1, cum=cum)
        no = np.zeros(1, bool)
        for t in range(3 * STALL):
            core.at(10.0 * t)
            hk(t, None, None, no, no)
        assert (hk.state["n"] > 0) is want, (cum, hk.state["n"])


@needs_run
def test_trainer_smoke_config_restore_header_and_record():
    run = "scum_on"
    r = _train(run, ["--stall-cum", "1", "--eval-stall", "1", "--record-every", "2048",
                     "--eval-eps", "1"], steps="4096")
    assert "--stall-cum: the stall timer re-arms on a CUMULATIVE improvement" in r.stdout
    cfg = json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["stall_cum"] == 1
    trajs = sorted((ROOT / "runs" / run).glob("traj_*.jsonl"))
    assert trajs
    hdr = json.loads(trajs[-1].read_text(encoding="utf-8").splitlines()[0])
    assert hdr["eval_stall"] == 1 and hdr["stall_cum"] == 1
    run2 = "scum_bare"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + ["--steps", "6144"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "stall_cum=1" in r2.stdout
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--episodes", "1", "--ep-ticks", "300"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    rec = sorted((ROOT / "runs" / run).glob("traj_*_rec*.jsonl"))
    assert rec
    h3 = json.loads(rec[-1].read_text(encoding="utf-8").splitlines()[0])
    assert h3["eval_stall"] == 1 and h3["stall_cum"] == 1
    for dd in (ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)
