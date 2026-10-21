"""The --respawn-novel ceiling is a distance in the REWARD field (2026-10-21, autoresearch r4b4).

A resume that changes the reward field (--race-kill-aware, --goal-field-file) cannot keep the
saved ceiling: in the new field the same place lies at another distance, and on sidistic the
kill-masked field puts the frontier ~900 u farther, so the stale number would let novel rows
past the real ceiling (the user's rule: never past our best potential).

(a) The sampler saves the field's start distance d0 with its state; a load under another d0
    forgets d_best (re-measured from real episodes), a load under the same d0 keeps it, and a
    state without d0 (saved before the key) is kept for the caller to judge; the visits stay.
(b) The trainer (CPU, sidistic): resuming a novel-sampler checkpoint with --goal-field-file
    (here a shifted copy of the map's field, same lattice) announces the re-measure and runs;
    a plain resume does not.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ez_viewc import SID, SID_FIELD, SMOKE, TRAIN, _run, needs_run   # noqa: E402


def _field():
    from surfgym.goalfield import GoalField
    z = np.load(SID_FIELD, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def _sampler(d0):
    from surfgym import SurfCore, default_config
    from surfgym.respawn import NovelSpawnSampler
    core = SurfCore(str(SID), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    return NovelSpawnSampler(core, _field(), 4, d0=d0)


@needs_run
def test_a_load_under_another_field_forgets_the_ceiling_and_keeps_the_visits():
    a = _sampler(97768.0)
    a.d_best = 41000.0
    a.visit[3, 4, 5] = True
    st = a.state_dict()
    assert st["d0"] == 97768.0
    same = _sampler(97768.0)
    assert same.load_state_dict(st) and same.d_best == 41000.0
    other = _sampler(98888.0)
    assert other.load_state_dict(st) and not np.isfinite(other.d_best)
    assert other.visit[3, 4, 5] and other.visit.sum() == 1
    old = dict(st)
    old.pop("d0")
    legacy = _sampler(98888.0)
    assert legacy.load_state_dict(old) and legacy.d_best == 41000.0


NOVEL_SMOKE = ["--respawn-reservoir", "5000", "--respawn-novel", "--record-every", "1e12"]


@needs_run
def test_a_resume_that_changes_the_reward_field_re_measures_the_ceiling(tmp_path):
    from test_respawn_greedy import _seeded_ckpt
    d0, seeded = _seeded_ckpt("novceil_base")
    run = "novceil_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + NOVEL_SMOKE + ["--steps", "8192"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    ck = ROOT / "runs" / run / "ckpt_final.pt"
    # a reward field on the same lattice, every distance (and the unreachable sentinel, and
    # reach_max with it) 500 u longer: the same reachable set, another start distance
    z = dict(np.load(SID_FIELD, allow_pickle=False))
    q = float(z["quant"])
    z["grid"] = (z["grid"].astype(np.int64) + int(round(500.0 / q))).astype(z["grid"].dtype)
    z["reach_max"] = np.asarray(float(z["reach_max"]) + 500.0, dtype=z["reach_max"].dtype)
    shifted = tmp_path / "shifted_goal_32.npz"
    np.savez_compressed(shifted, **z)
    msg = "--respawn-novel: the reward field differs from the checkpoint's"
    plain = _run([sys.executable, "-u", str(TRAIN), "--run", "novceil_plain", "--ckpt", str(ck)]
                 + SMOKE + NOVEL_SMOKE + ["--steps", "10240"])
    assert plain.returncode == 0, plain.stdout[-4000:] + plain.stderr[-4000:]
    assert msg not in plain.stdout and "restored --respawn-novel state" in plain.stdout
    moved = _run([sys.executable, "-u", str(TRAIN), "--run", "novceil_moved", "--ckpt", str(ck)]
                 + SMOKE + NOVEL_SMOKE + ["--steps", "10240", "--goal-field-file", str(shifted)])
    assert moved.returncode == 0, moved.stdout[-4000:] + moved.stderr[-4000:]
    assert msg in moved.stdout
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / "novceil_plain",
               ROOT / "runs" / "novceil_moved"):
        shutil.rmtree(dd, ignore_errors=True)
