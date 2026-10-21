"""--int-real-only 1 (2026-10-21, autoresearch r3b3): an episode started from a NOVEL row is
neither paid nor counted by the count-based novelty bonus (RaceReward.int_skip = the novel
sampler's novel_env), so a place only novel rows have shown stays novel for the policy's
REAL episodes - the novel sampler's own rule for its visit grid.

(a) RaceReward: a skipped env gets no bonus and adds no count (entry and dwell counting);
    int_skip None is the shipped path.
(b) The trainer (CPU, sidistic): with --respawn-novel on a filled reservoir the flag runs,
    the config carries it and a bare resume restores it; without --respawn-novel or the
    bonus it is refused; the recorder lists it as training-only.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.rewards import RaceReward                             # noqa: E402

from test_ez_viewc import DLL, SID, SID_FIELD, SMOKE, TRAIN, _run, needs_run  # noqa: E402
from test_respawn_greedy import _seeded_ckpt                       # noqa: E402

needs_core = pytest.mark.skipif(not (SID.exists() and DLL.exists()),
                                reason="needs the core + sidistic")


class _Flat:
    reach_max = 1e9
    cell = 32.0

    def sample(self, pos):
        return np.full(len(np.atleast_2d(pos)), 1000.0)


@pytest.fixture(scope="module")
def core():
    from surfgym import SurfCore, default_config
    c = SurfCore(str(SID), default_config(num_envs=4, max_episode_ticks=200,
                                          lidar_w=0, lidar_h=0))
    c.reset(0)
    return c


def _put(core, xs):
    st = core.get_states()
    for i, x in enumerate(xs):
        s = st[i].copy()
        s["origin"][0] = x
        core.set_state(i, s)


def _two_calls(core, skip, **kw):
    rr = RaceReward(_Flat(), scale=1.0, int_coef=0.1, time_pen=0.0, **kw)
    z = np.zeros(4, bool)
    _put(core, [0.0] * 4)
    rr(None, None, None, None, z, z, core)                 # on_reset
    if skip is not None:
        rr.int_skip = np.asarray(skip, bool)
    _put(core, [1000.0, 3000.0, 5000.0, 7000.0])           # four new cells
    r1 = rr(None, None, None, None, z, z, core)
    r2 = rr(None, None, None, None, z, z, core)            # nobody moves
    return r1, r2, rr


@needs_core
def test_a_skipped_env_is_neither_paid_nor_counted(core):
    r1, _, rr = _two_calls(core, [False, True, False, True])
    np.testing.assert_allclose(r1, [0.1, 0.0, 0.1, 0.0], atol=1e-7)
    assert int(rr._counts.sum()) == 2
    keys = rr._cells(core.get_states())
    assert rr._counts[keys[1]] == 0 and rr._counts[keys[3]] == 0
    assert rr._counts[keys[0]] == 1 and rr._counts[keys[2]] == 1


@needs_core
def test_no_mask_is_the_shipped_path(core):
    r1, r2, rr = _two_calls(core, None)
    np.testing.assert_allclose(r1, [0.1] * 4, atol=1e-7)
    assert int(rr._counts.sum()) == 4
    a1, a2, ra = _two_calls(core, [False] * 4)
    assert a1.tobytes() == r1.tobytes() and a2.tobytes() == r2.tobytes()
    assert ra._counts.tobytes() == rr._counts.tobytes()


@needs_core
def test_dwell_counting_skips_the_masked_envs_too(core):
    _, _, rr = _two_calls(core, [False, True, False, True], int_dwell=True)
    keys = rr._cells(core.get_states())
    # entry + one dwell call each for envs 0 and 2, nothing for 1 and 3
    assert rr._counts[keys[0]] == 2 and rr._counts[keys[2]] == 2
    assert rr._counts[keys[1]] == 0 and rr._counts[keys[3]] == 0


def test_the_recorder_treats_the_flag_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"int_real_only"' in src


@needs_run
def test_with_novel_rows_it_runs_is_dumped_and_restored():
    import torch
    from surfgym import SurfCore, default_config
    from surfgym.goalfield import GoalField
    from surfgym.respawn import NovelSpawnSampler
    d0, seeded = _seeded_ckpt("intreal_base")
    # a sampler state as deep as the seeded reservoir, so novel rows are drawn at once
    ck = torch.load(seeded, map_location="cpu", weights_only=False)
    z = np.load(SID_FIELD, allow_pickle=False)
    fld = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                    float(z["cell"]), float(z["reach_max"]))
    sc = SurfCore(str(SID), default_config(num_envs=1, spawn_mode=2, max_episode_ticks=200,
                                           sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))
    ns = NovelSpawnSampler(sc, fld, 64)
    ns.d_best = float(fld.sample(np.asarray(ck["respawn"]["states"]["origin"],
                                            np.float64)).min())
    ck["respawn_novel"] = {"sidistic": ns.state_dict()}
    torch.save(ck, seeded)
    run = "intreal_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)] + SMOKE
             + ["--steps", "18432", "--respawn-reservoir", "5000", "--stall-secs", "2",
                "--respawn-novel", "--int-real-only", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "--int-real-only: episodes started from novel rows" in r.stdout
    assert "novel ceil" in r.stdout
    cfg = json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["int_real_only"] == 1 and cfg["respawn_novel"] == 1
    run2 = "intreal_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE
              + ["--steps", "20480", "--respawn-reservoir", "5000"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "int_real_only=1" in r2.stdout and "--int-real-only: episodes" in r2.stdout
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
@pytest.mark.parametrize("extra", [["--int-real-only", "1"],
                                   ["--int-real-only", "1", "--respawn-novel",
                                    "--int-coef", "0"]])
def test_without_novel_rows_or_the_bonus_it_is_refused(extra):
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "intreal_bad"] + SMOKE
             + ["--steps", "2048"] + extra)
    shutil.rmtree(ROOT / "runs" / "intreal_bad", ignore_errors=True)
    assert r.returncode != 0
    assert "--int-real-only 1 needs --respawn-novel and --int-coef > 0" in r.stdout + r.stderr
