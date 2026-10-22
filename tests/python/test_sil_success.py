"""--sil-success 1 (2026-10-22, autoresearch r6b3): self-imitation of the SUCCESSES only - the SIL
buffer admits only rows of real episodes (never a --respawn-novel row's) that reach the finish
inside the rollout (surfgym.sil.success_rows + SILBuffer.add(mask=...)).

(a) success_rows: a row belongs to the episode that ends at the first done at or after it; only a
    finishing end marks its rows; the open last episode and novel episodes are never marked.
(b) SILBuffer.add with a mask keeps only masked rows with R > V; no mask = the shipped rule.
(c) The trainer (CPU, sidistic): the flag reaches the config and a bare resume; it is refused without
    --sil-coef.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.sil import SILBuffer, success_rows                    # noqa: E402

from test_ez_viewc import SMOKE, TRAIN, _run, needs_run              # noqa: E402


def test_success_rows_marks_the_finishing_episode_only():
    T, N = 8, 3
    done = np.zeros((T, N), bool)
    goal = np.zeros((T, N), bool)
    nov = np.zeros((T, N), bool)
    # env 0: an episode ends at t 2 (a death), the next at t 5 (a FINISH), then an open one
    done[2, 0] = True
    done[5, 0] = goal[5, 0] = True
    # env 1: finishes at t 3, but that episode was a novel row's
    done[3, 1] = goal[3, 1] = True
    nov[:4, 1] = True
    # env 2: never ends inside the rollout
    out = success_rows(done, goal, nov)
    assert out[:, 0].tolist() == [False, False, False, True, True, True, False, False]
    assert not out[:, 1].any()
    assert not out[:, 2].any()


def test_add_with_a_mask_keeps_only_masked_rows_that_beat_the_critic():
    b = SILBuffer(100, "cpu", torch.float32, 4, 6, 2)
    n = 10
    scal = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
    img = torch.zeros((n, 6))
    act = torch.zeros((n, 2), dtype=torch.long)
    ret = torch.ones(n)
    val = torch.zeros(n)
    val[3] = 5.0                                  # row 3: R < V
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[1, 3, 7]] = True
    assert b.add(scal, img, act, ret, val, mask=mask) == 2      # rows 1 and 7
    assert sorted(b.scal[:b.size, 0].tolist()) == [4.0, 28.0]
    c = SILBuffer(100, "cpu", torch.float32, 4, 6, 2)
    assert c.add(scal, img, act, ret, val) == 9                  # no mask: the shipped rule


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_wires_restores_and_refuses_it():
    run = "silsucc_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE
             + ["--steps", "4096", "--sil-coef", "0.1", "--sil-success", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "--sil-success: only rows of REAL episodes" in r.stdout
    assert _cfg(run)["sil_success"] == 1
    run2 = "silsucc_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "sil_success=1" in r2.stdout and _cfg(run2)["sil_success"] == 1
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "silsucc_bad"] + SMOKE
              + ["--steps", "2048", "--sil-success", "1"])
    assert r3.returncode != 0 and "--sil-success filters --sil-coef" in r3.stdout + r3.stderr
    for dd in (run, run2, "silsucc_bad"):
        shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)
