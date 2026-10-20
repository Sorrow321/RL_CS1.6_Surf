"""The predecessor archive (--archive-frac) and --respawn-novel together (2026-10-20, r2b1).

A novel episode starts from a MOVED state and is never harvested by the reservoir, so its flight
can never become anyone's spawn (NovelSpawnSampler, the user's "we cannot teleport forward"). The
archive did not know about the tag: it froze a novel episode's run-up before a rare entry and
replayed it as a REAL spawn (r2b1_pa: the real-episode ceiling jumped 28.8 -> 30.0% in 3M steps).

(a) PredecessorArchive.observe(skip=...): a skipped env takes no snapshot and no rare entry, and a
    pending prefix of its is dropped; an unskipped env beside it commits as before.
(b) The trainer, CPU smoke on sidistic: --archive-frac with --respawn-novel runs and logs.
(c) Without --respawn-novel the archive path is the trainer before this change, bit for bit.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.archive import PredecessorArchive                      # noqa: E402
from surfgym.core import STATE_DTYPE                                  # noqa: E402
from test_ez_viewc import SMOKE, TRAIN, _csv, needs_run               # noqa: E402
from test_spawn_states_novel import _run                              # noqa: E402


def _states(t):
    st = np.zeros(2, STATE_DTYPE)
    st["origin"][:, 0] = [t, 1000.0 + t]
    return st


def test_a_skipped_env_never_becomes_a_row():
    a = PredecessorArchive(2, window_ticks=8, hold_ticks=4, capacity=100, snap_every=2)
    no = np.zeros(2, bool)
    skip = np.array([False, True])
    for t in range(10):                                   # both fill their rings
        a.observe(_states(t), no, no, no, skip)
    assert a.ring_n[0] > 0 and a.ring_n[1] == 0           # the skipped env snapshots nothing
    a.observe(_states(10), np.array([True, True]), no, no, skip)   # both enter a rare cell
    assert 0 in a.pending and 1 not in a.pending
    for t in range(11, 20):                               # both survive the hold
        a.observe(_states(t), no, no, no, skip)
    assert a.size > 0 and a.n_commit == 1
    assert (a.store["origin"][:a.size, 0] < 1000.0).all()   # only env 0's run-up
    # a pending prefix of an env that becomes skipped (a novel episode started) is dropped
    b = PredecessorArchive(2, window_ticks=8, hold_ticks=6, capacity=100, snap_every=2)
    for t in range(10):
        b.observe(_states(t), no, no, no, None)
    b.observe(_states(10), np.array([True, False]), no, no, None)
    assert 0 in b.pending
    b.observe(_states(11), no, no, no, np.array([True, False]))
    assert 0 not in b.pending and b.n_skip == 1
    # skip=None is the archive before this change
    c = PredecessorArchive(2, window_ticks=8, hold_ticks=4, capacity=100, snap_every=2)
    for t in range(10):
        c.observe(_states(t), no, no, no)
    assert c.ring_n.tolist() == [4, 4]


def _train(run, extra, steps="20480", script=TRAIN):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE
             + ["--ep-ticks", "400", "--respawn-margin", "0.1", "--steps", steps] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


@needs_run
def test_the_archive_runs_beside_novel_spawns():
    run = "archnov_on"
    r = _train(run, ["--respawn-novel", "--archive-frac", "0.2", "--int-rare", "2",
                     "--archive-hold", "0.5"], steps="40960")
    assert "--archive-frac 0.2" in r.stdout and "respawn NOVEL" in r.stdout
    cfg = json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["archive_frac"] == 0.2 and cfg["respawn_novel"] == 1
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"_nv.novel_env" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_without_novel_the_archive_is_the_trainer_before_the_change():
    import torch
    extra = ["--archive-frac", "0.2", "--int-rare", "2", "--archive-hold", "0.5"]
    pre = ROOT / "python" / "_train_fast_pre_archnov.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("archnov_off_new", extra)
        _train("archnov_off_old", extra, script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "archnov_off_new", ROOT / "runs" / "archnov_off_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb
    ra, rb = _csv(a.name), _csv(b.name)
    assert ra and len(ra) == len(rb)
    for x, y in zip(ra, rb):
        for k in x:
            if k != "time/fps":
                assert x[k] == y[k], k
    ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
    for p, q in zip(ta, tb):
        assert p.read_bytes() == q.read_bytes(), p.name
    sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    for dd in (a, b):
        shutil.rmtree(dd, ignore_errors=True)
