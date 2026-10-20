"""--respawn-rare F: Go-Explore's per-cell return over the reservoir (autoresearch r2b3, 2026-10-20).

(a) RareCellReturn: only REAL envs' cell entries are counted (a --respawn-novel episode is not);
    a cell's draw is 1/sqrt(1 + N) split evenly over the reservoir states standing in it, mixed
    with the uniform draw by the share F; the counts round-trip through the checkpoint payload.
(b) CPU trainer smoke on sidistic (abs view, keys-hold, novel rows on): the draw is live (the
    step line and progress.csv carry rare/*), the config and the checkpoint carry the state, a
    flagless resume restores it, and the conflicting reservoir draws are refused.
(c) flag off: the trainer before this change and after it give the same config, progress.csv
    (fps aside), eval trajectory bytes and weights on the same smoke.
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
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.respawn import RareCellReturn  # noqa: E402
from test_ez_viewc import (DLL, SID, SID_FIELD, SMOKE, TRAIN, _csv,  # noqa: E402
                           _run, _train)

needs_run = pytest.mark.skipif(
    not (SID.exists() and SID_FIELD.exists() and DLL.exists()),
    reason="needs the built core + sidistic and its prebaked goal field")


def _rc(n=4, frac=0.5, cell=100.0):
    return RareCellReturn((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0), n, cell=cell, frac=frac)


def test_only_real_entries_are_counted():
    r = _rc(n=3)
    pos = np.array([[50, 50, 50], [150, 50, 50], [250, 50, 50]], float)
    r.observe(pos, real=np.array([True, True, False]))      # first decision: every env enters
    k = r.keys(pos)
    assert r.counts[k[0]] == 1 and r.counts[k[1]] == 1 and r.counts[k[2]] == 0
    r.observe(pos + 10.0, real=np.array([True, True, True]))  # same cells: no entry
    assert r.counts[k].tolist() == [1, 1, 0] and r.entries == 2
    moved = pos.copy()
    moved[:, 0] += 100.0                                     # every env enters the next cell
    r.observe(moved, real=np.array([True, False, True]))
    km = r.keys(moved)
    assert r.counts[km[0]] == 1 + (km[0] == k[1])            # env 0 entered env 1's old cell
    assert r.counts[km[2]] == 1 and r.entries == 4
    r.observe(moved, real=None)                              # no mask = every env is real
    assert r.entries == 4


def test_draw_is_per_cell_and_mixed_with_uniform():
    r = _rc(frac=1.0)
    # 3 states in a cell entered 99 times, 1 state in a cell never entered
    origins = np.array([[50, 50, 50]] * 3 + [[550, 550, 550]], float)
    r.counts[r.keys(origins[:1])[0]] = 99
    p = r.weights(origins)
    assert p.shape == (4,) and abs(p.sum() - 1.0) < 1e-12
    w_busy, w_rare = 1.0 / np.sqrt(100.0), 1.0
    pc = np.array([w_busy, w_rare]) / (w_busy + w_rare)
    assert np.allclose(p[:3], pc[0] / 3) and np.isclose(p[3], pc[1])
    # density bias removed: the busy cell's 3 states do not triple its share
    assert p[3] > p[:3].sum()
    r2 = _rc(frac=0.25)
    r2.counts[:] = r.counts
    p2 = r2.weights(origins)
    assert np.allclose(p2, 0.75 / 4 + 0.25 * p) and abs(p2.sum() - 1.0) < 1e-12
    st = r2.stats()
    assert st["cells"] == 2 and 1.0 < st["eff"] <= 2.0


def test_state_round_trip_and_bad_share():
    r = _rc()
    r.observe(np.array([[50, 50, 50], [150, 50, 50], [250, 50, 50], [350, 50, 50]], float))
    d = r.state_dict()
    r2 = _rc()
    assert r2.load_state_dict(d) and np.array_equal(r2.counts, r.counts)
    assert r2.entries == r.entries == 4
    assert not _rc(cell=50.0).load_state_dict(d)             # another grid: refused, fresh
    with pytest.raises(ValueError):
        _rc(frac=0.0)
    with pytest.raises(ValueError):
        _rc(frac=1.5)


# 4 s episodes, a 0.1 s margin and a 3,000-state reservoir: the trainer rebuilds the spawn pool
# from the reservoir only past 2,000 states, which these settings reach after ~400k steps
RARE = ["--ep-ticks", "400", "--respawn-margin", "0.1", "--respawn-reservoir", "3000",
        "--respawn-novel"]
LONG = "450560"


@needs_run
def test_rare_draw_runs_logs_checkpoints_and_resumes():
    """The reservoir passes the pool-rebuild floor inside the smoke, so the per-cell draw is live
    next to the novel rows."""
    run = "rare_on"
    r = _train(run, RARE + ["--respawn-rare", "0.5"], steps=LONG)
    assert "respawn RARE: 50% of the reservoir draw" in r.stdout
    assert re.search(r"  rare eff [0-9,]+/[0-9,]+ N [0-9,]+", r.stdout), r.stdout[-3000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["respawn_rare"] == 0.5 and cfg["respawn_rare_cell"] == 256.0
    rows = _csv(run)
    assert "rare/eff" in rows[0] and any(x["rare/eff"] for x in rows)
    assert int(rows[-1]["rare/cells"]) > 0
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    rr = ck["respawn_rare"]
    assert isinstance(rr, dict) and len(rr) == 1
    counts = next(iter(rr.values()))["counts"]
    assert int((counts > 0).sum()) == int(rows[-1]["rare/cells"])
    # a flagless resume restores the share and the counts
    run2 = "rare_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--respawn-reservoir", "3000",
                                                    "--steps", "471040"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_rare=0.5" in r2.stdout and "restored --respawn-rare counts" in r2.stdout
    assert re.search(r"  rare eff [0-9,]+/[0-9,]+", r2.stdout)
    # another reservoir draw is refused
    bad = _run([sys.executable, "-u", str(TRAIN), "--run", "rare_bad"] + SMOKE
               + ["--steps", "2048", "--respawn-rare", "0.5", "--respawn-binned", "1"])
    assert bad.returncode != 0 and "--respawn-rare is not wired with" in (bad.stdout + bad.stderr)
    for dd in (d, ROOT / "runs" / run2, ROOT / "runs" / "rare_bad"):
        shutil.rmtree(dd, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before --respawn-rare."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--respawn-rare" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    pre = ROOT / "python" / "_train_fast_pre_rare.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("rare_off_new", RARE, steps=LONG)
        _train("rare_off_old", RARE, script=pre, steps=LONG)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "rare_off_new", ROOT / "runs" / "rare_off_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb and "respawn_rare" not in ca
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
    assert "respawn_rare" not in sa
    for dd in (a, b):
        shutil.rmtree(dd, ignore_errors=True)
