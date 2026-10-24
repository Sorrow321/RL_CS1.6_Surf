"""explore_phase1 --view-vel + tools/ge_replay.py (pbr2b1, 2026-10-24).

(a) with the flag the archive's runs act in the core's --view-absolute velocity view (the yaw
    column indexes VIEW_VEL_CMDS), the meta records the table, the archive's fast states look
    roughly along their velocity, and ge_replay replays the deepest chain exactly in that mode;
(b) flag off: the archive and its meta are bit-identical to the explorer before the change.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_exact_chains import ARGS, FIELD, GE  # noqa: E402
from test_ge_minutes import PB, _run, needs_pb  # noqa: E402


@needs_pb
def test_view_vel_chains_replay_exactly(tmp_path):
    out = tmp_path / "ge"
    r = _run([sys.executable, "-u", str(GE), "--out", str(out), "--exact-chains", "--view-vel"]
             + ARGS, timeout=900)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    z = np.load(out / "archive.npz", allow_pickle=False)
    meta = json.loads((out / "archive.meta.json").read_text())
    assert meta["view_vel"] is True and len(meta["view_vel_cmds"]) == 15
    st = z["state"]
    v = st["velocity"].astype(np.float64)
    fast = np.hypot(v[:, 0], v[:, 1]) >= 150.0
    if fast.sum() >= 20:
        off = (st["yaw"][fast] - np.degrees(np.arctan2(v[fast, 1], v[fast, 0])) + 180.0) % 360.0
        off = np.abs(off - 180.0)
        # the commands are offsets of at most 60 deg; the view lags a turning velocity by a few
        assert np.median(off) < 45.0, np.median(off)
    line = tmp_path / "line.npy"
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "ge_replay.py"), "--map", str(PB),
              "--archive", str(out / "archive.npz"), "--field", str(FIELD), "--out", str(line)],
             timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "DIVERGED" not in r.stdout and "--view-vel archive" in r.stdout
    ln = np.load(line, allow_pickle=False)
    assert len(ln) > 1


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:tools/explore_phase1.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--view-vel" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "tools" / "_explore_phase1_pre_vv.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change explorer in the first-parent history")
        for name, script in (("new", GE), ("old", pre)):
            r = _run([sys.executable, "-u", str(script), "--out", str(tmp_path / name),
                      "--exact-chains"] + ARGS, timeout=900)
            assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    finally:
        pre.unlink(missing_ok=True)
    a = np.load(tmp_path / "new" / "archive.npz")
    b = np.load(tmp_path / "old" / "archive.npz")
    assert a.files == b.files
    for k in a.files:
        assert np.array_equal(a[k], b[k]), k
    assert (json.loads((tmp_path / "new" / "archive.meta.json").read_text())
            == json.loads((tmp_path / "old" / "archive.meta.json").read_text()))
