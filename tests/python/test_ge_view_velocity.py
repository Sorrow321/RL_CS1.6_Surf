"""explore_phase1 --view-velocity + tools/ge_replay.py (pbr2b2, 2026-10-24).

(a) with the flag the bursts act in the --view-absolute velocity frame (core view_mode 1, the
    view command off_warp(q / 127) from action column 0); the meta says so, and with
    --exact-chains ge_replay replays the deepest chain exactly in that frame;
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
sys.path.insert(0, str(ROOT / "tools"))
from test_exact_chains import ARGS, FIELD, GE  # noqa: E402
from test_ge_minutes import PB, _run, needs_pb  # noqa: E402


def test_view_row():
    from explore_phase1 import VIEW_Q, view_row
    from surfgym.view import off_warp
    a = np.zeros((4, 6), np.int32)
    a[:, 0] = [0, VIEW_Q, -VIEW_Q, 64]
    v = view_row(a)
    assert v.dtype == np.float32 and v.shape == (4, 2)
    np.testing.assert_allclose(v[:, 0], [0.0, 180.0, -180.0, off_warp(64 / VIEW_Q)], atol=1e-4)
    assert (v[:, 1] == 0).all()


@needs_pb
def test_view_velocity_chains_replay_exactly(tmp_path):
    out = tmp_path / "ge"
    r = _run([sys.executable, "-u", str(GE), "--out", str(out), "--exact-chains",
              "--view-velocity"] + ARGS, timeout=900)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    meta = json.loads((out / "archive.meta.json").read_text())
    assert meta["view_velocity"] is True and meta["view_q"] == 127
    z = np.load(out / "archive.npz", allow_pickle=False)
    q = z["link_acts"][:, 0].astype(np.int64)
    assert q.min() < -20 and q.max() > 20            # signed offsets, not yaw bins 0..14
    line = tmp_path / "line.npy"
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "ge_replay.py"), "--map", str(PB),
              "--archive", str(out / "archive.npz"), "--field", str(FIELD), "--out", str(line)],
             timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "DIVERGED" not in r.stdout
    assert len(np.load(line, allow_pickle=False)) > 1


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:tools/explore_phase1.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--view-velocity" not in s.stdout:
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
