"""explore_phase1 --view-velocity + tools/ge_replay.py (pbr3b1, 2026-10-24).

(a) with the flag (and --exact-chains) the archive keeps the runs' per-tick view TARGETS
    (link_views, same offsets as link_acts), its meta says so, and ge_replay replays every link
    of the deepest chain exactly through a view_mode 1 core;
(b) the explored lines keep their views near the velocity (the policy's own frame), which the
    uniform yaw-delta explorer does not;
(c) flag off is covered by test_exact_chains.test_flag_off_is_bit_identical (the archive and
    meta of the pre-exact-chains explorer, byte for byte).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_ge_minutes import PB, _run, needs_pb  # noqa: E402
from test_exact_chains import ARGS, FIELD, GE  # noqa: E402


@needs_pb
def test_view_velocity_chains_replay_exactly(tmp_path):
    out = tmp_path / "ge"
    r = _run([sys.executable, "-u", str(GE), "--out", str(out), "--exact-chains",
              "--view-velocity"] + ARGS, timeout=900)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    z = np.load(out / "archive.npz", allow_pickle=False)
    meta = json.loads((out / "archive.meta.json").read_text())
    assert meta["view_velocity"] is True and meta["exact_chains"] is True
    assert "link_views" in z.files
    lv, la, off = z["link_views"], z["link_acts"], z["link_off"]
    assert lv.shape == (len(la), 2) and lv.dtype == np.float32
    assert len(off) == len(z["parent"]) + 1
    # the draws are targets in degrees (off_warp: within +-180, pitch 0)
    assert np.abs(lv[:, 0]).max() <= 180.0 and (lv[:, 1] == 0.0).all()
    line = tmp_path / "line.npy"
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "ge_replay.py"), "--map", str(PB),
              "--archive", str(out / "archive.npz"), "--field", str(FIELD), "--out", str(line)],
             timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "DIVERGED" not in r.stdout
    st = np.load(line, allow_pickle=False)
    assert len(st) > 1
    # (b) views near the velocity on the moving rows of the replayed line
    v = st["velocity"].astype(np.float64)
    fast = np.hypot(v[:, 0], v[:, 1]) >= 200.0
    if fast.sum() >= 20:
        hd = np.degrees(np.arctan2(v[fast, 1], v[fast, 0]))
        off_deg = np.abs((st["yaw"][fast] - hd + 180.0) % 360.0 - 180.0)
        assert np.median(off_deg) < 45.0, np.median(off_deg)
