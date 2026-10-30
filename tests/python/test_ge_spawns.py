"""tools/ge_spawns.py: a Go-Explore archive -> --spawn-states rows (CPU, no core, no map)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402
import ge_spawns  # noqa: E402


def _archive(tmp_path: Path) -> tuple[Path, np.ndarray]:
    st = np.zeros(6, STATE_DTYPE)
    # places (64 u): A = rows 0, 1, 2; B = rows 3, 4; C = row 5
    st["origin"] = [[10, 10, 100], [20, 30, 110], [40, 50, 90],
                    [100, 10, 100], [120, 20, 100], [10, 10, 300]]
    st["velocity"] = [[0, 0, 0], [400, 0, 0], [800, 0, 0],
                      [0, 300, 0], [0, 100, 0], [0, 0, 0]]
    st["tick"] = [5, 6, 7, 8, 9, 10]
    st["stuck_ticks"] = 3
    st["yaw"] = np.arange(6)
    p = tmp_path / "archive.npz"
    np.savez(p, key=np.arange(6), state=st, seen=np.zeros(6, np.int64))
    return p, st


def test_best_per_place_keeps_the_most_energetic_state(tmp_path):
    p, st = _archive(tmp_path)
    out = tmp_path / "spawns.npy"
    assert ge_spawns.main(["--archive", str(p), "--out", str(out)]) == 0
    rows = np.load(out, allow_pickle=False)
    assert rows.dtype == STATE_DTYPE
    # A: h = 100, 210, 490 -> row 2; B: 156.25 vs 106.25 -> row 3; C: row 5
    assert rows["yaw"].tolist() == [2.0, 3.0, 5.0]
    assert (rows["tick"] == 0).all() and (rows["stuck_ticks"] == 0).all()
    np.testing.assert_array_equal(rows["origin"], st["origin"][[2, 3, 5]])
    assert not (tmp_path / "spawns.tmp.npy").exists()


def test_all_and_max(tmp_path):
    p, _ = _archive(tmp_path)
    out = tmp_path / "spawns.npy"
    assert ge_spawns.main(["--archive", str(p), "--out", str(out), "--all"]) == 0
    assert len(np.load(out)) == 6
    assert ge_spawns.main(["--archive", str(p), "--out", str(out), "--all", "--max", "4"]) == 0
    rows = np.load(out)
    assert len(rows) == 4 and len(set(rows["yaw"].tolist())) == 4


def test_unreadable_archive_is_not_converted(tmp_path):
    p = tmp_path / "archive.npz"
    p.write_bytes(b"PK\x03\x04 half a zip")
    out = tmp_path / "spawns.npy"
    assert ge_spawns.main(["--archive", str(p), "--out", str(out)]) == 1
    assert not out.exists()


def test_speed_scale(tmp_path):
    p, st = _archive(tmp_path)
    out = tmp_path / "spawns.npy"
    assert ge_spawns.main(["--archive", str(p), "--out", str(out), "--all",
                           "--speed-scale", "2", "2"]) == 0
    rows = np.load(out)
    np.testing.assert_allclose(rows["velocity"], st["velocity"] * 2.0)
    np.testing.assert_array_equal(rows["origin"], st["origin"])
