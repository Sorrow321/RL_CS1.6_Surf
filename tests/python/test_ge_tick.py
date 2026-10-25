"""explore_phase1 / ge_replay --tick-ms (pbr4b1, 2026-10-25).

(a) explore_phase1 --exact-chains --tick-ms 8 runs its core at 8 ms, records the tick in the
    archive meta, and ge_replay replays the deepest chain exactly at that tick - while the same
    chain replayed at the 10 ms default diverges (the tick really is part of the dynamics);
(b) flag off: the archive and its meta are bit-identical to the explorer before the flag;
(c) a non-integer tick (a [8, 8, 7] pattern) is refused: its chains would not replay exactly.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_ge_minutes import PB, _run, needs_pb  # noqa: E402

GE = ROOT / "tools" / "explore_phase1.py"
REPLAY = ROOT / "tools" / "ge_replay.py"
FIELD = ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz"
ARGS = ["--map", str(PB), "--envs", "128", "--cell", "64", "--cell-energy", "32",
        "--decisions", "50", "--act-every", "4", "--maxvel", "3500", "--max-iters", "3",
        "--threads", "2"]


def _replay(arch: Path, out: Path):
    return _run([sys.executable, "-u", str(REPLAY), "--map", str(PB), "--archive", str(arch),
                 "--field", str(FIELD), "--out", str(out)], timeout=600)


@needs_pb
def test_tick_chains_replay_at_their_tick(tmp_path):
    out = tmp_path / "ge8"
    r = _run([sys.executable, "-u", str(GE), "--out", str(out), "--exact-chains",
              "--tick-ms", "8"] + ARGS, timeout=900)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    meta = json.loads((out / "archive.meta.json").read_text())
    assert meta["tick_ms"] == 8 and meta["exact_chains"] is True
    r = _replay(out / "archive.npz", tmp_path / "line8.npy")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "DIVERGED" not in r.stdout
    assert len(np.load(tmp_path / "line8.npy")) > 1
    # the same archive read as a 10 ms one: its first moving link no longer lands on the child
    meta.pop("tick_ms")
    (out / "archive.meta.json").write_text(json.dumps(meta))
    r = _replay(out / "archive.npz", tmp_path / "line10.npy")
    assert "DIVERGED" in r.stdout, r.stdout[-3000:]


@needs_pb
def test_tick_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "tools" / "_explore_phase1_pre_tick.py"
    s = subprocess.run(["git", "show", "9fca3235019e6e78ae1c6f6ce7100a1ab03b17d4:"
                        "tools/explore_phase1.py"], capture_output=True, cwd=str(ROOT))
    assert s.returncode == 0 and b"--tick-ms" not in s.stdout
    pre.write_bytes(s.stdout)
    try:
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


@needs_pb
def test_pattern_tick_refused(tmp_path):
    r = _run([sys.executable, "-u", str(GE), "--out", str(tmp_path / "ge"), "--tick-ms",
              "7.63"] + ARGS, timeout=300)
    assert r.returncode != 0 and "whole-ms" in (r.stdout + r.stderr)


def test_select_energy_weights():
    """--select-energy T: W = 1/sqrt(C_seen + 1) x exp((h - h_max) / T); 0 = the Nature weight."""
    sys.path.insert(0, str(ROOT / "tools"))
    sys.path.insert(0, str(ROOT / "python"))
    from explore_phase1 import Archive, CellHash
    from surfgym.core import STATE_DTYPE
    a = Archive(CellHash((0.0,) * 3, (1e5,) * 3, 256.0), capacity=4)
    for j, (z, vx) in enumerate([(100.0, 0.0), (100.0, 400.0), (164.0, 0.0)]):
        s = np.zeros(1, STATE_DTYPE)[0]
        s["origin"][2] = z
        s["velocity"][0] = vx
        a.add(1000 + j, s, depth=j, parent=-1)
    a.seen[:3] = 0
    a.build_sampler()
    w0 = np.diff(np.concatenate([[0.0], a._cum]))
    assert np.allclose(w0, 1.0)                       # off: every unseen cell weighs 1
    a.select_energy, a.two_g = 64.0, 1600.0
    a.build_sampler()
    w = np.diff(np.concatenate([[0.0], a._cum]))
    h = np.array([100.0, 100.0 + 400.0 ** 2 / 1600.0, 164.0])   # 100, 200, 164
    assert np.allclose(w, np.exp((h - h.max()) / 64.0))
    assert w[1] > w[2] > w[0]


@needs_pb
def test_select_energy_meta(tmp_path):
    out = tmp_path / "gee"
    r = _run([sys.executable, "-u", str(GE), "--out", str(out), "--exact-chains", "--tick-ms",
              "8", "--select-energy", "64"] + ARGS, timeout=900)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    meta = json.loads((out / "archive.meta.json").read_text())
    assert meta["select_energy"] == 64.0 and meta["tick_ms"] == 8
    r = _replay(out / "archive.npz", tmp_path / "line.npy")
    assert r.returncode == 0 and "DIVERGED" not in r.stdout, r.stdout[-3000:]
