"""explore_phase1 --tick-ms / ge_replay / auto_spine.ckpt_tick (pbr4b3, 2026-10-25).

A policy trained under --tick-ms-schedule flies another physics tick than the explorer's 10 ms;
the explorer now takes a whole-ms --tick-ms, records it in its meta, ge_replay replays the chains
at that tick, and auto_spine reads the tick off the checkpoint it searches with.

(a) --tick-ms 8: the meta says so, the chains replay exactly, and the archive differs from 10 ms;
(b) flag off: archive and meta bit-identical to the explorer before the change;
(c) auto_spine.ckpt_tick: the config's realised tick, else tick_ms, else 10.
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
from test_ge_minutes import PB, _run, needs_pb  # noqa: E402

GE = ROOT / "tools" / "explore_phase1.py"
FIELD = ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz"
ARGS = ["--map", str(PB), "--envs", "128", "--cell", "64", "--cell-energy", "32",
        "--decisions", "50", "--act-every", "4", "--maxvel", "3500", "--max-iters", "3",
        "--threads", "2", "--exact-chains"]


@needs_pb
def test_tick_ms_chains_replay_exactly(tmp_path):
    for name, extra in (("t8", ["--tick-ms", "8"]), ("t10", [])):
        r = _run([sys.executable, "-u", str(GE), "--out", str(tmp_path / name)] + ARGS + extra,
                 timeout=900)
        assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    meta = json.loads((tmp_path / "t8" / "archive.meta.json").read_text())
    assert meta["tick_ms"] == 8
    assert "tick_ms" not in json.loads((tmp_path / "t10" / "archive.meta.json").read_text())
    a8 = np.load(tmp_path / "t8" / "archive.npz")["state"]
    a10 = np.load(tmp_path / "t10" / "archive.npz")["state"]
    assert len(a8) != len(a10) or not np.array_equal(a8["origin"], a10["origin"])
    line = tmp_path / "line.npy"
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "ge_replay.py"), "--map", str(PB),
              "--archive", str(tmp_path / "t8" / "archive.npz"), "--field", str(FIELD),
              "--out", str(line)], timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "DIVERGED" not in r.stdout
    assert len(np.load(line, allow_pickle=False)) > 1


def _pre_change(dst: Path):
    s = subprocess.run(["git", "show", "HEAD:tools/explore_phase1.py"], capture_output=True,
                       cwd=str(ROOT))
    if s.returncode != 0 or b'"--tick-ms"' in s.stdout:
        return None
    dst.write_bytes(s.stdout)
    return True


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "tools" / "_explore_phase1_pre_tick.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("HEAD's explorer already has --tick-ms (run before committing)")
        for name, script in (("new", GE), ("old", pre)):
            r = _run([sys.executable, "-u", str(script), "--out", str(tmp_path / name)] + ARGS,
                     timeout=900)
            assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
    finally:
        pre.unlink(missing_ok=True)
    a = np.load(tmp_path / "new" / "archive.npz")
    b = np.load(tmp_path / "old" / "archive.npz")
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        assert np.array_equal(a[k], b[k]), k
    ma = json.loads((tmp_path / "new" / "archive.meta.json").read_text())
    mb = json.loads((tmp_path / "old" / "archive.meta.json").read_text())
    for m in (ma, mb):
        m.pop("wall_s", None), m.pop("minutes", None), m.pop("argv", None)
    assert ma.keys() == mb.keys()


def test_auto_spine_reads_the_checkpoint_tick(tmp_path):
    import torch
    import auto_spine
    for cfg, want in (({"tick_ms_eff": 7.6667, "tick_ms": 7.63}, 7.6667),
                      ({"tick_ms": 9.5}, 9.5), ({}, 10.0)):
        p = tmp_path / "c.pt"
        torch.save({"config": cfg}, p)
        assert auto_spine.ckpt_tick(p) == pytest.approx(want)
    assert auto_spine.ckpt_tick(tmp_path / "missing.pt") == 10.0
