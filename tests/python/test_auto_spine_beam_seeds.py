"""auto_spine --beam-seeds N (pbr6b1, 2026-10-26): when the plain beam's line does not reach past the
greedy frontier by --gain, the plain beam runs again with other torch seeds (--seed + 1000 j) until one
does; the deepest line is kept. On pbr6b1_a1's 29.72B weights seed 7 stopped at the 62.3% drop and
seed 9 went past it. N = 1 is the old single beam."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import auto_spine  # noqa: E402
from surfgym.core import STATE_DTYPE  # noqa: E402


def _field(tmp_path):
    nx, ny, nz, cell = 64, 4, 4, 32.0                       # d = 10000 - x, reachable everywhere
    grid = np.zeros((nz, ny, nx), np.float32)
    grid[:] = (10000.0 - (np.arange(nx) + 0.5) * cell)[None, None, :]
    fp = tmp_path / "field.npz"
    np.savez(fp, grid=grid, quant=np.float64(1.0), mins=np.zeros(3), cell=np.float64(cell),
             reach_max=np.float64(20000.0))
    return fp


def _line(x_end, n=400):
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.linspace(100.0, x_end, n)
    s["origin"][:, 1:] = 50.0
    s["velocity"][:, 0] = 500.0
    return s


END_X = {0: 600.0, 1000: 1200.0, 2000: 1500.0}               # seed -> where its line ends


def _run(tmp_path, monkeypatch, extra, ends=END_X):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fp = _field(tmp_path)
    calls = []

    def fake_beam(a, out, branch_tick=0, tag=None, seed=None):
        s = int(a.seed if seed is None else seed)
        calls.append((tag or "beam", s))
        return _line(ends[s]), {f"{tag or 'beam'}_rc": 0}
    monkeypatch.setattr(auto_spine, "beam", fake_beam)
    out = tmp_path / "k01"
    rc = auto_spine.main(["--ckpt", "c.pt", "--map", "m.bsp", "--field", str(fp), "--out", str(out),
                          "--greedy-d", "9400", "--gain", "500", "--ge-minutes", "0",
                          "--branch-energy", "0", "--seed", "0"] + extra)
    return rc, calls, out


def test_one_seed_is_the_old_single_beam(tmp_path, monkeypatch):
    rc, calls, out = _run(tmp_path, monkeypatch, [])
    assert rc == 2 and calls == [("beam", 0)]
    assert not (out / "spine.json").exists()


def test_more_seeds_until_one_passes_the_gain(tmp_path, monkeypatch):
    # seed 0 ends at d 9,400 (short of 8,900); seed 1000 at 8,800 passes; seed 2000 never runs
    rc, calls, out = _run(tmp_path, monkeypatch, ["--beam-seeds", "3"])
    assert rc == 0 and calls == [("beam", 0), ("beam_s2", 1000)]
    info = json.loads((out / "spine.json").read_text())
    assert info["source"] == "beam_s2" and info["beam_tag"] == "beam_s2"
    sp = np.load(out / "spine.npy")
    assert abs(float(sp[-1]["origin"][0]) - 1200.0) < 1e-3


def test_all_seeds_short_keeps_the_deepest_line(tmp_path, monkeypatch, capsys):
    ends = {0: 600.0, 1000: 650.0, 2000: 620.0}               # none reaches d 8,900
    rc, calls, out = _run(tmp_path, monkeypatch, ["--beam-seeds", "3"], ends=ends)
    assert rc == 2 and [c[1] for c in calls] == [0, 1000, 2000]
    assert not (out / "spine.json").exists()
    # the deepest of the three (seed 1000, x 650 -> field d 9,350) is the line the later stages
    # (branched beams, Go-Explore roots) work on
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("beam line:")][-1]
    assert "best d 9,350" in line
