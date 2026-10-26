"""auto_spine --near-gain M / train_fast --auto-spine-near-gain M (pbr7b4, 2026-10-26): when a more
boosted Go-Explore stage passes the gain, an earlier, less boosted stage's line that ended within
M u of the gain line is the spine instead (pbr7b4_c13's search: stage 1's x1.04 chain 12 u short
of the 500 u gain, stage 2's x1.08 chain taken; the curriculum then stalled at the x1.08 root).
CPU: the ladder with fakes (the near miss taken, off = the old pick, a miss outside the margin
ignored, no stage passing = no spine), the trainer's refusal and TRAIN_ONLY."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))


def _setup(tmp_path, monkeypatch, ends):
    import auto_spine
    from surfgym.core import STATE_DTYPE
    nx, ny, nz, cell = 64, 4, 4, 32.0
    grid = np.zeros((nz, ny, nx), np.float32)
    grid[:] = (10000.0 - (np.arange(nx) + 0.5) * cell)[None, None, :]
    fp = tmp_path / "field.npz"
    np.savez(fp, grid=grid, quant=np.float64(1.0), mins=np.zeros(3), cell=np.float64(cell),
             reach_max=np.float64(20000.0))

    def line_to(x_end, n=50):
        s = np.zeros(n, STATE_DTYPE)
        s["origin"][:, 0] = np.linspace(100.0, x_end, n)
        s["origin"][:, 1:] = 50.0
        s["velocity"][:, 0] = 500.0
        return s
    monkeypatch.setattr(auto_spine, "beam",
                        lambda a, out, branch_tick=0, tag=None: (line_to(600.0), {"beam_rc": 0}))
    calls = []

    def fake_ge(a, out, line, end, boosts=None, tag="ge", line_keys=None, **kw):
        calls.append(tag)
        return line_to(ends[len(calls) - 1]), {"ge_rc": 0}
    monkeypatch.setattr(auto_spine, "go_explore", fake_ge)
    return auto_spine, fp, calls


def _run(auto_spine, fp, out, extra=()):
    # greedy-d 9,400, gain 500: a line passes at d < 8,900; d = 10,000 - x on this field
    return auto_spine.main(["--ckpt", "c.pt", "--map", "m.bsp", "--field", str(fp),
                            "--out", str(out), "--greedy-d", "9400", "--gain", "500",
                            "--ge-boost", "1.0,1.04;1.08;1.12", "--ge-minutes", "1", *extra])


def test_the_near_miss_is_taken(tmp_path, monkeypatch):
    # stage 1 ends at x 1,050 (d ~8,950: 50 u short of the gain line), stage 2 passes
    auto_spine, fp, calls = _setup(tmp_path, monkeypatch, [1050.0, 1500.0, 1600.0])
    out = tmp_path / "k01"
    assert _run(auto_spine, fp, out, ["--near-gain", "100"]) == 0
    assert calls == ["ge", "ge2"]
    info = json.loads((out / "spine.json").read_text())
    assert info["source"] == "beam+ge" and info["near_pick"] == "beam+ge"
    assert info["near_over"] == "beam+ge2"
    assert np.load(out / "spine.npy")["origin"][-1, 0] == 1050.0


def test_off_takes_the_first_stage_past_the_gain(tmp_path, monkeypatch):
    auto_spine, fp, calls = _setup(tmp_path, monkeypatch, [1050.0, 1500.0, 1600.0])
    out = tmp_path / "k01"
    assert _run(auto_spine, fp, out) == 0
    info = json.loads((out / "spine.json").read_text())
    assert info["source"] == "beam+ge2" and "near_pick" not in info


def test_a_miss_outside_the_margin_is_ignored(tmp_path, monkeypatch):
    # stage 1 ends 210 u short of the gain line: outside a 100 u margin
    auto_spine, fp, calls = _setup(tmp_path, monkeypatch, [890.0, 1500.0, 1600.0])
    out = tmp_path / "k01"
    assert _run(auto_spine, fp, out, ["--near-gain", "100"]) == 0
    info = json.loads((out / "spine.json").read_text())
    assert info["source"] == "beam+ge2" and "near_pick" not in info


def test_a_near_miss_alone_is_no_spine(tmp_path, monkeypatch):
    auto_spine, fp, calls = _setup(tmp_path, monkeypatch, [1050.0, 1040.0, 1030.0])
    out = tmp_path / "k01"
    assert _run(auto_spine, fp, out, ["--near-gain", "100"]) == 2
    assert calls == ["ge", "ge2", "ge3"] and not (out / "spine.npy").exists()


def test_train_only_and_refusal(tmp_path):
    assert '"auto_spine_near_gain"' in (ROOT / "tools" / "record_ckpt.py").read_text(
        encoding="utf-8")
    from test_ge_minutes import needs_pb
    if needs_pb.args[0]:
        pytest.skip("needs the built core + princessburglar")
    from test_demo_success_d import _run_demo
    import shutil
    bad = _run_demo("asng_bad", ["--demo-success-d", "50", "--auto-spine-near-gain", "100"],
                    tmp_path)
    assert bad.returncode != 0 and "--auto-spine-near-gain M (> 0) is a rule of" in (
        bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "asng_bad", ignore_errors=True)
