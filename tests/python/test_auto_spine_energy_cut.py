"""auto_spine --energy-cut / train_fast --auto-spine-energy-cut (pbr6b3, 2026-10-26): a depth spine
the search found from a Go-Explore root boosted above x1.04 is cut at its DEEPEST row keeping >= G u
more energy height than the policy's line at the same progress and handed back as an ENERGY spine,
so the trainer teaches the energy first (princessburglar's x1.12 line past the 62.3% drop stalled a
depth curriculum; its energy cut was learned in ~15 min and the next plain beam passed unboosted).
(a) the cut rule; (b) main cuts a boosted chain, only when asked, and not an x1.04 one; (c) the
trainer passes the flag to the search and refuses it without --auto-spine-energy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import auto_spine  # noqa: E402
from surfgym.core import STATE_DTYPE  # noqa: E402
from test_auto_spine_energy import _field  # noqa: E402

N_LINE, X0, X1 = 400, 100.0, 450.0
DX = (X1 - X0) / (N_LINE - 1)


def _line():
    """the policy's line: d = 10000 - x falls to 9,550; 500 u/s (h 206) until its loss at row 250,
    then 200 u/s (h 75)."""
    s = np.zeros(N_LINE, STATE_DTYPE)
    s["origin"][:, 0] = X0 + np.arange(N_LINE) * DX
    s["origin"][:, 1:] = 50.0
    s["velocity"][:, 0] = 500.0
    s["velocity"][250:, 0] = 200.0
    return s


def _spine(line, boost=1.12, j=200, x_end=520.0, x_keep=400.0):
    """line[:j+1] + a chain from row j as fast as the line (DX per row) to x_end, at 500 x boost
    u/s while x <= x_keep, then 250 u/s (h 89)."""
    n = int(round((x_end - float(line["origin"][j, 0])) / DX))
    ch = np.zeros(n, STATE_DTYPE)
    ch["origin"][:, 0] = float(line["origin"][j, 0]) + np.arange(1, n + 1) * DX
    ch["origin"][:, 1:] = 50.0
    ch["velocity"][:, 0] = np.where(ch["origin"][:, 0] <= x_keep, 500.0 * boost, 250.0)
    return np.concatenate([line[:j + 1], ch])


def test_energy_cut_is_the_deepest_row_keeping_the_gain(tmp_path):
    f = auto_spine.load_field(_field(tmp_path))
    line = _line()
    sp = _spine(line)
    d0 = float(f.sample(np.asarray(line["origin"][:1], np.float64))[0])
    r, c = auto_spine.energy_cut(sp, 200, line, f, d0, 100.0)
    x = np.asarray(sp["origin"][:, 0], np.float64)
    # h = 50 + v^2/1600: 560 u/s = 246 against the line's 75 past its loss (+171) up to x 400;
    # 250 u/s = 89 (+14) after; past x 450 the line never went: not comparable, not taken
    last_keep = int(np.flatnonzero(x <= 400.0)[-1])
    assert r == last_keep and 165.0 < c["gain"] < 175.0, (r, last_keep, c)
    r2, _ = auto_spine.energy_cut(sp, 200, line, f, d0, 200.0)
    assert r2 is None
    # an unboosted chain that keeps exactly the line's energy before the loss keeps +131 after it
    r3, c3 = auto_spine.energy_cut(_spine(line, boost=1.0), 200, line, f, d0, 100.0)
    assert r3 == last_keep and 125.0 < c3["gain"] < 135.0
    # a chain that crawls (10x slower than the line) is not taken
    slow = _spine(line)
    slow = np.concatenate([slow[:201], np.repeat(slow[201:], 10)])
    r4, _ = auto_spine.energy_cut(slow, 200, line, f, d0, 100.0)
    assert r4 is None


def _main(tmp_path, monkeypatch, extra, boost=1.12):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fp = _field(tmp_path)
    line = _line()
    monkeypatch.setattr(auto_spine, "beam",
                        lambda a, out, branch_tick=0, tag=None: (line, {"beam_rc": 0}))

    def fake_ge(a, out, ln, end, boosts=None, tag="ge", line_keys=None, roots=None, pick=None):
        return _spine(ln, boost=boost), {"ge_rc": 0, "ge_root_index": 200,
                                         "ge_root_boost": float(boost)}
    monkeypatch.setattr(auto_spine, "go_explore", fake_ge)
    out = tmp_path / "k01"
    rc = auto_spine.main(["--ckpt", "c.pt", "--map", "m.bsp", "--field", str(fp), "--out",
                          str(out), "--greedy-d", "9900", "--gain", "400", "--ge-boost", "1.0",
                          "--ge-minutes", "1", "--stall-margin", "50", "--branch-energy", "0"]
                         + extra)
    info = json.loads((out / "spine.json").read_text())
    return rc, info, np.load(out / "spine.npy")


def test_main_cuts_a_boosted_chain_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_spine, "ckpt_tick", lambda ckpt: 10.0)
    rc, info, sp = _main(tmp_path / "off", monkeypatch, ["--energy-gain", "100"])
    full = info["ge_end"] + 1                    # the depth spine: the chain to its best d
    assert rc == 0 and len(sp) == full and not info.get("energy") and "energy_cut" not in info
    assert full > 201 + 200 and info["source"] == "beam+ge"
    rc, info, sp = _main(tmp_path / "on", monkeypatch, ["--energy-gain", "100", "--energy-cut"])
    x = np.asarray(sp["origin"][:, 0], np.float64)
    assert rc == 0 and info["energy"] is True and info["energy_cut"]["root_boost"] == 1.12
    assert len(sp) == info["energy_cut"]["row"] + 1 < full and x[-1] <= 400.0 < x[-1] + 2 * DX
    assert info["energy_cut"]["depth_n"] == full and 165.0 < info["energy_gain"] < 175.0
    assert abs(info["end_d"] - (10000.0 - float(x[-1]))) < 32.0
    rc, info, sp = _main(tmp_path / "x104", monkeypatch, ["--energy-gain", "100", "--energy-cut"],
                         boost=1.04)
    assert rc == 0 and len(sp) == full and not info.get("energy")
    rc, info, sp = _main(tmp_path / "nogain", monkeypatch, ["--energy-cut"])
    assert rc == 0 and len(sp) == full and not info.get("energy")


def test_the_trainer_passes_and_refuses_the_energy_cut(tmp_path):
    import shutil
    from test_demo_success_d import _run_demo
    from test_ge_minutes import PB
    if not PB.exists():
        import pytest
        pytest.skip("no princessburglar")
    run = "aspec_smoke"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--auto-spine", "--auto-spine-quorum", "1",
                            "--auto-spine-patience", "1", "--auto-spine-envs", "16",
                            "--auto-spine-ge-minutes", "0", "--auto-spine-extra-secs", "1",
                            "--auto-spine-retry", "1", "--record-every", "2048",
                            "--eval-eps", "2", "--ep-ticks", "300",
                            "--auto-spine-energy", "150", "--auto-spine-energy-cut"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-spine-energy-cut: a depth spine from a GE root boosted" in out
        line = [ln for ln in out.splitlines() if "auto-spine search 1: " in ln and "auto_spine.py" in ln]
        assert line and "--energy-gain 150" in line[0] and "--energy-cut" in line[0], out[-6000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine_energy_cut"] == 1
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("aspec_bad", ["--demo-success-d", "50", "--demo-success-bonus", "50",
                                  "--auto-spine", "--auto-spine-energy-cut"], tmp_path)
    assert bad.returncode != 0 and "--auto-spine-energy-cut keeps --auto-spine-energy" in \
        (bad.stdout + bad.stderr), (bad.stdout + bad.stderr)[-3000:]
    shutil.rmtree(ROOT / "runs" / "aspec_bad", ignore_errors=True)


def test_record_ckpt_knows_the_flag():
    import record_ckpt
    assert "auto_spine_energy_cut" in record_ckpt.TRAIN_ONLY
