"""auto_spine --energy-gain G (pbr5b2, 2026-10-26): when the depth ladder finds no deeper line, an
energy stage searches before the line's largest energy losses and hands back the line to the
DEEPEST archive cell keeping >= G u more energy height than the line at the same progress, marked
"energy" for the trainer's energy success rule (gate 17's playbook, by the loop). (a) the pick rule;
(b) go_explore replays the picked cell (ge_replay --cell) and nothing when the pick is None;
(c) main runs the stage only when asked and only after the ladder failed."""
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


def _line(x_end=600.0, n=400):
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.linspace(100.0, x_end, n)
    s["origin"][:, 1:] = 50.0
    s["velocity"][:, 0] = 500.0
    s["velocity"][250:, 0] = 200.0                           # the line's one loss, at tick 249
    return s


def _archive(line, path, xs=(450.0, 500.0, 560.0), v=(500.0, 500.0, 230.0)):
    """root = the line's tick 200; children at xs with speeds v, as fast as the line there."""
    n = 1 + len(xs)
    st = np.zeros(n, STATE_DTYPE)
    st[0] = line[200]
    parent = np.full(n, -1, np.int64)
    depth = np.zeros(n, np.int64)
    lx = np.asarray(line["origin"][:, 0], np.float64)
    for i, (x, vx) in enumerate(zip(xs, v), 1):
        st[i]["origin"] = (x, 50.0, 50.0)
        st[i]["velocity"] = (vx, 0.0, 0.0)
        parent[i] = 0
        depth[i] = int(np.searchsorted(lx, x)) - 200
    np.savez(path, state=st, parent=parent, depth=depth)
    return st, parent, depth


def test_energy_cell_is_the_deepest_with_the_gain(tmp_path):
    f = auto_spine.load_field(_field(tmp_path))
    line = _line()
    st, parent, depth = _archive(line, tmp_path / "a.npz")
    d0 = float(f.sample(np.asarray(line["origin"][:1], np.float64))[0])
    c, n = auto_spine.energy_cell(st, parent, depth, line, f, d0, 0.0, 100.0, 100.0)
    # h = 50 + v^2/1600: the line has 75 past its loss; 500 u/s = 206 (+131) at x 450 and 500,
    # 230 u/s at x 560 = +8: the deepest with >= 100 is x 500, not the largest gain's first
    assert n >= 3 and c is not None and c["cell"] == 2 and 125.0 < c["gain"] < 135.0
    c2, _ = auto_spine.energy_cell(st, parent, depth, line, f, d0, 0.0, 100.0, 200.0)
    assert c2 is None
    st2, p2, dp2 = st.copy(), parent, depth.copy()
    dp2[2] += 1000                                           # a chain that wandered: not taken
    c3, _ = auto_spine.energy_cell(st2, p2, dp2, line, f, d0, 0.0, 100.0, 100.0)
    assert c3 is not None and c3["cell"] == 1


def test_go_explore_replays_the_picked_cell(tmp_path, monkeypatch):
    seen = []

    def fake_run(cmd, log, timeout):
        cmd = [str(c) for c in cmd]
        seen.append(cmd)
        if cmd[2].endswith("explore_phase1.py"):
            o = Path(cmd[cmd.index("--out") + 1])
            o.mkdir(parents=True, exist_ok=True)
            np.savez(o / "archive.npz", state=np.zeros(1, STATE_DTYPE))
        return 0
    monkeypatch.setattr(auto_spine, "run", fake_run)
    ns = auto_spine.main.__globals__["argparse"].Namespace
    a = ns(map="m.bsp", envs=64, ge_cell=64.0, ge_energy=32.0, act_every=4, ge_minutes=1.0, seed=0,
           maxvel=None, threads=0, tick=10.0, ge_view_follow="none", roots_secs=1.0, roots_n=8,
           ge_boost="1.0", ge_cell_heading=1, ge_keep_all=1, field="f.npz")
    line = _line()
    got = []
    gl, info = auto_spine.go_explore(a, tmp_path / "p", line, 150, boosts=[1.0], tag="ge_energy",
                                     pick=lambda arch: got.append(arch) or 7)
    assert gl is None and info["ge_pick"] == 7 and got and str(got[0]).endswith("archive.npz")
    rep = seen[-1]
    assert rep[2].endswith("ge_replay.py") and rep[rep.index("--cell") + 1] == "7"
    seen.clear()
    gl, info = auto_spine.go_explore(a, tmp_path / "q", line, 150, boosts=[1.0], tag="ge_energy",
                                     pick=lambda arch: None)
    assert gl is None and info["ge_pick"] is None
    assert not any(c[2].endswith("ge_replay.py") for c in seen)
    seen.clear()
    auto_spine.go_explore(a, tmp_path / "r", line, 150, boosts=[1.0])
    rep = [c for c in seen if c[2].endswith("ge_replay.py")][0]
    assert "--cell" not in rep                               # no pick: the deepest, as before


def _main(tmp_path, monkeypatch, extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fp = _field(tmp_path)
    monkeypatch.setattr(auto_spine, "beam",
                        lambda a, out, branch_tick=0, tag=None: (_line(), {"beam_rc": 0}))
    calls = []

    def fake_ge(a, out, line, end, boosts=None, tag="ge", line_keys=None, roots=None, pick=None):
        calls.append((tag, tuple(boosts), roots, pick))
        if pick is None:
            return _line(650.0), {"ge_rc": 0}                # no deeper line: the ladder fails
        arch = out / tag / "archive.npz"
        arch.parent.mkdir(parents=True, exist_ok=True)
        _archive(line, arch)
        cell = pick(arch)
        if cell is None:
            return None, {"ge_rc": 0, "ge_pick": None}
        z = np.load(arch)
        chain = np.concatenate([line[201:300], z["state"][cell:cell + 1]])
        return np.concatenate([line[:201], chain]), {"ge_rc": 0, "ge_pick": cell}
    monkeypatch.setattr(auto_spine, "go_explore", fake_ge)
    out = tmp_path / "k01"
    rc = auto_spine.main(["--ckpt", "c.pt", "--map", "m.bsp", "--field", str(fp), "--out",
                          str(out), "--greedy-d", "9400", "--gain", "500", "--ge-boost", "1.0",
                          "--ge-minutes", "1", "--stall-margin", "50", "--branch-energy", "0"]
                         + extra)
    return rc, calls, out


def test_main_runs_the_energy_stage_only_when_asked(tmp_path, monkeypatch):
    rc, calls, out = _main(tmp_path / "off", monkeypatch, [])
    assert rc == 2 and [c[0] for c in calls] == ["ge"]
    rc, calls, out = _main(tmp_path / "on", monkeypatch, ["--energy-gain", "100"])
    assert rc == 0 and [c[0] for c in calls] == ["ge", "ge_energy"]
    tag, boosts, roots, pick = calls[1]
    assert boosts == (1.0, 1.04) and roots is not None and np.asarray(roots).max() <= 249
    info = json.loads((out / "spine.json").read_text())
    assert info["source"] == "beam+ge_energy" and info["energy"] is True
    assert 125.0 < info["energy_gain"] < 135.0 and info["energy_losses"][0][0] == 249
    sp = np.load(out / "spine.npy")
    assert len(sp) == 201 + 100 and abs(float(sp[-1]["origin"][0]) - 500.0) < 1e-3
    rc, calls, out = _main(tmp_path / "high", monkeypatch, ["--energy-gain", "300"])
    assert rc == 2 and [c[0] for c in calls] == ["ge", "ge_energy"]
    info = json.loads((out / "spine.json").read_text()) if (out / "spine.json").exists() else {}
    assert not info.get("energy")


def test_trainer_energy_rules():
    from train_fast import crossed_with_energy, spine_energy_rule
    d = np.array([900.0, 700.0, 480.0, 400.0])
    h = np.array([2600.0, 2550.0, 2490.0, 2700.0])
    assert crossed_with_energy(d, h, 500.0, 2480.0)          # first at the line: 2,490
    assert not crossed_with_energy(d, h, 500.0, 2500.0)      # 2,700 later does not count
    assert not crossed_with_energy(d, h, 300.0, 0.0)         # never crossed
    s = np.zeros(1, STATE_DTYPE)[0]
    s["origin"] = (0.0, 0.0, 2400.0)
    s["velocity"] = (400.0, 0.0, 0.0)                        # h = 2,400 + 100
    assert spine_energy_rule({"energy": True}, s, 150.0) == 2500.0 - 75.0
    assert spine_energy_rule({"source": "beam"}, s, 150.0) is None
    assert spine_energy_rule({"energy": True}, s, None) is None


def test_the_trainer_passes_and_refuses_the_energy_stage(tmp_path):
    sys.path.insert(0, str(ROOT / "tests" / "python"))
    import shutil
    from test_demo_success_d import _run_demo
    from test_ge_minutes import needs_pb  # noqa: F401
    from test_ge_minutes import PB
    if not PB.exists():
        import pytest
        pytest.skip("no princessburglar")
    run = "aspe_smoke"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--auto-spine", "--auto-spine-quorum", "1",
                            "--auto-spine-patience", "1", "--auto-spine-envs", "16",
                            "--auto-spine-ge-minutes", "0", "--auto-spine-extra-secs", "1",
                            "--auto-spine-retry", "1", "--record-every", "2048",
                            "--eval-eps", "2", "--ep-ticks", "300",
                            "--auto-spine-energy", "150"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-spine-energy 150: when no line is deeper" in out
        assert "auto-spine search 1:" in out and "--energy-gain 150" in out, out[-6000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine_energy"] == 150.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    for name, extra, msg in (
            ("aspe_bad1", ["--demo-success-d", "50", "--demo-success-bonus", "50",
                           "--auto-spine-energy", "150"], "is a stage of --auto-spine"),
            ("aspe_bad2", ["--demo-success-d", "50", "--auto-spine", "--auto-spine-energy", "150"],
             "gates the --demo-success-bonus")):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr), \
            (bad.stdout + bad.stderr)[-3000:]
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


def test_record_ckpt_knows_the_flag():
    import record_ckpt
    assert "auto_spine_energy" in record_ckpt.TRAIN_ONLY
