"""--auto-spine (pbr3b4, 2026-10-24): the search -> curriculum loop run by the trainer
(tools/auto_spine.py: beam from the true start, the frontier line cut at its last contact and
its best d, Go-Explore from the line's own last seconds when the beam falls short). CPU,
princessburglar: the cut rules' selftest, DemoCurriculum.reset_spine, a smoke run in which the
trainer calls the search and logs its outcome, the refusals, and flag off bit-identical to the
trainer before the change."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


def test_cut_rules_selftest():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "auto_spine.py"), "--selftest"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FAIL" not in r.stdout


def test_stall_tick():
    sys.path.insert(0, str(ROOT / "tools"))
    import auto_spine
    d = np.concatenate([np.linspace(1000, 300, 70), np.full(30, 290.0), [280.0]])
    # first within 50 u of the best (280): 330 at index ~67
    st = auto_spine.stall_tick(d, 50.0)
    assert d[st] <= 330.0 and (st == 0 or d[st - 1] > 330.0)


def test_reset_spine():
    from surfgym.core import STATE_DTYPE
    from surfgym.respawn import DemoCurriculum
    a = np.zeros(30, STATE_DTYPE)
    a["origin"][:, 0] = np.arange(30)
    b = np.zeros(50, STATE_DTYPE)
    b["origin"][:, 0] = 100 + np.arange(50)
    c = DemoCurriculum(a, window=5, rate=0.3, min_ep=1, grow=5, front_frac=0.4,
                       cooldown=1, grow_max=20)
    c.note_outcomes(np.arange(25, 30), np.ones(5))
    c.build_pool(a[:1], pool_size=64)
    assert c.tau < 29
    c.reset_spine(b)
    assert c.n == 50 and c.tau == 49 and c.ep.sum() == 0 and c.win.sum() == 0
    assert c._step == c.grow and c._cool == 0
    assert c.match(b["origin"][[0, 49]]).tolist() == [0, 49]
    assert c.match(a["origin"][[3]]).tolist() == [-1]
    pool = c.build_pool(a[:1], pool_size=64)
    assert len(pool) == 64


@needs_pb
def test_the_trainer_runs_the_search_and_logs_it(tmp_path):
    run = "aspn_smoke"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--auto-spine", "--auto-spine-quorum", "1",
                            "--auto-spine-patience", "1", "--auto-spine-envs", "16",
                            "--auto-spine-ge-minutes", "0", "--auto-spine-extra-secs", "1",
                            "--auto-spine-retry", "1", "--record-every", "2048",
                            "--eval-eps", "2", "--ep-ticks", "300"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-spine: a greedy eval with >= 1 episodes" in out
        assert "auto-spine: " in out and "greedy episodes past the success line" in out
        assert "auto-spine search 1:" in out, out[-6000:]
        assert ("NEW SPINE" in out) or ("NO new spine" in out)
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine"] == 1 and cfg["auto_spine_quorum"] == 1
        k1 = ROOT / "runs" / run / "auto" / "k01"
        assert (k1 / "auto_spine.log").exists()
        assert "AUTO_SPINE " in (k1 / "auto_spine.log").read_text()
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("aspn_bad", ["--auto-spine"], tmp_path)
    assert bad.returncode != 0 and "--auto-spine moves the --demo-success-d line" in (
        bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "aspn_bad", ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--auto-spine" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_aspn.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("aspn_off_new", TRAIN), ("aspn_off_old", pre)):
            r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50"],
                          tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "aspn_off_new", ROOT / "runs" / "aspn_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "auto_spine" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)


def test_branched_beam_command_parses(tmp_path, monkeypatch):
    """The branched beam's flags are ones beam_tas.py's own parser accepts (round 3: a float
    --branch-jitter killed the first live branched beam with an argparse error)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import auto_spine
    seen = {}

    def fake_run(cmd, log, timeout):
        seen["cmd"] = [str(c) for c in cmd]
        return 0
    monkeypatch.setattr(auto_spine, "run", fake_run)
    a = auto_spine.main.__globals__["argparse"].Namespace(
        ckpt="c.pt", map="m.bsp", envs=64, greedy_envs=4, beam_ticks=100, seed=0, timeout=10,
        branch_burst_secs=2.0, branch_hold_secs=0.5, branch_jitter=45, branch_protect=3)
    auto_spine.beam(a, tmp_path, branch_tick=1700)
    cmd = seen["cmd"]
    i = cmd.index("--branch-at")
    assert cmd[i + 1] == "t1700:32"
    import beam_tas
    ap_src = Path(beam_tas.__file__).read_text(encoding="utf-8")
    for flag in ("--branch-burst", "--branch-hold", "--branch-jitter", "--branch-protect"):
        v = cmd[cmd.index(flag) + 1]
        assert v.lstrip("-").isdigit(), (flag, v)        # beam_tas parses these as int
        assert f'"{flag}", type=int' in ap_src
    assert "--branch-view-only" in cmd


def test_ge_ladder_escalates_until_a_stage_passes(tmp_path, monkeypatch):
    """--ge-boost 'S1;S2': stage 2 runs only because stage 1 found no line past the greedy
    frontier, and its line becomes the spine (source beam+ge2)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import auto_spine
    from surfgym.core import STATE_DTYPE
    # a field whose d is 10,000 - x along the x axis (cells of 32 u, 64 x 4 x 4)
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

    def fake_ge(a, out, line, end, boosts=None, tag="ge"):
        calls.append((tuple(boosts), tag))
        if len(calls) == 1:
            return line_to(650.0), {"ge_rc": 0}            # not past greedy - gain
        return line_to(1500.0), {"ge_rc": 0}               # passes
    monkeypatch.setattr(auto_spine, "go_explore", fake_ge)
    out = tmp_path / "k01"
    rc = auto_spine.main(["--ckpt", "c.pt", "--map", "m.bsp", "--field", str(fp),
                          "--out", str(out), "--greedy-d", "9400", "--gain", "500",
                          "--ge-boost", "1.0,1.04;1.08,1.12", "--ge-minutes", "1"])
    assert rc == 0
    assert calls == [((1.0, 1.04), "ge"), ((1.08, 1.12), "ge2")]
    info = json.loads((out / "spine.json").read_text())
    assert info["source"] == "beam+ge2" and info["n"] == 50
    assert "ge2_end_d" in info and "ge_end_d" in info
    assert np.load(out / "spine.npy")["origin"][-1, 0] == 1500.0


@needs_pb
def test_milestones_pay_and_refuse(tmp_path):
    """--demo-milestones: a line 40 u in front of the true spawn is crossed and paid; without the
    success bonus it is refused."""
    from test_demo_success_d import _spine
    from surfgym.goalfield import GoalField
    z = np.load(ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz", allow_pickle=False)
    f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]),
                  float(z["reach_max"]))
    sp = np.load(_spine(tmp_path))
    # the test spine walks EAST of the spawn (away from the goal): a line 5 u inside its far
    # end is crossed by the episodes spawned there that step back toward the goal
    m = float(np.round(f.sample(np.asarray(sp[-1]["origin"], np.float64)[None])[0] - 5.0))
    run = "aspn_miles"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--demo-milestones", f"{m:g}", "--auto-spine",
                            "--auto-spine-milestones", "--auto-spine-envs", "16",
                            "--auto-spine-ge-minutes", "0", "--auto-spine-delay", "1e12",
                            "--steps", "16384"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert f"--demo-milestones: {m:,.0f} - each paid 50 once per episode" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_milestones"] == [m] and cfg["auto_spine_milestones"] == 1
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("aspn_miles_bad", ["--demo-success-d", "50", "--demo-milestones", "57100"],
                    tmp_path)
    assert bad.returncode != 0 and "--demo-milestones pays the --demo-success-bonus" in (
        bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "aspn_miles_bad", ignore_errors=True)


def test_milestone_pay_rule():
    """The milestone arithmetic: paid once per crossing episode, never for a row that is past
    the line when it appears or when it respawns, and a milestone added later extends the
    table without re-paying the old ones."""
    sys.path.insert(0, str(ROOT / "python"))
    from train_fast import milestone_pay
    ms = [100.0, 50.0]
    rec = np.array([120.0, 90.0, 40.0])          # row 1 past 100 already, row 2 past both
    n, paid = milestone_pay(rec, np.zeros(3, bool), ms, None)
    assert n.tolist() == [0, 0, 0] and paid.tolist() == [[False, False], [True, False],
                                                          [True, True]]
    rec = np.array([95.0, 45.0, 30.0])           # row 0 crosses 100, row 1 crosses 50
    n, paid = milestone_pay(rec, np.zeros(3, bool), ms, paid)
    assert n.tolist() == [1, 1, 0]
    n, paid = milestone_pay(rec, np.zeros(3, bool), ms, paid)
    assert n.tolist() == [0, 0, 0]               # once per episode
    rec = np.array([200.0, 45.0, 30.0])          # row 0 ended and respawned at 200
    n, paid = milestone_pay(rec, np.array([True, False, False]), ms, paid)
    assert n.tolist() == [0, 0, 0] and paid[0].tolist() == [False, False]
    rec = np.array([99.0, 45.0, 30.0])           # it crosses 100 again in its new episode
    n, paid = milestone_pay(rec, np.zeros(3, bool), ms, paid)
    assert n.tolist() == [1, 0, 0]
    n, paid = milestone_pay(rec, np.zeros(3, bool), ms + [20.0], paid)   # a new milestone
    assert n.tolist() == [0, 0, 0] and paid.shape == (3, 3)
    rec = np.array([99.0, 45.0, 10.0])           # row 2 was at 30 when 20 appeared: pays now
    n, paid = milestone_pay(rec, np.zeros(3, bool), ms + [20.0], paid)
    assert n.tolist() == [0, 0, 1]


def test_branched_beams_fork_at_the_tick_and_earlier(tmp_path, monkeypatch):
    """--branch-tick T --branch-extra-secs 1,2: branched beams fork at T, T - 1 s and T - 2 s
    until one passes; the passing one is the spine."""
    sys.path.insert(0, str(ROOT / "tools"))
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
        return s
    forks = []

    def fake_beam(a, out, branch_tick=0, tag=None):
        if branch_tick <= 0:
            return line_to(600.0), {"beam_rc": 0}
        forks.append((branch_tick, tag))
        return line_to(1500.0 if branch_tick == 1500 else 650.0), {f"{tag}_rc": 0}
    monkeypatch.setattr(auto_spine, "beam", fake_beam)
    out = tmp_path / "k01"
    rc = auto_spine.main(["--ckpt", "c.pt", "--map", "m.bsp", "--field", str(fp),
                          "--out", str(out), "--greedy-d", "9400", "--gain", "500",
                          "--branch-tick", "1700", "--branch-extra-secs", "1,2",
                          "--ge-minutes", "0"])
    assert rc == 0
    assert forks == [(1700, "beam_br"), (1600, "beam_br2"), (1500, "beam_br3")]
    info = json.loads((out / "spine.json").read_text())
    assert info["source"] == "beam_br3" and info["br3_fork"] == 1500


def test_ge_command_carries_tick_and_energy_selection(tmp_path, monkeypatch):
    """auto_spine --tick-ms 8 --ge-select-energy 64 (pbr4b1): the explorer's command carries both
    and explore_phase1's own parser accepts it; the roots window is counted at the run's tick
    (4 s = 500 ticks at 8 ms); off (no flags) the command has neither."""
    sys.path.insert(0, str(ROOT / "tools"))
    import auto_spine
    import explore_phase1
    from surfgym.core import STATE_DTYPE
    seen = []

    def fake_run(cmd, log, timeout):
        seen.append([str(c) for c in cmd])
        return 0
    monkeypatch.setattr(auto_spine, "run", fake_run)
    line = np.zeros(2000, STATE_DTYPE)
    line["origin"][:, 0] = np.arange(2000, dtype=np.float64)
    fp = tmp_path / "field.npz"
    np.savez(fp, grid=np.zeros((4, 4, 4), np.float32), quant=np.float64(1.0), mins=np.zeros(3),
             cell=np.float64(32.0), reach_max=np.float64(20000.0))
    base = ["--ckpt", "c.pt", "--map", "m.bsp", "--field", str(fp), "--out", str(tmp_path)]
    for flags, tps in ((["--tick-ms", "8", "--ge-select-energy", "64"], 125.0), ([], 100.0)):
        monkeypatch.setattr(auto_spine, "TICKS_PER_S", 100)
        ap_ns = {}

        def fake_beam(a, out, branch_tick=0, tag=None):
            ap_ns["a"] = a
            return None, {}
        monkeypatch.setattr(auto_spine, "beam", fake_beam)
        assert auto_spine.main(base + flags) == 2          # no beam line: no spine
        assert auto_spine.TICKS_PER_S == tps
        a = ap_ns["a"]
        a.roots_secs, a.roots_n, a.ge_minutes, a.threads = 4.0, 8, 1.0, 0
        seen.clear()
        auto_spine.go_explore(a, tmp_path, line, 1999, boosts=[1.0])
        cmd = seen[0]
        roots = np.load(tmp_path / "ge" / "roots.npy")
        assert len(roots) >= 2
        assert float(roots["origin"][0, 0]) == 1999 - round(4.0 * tps)
        args = explore_phase1.build_parser().parse_args(cmd[3:])
        if flags:
            assert args.tick_ms == 8.0 and args.select_energy == 64.0
        else:
            assert "--tick-ms" not in cmd and "--select-energy" not in cmd
            assert args.tick_ms is None and args.select_energy == 0.0
