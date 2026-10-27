"""auto_spine --energy-minutes / --energy-cell-energy / --energy-select and train_fast
--auto-spine-search-args (pbr7b4, 2026-10-27): the energy stage's own explorer budget, energy-cell
size and energy-weighted selection (0 = the GE ladder's), and a pass-through of auto_spine options
for the trainer's searches. pbr7b4's focused energy search (15 min, 16 u cells, select 1, roots in
the last 6 s) found the line keeping +125 u to the 74.4% climb where the loop's 5-min default kept
it only through the bounce. CPU: the overrides apply to the energy stage only and are restored, off
= the ladder's values, the trainer appends the arguments to its search and refuses them without
--auto-spine, TRAIN_ONLY."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import auto_spine  # noqa: E402
from test_auto_spine_energy import _archive, _field, _line  # noqa: E402
from test_ge_minutes import needs_pb  # noqa: E402


def _run(tmp_path, monkeypatch, extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fp = _field(tmp_path)
    monkeypatch.setattr(auto_spine, "beam",
                        lambda a, out, branch_tick=0, tag=None: (_line(), {"beam_rc": 0}))
    seen = []

    def fake_ge(a, out, line, end, boosts=None, tag="ge", line_keys=None, roots=None, pick=None):
        seen.append((tag, float(a.ge_minutes), float(a.ge_energy), float(a.ge_select_energy)))
        if pick is None:
            return _line(650.0), {"ge_rc": 0}
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
                          "--ge-minutes", "1", "--ge-energy", "32", "--stall-margin", "50",
                          "--branch-energy", "0", "--energy-gain", "100"] + extra)
    return rc, seen


def test_off_the_energy_stage_uses_the_ladder_budget(tmp_path, monkeypatch):
    rc, seen = _run(tmp_path / "off", monkeypatch, [])
    assert rc == 0 and [s[0] for s in seen] == ["ge", "ge_energy"]
    assert seen[1][1:] == (1.0, 32.0, 0.0)


def test_the_energy_stage_gets_its_own_budget(tmp_path, monkeypatch):
    rc, seen = _run(tmp_path / "on", monkeypatch,
                    ["--energy-minutes", "15", "--energy-cell-energy", "16", "--energy-select", "1"])
    assert rc == 0 and [s[0] for s in seen] == ["ge", "ge_energy"]
    assert seen[0][1:] == (1.0, 32.0, 0.0)          # the ladder's stage: unchanged
    assert seen[1][1:] == (15.0, 16.0, 1.0)         # the energy stage: its own


@needs_pb
def test_the_trainer_passes_the_search_args(tmp_path):
    from test_demo_success_d import _run_demo
    run = "asa_smoke"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--auto-spine", "--auto-spine-quorum", "1",
                            "--auto-spine-patience", "1", "--auto-spine-envs", "16",
                            "--auto-spine-ge-minutes", "0", "--auto-spine-extra-secs", "1",
                            "--auto-spine-retry", "1", "--record-every", "2048",
                            "--eval-eps", "2", "--ep-ticks", "300",
                            "--auto-spine-search-args", "--energy-minutes 15 --energy-select 1"],
                      tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert re.search(r"auto-spine search 1: \S+ -u \S+auto_spine.py .*--energy-minutes 15 "
                         r"--energy-select 1", out), out[-4000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine_search_args"] == "--energy-minutes 15 --energy-select 1"
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("asa_bad", ["--demo-success-d", "50", "--auto-spine-search-args", "--x 1"],
                    tmp_path)
    assert bad.returncode != 0 and "--auto-spine-search-args extends --auto-spine's search" in (
        bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "asa_bad", ignore_errors=True)
    assert '"auto_spine_search_args"' in (ROOT / "tools" / "record_ckpt.py").read_text(
        encoding="utf-8")
