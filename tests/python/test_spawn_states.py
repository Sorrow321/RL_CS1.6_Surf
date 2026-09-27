"""--spawn-states FILE --spawn-states-frac F (2026-09-27): a share of every training spawn pool
drawn uniformly from a file of the agent's OWN states (edge_archive.py --dump-states).

(a) section 0: the trainer refuses the file unless SELF_STATES=1 declares it policy-derived;
(b) with the declaration, a CPU smoke run trains, names the file in its provenance line and
    config, and a bad fraction is refused.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_unstuck import ABS, TRAIN, _env                     # noqa: E402
from test_view_continuous import CANNONBALL, SMOKE_FLAGS, needs_run  # noqa: E402


def _states_file(tmp_path):
    from surfgym.core import SurfCore, SurfEnvConfig
    core = SurfCore(str(CANNONBALL), SurfEnvConfig(num_envs=4))
    core.reset(0)
    st = core.get_states().copy()
    st["origin"][:, 2] += np.arange(4, dtype=np.float32)       # four distinct rows
    p = tmp_path / "own_states.npy"
    np.save(p, st)
    return p


def _run_trainer(run, extra, self_states):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    env = _env()
    if self_states:
        env["SELF_STATES"] = "1"
    else:
        env.pop("SELF_STATES", None)
    return subprocess.run([sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE_FLAGS
                          + ["--steps", "4096"] + ABS + list(extra), capture_output=True,
                          text=True, env=env, cwd=str(ROOT), timeout=1800, encoding="utf-8",
                          errors="replace")


@needs_run
def test_spawn_states_needs_self_states(tmp_path):
    f = _states_file(tmp_path)
    r = _run_trainer("ss_refuse", ["--spawn-states", str(f), "--spawn-states-frac", "0.5"],
                     self_states=False)
    assert r.returncode != 0
    assert "SELF_STATES" in (r.stdout + r.stderr)


@needs_run
def test_spawn_states_trains_and_records_its_provenance(tmp_path):
    f = _states_file(tmp_path)
    run = "ss_smoke"
    r = _run_trainer(run, ["--spawn-states", str(f), "--spawn-states-frac", "0.5"],
                     self_states=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "--spawn-states: 4 own states" in r.stdout
    assert "provenance: spawn_states" in r.stdout
    cfg = json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["spawn_states"] == str(f) and cfg["spawn_states_frac"] == 0.5
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_trainer("ss_bad", ["--spawn-states", str(f), "--spawn-states-frac", "1.5"],
                       self_states=True)
    assert bad.returncode != 0 and "--spawn-states-frac in (0, 1)" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "ss_bad", ignore_errors=True)
