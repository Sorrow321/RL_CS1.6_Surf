"""--spawn-states-curric (2026-10-23, autoresearch pbr1b3): a reverse curriculum over the rows
of a --spawn-states file (surfgym/ss_curric.py, Florensa et al. 2017).

(a) the weights: an untried row reads p = 0.5 (the band's centre, weight 0.25); a row that
    always succeeds or always fails falls to the floor; duplicate rows share one count;
(b) note(): success is the record G x d0 past the row's own field distance;
(c) set_states() keeps the counts of rows whose origin survives a re-read;
(d) draws follow the weights;
(e) a CPU smoke on princessburglar: the flag trains, logs its line, writes its config key.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.core import STATE_DTYPE           # noqa: E402
from surfgym.ss_curric import SpawnStateCurriculum  # noqa: E402


class _Field:
    """d = 1000 - x (progress grows with x)."""
    def sample(self, p):
        return 1000.0 - np.asarray(p, np.float64)[:, 0]


def _rows(xs):
    a = np.zeros(len(xs), STATE_DTYPE)
    a["origin"][:, 0] = xs
    return a


def test_weights_and_notes():
    c = SpawnStateCurriculum(_rows([0.0, 100.0, 200.0, 100.0, 900.0]), _Field(), d0=1000.0,
                             gain_frac=0.05)          # gain 50 u (x 900: the set's deep end)
    w = c.weights()
    assert np.allclose(w, 0.25) and len(w) == 5
    assert list(c.match(_rows([100.0, 7.0])["origin"])) == [1, -1]
    # row 0: always succeeds (record 60 u past its d = 1000); row 2: always fails (10 u)
    for _ in range(300):
        c.note([0], [1000.0 - 60.0])
        c.note([2], [800.0 - 10.0])
    p = c.p()
    assert p[0] > 0.95 and p[2] < 0.05
    w = c.weights()
    assert w[0] == pytest.approx(0.02) and w[2] == pytest.approx(0.02)
    assert w[1] == pytest.approx(0.25) and w[3] == pytest.approx(0.25)   # duplicate of row 1
    # half successes -> the band's centre
    for k in range(400):
        c.note([1], [900.0 - (60.0 if k % 2 else 10.0)])
    assert c.p()[1] == pytest.approx(0.5, abs=0.03)


def test_set_states_keeps_counts_and_draws_follow_weights():
    c = SpawnStateCurriculum(_rows([0.0, 100.0]), _Field(), d0=1000.0, gain_frac=0.05)
    for _ in range(200):
        c.note([0], [1000.0])                          # row 0 always fails
    c.set_states(_rows([100.0, 0.0, 300.0]))           # reordered, one new row
    assert c.ep[1] > 50 and c.ep[0] == 0 and c.ep[2] == 0
    rng = np.random.default_rng(0)
    d = c.draw(rng, 20000)
    frac0 = float(np.mean(d["origin"][:, 0] == 0.0))
    want = 0.02 / (0.02 + 0.25 + 0.25)
    assert frac0 == pytest.approx(want, abs=0.01)
    assert "rows" in c.line()


PB = ROOT / "maps" / "surf_princessburglar_b26.bsp"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_run = pytest.mark.skipif(not (PB.exists() and DLL.exists()),
                               reason="needs the built core + princessburglar")


@needs_run
def test_cpu_smoke_with_the_curriculum(tmp_path):
    from test_view_ent_squash import SMOKE
    from surfgym import SurfCore, default_config
    core = SurfCore(str(PB), default_config(num_envs=1))
    st = core.get_states().copy()
    rows = np.repeat(st, 4)
    rows["origin"] = [(860.0, -3005.0, 3653.0), (760.0, -3005.0, 3653.0),
                      (960.0, -3005.0, 3653.0), (860.0, -2950.0, 3653.0)]
    rows["velocity"] = 0.0
    f = tmp_path / "ss.npy"
    np.save(f, rows)
    run = "_t_sscur"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2")
    try:
        r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                            "--run", run] + SMOKE + ["--steps", "2048", "--race-ratchet",
                                                     "--spawn-states", str(f),
                                                     "--spawn-states-frac", "0.5",
                                                     "--spawn-states-curric", "0.01"],
                           capture_output=True, text=True, env=dict(env, SELF_STATES="1"),
                           cwd=str(ROOT), timeout=1800, encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        assert "--spawn-states-curric 0.01:" in r.stdout
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg.get("spawn_states_curric") == 0.01
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def test_backward_window_slides_with_success_and_draws_from_it():
    xs = np.arange(0.0, 400.0, 2.0)                    # 200 rows, progress = x
    c = SpawnStateCurriculum(_rows(xs), _Field(), d0=1000.0, gain_frac=0.05, mode="backward",
                             min_ep=30.0)
    assert c.D == 16 and c.step_n == 4
    w = c._window()
    assert np.allclose(np.sort(c.spawn_d[w]), np.sort(1000.0 - xs[-16:]))   # deepest first
    rng = np.random.default_rng(1)
    d = c.draw(rng, 3000)
    deep = d["origin"][:, 0] >= xs[-16]
    assert deep.mean() > 0.99                          # tau 0: window = passed = the deep rows
    # the window succeeds -> it slides toward the start
    for _ in range(40):
        c.note(w, 1000.0 - xs[w] - 60.0)               # record 60 u past the row (gain 50)
    c.step()
    assert c.tau == 4 and c.advances == 1
    # the new window (it overlaps the old one, whose successes decay) keeps failing -> back off
    w2 = c._window()
    for _ in range(400):
        c.note(w2, 1000.0 - xs[w2] - 5.0)
    c.step()
    assert c.tau == 0 and c.advances == 0
    assert "backward" in c.line()


def test_rows_at_the_end_of_the_set_can_succeed():
    c = SpawnStateCurriculum(_rows([0.0, 100.0, 190.0, 200.0]), _Field(), d0=1000.0,
                             gain_frac=0.05)          # gain 50 u; the deepest row is x 200
    c.note([3], [800.0])                               # the deepest row: no gain needed
    c.note([2], [800.0])                               # 10 u from the end: reaching it is enough
    c.note([2], [805.0])                               # 5 u short of the end: a failure
    c.note([0], [960.0])                               # 40 u of 50 needed: a failure
    assert c.win[3] == 1.0 and c.win[2] == pytest.approx(0.99) and c.ep[2] == pytest.approx(1.99)
    assert c.win[0] == 0.0
