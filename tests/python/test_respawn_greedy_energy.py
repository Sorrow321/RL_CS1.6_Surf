"""--respawn-greedy-energy N (pbr5b2, 2026-10-25): the greedy-line pool also takes the HI..LO
seconds before each of a failed greedy episode's N largest energy-height losses in its last
--respawn-greedy-energy-secs before its stop. Synthetic episodes; the flag off reproduces the
pool of the code before the flag exactly (the module at git HEAD~ is loaded side by side)."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import needs_pb  # noqa: E402

from surfgym.core import STATE_DTYPE                                 # noqa: E402
from surfgym.greedy_spawn import GreedyLinePool, energy_losses      # noqa: E402

BASE = "ef2d3db"      # the commit before the flag


def _episode(n=3000, losses=((1200, 400.0), (2000, 300.0), (2500, 100.0))):
    """A fast horizontal line that loses speed at the given ticks; origin x = the tick."""
    ep = np.zeros(n, STATE_DTYPE)
    v = np.full(n, 2200.0)
    for t, dv in losses:
        v[t + 1:] -= dv
    ep["velocity"][:, 0] = v
    ep["origin"][:, 0] = np.arange(n)
    ep["origin"][:, 2] = 2000.0
    return ep


def _build(p, eps, ends, d_fn=None):
    p._eps = list(eps)
    p._cur = []
    p._par = None
    return p.build([{"end": e} for e in ends], 10.0, d_fn=d_fn)


def test_energy_losses_ranks_and_spaces():
    ep = _episode()
    out = energy_losses(ep, len(ep) - 1, 100.0, 30.0, 2)
    assert [m for m, _ in out] == [1200, 2000]
    assert out[0][1] < out[1][1] < 0
    # the window: only the last 6 s before tick 2999 -> the losses at 2000 and 2500
    out2 = energy_losses(ep, len(ep) - 1, 100.0, 10.0, 3)
    assert sorted(m for m, _ in out2) == [2000, 2500]


def test_energy_anchor_windows():
    ep = _episode()
    p = GreedyLinePool(0.2, window=(0.25, 2.0), seed=0, energy=2, energy_secs=30.0)
    _build(p, [ep], ["fail"])
    x = np.sort(p.rows["origin"][:, 0].astype(int))
    for m in (1200, 2000):
        w = x[(x >= m - 200) & (x < m - 25 + 1)]
        assert len(w) == 175 and w.min() == m - 200 and w.max() == m - 26
    # the death anchor is still there
    assert x.max() == len(ep) - 26
    assert [s for s, _ in p.info["energy_losses"]] == [12.0, 20.0]
    assert "energy losses anchored" in p.line()


def _base_module():
    src = subprocess.run(["git", "show", f"{BASE}:python/surfgym/greedy_spawn.py"],
                         capture_output=True, text=True, cwd=str(ROOT), check=True).stdout
    path = Path(__file__).with_name("_greedy_spawn_base.py")
    path.write_text(src.replace("from .core import", "from surfgym.core import"),
                    encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location("_greedy_spawn_base", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        path.unlink()
    return mod


def test_flag_off_is_the_code_before_the_flag():
    base = _base_module()
    rng = np.random.default_rng(3)
    eps = [_episode(), _episode(2500, ((700, 250.0),)), _episode(1800, ())]
    for ep in eps:
        ep["origin"][:, 1] = rng.normal(0.0, 50.0, len(ep))
    ends = ["fail", "trunc", "fail"]

    def d_fn(o):
        d = 5000.0 - np.asarray(o, np.float64)[:, 0]
        d[-50:] += 200.0                         # a stop 50 ticks before the end
        return d
    for kw in ({}, {"stuck": 7}, {"contact": True}, {"turn": 30.0, "speed": (1.0, 1.1)}):
        for df in (None, d_fn):
            a = base.GreedyLinePool(0.2, window=(0.25, 2.0), seed=5, **kw)
            b = GreedyLinePool(0.2, window=(0.25, 2.0), seed=5, **kw)
            ia = _build(a, [e.copy() for e in eps], ends, df)
            ib = _build(b, [e.copy() for e in eps], ends, df)
            assert np.array_equal(a.rows, b.rows), kw
            assert ia == ib and a.line() == b.line()
            pool = np.zeros(400, STATE_DTYPE)
            assert np.array_equal(a.mix(pool, 10), b.mix(pool, 10))


@needs_pb
def test_trainer_says_it_dumps_it_and_refuses_it_alone(tmp_path):
    run = "rge_on"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--respawn-greedy", "0.2",
                            "--respawn-greedy-energy", "3",
                            "--respawn-greedy-energy-secs", "10"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert ("+ the 3 largest energy-height losses of each episode's last 10 s before its "
                "stop (--respawn-greedy-energy)") in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["respawn_greedy_energy"] == 3 and cfg["respawn_greedy_energy_secs"] == 10.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("rge_bad", ["--respawn-greedy-energy", "3"], tmp_path)
    assert bad.returncode != 0 and ("--respawn-greedy-energy anchors the --respawn-greedy rows"
                                    in (bad.stdout + bad.stderr))
    shutil.rmtree(ROOT / "runs" / "rge_bad", ignore_errors=True)


@needs_pb
def test_trainer_flag_off_is_the_trainer_before_it(tmp_path):
    """The trainer with --respawn-greedy and no --respawn-greedy-energy against the trainer at
    the commit before the flag: the same config dump and the same per-iteration numbers."""
    from test_ge_minutes import _csv
    pre = ROOT / "python" / "_train_fast_pre_rge.py"
    src = subprocess.run(["git", "show", f"{BASE}:python/train_fast.py"], capture_output=True,
                         cwd=str(ROOT), check=True).stdout
    assert b"--respawn-greedy-energy" not in src
    pre.write_bytes(src)
    a, b = ROOT / "runs" / "rge_off_new", ROOT / "runs" / "rge_off_old"
    try:
        for run, script in ((a.name, ROOT / "python" / "train_fast.py"), (b.name, pre)):
            r = _run_demo(run, ["--demo-success-d", "50", "--respawn-greedy", "0.2"], tmp_path,
                          script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "respawn_greedy_energy" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        pre.unlink(missing_ok=True)
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
