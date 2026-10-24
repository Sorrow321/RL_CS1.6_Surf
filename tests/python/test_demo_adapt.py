"""--demo-cooldown C / --demo-adapt M / --demo-replay R (pbr2b4, 2026-10-24): the backward
curriculum's pace, and practice of the spine before tau.

* the class with the defaults (cooldown 20, adapt 0) is byte-identical to DemoCurriculum before
  the change (loaded from git): same pools, same tau path, same log line;
* --demo-adapt doubles the step over an easy stretch and halves it into a precise one, so tau
  runs fast to a hard state and then settles beside it;
* --demo-cooldown spaces the moves;
* --demo-replay draws that share uniformly over the whole spine, before tau included;
* the trainer accepts both, prints them and dumps them (CPU, princessburglar)."""
from __future__ import annotations

import importlib.util
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

from surfgym.core import STATE_DTYPE  # noqa: E402
from surfgym.respawn import DemoCurriculum  # noqa: E402


def _old_class(tmp_path):
    """DemoCurriculum as it was before --demo-adapt (the newest commit without it)."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in r.stdout.split():
        s = subprocess.run(["git", "show", f"{ref}:python/surfgym/respawn.py"],
                           capture_output=True, text=True, cwd=str(ROOT), timeout=60)
        if s.returncode == 0 and "adapt" not in s.stdout:
            pkg = tmp_path / "oldpkg"
            pkg.mkdir(exist_ok=True)
            src = s.stdout.replace("from .core import STATE_DTYPE",
                                   "from surfgym.core import STATE_DTYPE")
            (pkg / "respawn_old.py").write_text(src)
            spec = importlib.util.spec_from_file_location("respawn_old", pkg / "respawn_old.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.DemoCurriculum
    pytest.skip("no pre-change respawn.py in the history")


def _spine(n):
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(n, dtype=np.float32) * 7.0
    s["origin"][:, 1] = 11.0
    return s


def _drive(c, hard, calls, seed=3, p_ok=0.9):
    """Episodes from spine states >= hard succeed with p_ok, earlier ones never; one
    build_pool per call (one training iteration), outcomes of every demo row of the pool."""
    rng = np.random.default_rng(seed)
    start = np.zeros(8, STATE_DTYPE)
    taus, steps, pools = [], [], []
    for _ in range(calls):
        pool = c.build_pool(start, pool_size=512)
        idx = c.match(pool["origin"])
        wins = (idx >= hard) & (rng.random(len(idx)) < p_ok)
        c.note_outcomes(idx, wins)
        taus.append(c.tau)
        steps.append(getattr(c, "step", None))
        pools.append(pool["origin"].copy())
    return taus, steps, pools


def test_defaults_are_the_old_curriculum(tmp_path):
    Old = _old_class(tmp_path)
    for kw in ({}, {"grow": 20, "front_frac": 0.67}, {"grow": 5, "tau0": 300}):
        a = Old(_spine(400), window=10, **kw)
        b = DemoCurriculum(_spine(400), window=10, **kw)
        ta, _, pa = _drive(a, hard=150, calls=400)
        tb, _, pb = _drive(b, hard=150, calls=400)
        assert ta == tb, kw
        assert all(np.array_equal(x, y) for x, y in zip(pa, pb)), kw
        assert a.last_info == b.last_info, kw
        assert len(set(ta)) > 3, (kw, "the curriculum must actually move in this test")


def test_adapt_runs_fast_then_settles_at_the_hard_state():
    n, hard = 3000, 600
    c = DemoCurriculum(_spine(n), window=10, grow=8, front_frac=0.67, cooldown=2, adapt=64)
    taus, steps, _ = _drive(c, hard=hard, calls=900)
    assert max(steps) == 64, "an easy stretch doubles the step up to the cap"
    first = next(i for i, t in enumerate(taus) if t <= hard + 64)
    # 2,400 states at <= 64 per move every 2 calls: well under the fixed step's 300 moves
    assert first < 200, first
    tail = taus[-200:]
    assert min(tail) >= hard - 10 and max(tail) <= hard + 16, (min(tail), max(tail))
    assert max(steps[-200:]) <= 4, "the step shrinks into the precise stretch"
    fixed = DemoCurriculum(_spine(n), window=10, grow=8, front_frac=0.67, cooldown=2)
    ft, _, _ = _drive(fixed, hard=hard, calls=200)
    assert ft[-1] > taus[199], "the adaptive step is ahead of the fixed one at matched calls"


def test_cooldown_spaces_the_moves():
    c = DemoCurriculum(_spine(500), window=10, grow=4, cooldown=5)
    taus, _, _ = _drive(c, hard=0, calls=120)
    moves = [i for i in range(1, len(taus)) if taus[i] != taus[i - 1]]
    assert moves and all(b - a >= 5 for a, b in zip(moves, moves[1:])), moves
    with pytest.raises(ValueError):
        DemoCurriculum(_spine(50), adapt=8)            # adapt needs grow
    with pytest.raises(ValueError):
        DemoCurriculum(_spine(50), grow=4, cooldown=-1)


def test_replay_draws_the_whole_spine():
    n, tau = 1000, 700
    for front in (0.0, 0.67):
        c = DemoCurriculum(_spine(n), window=10, grow=8, front_frac=front, tau0=tau,
                           replay=0.2, min_ep=1e9)        # min_ep: tau never moves here
        pool = c.build_pool(np.zeros(8, STATE_DTYPE), pool_size=20000, fresh_frac=0.1)
        idx = c.match(pool["origin"])
        demo = idx[idx >= 0]
        before = float((demo < tau).mean())
        # 20% uniform over [0, n-1] -> 0.2 x 700/1000 = 14% of the draws land before tau
        assert 0.12 < before < 0.16, (front, before)
        assert demo.min() < 50 and demo.max() == n - 1
        off = DemoCurriculum(_spine(n), window=10, grow=8, front_frac=front, tau0=tau,
                             min_ep=1e9)
        p0 = off.build_pool(np.zeros(8, STATE_DTYPE), pool_size=20000, fresh_frac=0.1)
        i0 = off.match(p0["origin"])
        assert (i0[i0 >= 0] >= tau).all(), "without replay every demo draw is in [tau, n-1]"
    with pytest.raises(ValueError):
        DemoCurriculum(_spine(50), replay=0.2)          # replay needs grow


@pytest.mark.skipif(not (ROOT / "maps" / "surf_princessburglar_b26.bsp").exists(),
                    reason="needs the princessburglar map")
def test_trainer_takes_and_dumps_the_flags(tmp_path):
    from test_demo_success_d import _run_demo
    run = "dadapt_smoke"
    try:
        r = _run_demo(run, ["--demo-grow", "4", "--demo-adapt", "16", "--demo-cooldown", "3",
                            "--demo-replay", "0.2"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-adapt 16: the step adapts in [1, 16] from 4" in out
        assert "--demo-cooldown 3: tau moves at most once every 3 iterations" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert "--demo-replay 0.2: that share of the demo draws is uniform over the whole" in out
        assert cfg["demo_adapt"] == 16 and cfg["demo_cooldown"] == 3
        assert cfg["demo_replay"] == 0.2
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dadapt_bad", ["--demo-adapt", "16"], tmp_path)
    assert bad.returncode != 0 and "--demo-adapt adapts the --demo-grow step" in (bad.stdout
                                                                                   + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "dadapt_bad", ignore_errors=True)
