"""--demo-replay-frac (pbr7b4, 2026-10-26): under --demo-grow a share R of the demo draws is
uniform over the spine rows BEHIND the curriculum, [0, tau) - rehearsal of the policy's own line
from the start, which the grown range [tau, n-1] never draws (pbr7b4_c14 / c16: greedy evals
collapsing to gates 54.6-62% while the curriculum hammered a hard band at 70.6%). CPU: off draws
byte-identical to DemoCurriculum before the change (loaded from git), the shares on, the
refusals, and a princessburglar trainer smoke run (notice, config key, refusal, TRAIN_ONLY)."""
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


def _spine(n=100):
    a = np.zeros(n, STATE_DTYPE)
    a["origin"][:, 0] = np.arange(n, dtype=np.float64) * 10.0
    return a


def _old_respawn(tmp_path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/surfgym/respawn.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"replay_frac" not in s.stdout:
            p = tmp_path / "respawn_pre.py"
            p.write_bytes(s.stdout)
            spec = importlib.util.spec_from_file_location("surfgym._respawn_pre", p,
                                                          submodule_search_locations=None)
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = "surfgym"
            spec.loader.exec_module(mod)
            return mod
    return None


@pytest.mark.parametrize("front", [0.0, 0.8])
def test_off_draws_are_byte_identical_to_before(tmp_path, front):
    old = _old_respawn(tmp_path)
    if old is None:
        pytest.skip("no pre-change respawn.py in the first-parent history")
    sp = _spine()
    kw = dict(window=5, rate=0.3, min_ep=1, grow=5, front_frac=front, tau0=60, cooldown=1,
              grow_max=20, match_band=True)
    a, b = DemoCurriculum(sp, **kw), old.DemoCurriculum(sp, **kw)
    start = sp[:1]
    for it in range(6):
        pa, pb = a.build_pool(start, pool_size=256), b.build_pool(start, pool_size=256)
        assert pa.tobytes() == pb.tobytes(), it
        assert a.last_info == b.last_info
        w = np.ones(5) if it % 2 == 0 else np.zeros(5)
        a.note_outcomes(np.arange(a.tau, a.tau + 5), w)
        b.note_outcomes(np.arange(b.tau, b.tau + 5), w)
        assert a.tau == b.tau


def test_the_replay_share_lands_behind_tau():
    sp = _spine()
    c = DemoCurriculum(sp, window=5, rate=0.3, min_ep=1e9, grow=5, front_frac=0.5, tau0=60,
                       cooldown=1, replay_frac=0.25)
    pool = c.build_pool(sp[:1], pool_size=4096, fresh_frac=0.10)
    rows = (pool["origin"][:, 0] / 10.0).round().astype(int)[int(round(4096 * 0.10)):]
    n = len(rows)
    behind = (rows < 60).mean()
    band = ((rows >= 60) & (rows <= 64)).mean()
    ahead = (rows > 64).mean()
    # the band: its 50% plus its 5/40 of the 25% left uniform over [tau, n-1]
    assert abs(behind - 0.25) < 0.01 and abs(band - (0.5 + 0.25 * 5 / 40)) < 0.02, (behind, band)
    assert behind + band + ahead == pytest.approx(1.0)
    assert rows.min() >= 0 and rows.max() <= 99 and n > 3000
    assert "behind it [0,59]" in c.last_info
    # at tau 0 there is nothing behind: the share falls back to the grown range
    c0 = DemoCurriculum(sp, window=5, rate=0.3, min_ep=1e9, grow=5, front_frac=0.5, tau0=0,
                        cooldown=1, replay_frac=0.25)
    p0 = c0.build_pool(sp[:1], pool_size=512)
    assert len(p0) == 512 and "behind it" not in c0.last_info
    # a spine swap keeps the share
    c.reset_spine(_spine(50))
    assert c.replay_frac == 0.25


def test_refusals():
    sp = _spine()
    with pytest.raises(ValueError):
        DemoCurriculum(sp, window=5, grow=0, replay_frac=0.2)
    with pytest.raises(ValueError):
        DemoCurriculum(sp, window=5, grow=5, front_frac=0.9, replay_frac=0.2)
    with pytest.raises(ValueError):
        DemoCurriculum(sp, window=5, grow=5, replay_frac=1.0)


def test_train_only_in_record_ckpt():
    assert '"demo_replay_frac"' in (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")


def _needs_pb():
    from test_ge_minutes import needs_pb
    return needs_pb


@_needs_pb()
def test_the_trainer_runs_it(tmp_path):
    from test_demo_success_d import _run_demo
    run = "drf_smoke"
    try:
        r = _run_demo(run, ["--demo-grow", "5", "--demo-front-frac", "0.5",
                            "--demo-replay-frac", "0.2", "--demo-tau0", "30"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--demo-replay-frac 0.2: that share of the demo draws" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_replay_frac"] == 0.2
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("drf_bad", ["--demo-replay-frac", "0.2"], tmp_path)
    assert bad.returncode != 0 and "--demo-replay-frac draws the rows behind a --demo-grow" in (
        bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "drf_bad", ignore_errors=True)
