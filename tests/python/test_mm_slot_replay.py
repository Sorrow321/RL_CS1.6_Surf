"""Multi-map training data, per env: slot K of a --maps run must be the single-map run of map K.

tests/python/mm_slot_replay.py records a --maps run of recipe v14 at CPU smoke scale (the
policy weights of every iteration and eval, every decision's sampled actions, every eval seed,
every training spawn pool, the rollout buffers at the update) and replays each map ALONE on the
single-map code path with the same weights, actions, eval seeds and slot K's per-slot seeds;
compare() checks slot K's rows bit for bit: the observation row (scalars and image), reward,
done, action, log-prob, value (1e-4), the sampling temperatures, the per-row entropy
coefficient, the curriculum's tables, the novelty counts, the reservoir, the coefficients, the
band edges, demo_idx / greedy flags / episode clocks, the spawn pools.

Two differences are reported as notes, not failures: decision 0 of every rollout (the
single-map path's set_unstuck_temp() overwrites the per-env temperature tensors with the
uniform 1 + T at the end of every iteration; --maps never calls it), and float noise of
gamma x V(s_T) on truncated rows (V is evaluated on a different batch of rows).

Maps: MM_TEST_MAPS (comma-separated .bsp paths with their six cell-32 caches; the first three
are used), SURFCORE_DLL. MM_REPLAY_ITERS sets the iterations (default 6). The full v14 line.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v14_smoke as V  # noqa: E402

HERE = Path(__file__).resolve().parent
MAPS = V.MM_MAPS[:3]
needs = pytest.mark.skipif(not V.have_maps(MAPS) or len(MAPS) < 2,
                           reason="needs two or three maps with their six cell-32 caches "
                                  "(MM_TEST_MAPS) and SURFCORE_DLL")


def _maxvel(m):
    return "3500" if ("unitfarmer2" in m.stem or "princessburglar" in m.stem) else "4000"


def _run(out, extra=(), env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    iters = os.environ.get("MM_REPLAY_ITERS", "6")
    cmd = [sys.executable, str(HERE / "mm_slot_replay_run.py"), str(out),
           ",".join(str(m) for m in MAPS), ",".join(_maxvel(m) for m in MAPS), iters]
    if extra:
        cmd += ["--extra", " ".join(extra)]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)


@needs
def test_every_slot_is_its_map_alone(tmp_path):
    r = _run(tmp_path / "rep")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    for k, m in enumerate(MAPS):
        assert f"=== slot {k} ({m.stem}) vs alone: rc 0" in out, out[-6000:]
    assert out.count("IDENTICAL") == len(MAPS), out[-6000:]


@needs
def test_the_replay_catches_a_slot_misalignment(tmp_path):
    """the harness is sensitive: every env's compass read off slot 0's field (an injected
    slot-0 leak, record side only) must fail the other slots"""
    r = _run(tmp_path / "mut", env_extra={"MM_MUTATE": "compass_slot0"})
    out = r.stdout + r.stderr
    assert "=== slot 0" in out, out[-4000:]
    assert any(f"=== slot {k} ({m.stem}) vs alone: rc 1" in out
               for k, m in enumerate(MAPS) if k), out[-6000:]
