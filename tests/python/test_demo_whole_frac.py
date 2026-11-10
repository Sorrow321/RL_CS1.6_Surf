"""--demo-whole-frac (azr1b1, 2026-11-10): a share W of the curriculum's draws is uniform over the
WHOLE spine [0, n-1]; the rest is drawn as before (front_frac of it on the band, the remainder on
[tau, n-1]). Florensa et al. 2017's replayed old starts for a curriculum that restarts at the end
of every new search line (azr1b1_pool1: one pool, the greedy lost the CP2 block whenever the
line's tail sat elsewhere).

(a) off (whole_frac 0, the default) the draws are byte-identical to the code before the flag.
(b) on, the pool holds round(W x demo draws) rows from the whole spine, rows behind tau among
    them, the band and the tail keep their split of the rest, and the map-start share is
    unchanged; draw(m) (the --demo-share path) does the same.
(c) the trainer: validation, config only when set, the recorder lists it TRAIN_ONLY.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE  # noqa: E402
from surfgym.respawn import DemoCurriculum  # noqa: E402


def _spine(n: int) -> np.ndarray:
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(n, dtype=np.float32) * 16.0    # distinct origins, row = x/16
    return s


def _start_pool(m: int) -> np.ndarray:
    p = np.zeros(m, STATE_DTYPE)
    p["origin"][:, 0] = -1000.0                                     # the map start, off the spine
    return p


def _rows(pool: np.ndarray) -> np.ndarray:
    x = pool["origin"][:, 0]
    return (x[x >= 0.0] / 16.0).astype(np.int64)


def _legacy_build_pool(c: DemoCurriculum, start_pool, pool_size=4096, fresh_frac=0.10):
    """build_pool exactly as it was before --demo-whole-frac (git 09099d2a)."""
    n_fresh = max(1, int(round(pool_size * fresh_frac)))
    n_demo = pool_size - n_fresh
    c._move()
    d_lo, d_hi = c._draw()
    if c.front_frac > 0.0:
        b_lo, b_hi = c._band()
        n_front = int(round(n_demo * c.front_frac))
        idx = np.concatenate([
            c.rng.integers(b_lo, b_hi + 1, n_front),
            c.rng.integers(d_lo, d_hi + 1, n_demo - n_front)])
    else:
        idx = c.rng.integers(d_lo, d_hi + 1, n_demo)
    fresh = start_pool[c.rng.integers(0, len(start_pool), n_fresh)]
    return np.concatenate([fresh, c.S[idx].copy()])


def test_off_is_byte_identical():
    sp, st = _spine(500), _start_pool(64)
    for kw in ({}, {"grow": 5, "front_frac": 0.5}, {"grow": 5}):
        a = DemoCurriculum(sp, window=5, seed=11, tau0=400, **kw)
        b = DemoCurriculum(sp, window=5, seed=11, tau0=400, **kw)
        for _ in range(5):
            pa = a.build_pool(st)
            pb = _legacy_build_pool(b, st)
            assert pa.tobytes() == pb.tobytes()
        # draw(m) as well: the old body is the build_pool draw without the fresh part
        c = DemoCurriculum(sp, window=5, seed=3, tau0=400, **kw)
        d = DemoCurriculum(sp, window=5, seed=3, tau0=400, whole_frac=0.0, **kw)
        assert c.draw(300).tobytes() == d.draw(300).tobytes()


def test_on_draws_over_the_whole_spine():
    n, tau = 1000, 900
    sp, st = _spine(n), _start_pool(64)
    c = DemoCurriculum(sp, window=5, seed=5, tau0=tau, grow=5, front_frac=0.5,
                       whole_frac=0.5, min_ep=1e9)       # min_ep: tau does not move
    pool = c.build_pool(st, pool_size=4096, fresh_frac=0.10)
    assert len(pool) == 4096
    n_fresh = int(round(4096 * 0.10))
    assert int((pool["origin"][:, 0] < 0).sum()) == n_fresh          # map starts unchanged
    rows = _rows(pool)
    n_demo = 4096 - n_fresh
    n_whole = int(round(n_demo * 0.5))
    behind = rows < tau
    # only the whole-spine draws can land behind tau; they spread over [0, n-1]
    assert 0.8 * n_whole * tau / n < behind.sum() < 1.2 * n_whole * tau / n
    assert rows[behind].min() < 100 and rows[behind].max() > 800
    # the band and the tail keep their split of the rest
    n_rest = n_demo - n_whole
    band = (rows >= tau) & (rows <= tau + 4)
    assert band.sum() >= int(round(n_rest * 0.5))


def test_draw_m_on():
    n, tau = 1000, 900
    c = DemoCurriculum(_spine(n), window=5, seed=9, tau0=tau, grow=5, whole_frac=0.25,
                       min_ep=1e9)
    rows = _rows(c.draw(800))
    assert len(rows) == 800
    assert 0.8 * 200 * tau / n < (rows < tau).sum() < 1.2 * 200 * tau / n


def test_bounds():
    import pytest
    with pytest.raises(ValueError):
        DemoCurriculum(_spine(10), whole_frac=1.0)
    with pytest.raises(ValueError):
        DemoCurriculum(_spine(10), whole_frac=-0.1)


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--demo-whole-frac", type=float, default=None' in src
    assert 'raise SystemExit("--demo-whole-frac must be in (0, 1)")' in src
    assert 'meta["config"]["demo_whole_frac"] = float(args.demo_whole_frac)' in src
    i = src.index('meta["config"]["demo_whole_frac"]')
    assert "if args.demo_file and args.demo_whole_frac is not None:" in src[i - 160:i]
    assert "**_dkw, **_wkw)" in src


def test_the_recorder_lists_it_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    block = src[src.index("TRAIN_ONLY = frozenset({"):]
    block = block[:block.index("})")]
    assert re.search(r'"demo_whole_frac"', block)
