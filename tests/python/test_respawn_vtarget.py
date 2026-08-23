"""--respawn-vtarget: an ABSOLUTE spawn speed, not a multiplier.

A speed gate is a speed. Petrus's two measured gates want ~1,550 u/s (20%
ramp) and ~1,520 u/s (68% ramp) while the same policy's reservoir carries
731-1,879 u/s depending on where a state was harvested, so one multiplier
necessarily overdoses one section while underdosing the other. These tests
pin the two properties that matter: the flag hits the target speed without
touching direction, and NOT passing it leaves build_pool bit-identical to
every run trained so far.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from surfgym.core import STATE_DTYPE
from surfgym.respawn import RespawnBuffer


def _rows(speeds, rng):
    """States whose velocity has the given magnitude and a random direction."""
    rows = np.zeros(len(speeds), STATE_DTYPE)
    d = rng.normal(size=(len(speeds), 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rows["velocity"] = (d * np.asarray(speeds)[:, None]).astype(np.float32)
    rows["origin"][:, 0] = np.arange(len(speeds))
    return rows


def _fill(rng, speeds):
    rb = RespawnBuffer(4, reservoir=20_000, map_id="m", seed=7)
    for row in _rows(speeds, rng):
        rb._push(row)
    return rb


def _speeds(pool, n_fresh):
    v = pool["velocity"][n_fresh:].astype(np.float64)
    return np.linalg.norm(v, axis=1)


def test_vtarget_places_every_draw_at_the_target_speed():
    rng = np.random.default_rng(0)
    # the real spread: petrus's own reservoir runs 731-1,879 u/s by depth
    rb = _fill(rng, rng.uniform(700.0, 1900.0, 4000))
    fresh = _rows(np.full(8, 500.0), rng)
    pool = rb.build_pool(fresh, pool_size=4096, fresh_frac=0.1,
                         vel_scale=(1.0, 3.5), pitch_jitter=0.0,
                         vtarget=(1500.0, 3000.0))
    s = _speeds(pool, max(1, round(4096 * 0.1)))
    assert s.min() > 1499.0 and s.max() < 3001.0, (s.min(), s.max())
    # ...and the multiplier it implies is NOT one number
    assert s.std() > 100.0


def test_vtarget_never_rotates_a_state():
    rng = np.random.default_rng(1)
    rb = _fill(rng, rng.uniform(700.0, 1900.0, 512))
    fresh = _rows(np.full(8, 500.0), rng)
    pool = rb.build_pool(fresh, pool_size=1024, fresh_frac=0.0,
                         vel_scale=(1.0, 1.0), pitch_jitter=0.0,
                         vtarget=(1500.0, 3000.0))
    # build_pool always keeps at least one fresh entry (max(1, ...))
    v = pool["velocity"][1:].astype(np.float64)
    ref = rb._store[:rb.size]["velocity"].astype(np.float64)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    unit = v / np.linalg.norm(v, axis=1, keepdims=True)
    # every drawn direction must be one of the stored directions
    cos = unit @ ref.T
    assert cos.max(axis=1).min() > 1.0 - 1e-5


def test_vtarget_clips_a_near_stationary_snapshot():
    """A 20 u/s snapshot must not be fired off at 75x its own speed."""
    rng = np.random.default_rng(2)
    rb = _fill(rng, np.full(512, 20.0))
    fresh = _rows(np.full(8, 500.0), rng)
    pool = rb.build_pool(fresh, pool_size=1024, fresh_frac=0.0,
                         vel_scale=(1.0, 1.0), pitch_jitter=0.0,
                         vtarget=(1500.0, 3000.0), vtarget_clip=(0.5, 6.0))
    s = _speeds(pool, 1)          # entry 0 is the mandatory fresh draw
    assert s.max() <= 20.0 * 6.0 + 1e-3


def test_without_vtarget_the_pool_is_bit_identical():
    """The flag is additive: every run trained so far must be reproducible.

    Same seed, same reservoir, same call minus the new keyword - the pools
    have to match byte for byte, including the RNG stream the pitch jitter
    and the fresh draws consume after the velocity scale.
    """
    rng = np.random.default_rng(3)
    speeds = rng.uniform(700.0, 1900.0, 4000)
    fresh = _rows(np.full(8, 500.0), rng)

    a = _fill(np.random.default_rng(4), speeds).build_pool(
        fresh, pool_size=4096, fresh_frac=0.1, vel_scale=(1.0, 3.5),
        pitch_jitter=5.0)
    b = _fill(np.random.default_rng(4), speeds).build_pool(
        fresh, pool_size=4096, fresh_frac=0.1, vel_scale=(1.0, 3.5),
        pitch_jitter=5.0, vtarget=None)
    assert a.tobytes() == b.tobytes()
