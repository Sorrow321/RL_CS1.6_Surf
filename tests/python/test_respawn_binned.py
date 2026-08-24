"""S1 (progress-binned respawn sampling) + S2 (kill-zone extraction)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from surfgym.core import STATE_DTYPE
from surfgym.respawn import RespawnBuffer
from surfgym import zones

ROOT = Path(__file__).resolve().parents[2]


def _rows(xs):
    rows = np.zeros(len(xs), STATE_DTYPE)
    rows["origin"][:, 0] = xs
    return rows


def _dist(o):
    return np.asarray(o)[:, 0]


def test_binned_pool_is_flat_over_occupied_bins():
    # visitation skew: 90% of the reservoir sits in one distance decile
    rb = RespawnBuffer(4, reservoir=20_000, map_id="m",
                       dist_fn=_dist, dist_max=1000.0, bins=10)
    xs = np.concatenate([np.full(9000, 50.0),
                         np.random.default_rng(0).uniform(100, 1000, 1000)])
    for k, row in enumerate(_rows(xs)):
        rb._push(row, float(xs[k]))
    pool = rb.build_pool(_rows(np.zeros(8)), pool_size=4096, fresh_frac=0.1,
                         vel_scale=(1.0, 1.0), pitch_jitter=0.0)
    d = pool["origin"][:, 0]
    re = d[d > 0]                                  # drop the fresh entries
    # every non-fresh draw must land on a REAL state (an unsliced _d read
    # would draw zeroed slots that this filter silently hides)
    assert len(re) == 4096 - max(1, round(4096 * 0.1))
    hist, _ = np.histogram(re, bins=np.linspace(0.0, 1000.0, 11))
    occ = hist[hist > 0]
    assert len(occ) == 10                          # all deciles occupied
    assert occ.min() > 0.85 * occ.mean()           # actually flat
    # the dominant decile must no longer dominate the pool
    assert (re < 100.0).sum() < 0.35 * len(re)


def test_binned_cap_limits_frontier_cloning():
    # a 4-state frontier bin must not be cloned into a quarter of the pool
    rb = RespawnBuffer(2, reservoir=10_000, map_id="m",
                       dist_fn=_dist, dist_max=1000.0, bins=10)
    xs = np.concatenate([np.full(5000, 50.0), np.full(4, 950.0)])
    for k, row in enumerate(_rows(xs)):
        rb._push(row, float(xs[k]))
    pool = rb.build_pool(_rows(np.zeros(8)), pool_size=4096, fresh_frac=0.1,
                         vel_scale=(1.0, 1.0), pitch_jitter=0.0)
    d = pool["origin"][:, 0]
    assert (d > 900.0).sum() <= 4 * 4              # 4x-population draw cap


def test_uniform_path_unchanged_without_dist_fn():
    rb = RespawnBuffer(2, reservoir=1000, map_id="m")
    for row in _rows(np.linspace(1, 999, 500)):
        rb._push(row)
    pool = rb.build_pool(_rows(np.zeros(8)), pool_size=256, fresh_frac=0.1,
                         vel_scale=(1.0, 1.0), pitch_jitter=0.0)
    assert len(pool) == 256 and rb._d is None


def test_load_state_dict_recomputes_distances():
    # F2-era payloads predate the distance column: d must be recomputed
    src = RespawnBuffer(2, reservoir=100, map_id="m")
    for row in _rows(10.0 * np.arange(50)):
        src._push(row)
    dst = RespawnBuffer(2, reservoir=100, map_id="m",
                        dist_fn=_dist, dist_max=1000.0)
    dst.load_state_dict(src.state_dict())
    assert dst.size == 50
    assert np.allclose(dst._d[:50], 10.0 * np.arange(50))


def test_observe_keeps_d_aligned_with_store():
    # the harvest batching is the only path that fills _d during training;
    # drive the REAL observe() loop across ring wraparound and multi-env
    # same-tick endings, then require exact _d/_store alignment
    rng = np.random.default_rng(7)
    rb = RespawnBuffer(4, reservoir=97, margin_ticks=20, snap_every=5,
                       map_id="m", dist_fn=_dist, dist_max=1000.0)
    states = np.zeros(4, STATE_DTYPE)
    for _ in range(3000):
        states["origin"][:, 0] = rng.uniform(1.0, 999.0, 4)
        ended = rng.random(4) < 0.02
        stag = rng.random(4) < 0.1
        rb.observe(states, ended, stagnant=stag)
    rb.flush_harvest()      # the trainer drains once per iteration
    assert rb.size > 0
    assert np.array_equal(rb._store[:rb.size]["origin"][:, 0],
                          rb._d[:rb.size])


def test_sentinel_states_never_sampled_by_binned_path():
    rb = RespawnBuffer(2, reservoir=1000, map_id="m", dist_fn=_dist,
                       dist_max=1000.0, dist_valid_max=1100.0, bins=10)
    xs = np.concatenate([np.full(200, 500.0), np.full(300, 1500.0)])  # 300 sentinels
    for k, row in enumerate(_rows(xs)):
        rb._push(row, float(xs[k]))
    pool = rb.build_pool(_rows(np.zeros(8)), pool_size=512, fresh_frac=0.1,
                         vel_scale=(1.0, 1.0), pitch_jitter=0.0)
    d = pool["origin"][:, 0]
    assert (d > 1100.0).sum() == 0


def test_kill_rule_boundaries():
    models = [([0, 0, 0], [10, 10, 10])]
    def hurt(dmg):
        return [{"classname": "trigger_hurt", "model": "*0", "dmg": dmg}]
    assert not zones._kill_entities(hurt("89"), models)
    assert zones._kill_entities(hurt("90"), models)
    assert not zones._kill_entities(hurt("garbage"), models)   # kv_float parity
    # destless teleport inert; destful deadly
    tp = [{"classname": "trigger_teleport", "model": "*0", "target": "x"}]
    assert not zones._kill_entities(tp, models)
    tp2 = tp + [{"classname": "info_teleport_destination", "targetname": "x"}]
    assert zones._kill_entities(tp2, models)


def test_hull_probe_is_tighter_than_aabb():
    bsp = ROOT / "maps" / "surf_src_cannonball.bsp"
    kz = zones.kill_zones(bsp)
    net = next(k for k in kz if k["model"] == "*30")
    contains = zones.hull_probe(bsp)
    rng = np.random.default_rng(3)
    mn, mx = np.asarray(net["mins"]), np.asarray(net["maxs"])
    pts = rng.uniform(mn, mx, (20000, 3))
    inside = contains(30, pts - np.asarray(net["origin"]))
    frac = inside.mean()
    assert 0.0 < frac < 1.0     # the hull is neither empty nor the full box
    # points far outside the AABB are never inside
    far = pts + (mx - mn) * 3.0
    assert not contains(30, far - np.asarray(net["origin"])).any()
    kz = zones.kill_zones(ROOT / "maps" / "surf_src_cannonball.bsp")
    assert kz, "cannonball has fail nets; extraction found none"
    models = {k["model"] for k in kz}
    # the wall-2 fail net: trigger_teleport *30 -> mapstart
    assert "*30" in models
    for k in kz:
        assert k["class"] in ("trigger_teleport", "trigger_hurt")
        assert all(a < b for a, b in zip(k["mins"], k["maxs"]))
    # destless pads (inert in GoldSrc and in src/env.c) must be excluded
    ents, _ = zones.parse_bsp(ROOT / "maps" / "surf_src_cannonball.bsp")
    dests = {e.get("targetname") for e in ents if e.get("targetname")}
    destless = [e for e in ents
                if e.get("classname") == "trigger_teleport"
                and e.get("target") not in dests]
    assert not ({e.get("model") for e in destless} & models)
