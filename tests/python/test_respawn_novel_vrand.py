"""--respawn-novel-vrand F (2026-10-22, autoresearch r5b2): F of the novel rows are the reservoir's
own FRONTIER states kept in place with a random velocity direction (exploring starts in velocity
where the line ends; surfgym.respawn.NovelSpawnSampler.sample_vrand).

* F = 0 (the default) is the sampler as shipped: mix() returns the same pool, byte for byte, and
  the moved rows draw from the same generator;
* F > 0: round(F x n) of the n novel rows are frontier rows - their origins are reservoir states
  within vrand_band x d0 of the reservoir's deepest state, their speeds are speeds of such states,
  their directions are not the sources' (a uniform draw on the sphere); the rest are moved rows;
  every row's origin is a novel key, so note_spawns tags the episodes novel (never harvested);
* the view yaw turns with the heading (the offset from it is kept);
* bad values are refused, and the recorder lists both flags as training-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.core import STATE_DTYPE                               # noqa: E402
from surfgym.respawn import NovelSpawnSampler                      # noqa: E402

from test_respawn_novel_turn import core, field, needs_sid, reservoir  # noqa: E402,F401

D0 = 97768.0


def _sampler(core, field, reservoir, **kw):
    s = NovelSpawnSampler(core, field, 4, seed=5, d0=D0, **kw)
    store = reservoir._store[:reservoir.size]
    s.d_best = float(np.quantile(field.sample(np.asarray(store["origin"], np.float64)), 0.2))
    return s


def _pool(reservoir, n=600):
    pool = np.zeros(n, STATE_DTYPE)
    pool["origin"][:, 0] = np.arange(n, dtype=np.float32)
    return pool


@needs_sid
def test_off_is_the_shipped_sampler(core, field, reservoir):
    a = _sampler(core, field, reservoir)
    b = _sampler(core, field, reservoir, vrand=0.0, vrand_band=0.05)
    pa = a.mix(_pool(reservoir), 100, 300, reservoir)
    pb = b.mix(_pool(reservoir), 100, 300, reservoir)
    assert pa.tobytes() == pb.tobytes()
    assert a._keys == b._keys
    assert b.vrand_rng is None and b.vrand_kept == 0


@needs_sid
def test_frontier_rows_keep_place_and_speed_scale_and_are_novel(core, field, reservoir):
    band = 0.05
    s = _sampler(core, field, reservoir, vrand=0.5, vrand_band=band)
    out = s.mix(_pool(reservoir), 100, 300, reservoir)
    new = out[100:400]
    assert len(new) == 300
    store = reservoir._store[:reservoir.size]
    d_all = np.asarray(field.sample(np.asarray(store["origin"], np.float64)), np.float64)
    lim = field.reach_max - 0.5 * field.cell
    front = (d_all < lim) & (d_all <= d_all[d_all < lim].min() + band * D0)
    keys_front = {o.tobytes() for o in np.asarray(store["origin"][front], np.float32)}
    spd_front = np.linalg.norm(np.asarray(store["velocity"][front], np.float64), axis=1)
    vr = new[150:]                                  # the vrand rows follow the moved rows
    assert s._last_vrand == 150 and s.vrand_kept == 150
    for o in np.asarray(vr["origin"], np.float32):
        assert o.tobytes() in keys_front              # kept in place, a frontier state
    sp = np.linalg.norm(np.asarray(vr["velocity"], np.float64), axis=1)
    assert np.all(np.min(np.abs(sp[:, None] - spd_front[None, :]), axis=1) < 1e-2)
    # directions: not the sources' - spread over the sphere (both signs of vz, all headings)
    u = np.asarray(vr["velocity"], np.float64) / np.maximum(sp[:, None], 1e-9)
    assert (u[:, 2] > 0.3).any() and (u[:, 2] < -0.3).any()
    assert np.ptp(np.degrees(np.arctan2(u[:, 1], u[:, 0]))) > 300.0
    assert np.all(vr["onground"] == -1)
    # every row is a novel key: their episodes are tagged novel, never harvested
    st = np.zeros(4, STATE_DTYPE)
    st["origin"][:] = vr["origin"][:4]
    s.note_spawns(st, np.ones(4, bool))
    assert s.novel_env.all()
    assert "vrand rows" in s.line()


@needs_sid
def test_view_yaw_turns_with_the_heading(core, field, reservoir):
    s = _sampler(core, field, reservoir, vrand=1.0, vrand_band=0.05)
    store = reservoir._store[:reservoir.size]
    src = {o.tobytes(): i for i, o in enumerate(np.asarray(store["origin"], np.float32))}
    rows = s.sample_vrand(200, reservoir)
    for r in rows:
        b = store[src[np.asarray(r["origin"], np.float32).tobytes()]]
        v0, v1 = np.asarray(b["velocity"], np.float64), np.asarray(r["velocity"], np.float64)
        if np.hypot(v0[0], v0[1]) < 1.0 or np.hypot(v1[0], v1[1]) < 1.0:
            continue
        off0 = (float(b["yaw"]) - np.degrees(np.arctan2(v0[1], v0[0]))) % 360.0
        off1 = (float(r["yaw"]) - np.degrees(np.arctan2(v1[1], v1[0]))) % 360.0
        assert min(abs(off0 - off1), 360.0 - abs(off0 - off1)) < 1e-2


@needs_sid
def test_speed_mode_keeps_direction_and_scales_speed(core, field, reservoir):
    band = 0.05
    s = _sampler(core, field, reservoir, vrand=1.0, vrand_band=band, vrand_speed=(1.0, 1.5))
    store = reservoir._store[:reservoir.size]
    src = {o.tobytes(): i for i, o in enumerate(np.asarray(store["origin"], np.float32))}
    d_all = np.asarray(field.sample(np.asarray(store["origin"], np.float64)), np.float64)
    lim = field.reach_max - 0.5 * field.cell
    dmin = d_all[d_all < lim].min()
    rows = s.sample_vrand(300, reservoir)
    assert len(rows) == 300 and "speed x 1-1.5" in s.line()
    ratios = []
    for r in rows:
        i = src[np.asarray(r["origin"], np.float32).tobytes()]
        assert d_all[i] <= dmin + band * D0
        v0 = np.asarray(store["velocity"][i], np.float64)
        v1 = np.asarray(r["velocity"], np.float64)
        n0 = np.linalg.norm(v0)
        if n0 < 1.0:
            continue
        k = np.linalg.norm(v1) / n0
        ratios.append(k)
        assert np.allclose(v1, v0 * k, atol=1e-2 * n0)       # same direction
        assert float(r["yaw"]) == float(store["yaw"][i])    # view unchanged
    ratios = np.asarray(ratios)
    assert ratios.min() >= 1.0 - 1e-4 and ratios.max() <= 1.5 + 1e-4 and np.ptp(ratios) > 0.3
    with pytest.raises(ValueError):
        NovelSpawnSampler(core, field, 4, d0=D0, vrand=0.3, vrand_speed=(1.5, 1.0))


@needs_sid
def test_refusals(core, field):
    with pytest.raises(ValueError):
        NovelSpawnSampler(core, field, 4, d0=D0, vrand=1.5)
    with pytest.raises(ValueError):
        NovelSpawnSampler(core, field, 4, d0=D0, vrand=0.3, vrand_band=0.0)
    with pytest.raises(ValueError):
        NovelSpawnSampler(core, field, 4, d0=0.0, vrand=0.3)


def test_recorder_lists_the_flags_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"respawn_novel_vrand", "respawn_novel_vrand_band", "respawn_novel_vrand_speed"' in src


def test_the_trainer_wires_and_restores_it():
    import json
    import re
    import shutil
    from test_ez_viewc import SMOKE, TRAIN, _run, needs_run  # noqa: F401
    from test_respawn_greedy import _seeded_ckpt
    if not (ROOT / "build" / "libsurfcore.so").exists() and not (ROOT / "build" / "surfcore.dll").exists():
        pytest.skip("needs the built core")
    d0, seeded = _seeded_ckpt("vrand_base")
    run = "vrand_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)] + SMOKE
             + ["--steps", "10240", "--respawn-reservoir", "5000", "--respawn-novel",
                "--respawn-novel-vrand", "0.5", "--respawn-novel-vrand-band", "0.2"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "50% of the novel rows are frontier states (within 0.2 x d0" in r.stdout
    assert re.search(r"[0-9,]+ vrand rows \(frontier states, random velocity\)", r.stdout)
    cfg = json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["respawn_novel_vrand"] == 0.5 and cfg["respawn_novel_vrand_band"] == 0.2
    run2 = "vrand_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE
              + ["--steps", "14336", "--respawn-reservoir", "5000"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "50% of the novel rows are frontier states" in r2.stdout
    # off: no key in the config dump (a control run's dump is unchanged)
    run3 = "vrand_off"
    shutil.rmtree(ROOT / "runs" / run3, ignore_errors=True)
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", run3, "--ckpt", str(seeded)] + SMOKE
              + ["--steps", "6144", "--respawn-reservoir", "5000", "--respawn-novel"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    cfg3 = json.loads((ROOT / "runs" / run3 / "run.json").read_text(encoding="utf-8"))["config"]
    assert "respawn_novel_vrand" not in cfg3 and "vrand rows" not in r3.stdout
    # the speed mode: own direction, speed x U(LO, HI)
    run4 = "vrand_spd"
    shutil.rmtree(ROOT / "runs" / run4, ignore_errors=True)
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", run4, "--ckpt", str(seeded)] + SMOKE
              + ["--steps", "6144", "--respawn-reservoir", "5000", "--respawn-novel",
                 "--respawn-novel-vrand", "0.5", "--respawn-novel-vrand-band", "0.2",
                 "--respawn-novel-vrand-speed", "1.0", "1.5"])
    assert r4.returncode == 0, r4.stdout[-4000:] + r4.stderr[-4000:]
    assert "with their own velocity x U(1, 1.5)" in r4.stdout
    assert re.search(r"[0-9,]+ vrand rows \(frontier states, speed x 1-1.5\)", r4.stdout)
    cfg4 = json.loads((ROOT / "runs" / run4 / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg4["respawn_novel_vrand_speed"] == [1.0, 1.5]
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2, ROOT / "runs" / run3,
               ROOT / "runs" / run4):
        shutil.rmtree(dd, ignore_errors=True)
