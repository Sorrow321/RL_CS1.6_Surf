"""Stage 1 of the multi-map build (mm/s1): v14's refused parts, per map. CPU runs of the v14 line
on two maps (MM_TEST_MAPS, default maps/surf_edgeflow_blue200 + surf_unitfarmer2, each with its
six cell-32 caches; SURFCORE_DLL), with the families not ported yet stripped
(tests/python/v14_smoke.py FAMILIES). The single-map identity is tests/python/test_mm_identity.py.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import v14_smoke as V  # noqa: E402
from surfgym.mapfleet import map_tags  # noqa: E402

needs_mm = pytest.mark.skipif(not V.have_maps(), reason="needs two maps with their six "
                                                        "cell-32 caches (MM_TEST_MAPS)")
# the families still refused with --maps; each S1 item removes its own
NOT_YET = ["front_temp", "demo_front_temp", "greedy_temp", "energy_front", "int_gate",
           "unstuck", "front_perturb", "greedy", "loop"]
MAPS_ARG = ["--maps", ",".join(str(m) for m in V.MM_MAPS)]


def _headers(path):
    hs = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("{") and '"map"' in ln:
                hs.append(json.loads(ln))
    return hs


@needs_mm
def test_a_eval_parallel_records_every_map_side_by_side():
    r, d = V.run_line(V.ROOT, "mms1_a", MAPS_ARG, drop=NOT_YET, steps=4096)
    try:
        assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
        for m, tag in zip(V.MM_MAPS, map_tags([m.stem for m in V.MM_MAPS])):
            files = sorted(d.glob(f"traj_*_{tag}.jsonl"))
            assert files, f"no eval of {tag}"
            for f in files:
                hs = _headers(f)
                assert len(hs) == 2 and all(h["map"] == m.stem for h in hs), f
                assert all(h.get("eval_parallel") == 1 for h in hs), f
    finally:
        shutil.rmtree(d, ignore_errors=True)


@needs_mm
def test_c_the_novelty_gate_and_its_share_cap_per_map():
    import torch
    tags = map_tags([m.stem for m in V.MM_MAPS])
    drop = [f for f in NOT_YET if f != "int_gate"]
    r, d = V.run_line(V.ROOT, "mms1_c", MAPS_ARG, drop=drop, steps=4096)
    try:
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        for tag in tags:
            assert f"--int-coef-progress[{tag}]: greedy frontier" in out, tag
        assert "share-cap m " + ", ".join(f"{t} off" for t in tags) in out
        ck = torch.load(d / "ckpt_latest.pt", map_location="cpu", weights_only=False)
        assert sorted(ck["int_coef_progress"]) == sorted(tags)
        assert sorted(ck["int_share_max"]) == sorted(tags)
        assert all(set(v) == {"progress", "coef"} for v in ck["int_coef_progress"].values())
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_d_reservoir_depth_per_map():
    """MapFleet.reservoir_min_depths: each slot's deepest reservoir state over ITS d0 (the
    per-map band edges of --energy-pot-front / --front-temp / --unstuck), while
    reservoir_min_depth keeps the fleet minimum"""
    import numpy as np
    from surfgym.mapfleet import MapFleet, MapSlot

    class _Core:
        obs_dim = 15

        def __init__(self, n):
            self.num_envs = n

    class _Res:
        def __init__(self, d):
            self._d = np.asarray(d, np.float64)
            self.size = len(d)

    slots = []
    for i, (n, d, d0) in enumerate([(4, [900.0, 300.0, 500.0], 1000.0),
                                    (4, [20000.0, 15000.0], 40000.0),
                                    (4, [], 5000.0)]):
        s = MapSlot(f"m{i}", f"m{i}.bsp", _Core(n), 4 * i, 4 * i + 4)
        s.respawn, s.rf_d0 = _Res(d), d0
        slots.append(s)
    f = MapFleet(slots)
    got = f.reservoir_min_depths()
    assert got[0] == 300.0 / 1000.0 and got[1] == 15000.0 / 40000.0 and got[2] != got[2]
    assert f.reservoir_min_depth() == 0.3


@needs_mm
def test_d_the_energy_band_runs_per_map():
    drop = [f for f in NOT_YET if f not in ("int_gate", "energy_front")]
    r, d = V.run_line(V.ROOT, "mms1_d", MAPS_ARG, drop=drop, steps=4096)
    try:
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--energy-pot-front 0.01: the energy term is paid only until" in out
    finally:
        shutil.rmtree(d, ignore_errors=True)


@needs_mm
def test_e_the_front_temperature_band_per_map():
    drop = [f for f in NOT_YET if f not in ("int_gate", "energy_front", "front_temp")]
    r, d = V.run_line(V.ROOT, "mms1_e", MAPS_ARG, drop=drop, steps=4096)
    try:
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--front-temp 1: an env whose progress is >=" in out
        assert any(ln.startswith("step ") and " fronttemp " in ln for ln in out.splitlines())
    finally:
        shutil.rmtree(d, ignore_errors=True)
    bad = V.run_line(V.ROOT, "mms1_e2", MAPS_ARG, drop=drop, steps=2048,
                     extra=["--front-contact", "1", "--ratchet-contact", "1"])[0]
    assert bad.returncode != 0 and "--maps with --front-contact" in (bad.stdout + bad.stderr)
    shutil.rmtree(V.ROOT / "runs" / "mms1_e2", ignore_errors=True)


@needs_mm
def test_f_g_greedy_and_front_perturb_rows_per_map():
    import torch
    tags = map_tags([m.stem for m in V.MM_MAPS])
    drop = [f for f in NOT_YET if f not in ("int_gate", "energy_front", "front_temp",
                                            "greedy", "greedy_temp", "front_perturb")]
    r, d = V.run_line(V.ROOT, "mms1_fg", MAPS_ARG, drop=drop, steps=6144)
    try:
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        for tag in tags:
            assert f"[{tag}] greedy-line pool: " in out, tag      # each map's pool, its own eval
        head = (d / "progress.csv").read_text().splitlines()[0].split(",")
        assert all(f"fpert/ended.{t}" in head for t in tags)
        assert any(ln.startswith("step ") and " greedy " in ln and "," in ln.split(" greedy ")[1]
                   for ln in out.splitlines())
        ck = torch.load(d / "ckpt_latest.pt", map_location="cpu", weights_only=False)
        assert sorted(ck["respawn_greedy"]) == sorted(tags)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_f_one_greedy_pool_per_map_seeded_like_the_reservoir():
    src = (V.ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert "GREEDY = _greedy_pool(int(args.seed) + 9191)" in src
    assert "_greedy_pool(int(args.seed) + 9191 + 101 * _gi)" in src
    assert "GREEDY_S[_s.tag].build(" in src and "GREEDY_S[_s.tag].hook(N_EVAL_PAR)" in src


@needs_mm
def test_h_unstuck_per_map_with_per_row_entropy():
    import torch
    tags = map_tags([m.stem for m in V.MM_MAPS])
    drop = [f for f in NOT_YET if f not in ("int_gate", "energy_front", "front_temp",
                                            "greedy", "greedy_temp", "front_perturb",
                                            "unstuck")]
    r, d = V.run_line(V.ROOT, "mms1_h", MAPS_ARG, drop=drop, steps=12288)
    try:
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        head = (d / "progress.csv").read_text().splitlines()[0].split(",")
        for t in tags:
            assert all(f"unstuck/{c}.{t}" in head for c in ("T", "stuck_steps", "best")), t
        steps = [ln for ln in out.splitlines() if ln.startswith("step ")]
        assert steps and all(f"{tags[0]} T " in ln and f"{tags[1]} T " in ln for ln in steps)
        ck = torch.load(d / "ckpt_latest.pt", map_location="cpu", weights_only=False)
        assert sorted(ck["unstuck"]) == sorted(tags)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_h_the_entropy_weight_is_per_row_only_under_maps():
    src = (V.ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert "ENT_ROW = bool(MULTI and UNSTUCK_ENT)" in src
    assert "el = (-(ent_w_env[idx % N] * ent).mean() if ENT_ROW else -ent.mean())" in src
    assert "return global_step if not MULTI else (int(global_step) * slots[si].n) // N" in src
