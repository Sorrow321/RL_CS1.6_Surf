"""--maxvel V1,V2,... (mm/s0, 2026-11-16): one sv_maxvelocity per --maps map (then per
--heldout-maps map), each map server's physics; one value keeps every run as it was.

1. The pieces: the argument type, SurfFlag's per-env clamp, cfg_maxvel's lookup, the readers
   (record_ckpt, beam_tas) and the trainer's wiring (source).
2. CPU runs (the built core + maps with their prebaked caches; MAXVEL_TEST_MAP for the
   single-map run, MAXVEL_TEST_MAPS for the two-map run, SURFCORE_DLL):
   * one value is byte-identical to the parent commit (config, eval trajectories, progress
     rows), --obs-surf on so the surf flag's scalar path is compared too;
   * two maps at 3500 and 4000: each map's eval core runs its own value, the config records
     {tag: value}, and a list of the wrong length is refused.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


# --- 1. the pieces --------------------------------------------------------------------------------

def test_the_argument_is_a_float_or_a_tuple():
    import argparse
    from train_fast import _maxvel_arg
    v = _maxvel_arg("3500")
    assert isinstance(v, float) and v == 3500.0          # what type=float gave
    assert _maxvel_arg("3500,4000, 4000") == (3500.0, 4000.0, 4000.0)
    for bad in ("x", "3500,y", "3500,-1", "3500,0", ","):
        with pytest.raises(argparse.ArgumentTypeError):
            _maxvel_arg(bad)


def test_the_surf_flag_takes_a_clamp_per_env():
    from surfgym.surfflag import SurfFlag
    rng = np.random.default_rng(0)
    n = 64
    vmax = np.where(np.arange(n) < 40, 3500.0, 4000.0)
    both = SurfFlag(n, 24.5, 4, vmax=vmax)
    a = SurfFlag(40, 24.5, 4, vmax=3500.0)
    b = SurfFlag(24, 24.5, 4, vmax=4000.0)
    for _ in range(50):
        og = rng.choice([-1, 0], size=n, p=[0.8, 0.2])
        vz = rng.uniform(-4200.0, 300.0, size=n)        # crosses both clamps
        f = both.features(og, vz).copy()
        assert np.array_equal(f[:40], a.features(og[:40], vz[:40]))
        assert np.array_equal(f[40:], b.features(og[40:], vz[40:]))
    # a scalar keeps the scalar path; a wrong-length array is refused
    assert isinstance(SurfFlag(4, 1.0, 4, vmax=3500.0).vmax, float)
    with pytest.raises(ValueError):
        SurfFlag(4, 1.0, 4, vmax=np.ones(3))


def test_the_config_lookup():
    from surfgym.mapfleet import cfg_maxvel
    one = {"maxvel": 3500.0}
    assert cfg_maxvel(one, "maps/surf_src_cannonball.bsp") == 3500.0
    assert cfg_maxvel({}, "x.bsp") == 2000.0              # the old default
    per = {"maxvel": 3500.0, "maxvel_maps": {"unitfarmer2": 3500.0, "cannonball": 4000.0}}
    assert cfg_maxvel(per, "/a/b/surf_src_cannonball.bsp") == 4000.0
    assert cfg_maxvel(per, "surf_unitfarmer2") == 3500.0
    full = {"maxvel": 3500.0, "maxvel_maps": {"surf_x": 3000.0}}   # full stems after a collision
    assert cfg_maxvel(full, "maps/surf_x.bsp") == 3000.0
    with pytest.raises(KeyError):
        cfg_maxvel(per, "maps/surf_petrus_lite.bsp")


def test_the_readers_and_the_wiring():
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert 'cfg.get("maxvel_maps")' in rc and "_map_maxvel(cfg, map_path)" in rc
    bt = (ROOT / "tools" / "beam_tas.py").read_text(encoding="utf-8")
    assert "sv_maxvelocity=cfg_maxvel(cfg, map_path, 2000.0)" in bt
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    for s in ("sv_maxvelocity=SLOT_MAXVEL[_i],", "sv_maxvelocity=HELD_MAXVEL[_j],",
              '**({"maxvel_maps": dict(MAXVEL_MAPS)} if MAXVEL_MAPS else {})',
              "vmax=(float(_ph0.sv_maxvelocity) if len(set(_svm)) == 1 else"):
        assert s in src, s
    assert "sv_maxvelocity=args.maxvel" not in src


def test_record_ckpt_does_not_refuse_the_key():
    """the record gate's audit: a config key passes when it is TRAIN_ONLY or read by name"""
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"maxvel_maps"' in rc


# --- 2. CPU runs ----------------------------------------------------------------------------------

MAP = Path(os.environ.get("MAXVEL_TEST_MAP", str(ROOT / "maps" / "surf_unitfarmer2.bsp")))
MAPS = [Path(p) for p in os.environ.get(
    "MAXVEL_TEST_MAPS", ",".join(str(ROOT / "maps" / f"{s}.bsp") for s in
                                 ("surf_petrus_lite", "surf_edgeflow_blue200"))).split(",")]
DLL = Path(os.environ.get("SURFCORE_DLL", str(
    ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))))


def _have(m):
    return m.exists() and m.with_suffix(".goal_32.npz").exists()


needs_one = pytest.mark.skipif(not (_have(MAP) and DLL.exists()),
                               reason="needs the built core + a map with its prebaked caches "
                                      "(MAXVEL_TEST_MAP, SURFCORE_DLL)")
needs_two = pytest.mark.skipif(not (all(_have(m) for m in MAPS) and DLL.exists()),
                               reason="needs the built core + two maps with their prebaked "
                                      "caches (MAXVEL_TEST_MAPS, SURFCORE_DLL)")
BASE = ["--reward", "race", "--spawn", "platform",
        "--lidar-w", "16", "--lidar-h", "8", "--lidar-cell", "32", "--lidar-range", "11500",
        "--lidar-near", "2000", "--emb", "64", "--hidden", "64", "--act-every", "4",
        "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4", "--gamma", "0.9995",
        "--gae", "0.95", "--clip", "0.2", "--vf", "0.5", "--ent", "0.005", "--n-steps", "8",
        "--epochs", "1", "--minibatches", "2", "--ep-ticks", "600", "--time-pen", "0.005",
        "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
        "--race-dist", "geodesic", "--train-stride", "1", "--yaw-adaptive",
        "--respawn-frac", "0.9", "--respawn-margin", "10", "--respawn-reservoir", "1000",
        "--int-coef", "2.5", "--int-view", "0", "--int-speed", "8", "--ckpt-every", "1e9",
        "--eval-eps", "3", "--eval-greedy-only", "--obs-surf", "1", "--race-ratchet",
        "--no-compile", "--fp32", "--seed", "0"]


def _env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
                OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", SURFCORE_DLL=str(DLL))


def _train(root, run, extra, timeout=1800):
    shutil.rmtree(Path(root) / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-u", str(Path(root) / "python" / "train_fast.py"), "--run", run] \
        + BASE + extra
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(), cwd=str(root),
                          timeout=timeout)


def _base_tree(tmp_path):
    """<tmp>/base/python before the per-map --maxvel: the parent of the first commit that
    writes maxvel_maps (MAXVEL_BASE_REF overrides, e.g. fea35b06 before the commit exists)."""
    ref = os.environ.get("MAXVEL_BASE_REF")
    if not ref:
        r = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-S", "maxvel_maps",
                            "--", "python/train_fast.py"], capture_output=True, text=True)
        shas = r.stdout.split()
        if r.returncode != 0 or not shas:
            pytest.skip("the change is not committed yet; set MAXVEL_BASE_REF")
        ref = shas[-1] + "^"
    base = tmp_path / "base"
    base.mkdir(parents=True, exist_ok=True)
    tar = tmp_path / "base.tar"
    r = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "-o", str(tar), ref,
                        "python"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git archive {ref} failed: {r.stderr[-400:]}")
    with tarfile.open(tar) as tf:
        tf.extractall(base)
    assert "maxvel_maps" not in (base / "python" / "train_fast.py").read_text(encoding="utf-8")
    return base


def _rows(path, drop=("time/fps",)):
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    keep = [i for i, h in enumerate(head) if h not in drop]
    return [[r.split(",")[i] for i in keep] for r in lines[1:]]


@needs_one
def test_one_value_is_byte_identical_to_the_parent_commit(tmp_path):
    base = _base_tree(tmp_path)
    extra = ["--map", str(MAP), "--envs", "64", "--maxvel", "3500", "--steps", "6144",
             "--record-every", "2048"]
    r_old = _train(base, "maxvel_ctl", extra)
    assert r_old.returncode == 0, r_old.stdout[-3000:] + r_old.stderr[-3000:]
    r_new = _train(ROOT, "maxvel_ctl", extra)
    assert r_new.returncode == 0, r_new.stdout[-3000:] + r_new.stderr[-3000:]
    d_old, d_new = base / "runs" / "maxvel_ctl", ROOT / "runs" / "maxvel_ctl"
    c_old = json.loads((d_old / "run.json").read_text(encoding="utf-8"))["config"]
    c_new = json.loads((d_new / "run.json").read_text(encoding="utf-8"))["config"]
    assert "maxvel_maps" not in c_new and c_new == c_old and c_new["maxvel"] == 3500.0
    t_old, t_new = sorted(d_old.glob("traj_*.jsonl")), sorted(d_new.glob("traj_*.jsonl"))
    assert len(t_old) >= 3 and [p.name for p in t_old] == [p.name for p in t_new]
    for a, b in zip(t_old, t_new):
        assert a.read_bytes() == b.read_bytes(), f"{a.name} differs"
    assert _rows(d_old / "progress.csv") == _rows(d_new / "progress.csv")
    shutil.rmtree(d_new, ignore_errors=True)


def _phys_maxvel(path):
    with open(path, encoding="utf-8") as fh:
        hdr = json.loads(fh.readline())
    return float(hdr["phys"]["sv_maxvelocity"])


@needs_two
def test_two_maps_run_their_own_value():
    from surfgym.mapfleet import map_tags
    run = "maxvel_two"
    tags = map_tags([m.stem for m in MAPS])
    try:
        r = _train(ROOT, run, ["--maps", ",".join(str(m) for m in MAPS), "--envs", "64",
                               "--maxvel", "3500,4000", "--steps", "2048",
                               "--record-every", "2048"])
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert f"--maxvel per map: {tags[0]} 3500, {tags[1]} 4000" in out
        d = ROOT / "runs" / run
        cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
        assert cfg["maxvel"] == 3500.0
        assert cfg["maxvel_maps"] == {tags[0]: 3500.0, tags[1]: 4000.0}
        for tag, want in zip(tags, (3500.0, 4000.0)):
            files = sorted(d.glob(f"traj_*_{tag}.jsonl"))
            assert files, f"no eval file for {tag}"
            assert all(_phys_maxvel(f) == want for f in files)
        # record_ckpt (the record gate's and the dashboard's backend) records each map at its own
        ck = d / "ckpt_final.pt" if (d / "ckpt_final.pt").exists() else d / "ckpt_latest.pt"
        for m, want in zip(MAPS, (3500.0, 4000.0)):
            rec = d / f"rec_{m.stem}.jsonl"
            rr = subprocess.run([sys.executable, str(ROOT / "tools" / "record_ckpt.py"), str(ck),
                                 "--episodes", "1", "--ep-ticks", "200", "--out", str(rec),
                                 "--map", str(m)], capture_output=True, text=True, env=_env(),
                                cwd=str(ROOT), timeout=900)
            assert rr.returncode == 0 and rec.exists(), (rr.stdout + rr.stderr)[-3000:]
            assert _phys_maxvel(rec) == want, (m.stem, _phys_maxvel(rec))
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _train(ROOT, "maxvel_bad", ["--maps", ",".join(str(m) for m in MAPS), "--envs", "64",
                                      "--maxvel", "3500,4000,4000", "--steps", "2048"])
    assert bad.returncode != 0
    assert "--maxvel lists 3 values for 2 training map(s)" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "maxvel_bad", ignore_errors=True)


@needs_two
def test_beam_tas_builds_each_map_at_its_own_value():
    import beam_tas
    from surfgym.mapfleet import map_tags
    tags = map_tags([m.stem for m in MAPS])
    cfg = {"maxvel": 3500.0, "maxvel_maps": {tags[0]: 3500.0, tags[1]: 4000.0}}
    for m, want in zip(MAPS, (3500.0, 4000.0)):
        core = beam_tas.build_sim(cfg, str(m), 1, 100)
        assert float(core.config.phys.sv_maxvelocity) == want
    core = beam_tas.build_sim({"maxvel": 3500.0}, str(MAPS[1]), 1, 100)
    assert float(core.config.phys.sv_maxvelocity) == 3500.0         # one value: as before
