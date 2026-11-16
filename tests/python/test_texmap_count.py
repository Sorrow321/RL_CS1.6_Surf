"""texmap.count_set (mm/s0, 2026-11-16): the face-id grid's set voxels counted in chunks instead of
`int((fid >= 0).sum())`, which built a bool and an int64 copy of the whole grid (5.6 GiB on
cannonball) for a log line. Same count, same message.

1. count_set == the one-shot expression on random grids, across chunk boundaries.
2. On a real map with a cached face-id grid (TEXMAP_TEST_MAP, SURFCORE_DLL; CPU): the
   TextureMap description is the same string as the parent commit's.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


@pytest.mark.parametrize("n, chunk", [(0, 4), (1, 4), (3, 4), (4, 4), (5, 4), (4 * 7 + 3, 4),
                                      (100_003, 1 << 22), (100_003, 4096)])
def test_the_count_is_the_one_shot_count(n, chunk):
    import torch
    from surfgym.texmap import count_set
    g = torch.Generator().manual_seed(n + chunk)
    fid = torch.randint(-3, 9, (n,), generator=g).to(torch.int16)
    assert count_set(fid, chunk) == int((fid >= 0).sum())
    assert count_set(fid.reshape(1, -1), chunk) == int((fid >= 0).sum())


def test_the_loader_uses_it():
    src = (ROOT / "python" / "surfgym" / "texmap.py").read_text(encoding="utf-8")
    assert "n_set = count_set(fid)" in src
    assert "int((fid >= 0).sum())" not in src.split("def count_set")[0]


MAP = Path(os.environ.get("TEXMAP_TEST_MAP", str(ROOT / "maps" / "surf_unitfarmer2.bsp")))
DLL = Path(os.environ.get("SURFCORE_DLL", str(
    ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))))
needs_map = pytest.mark.skipif(
    not (MAP.exists() and DLL.exists() and MAP.with_suffix(".faceid_32.npz").exists()
         and MAP.with_suffix(".sdf_32.npz").exists()),
    reason="needs the built core + a map with cached sdf_32 and faceid_32 (TEXMAP_TEST_MAP)")

PROBE = r"""
import sys
sys.path.insert(0, sys.argv[1])
from surfgym import SurfCore, default_config
from surfgym.vision import GpuLidar
core = SurfCore(sys.argv[2], default_config(num_envs=1, spawn_mode=2, lidar_w=0, lidar_h=0))
ld = GpuLidar(core, 16, 8, texture=True, texture_mode="normal", cell=32.0, device="cpu")
print("DESC", ld.texmap.describe())
"""


def _describe(python_dir):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", SURFCORE_DLL=str(DLL))
    r = subprocess.run([sys.executable, "-c", PROBE, str(python_dir), str(MAP)],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    return [ln for ln in r.stdout.splitlines() if ln.startswith("DESC ")][-1]


@needs_map
def test_the_description_is_the_parent_commits(tmp_path):
    ref = os.environ.get("TEXMAP_BASE_REF")
    if not ref:
        r = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-S", "count_set(",
                            "--", "python/surfgym/texmap.py"], capture_output=True, text=True)
        shas = r.stdout.split()
        if r.returncode != 0 or not shas:
            pytest.skip("the change is not committed yet; set TEXMAP_BASE_REF")
        ref = shas[-1] + "^"
    tar = tmp_path / "base.tar"
    r = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "-o", str(tar), ref,
                        "python"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git archive {ref} failed: {r.stderr[-400:]}")
    with tarfile.open(tar) as tf:
        tf.extractall(tmp_path / "base")
    old = _describe(tmp_path / "base" / "python")
    new = _describe(ROOT / "python")
    assert "face-id grid cached" in new and new == old
