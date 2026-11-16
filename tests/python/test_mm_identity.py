"""The multi-map build's two standing invariants (mm/s1), on recipe v14 at CPU smoke scale:

1. a single-map v14 run is byte-identical to fea35b06 (config dump, every eval trajectory,
   progress rows minus time/fps) - the guarantee every mm/s1 commit keeps;
2. `--maps A` (one map in the list) is byte-identical to `--map A`.

Needs the built core and a map with all six cell-32 caches (V14_SMOKE_MAP, default
maps/surf_unitfarmer2.bsp; SURFCORE_DLL). V14_BASE_TREE reuses an extracted fea35b06 tree.
The search never starts here (tests/python/v14_smoke.py: identity runs need no async timing).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v14_smoke as V  # noqa: E402

needs = pytest.mark.skipif(not V.have_map(), reason="needs the built core + a map with its six "
                                                     "cell-32 caches (V14_SMOKE_MAP)")


@needs
def test_single_map_v14_is_byte_identical_to_fea35b06(tmp_path):
    base = V.base_tree(tmp_path / "base")
    r0, d0 = V.run(base, "v14id", ["--map", str(V.MAP)])
    assert r0.returncode == 0, (r0.stdout + r0.stderr)[-4000:]
    r1, d1 = V.run(V.ROOT, "v14id", ["--map", str(V.MAP)])
    assert r1.returncode == 0, (r1.stdout + r1.stderr)[-4000:]
    try:
        V.same_run(d0, d1, drop_cfg=("demo_file",))      # the spine path names the checkout
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        if not V.os.environ.get("V14_BASE_TREE"):
            shutil.rmtree(d0, ignore_errors=True)


@needs
def test_maps_with_one_map_is_the_map_run(tmp_path):
    r0, d0 = V.run(V.ROOT, "v14one", ["--map", str(V.MAP)])
    assert r0.returncode == 0, (r0.stdout + r0.stderr)[-4000:]
    keep = tmp_path / "map_run"
    shutil.move(str(d0), str(keep))
    r1, d1 = V.run(V.ROOT, "v14one", ["--maps", str(V.MAP)])
    assert r1.returncode == 0, (r1.stdout + r1.stderr)[-4000:]
    try:
        V.same_run(keep, d1)
    finally:
        shutil.rmtree(d1, ignore_errors=True)
