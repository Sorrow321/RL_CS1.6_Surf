"""The entity tag (mm/s1): a map with more solid brush entities than the pre-6bd7f931 core's
128-entry table gets "_e1" on every cache sampled from the core's solid set, so its grids built
by that core (princessburglar_b26: four func_walls missing) miss and rebuild; every other map's
signatures are unchanged. Needs a staging dir with the maps' caches (ENTITY_TAG_MAPS, flat
<stem>.* files, e.g. the seven maps' tarballs unpacked and restamped) and the built core.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

STAGE = Path(os.environ.get("ENTITY_TAG_MAPS", str(ROOT / "maps")))
DLL = Path(os.environ.get("SURFCORE_DLL", str(ROOT / "build" / "libsurfcore.so")))
PB, UF2 = "surf_princessburglar_b26", "surf_unitfarmer2"


def _have(stem, kinds=("goal", "goalc", "sdf", "occ", "slabocc", "faceid")):
    return DLL.exists() and (STAGE / f"{stem}.bsp").exists() and all(
        (STAGE / f"{stem}.{k}_32.npz").exists() for k in kinds)


def _core(stem):
    from surfgym.core import SurfCore, default_config
    return SurfCore(str(STAGE / f"{stem}.bsp"),
                    default_config(num_envs=1, spawn_mode=2, lidar_w=0, lidar_h=0))


@pytest.mark.skipif(not _have(UF2), reason="needs unitfarmer2 staged with its caches")
def test_a_map_under_the_old_table_keeps_every_signature():
    import mm_assets
    from surfgym.vision import entity_tag, solid_entity_count
    assert solid_entity_count(STAGE / f"{UF2}.bsp") == 7 and entity_tag(_core(UF2)) == ""
    r = mm_assets.check_map(STAGE, UF2)
    assert r["ok"] and all(v == "OK" for v in r["files"].values()), r


@pytest.mark.skipif(not _have(PB), reason="needs princessburglar staged with its caches")
def test_princessburglars_old_core_grids_now_miss():
    import mm_assets
    from surfgym.vision import entity_tag, solid_entity_count
    assert solid_entity_count(STAGE / f"{PB}.bsp") == 135 and entity_tag(_core(PB)) == "_e1"
    r = mm_assets.check_map(STAGE, PB)
    assert not r["ok"]
    # the six caches carry the old signatures: every one misses and is rebuilt by the fixed core
    assert all(v.startswith("MISS") for v in r["files"].values()), r["files"]


def test_every_core_sampled_cache_appends_the_tag():
    v = (ROOT / "python" / "surfgym" / "vision.py").read_text(encoding="utf-8")
    g = (ROOT / "python" / "surfgym" / "goalfield.py").read_text(encoding="utf-8")
    assert 'sig = f"{_map_sig(bsp)}_{_OCC_SEMANTICS}{entity_tag(core)}"' in v
    assert v.count("{_SDF_SEMANTICS}{pane_tag(core)}{entity_tag(core)}") == 2     # slab, sdf
    assert 'f"{entity_tag(core)}",' in v                                           # faceid
    assert "+ pane_tag(core)\n           + entity_tag(core))" in g                 # goal fields
    assert '+ "_".join(f"{v:g}" for v in box) + entity_tag(core))' in g            # ground field
