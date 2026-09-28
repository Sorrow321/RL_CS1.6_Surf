"""--vision-clip (2026-09-28): the depth image of the PLAYER's collision geometry. hlcsg compiles
CLIP brushes into the player hulls only, so the point-hull vision grid showed a clip-brush ramp as
open air while the player surfed it (surf_src_utopia's curved ramp at x 4,600-8,700, where the
jt3ANCHU finisher lands at 44.9 s; most ramps of surf_src_kairo_b2, surf_hamburglar_love,
surf_gi_rino, surf_src_raphaello). core.occupancy_grid(player=True): 1 = the point grid exactly,
2 = player-hull-only solid; a map with none gets the identical grid."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import SurfCore, default_config  # noqa: E402
from surfgym import vision  # noqa: E402

PETRUS = ROOT / "maps" / "surf_petrus_lite.bsp"
UTOPIA = ROOT / "maps_pool" / "surf_src_utopia.bsp"


def _core(bsp):
    core = SurfCore(str(bsp), default_config(num_envs=1))
    if getattr(core._lib, "surf_occupancy_grid_player", None) is None:
        pytest.skip("this surfcore build predates surf_occupancy_grid_player")
    return core


@pytest.mark.skipif(not PETRUS.exists(), reason="petrus_lite not present")
def test_no_player_only_solid_means_the_identical_grid(tmp_path):
    core = _core(PETRUS)
    cell = vision.pick_cell(core)
    mins, nx, ny, nz = vision.grid_dims(core, cell)
    pt = core.occupancy_grid(mins, cell, nx, ny, nz)
    pl = core.occupancy_grid(mins, cell, nx, ny, nz, player=True)
    assert np.array_equal(pl == 1, pt == 1)           # bit 0 IS the point grid
    assert int((pl == 2).sum()) == 0                  # petrus has no CLIP-only solid
    # the slab bake the SDF is built from (13 samplings) is identical too, in its own cache file
    a, _ = vision.slab_occupancy(core, cell, cache_dir=tmp_path)
    b, _ = vision.slab_occupancy(core, cell, cache_dir=tmp_path, player=True)
    assert np.array_equal(a, b)
    assert (tmp_path / f"{PETRUS.stem}.slabocc_{cell:g}.npz").exists()
    zp = np.load(tmp_path / f"{PETRUS.stem}.slaboccp_{cell:g}.npz")
    assert str(zp["sig"]).endswith("_" + vision._PLAYER_SEMANTICS)


@pytest.mark.skipif(not UTOPIA.exists(), reason="surf_src_utopia not present")
def test_utopia_clip_ramp_is_player_solid_and_point_empty():
    core = _core(UTOPIA)
    # the surface the finisher rides at 45.1 s: plane normal (0, .781, .625)
    s = np.array([8005.0, -3586.0, -2767.0])
    n = np.array([0.0, 0.781, 0.625])
    n /= np.linalg.norm(n)
    # the physics: the ducked and standing hulls collide with it, a point passes straight through
    a, b = s + n * 80.0, s - n * 80.0
    assert core.trace(a.tolist(), b.tolist(), 2).fraction == 1.0
    for hull in (0, 1):
        tr = core.trace(a.tolist(), b.tolist(), hull)
        assert tr.fraction < 1.0 and float(np.array(tr.normal[:]) @ n) > 0.99
    # the grids: behind the surface the point grid is open, the player grid is solid (2)
    cell = 32.0
    mins = np.array([7840.0, -3776.0, -2944.0])
    g0 = core.occupancy_grid(mins, cell, 12, 12, 12)
    g1 = core.occupancy_grid(mins, cell, 12, 12, 12, player=True)
    assert np.array_equal(g1 == 1, g0 == 1)
    for depth in (8.0, 24.0):
        q = s - n * depth
        i = np.floor((q - mins) / cell).astype(int)
        assert g0[i[2], i[1], i[0]] == 0 and g1[i[2], i[1], i[0]] == 2
    # and open air in front of it stays open
    q = s + n * 96.0
    i = np.floor((q - mins) / cell).astype(int)
    assert g1[i[2], i[1], i[0]] == 0
