"""The engine rule for a brush timer trigger (uf2r2b2 + uf2r2b4, 2026-10-31; harness merge by the
operator): GoldSrc fires a brush trigger only with the player origin inside the model's CLIP HULL
at the end of the move (SV_TouchLinks); its bounding box is only the pre-check. surf_set_goal_box
finds the brush model whose bounds ARE the goal box (zones.json's "end" on a type-1 map) and
completes on its clip hull (goal_hull on, the default) - unless that model is a PLAIN box, whose
hull is the box itself, where the swept box stays (byte-identical on such maps). uf2r2b4's
surf_set_goal_model names the model explicitly; -1 = the swept box.

surf_unitfarmer2's finish *23 (counter_stop_button) is three 1 u curtains - x 3328 for
y <= -2144, the diagonal (3328, -2144)-(3584, -1888), y -1888 for x >= 3584 - and its box adds an
empty triangle north-west of the diagonal: every finish of the round-1 incumbent's lineage but
one, and every beam line of round 2, only entered that triangle. Its start *16 is a plain box."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
UF2 = ROOT / "maps" / "surf_unitfarmer2.bsp"
ZONES = ROOT / "maps" / "surf_unitfarmer2.zones.json"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else ROOT / "build" / "libsurfcore.so")
needs_uf2 = pytest.mark.skipif(not (UF2.exists() and ZONES.exists() and DLL.exists()),
                               reason="needs the built core + surf_unitfarmer2")

CORNER = (3315.0, -1950.0, -500.0)        # inside the box's NW triangle, off every curtain


def _probe(hull, pos, vel, ticks=6, box=None, model=None):
    """hull: None = the default, True/False = set_goal_hull; model = set_goal_model(model)."""
    from surfgym.core import SurfCore, default_config
    z = json.loads(ZONES.read_text())
    core = SurfCore(str(UF2), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    if getattr(core._lib, "surf_set_goal_model", None) is None:
        pytest.skip("this surfcore build predates the clip-hull goal")
    lo, hi = (box or (z["end"]["mins"], z["end"]["maxs"]))
    core.set_goal_box(lo, hi)
    if hull is not None:
        core.set_goal_hull(hull)
    if model is not None:
        core.set_goal_model(model)
    gm = core.goal_model()
    core.reset(0)
    st = core.get_states()[0].copy()
    st["origin"][:] = pos
    st["velocity"][:] = vel
    st["onground"] = -1
    core.set_state(0, st)
    a = np.zeros((1, 6), np.int32)
    a[0, 0], a[0, 1] = 7, 3
    hits = []
    for _ in range(ticks):
        core.step(a)
        hits.append(int(core.goal_hits[0]))
    return gm, hits


@needs_uf2
def test_default_is_the_engine_rule_and_finds_the_trigger_model():
    gm, hits = _probe(None, CORNER, (200, 0, 0))
    assert gm == 23 and not any(hits), (gm, hits)      # the empty triangle is no finish


@needs_uf2
def test_the_box_rule_still_counts_the_triangle_when_asked():
    gm, hits = _probe(False, CORNER, (200, 0, 0))
    assert gm == 0 and hits[0] == 1, (gm, hits)


@needs_uf2
def test_crossing_a_real_curtain_finishes_under_both_rules():
    for hull in (True, False):
        gm, hits = _probe(hull, (3290, -2300, -400), (1000, 0, 0))     # the west face
        assert any(hits), (hull, gm, hits)
        gm, hits = _probe(hull, (3480, -1960, -300), (700, -700, 0))   # the diagonal
        assert any(hits), (hull, gm, hits)


@needs_uf2
def test_a_box_that_is_no_model_keeps_the_swept_box():
    z = json.loads(ZONES.read_text())
    lo = [v - 64.0 for v in z["end"]["mins"]]           # a padded box (a button finish)
    hi = [v + 64.0 for v in z["end"]["maxs"]]
    gm, hits = _probe(True, CORNER, (200, 0, 0), box=(lo, hi))
    assert gm == 0 and hits[0] == 1, (gm, hits)


@needs_uf2
def test_a_plain_box_trigger_keeps_the_swept_box():
    # the start curtain *16 is one axial box brush: its clip hull IS its grown bounds, so the
    # auto match leaves the swept box (the pre-fix core, byte-identical) in charge
    z = json.loads(ZONES.read_text())
    st = (z["start"]["mins"], z["start"]["maxs"])
    gm, hits = _probe(None, (-2176.0, -2380.0, 1200.0), (0, -300, 0), box=st)
    assert gm == 0 and any(hits), (gm, hits)
    # named explicitly (uf2r2b4's API) it is the clip hull, which fires there too
    gm, hits = _probe(None, (-2176.0, -2380.0, 1200.0), (0, -300, 0), box=st, model=16)
    assert gm == 16 and any(hits), (gm, hits)


@needs_uf2
def test_the_explicit_model_api():
    # uf2r2b4's surf_set_goal_model: the finish model named, or -1 / out of range = the box
    gm, hits = _probe(None, CORNER, (100, -100, 0), model=23)
    assert gm == 23 and not any(hits), (gm, hits)
    for bad in (-1, 10 ** 6):
        gm, hits = _probe(None, CORNER, (100, -100, 0), model=bad)
        assert gm == 0 and hits[0] == 1, (bad, gm, hits)


@needs_uf2
def test_the_lineage_finish_and_the_record_finish():
    # the last tick of beam line pf502 (35.80 s on the box rule): the triangle, below the
    # platform's floor - no finish under the engine rule; Fizz's record crosses the west face
    gm, hits = _probe(None, (3306.4, -1871.1, -643.1), (918.0, -1038.0, -319.0), ticks=1)
    assert gm == 23 and not any(hits), (gm, hits)
    gm, hits = _probe(False, (3306.4, -1871.1, -643.1), (918.0, -1038.0, -319.0), ticks=1)
    assert hits == [1], hits
    gm, hits = _probe(None, (3321.6, -2116.7, -506.0), (1086.0, -693.0, -305.0), ticks=3)
    assert gm == 23 and any(hits), (gm, hits)


def test_the_model_lookup_finds_the_finish_trigger():
    if not (UF2.exists() and ZONES.exists()):
        pytest.skip("surf_unitfarmer2 map files not present")
    from surfgym.zones import goal_trigger_model
    z = json.loads(ZONES.read_text())
    assert goal_trigger_model(str(UF2), z["end"]) == 23
    assert goal_trigger_model(str(UF2), {"mins": [0, 0, 0], "maxs": [1, 1, 1]}) == -1
