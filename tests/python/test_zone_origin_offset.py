"""detect_zones must offset a brush-entity AABB by its "origin" key.

A trigger built around an ORIGIN BRUSH stores its model vertices relative to
that origin; the world AABB is model + origin. goalfield's kill-volume mask
already assumes that convention (`hull_probe.contains(mi, pts - origin)`),
but detect_zones did not, so on such a map the race zones landed near the
world origin instead of on the map — a phantom finish box.

Regression guards, both directions:
  * a zero/absent origin must leave the box bit-identical (every map in
    maps/ except sidistic is in this class, so no trained checkpoint's zone
    boxes move);
  * a non-zero origin must shift the box by exactly that vector.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import zones  # noqa: E402

_BB = [([-8.0, -64.0, -4.0], [8.0, 64.0, 4.0]),        # model 0 (worldspawn)
       ([-88.0, -88.0, -4.0], [88.0, 88.0, 4.0])]      # model 1 (the trigger)


def _ents(origin):
    trig = {"classname": "trigger_multiple", "model": "*1",
            "target": "counter_stop_button"}
    if origin is not None:
        trig["origin"] = origin
    return [{"classname": "worldspawn", "model": "*0"},
            {"classname": "func_button", "targetname": "counter_stop_button",
             "target": "mapend"},
            trig]


@pytest.mark.parametrize("origin,shift", [
    (None,      (0.0, 0.0, 0.0)),
    ("0 0 0",   (0.0, 0.0, 0.0)),
    ("0 0 512", (0.0, 0.0, 512.0)),          # surf_src_sidistic's finish
    ("2136 -912 3400", (2136.0, -912.0, 3400.0)),   # surf_sg_china's finish
    ("garbage", (0.0, 0.0, 0.0)),            # kv_float parity: never raise
])
def test_origin_offsets_the_zone_box(monkeypatch, origin, shift):
    monkeypatch.setattr(zones, "parse_bsp",
                        lambda p: (_ents(origin), _BB))
    end = zones.detect_zones("fake.bsp")["end"]
    assert end is not None
    assert end["mins"] == pytest.approx([_BB[1][0][i] + shift[i]
                                         for i in range(3)])
    assert end["maxs"] == pytest.approx([_BB[1][1][i] + shift[i]
                                         for i in range(3)])


def test_trained_maps_are_unchanged():
    """The three maps every checkpoint was trained on carry no origin brush
    on their timer triggers, so this fix cannot move their zone boxes."""
    for stem in ("surf_petrus_lite", "surf_src_cannonball", "surf_ski_2"):
        bsp = ROOT / "maps" / f"{stem}.bsp"
        if not bsp.is_file():
            pytest.skip(f"{stem}.bsp not in this checkout")
        ents, _ = zones.parse_bsp(bsp)
        offset = [e for e in ents
                  if e.get("classname") == "trigger_multiple"
                  and (e.get("origin") or "0 0 0").split()[:3] != ["0", "0", "0"]]
        assert not offset, f"{stem} has an origin-offset trigger: {offset}"
