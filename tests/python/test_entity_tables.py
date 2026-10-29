"""The core loads EVERY brush entity of a map, and refuses a map that overflows one of its entity
tables instead of dropping the excess (src/sim.h MAX_*).

Why (pbx, 2026-10-29): the tables held 128 solid brush entities and 128 triggers and dropped the
rest silently. surf_princessburglar_b26 has 135 solid brush entities; the 131st, func_conveyor
*250, is a 3 u slab (z 3678..3681) on the floor of the doorway west of the spawn. Without it the
doorway floor was 62 u above the spawn floor and a 63 u crouch-jump reached it; in the game it is
65 u - every princessburglar finish of the pbr / pbEXIT2 lineage climbed through that gap.
surf_src_sidistic lost 10 of its 138 teleport / push / hurt triggers the same way.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
PB = REPO / "maps" / "surf_princessburglar_b26.bsp"
SKI = REPO / "maps" / "surf_ski_2.bsp"


def _core(path):
    from surfgym.core import SurfCore, default_config
    return SurfCore(str(path), default_config())


def _feet_z(core, x, y):
    tr = core.trace([x, y, 3760.0], [x, y, 3640.0], 0)     # standing hull, straight down
    return float(tr.endpos[2]) - 36.0


def test_princessburglar_doorway_keeps_its_conveyor_slab():
    if not PB.exists():
        pytest.skip("no princessburglar")
    core = _core(PB)
    # over the north strip of func_conveyor *250 (AABB x 648..718, y -3054..-2946; its middle is
    # open down to *249 at z 3657): its top, not the 3678 floor under it
    assert abs(_feet_z(core, 680.0, -2940.0) - 3681.031) < 0.01
    # the north edge the 60.069 s search line stood on (origin y -2930.13..-2930.85): the
    # standing hull meets the slab from y -2930 (its face -2946 + the 16 u hull) and is 65 u
    # above the spawn floor there; north of the edge it is the spawn floor itself
    assert abs(_feet_z(core, 649.016, -2930.5) - 3681.031) < 0.01
    assert abs(_feet_z(core, 649.016, -2929.0) - 3616.031) < 0.01


def _with_extra_walls(src: Path, dst: Path, n: int) -> None:
    """A copy of a v30 BSP whose entity lump gains n func_walls on brush model *1."""
    data = bytearray(src.read_bytes())
    off, length = struct.unpack_from("<ii", data, 4)               # lump 0: entities
    ents = bytes(data[off:off + length]).rstrip(b"\0")
    ents += b"".join(b'{\n"classname" "func_wall"\n"model" "*1"\n}\n' for _ in range(n)) + b"\0"
    new_off = len(data)
    data += ents
    struct.pack_into("<ii", data, 4, new_off, len(ents))
    dst.write_bytes(bytes(data))


def test_a_map_that_overflows_the_solid_table_is_refused(tmp_path):
    if not SKI.exists():
        pytest.skip("no surf_ski_2")
    from surfgym.core import SurfCore  # noqa: F401  (the library must load)
    ok = tmp_path / "under.bsp"
    _with_extra_walls(SKI, ok, 400)            # ski_2's own 11 + 400: inside the 512 table
    _core(ok)
    bad = tmp_path / "over.bsp"
    _with_extra_walls(SKI, bad, 600)
    with pytest.raises(RuntimeError, match="solid brush entities"):
        _core(bad)
