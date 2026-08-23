"""load_zones must trust a zones.json it cannot regenerate from the BSP.

447 of the 620 maps in the dataset have NO timer entities: the timer lives
in a server plugin, and the start/stop button positions come from the Surf
Gateway buttons service (``tools/fetch_gateway_buttons.py``), which writes
``"source": "gateway"``.

The old rule trusted only ``"source": "manual"`` and otherwise re-ran
``detect_zones``. On exactly these maps detection finds nothing, so the
gateway file was read, discarded, and ``{"start": None, "end": None}``
handed back — ``--reward race`` would then refuse the map it had just been
given a finish for. The rule is therefore inverted: only ``"auto"`` is
regenerable from the BSP and re-extracts on a signature miss; every other
source is trusted verbatim.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import zones  # noqa: E402

_BOX = {"mins": [1.0, 2.0, 3.0], "maxs": [4.0, 5.0, 6.0]}


def _fake_bsp(tmp_path):
    """A file that stands in for the map. load_zones only stat()s it for the
    signature; detect_zones is monkeypatched, so the bytes never matter."""
    bsp = tmp_path / "surf_nozones.bsp"
    bsp.write_bytes(b"not a bsp")
    return bsp


@pytest.mark.parametrize("source", ["gateway", "manual", "hand", None])
def test_non_auto_source_is_trusted_and_never_overwritten(
        tmp_path, monkeypatch, source):
    bsp = _fake_bsp(tmp_path)
    doc = {"map": bsp.stem, "start": _BOX, "end": _BOX}
    if source is not None:
        doc["source"] = source
    zp = zones.zones_path(bsp)
    zp.write_text(json.dumps(doc), encoding="utf-8")

    # the whole point: this map has nothing in the BSP to detect
    monkeypatch.setattr(zones, "detect_zones",
                        lambda p: {"start": None, "end": None})

    got = zones.load_zones(str(bsp))
    assert got["end"] == _BOX, "an unregenerable zone file was discarded"
    assert got["start"] == _BOX
    assert json.loads(zp.read_text(encoding="utf-8")) == doc, \
        "load_zones rewrote a file it cannot reproduce"


def test_auto_source_still_re_extracts_on_a_signature_miss(
        tmp_path, monkeypatch):
    bsp = _fake_bsp(tmp_path)
    stale = {"mins": [-9.0, -9.0, -9.0], "maxs": [-8.0, -8.0, -8.0]}
    zp = zones.zones_path(bsp)
    zp.write_text(json.dumps({"map": bsp.stem, "source": "auto",
                              "bsp_sig": "0_0", "start": stale,
                              "end": stale}), encoding="utf-8")
    monkeypatch.setattr(zones, "detect_zones",
                        lambda p: {"start": _BOX, "end": _BOX})

    got = zones.load_zones(str(bsp))
    assert got["end"] == _BOX, "a stale auto file was not re-extracted"
    assert got["bsp_sig"] == zones._bsp_sig(bsp)


def test_auto_source_with_a_matching_signature_is_reused(
        tmp_path, monkeypatch):
    bsp = _fake_bsp(tmp_path)
    zp = zones.zones_path(bsp)
    zp.write_text(json.dumps({"map": bsp.stem, "source": "auto",
                              "bsp_sig": zones._bsp_sig(bsp),
                              "start": _BOX, "end": _BOX}), encoding="utf-8")

    def _boom(p):
        raise AssertionError("re-extracted a fresh auto file")

    monkeypatch.setattr(zones, "detect_zones", _boom)
    assert zones.load_zones(str(bsp))["end"] == _BOX
