"""tools/mm_assets.py (mm/s0, 2026-11-16): the seven-map assets check and pack (CPU only).

Needs a staging directory with prebaked caches (MM_ASSETS_TEST_DIR, flat <stem>.* files, e.g.
the maps' tarballs unpacked and restamped) and the built core (SURFCORE_DLL).
* the signatures the tool computes equal the stored ones of valid caches (so it predicts the
  builders' cache hits), a re-dated bsp is a MISS with the restamp hint, a missing cache is
  MISSING, and pack writes the maps/ layout;
* princessburglar (when staged): 135 solid brush entities and STALE grids (built by the core
  that held 128), the mm/s0 finding.
"""
from __future__ import annotations

import os
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

STAGE = Path(os.environ.get("MM_ASSETS_TEST_DIR", str(ROOT / "maps")))
DLL = Path(os.environ.get("SURFCORE_DLL", str(ROOT / "build" / "libsurfcore.so")))
SMALL = "surf_edgeflow_blue200"
KINDS = ("goal", "goalc", "sdf", "occ", "slabocc", "faceid")


def _staged(stem):
    return all((STAGE / f"{stem}.{k}_32.npz").exists() for k in KINDS) and \
        (STAGE / f"{stem}.bsp").exists() and (STAGE / f"{stem}.zones.json").exists()


needs_small = pytest.mark.skipif(not (_staged(SMALL) and DLL.exists()),
                                 reason="needs MM_ASSETS_TEST_DIR with blue200's six caches")
needs_pb = pytest.mark.skipif(not ((STAGE / "surf_princessburglar_b26.bsp").exists()
                                   and DLL.exists()),
                              reason="needs princessburglar staged in MM_ASSETS_TEST_DIR")


def _copy(stem, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for p in STAGE.glob(f"{stem}.*"):
        shutil.copy2(p, dst / p.name)              # copy2 keeps the mtime (ns)


@needs_small
def test_valid_caches_check_ok_and_pack(tmp_path):
    import mm_assets
    d = tmp_path / "stage"
    _copy(SMALL, d)
    r = mm_assets.check_map(d, SMALL)
    assert r["ok"], r
    assert all(v == "OK" for v in r["files"].values()), r
    out = tmp_path / "m.tar.gz"
    assert mm_assets.main(["pack", "--dir", str(d), "--out", str(out)]) == 0
    with tarfile.open(out) as tf:
        names = sorted(tf.getnames())
    assert f"maps/{SMALL}.bsp" in names and f"maps/{SMALL}.goalc_32.npz" in names
    assert all(n.startswith("maps/") for n in names)


@needs_small
def test_a_redated_bsp_misses_and_a_missing_cache_is_named(tmp_path):
    import mm_assets
    d = tmp_path / "stage"
    _copy(SMALL, d)
    st = (d / f"{SMALL}.bsp").stat()
    os.utime(d / f"{SMALL}.bsp", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    (d / f"{SMALL}.faceid_32.npz").unlink()
    r = mm_assets.check_map(d, SMALL)
    assert not r["ok"]
    assert "run restamp_maps.py first" in r["files"][f"{SMALL}.occ_32.npz"]
    assert r["files"][f"{SMALL}.faceid_32.npz"].startswith("MISSING")
    assert mm_assets.main(["pack", "--dir", str(d), "--out", str(tmp_path / "x.tar.gz")]) == 1


@needs_pb
def test_princessburglars_grids_are_flagged_stale(tmp_path):
    import mm_assets
    assert mm_assets.solid_entity_count(STAGE / "surf_princessburglar_b26.bsp") == 135
    if not (STAGE / "surf_princessburglar_b26.occ_32.npz").exists():
        pytest.skip("princessburglar's occ_32 is not staged")
    r = mm_assets.check_map(STAGE, "surf_princessburglar_b26")
    if r.get("occ_vs_core", 0) == 0:
        pytest.skip("this princessburglar occ_32 was rebuilt with the fixed core")
    assert not r["ok"] and any(n.startswith("STALE") for n in r["notes"])
