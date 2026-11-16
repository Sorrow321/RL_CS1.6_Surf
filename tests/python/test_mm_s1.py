"""Stage 1 of the multi-map build (mm/s1): v14's refused parts, per map. CPU runs of the v14 line
on two maps (MM_TEST_MAPS, default maps/surf_edgeflow_blue200 + surf_unitfarmer2, each with its
six cell-32 caches; SURFCORE_DLL), with the families not ported yet stripped
(tests/python/v14_smoke.py FAMILIES). The single-map identity is tests/python/test_mm_identity.py.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import v14_smoke as V  # noqa: E402
from surfgym.mapfleet import map_tags  # noqa: E402

needs_mm = pytest.mark.skipif(not V.have_maps(), reason="needs two maps with their six "
                                                        "cell-32 caches (MM_TEST_MAPS)")
# the families still refused with --maps; each S1 item removes its own
NOT_YET = ["front_temp", "demo_front_temp", "greedy_temp", "energy_front", "int_gate",
           "unstuck", "front_perturb", "greedy", "loop"]
MAPS_ARG = ["--maps", ",".join(str(m) for m in V.MM_MAPS)]


def _headers(path):
    hs = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("{") and '"map"' in ln:
                hs.append(json.loads(ln))
    return hs


@needs_mm
def test_a_eval_parallel_records_every_map_side_by_side():
    r, d = V.run_line(V.ROOT, "mms1_a", MAPS_ARG, drop=NOT_YET, steps=4096)
    try:
        assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
        for m, tag in zip(V.MM_MAPS, map_tags([m.stem for m in V.MM_MAPS])):
            files = sorted(d.glob(f"traj_*_{tag}.jsonl"))
            assert files, f"no eval of {tag}"
            for f in files:
                hs = _headers(f)
                assert len(hs) == 2 and all(h["map"] == m.stem for h in hs), f
                assert all(h.get("eval_parallel") == 1 for h in hs), f
    finally:
        shutil.rmtree(d, ignore_errors=True)
