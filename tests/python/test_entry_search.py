"""tools/entry_search.py (pbr1b1, 2026-10-23): random bursts from a root state into a target
region, keeping each success's per-tick states. CPU, princessburglar: from the true spawn onto
the start room's conveyor-plate ledge (feet >= 3674 on the ground), then a chained second stage
from those landings; the output carries ordered STATE_DTYPE lines that start at the root and end
inside the target, plus the provenance meta."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
PB = ROOT / "maps" / "surf_princessburglar_b26.bsp"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_pb = pytest.mark.skipif(not (PB.exists() and DLL.exists()),
                              reason="needs the built core + princessburglar")
LEDGE = "640 725 -2965 -2900 3600 3800"


def _run(args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2")
    return subprocess.run([sys.executable, str(ROOT / "tools" / "entry_search.py"), "--map",
                           str(PB)] + args, capture_output=True, text=True, env=env,
                          cwd=str(ROOT), timeout=900)


@needs_pb
def test_lines_start_at_the_root_and_end_in_the_target(tmp_path):
    from surfgym.core import STATE_DTYPE
    a = tmp_path / "a.npz"
    r = _run(["--out", str(a), "--root-spawn", "--target", LEDGE, "--onground", "--feet-min",
              "3674", "--k", "1", "--envs", "512", "--ticks", "300", "--minutes", "3"])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    z = np.load(a, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    assert meta["hits"] == len(z["lines"]) == 1 and "no demo" in meta["provenance"]
    line = z["lines"][0]
    assert line.dtype == STATE_DTYPE and len(line) >= 2 and len(z["acts"][0]) == len(line)
    assert np.allclose(line[0]["origin"], [860.0, -3005.0, 3653.0], atol=1.0)   # the true spawn
    e = line[-1]
    assert 640 <= e["origin"][0] <= 725 and -2965 <= e["origin"][1] <= -2900
    assert e["onground"] != -1
    feet = e["origin"][2] - (18.0 if e["ducked"] else 36.0)
    assert feet >= 3674.0
    # every other row is short of the target (the line stops at its FIRST entry)
    o = line["origin"][:-1]
    ft = o[:, 2] - np.where(line["ducked"][:-1] > 0, 18.0, 36.0)
    inside = ((o[:, 0] >= 640) & (o[:, 0] <= 725) & (o[:, 1] >= -2965) & (o[:, 1] <= -2900)
              & (line["onground"][:-1] != -1) & (ft >= 3674.0))
    assert not inside.any()
    # a chained stage takes each line's last state as its root
    b = tmp_path / "b.npz"
    r2 = _run(["--out", str(b), "--stage-roots", str(a), "--target", LEDGE, "--k", "1",
               "--envs", "64", "--ticks", "20", "--minutes", "0.2"])
    assert r2.returncode in (0, 1), r2.stdout[-2000:] + r2.stderr[-2000:]
    zb = np.load(b, allow_pickle=True)
    for k, lb in enumerate(zb["lines"]):
        assert np.allclose(lb[0]["origin"], line[-1]["origin"])
