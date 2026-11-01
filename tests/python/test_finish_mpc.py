"""tools/finish_mpc.py: a root already inside the reachable finish region finishes at once and
replays exactly; a root far below it does not finish (zlr3b1, 2026-11-01)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

BSP = ROOT / "maps" / "surf_za_loop.bsp"


@pytest.mark.skipif(not BSP.exists(), reason="map not present")
def test_finish_mpc_finishes_and_replays(tmp_path):
    torch = pytest.importorskip("torch")
    from surfgym.core import STATE_DTYPE
    import finish_mpc
    cfg = {"map": "surf_za_loop", "tick_ms": 7.0, "maxvel": 3500.0, "view_continuous": 1,
           "view_absolute": "velocity", "teleport_fail": True, "reward": "race",
           "yaw_adaptive": True, "pitch_rate": 1.33}
    ck = tmp_path / "ck.pt"
    torch.save({"config": cfg}, ck)
    st = np.zeros(2, STATE_DTYPE)
    st["origin"][0] = (-200.0, 2000.0, 3060.0)      # standing over the strip, clear of its floor
    st["velocity"][0] = (0.0, 0.0, 50.0)
    st["origin"][1] = (500.0, 1600.0, 2400.0)       # far below the platform, falling
    st["velocity"][1] = (0.0, 0.0, -300.0)
    for k in ("onground",):
        st[k] = -1
    sf = tmp_path / "st.npy"
    np.save(sf, st)
    out = tmp_path / "o"
    finish_mpc.main(["--ckpt", str(ck), "--map", str(BSP), "--states", str(sf), "--rows", "0,1",
                     "--out", str(out), "--pop", "32", "--iters", "1", "--plan", "0.2",
                     "--duration", "0.2", "--exec", "4"])
    summ = json.loads((out / "summary.json").read_text())
    assert summ[0]["finish_tick"] is not None and summ[0]["finish_tick"] <= 2, summ[0]
    assert summ[0]["replay_ok"], summ[0]
    assert (out / "line_r0.npy").exists()
    assert summ[1]["finish_tick"] is None and summ[1]["best_gap"] > 500, summ[1]
