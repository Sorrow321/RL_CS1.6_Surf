"""tools/transfer_ckpt.py: a cross-map warm start keeps the weights and the Adam moments, sets
the target map, drops every piece of the source map's run state and restarts the step count."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import transfer_ckpt as T  # noqa: E402


def _src():
    return {"policy": {"w": 1}, "optimizer": {"state": 2}, "global_step": 25_394_937_856,
            "config": {"map": "surf_src_sidistic", "maxvel": 4000.0,
                       "obs_potential_d0": {"sidistic": 99674.75}, "demo_file": None},
            "int_counts": [1, 2], "unstuck": {"T": 0.2}, "respawn_novel": {"a": 1},
            "respawn_greedy": {"rows": []}, "respawn": {"states": []}}


def test_transfer_keeps_weights_drops_run_state():
    src = _src()
    out = T.transfer(src, "surf_princessburglar_b26", src="x.pt", src_md5="abc")
    assert set(out) == {"policy", "optimizer", "config", "global_step"}
    assert out["policy"] is src["policy"] and out["optimizer"] is src["optimizer"]
    assert out["global_step"] == 0
    c = out["config"]
    assert c["map"] == "surf_princessburglar_b26"
    assert "obs_potential_d0" not in c
    assert c["maxvel"] == 4000.0                       # flags are the launch's business
    t = c["transfer"]
    assert t["src_map"] == "surf_src_sidistic" and t["src_step"] == 25_394_937_856
    assert t["dropped"] == ["int_counts", "respawn", "respawn_greedy", "respawn_novel",
                            "unstuck"]
    assert t["optimizer"] and t["step_reset"]
    # pure: the source dict is untouched
    assert src["config"]["map"] == "surf_src_sidistic" and "unstuck" in src


def test_keep_step():
    out = T.transfer(_src(), "m", keep_step=True)
    assert out["global_step"] == 25_394_937_856 and not out["config"]["transfer"]["step_reset"]


@pytest.mark.parametrize("key", T.DEMO_KEYS)
def test_refuses_demo_trained_source(key):
    src = _src()
    src["config"][key] = "runs/x/demo.npz"
    with pytest.raises(SystemExit):
        T.transfer(src, "m")
