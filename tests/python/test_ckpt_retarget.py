"""tools/ckpt_retarget.py: a single-map checkpoint retargeted to another map (CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import ckpt_retarget  # noqa: E402


def test_retarget_clears_map_bound_entries_and_keeps_weights(tmp_path):
    w = {"a": torch.arange(4.0)}
    ck = {"policy": w, "optimizer": {"state": 1}, "global_step": 123,
          "config": {"map": "surf_old", "demo_file": "/x/spine.npy", "demo_front_frac": 0.8,
                     "bc_file": "/x/bc.npz", "keys_hold": True, "lr": 2e-5, "n_maps": 1},
          "respawn": {"states": [1]}, "int_counts": [1, 2], "unstuck": {"T": 1.0}}
    src, dst = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save(ck, src)
    assert ckpt_retarget.main(["--ckpt", str(src), "--map", "surf_new", "--out", str(dst)]) == 0
    out = torch.load(dst, map_location="cpu", weights_only=False)
    c = out["config"]
    assert c["map"] == "surf_new" and c["demo_file"] is None and c["bc_file"] is None
    assert c["demo_front_frac"] == 0.0 and c["keys_hold"] is True and c["lr"] == 2e-5
    assert out["global_step"] == 123 and torch.equal(out["policy"]["a"], w["a"])
    assert "respawn" not in out and "int_counts" not in out and "unstuck" not in out
    # the source is untouched
    assert torch.load(src, map_location="cpu", weights_only=False)["config"]["map"] == "surf_old"
