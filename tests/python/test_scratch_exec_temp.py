"""record_ckpt --plan-scratch: the scratch executor factory must build the SAME wrapper as the
recorded policy, temperature included. Until 2026-09-27 it rebuilt type(_pol), which dropped the
functools.partial carrying --exec-temp / --exec-keys-temp / --exec-view-scale, so every
edge_archive flight sampled at the native temperature whatever was asked."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

CKPT = ROOT / "runs" / "research" / "stage" / "prim1_mover.pt"
MAP = ROOT / "maps_pool" / "surf_edgeflow_blue025.bsp"


@pytest.mark.skipif(not (CKPT.exists() and MAP.exists()),
                    reason="needs the local stage-1 mover checkpoint and the edgeflow map")
def test_scratch_executor_keeps_the_view_scale():
    import record_ckpt
    ctx = record_ckpt.build([str(CKPT), "--episodes", "1", "--plan-scratch", "4", "--stochastic",
                             "--exec-view-scale", "0.1", "--map", str(MAP)], device="cpu")
    pol = ctx.scratch.make_policy(ctx.scratch.core, ctx.scratch.line)
    assert type(pol).__name__ == "TemperedTorchPolicy"
    assert pol.view_scale is not None
    assert all(abs(float(s) - 0.1) < 1e-6 for s in pol.view_scale)   # float32
