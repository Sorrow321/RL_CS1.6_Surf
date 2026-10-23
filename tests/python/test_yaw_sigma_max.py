"""``--yaw-sigma-max S``: a ceiling on the yaw head(s)' sigma under --view-absolute.

Why (pbr1b4, 2026-10-23): on a stuck start the entropy bonus is the only force on the yaw
sigma, and three arms took it from the sidistic finisher's 0.08 to the global clamp 2.718 in
~0.8B steps - every rollout's movement direction became noise. What is pinned:

1. with the flag, project_log_std() pulls a raw yaw log sigma above log S onto it and log_std()
   never reports more; the pitch head keeps its own cap (log 0.5);
2. without it, the ceiling list is the pre-flag one (yaw at LOG_STD_MAX);
3. the trainer writes ``yaw_sigma_max`` to the config only when set, the trained yaw sigma
   respects it, and the recorder mentions the key (its config audit passes).

    CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/python/test_yaw_sigma_max.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

from train_fast import (LOG_STD_MAX, N_SCALAR, PITCH_LOG_STD_MAX_ABS,  # noqa: E402
                        Policy)

W, H = 8, 4


def _pol(**kw):
    torch.manual_seed(0)
    return Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24, view_continuous=True,
                  view_absolute="velocity", **kw)


def test_the_yaw_ceiling_is_enforced_and_pitch_keeps_its_own():
    p = _pol(yaw_sigma_max=0.5)
    hi = p.view_std.log_std_hi.tolist()
    assert all(abs(v - math.log(0.5)) < 1e-6 for v in hi[:-1])
    assert abs(hi[-1] - PITCH_LOG_STD_MAX_ABS) < 1e-6
    with torch.no_grad():
        p.view_std.log_std.fill_(2.0)
    assert p.project_log_std() == len(hi)
    assert float(p.view_std.log_std.max()) <= math.log(0.5) + 1e-6
    assert float(p.log_std().max()) <= math.log(0.5) + 1e-6


def test_no_flag_is_the_pre_flag_ceiling():
    p = _pol()
    hi = p.view_std.log_std_hi.tolist()
    assert all(abs(v - LOG_STD_MAX) < 1e-6 for v in hi[:-1])
    assert abs(hi[-1] - PITCH_LOG_STD_MAX_ABS) < 1e-6
    q = _pol(yaw_sigma_max=0.0)
    assert q.view_std.log_std_hi.tolist() == hi


def test_the_recorder_mentions_the_key():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    assert "yaw_sigma_max" in record_ckpt._mentioned_keys()


try:
    import test_init_weights as tiw          # the princessburglar smoke harness
except Exception:                            # pragma: no cover
    tiw = None


@pytest.mark.skipif(tiw is None or tiw.MAPS is None or not tiw.DLL.exists(),
                    reason="needs the built core + the prebaked princessburglar fields")
def test_the_trainer_writes_the_key_only_when_set_and_respects_the_cap():
    view = ["--view-continuous", "--view-absolute", "velocity"]
    try:
        _, on = tiw._train("ysm_on", view + ["--steps", "1024", "--ent", "0.5",
                                             "--yaw-sigma-max", "0.4"])
        cfg = json.loads((on / "run.json").read_text(encoding="utf-8"))["config"]
        assert cfg["yaw_sigma_max"] == 0.4
        ck = torch.load(on / "ckpt_final.pt", map_location="cpu", weights_only=False)
        ls = ck["policy"]["view_std.log_std"]
        assert float(ls[:-1].max()) <= math.log(0.4) + 1e-5
        _, off = tiw._train("ysm_off", view + ["--steps", "512"])
        cfg = json.loads((off / "run.json").read_text(encoding="utf-8"))["config"]
        assert "yaw_sigma_max" not in cfg
    finally:
        import shutil
        for run in ("ysm_on", "ysm_off"):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
