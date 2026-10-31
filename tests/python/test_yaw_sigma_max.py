"""--yaw-sigma-max S (zlr1b2, 2026-10-31): a ceiling on the yaw head(s)' sigma under
--view-absolute, the pitch head's cap applied to the yaw.

In the velocity frame the yaw is an offset from the heading through off_warp(tanh z): sigma 0.3
(the init) keeps the view within a few degrees of the velocity (the strafe optimum), the 2.718
clamp draws offsets near +-180 deg, and a strafe with a backward component brakes. A stuck run
walks sigma to the clamp (princessburglar; surf_za_loop A2-A4: 1.2-2.718) and can no longer pump.

(a) the policy: the yaw head is capped, the pitch keeps its own cap, projection pulls a raw
    parameter under the ceiling, the state_dict keys do not change, the flag off is the old
    ceiling exactly, and the delta policy refuses it;
(b) the recorder and beam search mirror the key.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def _policy(**kw):
    import torch
    from train_fast import Policy
    torch.manual_seed(0)
    return Policy(15 + 32, 8, 4, emb=16, hidden=16, view_continuous=True,
                  view_absolute="velocity", **kw)


def test_the_yaw_head_is_capped_and_the_pitch_keeps_its_own():
    import torch
    p = _policy(yaw_sigma_max=0.3)
    with torch.no_grad():
        p.view_std.log_std.fill_(0.9)                          # both heads parked high
    s = p.log_std().exp().tolist()
    assert s[0] == pytest.approx(0.3, rel=1e-6)                # yaw: the new ceiling
    assert s[-1] == pytest.approx(0.5, rel=1e-6)               # pitch: its log 0.5 cap


def test_project_pulls_a_raw_yaw_parameter_under_the_ceiling():
    import torch
    p = _policy(yaw_sigma_max=0.3)
    with torch.no_grad():
        p.view_std.log_std.copy_(torch.tensor([1.0, -1.0]))
    assert p.project_log_std() == 1
    assert float(p.view_std.log_std[0]) == pytest.approx(float(np.log(0.3)), rel=1e-6)
    assert float(p.view_std.log_std[-1]) == pytest.approx(-1.0)
    assert p.project_log_std() == 0                            # idempotent


def test_the_state_dict_keys_do_not_change():
    a, b = _policy(), _policy(yaw_sigma_max=0.3)
    assert list(a.state_dict()) == list(b.state_dict())


def test_flag_off_is_the_old_ceiling_and_the_delta_policy_refuses_it():
    import torch
    from train_fast import LOG_STD_MAX, PITCH_LOG_STD_MAX_ABS, Policy
    for p in (_policy(), _policy(yaw_sigma_max=0.0)):
        hi = p.view_std.log_std_hi.tolist()
        assert hi == pytest.approx([LOG_STD_MAX] * (len(hi) - 1) + [PITCH_LOG_STD_MAX_ABS])
    torch.manual_seed(0)
    with pytest.raises(SystemExit):
        Policy(15 + 32, 8, 4, emb=16, hidden=16, view_continuous=True,
               view_absolute=None, yaw_sigma_max=0.3)


def test_the_sampled_offset_stays_near_the_velocity():
    """What the cap buys: offsets drawn at z ~ N(0, 0.3) stay within a few degrees of the
    velocity heading; at the 2.718 clamp most of them point more than 45 deg off it."""
    from surfgym.view import off_warp
    rng = np.random.default_rng(0)
    for sig, want in ((0.3, 0.01), (2.718, 0.5)):
        off = np.abs(off_warp(np.tanh(rng.normal(0.0, sig, 20000))))
        frac = float((off > 45.0).mean())
        assert (frac < want) if sig < 1 else (frac > want), (sig, frac)


def test_the_recorder_and_beam_search_mirror_it():
    import record_ckpt as rc
    assert "yaw_sigma_max" in rc._mentioned_keys()
    assert "yaw_sigma_max" in (ROOT / "tools" / "beam_tas.py").read_text(encoding="utf-8")
