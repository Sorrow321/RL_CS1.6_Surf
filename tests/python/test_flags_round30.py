"""Round 30's seven small flags (train_fast.py / rewards.py / record_ckpt.py).

Each flag is default-off and byte-identical to the pre-flag trainer when off,
each is recorded in ``run.json`` and restored on ``--ckpt``, and each is either
mirrored by ``tools/record_ckpt.py`` or listed in its ``TRAIN_ONLY`` allowlist.
What is pinned here, item by item:

1. ``--pitch-fixed`` - the states' pitch column is written immediately before
   every lidar render, so the RECORDED pitch column is constant even when the
   pitch head is asking to look elsewhere. The control (no flag, stock pitch
   rate) is checked to actually move, or the treatment proves nothing.
2. ``--tower-depth`` / ``--conv-mult`` - the tower is N tanh layers, the plain
   trunk's three convs scale by M and the Linear after the pool follows; the
   defaults (2, 1) rebuild the pre-flag Policy tensor for tensor, and the
   route/rnn zero-pad warm start still works on top.
3. ``--fp32-heads`` - the action and value heads run outside autocast, so
   under bf16 their output is the fp32 result rather than a bf16-quantized
   one; off is the same op it always was.
4. ``--max-step`` - RaceReward's per-tick teleport clip is a parameter, and
   the clip scales by ``every`` exactly as the hardcoded 100 did.
5. ``--stall-eps`` - the per-CALL improvement threshold reaches RaceReward AND
   the eval-stall mirror, which reads it off the reward object.
6. ``--n-steps`` / ``--epochs`` / ``--minibatches`` are restored from the
   checkpoint config, with an explicit flag still winning.
7. ``d0_per_env`` (per-env euclid potential) refuses to combine with ``ng`` /
   ``d_floor`` / ``d_latch``, which are geodesic-scale quantities.

    python -m pytest tests/python/test_flags_round30.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from train_fast import _TorchPolicyBase                       # noqa: E402

CANNONBALL = ROOT / "maps" / "surf_src_cannonball.bsp"
TRAIN_SRC = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
REC_SRC = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")


# ==========================================================================
# 1. --pitch-fixed
# ==========================================================================
class _StubLidar:
    """Records the pitch it is handed and returns a 4-pixel black frame."""

    def __init__(self):
        self.seen = []

    def render(self, origin, yaw, pitch, ducked):
        self.seen.append(np.asarray(pitch.cpu()).copy())
        return torch.zeros(origin.shape[0], 4)


class _PinProbe(_TorchPolicyBase):
    """Runs the real _obs (where the pin lives) and holds one action.

    ``pitch_bin`` 6 is the maximum LOOK UP delta, so the control drifts to
    the +30 clamp within a few ticks and a constant column cannot be an
    accident of a neutral action.
    """

    def __init__(self, *a, pitch_bin: int = 6, **kw):
        super().__init__(*a, **kw)
        self._act = np.array([[7, pitch_bin, 1, 1, 0, 0]], np.int32)

    def _decide(self, obs):
        self._obs(obs)
        return self._act


def _record_pitch(tmp_path, name, *, pitch_rate, pitch_fixed):
    from surfgym import SurfCore, default_config
    from surfgym.record import record_rollout

    core = SurfCore(str(CANNONBALL), default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=200, water_fail=1,
        sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0,
        pitch_rate_max_deg=pitch_rate))
    lidar = _StubLidar()
    pol = _PinProbe(None, None, torch.device("cpu"), lidar, core,
                    act_every=1, pitch_fixed=pitch_fixed)
    out = tmp_path / f"{name}.traj.jsonl"
    record_rollout(core, pol, out, episodes=1, max_ticks=120, seed=0)
    col = []
    for line in out.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if isinstance(row, list):
            col.append(row[12])          # index 12 = view pitch (record.py)
    return np.array(col, np.float64), lidar


@pytest.mark.skipif(not CANNONBALL.exists(),
                    reason="surf_src_cannonball.bsp absent")
def test_pitch_fixed_holds_the_recorded_pitch_column(tmp_path):
    # control: stock pitch rate, the head asking to look up -> it moves
    ctl, _ = _record_pitch(tmp_path, "ctl", pitch_rate=-1.0, pitch_fixed=None)
    assert len(ctl) > 20
    assert ctl.std() > 1.0, "control gaze did not move; the probe is inert"

    # treatment: --pitch-fixed pins the column and zeroes the rate
    arm, lidar = _record_pitch(tmp_path, "arm", pitch_rate=0.0,
                               pitch_fixed=-12.5)
    assert len(arm) > 20
    # tick 0 is the spawn snapshot taken BEFORE the first decision; every
    # snapshot after it is post-pin
    assert np.all(arm[1:] == -12.5), f"pitch column drifted: {set(arm[1:])}"
    assert lidar.seen, "the stub lidar was never asked to render"
    assert all(float(p[0]) == -12.5 for p in lidar.seen)


@pytest.mark.skipif(not CANNONBALL.exists(),
                    reason="surf_src_cannonball.bsp absent")
def test_pitch_fixed_off_touches_nothing(tmp_path):
    """pitch_fixed=None must leave the states exactly as the core wrote them."""
    from surfgym import SurfCore, default_config
    core = SurfCore(str(CANNONBALL), default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=200, water_fail=1,
        sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))
    obs = core.reset(0)
    lidar = _StubLidar()
    pol = _PinProbe(None, None, torch.device("cpu"), lidar, core, act_every=1)
    before = float(core.states_view["pitch"][0])
    pol._obs(obs)
    assert float(core.states_view["pitch"][0]) == before
    assert pol.pitch_fixed is None


def test_pitch_fixed_is_plumbed_through_the_trainer():
    """run.json, resume restore, both render sites and the eval mirror."""
    assert '"pitch_fixed": args.pitch_fixed' in TRAIN_SRC        # run.json
    assert 'ck_cfg.get("pitch_fixed")' in TRAIN_SRC              # resume
    assert 'restored.append(f"pitch_fixed=' in TRAIN_SRC
    assert "PITCH_PIN = args.pitch_fixed" in TRAIN_SRC
    # the rollout render, the truncation bootstrap and the in-trainer eval
    assert TRAIN_SRC.count("PITCH_PIN") >= 4
    assert "pitch_fixed=args.pitch_fixed" in TRAIN_SRC           # eval policy
    # ... and record_ckpt mirrors it rather than letting the gaze drift
    assert 'cfg.get("pitch_fixed")' in REC_SRC
    assert "pitch_fixed=pitch_fixed" in REC_SRC
