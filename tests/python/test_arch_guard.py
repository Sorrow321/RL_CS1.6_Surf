"""A resume with a DIFFERENT architecture must say so in its own words.

The observation-block wideners can only ever see
``checkpoint_tensor_width - policy.feat_dim`` and have to assume the two
trunks agree, so when they do not the failure comes out in whichever
flag's language happens to run first. That cost a debugging session on
2026-09-04: a CPU pre-flight of tools/expert_loop.py from
runs/research/xENT131 (emb 512, hidden 448) died in the train phase with

    --route cannot warm-start this checkpoint: pi.0.weight is (448, 524),
    i.e. 449 route-side columns over a 75-wide trunk, and this run wants 1

- a route-block message, for a run with no route file, about a checkpoint
with none either. The whole discrepancy was the conv embedding: 524 is
11 scalars + 512 conv + 1 latch and 75 is 11 + 64, because --dry-run
passed ``--emb 64 --hidden 64`` to a WARM resume.

Two guards, one per direction, both before any widener:

  * ``check_arch_matches`` compares the checkpoint's own config keys with
    this run's flags and names the flag to drop. It catches the case above
    (checkpoint WIDER than the model);
  * ``ck_trunk_mismatch`` is the tensor backstop for a checkpoint too old
    to carry those keys, and catches the other direction (checkpoint
    NARROWER than this run's trunk alone), which is arithmetically
    impossible for an observation block.

    python -m pytest tests/python/test_arch_guard.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.bc import N_SCALAR                                   # noqa: E402
from train_fast import (ARCH_KEYS, Policy, check_arch_matches,    # noqa: E402
                        ck_obs_block, ck_trunk_mismatch, widen_for_obs)


class _Args:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _policy(emb=16, hidden=12, route_dim=0):
    # obs_dim = 15 core + route block + the depth image
    return Policy(N_SCALAR + route_dim + 8 * 4, 8, 4, emb=emb, hidden=hidden,
                  route_dim=route_dim)


# --------------------------------------------------------------------------
# 1. the config guard: the real xENT131 case, in its own words
# --------------------------------------------------------------------------
def test_a_narrower_emb_on_a_warm_resume_is_refused_by_name():
    p = _policy()
    ck_cfg = {"emb": 512, "hidden": 448, "trunk": "plain", "lidar_w": 64,
              "lidar_h": 32, "conv_mult": 1, "tower_depth": 2}
    args = _Args(emb=64, hidden=64, trunk="plain", lidar_w=64, lidar_h=32,
                 conv_mult=1, tower_depth=2)
    with pytest.raises(SystemExit) as e:
        check_arch_matches(ck_cfg, args, p)
    msg = str(e.value)
    assert "--emb 64" in msg and "trained at 512" in msg
    assert "--hidden 64" in msg and "trained at 448" in msg
    # it must point at what a smoke run should shrink instead
    assert "--envs" in msg and "--n-steps" in msg
    # and it must NOT be phrased as a route/observation-block problem
    assert "route" not in msg.lower()


def test_matching_or_absent_flags_pass():
    p = _policy()
    ck_cfg = {"emb": 512, "hidden": 448, "trunk": "plain"}
    # equal, including across int/str (a config round-trips through JSON)
    check_arch_matches(ck_cfg, _Args(emb=512, hidden=448, trunk="plain"), p)
    check_arch_matches(ck_cfg, _Args(emb="512", hidden=448, trunk="plain"), p)
    # None = "the flag was not given", which is where the restore happens
    check_arch_matches(ck_cfg, _Args(emb=None, hidden=None, trunk=None), p)
    # a key the checkpoint does not carry is not evidence of anything
    check_arch_matches({}, _Args(emb=64, hidden=64), p)
    # every key in the list is a real trainer arg
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    for key, flag in ARCH_KEYS:
        assert f'"{flag}"' in src, flag


# --------------------------------------------------------------------------
# 2. the tensor backstop: the OTHER direction, where no config exists
# --------------------------------------------------------------------------
def _fake_ck(width, rows=12, rnn=0):
    sd = {"pi.0.weight": torch.zeros(rows, width),
          "vf.0.weight": torch.zeros(rows, width)}
    if rnn:
        sd["gru.weight_hh_l0"] = torch.zeros(3 * rnn, rnn)
    return {"policy": sd}


def test_a_checkpoint_narrower_than_the_trunk_is_named_as_a_trunk_mismatch():
    p = _policy(route_dim=1)
    feat = int(p.feat_dim)
    # the checkpoint's towers are NARROWER than this run's trunk alone -
    # arithmetically impossible for an observation block, which is >= 0
    got = ck_trunk_mismatch(_fake_ck(feat - 8), p)
    assert got is not None
    name, width, rem = got
    assert name == "pi.0.weight" and width == feat - 8 and rem == -8
    # exactly the trunk width, or wider (a real block), is not a mismatch
    assert ck_trunk_mismatch(_fake_ck(feat), p) is None
    assert ck_trunk_mismatch(_fake_ck(feat + 1), p) is None
    # the GRU block counts toward the width, so a recurrent checkpoint of
    # the same trunk is not a mismatch either
    assert ck_trunk_mismatch(_fake_ck(feat + 16, rnn=16), p) is None
    assert ck_trunk_mismatch({"policy": {}}, p) is None


# --------------------------------------------------------------------------
# 3. what the guards prevent: the widener's own message, reproduced
# --------------------------------------------------------------------------
def test_without_the_guard_the_widener_blames_the_route_block():
    """The exact misdiagnosis, so the guard's reason cannot be deleted by
    someone who reads it as belt-and-braces. A checkpoint 448 columns WIDER
    than this run's trunk reads as 449 route-side columns over a 75-wide
    trunk - `ck_obs_block` charges the whole difference to the block it
    happens to be measuring, because it cannot see the other term."""
    p = _policy(route_dim=1)
    feat = int(p.feat_dim)
    ck = _fake_ck(feat + 448 + 1)          # + a wider trunk + the latch
    assert ck_obs_block(ck, p) == 449
    with pytest.raises(SystemExit) as e:
        widen_for_obs(ck, p, 1, flag="--route")
    msg = str(e.value)
    assert "449 route-side columns" in msg
    assert f"{feat}-wide trunk" in msg
    # ...which is why check_arch_matches runs FIRST, and says --emb
    assert "--emb" not in msg
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    i_guard = src.index("check_arch_matches(ck_cfg, args, policy)")
    i_widen = src.index("widen_for_obs(ck, policy, N_ROUTE", i_guard - 20000)
    assert i_guard < i_widen, "the guard must run before the wideners"
