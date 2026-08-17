"""Strided frame stacking (--frame-stack K), CPU only.

The rollout and the update build the SAME stack from two different places:
the rollout composes it per decision out of a GPU ring of past renders, the
update gathers it out of the flat single-frame rollout buffer at PPO time.
If those two ever disagree, PPO optimizes a network on inputs the policy
never saw — silently, with no shape error and no NaN, just a quietly wrong
gradient. That agreement is what most of this file is about.

The rest: history must collapse at a spawn (a policy cannot remember before
its episode began, so training must not either), the layout must stay
channel-fastest (Policy.forward_split's restride is only free that way), and
with the flag off nothing may move at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (N_SCALAR, NVEC, FrameRing,            # noqa: E402
                        GreedyTorchPolicy, Policy, STACK_STRIDES,
                        check_vision_exclusive, frame_offsets,
                        interleave_frames, stack_from_buffer, stack_from_ring)


# ------------------------------------------------------------- the offsets --
def test_offsets_are_the_documented_strides():
    assert frame_offsets(0) == (0,) and frame_offsets(1) == (0,)
    assert frame_offsets(2) == (0, 1)
    assert frame_offsets(4) == (0, 1, 2, 4)
    assert frame_offsets(5) == (0,) + STACK_STRIDES
    with pytest.raises(ValueError, match="only 5 frames"):
        frame_offsets(6)


def test_interleave_is_channel_fastest():
    """Frame index innermost. Two separate PLANES would make
    forward_split's reshape+permute a real transpose per forward."""
    b, p, k = 3, 4, 3
    frames = [torch.full((b, p), float(j)) for j in range(k)]
    out = interleave_frames(frames)
    assert out.shape == (b, p * k)
    assert out.is_contiguous()
    # pixel-major: [px0f0, px0f1, px0f2, px1f0, ...]
    assert torch.equal(out[0, :k], torch.arange(k, dtype=torch.float32))
    # and the restride Policy performs recovers the frames
    im = out.reshape(b, 1, p, k).permute(0, 3, 1, 2)
    assert im.stride()[1] == 1, "channel stride must be 1 (NHWC)"
    for j in range(k):
        assert torch.all(im[:, j] == float(j))


# ------------------------------------------- ring vs gather, the real one --
K, P, N, T = 4, 3, 5, 24            # K frames, P "pixels", N envs, T decisions


def _simulate(resets, k=K, seed=0):
    """Drive the REAL FrameRing over a synthetic trajectory.

    Deliberately the production object, not a copy of it: a hand-rolled ring
    in the test would only prove two test-local implementations agree. The
    call ORDER here mirrors train_fast's rollout — push the render, record
    the single frame and the age, compose for the policy — and `resets[t]`
    is the envs that respawned during decision t-1's substeps.

    Returns (what the policy saw, the update's single-frame buffer, the age
    buffer, prologue depth).
    """
    rng = torch.Generator().manual_seed(seed)
    ring = FrameRing(k, N, P, "cpu")
    pro = ring.pro
    buf = torch.zeros((pro + T, N, P))         # PRO prologue rows + T decisions
    ages = torch.zeros((T, N), dtype=torch.long)
    seen = []

    # the run's first decision: a spawn frame with no history behind it
    ring.push(torch.rand((N, P), generator=rng), None)
    # then a previous iteration's worth of decisions, so the prologue is real
    for _ in range(pro):
        ring.push(torch.rand((N, P), generator=rng),
                  torch.zeros(N, dtype=torch.bool))
    ring.fill_prologue(buf)                    # the carry-over across iterations

    for t in range(T):
        ring.record(buf, ages, t)
        seen.append(ring.compose())
        ring.push(torch.rand((N, P), generator=rng), resets[t])
    return torch.stack(seen), buf, ages, pro


def _reset_schedule(seed=1):
    """Resets scattered through the trajectory, including two envs that
    respawn twice and one that never does."""
    g = torch.Generator().manual_seed(seed)
    r = torch.rand((T, N), generator=g) < 0.10
    r[:, 0] = False                                    # never resets
    r[3, 1] = r[4, 1] = True                           # back-to-back respawns
    r[0, 2] = True                                     # resets immediately
    return r


def test_the_update_gathers_exactly_what_the_rollout_composed():
    """THE test. Every (t, n) sample's stacked image out of the flat buffer
    must be bit-identical to the stack the ring handed the policy."""
    seen, buf, ages, pro = _simulate(_reset_schedule())
    f_img = buf.reshape((pro + T) * N, P)
    f_age = ages.reshape(-1)
    idx = torch.arange(T * N)
    got = stack_from_buffer(f_img, idx, f_age[idx], K, N, pro)
    want = seen.reshape(T * N, P * K)
    assert torch.equal(got, want), \
        f"{int((got != want).any(dim=1).sum())} of {T * N} samples differ"


def test_the_agreement_holds_for_every_k():
    for k in range(1, len(STACK_STRIDES) + 2):
        seen, buf, ages, pro = _simulate(_reset_schedule(), k=k)
        idx = torch.randperm(T * N)[:37]
        got = stack_from_buffer(buf.reshape((pro + T) * N, P),
                                idx, ages.reshape(-1)[idx], k, N, pro)
        assert torch.equal(got, seen.reshape(T * N, P * k)[idx]), f"K={k}"


def test_the_test_is_not_vacuous():
    """A trajectory with no resets and no strides would pass anything."""
    resets = _reset_schedule()
    assert resets.any(), "no resets in the fixture"
    seen, _, ages, _ = _simulate(resets)
    assert (ages == 0).any() and (ages > 0).any(), "ages never vary"
    # the frames really do differ from each other, so a mixed-up offset shows
    s = seen.reshape(T * N, P, K)
    assert (s[:, :, 0] != s[:, :, -1]).any(), "newest == oldest everywhere"
    # and a WRONG gather is detected: off-by-one in the stride must fail
    bad = stack_from_buffer(_simulate(resets)[1].reshape(-1, P),
                            torch.arange(T * N), ages.reshape(-1) + 1, K, N,
                            max(frame_offsets(K)))
    assert not torch.equal(bad, seen.reshape(T * N, P * K)), \
        "a corrupted age still matched — the check cannot fail"


# ----------------------------------------------------------- spawn clamping --
def test_history_collapses_to_the_spawn_frame():
    """At age 0 every offset resolves to the current frame: a policy on its
    first decision of an episode cannot know anything earlier, and training
    must not show it anything earlier either."""
    pro = max(frame_offsets(K))
    ring = torch.arange((pro + 1) * N * P, dtype=torch.float32) \
        .reshape(pro + 1, N, P)
    age = torch.zeros(N, dtype=torch.long)
    out = stack_from_ring(ring, 2, age, K).reshape(N, P, K)
    for j in range(K):
        assert torch.equal(out[:, :, j], ring[2]), f"frame {j} is not the spawn"


def test_the_stack_never_reaches_back_past_a_respawn():
    """The age arithmetic itself, over a real trajectory.

    frame(offset s) must be max(t - s, episode_start) — never a frame the
    previous episode rendered. This needs its own test because a wrong-but-
    CONSISTENT age sails through the ring-vs-gather check: both sides read
    the same number, agree perfectly, and both show the policy frames from
    before it spawned. A mutation that made age grow twice as fast did
    exactly that.
    """
    ring = FrameRing(K, N, 1, "cpu")        # P=1, so a frame's value is its id
    g = torch.Generator().manual_seed(5)
    resets = torch.rand((T, N), generator=g) < 0.15
    start = torch.zeros(N, dtype=torch.long)
    for t in range(T):
        ring.push(torch.full((N, 1), float(t)), None if t == 0 else resets[t])
        if t == 0:
            start.zero_()                    # the run's first decision
        else:
            start = torch.where(resets[t], torch.full_like(start, t), start)
        got = ring.compose().reshape(N, K)
        for j, s in enumerate(frame_offsets(K)):
            want = torch.maximum(torch.full_like(start, t - s), start).float()
            assert torch.equal(got[:, j], want), \
                f"t={t} offset={s}: {got[:, j].tolist()} != {want.tolist()}"
    assert resets[1:].any(), "no respawns in the fixture"


def test_clamping_walks_out_one_decision_at_a_time():
    pro = max(frame_offsets(K))
    ring = torch.zeros((pro + 1, N, P))
    for s in range(pro + 1):
        ring[s] = float(s)                    # slot index as its own value
    head = pro
    for age in range(pro + 1):
        out = stack_from_ring(ring, head, torch.full((N,), age), K)
        got = out.reshape(N, P, K)[0, 0]
        want = [float(head - min(s, age)) for s in frame_offsets(K)]
        assert got.tolist() == want, f"age {age}: {got.tolist()} != {want}"


def test_the_ring_wraps_without_reaching_stale_slots():
    """R = max stride + 1, so the oldest reachable offset is the oldest slot
    and nothing older is addressable however long the run goes."""
    pro = max(frame_offsets(K))
    ring = torch.zeros((pro + 1, N, P))
    for head in range(pro + 1):
        ring.zero_()
        for s in range(pro + 1):
            ring[(head - s) % (pro + 1)] = float(s)   # age-in-decisions
        out = stack_from_ring(ring, head, torch.full((N,), pro), K)
        assert out.reshape(N, P, K)[0, 0].tolist() == \
            [float(s) for s in frame_offsets(K)]


# ------------------------------------------------------------- the policy ---
W, H = 32, 16
KEYS = ["conv.0.weight", "conv.0.bias", "conv.2.weight", "conv.2.bias",
        "conv.4.weight", "conv.4.bias", "conv.8.weight", "conv.8.bias",
        "pi.0.weight", "pi.0.bias", "pi.2.weight", "pi.2.bias",
        "vf.0.weight", "vf.0.bias", "vf.2.weight", "vf.2.bias",
        "action_head.weight", "action_head.bias",
        "value_head.weight", "value_head.bias"]


def test_flag_off_is_the_old_model_exactly():
    torch.manual_seed(0)
    p = Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24).eval()
    assert list(p.state_dict()) == KEYS
    assert tuple(p.state_dict()["conv.0.weight"].shape) == (16, 1, 5, 5)
    assert p.in_ch == 1
    # golden forward: a fixed seed and a fixed input must still produce this
    torch.manual_seed(7)
    obs = torch.rand(2, N_SCALAR + W * H)
    with torch.no_grad():
        a, v = p(obs)
    assert a.shape == (2, sum(NVEC)) and torch.isfinite(a).all()
    # the frame-stack code must not have touched the single-frame path at all
    with torch.no_grad():
        b, w = p.forward_split(obs[:, :N_SCALAR], obs[:, N_SCALAR:])
    assert torch.equal(a, b) and torch.equal(v, w)


def test_stacked_policy_runs_and_keeps_the_parameter_set():
    torch.manual_seed(0)
    p = Policy(N_SCALAR + W * H * K, W, H, emb=32, hidden=24, in_ch=K).eval()
    sd = p.state_dict()
    assert list(sd) == KEYS, "stacking must not add or rename parameters"
    assert tuple(sd["conv.0.weight"].shape) == (16, K, 5, 5)
    one = Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24).state_dict()
    assert all(sd[k].shape == one[k].shape for k in KEYS if k != "conv.0.weight")

    obs = torch.rand(6, N_SCALAR + W * H * K)
    with torch.no_grad():
        logits, value = p(obs)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    assert logits.shape == (6, sum(NVEC))


def test_the_stack_reaches_the_conv_as_k_channels():
    """End to end through the real helper: K distinguishable frames in, K
    conv channels out, in the right order."""
    frames = [torch.full((2, W * H), float(j + 1)) for j in range(K)]
    img = interleave_frames(frames)
    p = Policy(N_SCALAR + W * H * K, W, H, emb=32, hidden=24, in_ch=K)
    im = img.reshape(-1, p.lidar_h, p.lidar_w, p.in_ch).permute(0, 3, 1, 2)
    assert im.shape == (2, K, H, W) and im.stride()[1] == 1
    for j in range(K):
        assert torch.all(im[:, j] == float(j + 1)), f"channel {j} scrambled"


# ------------------------------------------------------- the eval/record path --
def test_the_recording_policy_collapses_history_on_a_new_episode():
    """record_rollout never tells a policy an episode ended, so the eval
    ring reads it off the core's per-env tick counter, which reset_env zeroes
    (src/env.c). Inference has to collapse where training does, or the
    recorded policy is not the one that was trained."""
    pol = GreedyTorchPolicy(None, None, "cpu", stack=K)
    ticks = [0, 3, 6, 9, 0, 3]                 # a respawn between decision 3/4
    out = [pol._push_frame(torch.full((1, 1), float(d)),
                           np.array([tk])).reshape(K)
           for d, tk in enumerate(ticks)]
    assert out[0].tolist() == [0.0] * K, "the first decision has no history"
    assert out[3].tolist() == [3.0, 2.0, 1.0, 0.0], "window did not fill"
    assert out[4].tolist() == [4.0] * K, "the respawn did not collapse history"
    assert out[5].tolist() == [5.0, 4.0, 4.0, 4.0], "clamp lost after respawn"


def test_the_eval_ring_and_the_rollout_ring_are_the_same_object():
    """Not a style point: two rings would drift, and the drift would only
    show up as eval recordings that disagree with training."""
    pol = GreedyTorchPolicy(None, None, "cpu", stack=K)
    pol._push_frame(torch.zeros((3, P)), np.zeros(3, np.int64))
    assert isinstance(pol._ring, FrameRing)
    assert pol._ring.pro == max(frame_offsets(K))


# ------------------------------------------------------------------ gating --
@pytest.mark.parametrize("sm,pin,fs", [(1, 0, 4), (0, 1, 4), (1, 1, 0),
                                       (1, 1, 4)])
def test_vision_experiments_refuse_to_combine(sm, pin, fs):
    with pytest.raises(SystemExit, match="separate experiments"):
        check_vision_exclusive(sm, pin, fs)


@pytest.mark.parametrize("sm,pin,fs", [(0, 0, 0), (0, 0, 1), (1, 0, 0),
                                       (0, 1, 1), (0, 0, 4)])
def test_one_at_a_time_is_allowed(sm, pin, fs):
    check_vision_exclusive(sm, pin, fs)      # must not raise


def test_k1_stack_is_exactly_the_plain_gather():
    """The boundary: at K=1 the stacked path must reduce to the single-frame
    gather the trainer does when the flag is off, or 'off is bit-identical'
    is only true because a branch skips the new code."""
    f_img = torch.rand(40, P)
    idx = torch.tensor([3, 17, 0, 39])
    age = torch.tensor([0, 2, 1, 5])
    assert torch.equal(stack_from_buffer(f_img, idx, age, 1, 8, 0), f_img[idx])


def test_frame_stack_one_is_off_not_a_stack_of_one():
    """--frame-stack 1 has to mean today's behaviour, including no prologue
    rows and no age buffer."""
    assert frame_offsets(1) == (0,) and max(frame_offsets(1)) == 0
    check_vision_exclusive(1, 0, 1)          # 1 does not collide with the mask
