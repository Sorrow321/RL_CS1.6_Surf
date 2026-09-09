"""``--keys-hold``: the movement keys become HELD STATE, not per-decision presses.

The absolute continuous view works because the policy commands where the view
IS rather than how much to turn it this tick.  ``--keys-hold`` applies that
logic to the fwd, side and duck keys: each of those three heads gains a
"keep" bin at index 0, the held value is carried per env across decisions and
resolved back into the engine's own absolute row in Python, and the policy
observes what it is holding as 7 columns at the tail of the scalar block.
Jump is deliberately untouched - an impulse, not a held state.

Five ways this could be silently wrong, each of which makes a one-hour arm
unreadable or, worse, plausible-but-wrong:

  * **the flag OFF must be bit-identical to what shipped** - same NVEC, same
    state_dict, same logits on the same input;
  * **the head sizes must be n+1 on exactly the three held heads**;
  * **"keep" must actually keep** - an all-keep sequence reproduces a
    continuously held key, and a change bin sets the value and persists;
  * **every episode start must reset the held state** - the engine's buttons
    do, so a policy that believed it was still holding A after a reservoir
    respawn would be acting on a state that does not exist;
  * **the observation must equal the RESOLVED state**, and the ``boot`` copy
    the truncation bootstrap reads must lag exactly one resolve.

Plus the two contract items that have cost real time before: the flag is
recorded in ``run.json`` and a resume that disagrees with the checkpoint is
REFUSED (the head widths differ, so there is no warm start in either
direction).

    python -m pytest tests/python/test_keys_hold.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import train_fast                                            # noqa: E402
from surfgym.core import ACTION_NVEC                         # noqa: E402
from surfgym.keyshold import (HELD_HEADS, KEEP, KeysHold,    # noqa: E402
                              N_FEATURES, NEUTRAL, nvec_with_keep)

SRC = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
NACT = 6


@pytest.fixture(autouse=True)
def _restore_nvec():
    """Every test runs against a known NVEC and leaves it at the default.

    ``set_keys_hold`` mutates a module global; a test that left it widened
    would silently change every later test in the same process.
    """
    train_fast.set_keys_hold(False)
    yield
    train_fast.set_keys_hold(False)


# --------------------------------------------------------------------------
# 1. head sizes
# --------------------------------------------------------------------------
def test_default_nvec_is_the_engine_action_space():
    assert train_fast.NVEC == ACTION_NVEC == (15, 7, 3, 3, 2, 2)
    assert train_fast.NVEC_CORE == ACTION_NVEC


def test_keep_bin_widens_exactly_the_three_held_heads():
    assert nvec_with_keep(ACTION_NVEC) == (15, 7, 4, 4, 2, 3)
    train_fast.set_keys_hold(True)
    got = train_fast.NVEC
    for h in range(NACT):
        want = ACTION_NVEC[h] + (1 if h in HELD_HEADS else 0)
        assert got[h] == want, f"head {h}: {got[h]} != {want}"
    # jump (head 4) is untouched on purpose - an impulse, not a held state
    assert got[4] == ACTION_NVEC[4] == 2
    # the padded logit table must keep its width, or every buffer moves
    assert max(got) == train_fast.NPAD == 15


def test_set_keys_hold_is_reversible():
    train_fast.set_keys_hold(True)
    assert train_fast.NVEC == (15, 7, 4, 4, 2, 3)
    train_fast.set_keys_hold(False)
    assert train_fast.NVEC == ACTION_NVEC


# --------------------------------------------------------------------------
# 2. flag OFF is bit-identical
# --------------------------------------------------------------------------
OBS_DIM = train_fast.N_SCALAR + 8 * 4        # 15 scalars + an 8x4 depth image


def _policy(seed=0, obs=OBS_DIM):
    torch.manual_seed(seed)
    return train_fast.Policy(obs, 8, 4, emb=16, hidden=16, in_ch=1)


def test_flag_off_state_dict_and_forward_are_bit_identical():
    """Toggling the flag on and back off must leave nothing behind."""
    a = _policy()
    train_fast.set_keys_hold(True)
    _ = _policy()                       # build a WIDE policy in between
    train_fast.set_keys_hold(False)
    b = _policy()
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    x = torch.randn(4, OBS_DIM)
    la, va = a(x)
    lb, vb = b(x)
    assert torch.equal(la, lb) and torch.equal(va, vb)


def test_action_head_width_tracks_the_flag():
    off = _policy()
    assert off.action_head.out_features == sum(ACTION_NVEC)
    train_fast.set_keys_hold(True)
    on = _policy()
    assert on.action_head.out_features == sum((15, 7, 4, 4, 2, 3))
    assert on.action_head.out_features == off.action_head.out_features + 3


def test_headpacker_follows_the_flag():
    dev = torch.device("cpu")
    train_fast.set_keys_hold(True)
    pk = train_fast.HeadPacker(dev)
    padded = pk.pad(torch.zeros(2, sum(train_fast.NVEC)))
    assert padded.shape == (2, NACT, 15)
    # exactly NVEC[h] live slots per head, the rest at NEG
    live = (padded[0] > train_fast.NEG / 2).sum(-1).tolist()
    assert live == list(train_fast.NVEC)


# --------------------------------------------------------------------------
# 3. resolve: keep keeps, a change bin sets and persists
# --------------------------------------------------------------------------
def _row(fwd=KEEP, side=KEEP, duck=KEEP, yaw=7, pitch=3, jump=0):
    a = np.zeros((1, NACT), np.int32)
    a[0] = (yaw, pitch, fwd, side, jump, duck)
    return a


def test_all_keep_reproduces_a_continuously_held_key():
    k = KeysHold(1)
    # press D (side engine bin 2 -> policy bin 3) once, then keep forever
    out = k.resolve(_row(side=3))
    assert out[0, 3] == 2
    for _ in range(50):
        out = k.resolve(_row())
        assert out[0, 3] == 2, "a keep bin must reproduce the held key"
    # ... and the untouched heads came back neutral, not zero
    assert out[0, 2] == NEUTRAL[0] and out[0, 5] == NEUTRAL[2]


def test_a_change_bin_sets_the_value_and_it_persists():
    k = KeysHold(1)
    # policy bin b (b >= 1) is engine bin b - 1, in the engine's own order
    for pol, eng in ((1, 0), (2, 1), (3, 2)):
        out = k.resolve(_row(fwd=pol))
        assert out[0, 2] == eng
        for _ in range(3):
            assert k.resolve(_row())[0, 2] == eng
    # duck is a 2-bin head: policy 1 -> off, policy 2 -> on
    assert k.resolve(_row(duck=2))[0, 5] == 1
    assert k.resolve(_row())[0, 5] == 1
    assert k.resolve(_row(duck=1))[0, 5] == 0


def test_jump_and_the_view_heads_pass_through_untouched():
    k = KeysHold(1)
    out = k.resolve(_row(yaw=11, pitch=5, jump=1, side=2))
    assert (out[0, 0], out[0, 1], out[0, 4]) == (11, 5, 1)
    # jump is NOT held: the next decision's own jump bin is what lands
    out = k.resolve(_row(yaw=0, pitch=0, jump=0))
    assert out[0, 4] == 0


def test_resolve_is_per_env():
    k = KeysHold(3)
    a = np.zeros((3, NACT), np.int32)
    a[:, 3] = (1, 3, KEEP)          # A, D, keep(neutral)
    out = k.resolve(a)
    assert out[:, 3].tolist() == [0, 2, 1]
    b = np.zeros((3, NACT), np.int32)   # all keep
    assert k.resolve(b)[:, 3].tolist() == [0, 2, 1]


def test_every_resolved_value_is_a_legal_engine_bin():
    rng = np.random.default_rng(0)
    n = 64
    k = KeysHold(n)
    wide = np.asarray(nvec_with_keep(ACTION_NVEC))
    for _ in range(200):
        a = (rng.random((n, NACT)) * wide).astype(np.int32)
        out = k.resolve(a)
        assert (out >= 0).all()
        assert (out < np.asarray(ACTION_NVEC)).all()


# --------------------------------------------------------------------------
# 4. the reset at every episode start
# --------------------------------------------------------------------------
def test_reset_collapses_to_neutral_for_the_masked_rows_only():
    k = KeysHold(4)
    a = np.zeros((4, NACT), np.int32)
    a[:, 2] = 1          # fwd -> engine 0 (S) on every env
    a[:, 3] = 3          # side -> engine 2 (D)
    a[:, 5] = 2          # duck -> engine 1 (on)
    k.resolve(a)
    assert k.state.tolist() == [[0, 2, 1]] * 4
    k.reset(np.array([True, False, True, False]))
    assert k.state.tolist() == [list(NEUTRAL), [0, 2, 1],
                                list(NEUTRAL), [0, 2, 1]]
    # and a keep after the reset holds NEUTRAL, not the pre-reset key
    out = k.resolve(np.zeros((4, NACT), np.int32))
    assert out[0, 3] == NEUTRAL[1] and out[1, 3] == 2


def test_reset_none_resets_everything_and_a_fresh_object_is_neutral():
    k = KeysHold(2)
    assert k.state.tolist() == [list(NEUTRAL)] * 2
    k.resolve(np.full((2, NACT), 2, np.int32))
    assert k.state.tolist() != [list(NEUTRAL)] * 2
    k.reset(None)
    assert k.state.tolist() == [list(NEUTRAL)] * 2


def test_the_trainer_resets_on_ended_acc_which_covers_every_spawn_source():
    """The reset must ride ``ended_acc``, next to ``obs_aux.reset``.

    ``ended_acc`` is the decision's accumulated (done | trunc) mask, and in
    this trainer EVERY start source raises it: the autoreset, a reservoir
    respawn (respawn_frac draws the new episode's start), a stall kill
    (force_fail -> done on the next step) and a demo start.  Pinning the call
    site is what stops a later refactor from resetting on ``done`` alone,
    which would leave a truncated episode holding the previous one's keys.
    """
    # the CALL, not the comment that names it (indentation distinguishes)
    call = "                    keys.reset(ended_acc)"
    assert call in SRC
    i_boot = SRC.index("keys.boot_features(ti)")
    i_reset = SRC.index(call)
    i_fill = SRC.index("fill_vision(static_obs, b_done[t]")
    assert i_boot < i_reset < i_fill, (
        "the reset must run AFTER the truncation bootstrap reads boot and "
        "BEFORE fill_vision builds the fresh episode's first observation")


# --------------------------------------------------------------------------
# 5. the observation columns and the boot copy
# --------------------------------------------------------------------------
def test_features_are_the_resolved_state_as_one_hots():
    k = KeysHold(1)
    f = k.features()
    assert f.shape == (1, N_FEATURES) == (1, 7)
    # neutral: fwd one-hot at 1, side one-hot at 1 (offset 3), duck 0
    assert f[0].tolist() == [0, 1, 0, 0, 1, 0, 0]
    k.resolve(_row(fwd=1, side=3, duck=2))       # engine S, D, duck on
    assert k.features()[0].tolist() == [1, 0, 0, 0, 0, 1, 1]
    # one-hot, always: exactly one fwd bit and one side bit
    rng = np.random.default_rng(1)
    wide = np.asarray(nvec_with_keep(ACTION_NVEC))
    for _ in range(50):
        k.resolve((rng.random((1, NACT)) * wide).astype(np.int32))
        f = k.features()
        assert f[0, 0:3].sum() == 1.0 and f[0, 3:6].sum() == 1.0
        assert f[0, 6] in (0.0, 1.0)


def test_features_out_is_filled_in_place_and_matches():
    k = KeysHold(3)
    k.resolve(np.full((3, NACT), 2, np.int32))
    buf = np.full((3, N_FEATURES), 9.0, np.float32)
    got = k.features(out=buf)
    assert got is buf
    assert np.array_equal(buf, k.features())


def test_boot_lags_exactly_one_resolve():
    """``boot`` is the state as of the LAST resolve.

    The truncation bootstrap runs after the engine has already autoreset the
    rows that ended, so ``state`` is about to be collapsed; ``boot`` is what
    ``s_T`` was holding.  It must therefore be written by ``resolve`` and
    survive ``reset``.
    """
    k = KeysHold(1)
    k.resolve(_row(side=3))                     # hold D
    assert k.boot.tolist() == k.state.tolist() == [[1, 2, 0]]
    k.reset(np.array([True]))                   # the episode ended
    assert k.state.tolist() == [list(NEUTRAL)]  # the fresh spawn
    assert k.boot.tolist() == [[1, 2, 0]]       # ... but s_T held D
    assert k.boot_features(np.array([0]))[0].tolist() == [0, 1, 0, 0, 0, 1, 0]
    # the NEXT resolve re-arms it
    k.resolve(_row(fwd=1))
    assert k.boot.tolist() == [[0, 1, 0]]


def test_boot_features_selects_rows():
    k = KeysHold(4)
    a = np.zeros((4, NACT), np.int32)
    a[:, 3] = (1, 2, 3, KEEP)
    k.resolve(a)
    got = k.boot_features(np.array([0, 2]))
    assert got.shape == (2, N_FEATURES)
    assert got[0, 3:6].tolist() == [1, 0, 0]     # engine A
    assert got[1, 3:6].tolist() == [0, 0, 1]     # engine D


# --------------------------------------------------------------------------
# 6. wiring: the obs block, the resolve site, the exclusions
# --------------------------------------------------------------------------
def test_the_obs_block_is_seven_columns_at_the_tail():
    assert "N_KEYS = keyshold.N_FEATURES if args.keys_hold else 0" in SRC
    assert ("N_ROUTE = N_FAN + N_LATCH + N_AUX + N_CC + N_KEYS + N_RATCHET"
            in SRC)
    assert "KEYS0 = N_SCALAR + N_FAN + N_LATCH + N_AUX + N_CC" in SRC
    # --race-ratchet keeps the trailing slot: it CAN be grown onto a
    # checkpoint by widen_for_obs' zero-pad and --keys-hold never can, so the
    # one block that needs the tail keeps it (and the ratchet arm's own
    # pinned test stays green)
    assert "RATCHET_COL = N_SCALAR + N_ROUTE - 1" in SRC
    # ... and the three writers agree on that order
    for txt in (SRC, SRC):
        i_k = txt.index("dst[:, KEYS0:KEYS0 + N_KEYS].copy_(keys_pin")
        i_r = txt.index("dst[:, RATCHET_COL:RATCHET_COL + 1].copy_(ratchet_pin")
        assert i_k < i_r
    assert (SRC.index("keys.boot_features(ti)")
            < SRC.index("gp = fleet.terminal_ratchet(ti, pos_np)"))
    assert (SRC.index("self.keys.features(), dtype=torch.float32")
            < SRC.index("self.ratchet_fn(self.core), dtype=torch.float32"))
    assert "dst[:, KEYS0:KEYS0 + N_KEYS].copy_(keys_pin" in SRC


def test_resolve_runs_between_the_policy_draw_and_the_engine_step():
    i_copy = SRC.index("np.copyto(act_np32, act_pin.numpy()")
    i_res = SRC.index("keys.resolve(act_np32)")
    i_step = SRC.index("o2, base_r, done, trunc, term_obs = fleet.step(")
    assert i_copy < i_res < i_step
    # b_act records the POLICY's draw (what PPO's ratio is over), and it is
    # written BEFORE the resolve mutates the row
    assert SRC.index("b_act[t].copy_(static_act") < i_res


def test_flag_is_off_by_default_saved_restored_and_a_mismatch_is_refused():
    assert '"--keys-hold", action="store_const", const=1, default=None' in SRC
    assert "args.keys_hold = bool(args.keys_hold)" in SRC
    assert '"keys_hold": bool(args.keys_hold),' in SRC
    assert 'ck_cfg.get("keys_hold")' in SRC
    assert '("keys_hold", "--keys-hold"),' in SRC
    assert "--keys-hold changes the action head's SIZE" in SRC


def test_the_incompatible_flags_are_refused():
    for frag in ("--keys-hold with --chunk",
                 "--keys-hold with --mask-forward-air",
                 "--keys-hold with --yaw-cond",
                 "--keys-hold with --bc-file"):
        assert frag in SRC, frag


def test_the_launcher_carries_keys_hold_on_the_scratch_line_only():
    """SUPERSEDES "the launcher carries no --keys-hold of its own".

    User decision 2026-09-09: ``--keys-hold`` is the FROM-SCRATCH default
    (see tests/python/test_keys_pot_default.py for the full contract).  It
    stays out of every WARM path, because it changes the action head's SIZE
    and a resumed checkpoint must keep restoring its own.  A control arm is
    now ``KEYS=off``.
    """
    sh = (ROOT / "tools" / "run_arm.sh").read_text(encoding="utf-8")
    ps = (ROOT / "tools" / "launch_local.ps1").read_text(encoding="utf-8")
    assert 'KEYS="${KEYS:-hold}"' in sh
    assert "$KEYS = if ($env:KEYS) { $env:KEYS } else { \"hold\" }" in ps
    # and the resume half of each launcher never PASSES it: the warm branch
    # may only say out loud that a resume ignores KEYS
    warm = sh[sh.index("# ARM_RESUME=1:"):]
    wargs = warm[warm.index('ARGS=(--ckpt "$CKPT"'):]
    wargs = wargs[:wargs.index('"$@")')]
    assert "keys-hold" not in wargs and "keys_hold" not in wargs
    assert "KEYS_ARGS" not in warm
    resume = ps[ps.index('    "resume" {'):
                ps.index('    default { throw "unknown preset')]
    assert "keys-hold" not in resume and "KEYSARGS" not in resume


def test_eval_and_recorder_resolve_the_same_way():
    assert "keys_hold=args.keys_hold" in SRC
    assert "self.keys.resolve(self._held)" in SRC
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert "train_fast.set_keys_hold(keys_hold)" in rc
    assert "keys_hold=keys_hold" in rc
    assert "route_dim += train_fast.keyshold.N_FEATURES" in rc
    # the two planner tools must REFUSE rather than mis-decode
    bt = (ROOT / "tools" / "beam_tas.py").read_text(encoding="utf-8")
    assert '"keys_hold"' in bt and "UNSUPPORTED" in bt
    pb = (ROOT / "tools" / "plan_to_bc.py").read_text(encoding="utf-8")
    assert 'cfg.get("keys_hold")' in pb


def test_act_diagnostics_read_the_resolved_keys():
    """``act/strafe_flip`` counts KEY changes, not decision changes.

    Under --keys-hold a run of "keep" decisions is one held key, and a
    diagnostic that counted policy bins would report the flip rate of the
    thing this arm exists to measure as if nothing had changed.
    """
    assert "_flip = ((_aS[1:] != _aS[:-1]) & _pair)" in SRC
    assert "_aS = b_kact[:, :, 1].long()" in SRC
    assert "_aS = b_act[:, :, H_SIDE]" in SRC      # the flag-off path
