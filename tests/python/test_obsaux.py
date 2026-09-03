"""--act-hist and --obs-compass (surfgym/obsaux.py).

The three things that can silently go wrong with a handcrafted observation
block, and one test each:

* it does not COLLAPSE at an episode start, so the first decision of a fresh
  episode reads the last decisions of the previous one;
* its EGO frame disagrees in sign with the fan and the velocity scalars, so
  "the finish is to my left" and "my velocity points left" are opposite
  numbers and the policy has to unlearn one of them;
* the trainer writes it and tools/record_ckpt.py does not, so evals measure
  a different network input than training wrote - which lands on
  race/eval_progress, the number every arm is judged by.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.obsaux import (ACT_FEAT, CMP_FEAT, ObsAux, pitch_hist_table,
                            yaw_hist_table)


# ---------------------------------------------------------------- --act-hist
def test_history_is_zero_before_any_decision():
    aux = ObsAux(4, k=3)
    assert aux.n_features == 3 * ACT_FEAT
    assert np.array_equal(aux.features(), np.zeros((4, 3 * ACT_FEAT),
                                                   np.float32))


def test_history_shifts_over_three_decisions_most_recent_first():
    aux = ObsAux(2, k=3)
    a1 = np.array([[14, 6, 2, 0, 1, 0]] * 2, np.int32)   # +max yaw, +max pitch
    a2 = np.array([[0, 0, 0, 2, 0, 1]] * 2, np.int32)    # -max yaw, -max pitch
    a3 = np.array([[7, 3, 1, 1, 0, 0]] * 2, np.int32)    # NEUTRAL_ACT
    for a in (a1, a2, a3):
        aux.push(a)
    rows = aux.features().reshape(2, 3, ACT_FEAT)
    # slot 0 is the MOST RECENT decision
    assert np.allclose(rows[:, 0], aux.encode(a3))
    assert np.allclose(rows[:, 1], aux.encode(a2))
    assert np.allclose(rows[:, 2], aux.encode(a1))
    # and the neutral action is the all-zero row, by construction
    assert np.allclose(rows[:, 0], 0.0)


def test_a_fourth_decision_drops_the_oldest():
    aux = ObsAux(1, k=3)
    for a in ([14, 3, 1, 1, 0, 0], [0, 3, 1, 1, 0, 0], [13, 3, 1, 1, 0, 0],
              [1, 3, 1, 1, 0, 0]):
        aux.push(np.array([a], np.int32))
    rows = aux.features().reshape(1, 3, ACT_FEAT)
    assert rows[0, 0, 0] == pytest.approx(-0.7)     # bin 1  = -7/10
    assert rows[0, 1, 0] == pytest.approx(0.7)      # bin 13 = +7/10
    assert rows[0, 2, 0] == pytest.approx(-1.0)     # bin 0  = -10/10
    # the very first decision (bin 14, +1.0) has fallen off the end
    assert not np.isclose(rows[:, :, 0], 1.0).any()


def test_history_is_zeroed_at_every_episode_start():
    aux = ObsAux(3, k=2)
    aux.push(np.array([[14, 6, 2, 2, 1, 1]] * 3, np.int32))
    aux.push(np.array([[14, 6, 2, 2, 1, 1]] * 3, np.int32))
    assert aux.features().any()
    aux.reset(np.array([True, False, True]))
    rows = aux.features()
    assert not rows[0].any() and not rows[2].any()
    assert rows[1].any()


def test_the_eval_path_collapses_history_on_the_tick_counter():
    """record_rollout never tells a policy an episode ended, so the eval
    reads the core's per-env tick counter going backwards - the same signal
    the frame ring uses (tests/python/test_framestack.py)."""
    aux = ObsAux(2, k=2)
    seen = []
    for tick, act in ((np.array([0, 0]), [14, 6, 2, 2, 1, 1]),
                      (np.array([3, 3]), [14, 6, 2, 2, 1, 1]),
                      (np.array([6, 6]), [14, 6, 2, 2, 1, 1]),
                      (np.array([0, 9]), [14, 6, 2, 2, 1, 1])):
        seen.append(aux.eval_features(np.zeros((2, 3)), np.zeros(2),
                                      tick).copy())
        aux.push(np.array([act] * 2, np.int32))
    assert not seen[0].any()                 # nothing decided yet
    assert seen[2].all()                     # two full decisions of history
    assert not seen[3][0].any()              # env 0 respawned
    assert seen[3][1].all()                  # env 1 did not


@pytest.mark.parametrize("yaw_adaptive", [False, True])
def test_every_encoded_value_is_in_minus_one_to_one(yaw_adaptive):
    from surfgym.core import ACTION_NVEC
    aux = ObsAux(1, k=1, yaw_adaptive=yaw_adaptive)
    rng = np.random.default_rng(0)
    acts = np.stack([rng.integers(0, n, 4096) for n in ACTION_NVEC], axis=1)
    # plus the corners, which random sampling can miss
    acts = np.concatenate([acts, np.zeros((1, 6), np.int64),
                           np.array(ACTION_NVEC, np.int64)[None] - 1])
    enc = aux.encode(acts.astype(np.int32))
    assert enc.dtype == np.float32
    assert enc.min() >= -1.0 and enc.max() <= 1.0
    # the tables reach BOTH ends, or the normalisation is not the bin's own
    assert enc[:, 0].min() == pytest.approx(-1.0)
    assert enc[:, 0].max() == pytest.approx(1.0)
    assert enc[:, 1].min() == pytest.approx(-1.0)
    assert enc[:, 1].max() == pytest.approx(1.0)


def test_the_neutral_bins_encode_to_zero():
    """NEUTRAL_ACT is 'hold everything', and the history has to say so with
    the same number it uses for 'no history yet' - otherwise a fresh episode
    and a coasting one are different inputs for no physical reason."""
    from train_fast import NEUTRAL_ACT
    aux = ObsAux(1, k=1)
    assert np.allclose(aux.encode(np.array([NEUTRAL_ACT], np.int32)), 0.0)


def test_yaw_table_follows_the_engine_ladder():
    from surfgym.core import PITCH_BINS, YAW_BINS
    assert np.allclose(yaw_hist_table(False), np.asarray(YAW_BINS) / 10.0)
    assert np.allclose(pitch_hist_table(), np.asarray(PITCH_BINS) / 10.0)
    # --yaw-adaptive: the bin means a MULTIPLE of the optimal-strafe rate
    # (src/env.c K_BINS), so a different table but the same [-1, 1] range
    ka = yaw_hist_table(True)
    assert ka[7] == 0.0 and ka[0] == pytest.approx(-1.0)
    assert ka[-1] == pytest.approx(1.0)
    assert np.all(np.diff(ka) > 0)           # ascending, like the engine's


# ------------------------------------------------------------ --obs-compass
class _LinearField:
    """d(x) = |x - goal| measured along +x only: a field whose downhill
    direction is exactly -x everywhere, with a sentinel region so the
    unreachable branch is exercised."""

    def __init__(self, goal_x=1000.0, cell=32.0, reach_max=1e5,
                 sentinel_above_y=None):
        self.goal_x = float(goal_x)
        self.cell = float(cell)
        self.reach_max = float(reach_max)
        self.sentinel = self.reach_max + 2.0 * self.cell
        self._sy = sentinel_above_y

    def sample(self, pos):
        p = np.atleast_2d(np.asarray(pos, np.float64))
        d = np.abs(p[:, 0] - self.goal_x)
        if self._sy is not None:
            d = np.where(p[:, 1] > self._sy, self.sentinel, d)
        return d.astype(np.float32)

    def reachable(self, pos):
        return self.sample(pos) < self.reach_max - 0.5 * self.cell


def _compass(yaw_deg, pos=(0.0, 0.0, 0.0), field=None):
    aux = ObsAux(1, k=0, field=(field or _LinearField()))
    return aux.compass_features(np.array([pos], np.float64),
                                np.array([yaw_deg], np.float64))[0]


@pytest.mark.parametrize("yaw,want_fwd,want_left", [
    (0.0, 1.0, 0.0),          # facing +x, the finish is dead ahead
    (90.0, 0.0, -1.0),        # facing +y, the finish is to my RIGHT
    (180.0, -1.0, 0.0),       # facing -x, the finish is behind me
    (270.0, 0.0, 1.0),        # facing -y, the finish is to my LEFT
])
def test_compass_points_the_right_way_in_the_ego_frame(yaw, want_fwd,
                                                       want_left):
    """Sign convention is route.RouteLine.features' and src/env.c's:
    forward = (cos yaw, sin yaw), left = (-sin yaw, cos yaw). Pinned at the
    four cardinals so a transposed or negated rotation cannot pass."""
    f = _compass(yaw)
    assert f[0] == pytest.approx(want_fwd, abs=1e-5)
    assert f[1] == pytest.approx(want_left, abs=1e-5)
    assert f[2] == pytest.approx(0.0, abs=1e-5)


def test_compass_agrees_with_the_route_fan_on_the_same_geometry():
    """Not a restatement of the test above: it drives the ACTUAL fan code
    with the same direction and compares column for column, so the two
    ego frames cannot drift apart later."""
    torch = pytest.importorskip("torch")
    from surfgym.route import RouteLine
    # a straight line along +x, i.e. the same direction the field descends
    line = np.stack([np.linspace(0.0, 4096.0, 64),
                     np.zeros(64), np.zeros(64)], axis=1)
    rl = RouteLine(line, offsets=(1.0,))
    for yaw in (0.0, 37.0, 90.0, 180.0, 271.0):
        fan = rl.features(torch.zeros(1, 3), torch.tensor([yaw]),
                          torch.tensor([1000.0])).numpy()[0]
        cmp_ = _compass(yaw)
        # fan columns 3..5 are the first lookahead point's (fwd, left, up)
        v = fan[3:6]
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        assert np.allclose(v[:2], cmp_[:2], atol=1e-4), yaw


def test_compass_reads_zero_on_the_unreachable_sentinel():
    fld = _LinearField(sentinel_above_y=100.0)
    ok = _compass(0.0, pos=(0.0, 0.0, 0.0), field=fld)
    bad = _compass(0.0, pos=(0.0, 500.0, 0.0), field=fld)
    assert ok.any()
    assert np.array_equal(bad, np.zeros(CMP_FEAT, np.float32))


def test_compass_gradient_magnitude_is_one_on_a_true_distance_field():
    f = _compass(0.0)
    assert f[4] == pytest.approx(1.0, abs=1e-4)


def test_compass_d_over_d0_starts_at_one_and_falls_toward_the_goal():
    aux = ObsAux(1, k=0, field=_LinearField(goal_x=1000.0))
    first = aux.compass_features(np.array([[0.0, 0.0, 0.0]]),
                                 np.zeros(1))[0].copy()
    assert first[3] == pytest.approx(1.0)
    later = aux.compass_features(np.array([[750.0, 0.0, 0.0]]),
                                 np.zeros(1))[0]
    assert later[3] == pytest.approx(0.25, abs=1e-4)
    # a new episode re-anchors d0 on wherever it spawned
    aux.reset(np.array([True]))
    again = aux.compass_features(np.array([[750.0, 0.0, 0.0]]),
                                 np.zeros(1))[0]
    assert again[3] == pytest.approx(1.0)


def test_the_bootstrap_read_does_not_re_anchor_d0():
    """The truncation bootstrap evaluates a RECONSTRUCTED terminal pose for
    rows whose live state has already autoreset. latch=False keeps it from
    stamping that pose as the live episodes' d0."""
    aux = ObsAux(1, k=0, field=_LinearField(goal_x=1000.0))
    aux.compass_features(np.array([[0.0, 0.0, 0.0]]), np.zeros(1))
    d0 = aux._d0.copy()
    aux.compass_features(np.array([[900.0, 0.0, 0.0]]), np.zeros(1),
                         latch=False)
    assert np.array_equal(aux._d0, d0)


def test_a_flat_field_reads_zero_direction_and_zero_magnitude():
    class _Flat:
        cell = 32.0

        def sample(self, pos):
            return np.zeros(len(np.atleast_2d(pos)), np.float32)

    f = _compass(0.0, field=_Flat())
    assert np.allclose(f[:3], 0.0)
    assert f[4] == pytest.approx(0.0)


def test_per_slot_fields_sample_each_env_against_its_own_map():
    a, b = _LinearField(goal_x=1000.0), _LinearField(goal_x=-1000.0)
    aux = ObsAux(2, k=0, field=[(slice(0, 1), a), (slice(1, 2), b)])
    f = aux.compass_features(np.zeros((2, 3)), np.zeros(2))
    assert f[0, 0] == pytest.approx(1.0, abs=1e-5)     # +x for slot 0
    assert f[1, 0] == pytest.approx(-1.0, abs=1e-5)    # -x for slot 1


def test_compass_refuses_a_short_sample():
    """GoalDistField.sample is row-aligned to envs, so a short array would
    silently measure env i against env j's goal."""
    aux = ObsAux(4, k=0, field=_LinearField())
    with pytest.raises(ValueError):
        aux.compass_features(np.zeros((2, 3)), np.zeros(2))


# ------------------------------------------------------------------- layout
def test_the_block_is_history_then_compass():
    aux = ObsAux(1, k=2, field=_LinearField())
    assert aux.n_features == 2 * ACT_FEAT + CMP_FEAT
    aux.push(np.array([[14, 3, 1, 1, 0, 0]], np.int32))
    row = aux.features(np.zeros((1, 3)), np.zeros(1))[0]
    assert row[0] == pytest.approx(1.0)              # newest yaw, first col
    assert np.allclose(row[ACT_FEAT:2 * ACT_FEAT], 0.0)   # only one decision
    assert row[2 * ACT_FEAT] == pytest.approx(1.0)   # compass fwd
    assert row[2 * ACT_FEAT + 3] == pytest.approx(1.0)    # d/d0 at the start


def test_features_writes_into_a_caller_buffer():
    """The trainer hands it a pinned host row, so the per-decision path must
    fill in place rather than allocate."""
    aux = ObsAux(2, k=1, field=_LinearField())
    buf = np.full((2, aux.n_features), 7.0, np.float32)
    got = aux.features(np.zeros((2, 3)), np.zeros(2), out=buf)
    assert got is buf
    assert not np.any(buf == 7.0)


# ------------------------------------------------- the trainer and the tools
def test_the_flags_exist_default_off_and_are_recorded():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--act-hist", type=int, default=None' in src
    assert '"--obs-compass", type=int, default=None' in src
    assert '"act_hist": int(args.act_hist or 0),' in src
    assert '"obs_compass": int(args.obs_compass or 0),' in src
    # the width, and the ONE place the block's column offset is defined
    assert "N_HIST = ACT_FEAT * int(args.act_hist or 0)" in src
    assert "N_ROUTE = N_FAN + N_LATCH + N_AUX" in src
    assert "AUX0 = N_SCALAR + N_FAN + N_LATCH" in src
    # the latch keeps its own column now that the block grew past it
    assert "LATCH_COL = N_SCALAR + N_FAN + N_LATCH - 1" in src
    assert "dst[:, SCAL - 1:SCAL]" not in src
    # pushed once per DECISION, reset on every episode end, and rebuilt for
    # V(s_T) - the three places a history feature goes wrong
    assert "obs_aux.push(act_np32)" in src
    assert "obs_aux.reset(ended_acc)" in src
    assert "aux_t = obs_aux.features(p_t, y_t, latch=False)" in src
    # and it reads the FLEET's pose, not slot 0's: sv_view is one map's
    # states, so --maps would hand the compass a short array
    assert "p_t = np.array(vis_np[:, 0:3], np.float64)" in src


def test_the_recorder_mirrors_the_two_new_keys():
    """tools/record_ckpt.py refuses to emit a trajectory for a config key it
    never mentions (its audit_cfg gate). A new observation flag is exactly
    the case that guard exists for."""
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    known = record_ckpt._mentioned_keys()
    for k in ("act_hist", "obs_compass"):
        assert k in known and k not in record_ckpt.TRAIN_ONLY, k
    record_ckpt.audit_cfg({"act_hist": 4, "obs_compass": 1}, strict=True)


def test_the_recorder_and_the_trainer_drive_the_same_class():
    """Two rings would drift, and the drift would only show up as eval
    recordings that disagree with training (the frame-ring argument)."""
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert "from surfgym.obsaux import ObsAux" in src
    assert "route_dim += obs_aux.n_features" in src
    assert "aux=obs_aux" in src
    tsrc = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert "from surfgym.obsaux import ACT_FEAT, CMP_FEAT, ObsAux" in tsrc
    # the eval side of the trainer uses the same object as the recorder does
    assert "aux=_s.eval_aux" in tsrc
    assert "self.aux.eval_features(" in tsrc


def test_the_recording_policy_assembles_the_row_in_the_trainer_s_order():
    """The eval row is [core | fan | latch | aux | image]; the trainer's
    fill_vision writes AUX0..SCAL last of the scalar half. A recorder that
    concatenated the aux block before the latch would build a row of the
    right WIDTH and the wrong CONTENT - the exact bug the fan and latch
    hooks already exist to prevent."""
    torch = pytest.importorskip("torch")
    from train_fast import GreedyTorchPolicy

    class _Core:
        num_envs = 1
        states_view = np.zeros(1, dtype=[("origin", np.float32, 3),
                                         ("yaw", np.float32),
                                         ("tick", np.int32)])

    core = _Core()
    aux = ObsAux(1, k=2, field=_LinearField())
    pol = GreedyTorchPolicy(None, None, "cpu", core=core, aux=aux,
                            latch_fn=lambda c: np.array([0.5]))
    aux.push(np.array([[14, 3, 1, 1, 0, 0]], np.int32))
    row = pol._obs(np.zeros((1, 15), np.float32)).numpy()[0]
    assert row.shape[0] == 15 + 1 + aux.n_features
    assert row[15] == pytest.approx(0.5)                  # latch, then aux
    assert row[16] == pytest.approx(1.0)                  # newest yaw bin
    assert row[16 + 2 * ACT_FEAT] == pytest.approx(1.0)   # compass fwd
