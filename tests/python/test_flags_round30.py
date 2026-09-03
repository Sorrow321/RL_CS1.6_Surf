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


# ==========================================================================
# 2. --tower-depth / --conv-mult
# ==========================================================================
# The last commit before either flag existed. PINNED to a sha for the reason
# test_trunk.py pins one: at "HEAD" this would compare the flag against
# itself the moment it landed.
BASE_REV = "246aa4f"
W, H = 64, 32


def _baseline_module():
    """python/train_fast.py as of BASE_REV, imported under a private name."""
    import importlib.util
    import subprocess
    try:
        src = subprocess.run(["git", "show", f"{BASE_REV}:python/train_fast.py"],
                             cwd=ROOT, capture_output=True, check=True).stdout
    except Exception as exc:                    # no git / shallow checkout
        pytest.skip(f"cannot read the baseline from git: {exc!r}")
    tmp = ROOT / "python" / "_train_fast_r30_base_tmp.py"
    tmp.write_bytes(src)
    try:
        spec = importlib.util.spec_from_file_location(
            "_train_fast_r30_base_tmp", tmp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_train_fast_r30_base_tmp"] = mod
        spec.loader.exec_module(mod)
        if "tower_depth" in mod.Policy.__init__.__code__.co_varnames:
            pytest.skip(f"{BASE_REV} already has --tower-depth")
        return mod
    finally:
        tmp.unlink(missing_ok=True)
        sys.modules.pop("_train_fast_r30_base_tmp", None)


@pytest.mark.parametrize("kw", [
    {},                                     # the shipped baseline
    {"route_dim": 6},                       # --route zero-pad warm start
    {"rnn": "gru", "rnn_size": 16},         # --rnn widen_for_rnn
])
def test_defaults_are_the_pre_flag_policy_bit_for_bit(kw):
    from train_fast import N_SCALAR, Policy
    base = _baseline_module()
    obs = N_SCALAR + kw.get("route_dim", 0) + W * H

    torch.manual_seed(1234)
    old = base.Policy(obs, W, H, emb=32, hidden=24, **kw)
    torch.manual_seed(1234)
    new = Policy(obs, W, H, emb=32, hidden=24, tower_depth=2, conv_mult=1,
                 **kw)
    a, b = old.state_dict(), new.state_dict()
    assert list(a) == list(b), "state_dict KEYS moved"
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k} differs at the defaults"


def test_tower_depth_adds_layers_and_keeps_the_pad_column():
    """N Linear+Tanh pairs; only tower.0 follows the observation width."""
    from train_fast import N_SCALAR, Policy, widen_for_route
    for n in (1, 2, 3, 5):
        p = Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24, tower_depth=n)
        assert len(p.pi) == 2 * n and len(p.vf) == 2 * n
        lin = [m for m in p.pi if isinstance(m, torch.nn.Linear)]
        assert len(lin) == n
        assert lin[0].in_features == 32 + len(p.feat_idx)
        assert all(x.in_features == 24 and x.out_features == 24
                   for x in lin[1:])

    # --route/--rnn still warm-start: the widened column is tower.0 at any
    # depth, so widen_for_route's trailing zero-pad applies unchanged
    narrow = Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24, tower_depth=4)
    wide = Policy(N_SCALAR + 6 + W * H, W, H, emb=32, hidden=24,
                  tower_depth=4, route_dim=6)
    ck = {"policy": {k: v.clone() for k, v in narrow.state_dict().items()}}
    assert widen_for_route(ck, wide) == 2        # pi.0.weight and vf.0.weight
    wide.load_state_dict(ck["policy"])
    x = torch.randn(3, N_SCALAR + 6 + W * H)
    x[:, N_SCALAR:N_SCALAR + 6] = 0.0            # a zero fan reproduces...
    y = torch.randn(3, N_SCALAR + W * H)
    y[:, :N_SCALAR] = x[:, :N_SCALAR]
    y[:, N_SCALAR:] = x[:, N_SCALAR + 6:]
    with torch.no_grad():
        assert torch.allclose(wide(x)[0], narrow(y)[0], atol=1e-6)


def test_conv_mult_scales_the_three_convs_and_the_pool_linear():
    from train_fast import N_SCALAR, Policy
    for m in (1, 2, 3):
        p = Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24, conv_mult=m)
        convs = [x for x in p.conv if isinstance(x, torch.nn.Conv2d)]
        assert [c.out_channels for c in convs] == [16 * m, 32 * m, 64 * m]
        lin = [x for x in p.conv if isinstance(x, torch.nn.Linear)][0]
        assert lin.in_features == 64 * m * 4 * 8
        with torch.no_grad():                    # and it still runs
            assert p(torch.randn(2, N_SCALAR + W * H))[1].shape == (2,)


def test_bad_capacity_values_are_refused():
    from train_fast import N_SCALAR, Policy
    with pytest.raises(SystemExit):
        Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24, tower_depth=0)
    with pytest.raises(SystemExit):
        Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24, conv_mult=0)
    with pytest.raises(SystemExit):              # resnet has a fixed table
        Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24, trunk="resnet",
               conv_mult=2)


def test_capacity_flags_are_plumbed_and_guarded():
    assert '"tower_depth": args.tower_depth' in TRAIN_SRC     # run.json
    assert '"conv_mult": args.conv_mult' in TRAIN_SRC
    assert '("tower_depth", 2), ("conv_mult", 1)' in TRAIN_SRC  # restore+guard
    assert "there is no warm start across it" in TRAIN_SRC      # the message
    assert "tower_depth=args.tower_depth" in TRAIN_SRC          # construction
    assert 'cfg.get("tower_depth")' in REC_SRC                  # mirrored
    assert 'cfg.get("conv_mult")' in REC_SRC


# ==========================================================================
# 3. --fp32-heads
# ==========================================================================
def test_fp32_heads_off_is_the_pre_flag_expression():
    """Off must be the same op, and the flag must not touch the state_dict."""
    from train_fast import N_SCALAR, Policy
    torch.manual_seed(7)
    a = Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24)
    torch.manual_seed(7)
    b = Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24, fp32_heads=True)
    assert list(a.state_dict()) == list(b.state_dict())
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k])
               for k in a.state_dict())
    x = torch.randn(4, N_SCALAR + W * H)
    with torch.no_grad():                     # outside autocast they agree
        la, va = a(x)
        lb, vb = b(x)
    assert torch.equal(la, lb) and torch.equal(va, vb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp32_heads_keeps_the_heads_out_of_bf16():
    """Under bf16 autocast the flag changes the OUTPUT dtype and the value."""
    from train_fast import N_SCALAR, Policy
    dev = torch.device("cuda")
    torch.manual_seed(11)
    off = Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24).to(dev)
    torch.manual_seed(11)
    on = Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24,
                fp32_heads=True).to(dev)
    x = torch.randn(8, N_SCALAR + W * H, device=dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), \
            torch.no_grad():
        lo, vo = off(x)
        ln, vn = on(x)
    assert lo.dtype == torch.bfloat16 and vo.dtype == torch.bfloat16
    assert ln.dtype == torch.float32 and vn.dtype == torch.float32
    # the bf16 head is the fp32 head ROUNDED: same value to bf16 precision,
    # not the same bits once cast back up
    assert torch.allclose(lo.float(), ln, atol=0.05, rtol=0.05)
    assert not torch.equal(lo.float(), ln)


def test_fp32_heads_is_plumbed_and_train_only_in_the_recorder():
    assert '"fp32_heads": args.fp32_heads' in TRAIN_SRC          # run.json
    assert 'ck_cfg.get("fp32_heads")' in TRAIN_SRC               # resume
    assert "fp32_heads=bool(args.fp32_heads)" in TRAIN_SRC       # construction
    assert '"fp32_heads",' in REC_SRC                            # TRAIN_ONLY
    from record_ckpt import TRAIN_ONLY
    assert "fp32_heads" in TRAIN_ONLY


# ==========================================================================
# 4. --max-step, and record_ckpt.py --maxvel
# ==========================================================================
class _FakeCore:
    """Only what RaceReward reads (the test_race_dfloor.py FakeCore idiom)."""

    def __init__(self, pts):
        from surfgym.core import STATE_DTYPE
        self.num_envs = len(pts)
        self.states_view = np.zeros(len(pts), dtype=STATE_DTYPE)
        self.states_view["origin"] = pts
        self.goal_hits = np.zeros(len(pts), np.uint8)

    def map_bounds(self):
        return (np.full(3, -3e4, np.float32), np.full(3, 3e4, np.float32))

    def move(self, pts):
        self.states_view["origin"] = pts


class _AxisField:
    """d = x: a distance field whose gradient is exactly one unit per unit."""

    def sample(self, pts):
        return np.asarray(pts, np.float64)[:, 0].copy()


def _race_delta(dx, *, max_step, every=1):
    """Reward paid for a single step of +dx toward the goal (scale 1, no
    time penalty), i.e. the clipped shaping delta."""
    from surfgym.rewards import RaceReward
    core = _FakeCore(np.array([[1000.0, 0.0, 0.0]]))
    r = RaceReward(_AxisField(), scale=1.0, time_pen=0.0, success_bonus=0.0,
                   stall_ticks=0, max_step=max_step, every=every)
    z = np.zeros(1, np.uint8)
    r(None, None, None, np.zeros(1, np.float32), z, z, core)   # arms _d
    core.move(np.array([[1000.0 - dx, 0.0, 0.0]]))
    return float(r(None, None, None, np.zeros(1, np.float32), z, z, core)[0])


def test_max_step_is_the_teleport_clip_and_scales_with_every():
    from surfgym.rewards import RaceReward
    import inspect
    # the default is still 100, so an unflagged run is unchanged
    assert inspect.signature(RaceReward).parameters["max_step"].default == 100.0
    assert _race_delta(40.0, max_step=100.0) == pytest.approx(40.0)
    assert _race_delta(400.0, max_step=100.0) == pytest.approx(100.0)
    # raising it lets a bigger single-tick move cash
    assert _race_delta(400.0, max_step=500.0) == pytest.approx(400.0)
    # lowering it clips a legal-looking one
    assert _race_delta(40.0, max_step=10.0) == pytest.approx(10.0)
    # the clip is per TICK: `every` decisions of it are allowed per call
    assert _race_delta(400.0, max_step=100.0, every=4) == pytest.approx(400.0)


def test_max_step_is_plumbed_through_the_trainer_and_the_feeds():
    assert '"max_step": (args.max_step' in TRAIN_SRC              # run.json
    assert 'ck_cfg.get("max_step")' in TRAIN_SRC                  # resume
    assert "max_step=args.max_step" in TRAIN_SRC                  # RaceReward
    # the two --obs-reward eval mirrors clip at the same width as training
    assert "-100.0 * k, 100.0 * k" not in TRAIN_SRC
    assert "max_step=_s.reward_fn.max_step" in TRAIN_SRC
    # ... and so does record_ckpt.py's own feed
    assert 'cfg.get("max_step")' in REC_SRC
    assert "100.0 * _k" not in REC_SRC


def test_record_ckpt_maxvel_defaults_to_physics_parity():
    """The flag exists, defaults to None, and the ckpt value is read first."""
    import record_ckpt
    assert "--maxvel" in REC_SRC
    assert "cfg_maxvel = float(cfg.get(\"maxvel\", 2000.0))" in REC_SRC
    assert "maxvel = cfg_maxvel if args.maxvel is None else float(args.maxvel)" \
        in REC_SRC
    # loud on override, and written into the traj header
    assert "--maxvel OVERRIDE" in REC_SRC
    assert 'header_extra["maxvel"] = maxvel' in REC_SRC
    assert 'header_extra["maxvel_ckpt"] = cfg_maxvel' in REC_SRC
    # "maxvel" must not become a config key the audit thinks is unmirrored
    assert "maxvel" not in record_ckpt.TRAIN_ONLY


# ==========================================================================
# 5. --stall-eps
# ==========================================================================
class _StallCore:
    """A 1-env core that walks a straight line and remembers force_fail."""

    def __init__(self, step):
        from surfgym.core import STATE_DTYPE
        self.num_envs = 1
        self.states_view = np.zeros(1, dtype=STATE_DTYPE)
        self.step = float(step)
        self.kills = 0

    def advance(self):
        self.states_view["origin"][0, 0] += self.step

    def force_fail(self, mask):
        self.kills += int(np.asarray(mask).sum())


def _stall_kills(step_u, eps, ticks=400, every=1):
    """Run make_eval_stall_hook over a constant-rate approach."""
    from train_fast import make_eval_stall_hook
    core = _StallCore(-step_u)              # d = x, so -step_u is progress
    hook = make_eval_stall_hook(core, _AxisField(), 100, eps, every)
    z = np.zeros(1, np.uint8)
    for t in range(ticks):
        core.advance()
        hook(t, core.states_view, np.zeros(1, np.float32), z, z)
    return core.kills


def test_stall_eps_is_the_per_call_threshold_in_the_eval_mirror():
    # 40u per call clears a 32u threshold every call -> never killed
    assert _stall_kills(40.0, 32.0) == 0
    # the SAME flight against a 50u threshold is a stall, repeatedly
    assert _stall_kills(40.0, 50.0) > 0
    # ... and a crawl that a 32u threshold kills survives a 1u one
    assert _stall_kills(2.0, 32.0) > 0
    assert _stall_kills(2.0, 1.0) == 0


def test_stall_eps_reaches_the_reward_and_is_plumbed():
    from surfgym.rewards import RaceReward
    import inspect
    assert inspect.signature(RaceReward).parameters["stall_eps"].default == 32.0
    r = RaceReward(_AxisField(), scale=1.0, stall_eps=7.5)
    assert r.stall_eps == 7.5
    assert 'ap.add_argument("--stall-eps"' in TRAIN_SRC
    assert 'ck_cfg.get("stall_eps")' in TRAIN_SRC                 # resume
    assert 'restored.append(f"stall_eps=' in TRAIN_SRC
    assert '"stall_eps": (args.stall_eps' in TRAIN_SRC            # run.json
    assert "stall_eps=args.stall_eps," in TRAIN_SRC               # RaceReward
    # the eval-stall hook takes it off the REWARD OBJECT, so the flag
    # reaches it without a second source of truth
    assert "_s.reward_fn.stall_eps" in TRAIN_SRC
    # ... and record_ckpt's mirror no longer hardcodes 32
    assert '_stall_eps = float(cfg.get("stall_eps") or 32.0)' in REC_SRC
    assert "_stall_eps = 32.0" not in REC_SRC


# ==========================================================================
# 6. --n-steps / --epochs / --minibatches restore on resume
# ==========================================================================
def test_batch_shape_is_restored_from_the_checkpoint():
    """They have real argparse defaults, so the None-based loop misses them."""
    assert ('("n_steps", "--n-steps"), ("epochs", "--epochs"),\n'
            '                         ("minibatches", "--minibatches")'
            in TRAIN_SRC)
    assert "restored.append(f\"{k}={getattr(args, k)}\")" in TRAIN_SRC
    # explicit flags still win - the flag_given test, not a None test
    i = TRAIN_SRC.index('("n_steps", "--n-steps")')
    tail = TRAIN_SRC[i:i + 400]
    assert "not flag_given(_flag)" in tail
    # ... and they are recorded, so there is something to restore
    for k in ('"n_steps": args.n_steps', '"epochs": args.epochs',
              '"minibatches": args.minibatches'):
        assert k in TRAIN_SRC


def test_flag_given_sees_both_spellings():
    """The restore hinges on it: --n-steps 32 and --n-steps=32 both count."""
    import ast
    import textwrap
    fn = None
    for node in ast.walk(ast.parse(TRAIN_SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == "flag_given":
            fn = ast.get_source_segment(TRAIN_SRC, node)
    assert fn, "flag_given moved"
    ns = {"sys": sys}
    exec(textwrap.dedent(fn), ns)
    old = sys.argv
    try:
        sys.argv = ["train_fast.py", "--n-steps=32", "--epochs", "8"]
        assert ns["flag_given"]("--n-steps")
        assert ns["flag_given"]("--epochs")
        assert not ns["flag_given"]("--minibatches")
    finally:
        sys.argv = old
