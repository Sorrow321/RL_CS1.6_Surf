"""Record a greedy trajectory from a train_fast checkpoint, on demand.

Runs in its own process with its own env — safe to point at ckpt_latest.pt
of a training run that is still going. The output lands in the run's
directory named traj_<global_step>.jsonl, so the dashboard picks it up like
the trainer's own recordings.

    python tools\record_ckpt.py runs\marathon_10B\ckpt_latest.pt
    python tools\record_ckpt.py runs\marathon_10B\ckpt_latest.pt --episodes 5
"""
from __future__ import annotations

import argparse
import json
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import torch

from surfgym import SurfCore, default_config
from surfgym.record import record_rollout
from surfgym.route import RouteLine
from surfgym.rewards import (drop_spawn_pool, map_spawn_pool,
                             platform_spawn_pool, ramp_spawn_pool)
from train_fast import (GreedyChunkPolicy, GreedyTorchPolicy, HeadPacker,
                        Policy, SampledChunkPolicy, SampledTorchPolicy)



# Every misleading recording we shipped had the same shape: train_fast.py
# grew a flag, record_ckpt.py was never taught to mirror it, and the
# recorder happily produced a trajectory under the WRONG semantics. It
# happened three times in one day - --obs-reward (523-vs-522, caught only
# because a strict state_dict load throws), its reward feed (agent fed
# absolute position where it expected tanh(reward); died in seconds), and
# --yaw-adaptive (every steering action reinterpreted; 42k measured vs the
# trainer's own 98k on identical weights). Only the first was loud.
#
# Checking the three known fields would just wait for the fourth flag, so
# this checks the property instead: a checkpoint key the recorder never
# READS is a key it cannot be mirroring. TRAIN_ONLY lists the keys that
# genuinely do not change what a recording means; anything else unread is
# a new flag nobody taught this file about, and we refuse to emit a
# trajectory rather than emit a plausible wrong one.
TRAIN_ONLY = frozenset({
    # optimizer / schedule / plumbing - no effect on a rollout
    "trainer", "envs", "steps", "lr", "epochs", "gamma", "gae", "clip",
    "vf", "ent", "ent_final", "graphs", "compile", "bf16", "train_stride",
    "reward_per_decision", "eval_eps", "eval_greedy_only", "blend",
    # exploration: training-time only, a recording is greedy or samples pi
    "ez_eps", "ez_max", "ez_mu",
    # post-respawn random bursts / start-state selection modes (explore
    # arms): training-time start distribution and exploration only, never
    # what an action means or what the policy sees
    "spawn_burst", "spawn_burst_p", "respawn_mode", "respawn_bins",
    "respawn_killsafe", "demo_file", "demo_window", "demo_rate",
    "demo_min_ep",
    # spawn-distribution knobs: --spawn selects the pool we record from
    "respawn_frac", "respawn_margin", "respawn_binned", "respawn_reservoir",
    "respawn_speed", "race_kill_aware",
    # reward TERMS. These shape training but are not observed by the policy
    # -- except under --obs-reward, where the fed value is shaping-only and
    # omits the intrinsic bonus (~0.007/decision vs shaping's ~0.023). That
    # approximation is deliberate and logged; revisit if it ever matters.
    "revisit_pen", "success_bonus", "finish_k", "finish_tref", "stall_secs",
    "fail_pen", "speed_coef", "int_coef", "int_view", "rnd_coef",
    "speed_equiv", "int_speed",
    # --chunk's LEARNABLE decoder is a Parameter of the policy, so it ships
    # inside ck["policy"] and a recording reads it from there. dec_ent is a
    # loss coefficient; codebook/codebook_bias only seeded the decoder at
    # iteration 0 and have had no effect on it since. chunk and n_codes are
    # NOT here - they change what one decision means, and are mirrored below.
    "dec_ent", "codebook", "codebook_bias",
    # --quantiles' Huber threshold: a coefficient of the critic's LOSS, and
    # the critic does not act. (`quantiles` itself is NOT here - it changes
    # the value head's shape and is mirrored where the Policy is built.)
    "quantile_kappa",
})


def _mentioned_keys():
    """Every string literal in this file.

    v1 of this guard tracked which keys were read at RUNTIME. That is
    branch-dependent and wrong: time_pen is only read inside
    `if cfg.get("obs_reward")`, so every run WITHOUT obs-reward looked like
    it had an unmirrored key and refused to record - the guard broke the
    dashboard for every arm except the one I happened to test it on.

    A key this file never mentions anywhere is a key it cannot be
    mirroring, whatever branch runs. That is static, order-independent,
    and still catches the real case: a flag the trainer grew that nobody
    wrote into this file.
    """
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def audit_cfg(cfg, strict=True):
    known = _mentioned_keys()
    missing = sorted(k for k in cfg
                     if k not in known and k not in TRAIN_ONLY
                     and cfg.get(k) is not None)
    if not missing:
        return
    nl = chr(10)
    msg = ('this checkpoint sets '
           + ', '.join('%s=%r' % (k, cfg.get(k)) for k in missing)
           + ' but record_ckpt.py never mentions '
           + ('it' if len(missing) == 1 else 'them') + '.' + nl
           + 'If the setting changes what an action MEANS or what the'
           + ' policy SEES, mirror it here.' + nl
           + 'If it is training-only, add it to TRAIN_ONLY with a'
           + ' one-line reason.' + nl
           + 'Refusing to record under semantics that may not match'
           + ' training.')
    if strict:
        raise SystemExit('CONFIG MISMATCH: ' + msg)
    print('WARNING: ' + msg)

def _phase_writer(path):
    """Report what the recorder is DOING, not just tick counts.

    A recording is dominated by startup - torch import, CUDA context, ckpt
    load, goal-field build - not by stepping. A percentage of ticks therefore
    sits at 0 for most of the wall time and tells the user nothing, which is
    indistinguishable from a hung job. Startup owns 0-30%, stepping 30-100%.
    """
    if not path:
        return lambda *a, **k: None
    pf = Path(path)
    pf.parent.mkdir(parents=True, exist_ok=True)

    def write(phase, pct, **extra):
        d = {"phase": phase, "pct": int(pct)}
        d.update(extra)
        tmp = pf.with_suffix(".tmp")
        tmp.write_text(json.dumps(d), encoding="utf-8")
        tmp.replace(pf)           # atomic: a reader never sees a half file
    return write


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--map", default=None, help="defaults to the ckpt's map")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default=None,
                    help="defaults to <ckpt dir>/traj_<global_step>.jsonl")
    ap.add_argument("--seed", type=int, default=None,
                    help="spawn seed (default: derived from the ckpt step, "
                         "so successive snapshots sample different spawns)")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of argmax — what "
                         "rollout/ep_rew_mean actually measures")
    ap.add_argument("--spawn", choices=["platform", "ramp", "mixed",
                                        "reservoir"],
                    default=None,
                    help="spawn pool (default: the ckpt's training pool, "
                         "i.e. what rollout/ep_rew_mean averages over; "
                         "reservoir = the ckpt's respawn buffer — states "
                         "agents ACTUALLY reached, i.e. the live frontier)")
    ap.add_argument("--ep-ticks", type=int, default=None,
                    help="episode length for the recording (default: the "
                         "ckpt's training length; the policy has no episode "
                         "clock, so longer rollouts are fine)")
    ap.add_argument("--progress-file", default=None,
                    help="write live tick progress here as JSON, so a caller "
                         "(the dashboard) can show a real percentage instead "
                         "of an opaque spinner")
    ap.add_argument("--no-config-audit", action="store_true",
                    help="downgrade unmirrored-config errors to warnings")
    args = ap.parse_args()

    say = _phase_writer(args.progress_file)
    say("loading checkpoint", 5)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    step = int(ck.get("global_step", 0))
    cfg_map = cfg.get('map', 'surf_ski_2')
    map_path = args.map or str(ROOT / "maps" / f"{cfg_map}.bsp")
    # read the ckpt value FIRST, unconditionally. "args.X or cfg.get(X)"
    # short-circuits when the CLI overrides it, so the audit below never
    # sees the key as read and refuses to record - which is exactly how
    # the guard broke the dashboard's --spawn reservoir button.
    cfg_ep_ticks = cfg.get("ep_ticks", 700)
    ep_ticks = int(args.ep_ticks or cfg_ep_ticks)
    if (args.ep_ticks is None and cfg.get("reward") == "race"
            and ep_ticks < int(cfg.get("ep_ticks", 0))):
        # A race episode runs until the finish (or the cap), so a DEFAULT
        # that is shorter than training would cut good runs off. But an
        # EXPLICIT --ep-ticks is the caller saying how long a recording it
        # wants, and silently overriding it made the dashboard's "2000"
        # actually mean 12000: 2 episodes x 12000 = 24,000 single-env ticks,
        # ~100 s, which is the "why does this take 2 minutes" report. Worse,
        # it scaled with how GOOD the agent is - a policy that survives
        # longer burns more of the budget.
        ep_ticks = int(cfg["ep_ticks"])
        print(f"race ckpt: episode cap restored to {ep_ticks} ticks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    fix_pitch = cfg.get("fix_pitch")
    pitch_rate = 0.0 if fix_pitch is not None else float(cfg.get("pitch_rate", -1.0))
    say("starting sim", 12)
    core = SurfCore(map_path, default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=ep_ticks, water_fail=1,
        sv_maxvelocity=float(cfg.get("maxvel", 2000.0)),  # physics parity
        # --yaw-adaptive REDEFINES what a yaw bin means (k * atan(30/|v|)
        # instead of a fixed deg/tick). Recording such a ckpt on a stock core
        # silently reinterprets every steering action: measured 42k track vs
        # the trainer's own 98k on the same weights.
        yaw_adaptive=1 if cfg.get("yaw_adaptive") else 0,
        lidar_w=0, lidar_h=0,           # eyeless core; vision is GPU-side
        pitch_rate_max_deg=pitch_rate))
    cfg_spawn = cfg.get("spawn", "platform")
    spawn = args.spawn or cfg_spawn
    drop_rng = (float(cfg.get("drop_min", 400.0)),
                float(cfg.get("drop_max", 800.0)))
    punch = (float(cfg.get("punch_min", 100.0)),
             float(cfg.get("punch_max", 400.0)))
    from surfgym.vision import GpuLidar, pick_cell
    cell = float(cfg.get("lidar_cell") or pick_cell(core))
    gf = None
    if cfg.get("reward") == "race":
        # finish zone is armed for ANY race recording, whatever the spawns
        from surfgym.goalfield import EuclidField, build_goal_field
        from surfgym.zones import load_zones
        zones = load_zones(core.bsp_path)
        say("building goal field", 18)
        gf = (EuclidField(zones["end"]) if cfg.get("race_dist") == "euclid"
              else build_goal_field(core, zones["end"], cell=cell))
        core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])

    def race_start_pool():
        raw = map_spawn_pool(core)
        p = map_spawn_pool(core, yaw=gf.descent_yaw(raw["origin"]))
        p["pitch"] = -10.0
        print(f"race: start geodesic "
              f"{float(np.mean(gf.sample(raw['origin']))):.0f}u")
        return p

    if gf is not None and args.spawn is None:
        # race default: the run is judged from the map's real start line
        spawn = "start"
        pool = race_start_pool()
    elif spawn == "reservoir":
        rs = (ck.get("respawn") or {}).get("states")
        if rs is None or len(rs) == 0:
            raise SystemExit("this ckpt has no respawn reservoir "
                             "(run trained without --respawn-frac?)")
        pool = np.asarray(rs)
        print(f"reservoir pool: {len(pool)} frontier states from the ckpt")
    elif spawn in ("ramp", "mixed"):
        dp = drop_spawn_pool(core, h_range=drop_rng, speed_range=punch)
        if gf is not None:
            keep = gf.reachable(dp["origin"]) & (gf.sample(dp["origin"]) > 400.0)
            dp = dp[keep]              # training parity: on-track drops only
        if spawn == "mixed":
            base = race_start_pool() if gf is not None \
                else platform_spawn_pool(core)
            # the env resets by UNIFORM pool draw, so entry counts are the
            # probabilities: a few start entries beside thousands of drops
            # is not a mix, it is drops. Replicate to a real 50/50.
            reps = max(1, int(round(len(dp) / max(len(base), 1))))
            pool = np.concatenate([np.concatenate([base] * reps), dp])
        else:
            pool = dp
    else:
        # a race ckpt trains from the map's start entities, not the
        # walk-off-the-edge audition pool: asking for 'platform' on one
        # used to raise 'no edge-facing-ramp spawn found'
        pool = race_start_pool() if gf is not None else platform_spawn_pool(core)
    if fix_pitch is not None:
        pool["pitch"] = float(fix_pitch)
    if cfg.get("teleport_fail") or cfg.get("reward") == "race":
        core.set_teleport_fail(True)     # eval parity with training semantics
    core.set_spawn_pool(pool)
    print(f"spawn pool: {spawn} ({len(pool)} points)"
          + (f", pitch fixed {fix_pitch:g}" if fix_pitch is not None else ""))

    say("initialising vision", 25)
    lidar = GpuLidar(core, lw, lh,
                     range_units=float(cfg.get("lidar_range", 2000.0)),
                     near_range=cfg.get("lidar_near"),
                     cell=cell,
                     device=device,
                     surf_mask=bool(cfg.get("surf_mask", 0)),
                     pinhole=bool(cfg.get("pinhole", 0)))
    # --frame-stack: the recording policy keeps its own ring (see
    # _TorchPolicyBase._push_frame), so a stacked ckpt records honestly
    stack = max(1, int(cfg.get("frame_stack") or 1))
    # --obs-reward re-enables scalar slot 12 (an absolute-position channel
    # the no-GPS mask normally hides) to carry the previous reward, which
    # widens the scalar tower by one. Without this the state_dict load fails
    # with a 523-vs-522 size mismatch.
    extra = (12,) if cfg.get("obs_reward") else ()
    # --chunk REDEFINES what one policy decision is: the head emits a CODE and
    # a learned decoder expands it into H per-decision action distributions.
    # Recording such a ckpt with the flat six-head sampler would read K code
    # logits as padded action logits and emit a plausible, completely wrong
    # trajectory - exactly the failure the config audit exists to stop. The
    # decoder is a Parameter, so the SHAPE in the state_dict is authoritative
    # and the config is cross-checked against it rather than trusted.
    chunk = int(cfg.get("chunk") or 0)
    n_codes = int(cfg.get("n_codes") or 0)
    dec_w = (ck.get("policy") or {}).get("decoder")
    if (dec_w is not None) != (chunk > 0):
        raise SystemExit(
            "checkpoint/config disagree about --chunk: config says chunk=%r "
            "but the state_dict %s a decoder. Refusing to guess what an "
            "action means." % (cfg.get("chunk"),
                               "has" if dec_w is not None else "has no"))
    if chunk > 0:
        if tuple(dec_w.shape)[:2] != (n_codes, chunk):
            raise SystemExit(
                "checkpoint decoder is %s but the config says n_codes=%r "
                "chunk=%r" % (tuple(dec_w.shape), n_codes, chunk))
        print(f"chunk {chunk}: {n_codes} codes, decoder from the checkpoint "
              f"({chunk} decisions x {int(cfg.get('act_every', 1))} ticks)")
    # --route: the lookahead fan is part of the OBSERVATION, so a recording
    # that skipped it would feed these weights a row of the wrong width (and,
    # if it happened to fit, of the wrong content). The route file travels
    # with the checkpoint config; a missing file is fatal, not a warning.
    route = None
    if cfg.get("route_file"):
        rp = Path(cfg["route_file"])
        if not rp.exists():
            rp = ROOT / "maps" / rp.name
        if not rp.exists():
            raise SystemExit(
                f"this ckpt was trained with --route {cfg['route_file']} and "
                "that file is not here: a recording without the lookahead fan "
                "is not the trained policy")
        npts = int(cfg.get("route_points") or 8)
        span = float(cfg.get("route_span") or 6.0)
        offs = tuple(float(span * (24.0 ** (-(npts - 1 - i) / max(1, npts - 1))))
                     for i in range(npts))
        route = RouteLine.load(rp, offsets=offs, device=device)
        print(route.describe())
    route_dim = route.n_features if route is not None else 0
    # --race-latch: the flag is a 1-wide OBSERVATION block concatenated
    # LAST, exactly where the route fan's columns go (train_fast.py
    # N_LATCH). A recording that skipped it would build a row one column
    # narrow and the state_dict load would fail; one that fed a constant
    # would not be the trained policy in the only states that matter.
    d_latch = float(cfg.get("race_latch") or 0.0)
    latch_dim = 1 if d_latch > 0.0 else 0
    if latch_dim and gf is None:
        raise SystemExit("this ckpt uses --race-latch but has no goal "
                         "field to recompute the flag from")
    route_dim += latch_dim
    policy = Policy(core.obs_dim + route_dim + lw * lh * lidar.channels * stack,
                    lw, lh,
                    emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)),
                    extra_feat=extra,
                    in_ch=lidar.channels * stack,
                    n_codes=n_codes, chunk=chunk,
                    route_dim=route_dim,
                    route_critic_only=bool(cfg.get("route_critic_only")),
                    # --quantiles changes the value head's SHAPE, (N, hidden)
                    # instead of (1, hidden), so a recorder that ignored it
                    # would fail the strict state_dict load below. It changes
                    # NOTHING a recording means - the actor is untouched and
                    # Policy.forward returns the quantile MEAN - but the guard
                    # above refuses any key this file does not mention, and
                    # rightly: the shape has to be mirrored, not assumed.
                    n_quant=int(cfg.get("quantiles") or 0)
                    ).to(device)
    say("loading policy", 29)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    suffix = f"_{args.spawn}" if args.spawn else ""
    suffix += "_stoch" if args.stochastic else ""
    out = Path(args.out) if args.out else \
        Path(args.ckpt).parent / f"traj_{step:010d}{suffix}.jsonl"
    seed = args.seed if args.seed is not None else step & 0x7FFFFFFF
    if chunk > 0:
        # greedy = argmax code, then argmax per decision out of that code's
        # decoder row; stochastic = sample both, which is what the trainer's
        # rollout does. The shim holds each decoded 6-tuple for act_every
        # ticks, mirroring _TorchPolicyBase.act one level up.
        cls = SampledChunkPolicy if args.stochastic else GreedyChunkPolicy
    else:
        cls = SampledTorchPolicy if args.stochastic else GreedyTorchPolicy
    act_every = int(cfg.get("act_every", 1))
    # --obs-reward ckpts read a side-channel value from scalar slot 12 that
    # the core does not produce. Without feeding it here the recording hands
    # the policy absolute position (magnitude ~10) where it expects
    # tanh(reward) in [-1,1], and the agent dies within seconds - the same
    # train/eval mismatch that made sOBSR's in-trainer evals meaningless.
    # the flag: set on any tick with d <= race_latch, cleared at an
    # episode start, which reset_env marks by zeroing the tick counter
    latch_fn = None
    if latch_dim:
        _ls = {"f": None, "tick": None}

        def latch_fn(c, _f=gf, _L=d_latch):
            sv = c.states_view
            d = _f.sample(sv["origin"]).astype(np.float64)
            tick = np.asarray(sv["tick"], np.int64).copy()
            f, pt = _ls["f"], _ls["tick"]
            if f is None or len(f) != len(d) or pt is None:
                f = np.zeros(len(d), bool)
            else:
                f = f & ~(tick <= pt)
            f |= d <= _L
            _ls["f"], _ls["tick"] = f, tick
            return f.astype(np.float32)

        latch_fn.state = _ls
        print(f"--race-latch {d_latch:,.0f}u: shaping switches OFF for the "
              f"rest of an episode once it reaches that distance; the flag "
              f"is obs column {core.obs_dim + route_dim - 1}")
    extra_slot, extra_fn = -1, None
    if cfg.get("obs_reward"):
        if gf is None:
            raise SystemExit("this ckpt uses --obs-reward but has no goal "
                             "field to recompute it from")
        d0 = float(np.mean(gf.sample(core.get_states()["origin"])))
        # --race-shaping scales the trainer's potential; the feed mirrors it
        scale = 100.0 / max(d0, 1.0) * float(cfg.get("race_shaping") or 1.0)
        tp = float(cfg.get("time_pen") or 0.005)
        _st = {"d": None}
        # --race-dfloor clamps the potential and --race-latch switches it
        # off; slot 12 carries the policy's OWN shaping, so a mirror that
        # reported the raw term would feed these weights a feature they
        # were never trained on.
        d_floor = float(cfg.get("race_dfloor") or 0.0)

        def _feed(c, _f=gf, _s=scale, _tp=tp, _k=act_every,
                  _fl=d_floor):
            d = _f.sample(c.states_view["origin"]).astype(np.float64)
            if _fl > 0.0:
                d = np.maximum(d, _fl)
            prev, _st["d"] = _st["d"], d
            if prev is None or len(prev) != len(d):
                return np.zeros(len(d), np.float32)
            delta = np.clip(prev - d, -100.0 * _k, 100.0 * _k)
            if latch_fn is not None:
                # the flag as of the PREVIOUS decision - the one that
                # governed the reward this slot reports
                was = latch_fn.state["f"]
                if was is not None and len(was) == len(delta):
                    delta = np.where(was, 0.0, delta)
            return np.tanh((delta * _s - _tp * _k) / 0.1).astype(np.float32)

        extra_slot, extra_fn = 12, _feed
    audit_cfg(cfg, strict=not args.no_config_audit)
    total_budget = args.episodes * ep_ticks
    on_tick = None
    if args.progress_file:
        pf = Path(args.progress_file)
        pf.parent.mkdir(parents=True, exist_ok=True)
        # Percent-of-tick-budget is a LIE here: a race episode usually ends
        # early (death), so the counter crawls to 2% and then the job is
        # done. Progress is really "episodes finished, plus how far into the
        # current one" - that is monotonic AND actually reaches 100.
        st = {"last": -1, "eps": 0, "ep0": 0}

        def on_tick(t, _states, _rew, done, trunc,
                    _eps=args.episodes, _cap=ep_ticks):
            if bool(done[0]) or bool(trunc[0]):
                st["eps"] += 1
                st["ep0"] = t + 1
            if t - st["last"] < 20:
                return
            st["last"] = t
            frac = min(1.0, (t - st["ep0"]) / float(max(_cap, 1)))
            done_frac = min(1.0, (st["eps"] + frac) / float(max(_eps, 1)))
            say("recording", 30 + 70 * done_frac,
                episode=min(st["eps"] + 1, int(_eps)), episodes=int(_eps),
                ticks=int(t))

    record_rollout(core, cls(policy, HeadPacker(device), device, lidar, core,
                             act_every, stack, extra_slot=extra_slot,
                             extra_fn=extra_fn, route=route,
                             latch_fn=latch_fn),
                   out, episodes=args.episodes, max_ticks=total_budget,
                   seed=seed, on_tick=on_tick)
    kind = "stochastic" if args.stochastic else "greedy"
    print(f"recorded {args.episodes} {kind} episode(s) at step {step:,} -> {out}")


if __name__ == "__main__":
    main()
