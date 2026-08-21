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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import torch

from surfgym import SurfCore, default_config
from surfgym.record import record_rollout
from surfgym.rewards import (drop_spawn_pool, map_spawn_pool,
                             platform_spawn_pool, ramp_spawn_pool)
from train_fast import (GreedyTorchPolicy, HeadPacker, Policy,
                        SampledTorchPolicy)



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
})


class _AuditedCfg(dict):
    """dict that remembers which keys were actually consulted."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


def audit_cfg(cfg, strict=True):
    unread = sorted(k for k in cfg
                    if k not in cfg.read and k not in TRAIN_ONLY
                    and dict.get(cfg, k) is not None)
    if not unread:
        return
    nl = chr(10)
    msg = ("this checkpoint sets "
           + ", ".join("%s=%r" % (k, dict.get(cfg, k)) for k in unread)
           + " but record_ckpt.py never reads "
           + ("it" if len(unread) == 1 else "them") + "." + nl
           + "If the setting changes what an action MEANS or what the"
           + " policy SEES, mirror it here." + nl
           + "If it is training-only, add it to TRAIN_ONLY with a"
           + " one-line reason." + nl
           + "Refusing to record under semantics that may not match"
           + " training.")
    if strict:
        raise SystemExit("CONFIG MISMATCH: " + msg)
    print("WARNING: " + msg)

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
    ap.add_argument("--no-config-audit", action="store_true",
                    help="downgrade unmirrored-config errors to warnings")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _AuditedCfg(ck.get("config") or {})
    step = int(ck.get("global_step", 0))
    cfg_map = cfg.get('map', 'surf_ski_2')
    map_path = args.map or str(ROOT / "maps" / f"{cfg_map}.bsp")
    # read the ckpt value FIRST, unconditionally. "args.X or cfg.get(X)"
    # short-circuits when the CLI overrides it, so the audit below never
    # sees the key as read and refuses to record - which is exactly how
    # the guard broke the dashboard's --spawn reservoir button.
    cfg_ep_ticks = cfg.get("ep_ticks", 700)
    ep_ticks = int(args.ep_ticks or cfg_ep_ticks)
    if cfg.get("reward") == "race" and ep_ticks < int(cfg.get("ep_ticks", 0)):
        # a race episode runs until the finish (or the cap) — a shorter
        # recording cap (dashboard hand-records pass 3000) would cut runs off
        ep_ticks = int(cfg["ep_ticks"])
        print(f"race ckpt: episode cap restored to {ep_ticks} ticks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    fix_pitch = cfg.get("fix_pitch")
    pitch_rate = 0.0 if fix_pitch is not None else float(cfg.get("pitch_rate", -1.0))
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
    policy = Policy(core.obs_dim + lw * lh * lidar.channels * stack, lw, lh,
                    emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)),
                    extra_feat=extra,
                    in_ch=lidar.channels * stack).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    suffix = f"_{args.spawn}" if args.spawn else ""
    suffix += "_stoch" if args.stochastic else ""
    out = Path(args.out) if args.out else \
        Path(args.ckpt).parent / f"traj_{step:010d}{suffix}.jsonl"
    seed = args.seed if args.seed is not None else step & 0x7FFFFFFF
    cls = SampledTorchPolicy if args.stochastic else GreedyTorchPolicy
    act_every = int(cfg.get("act_every", 1))
    # --obs-reward ckpts read a side-channel value from scalar slot 12 that
    # the core does not produce. Without feeding it here the recording hands
    # the policy absolute position (magnitude ~10) where it expects
    # tanh(reward) in [-1,1], and the agent dies within seconds - the same
    # train/eval mismatch that made sOBSR's in-trainer evals meaningless.
    extra_slot, extra_fn = -1, None
    if cfg.get("obs_reward"):
        if gf is None:
            raise SystemExit("this ckpt uses --obs-reward but has no goal "
                             "field to recompute it from")
        d0 = float(np.mean(gf.sample(core.get_states()["origin"])))
        scale = 100.0 / max(d0, 1.0)
        tp = float(cfg.get("time_pen") or 0.005)
        _st = {"d": None}

        def _feed(c, _f=gf, _s=scale, _tp=tp, _k=act_every):
            d = _f.sample(c.states_view["origin"]).astype(np.float64)
            prev, _st["d"] = _st["d"], d
            if prev is None or len(prev) != len(d):
                return np.zeros(len(d), np.float32)
            delta = np.clip(prev - d, -100.0 * _k, 100.0 * _k)
            return np.tanh((delta * _s - _tp * _k) / 0.1).astype(np.float32)

        extra_slot, extra_fn = 12, _feed
    audit_cfg(cfg, strict=not args.no_config_audit)
    record_rollout(core, cls(policy, HeadPacker(device), device, lidar, core,
                             act_every, stack, extra_slot=extra_slot,
                             extra_fn=extra_fn),
                   out, episodes=args.episodes, max_ticks=args.episodes * ep_ticks,
                   seed=seed)
    kind = "stochastic" if args.stochastic else "greedy"
    print(f"recorded {args.episodes} {kind} episode(s) at step {step:,} -> {out}")


if __name__ == "__main__":
    main()
