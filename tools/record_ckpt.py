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


class _RouteProbe:
    """--route-mode: eval-side ablation of the lookahead fan (the
    --depth-mode idiom, research-plan-goallines.md E2). Wraps a RouteLine;
    the policy keeps its trained input WIDTH, only the fan CONTENT changes:
    'off' zeroes the block, 'frozen' holds the first decision's features
    for the whole recording. A policy that ignores the fan is unaffected
    by either; one that reads it collapses."""

    def __init__(self, route, mode: str):
        self.route, self.mode = route, str(mode)
        self.n_features = route.n_features
        self._held = None

    def features(self, origin, yaw_deg, speed):
        import torch
        if self.mode == "off":
            return torch.zeros((origin.shape[0], self.n_features),
                               dtype=torch.float32, device=origin.device)
        f = self.route.features(origin, yaw_deg, speed)
        if self.mode == "frozen":
            if self._held is None or len(self._held) != len(f):
                self._held = f.clone()
            return self._held
        return f


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
    "respawn_min_speed",
    "respawn_speed", "race_kill_aware",
    # reward TERMS. These shape training but are not observed by the policy
    # -- except under --obs-reward, where the fed value is shaping-only and
    # omits the intrinsic bonus (~0.007/decision vs shaping's ~0.023). That
    # approximation is deliberate and logged; revisit if it ever matters.
    "revisit_pen", "success_bonus", "finish_k", "finish_tref", "stall_secs",
    "fail_pen", "speed_coef", "int_coef", "int_view", "rnd_coef",
    "speed_equiv", "int_speed",
    # --death-charge is a TERMINAL charge only ("no per-step tax, no goal
    # charge"), so like fail_pen/success_bonus it has no per-call value for
    # the --obs-reward slot to mirror - the episode ends where it applies.
    # --race-ng is NOT here: it also levies a PER-CALL tax that the trainer
    # feeds into slot 12, and _feed below mirrors it.
    "death_charge",
    # --chunk's LEARNABLE decoder is a Parameter of the policy, so it ships
    # inside ck["policy"] and a recording reads it from there. dec_ent is a
    # loss coefficient; codebook/codebook_bias only seeded the decoder at
    # iteration 0 and have had no effect on it since. chunk and n_codes are
    # NOT here - they change what one decision means, and are mirrored below.
    "dec_ent", "codebook", "codebook_bias",
    # DDP + multi-map. None of these reaches the policy: Policy.forward takes
    # `obs` alone, there is no map embedding and no rank input, so a rollout
    # is bit-identical however the training batch was sharded. "map_id" rides
    # along in the reservoir purely as a guard against restoring one map's
    # frontier into another (train_fast.py:2607), not as an input.
    "ddp", "world_size", "envs_per_rank", "envs_per_slot", "n_maps",
    "maps", "map_cells", "map_id",
    # batch shape / schedule, same category as epochs and envs above
    "n_steps", "minibatches", "seed",
    # --ret-norm rescales the VALUE HEAD's target. A recording is a rollout
    # of the ACTOR: Policy.forward's value output is never read here, so the
    # frame it is expressed in cannot change a single action. (--eval-stall
    # is NOT here - it changes when an episode ENDS, and is mirrored below.)
    "ret_norm",
    # --fp32-heads disables autocast around the two output Linears. This file
    # never ENTERS autocast (_TorchPolicyBase._decide runs the policy in
    # fp32), so the heads are already fp32 here whatever the training run
    # chose - it changes where the rounding happens in TRAINING, not what a
    # recording computes. If a recorder ever grows an autocast, mirror it.
    "fp32_heads",
    # --goals training-side knobs: the goal DISTRIBUTION during training
    # (horizon band, curriculum, air share). A recording draws its own
    # goals (--goal-band); what the policy SEES is mirrored via "goals",
    # "goal_radius" and "goal_holdout" below.
    "goal_kmin", "goal_kcap", "goal_air_frac", "goal_curriculum",
    "goal_route", "goal_route_frac", "goal_reward", "goal_frontier",
    "goal_route_uniform", "goal_fixed", "goal_fixed_spacing", "goal_fixed_air",
    "goal_euclid_scale", "goal_fixed_decay",
    "goal_front_start", "goal_front_band", "goal_front_step",
    "goal_front_rate", "goal_front_min_ep",
    # expert iteration (--bc-file, surfgym/bc.py): an auxiliary LOSS on
    # planner rows during training. It changes what the weights are fitted
    # to, never what an action means or what the policy sees.
    "bc_file", "bc_coef", "bc_coef_final", "bc_steps", "bc_batch",
    # tools/ckpt_qr_to_scalar.py stamps the source path of a collapsed
    # quantile checkpoint; provenance only
    "qr_source",
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
    ap.add_argument("--eval-stall", type=int, default=None,
                    choices=[0, 1],
                    help="apply TRAINING's stall rule to the recorded "
                         "episodes (kill an episode that has not improved "
                         "its best distance-to-finish by 32u within "
                         "--stall-secs, ending it as a fail). Default: "
                         "whatever the checkpoint trained with "
                         "(cfg['eval_stall']), so a recording reproduces "
                         "the conditions its in-trainer eval ran under")
    ap.add_argument("--ep-ticks", type=int, default=None,
                    help="episode length for the recording (default: the "
                         "ckpt's training length; the policy has no episode "
                         "clock, so longer rollouts are fine)")
    ap.add_argument("--maxvel", type=float, default=None,
                    help="OVERRIDE sv_maxvelocity for this recording (map "
                         "units/s). The default is PHYSICS PARITY: the value "
                         "the checkpoint trained under, because the server "
                         "speed cap is part of the dynamics the policy "
                         "learned and changing it makes the recording a "
                         "rollout of different physics. Overriding is a "
                         "deliberate probe (how does this policy behave under "
                         "a different cap?) and is logged loudly and written "
                         "into every episode header as maxvel/maxvel_ckpt")
    ap.add_argument("--progress-file", default=None,
                    help="write live tick progress here as JSON, so a caller "
                         "(the dashboard) can show a real percentage instead "
                         "of an opaque spinner")
    ap.add_argument("--goal-band", default=None,
                    help="--goals ckpts: lo,hi straight-line distance "
                         "(u) of the per-episode air goal from the spawn "
                         "(default 300 .. 1500*goal_kmax)")
    ap.add_argument("--goal-holdout-only", action="store_true",
                    help="--goals ckpts: sample goals ONLY inside the "
                         "ckpt's held-out geodesic band (the G3 probe)")
    ap.add_argument("--goal-seed", type=int, default=0)
    ap.add_argument("--route", default=None,
                    help="override the ckpt's route file for the lookahead "
                         "fan (cross-map zero-shot probe: pair with --map "
                         "and that map's own field line). The ckpt must "
                         "have been TRAINED with --route; point count/span "
                         "stay the ckpt's, so the feature width is unchanged")
    ap.add_argument("--route-mode", choices=["live", "off", "frozen"],
                    default="live",
                    help="eval-side ablation of the --route lookahead fan "
                         "(research-plan-goallines.md E2 probe): 'off' zeroes "
                         "the fan block, 'frozen' holds the first decision's "
                         "features for the whole recording. Width unchanged; "
                         "a score that survives these means the policy is "
                         "not reading the fan")
    ap.add_argument("--depth-mode", choices=["live", "off", "frozen",
                                             "shuffle"], default="live",
                    help="eval-side depth ablation (research question 3): "
                         "'off' feeds the clear-sky encoding on every ray, "
                         "'frozen' repeats each episode's first frame, "
                         "'shuffle' permutes image rows with a fixed seeded "
                         "permutation. Scalars untouched; the recording is "
                         "otherwise the trained policy")
    ap.add_argument("--no-config-audit", action="store_true",
                    help="downgrade unmirrored-config errors to warnings")
    ap.add_argument("--dump-states", default=None,
                    help="also write the FULL per-tick STATE_DTYPE states of "
                         "every recorded episode to this .npz (keys ep_0000, "
                         "ep_0001, ...). The .jsonl is lossy - it carries no "
                         "basevelocity/duck bookkeeping - so anything that "
                         "SPAWNS from a recording (a demo spine) must read "
                         "the states, not the trajectory")
    args = ap.parse_args()

    say = _phase_writer(args.progress_file)
    say("loading checkpoint", 5)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    step = int(ck.get("global_step", 0))
    cfg_map = cfg.get('map', 'surf_ski_2')
    # --maps: a multi-map checkpoint carries the whole list; "map" is still
    # the first one, so an unqualified recording keeps recording that. Say
    # the others exist rather than let a reader assume the ckpt is
    # single-map - the weights are shared, so every listed map is a valid
    # thing to record and they will not score alike.
    cfg_maps = list(cfg.get("maps") or [])
    if cfg_maps and args.map is None:
        print(f"multi-map checkpoint ({', '.join(cfg_maps)}); recording "
              f"{cfg_map} - pass --map <name> for another")
    if args.map and cfg_maps and Path(args.map).stem not in cfg_maps:
        print(f"WARNING: --map {Path(args.map).stem} is not one of the "
              f"checkpoint's maps ({', '.join(cfg_maps)})")
    if args.map and not args.map.lower().endswith(".bsp"):
        args.map = str(ROOT / "maps" / f"{args.map}.bsp")
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
    # --pitch-fixed is MIRRORED, not TRAIN_ONLY: it aims the lidar, so a
    # recording that let the gaze drift would feed these weights a view they
    # were never trained on. The trainer pins the states' pitch column before
    # every render; _TorchPolicyBase does the same here (pitch_fixed=).
    pitch_fixed = cfg.get("pitch_fixed")
    pitch_rate = (0.0 if (fix_pitch is not None or pitch_fixed is not None)
                  else float(cfg.get("pitch_rate", -1.0)))
    # read the ckpt value FIRST and unconditionally, the --ep-ticks idiom
    # above: "args.X or cfg.get(X)" short-circuits when the CLI overrides it,
    # and then the config audit never sees the key as read.
    cfg_maxvel = float(cfg.get("maxvel", 2000.0))
    maxvel = cfg_maxvel if args.maxvel is None else float(args.maxvel)
    if args.maxvel is not None:
        print(f"!! --maxvel OVERRIDE: recording at sv_maxvelocity {maxvel:g} "
              f"instead of the checkpoint's {cfg_maxvel:g}. The speed cap is "
              f"part of the dynamics these weights were trained under, so "
              f"this recording is NOT physics parity and its times are not "
              f"comparable to the trainer's own evals.")
    say("starting sim", 12)
    core = SurfCore(map_path, default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=ep_ticks, water_fail=1,
        sv_maxvelocity=maxvel,          # physics parity unless --maxvel
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
    from surfgym.mapfleet import map_tag
    # --maps: "lidar_cell" is the ONE global cell only when --lidar-cell was
    # passed; otherwise the trainer picked one per map and recorded them in
    # "map_cells". pick_cell is the same function on the same bounds, so the
    # fallback agrees with training either way.
    cell = float((cfg.get("map_cells") or {}).get(
        map_tag(Path(map_path).stem),
        cfg.get("lidar_cell") or pick_cell(core)))
    # The SHAPING field's cell is NOT the lidar cell any more: --goal-cell
    # decoupled them, and the pool ships each map's field at its GATED cell
    # (often 48) while the lidar stays at 32. Reading map_cells here asks for
    # a cell nobody baked, so the cache misses and a recording sits for
    # minutes rebuilding a field that is already on disk. Accepts a scalar or
    # a per-map comma list aligned to cfg["maps"].
    gcell = cell
    gcells = cfg.get("goal_cells")          # multi-map: {tag: cell}
    gc = cfg.get("goal_cell")               # single: a scalar, or the CLI list
    if isinstance(gcells, dict) and gcells:
        gcell = float(gcells.get(map_tag(Path(map_path).stem), gcell))
    elif isinstance(gc, str) and "," in gc:
        parts = [x.strip() for x in gc.split(",")]
        names = cfg.get("maps") or []
        tag = map_tag(Path(map_path).stem)
        idx = next((i for i, m in enumerate(names) if map_tag(m) == tag), None)
        if idx is not None and idx < len(parts) and parts[idx]:
            gcell = float(parts[idx])
    elif gc:
        gcell = float(gc)

    gf = None
    # --obs-compass on a --goal-reward euclid/geo ckpt: the per-env
    # distance field the compass reads, re-centred on every goal by
    # _goal_meta below. Bound for real further down, next to ObsAux.
    cmp_goal_field = None
    if cfg.get("reward") == "race":
        # finish zone is armed for ANY race recording, whatever the spawns
        from surfgym.goalfield import EuclidField, build_goal_field
        from surfgym.zones import load_zones
        zones = load_zones(core.bsp_path)
        say(f"goal field @ cell {gcell:g}", 18)
        # Seed from the box that is ARMED on the next line - the invariant
        # train_fast.py documents at length. Seeding anything smaller also
        # re-keys the cache (the seed box is in the signature) and rebakes a
        # field the trainer already has on disk.
        gf = (EuclidField(zones["end"]) if cfg.get("race_dist") == "euclid"
              else build_goal_field(core, zones["end"], cell=gcell))
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
        # Two layouts. Single-map: {"states": ..., "map_id": ...}. Multi-map:
        # ONE reservoir PER MAP, keyed by bsp stem - {"surf_x": {"states":...}}.
        # Reading .get("states") off the multi-map dict yields None, which
        # reported as "trained without --respawn-frac" on a run that plainly
        # had 20,000 frontier states per map.
        resv = ck.get("respawn") or {}
        rs = resv.get("states")
        if rs is None and resv:
            stem = Path(map_path).stem
            sub = resv.get(stem)
            if sub is None:                       # tag match, e.g. mellow
                tag = map_tag(stem)
                sub = next((v for k, v in resv.items()
                            if isinstance(v, dict) and map_tag(k) == tag), None)
            if sub is None:
                raise SystemExit(
                    f"ckpt has per-map reservoirs but none for {stem!r}; "
                    f"have: {', '.join(sorted(resv))}")
            rs = sub.get("states")
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
    if pitch_fixed is not None:
        pool["pitch"] = float(pitch_fixed)
    if cfg.get("teleport_fail") or cfg.get("reward") == "race":
        core.set_teleport_fail(True)     # eval parity with training semantics
    core.set_spawn_pool(pool)
    print(f"spawn pool: {spawn} ({len(pool)} points)"
          + (f", pitch fixed {fix_pitch:g}" if fix_pitch is not None else "")
          + (f", pitch PINNED {float(pitch_fixed):g} deg every render"
             if pitch_fixed is not None else ""))

    say("initialising vision", 25)
    # --normals and --lidar-hfov/--lidar-vfov are what the policy SEES (a
    # 4-channel image; a different pixel grid), so they are mirrored, not
    # TRAIN_ONLY. Old checkpoints have no such keys: depth only, 120 x 90.
    lidar = GpuLidar(core, lw, lh,
                     hfov_deg=float(cfg.get("lidar_hfov") or 120.0),
                     vfov_deg=float(cfg.get("lidar_vfov") or 90.0),
                     range_units=float(cfg.get("lidar_range", 2000.0)),
                     near_range=cfg.get("lidar_near"),
                     cell=cell,
                     device=device,
                     surf_mask=bool(cfg.get("surf_mask", 0)),
                     pinhole=bool(cfg.get("pinhole", 0)),
                     normals=bool(cfg.get("normals", 0)))
    _ball = None
    if cfg.get("goals") and str(cfg.get("goal_obs") or "fan") in ("ball", "both"):
        # --goal-obs ball: mirror the second depth channel (the recorder
        # sets the ball's goal per episode below)
        from surfgym.goalball import GoalBallLidar
        if args.depth_mode != "live":
            raise SystemExit("--depth-mode with a goal-ball ckpt is not "
                             "supported (both channels would be ablated)")
        _ball = GoalBallLidar(lidar, core.num_envs,
                              radius=float(cfg.get("goal_radius") or 192.0),
                              views=int(cfg.get("goal_views") or 4))
        lidar = _ball
        print(_ball.describe())
    # --depth-mode: ablate the policy's depth input at eval time (research
    # question 3, 2026-08-25). The wrapper still renders (frozen needs a
    # real first frame, and the cost is irrelevant at batch 1) and then
    # replaces what the policy sees. Scalars are untouched, so a score that
    # survives these means the policy is not reading the image.
    if args.depth_mode != "live":
        _real_render = lidar.render
        _dm = {"frame": None, "prev_tick": None}
        near = cfg.get("lidar_near")
        rng_u = float(cfg.get("lidar_range", 2000.0))
        # the encoding's clear-sky value: 1.0 legacy; with the far tail it
        # is the value of a ray that runs to full range (vision.py)
        sky = 1.0 if (near is None or float(near) >= rng_u) else \
            1.0 + 0.25 * (1.0 - float(np.exp(-(rng_u - float(near)) / 2500.0)))
        _perm = torch.randperm(
            lidar.H, generator=torch.Generator().manual_seed(0))

        def _ablated_render(origin, yaw, pitch, ducked):
            out = _real_render(origin, yaw, pitch, ducked)
            if args.depth_mode == "off":
                return torch.full_like(out, sky)
            if args.depth_mode == "shuffle":
                return out[:, _perm.to(out.device), :]
            # frozen: hold each episode's first frame (tick counter restarts
            # at reset, which is how an episode boundary looks from here)
            tick = int(np.asarray(core.states_view["tick"])[0])
            if _dm["frame"] is None or (_dm["prev_tick"] is not None
                                        and tick < _dm["prev_tick"]):
                _dm["frame"] = out.clone()
            _dm["prev_tick"] = tick
            return _dm["frame"]

        lidar.render = _ablated_render
        print(f"depth-mode {args.depth_mode}: depth input ablated "
              f"(clear-sky value {sky:.4f}, scalars untouched)")
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
    if args.route and not cfg.get("route_file"):
        raise SystemExit("--route override given but this ckpt was not "
                         "trained with a lookahead fan (no route_file in "
                         "its config): the policy has no input columns for "
                         "it")
    if cfg.get("route_file"):
        rp = Path(args.route) if args.route else Path(cfg["route_file"])
        if args.route:
            print(f"--route OVERRIDE: fan features from {rp.name}, not the "
                  f"ckpt's {Path(cfg['route_file']).name} (cross-map probe)")
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
        if args.route_mode != "live":
            # eval-side ablation of the fan, the --depth-mode idiom: the
            # policy keeps its trained input WIDTH, only the CONTENT is
            # perturbed. A policy that ignores the fan is unaffected by
            # either mode; one that reads it collapses. This is the
            # research-plan-goallines.md E2 probe.
            route = _RouteProbe(route, args.route_mode)
            print(f"--route-mode {args.route_mode}: the lookahead fan is "
                  f"{'ZEROED' if args.route_mode == 'off' else 'FROZEN at the first decision'}"
                  " for this recording")
    goal_hooks = None
    if cfg.get("goals"):
        # --goals: the fan rides a per-env line (train_fast --goals); the
        # recording gives env 0 a random reachable-air goal per episode
        # (chord line), kills it on sphere entry, and writes the goal +
        # line into the episode header for the viewer.
        from surfgym.goals import AirSampler, MultiLine, chord_line
        if route is not None:
            raise SystemExit("--goals ckpt with a route file: not a thing")
        if gf is None:
            raise SystemExit("--goals ckpt needs the map's goal field")
        _gobs = str(cfg.get("goal_obs") or "fan")
        _ml = None
        if _gobs in ("fan", "both"):
            _ml = MultiLine(core.num_envs, device=device)
            route = _ml if args.route_mode == "live" else _RouteProbe(_ml, args.route_mode)
            print(_ml.describe() + (f"  [route-mode {args.route_mode}]"
                                    if args.route_mode != "live" else ""))
        if _ball is not None and args.route_mode == "off":
            _ball.mode = "off"          # the goal probe zeroes the ball too
            print("goal ball: OFF (zeroed channel) for this recording")
        _mins, _maxs = core.map_bounds()
        _rad = float(cfg.get("goal_radius") or 192.0)
        _kmax = float(cfg.get("goal_kmax") or 5.0)
        if args.goal_band:
            _lo, _hi = (float(v) for v in args.goal_band.split(","))
        else:
            _lo, _hi = 300.0, max(1500.0, 1500.0 * _kmax)
        _reach = gf.reachable
        if args.goal_holdout_only:
            if not cfg.get("goal_holdout"):
                raise SystemExit("--goal-holdout-only: ckpt has no holdout")
            _flo, _fhi = (float(v) for v in str(cfg["goal_holdout"]).split(","))
            _d0 = float(np.mean(gf.sample(core.get_states()["origin"])))
            _d0 = float(cfg.get("goal_d0") or _d0)

            def _reach(p, _g=gf, _a=_flo * _d0, _b=_fhi * _d0):
                d = np.asarray(_g.sample(np.asarray(p, np.float64)), np.float64)
                return _g.reachable(p) & (d >= _a) & (d <= _b)
            print(f"goals: HELD-OUT band only, d in [{_flo * _d0:,.0f}, "
                  f"{_fhi * _d0:,.0f}]u")
        _air = AirSampler(_mins, _maxs, _reach)
        _rng = np.random.default_rng(args.goal_seed)
        _ev = {"n": 0, "succ": 0, "pending": False, "center": None,
               "t0": 0, "ticks": [], "dists": []}

        def _goal_meta(ep):
            o = core.states_view["origin"][0].astype(np.float64)
            try:
                g = _air.sample_near(64, o, _lo, _hi, _rng)[0]
            except RuntimeError:
                g = _air.sample(512, _rng)[0]
            g = np.asarray(g, np.float64)
            line = chord_line(o, g)
            if _ml is not None:
                _ml.set_lines(np.array([0]), [line])
            if _ball is not None:
                _ball.set_goals([0], [g])
            if cmp_goal_field is not None:
                # --obs-compass points at THIS episode's goal; a stale
                # centre would feed the policy a compass training never wrote
                cmp_goal_field.set([0], [g])
            _ev["center"] = g
            _ev["pending"] = False
            _ev["n"] += 1
            _ev["dists"].append(float(np.linalg.norm(g - o)))
            thin = line[:: max(1, len(line) // 64)]
            return {"goal": {"center": [float(v) for v in g], "radius": _rad},
                    "line": [[float(v) for v in p] for p in thin]}

        def _goal_tick(t, states, rewards, done, trunc):
            if bool(done[0]) or bool(trunc[0]):
                if _ev["pending"]:
                    _ev["succ"] += 1
                    _ev["ticks"].append(t - _ev["t0"])
                _ev["pending"] = False
                _ev["t0"] = t + 1
                return
            if _ev["pending"] or _ev["center"] is None:
                return
            o = core.states_view["origin"][0].astype(np.float64)
            if np.linalg.norm(o - _ev["center"]) <= _rad:
                m = np.zeros(core.num_envs, np.uint8)
                m[0] = 1
                core.force_fail(m)
                _ev["pending"] = True

        goal_hooks = (_goal_meta, _goal_tick, _ev)
    route_dim = route.n_features if route is not None else 0
    # --race-latch: the flag is a 1-wide OBSERVATION block concatenated
    # LAST, exactly where the route fan's columns go (train_fast.py
    # N_LATCH). A recording that skipped it would build a row one column
    # narrow and the state_dict load would fail; one that fed a constant
    # would not be the trained policy in the only states that matter.
    d_latch = float(cfg.get("race_latch") or 0.0)
    # --race-latch-frac is the SAME column with a per-map threshold: the
    # latch is frac * this map's own start geodesic, so recording a
    # multi-map ckpt on the short map must not reuse the long map's
    # distance.
    latch_frac = float(cfg.get("race_latch_frac") or 0.0)
    if latch_frac > 0.0 and gf is not None:
        d_latch = latch_frac * float(np.mean(gf.sample(
            map_spawn_pool(core)["origin"])))
    latch_dim = 1 if (d_latch > 0.0 or latch_frac > 0.0) else 0
    if latch_dim and gf is None:
        raise SystemExit("this ckpt uses --race-latch but has no goal "
                         "field to recompute the flag from")
    route_dim += latch_dim
    # --act-hist / --obs-compass: the TRAILING block of the scalar half,
    # after the fan and the latch (surfgym/obsaux.py carries the layout).
    # Same rule as the two above - a recording that skipped them would build
    # a row 6*K+5 columns narrow and the state_dict load would fail; one
    # that fed constants would not be the trained policy in the only states
    # that matter. The class is the SAME one the trainer drives, so there is
    # no second implementation to drift out of sync.
    obs_aux = None
    hist_k = int(cfg.get("act_hist") or 0)
    want_compass = int(cfg.get("obs_compass") or 0)
    if hist_k or want_compass:
        from surfgym.obsaux import ObsAux
        aux_field = None
        if want_compass:
            if gf is None:
                raise SystemExit("this ckpt uses --obs-compass but has no "
                                 "goal field to read the gradient from")
            # the compass follows whichever field RaceReward shaped on: the
            # per-env GoalDistField under --goal-reward euclid/geo (so it
            # points at the GOAL, re-centred by _goal_meta on every episode),
            # otherwise the map's own geodesic field
            if cfg.get("goals") and cfg.get("goal_reward") in ("euclid",
                                                               "geo"):
                from surfgym.goals import GoalDistField
                aux_field = GoalDistField(
                    core.num_envs,
                    geo=(gf if cfg.get("goal_reward") == "geo" else None))
                cmp_goal_field = aux_field
            else:
                aux_field = gf
        obs_aux = ObsAux(core.num_envs, k=hist_k, field=aux_field,
                         yaw_adaptive=bool(cfg.get("yaw_adaptive")))
        route_dim += obs_aux.n_features
        print(obs_aux.describe())
    policy = Policy(core.obs_dim + route_dim + lw * lh * lidar.channels * stack,
                    lw, lh,
                    emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)),
                    # --trunk is MIRRORED, not TRAIN_ONLY: it selects the
                    # image encoder, so it changes what the policy computes
                    # from the very same pixels. Old checkpoints have no such
                    # key and are the plain trunk by construction.
                    trunk=str(cfg.get("trunk") or "plain"),
                    # --rnn is MIRRORED too: the GRU is a module of the
                    # state_dict and its state is carried across decisions
                    # by _TorchPolicyBase._net, zeroed at every episode
                    # start off the core's tick counter - exactly the
                    # trainer's rule. Old checkpoints have no key = none.
                    rnn=str(cfg.get("rnn") or "none"),
                    rnn_size=int(cfg.get("rnn_size") or 256),
                    # --tower-depth / --conv-mult decide which tensors the
                    # policy has and how wide they are, so they are MIRRORED
                    # like --trunk. Old checkpoints have neither key and are
                    # the two-layer tower over the 16/32/64 stack.
                    tower_depth=int(cfg.get("tower_depth") or 2),
                    conv_mult=int(cfg.get("conv_mult") or 1),
                    extra_feat=extra,
                    in_ch=lidar.channels * stack,
                    n_codes=n_codes, chunk=chunk,
                    route_dim=route_dim,
                    route_critic_only=bool(cfg.get("route_critic_only"))
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
        tp = float(cfg.get("time_pen") or 0.005)
        # --max-step is RaceReward's per-TICK teleport clip on the shaping
        # delta, and slot 12 carries the policy's OWN shaping - a mirror that
        # clipped at a different width would feed these weights a feature
        # they were never trained on, exactly like --race-dfloor below. Old
        # checkpoints have no key and clipped at 100.
        ms = float(cfg.get("max_step") or 100.0)
        if cfg.get("race_arc"):
            # --race-arc replaces the geodesic potential with arc length
            # along a reference line, and under --obs-reward the policy READS
            # its own shaping in slot 12. Feeding the geodesic term to an
            # arc-trained policy is the same train/record mismatch that made
            # sOBSR's recordings meaningless - mirror the arc term instead.
            from surfgym.route import ArcProgress
            _ap = Path(str(cfg["race_arc"]))
            if not _ap.exists():
                _ap = ROOT / Path(str(cfg["race_arc"])).name
                if not _ap.exists():
                    _ap = ROOT / "maps" / Path(str(cfg["race_arc"])).name
            _arc = ArcProgress.load(
                _ap,
                corridor=float(cfg.get("race_arc_corridor") or 1500.0),
                window=int(cfg.get("race_arc_window") or 16))
            scale = 100.0 / _arc.length * float(cfg.get("race_shaping") or 1.0)
            _sp = {"p": None}

            def _feed(c, _a=_arc, _s=scale, _tp=tp, _k=act_every, _ms=ms):
                p = c.states_view["origin"].astype(np.float64)
                prev, _sp["p"] = _sp["p"], p.copy()
                if prev is None or len(prev) != len(p):
                    _a.reset(p)
                    return np.zeros(len(p), np.float32)
                jump = np.linalg.norm(p - prev, axis=1) > _ms * _k
                if jump.any():
                    _a.reset(p, mask=jump)
                delta, _in = _a.advance(p)
                delta = np.where(jump, 0.0, delta)
                delta = np.clip(delta, -_ms * _k, _ms * _k)
                return np.tanh((delta * _s - _tp * _k) / 0.1).astype(np.float32)
        else:
            if gf is None:
                raise SystemExit("this ckpt uses --obs-reward but has no "
                                 "goal field to recompute it from")
            d0 = float(np.mean(gf.sample(core.get_states()["origin"])))
            # --race-shaping scales the trainer's potential; the feed
            # mirrors it
            scale = 100.0 / max(d0, 1.0) * float(cfg.get("race_shaping")
                                                 or 1.0)
            _st = {"d": None}
            # --race-dfloor clamps the potential and --race-latch switches it
            # off; slot 12 carries the policy's OWN shaping, so a mirror that
            # reported the raw term would feed these weights a feature they
            # were never trained on.
            d_floor = float(cfg.get("race_dfloor") or 0.0)

            # --race-ng levies a PER-CALL conformant tax (1-gamma^k)*Phi that
            # the trainer writes into this very slot (train_fast
            # _make_eval_reward_feed): "mirror the conformant tax or an
            # ng-trained policy reads a slot the training never produced".
            # Only the terminal charge has no mirror - the episode ends
            # there. gamma is per PHYSICS TICK and the trainer raises it to
            # act_every, so the eval feed must too.
            ng = int(cfg.get("race_ng") or 0)
            ng_g = float(cfg.get("gamma", 0.9995)) ** act_every

            def _feed(c, _f=gf, _s=scale, _tp=tp, _k=act_every,
                      _fl=d_floor, _ng=ng, _ngg=ng_g, _ngd0=d0, _ms=ms):
                d = _f.sample(c.states_view["origin"]).astype(np.float64)
                if _fl > 0.0:
                    d = np.maximum(d, _fl)
                prev, _st["d"] = _st["d"], d
                if prev is None or len(prev) != len(d):
                    return np.zeros(len(d), np.float32)
                delta = np.clip(prev - d, -_ms * _k, _ms * _k)
                if latch_fn is not None:
                    # the flag as of the PREVIOUS decision - the one that
                    # governed the reward this slot reports
                    was = latch_fn.state["f"]
                    if was is not None and len(was) == len(delta):
                        delta = np.where(was, 0.0, delta)
                r = delta * _s - _tp * _k
                if _ng:
                    r = r - (1.0 - _ngg) * (_ngd0 - d) * _s
                return np.tanh(r / 0.1).astype(np.float32)

        extra_slot, extra_fn = 12, _feed
    audit_cfg(cfg, strict=not args.no_config_audit)
    total_budget = args.episodes * ep_ticks
    # --eval-stall: TRAINING's stall rule on the recording. Every constant is
    # read off the checkpoint's own config so this is a MIRROR, not a second
    # policy: the same --stall-secs window in physics ticks, the same 32u
    # improvement threshold against a running minimum, the same RAW geodesic
    # (the --race-dfloor clamp is deliberately not applied - "stall/stagnant
    # keep the raw d"), and the same PER-CALL cadence, which is act_every *
    # chunk under --reward-per-decision and 1 otherwise. The cadence is the
    # part that is easy to get wrong: the threshold is per call, so testing
    # it every tick at act_every 4 would quarter the effective step and kill
    # legitimate flight.
    ev_stall = int(cfg.get("eval_stall") or 0) if args.eval_stall is None \
        else int(args.eval_stall)
    stall_hook = None
    header_extra = {"eval_stall": 0}
    if ev_stall:
        if gf is None:
            raise SystemExit("--eval-stall needs a goal field to stall "
                             "against; this ckpt has none")
        _stall_ticks = int(float(cfg.get("stall_secs") or 15.0) * 100.0)
        # train_fast: `every = KH if --reward-per-decision else 1`, and
        # KH = act_every * max(1, chunk)
        _stall_every = (act_every * max(1, int(chunk or 0))
                        if cfg.get("reward_per_decision") else 1)
        # --stall-eps: the per-CALL improvement threshold. It is a MIRROR
        # of the training rule, so it has to be the run's own value - the
        # flag scales with --act-every, and a hardcoded 32 would apply a
        # different kill rule to the recording than to training. Old
        # checkpoints have no key and trained at RaceReward's default.
        _stall_eps = float(cfg.get("stall_eps") or 32.0)
        header_extra = {"eval_stall": 1, "stall_ticks": _stall_ticks,
                        "stall_eps": _stall_eps, "stall_every": _stall_every}
        _ss = {"best": None, "since": 0, "phase": 0, "n": 0}

        def stall_hook(t, states, rewards, done, trunc, _c=core, _f=gf,
                       _tk=_stall_ticks, _eps=_stall_eps, _ev=_stall_every):
            if bool(done[0]) or bool(trunc[0]):
                _ss["best"] = None
                _ss["since"] = 0
                _ss["phase"] = 0
                return
            if _tk <= 0:
                return
            _ss["phase"] += 1
            if _ss["phase"] < _ev:
                return
            _ss["phase"] = 0
            d = float(_f.sample(
                _c.states_view["origin"][0:1].astype(np.float64))[0])
            best = _ss["best"]
            if best is None:
                _ss["best"] = d
                return
            if d < best - _eps:
                _ss["best"] = d
                _ss["since"] = 0
                return
            _ss["best"] = min(best, d)
            _ss["since"] += _ev
            if _ss["since"] >= _tk:
                _ss["since"] = 0     # re-arm, exactly like pop_stall_mask
                _ss["n"] += 1
                m = np.zeros(_c.num_envs, np.uint8)
                m[0] = 1
                _c.force_fail(m)

        stall_hook.state = _ss
        print(f"--eval-stall: killing an episode after {_stall_ticks / 100:g}s "
              f"without a {_stall_eps:g}u improvement (checked every "
              f"{_stall_every} tick(s)) - training's rule, on the recording")
    if args.maxvel is not None:
        # a recording made under a different speed cap must SAY SO in its own
        # file: every downstream honesty tool reads the traj, not this log
        header_extra["maxvel"] = maxvel
        header_extra["maxvel_ckpt"] = cfg_maxvel
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

    # --dump-states: record.py hands on_tick the PRE-step snapshot of every
    # env, which is exactly the row the .jsonl line for that tick describes,
    # so episode i of the dump aligns 1:1 with episode i of the trajectory.
    dump = None
    if args.dump_states:
        dump = {"eps": [], "cur": []}
        _prev_hook = on_tick

        def on_tick(t, states, rew, done, trunc, _p=_prev_hook, _d=dump):
            _d["cur"].append(states[0].copy())
            if bool(done[0]) or bool(trunc[0]):
                _d["eps"].append(np.array(_d["cur"], dtype=states.dtype))
                _d["cur"] = []
            if _p is not None:
                _p(t, states, rew, done, trunc)

    if stall_hook is not None:
        _prev_tick = on_tick

        def on_tick(t, states, rewards, done, trunc, _a=stall_hook,
                    _b=_prev_tick):
            if _b is not None:
                _b(t, states, rewards, done, trunc)
            _a(t, states, rewards, done, trunc)

    episode_meta = None
    if goal_hooks is not None:
        _gm, _gt, _gev = goal_hooks
        episode_meta = _gm
        _prev_tick = on_tick

        def on_tick(t, states, rewards, done, trunc, _a=_gt, _b=_prev_tick):
            _a(t, states, rewards, done, trunc)
            if _b is not None:
                _b(t, states, rewards, done, trunc)
    record_rollout(core, cls(policy, HeadPacker(device), device, lidar, core,
                             act_every, stack, extra_slot=extra_slot,
                             extra_fn=extra_fn, route=route,
                             latch_fn=latch_fn, pitch_fixed=pitch_fixed, aux=obs_aux),
                   out, episodes=args.episodes, max_ticks=total_budget,
                   seed=seed, on_tick=on_tick, episode_meta=episode_meta,
                   header_extra=header_extra)
    if stall_hook is not None and stall_hook.state["n"]:
        print(f"--eval-stall: {stall_hook.state['n']} episode(s) killed for "
              f"stalling")
    if goal_hooks is not None:
        _gev = goal_hooks[2]
        _md = float(np.mean(_gev["dists"])) if _gev["dists"] else float("nan")
        _mt = (float(np.mean(_gev["ticks"])) / 100.0) if _gev["ticks"] else float("nan")
        print(f"goals: {_gev['succ']}/{_gev['n']} reached  mean dist "
              f"{_md:,.0f}u  mean time {_mt:.1f}s  [route-mode {args.route_mode}]")
    if dump is not None:
        if dump["cur"]:            # budget ran out mid-episode: keep the tail
            dump["eps"].append(np.array(dump["cur"],
                                        dtype=dump["eps"][0].dtype
                                        if dump["eps"] else None))
        np.savez(args.dump_states,
                 **{f"ep_{i:04d}": e for i, e in enumerate(dump["eps"])})
        print(f"dumped {len(dump['eps'])} episode state arrays "
              f"({sum(len(e) for e in dump['eps']):,} ticks) -> "
              f"{args.dump_states}")
    kind = "stochastic" if args.stochastic else "greedy"
    print(f"recorded {args.episodes} {kind} episode(s) at step {step:,} -> {out}")


if __name__ == "__main__":
    main()
