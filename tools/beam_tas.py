"""Policy-guided population search ("beam TAS") for a record run.

Runs N stochastic copies of a trained race checkpoint's policy in the
batched sim, all starting from ONE identical spawn state, and every R
decisions clones the leaders (ranked by geodesic distance-to-goal) over
the laggards via ``core.set_state`` - carrying each survivor's full
action history with it. Because the whole population runs in lockstep
wall-ticks, every valid env's elapsed episode time equals the global
tick, so the FIRST goal crossing is also the fastest run the search can
ever produce; later crossings only add to the finisher count. The
winning action sequence is then replayed open-loop (no policy) on a
fresh 1-env core from the same spawn state and must reproduce the finish
deterministically, tick for tick - the script asserts the finish tick
and the pre-finish origin/velocity are bit-identical.

This is the deterministic special case of MCTS: the policy is the
proposal distribution, the real simulator is the model, and truncation
selection replaces UCB.

Checkpoint-faithful loading (config handling, Policy construction,
GpuLidar setup, act_every hold, the --obs-reward slot-12 feed) mirrors
tools/record_ckpt.py, and the config audit is IMPORTED from it, so this
tool inherits its refuse-on-unknown-keys safety - every misleading
recording ever shipped came from hand-rolled loading. Config knobs that
carry PER-ENV INFERENCE STATE this tool cannot clone across a set_state
(frame ring, chunk plan, latch flag, route file) are refused outright.

Two per-env physics fields live OUTSIDE SurfState and are NOT copied by
set_state: consumed push-once trigger flags (``once_used``) and stuck-
nudge bookkeeping (``PmPersist``). surf_src_cannonball has zero
trigger_push entities, and a run that engages the stuck nudge is five
ticks from a fail anyway; the replay assert is the backstop that would
catch either. Same limitation as the trainer's own reservoir respawns.

The traj jsonl rows match surfgym.record byte-for-byte; the trailer
"end" label here is derived from the core's goal_hits flag (honest),
not from record.py's +50-reward heuristic, which mislabels a goal-box
finish as "fail" because the record path sets no waypoints and the core
therefore emits reward 0 on completion.

Usage:
    python tools/beam_tas.py                      # F_prime, full search
    python tools/beam_tas.py --greedy-only        # just the sanity gate
    python tools/beam_tas.py CKPT --envs 1024 --resample-every 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch

from surfgym import SurfCore, default_config
from surfgym.core import SURF_IN_DUCK, SURF_IN_JUMP, phys_to_dict
from surfgym.rewards import map_spawn_pool
from train_fast import (GreedyTorchPolicy, HeadPacker, Policy,
                        SampledTorchPolicy)
import record_ckpt as _rc   # audit_cfg: inherit refuse-on-unknown-keys

DEF_CKPT = "C:/RL_Surf/runs/frozen/F_prime.pt"
# ABSOLUTE main-checkout maps dir. The worktree's maps/ is a COPY with
# different mtimes, and every cache (goal field, SDF, occ) keys on
# size+mtime_ns of the bsp - resolving the map inside the worktree
# silently triggers a ~30-minute goal-field re-bake (CLAUDE.md).
MAIN_MAPS = Path("C:/RL_Surf/maps")

# Config knobs that add PER-ENV inference state (frame ring, chunk plan,
# latch flag) or need a side file (route). Cloning an env mid-episode
# must clone that state too; v1 does not implement it, so it refuses
# rather than run with silently wrong semantics.
UNSUPPORTED = ("route_file", "race_latch", "race_latch_frac", "chunk",
               "frame_stack")


def resolve_map(name_or_path, cfg_map):
    p = str(name_or_path or cfg_map)
    if p.lower().endswith(".bsp"):
        return p
    for base in (MAIN_MAPS, ROOT / "maps"):
        c = base / (p + ".bsp")
        if c.exists():
            if base != MAIN_MAPS:
                print(f"WARNING: {c} is not the main checkout - a stale "
                      f"mtime here re-bakes every cache")
            return str(c)
    raise SystemExit(f"map not found: {p!r}")


def build_sim(cfg, map_path, num_envs, ep_cap):
    """Physics core exactly as record_ckpt.py builds it (eyeless; vision
    is GPU-side)."""
    fix_pitch = cfg.get("fix_pitch")
    pitch_rate = 0.0 if fix_pitch is not None else float(
        cfg.get("pitch_rate", -1.0))
    return SurfCore(map_path, default_config(
        num_envs=num_envs, spawn_mode=2, max_episode_ticks=ep_cap,
        water_fail=1,
        sv_maxvelocity=float(cfg.get("maxvel", 2000.0)),
        yaw_adaptive=1 if cfg.get("yaw_adaptive") else 0,
        lidar_w=0, lidar_h=0,
        pitch_rate_max_deg=pitch_rate))


def run_episode(core, act_fn, obs, fout, max_ticks, header, episode_idx):
    """Roll env 0 ONE episode from the core's current state, writing traj
    rows in surfgym.record's exact format. Returns
    (end, ticks, finished, pre_finish_state)."""
    fout.write(json.dumps({**header, "episode": episode_idx},
                          separators=(",", ":")) + "\n")
    ep_ticks, best_progress = 0, 0.0
    finished, end, pre_state = False, "trunc", None
    for _ in range(max_ticks):
        s0 = core.get_states()[0]           # pre-step snapshot (copy)
        actions = act_fn(obs)
        obs, rew, done, trunc, _term = core.step(actions)
        r0 = float(rew[0])
        best_progress = max(best_progress, float(s0["best_progress"]),
                            float(s0["progress"]))
        buttons = (SURF_IN_JUMP if actions[0, 4] else 0) | (
            SURF_IN_DUCK if actions[0, 5] else 0)
        ox, oy, oz = (float(v) for v in s0["origin"])
        vx, vy, vz = (float(v) for v in s0["velocity"])
        line = [ep_ticks,
                round(ox, 2), round(oy, 2), round(oz, 2),
                round(vx, 2), round(vy, 2), round(vz, 2),
                round(float(s0["yaw"]), 2),
                int(buttons),
                int(int(s0["onground"]) >= 0),
                round(float(s0["progress"]), 2),
                round(r0, 5),
                round(float(s0["pitch"]), 2),
                int(actions[0, 2]), int(actions[0, 3])]
        fout.write(json.dumps(line, separators=(",", ":")) + "\n")
        ep_ticks += 1
        if done[0] or trunc[0]:
            finished = bool(core.goal_hits[0])
            end = "done" if finished else ("fail" if done[0] else "trunc")
            pre_state = s0
            break
    fout.write(json.dumps({"end": end, "ticks": ep_ticks,
                           "best_progress": round(best_progress, 2)},
                          separators=(",", ":")) + "\n")
    fout.flush()
    return end, ep_ticks, finished, pre_state


class Playback:
    """Open-loop replay: feed a recorded per-tick action sequence, ignore
    the observation entirely. No policy, no GPU."""

    def __init__(self, seq_ticks):
        self.seq, self.t = seq_ticks, 0

    def act(self, _obs):
        a = self.seq[min(self.t, len(self.seq) - 1)]
        self.t += 1
        return np.ascontiguousarray(a.reshape(1, 6), dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(
        description="policy-guided population search for a record run")
    ap.add_argument("ckpt", nargs="?", default=DEF_CKPT)
    ap.add_argument("--map", default=None, help="defaults to the ckpt's map, "
                    "resolved in the MAIN checkout's maps/ (cache mtimes)")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--resample-every", type=int, default=25,
                    help="decisions per generation (25 dec x act_every 3 "
                    "= 75 ticks = 0.75 s between clonings)")
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--max-ticks", type=int, default=12000)
    ap.add_argument("--gens", type=int, default=0,
                    help="stop this many generations after the first finish "
                    "(a later finish can never be faster; 0 = run to "
                    "--max-ticks and count finishers)")
    ap.add_argument("--seed", type=int, default=0, help="spawn draw seed")
    ap.add_argument("--torch-seed", type=int, default=0,
                    help="proposal sampling seed")
    ap.add_argument("--greedy-eps", type=int, default=3,
                    help="greedy sanity episodes to try (first finisher "
                    "supplies the matched spawn state and baseline time)")
    ap.add_argument("--greedy-only", action="store_true",
                    help="run the greedy sanity gate and exit")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "beam_tas"))
    ap.add_argument("--log-every", type=int, default=5,
                    help="print every Nth generation")
    ap.add_argument("--no-config-audit", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    step = int(ck.get("global_step", 0))
    if cfg.get("reward") != "race":
        raise SystemExit("beam_tas needs a race checkpoint (goal-box "
                         f"finish); this one has reward={cfg.get('reward')!r}")
    bad = [k for k in UNSUPPORTED if cfg.get(k)]
    if bad:
        raise SystemExit("beam_tas v1 cannot clone the per-env inference "
                         "state of: " + ", ".join(bad))
    _rc.audit_cfg(cfg, strict=not args.no_config_audit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.torch_seed)
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    act_every = int(cfg.get("act_every", 1))
    K = act_every
    cfg_ep_ticks = int(cfg.get("ep_ticks", 700))
    ep_cap = max(cfg_ep_ticks, int(args.max_ticks))
    map_path = resolve_map(args.map, cfg.get("map", "surf_ski_2"))
    print(f"ckpt step {step:,}  act_every {K}  map {map_path}")

    # ---- 1-env core: greedy sanity gate + spawn-state capture ----------
    core1 = build_sim(cfg, map_path, 1, ep_cap)
    from surfgym.mapfleet import map_tag
    from surfgym.vision import GpuLidar, pick_cell
    cell = float((cfg.get("map_cells") or {}).get(
        map_tag(Path(map_path).stem),
        cfg.get("lidar_cell") or pick_cell(core1)))
    # goal-field cell: --goal-cell decoupled it from the lidar cell;
    # asking for an unbaked cell would rebuild a field already on disk
    gcell = cell
    gcells = cfg.get("goal_cells")
    gc = cfg.get("goal_cell")
    if isinstance(gcells, dict) and gcells:
        gcell = float(gcells.get(map_tag(Path(map_path).stem), gcell))
    elif isinstance(gc, str) and "," in gc:
        parts = [x.strip() for x in gc.split(",")]
        names = cfg.get("maps") or []
        tag = map_tag(Path(map_path).stem)
        i = next((j for j, m in enumerate(names) if map_tag(m) == tag), None)
        if i is not None and i < len(parts) and parts[i]:
            gcell = float(parts[i])
    elif gc:
        gcell = float(gc)

    from surfgym.goalfield import EuclidField, build_goal_field
    from surfgym.zones import load_zones
    zones = load_zones(core1.bsp_path)
    t0 = time.time()
    gf = (EuclidField(zones["end"]) if cfg.get("race_dist") == "euclid"
          else build_goal_field(core1, zones["end"], cell=gcell))
    dt = time.time() - t0
    print(f"goal field @ cell {gcell:g} in {dt:.1f}s"
          + ("  ** WARNING: that smells like a RE-BAKE - wrong map "
             "path/mtime? **" if dt > 30 else ""))

    raw = map_spawn_pool(core1)
    pool = map_spawn_pool(core1, yaw=gf.descent_yaw(raw["origin"]))
    pool["pitch"] = -10.0
    if cfg.get("fix_pitch") is not None:
        pool["pitch"] = float(cfg["fix_pitch"])
    d0 = float(np.mean(gf.sample(raw["origin"])))
    print(f"race: start geodesic {d0:.0f}u, spawn pool {len(pool)} points")

    def arm(core):
        core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
        if cfg.get("teleport_fail") or cfg.get("reward") == "race":
            core.set_teleport_fail(True)
        core.set_spawn_pool(pool)

    arm(core1)
    lidar = GpuLidar(core1, lw, lh,
                     range_units=float(cfg.get("lidar_range", 2000.0)),
                     near_range=cfg.get("lidar_near"),
                     cell=cell, device=device,
                     surf_mask=bool(cfg.get("surf_mask", 0)),
                     pinhole=bool(cfg.get("pinhole", 0)))
    stack = max(1, int(cfg.get("frame_stack") or 1))   # 1: refused above
    extra = (12,) if cfg.get("obs_reward") else ()
    policy = Policy(core1.obs_dim + lw * lh * lidar.channels * stack, lw, lh,
                    emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)),
                    extra_feat=extra,
                    in_ch=lidar.channels * stack,
                    n_codes=0, chunk=0, route_dim=0,
                    route_critic_only=bool(cfg.get("route_critic_only"))
                    ).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    packer = HeadPacker(device)

    def mk_feed():
        """--obs-reward slot-12 feed, per record_ckpt.py (a missing feed
        hands the policy absolute position where it expects tanh(reward)
        and kills the agent in seconds). d0 is the trainer's own formula
        (train_fast.py: mean field over the RAW map spawns) - record_ckpt
        samples pre-reset states there, which is a latent bug this tool
        does not copy. Returns (slot, fn, state); state['d'] is per-env
        prev-d and must be cloned donor->loser at a resample."""
        if not cfg.get("obs_reward"):
            return -1, None, None
        scale = 100.0 / max(d0, 1.0) * float(cfg.get("race_shaping") or 1.0)
        tp = float(cfg.get("time_pen") or 0.005)
        d_floor = float(cfg.get("race_dfloor") or 0.0)
        st = {"d": None}

        def _feed(c, _f=gf, _s=scale, _tp=tp, _k=K, _fl=d_floor):
            dd = _f.sample(c.states_view["origin"]).astype(np.float64)
            if _fl > 0.0:
                dd = np.maximum(dd, _fl)
            prev, st["d"] = st["d"], dd
            if prev is None or len(prev) != len(dd):
                return np.zeros(len(dd), np.float32)
            delta = np.clip(prev - dd, -100.0 * _k, 100.0 * _k)
            return np.tanh((delta * _s - _tp * _k) / 0.1).astype(np.float32)

        return 12, _feed, st

    header1 = {"map": Path(core1.bsp_path).stem,
               "tick_ms": int(core1.config.phys.msec),
               "phys": phys_to_dict(core1.config.phys)}

    # ---- phase 1: greedy sanity gate -----------------------------------
    # Fresh wrapper + fresh reset per episode so every episode's act_every
    # hold is aligned to its own tick 0, exactly like the search and the
    # replay (record_rollout's single global cadence would misalign
    # episodes 2+). The first FINISHING episode supplies the matched
    # spawn state row0 and the baseline time.
    gpath = out_dir / "greedy_baseline.jsonl"
    greedy_ticks, row0, obs_start = None, None, None
    with open(gpath, "w", encoding="utf-8", newline="\n") as f:
        for e in range(max(1, args.greedy_eps)):
            obs = core1.reset(args.seed + e)
            row = core1.get_states()[0:1].copy()      # STATE_DTYPE copy
            o0 = obs[0].copy()
            es1, ef1, _ = mk_feed()
            gpol = GreedyTorchPolicy(policy, packer, device, lidar, core1,
                                     K, stack, extra_slot=es1, extra_fn=ef1)
            end, ticks, fin, _ = run_episode(core1, gpol.act, obs, f,
                                             ep_cap, header1, e)
            print(f"greedy ep{e} (spawn seed {args.seed + e}): {end} in "
                  f"{ticks} ticks ({ticks / 100:.2f}s)")
            if fin:
                greedy_ticks, row0, obs_start = ticks, row, o0
                break
    if greedy_ticks is None:
        raise SystemExit(
            f"GATE FAILED: {Path(args.ckpt).name} did not finish in "
            f"{args.greedy_eps} greedy episode(s) - wrong checkpoint for a "
            "beam search; stopping per plan. Trajectories: " + str(gpath))
    print(f"greedy baseline: {greedy_ticks} ticks = "
          f"{greedy_ticks / 100:.2f}s -> {gpath}")
    if args.greedy_only:
        return

    # ---- phase 2: the population search --------------------------------
    N = int(args.envs)
    R = int(args.resample_every)
    gen_ticks = R * K
    max_ticks = int(args.max_ticks)
    n_elite = max(1, int(round(N * args.elite_frac)))
    coreN = build_sim(cfg, map_path, N, ep_cap)
    arm(coreN)
    obs = np.array(coreN.reset(args.seed))        # copy, then overwrite
    for i in range(N):
        coreN.set_state(i, row0)
    obs[:] = obs_start[None, :]
    esN, efN, feed_state = mk_feed()
    spol = SampledTorchPolicy(policy, packer, device, lidar, coreN,
                              K, stack, extra_slot=esN, extra_fn=efN)

    D_total = (max_ticks + K - 1) // K
    hist = np.zeros((D_total, N, 6), np.int8)     # per-decision actions
    valid = np.ones(N, bool)   # history describes this env from tick 0
    idx = np.arange(N)
    finishes = []              # (tick, env, gen)
    best = None
    gen, best_gen = 0, None
    print(f"search: {N} envs, resample every {R} decisions "
          f"({gen_ticks} ticks), elite {n_elite}, cap {max_ticks} ticks")
    t_loop = time.time()
    for t in range(max_ticks):
        d = t // K
        a = spol.act(obs)                  # re-decides when t % K == 0
        if t % K == 0:
            hist[d] = a                    # bins < 15: int8 is lossless
        sv = coreN.states_view
        pre_o = sv["origin"].copy()
        pre_v = sv["velocity"].copy()
        obs, _rew, done, trunc, _term = coreN.step(a)
        gh = coreN.goal_hits
        if gh.any():
            for i in np.nonzero(gh)[0]:
                if not valid[i]:
                    continue               # respawned body: not a real run
                finishes.append((t + 1, int(i), gen))
                if best is None:           # lockstep: first hit is fastest
                    best = {"tick": t + 1,
                            "acts": hist[:d + 1, i].copy(),
                            "pre_origin": pre_o[i].copy(),
                            "pre_vel": pre_v[i].copy()}
                    best_gen = gen
                    print(f"FINISH: env {i} at tick {t + 1} "
                          f"({(t + 1) / 100:.2f}s), gen {gen}")
        dead = np.asarray(done, bool) | np.asarray(trunc, bool)
        if dead.any():
            valid &= ~dead
        if best is not None and args.gens > 0 and gen - best_gen >= args.gens:
            break
        if (t + 1) % gen_ticks == 0 and (t + 1) < max_ticks:
            gen += 1
            states = coreN.get_states()
            dgeo = gf.sample(states["origin"]).astype(np.float64)
            order = np.argsort(np.where(valid, dgeo, np.inf), kind="stable")
            elig = order[valid[order]]
            if len(elig) == 0:
                # every lineage finished or died inside this window; a
                # reseed would restart episode clocks against the global
                # tick and corrupt the finish-time accounting, so stop
                print(f"gen {gen}: population extinct at t={t + 1} "
                      f"(all lineages finished/died); stopping search")
                break
            # lockstep invariant: every valid env's episode clock is the
            # global tick (clones inherit the donor's; a violation means
            # the finish-time accounting is wrong)
            assert int(states["tick"][elig[0]]) == t + 1, \
                (int(states["tick"][elig[0]]), t + 1)
            keep = elig[:n_elite]
            keep_set = np.zeros(N, bool)
            keep_set[keep] = True
            losers = idx[~keep_set]
            donors = keep[np.arange(len(losers)) % len(keep)]
            for j, don in zip(losers, donors):
                coreN.set_state(int(j), states[don])
            hist[:d + 1, losers] = hist[:d + 1, donors]
            obs = np.array(obs)            # patch clones' scalar obs too
            obs[losers] = obs[donors]
            valid[:] = True
            if feed_state is not None and feed_state.get("d") is not None:
                feed_state["d"][losers] = feed_state["d"][donors]
            if gen % max(1, args.log_every) == 0:
                print(f"gen {gen:3d} t={t + 1:5d} valid={len(elig):4d} "
                      f"min_d={dgeo[keep[0]]:8.0f} "
                      f"med_d={np.median(dgeo[keep]):8.0f} "
                      f"finishes={len(finishes)}")
    dt_loop = time.time() - t_loop
    fps = (t + 1) * N / max(dt_loop, 1e-9)
    print(f"search done: {t + 1} ticks x {N} envs in {dt_loop:.0f}s "
          f"({fps:,.0f} env-steps/s), {len(finishes)} finishes, "
          f"{gen} generations")

    if best is None:
        raise SystemExit("search produced no finisher - nothing to replay "
                         f"(greedy baseline was {greedy_ticks} ticks; try "
                         "more ticks/envs or a different --torch-seed)")

    # ---- phase 3: deterministic open-loop replay of the winner ---------
    core1b = build_sim(cfg, map_path, 1, ep_cap)
    arm(core1b)
    core1b.reset(args.seed)                # arbitrary; state overwritten
    core1b.set_state(0, row0)
    acts_ticks = np.repeat(best["acts"].astype(np.int32), K, axis=0)
    rpath = out_dir / "beam_best.jsonl"
    hdr = {"map": Path(core1b.bsp_path).stem,
           "tick_ms": int(core1b.config.phys.msec),
           "phys": phys_to_dict(core1b.config.phys)}
    with open(rpath, "w", encoding="utf-8", newline="\n") as f:
        end, ticks, fin, pre_state = run_episode(
            core1b, Playback(acts_ticks).act,
            np.zeros((1, core1b.obs_dim), np.float32), f, ep_cap, hdr, 0)
    if not fin:
        raise SystemExit(f"REPLAY DIVERGED: open-loop replay ended "
                         f"'{end}' at tick {ticks}, search finished at "
                         f"{best['tick']} - determinism broken")
    assert ticks == best["tick"], \
        f"replay finished at {ticks}, search said {best['tick']}"
    same_o = np.array_equal(pre_state["origin"], best["pre_origin"])
    same_v = np.array_equal(pre_state["velocity"], best["pre_vel"])
    assert same_o and same_v, (
        "replay finish tick matched but the pre-finish state is not "
        f"bit-identical: origin {pre_state['origin']} vs "
        f"{best['pre_origin']}, velocity {pre_state['velocity']} vs "
        f"{best['pre_vel']}")
    print(f"replay: bit-exact finish reproduced at tick {ticks} "
          f"({ticks / 100:.2f}s) -> {rpath}")

    npz = out_dir / "beam_best.npz"
    np.savez(npz,
             acts=best["acts"], act_every=np.int32(K),
             finish_ticks=np.int32(best["tick"]),
             spawn_state=row0,
             greedy_ticks=np.int32(greedy_ticks),
             seed=np.int32(args.seed), torch_seed=np.int32(args.torch_seed),
             ckpt=np.str_(str(args.ckpt)), map=np.str_(map_path))
    summary = {
        "ckpt": str(args.ckpt), "map": map_path, "envs": N,
        "resample_every_decisions": R, "elite_frac": args.elite_frac,
        "greedy_ticks": greedy_ticks, "greedy_s": greedy_ticks / 100.0,
        "best_ticks": best["tick"], "best_s": best["tick"] / 100.0,
        "gain_s": (greedy_ticks - best["tick"]) / 100.0,
        "finishes": len(finishes),
        "finish_ticks": sorted(ft for ft, _, _ in finishes),
        "generations": gen, "search_wall_s": round(dt_loop, 1),
        "env_steps_per_s": round(fps),
        "replay_bit_exact": bool(same_o and same_v),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"beam TAS: greedy {greedy_ticks / 100:.2f}s -> best "
          f"{best['tick'] / 100:.2f}s "
          f"({(greedy_ticks - best['tick']) / 100:+.2f}s), "
          f"{len(finishes)} finishers, total wall "
          f"{time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
