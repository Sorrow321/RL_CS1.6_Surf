"""plan_to_bc.py - turn a beam_tas finisher into expert-iteration data.

The planner (tools/beam_tas.py) leaves beam_best.npz: the spawn state, the
per-decision action table of its fastest finishing lineage (and, with
--keep-finishers K, of the K fastest distinct ones), and the spawn's 15
core scalars. This replays every kept line OPEN-LOOP on a fresh 1-env core
- the same deterministic replay beam_tas asserted bit-exact - and records,
at every decision, the full physics state, the 15 scalars the core emitted
(slot 12 replaced by the --obs-reward mirror), the --race-latch flag and
the action the planner committed. Two files come out:

* ``--out``   the BC dataset (surfgym.bc, one row per decision) that
              train_fast.py --bc-file distils from;
* ``--spine`` the best line's per-tick STATE_DTYPE states, time-ordered,
              for --demo-file (loop_spine.py's format: a finisher, no trim)
              so the RL half of the round spawns along the planner's line.

Only lines that finish on replay are kept (a line that does not reproduce
its finish is reported and dropped - the winner MUST reproduce, since
beam_tas already proved it does). Rows of slower lines can be down-weighted
with --line-weight-decay (w = exp(-decay * (t - t_best) / t_best)).

    python tools/plan_to_bc.py --plan runs/exit/round_0/plan/beam_best.npz \
        --ckpt runs/exit/seed_scalar.pt --out runs/exit/round_0/bc.npz \
        --spine runs/exit/round_0/spine.npy
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

import beam_tas
from surfgym.bc import make_eval_feeds, replay_line, save_bc_dataset
from surfgym.rewards import map_spawn_pool


def goal_cell_for(cfg: dict, map_path: str, core) -> float:
    """The goal field's cell, resolved as beam_tas/record_ckpt resolve it
    (single-map form: --goal-cell if given, else the lidar cell)."""
    from surfgym.mapfleet import map_tag
    from surfgym.vision import pick_cell
    cell = float((cfg.get("map_cells") or {}).get(
        map_tag(Path(map_path).stem),
        cfg.get("lidar_cell") or pick_cell(core)))
    gc = cfg.get("goal_cell")
    gcells = cfg.get("goal_cells")
    if isinstance(gcells, dict) and gcells:
        return float(gcells.get(map_tag(Path(map_path).stem), cell))
    if isinstance(gc, str) and "," in gc:
        parts = [x.strip() for x in gc.split(",")]
        names = cfg.get("maps") or []
        tag = map_tag(Path(map_path).stem)
        i = next((j for j, m in enumerate(names) if map_tag(m) == tag), None)
        if i is not None and i < len(parts) and parts[i]:
            return float(parts[i])
        return cell
    if gc:
        return float(gc)
    return cell


def open_planner_core(cfg: dict, map_path: str, ep_cap: int):
    """(core, gf, d0, zones) armed exactly as beam_tas arms its cores."""
    from surfgym.goalfield import EuclidField, build_goal_field
    from surfgym.zones import load_zones
    core = beam_tas.build_sim(cfg, map_path, 1, ep_cap)
    zones = load_zones(core.bsp_path)
    gcell = goal_cell_for(cfg, map_path, core)
    t0 = time.time()
    gf = (EuclidField(zones["end"]) if cfg.get("race_dist") == "euclid"
          else build_goal_field(core, zones["end"], cell=gcell))
    dt = time.time() - t0
    if dt > 30:
        print(f"WARNING: goal field took {dt:.0f}s - that smells like a "
              "RE-BAKE (wrong map path/mtime?)")
    raw = map_spawn_pool(core)
    pool = map_spawn_pool(core, yaw=gf.descent_yaw(raw["origin"]))
    pool["pitch"] = -10.0
    if cfg.get("fix_pitch") is not None:
        pool["pitch"] = float(cfg["fix_pitch"])
    d0 = float(np.mean(gf.sample(raw["origin"])))
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    if cfg.get("teleport_fail") or cfg.get("reward") == "race":
        core.set_teleport_fail(True)
    core.set_spawn_pool(pool)
    return core, gf, d0, zones


def build(plan_npz, ckpt, out, spine=None, map_path=None, lines=0,
          line_weight_decay=0.0, summary_out=None):
    z = np.load(plan_npz, allow_pickle=False)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    K = int(z["act_every"])
    if K != int(cfg.get("act_every", 1)):
        raise SystemExit(f"plan act_every {K} != ckpt act_every "
                         f"{cfg.get('act_every')}")
    map_path = beam_tas.resolve_map(map_path or str(z["map"]), cfg.get("map"))
    spawn_state = np.asarray(z["spawn_state"])
    obs_start = np.asarray(z["obs_start"], np.float32).reshape(-1)
    gate_seed = int(z["gate_seed"])
    if "acts_all" in z.files:
        acts_all = np.asarray(z["acts_all"])
        acts_len = np.asarray(z["acts_len"])
        ticks_all = np.asarray(z["finish_ticks_all"])
    else:
        acts_all = np.asarray(z["acts"])[None]
        acts_len = np.array([len(z["acts"])])
        ticks_all = np.array([int(z["finish_ticks"])])
    if lines > 0:
        acts_all, acts_len, ticks_all = (acts_all[:lines], acts_len[:lines],
                                         ticks_all[:lines])
    ep_cap = max(int(cfg.get("ep_ticks", 12000)), int(ticks_all.max()) + K)
    core, gf, d0, _zones = open_planner_core(cfg, map_path, ep_cap)
    slot_probe, rf_probe, lf_probe = make_eval_feeds(cfg, gf, d0, K)
    n_latch = 0 if lf_probe is None else 1
    obs_reward = rf_probe is not None
    t_best = int(ticks_all.min())

    all_states, all_scal, all_latch, all_act, all_w, all_id = \
        [], [], [], [], [], []
    best_ticks_states = None
    kept, dropped = [], []
    for j in range(len(acts_all)):
        acts = np.asarray(acts_all[j, :int(acts_len[j])], np.int64)
        core.reset(gate_seed)             # fresh episode clocks; state next
        _slot, rf, lf = make_eval_feeds(cfg, gf, d0, K)   # fresh per line
        rows, tick_states, finished, ticks = replay_line(
            core, spawn_state, obs_start, acts, K, rf, lf,
            max_ticks=ep_cap)
        want = int(ticks_all[j])
        if not finished:
            dropped.append((j, want, ticks, "no finish on replay"))
            if j == 0:
                raise SystemExit("the planner's WINNER did not reproduce its "
                                 f"finish on replay (ended at tick {ticks}, "
                                 f"search said {want}) - determinism broken")
            continue
        if ticks != want:
            dropped.append((j, want, ticks, "finish tick differs"))
            if j == 0:
                raise SystemExit(f"winner replay finished at {ticks}, "
                                 f"search said {want}")
            continue
        w = float(np.exp(-line_weight_decay * (ticks - t_best)
                         / max(t_best, 1)))
        kept.append((j, ticks, len(rows), w))
        if best_ticks_states is None:
            best_ticks_states = np.array(tick_states, dtype=spawn_state.dtype)
        for st, scal, latch, act in rows:
            all_states.append(st)
            all_scal.append(scal)
            all_latch.append(latch)
            all_act.append(act)
            all_w.append(w)
            all_id.append(j)
    if not kept:
        raise SystemExit("no planner line reproduced its finish")
    meta = {"plan": str(plan_npz), "ckpt": str(ckpt), "map": map_path,
            "act_every": K, "obs_reward": bool(obs_reward),
            "n_latch": int(n_latch),
            "d_latch": (0.0 if lf_probe is None else float(lf_probe.d_latch)),
            "d0": d0, "lines": len(kept), "line_ticks": [k[1] for k in kept],
            "line_rows": [k[2] for k in kept],
            "line_weights": [k[3] for k in kept],
            "best_ticks": t_best, "best_s": t_best / 100.0,
            "greedy_ticks": int(z["greedy_ticks"]),
            "dropped": [list(map(str, d)) for d in dropped],
            "gate_seed": gate_seed, "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_bc_dataset(out, np.array(all_states, dtype=spawn_state.dtype),
                    np.array(all_scal, np.float32), np.array(all_latch, np.float32),
                    np.array(all_act, np.int64), np.array(all_w, np.float32),
                    np.array(all_id, np.int32), meta)
    n_rows = len(all_states)
    print(f"bc: {n_rows:,} rows from {len(kept)} line(s) "
          f"(best {t_best / 100:.2f}s, greedy {int(z['greedy_ticks']) / 100:.2f}s"
          f"; {len(dropped)} dropped) -> {out}")
    for j, want, got, why in dropped:
        print(f"  dropped line {j}: {why} (search {want}, replay {got})")
    if spine:
        np.save(spine, best_ticks_states)
        print(f"spine: {len(best_ticks_states):,} per-tick states of the "
              f"best line -> {spine}")
        meta["spine"] = str(spine)
        meta["spine_len"] = int(len(best_ticks_states))
    meta["rows"] = int(n_rows)
    if summary_out:
        Path(summary_out).write_text(json.dumps(meta, indent=2),
                                     encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="planner line -> BC rows + spine")
    ap.add_argument("--plan", required=True, help="beam_tas beam_best.npz")
    ap.add_argument("--ckpt", required=True,
                    help="the checkpoint the plan was searched with (its "
                         "config picks the side-channel columns)")
    ap.add_argument("--out", required=True, help="BC dataset .npz")
    ap.add_argument("--spine", default=None, help="demo spine .npy")
    ap.add_argument("--map", default=None,
                    help="absolute .bsp (default: the plan's own path)")
    ap.add_argument("--lines", type=int, default=0,
                    help="use at most this many kept finishers (0 = all)")
    ap.add_argument("--line-weight-decay", type=float, default=0.0,
                    help="row weight exp(-decay*(t-t_best)/t_best) per line")
    ap.add_argument("--summary-out", default=None)
    a = ap.parse_args()
    build(a.plan, a.ckpt, a.out, spine=a.spine, map_path=a.map,
          lines=a.lines, line_weight_decay=a.line_weight_decay,
          summary_out=a.summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
