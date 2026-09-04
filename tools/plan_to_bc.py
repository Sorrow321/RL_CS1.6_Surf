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

NON-FINISHING lineages (beam_tas --objective progress|auto, the from-scratch
loop) are accepted too. Each one is replayed the same way and must reach
the order-only corridor arc the search credited it with (eval_honesty's
--order-only rule, the replay twin of the finisher's finish-tick assert;
a line that does not is dropped). Then it is TRIMMED at its last map
contact - tools/pick_selfline.py's rule, the last tick whose vertical
acceleration departs from the gravity step - because every non-finishing
lineage ends in a fall and the fall is not a demonstration: neither the BC
rows nor the spine carry a decision taken after that tick. A short line
that spends most of its ticks on a ramp would fool the data-derived
gravity step, so the step recovered from the data is checked against the
engine's own (sv_gravity * tick) and the engine's is used when they
disagree. Slower non-finishers are down-weighted on their arc deficit,
w = exp(-decay * (arc_best - arc) / arc_best). Finishers, if any, rank
first and are never trimmed.

``--spine-spacing U`` writes the spine at (at least) U map units of travel
between states instead of every tick, so a spawn pool drawn uniformly from
it is uniform ALONG the line rather than in time (0 = every tick, the
finisher loop's behaviour).

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
from eval_honesty import corridor_progress_ordered, load_route
from pick_selfline import contact_cut
from surfgym.bc import (contact_rows, last_contact_cut, make_eval_feeds,
                        rank_lineages, replay_line, save_bc_dataset,
                        subsample_by_path)
from surfgym.core import phys_to_dict
from surfgym.rewards import map_spawn_pool

ARC_TOL = 0.5          # map units: a replayed arc must match the search's


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


def _opt(z, key, cast, default=None):
    return cast(z[key]) if key in z.files else default


def load_plans(plan_files):
    """Pool the kept lineages of one or more beam_best.npz files that were
    searched from the SAME spawn (same gate seed -> same spawn state; the
    driver's waves differ only in --torch-seed). -> dict with the spawn,
    its obs row, act_every, map, greedy ticks, the objective and the arc
    settings the planner used, and ``lines``: rank_lineages' order -
    finishers fastest first, then non-finishers by best arc - each a dict
    with finish_tick (0 = none), acts (D, 6) int64, src, best_arc,
    arc_tick, end_tick."""
    head = None
    lines = []
    for f in plan_files:
        z = np.load(f, allow_pickle=False)
        st = np.asarray(z["spawn_state"])
        if head is None:
            head = {"spawn_state": st,
                    "obs_start": np.asarray(z["obs_start"], np.float32).reshape(-1),
                    "gate_seed": int(z["gate_seed"]), "K": int(z["act_every"]),
                    "map": str(z["map"]), "greedy_ticks": int(z["greedy_ticks"]),
                    "objective": _opt(z, "objective", str, "finish"),
                    "route_file": _opt(z, "route_file", str),
                    "arc_corridor": _opt(z, "arc_corridor", float),
                    "arc_window": _opt(z, "arc_window", int),
                    "arc_total": _opt(z, "arc_total", float),
                    "arc_bank": _opt(z, "arc_bank", str, "contact"),
                    "contact_tol": _opt(z, "contact_tol", float),
                    "gravity_step": _opt(z, "gravity_step", float),
                    "files": [str(f)]}
        else:
            if (st.tobytes() != head["spawn_state"].tobytes()
                    or int(z["act_every"]) != head["K"]):
                raise SystemExit(f"{f}: a different spawn state / act_every "
                                 f"than {head['files'][0]} - the lines "
                                 "cannot share one replay")
            head["files"].append(str(f))
            if head["route_file"] is None and "route_file" in z.files:
                head["route_file"] = str(z["route_file"])
                head["arc_corridor"] = _opt(z, "arc_corridor", float)
                head["arc_window"] = _opt(z, "arc_window", int)
                head["arc_total"] = _opt(z, "arc_total", float)
        if "acts_all" in z.files:
            aa, al, ta = (np.asarray(z["acts_all"]), np.asarray(z["acts_len"]),
                          np.asarray(z["finish_ticks_all"]))
            has_arc = "arc_all" in z.files
            for j in range(len(aa)):
                ft = int(ta[j])
                lines.append({
                    "finish_tick": ft,
                    "acts": np.asarray(aa[j, :int(al[j])], np.int64),
                    "src": str(f),
                    "best_arc": float(z["arc_all"][j]) if has_arc else 0.0,
                    "arc_tick": int(z["arc_tick_all"][j]) if has_arc else 0,
                    "end_tick": (int(z["end_tick_all"][j]) if has_arc
                                 else ft)})
        else:
            ft = int(z["finish_ticks"])
            lines.append({"finish_tick": ft,
                          "acts": np.asarray(z["acts"], np.int64),
                          "src": str(f), "best_arc": 0.0, "arc_tick": 0,
                          "end_tick": ft})
    head["lines"] = rank_lineages(lines)
    return head


def trim_last_contact(states, g_phys: float, tol: float = 1.0):
    """-> (cut, gravity_step, source): the last tick the map pushed back.

    pick_selfline's rule with the ENGINE's gravity step (the planner banks
    arc with the same constant, so the trim keeps exactly the ticks the
    search credited). pick_selfline's own contact_cut - the same rule with
    the step recovered as the median of diff(vz) - is run alongside as a
    cross-check: 'physics=median' when both cut at the same tick, else
    'physics (median g=..., cut=...)' - which happens on a short line that
    spends most of its ticks on a ramp or standing, where the median is an
    on-map acceleration and the data-derived rule would keep the fall."""
    cut, g = last_contact_cut(states, gravity_step=g_phys, tol=tol)
    cut_m, g_m = contact_cut(contact_rows(states), tol)
    if int(cut_m) == int(cut):
        return int(cut), float(g), "physics=median"
    return int(cut), float(g), f"physics (median g={g_m:g}, cut={int(cut_m)})"


def build(plan_npz, ckpt, out, spine=None, map_path=None, lines=0,
          line_weight_decay=0.0, summary_out=None, route=None,
          corridor=None, arc_window=None, contact_tol=None,
          spine_spacing=0.0, no_trim=False):
    files = [plan_npz] if isinstance(plan_npz, (str, Path)) else list(plan_npz)
    plans = load_plans(files)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    K = int(plans["K"])
    if K != int(cfg.get("act_every", 1)):
        raise SystemExit(f"plan act_every {K} != ckpt act_every "
                         f"{cfg.get('act_every')}")
    map_path = beam_tas.resolve_map(map_path or plans["map"], cfg.get("map"))
    spawn_state = plans["spawn_state"]
    obs_start = plans["obs_start"]
    gate_seed = int(plans["gate_seed"])
    pl = plans["lines"][:lines] if lines > 0 else plans["lines"]
    if not pl:
        raise SystemExit("the plan carries no lineage at all")
    fin_ticks = [c["finish_tick"] for c in pl if c["finish_tick"] > 0]
    nonfin = [c for c in pl if c["finish_tick"] <= 0]
    end_all = [max(int(c["finish_tick"]), int(c["end_tick"]),
                   len(c["acts"]) * K) for c in pl]
    ep_cap = max(int(cfg.get("ep_ticks", 12000)), int(max(end_all)) + K)
    core, gf, d0, _zones = open_planner_core(cfg, map_path, ep_cap)
    slot_probe, rf_probe, lf_probe = make_eval_feeds(cfg, gf, d0, K)
    n_latch = 0 if lf_probe is None else 1
    obs_reward = rf_probe is not None
    t_best = int(min(fin_ticks)) if fin_ticks else None
    arc_best = max(c["best_arc"] for c in nonfin) if nonfin else None

    # the arc scorer that verifies a non-finisher's replay, set up exactly
    # as the planner scored it (its route, corridor and window)
    pts = spacing = None
    corr = win = None
    if nonfin:
        rp = route or plans.get("route_file") \
            or str(Path(map_path).with_suffix(".route.npz"))
        if not Path(rp).exists():
            raise SystemExit(f"route file for the arc check not found: {rp}")
        pts, spacing = load_route(Path(rp))
        corr = float(corridor if corridor is not None
                     else (plans.get("arc_corridor") or 1500.0))
        win = int(arc_window if arc_window is not None
                  else (plans.get("arc_window") or 16))
    pd = phys_to_dict(core.config.phys)
    g_phys = -float(pd.get("sv_gravity", 800.0)) * float(pd.get("msec", 10)) \
        / 1000.0
    if plans.get("gravity_step") is not None \
            and abs(float(plans["gravity_step"]) - g_phys) > 1e-6:
        raise SystemExit(f"the plan banked arc with gravity step "
                         f"{plans['gravity_step']:g}, this core's is "
                         f"{g_phys:g} - different physics")
    bank = str(plans.get("arc_bank") or "contact")
    if contact_tol is None:
        contact_tol = float(plans.get("contact_tol") or 1.0)

    all_states, all_scal, all_latch, all_act, all_w, all_id = \
        [], [], [], [], [], []
    best_states = None
    best_line = None
    kept, dropped = [], []
    for j, c in enumerate(pl):
        acts = c["acts"]
        is_fin = int(c["finish_tick"]) > 0
        core.reset(gate_seed)             # fresh episode clocks; state next
        _slot, rf, lf = make_eval_feeds(cfg, gf, d0, K)   # fresh per line
        rows, tick_states, finished, ticks = replay_line(
            core, spawn_state, obs_start, acts, K, rf, lf,
            max_ticks=ep_cap, keep_final=not is_fin)
        states_arr = np.array(tick_states, dtype=spawn_state.dtype)
        info = {"finished": bool(is_fin), "arc": float(c["best_arc"]),
                "arc_tick": int(c["arc_tick"]), "end_tick": int(c["end_tick"]),
                "rows_all": int(len(rows))}
        if is_fin:
            want = int(c["finish_tick"])
            if not finished:
                dropped.append((j, want, ticks, "no finish on replay"))
                if j == 0:
                    raise SystemExit("the planner's WINNER did not reproduce "
                                     "its finish on replay (ended at tick "
                                     f"{ticks}, search said {want}) - "
                                     "determinism broken")
                continue
            if ticks != want:
                dropped.append((j, want, ticks, "finish tick differs"))
                if j == 0:
                    raise SystemExit(f"winner replay finished at {ticks}, "
                                     f"search said {want}")
                continue
            w = float(np.exp(-line_weight_decay * (ticks - t_best)
                             / max(t_best, 1)))
            cut = len(states_arr) - 1
            info.update(ticks=int(ticks), cut=int(cut), trim_ticks=0,
                        arc_replay=None, gravity_step=None,
                        gravity_source=None, weight=w)
        else:
            if finished:
                dropped.append((j, 0, ticks, "finished on replay although "
                                              "the search said it did not"))
                continue
            cut, g, gsrc = trim_last_contact(states_arr, g_phys, contact_tol)
            # the arc this replay BANKED: over the states up to the last
            # contact under bank 'contact' (what the search credited), over
            # every state under 'raw'. Must equal the search's number.
            xyz = np.asarray(states_arr["origin"], np.float32)
            arc_full, _off = corridor_progress_ordered(xyz, pts, spacing,
                                                       corr, win)
            arc_trim, _off = corridor_progress_ordered(xyz[:cut + 1], pts,
                                                       spacing, corr, win)
            arc_rep = arc_trim if bank == "contact" else arc_full
            info["arc_replay"] = float(arc_rep)
            info["arc_replay_raw"] = float(arc_full)
            if abs(float(arc_rep) - float(c["best_arc"])) > ARC_TOL:
                dropped.append((j, int(c["arc_tick"]), ticks,
                                f"arc differs: search {c['best_arc']:.1f}u, "
                                f"replay {arc_rep:.1f}u (bank {bank}, cut "
                                f"{cut} of {len(states_arr) - 1})"))
                continue
            if no_trim:
                cut, g, gsrc = len(states_arr) - 1, 0.0, "none"
            # decision jd was taken at tick jd*K: keep those on the track
            rows = [r for jd, r in enumerate(rows) if jd * K <= cut]
            w = float(np.exp(-line_weight_decay
                             * (arc_best - float(c["best_arc"]))
                             / max(arc_best, 1.0)))
            info.update(ticks=int(ticks), cut=int(cut),
                        trim_ticks=int(len(states_arr) - 1 - cut),
                        gravity_step=float(g), gravity_source=gsrc,
                        weight=w)
        info["rows_kept"] = int(len(rows))
        kept.append((j, info, c["src"]))
        if best_states is None:
            best_states = np.ascontiguousarray(states_arr[:cut + 1]).copy()
            best_line = info
        for st, scal, latch, act in rows:
            all_states.append(st)
            all_scal.append(scal)
            all_latch.append(latch)
            all_act.append(act)
            all_w.append(w)
            all_id.append(j)
    if not kept or not all_states:
        raise SystemExit("no planner line reproduced (finish or arc) with at "
                         "least one decision on the track")
    meta = {"plan": plans["files"], "ckpt": str(ckpt), "map": map_path,
            "act_every": K, "obs_reward": bool(obs_reward),
            "n_latch": int(n_latch),
            "d_latch": (0.0 if lf_probe is None else float(lf_probe.d_latch)),
            "d0": d0, "objective": plans.get("objective", "finish"),
            "lines": len(kept),
            "line_ticks": [k[1]["ticks"] for k in kept],
            "line_rows": [k[1]["rows_kept"] for k in kept],
            "line_rows_all": [k[1]["rows_all"] for k in kept],
            "line_weights": [k[1]["weight"] for k in kept],
            "line_finished": [k[1]["finished"] for k in kept],
            "line_arc": [k[1]["arc"] for k in kept],
            "line_arc_replay": [k[1]["arc_replay"] for k in kept],
            "line_arc_replay_raw": [k[1].get("arc_replay_raw") for k in kept],
            "arc_bank": bank,
            "line_arc_tick": [k[1]["arc_tick"] for k in kept],
            "line_cut": [k[1]["cut"] for k in kept],
            "line_trim_ticks": [k[1]["trim_ticks"] for k in kept],
            "line_gravity_source": [k[1]["gravity_source"] for k in kept],
            "finishers": int(sum(1 for k in kept if k[1]["finished"])),
            "best_ticks": t_best,
            "best_s": (t_best / 100.0 if t_best is not None else None),
            "best_arc": (max(k[1]["arc"] for k in kept
                             if not k[1]["finished"])
                         if any(not k[1]["finished"] for k in kept) else None),
            "arc_total": plans.get("arc_total"),
            "gravity_step_physics": g_phys, "contact_tol": float(contact_tol),
            "greedy_ticks": int(plans["greedy_ticks"]),
            "line_src": [k[2] for k in kept],
            "dropped": [list(map(str, d)) for d in dropped],
            "gate_seed": gate_seed, "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_bc_dataset(out, np.array(all_states, dtype=spawn_state.dtype),
                    np.array(all_scal, np.float32), np.array(all_latch, np.float32),
                    np.array(all_act, np.int64), np.array(all_w, np.float32),
                    np.array(all_id, np.int32), meta)
    n_rows = len(all_states)
    lead = (f"best {t_best / 100:.2f}s" if t_best is not None
            else f"best arc {meta['best_arc']:,.0f}u")
    print(f"bc: {n_rows:,} rows from {len(kept)} line(s) "
          f"({lead}, greedy {plans['greedy_ticks'] / 100:.2f}s"
          f"; {len(dropped)} dropped) -> {out}")
    for j, want, got, why in dropped:
        print(f"  dropped line {j}: {why} (search {want}, replay {got})")
    if spine:
        sel = subsample_by_path(best_states, float(spine_spacing))
        spine_states = np.ascontiguousarray(best_states[sel]).copy()
        np.save(spine, spine_states)
        o = np.asarray(best_states["origin"], np.float64)
        path_len = float(np.linalg.norm(np.diff(o, axis=0), axis=1).sum()) \
            if len(o) > 1 else 0.0
        print(f"spine: {len(spine_states):,} states of the best line's "
              f"{len(best_states):,} ticks"
              + (f" (trimmed {best_line['trim_ticks']} ticks of fall at "
                 f"tick {best_line['cut']}, gravity step "
                 f"{best_line['gravity_step']:g} from "
                 f"{best_line['gravity_source']})"
                 if not best_line["finished"] else " (finisher: no trim)")
              + (f", one per {spine_spacing:g}u" if spine_spacing > 0 else "")
              + f" -> {spine}")
        meta["spine"] = str(spine)
        meta["spine_len"] = int(len(spine_states))
        meta["spine_ticks"] = int(len(best_states))
        meta["spine_spacing"] = float(spine_spacing)
        meta["spine_path_len"] = path_len
        meta["spine_trim_ticks"] = int(best_line["trim_ticks"])
        meta["spine_finished"] = bool(best_line["finished"])
    meta["rows"] = int(n_rows)
    if summary_out:
        Path(summary_out).write_text(json.dumps(meta, indent=2),
                                     encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="planner line -> BC rows + spine")
    ap.add_argument("--plan", required=True, nargs="+",
                    help="beam_tas beam_best.npz file(s); several waves "
                         "searched from the same spawn pool their kept "
                         "lineages (distinct, finishers fastest first, then "
                         "best arc first)")
    ap.add_argument("--ckpt", required=True,
                    help="the checkpoint the plan was searched with (its "
                         "config picks the side-channel columns)")
    ap.add_argument("--out", required=True, help="BC dataset .npz")
    ap.add_argument("--spine", default=None, help="demo spine .npy")
    ap.add_argument("--map", default=None,
                    help="absolute .bsp (default: the plan's own path)")
    ap.add_argument("--lines", type=int, default=0,
                    help="use at most this many kept lineages (0 = all)")
    ap.add_argument("--line-weight-decay", type=float, default=0.0,
                    help="row weight exp(-decay*(t-t_best)/t_best) per "
                         "finishing line; exp(-decay*(arc_best-arc)/"
                         "arc_best) per non-finishing one")
    ap.add_argument("--route", default=None,
                    help="route .npz for the non-finisher arc check "
                         "(default: the plan's own, else <map>.route.npz)")
    ap.add_argument("--corridor", type=float, default=None,
                    help="arc corridor (default: the plan's)")
    ap.add_argument("--arc-window", type=int, default=None,
                    help="arc local window (default: the plan's)")
    ap.add_argument("--contact-tol", type=float, default=None,
                    help="last-contact trim: how far diff(vz) may sit from "
                         "the gravity step and still count as free fall "
                         "(default: the plan's own --contact-tol, else 1.0)")
    ap.add_argument("--no-trim", action="store_true",
                    help="keep the fall of a non-finishing line (measured "
                         "on this map to teach falling; diagnostics only)")
    ap.add_argument("--spine-spacing", type=float, default=0.0,
                    help="spine states at (at least) this many map units of "
                         "travel apart, so uniform draws are uniform ALONG "
                         "the line; 0 = every tick (the finisher loop)")
    ap.add_argument("--summary-out", default=None)
    a = ap.parse_args()
    build(a.plan, a.ckpt, a.out, spine=a.spine, map_path=a.map,
          lines=a.lines, line_weight_decay=a.line_weight_decay,
          summary_out=a.summary_out, route=a.route, corridor=a.corridor,
          arc_window=a.arc_window, contact_tol=a.contact_tol,
          spine_spacing=a.spine_spacing, no_trim=a.no_trim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
