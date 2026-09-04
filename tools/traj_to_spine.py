#!/usr/bin/env python3
"""traj_to_spine.py - turn ONE recorded episode into a STATE_DTYPE spine.

``tools/loop_spine.py`` builds the xLOOP spawn distribution out of
``record_ckpt.py --dump-states``, because a trajectory row is lossy: it
carries origin / velocity / yaw / pitch and the action, but NOT
``ducked`` / ``induck`` / ``duck_time`` / ``fuser2`` / ``oldbuttons`` /
``basevelocity``, and ``onground`` is flattened to 0/1. A spawn pool
copies WHOLE states, so those fields decide the player's hull and its
duck bookkeeping at the moment the episode restarts - defaulting them is
guessing, and a wrong hull can start an episode inside geometry.

This tool recovers them exactly instead of guessing, from the recording
alone, by REPLAYING it through the same C core:

* every row carries the action's move bins (``fwd`` / ``side``) and its
  buttons (jump = 2, duck = 4) verbatim - four of the six action heads;
* the two view heads are INVERTIBLE. ``src/env.c`` applies
  ``yaw += surf_yaw_delta(state, yb)`` and ``pitch += PITCH_BINS[pb] *
  rate/10`` BEFORE the move, from the state the row already reports, so
  the bin is read back by matching the recorded per-tick view delta
  against the 15 (resp. 7) values the core could have produced from THIS
  state. At 3,000 u/s under ``--yaw-adaptive`` the candidate deltas are
  0.29 deg apart and the recorded yaw is rounded to 0.01, so the match is
  unambiguous (the clamp makes several bins coincide only where they
  produce the SAME state, which is all that matters here).

The replay is then free-running and its origins are compared against the
recording tick by tick: if they agree, the recovered states ARE the
episode's states and every hidden field is the core's own. The one input
the recorder rounds away is the spawn yaw (2 decimals of a value the
reset jittered), so ``--yaw-search`` re-runs the replay over that
0.01-degree interval and keeps the best.

    python tools/traj_to_spine.py --traj runs/.../traj_9661579264.jsonl \\
        --ckpt runs/.../ckpt.pt --pick fastest \\
        --out runs/research/xLOOP131/spine_r0.npy \\
        --summary-out runs/research/xLOOP131/pick.json

``--selftest`` exercises the parsing, the ranking and the bin inversion
on a synthetic recording made by the same core - no run artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch

FINISH_PAD = 64.0          # eval_honesty's finish tolerance


# ---------------------------------------------------------------------------
# the recording
# ---------------------------------------------------------------------------

def load_episodes(path):
    """[(header, rows (T, 15) float64, footer)] from a record_rollout file."""
    eps, hdr, cur = [], None, None
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            if ln[0] == "{":
                d = json.loads(ln)
                if "episode" in d:
                    hdr, cur = d, []
                else:
                    eps.append((hdr, np.asarray(cur, np.float64), d))
                    cur = None
            elif cur is not None:
                cur.append(json.loads(ln))
    if cur:                                  # file cut off mid-episode
        eps.append((hdr, np.asarray(cur, np.float64), {"end": "trunc"}))
    return eps


def finished(rows, zones, pad: float = FINISH_PAD) -> bool:
    """Did this episode end in the finish box?

    The recorder logs the PRE-step state, and the core autoresets in place
    on the done step, so a finisher's LAST row is the tick before it
    crosses - up to one tick short of the plane. Both the last row and the
    swept segment into it are tested against the padded box.
    """
    lo = np.asarray(zones["end"]["mins"], np.float64) - pad
    hi = np.asarray(zones["end"]["maxs"], np.float64) + pad
    tail = rows[-2:, 1:4]
    if np.any(np.all((tail >= lo) & (tail <= hi), axis=1)):
        return True
    # one tick of the last velocity, in case the plane sits between rows
    step = rows[-1, 1:4] + rows[-1, 4:7] * 0.01
    return bool(np.all((step >= lo) & (step <= hi)))


# ---------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------

def _yaw_candidates(core, st):
    """The 15 yaw deltas the core could apply to state ``st`` this tick -
    ``src/env.c`` surf_yaw_delta, in Python."""
    from surfgym.core import YAW_BINS
    cfg = core.config
    lim = float(cfg.yaw_rate_max_deg)
    if not int(getattr(cfg, "yaw_adaptive", 0)):
        return np.asarray(YAW_BINS, np.float64) * (lim / 10.0)
    v = np.asarray(st["velocity"], np.float64)
    vh = max(1.0, float(np.hypot(v[0], v[1])))
    w = np.degrees(np.arctan(30.0 / vh))
    return np.clip(np.asarray(K_BINS, np.float64) * w, -lim, lim)


K_BINS = (-20.0, -8.0, -3.0, -1.5, -1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0,
          1.5, 3.0, 8.0, 20.0)
PITCH_BINS = (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0)
PITCH_LO, PITCH_HI = -70.0, 30.0


def _pitch_candidates(core, pitch):
    rate = float(core.config.pitch_rate_max_deg) / 10.0
    return np.clip(pitch + np.asarray(PITCH_BINS, np.float64) * rate,
                   PITCH_LO, PITCH_HI) - pitch


def replay(core, rows, start, phase: int = 0):
    """Replay ``rows`` from ``start`` (a STATE_DTYPE scalar) and return
    ``(states (T,), max origin deviation, ended)``.

    ``states[t]`` is the PRE-step snapshot of tick ``t`` - exactly the row
    ``record_ckpt.py --dump-states`` writes, so the output is drop-in for
    ``tools/loop_spine.py``.
    """
    n = len(rows)
    core.reset(0)
    core.set_tick_phase(int(phase))
    core.set_state(0, np.asarray(start).reshape(1)[0])
    out = np.zeros(n, dtype=core.get_states().dtype)
    act = np.zeros((core.num_envs, 6), np.int32)
    dev = 0.0
    ended = None
    for t in range(n):
        st = core.get_states()[0]
        out[t] = st
        dev = max(dev, float(np.abs(np.asarray(st["origin"], np.float64)
                                    - rows[t, 1:4]).max()))
        if t + 1 < n:
            dyaw = (rows[t + 1, 7] - rows[t, 7] + 540.0) % 360.0 - 180.0
            dpit = rows[t + 1, 12] - rows[t, 12]
        else:                       # last tick: the post-state is not logged
            dyaw = dpit = 0.0
        act[0, 0] = int(np.argmin(np.abs(_yaw_candidates(core, st) - dyaw)))
        act[0, 1] = int(np.argmin(np.abs(
            _pitch_candidates(core, float(st["pitch"])) - dpit)))
        act[0, 2] = int(rows[t, 13])
        act[0, 3] = int(rows[t, 14])
        b = int(rows[t, 8])
        act[0, 4] = 1 if b & 2 else 0
        act[0, 5] = 1 if b & 4 else 0
        _o, _r, done, trunc, _to = core.step(act)
        if (done[0] or trunc[0]) and ended is None:
            ended = ("done" if done[0] else "trunc", t)
    return out, dev, ended


def build_core(cfg, map_path, ep_cap):
    """The core record_ckpt.py would have built for this checkpoint."""
    import beam_tas
    from surfgym.tick import TickClock
    tick = TickClock(float(cfg.get("tick_ms") or 10.0))
    core = beam_tas.build_sim(cfg, map_path, 1, int(ep_cap), tick=tick)
    if cfg.get("teleport_fail") or cfg.get("reward") == "race":
        core.set_teleport_fail(True)
    return core


def start_state(core, rows, zones=None):
    """The episode's first state: the map spawn entity the recording
    started from (exact float32 origin, everything else as
    ``rewards.map_spawn_pool`` builds it) with the recorded yaw and the
    race pool's pitch."""
    from surfgym.rewards import map_spawn_pool
    pool = map_spawn_pool(core)
    o = rows[0, 1:4]
    k = int(np.argmin(np.abs(np.asarray(pool["origin"], np.float64)
                             - o).max(axis=1)))
    off = float(np.abs(np.asarray(pool["origin"][k], np.float64) - o).max())
    if off > 0.01:
        raise SystemExit(
            f"row 0 origin {o.tolist()} is {off:.3f}u from the nearest map "
            f"spawn - this recording did not start from the race pool; "
            f"re-record with --dump-states instead")
    st = pool[k:k + 1].copy()
    st["yaw"] = rows[0, 7]
    st["pitch"] = rows[0, 12]
    st["velocity"] = rows[0, 4:7]
    return st


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", help="record_ckpt .jsonl")
    ap.add_argument("--ckpt", help="the checkpoint it was recorded from "
                                   "(for the core's config)")
    ap.add_argument("--map", default="C:/RL_Surf/maps/surf_src_cannonball.bsp")
    ap.add_argument("--route",
                    default="C:/RL_Surf/maps/surf_src_cannonball.route.npz")
    ap.add_argument("--out", help="spine .npy")
    ap.add_argument("--summary-out", default=None, help="pick summary .json")
    ap.add_argument("--pick", choices=("fastest", "deepest", "index"),
                    default="fastest",
                    help="fastest = fewest ticks among FINISHERS (a "
                         "finisher's line is ranked by its time, which is "
                         "the metric); deepest = the xLOOP rule, minimum "
                         "geodesic d; index = --ep")
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--yaw-search", type=int, default=41,
                    help="candidate spawn yaws inside the recorder's 0.01 "
                         "deg rounding; the replay with the smallest origin "
                         "deviation wins. 1 = take the rounded value")
    ap.add_argument("--corridor", type=float, default=1500.0)
    ap.add_argument("--contact-tol", type=float, default=1.0)
    ap.add_argument("--max-dev", type=float, default=1.0,
                    help="fail if the replay leaves the recorded line by "
                         "more than this (map units)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    for req in ("traj", "ckpt", "out"):
        if not getattr(args, req):
            ap.error(f"--{req} is required (or --selftest)")

    from surfgym.zones import load_zones
    from eval_honesty import corridor_progress, load_route
    from pick_selfline import contact_cut
    from surfgym.tick import episode_seconds

    cfg = (torch.load(args.ckpt, map_location="cpu", weights_only=False)
           .get("config") or {})
    eps = load_episodes(args.traj)
    if not eps:
        raise SystemExit(f"no episodes in {args.traj}")
    core = build_core(cfg, args.map, max(len(r) for _, r, _ in eps) + 16)
    zones = load_zones(core.bsp_path)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    pts, spacing = load_route(Path(args.route))

    gf = None
    if cfg.get("race_dist") != "euclid":
        from surfgym.goalfield import build_goal_field
        gcell = float(cfg.get("goal_cell") or cfg.get("lidar_cell") or 32.0)
        gf = build_goal_field(core, zones["end"], cell=gcell)

    rowsum = []
    for i, (hdr, rows, foot) in enumerate(eps):
        corr, _off = corridor_progress(rows[:, 1:4].astype(np.float32),
                                       pts, spacing, args.corridor)
        d = (gf.sample(rows[:, 1:4]) if gf is not None else None)
        rowsum.append({
            "ep": i, "ticks": int(len(rows)),
            "seconds": round(episode_seconds(hdr, len(rows), args.traj), 3),
            "finished": bool(finished(rows, zones)),
            "corridor": float(corr),
            "min_d": (float(d.min()) if d is not None else float("nan")),
            "end_d": (float(d[-1]) if d is not None else float("nan")),
            "phase": int(hdr.get("tick_phase", 0)),
        })
    fins = [r for r in rowsum if r["finished"]]
    print(f"{len(eps)} episodes, {len(fins)} finished; "
          f"times {min(r['seconds'] for r in rowsum):.3f}"
          f"..{max(r['seconds'] for r in rowsum):.3f}s")
    for r in sorted(rowsum, key=lambda r: (not r["finished"], r["seconds"])):
        print(f"  ep{r['ep']:3d} {r['ticks']:6d}t {r['seconds']:8.3f}s  "
              f"corridor {r['corridor']:9,.0f}u  min_d {r['min_d']:9,.0f}u"
              f"{'  FINISH' if r['finished'] else ''}")

    if args.pick == "index":
        if args.ep is None:
            raise SystemExit("--pick index needs --ep")
        best = rowsum[int(args.ep)]
    elif args.pick == "deepest":
        best = min(rowsum, key=lambda r: r["min_d"])
    else:
        if not fins:
            raise SystemExit("--pick fastest: no episode finished this map")
        best = min(fins, key=lambda r: r["seconds"])
    hdr, rows, foot = eps[best["ep"]]
    print(f"chosen ep{best['ep']}: {best['seconds']:.3f}s, "
          f"{'FINISH' if best['finished'] else 'no finish'}")

    st0 = start_state(core, rows)
    yaw0 = float(st0["yaw"][0])
    cands = ([yaw0] if args.yaw_search <= 1 else
             list(np.linspace(yaw0 - 0.005, yaw0 + 0.005,
                              int(args.yaw_search))))
    bestrun = None
    for y in cands:
        st = st0.copy()
        st["yaw"] = y
        states, dev, ended = replay(core, rows, st, best["phase"])
        if bestrun is None or dev < bestrun[1]:
            bestrun = (states, dev, ended, y)
        if dev < 1e-3:
            break
    states, dev, ended, ybest = bestrun
    print(f"replay: {len(states)} ticks, spawn yaw {ybest:.6f} "
          f"(recorded {yaw0:.2f}), max |origin - recorded| = {dev:.4f}u, "
          f"ended {ended}")
    if dev > args.max_dev:
        raise SystemExit(
            f"replay deviated {dev:.3f}u > --max-dev {args.max_dev:g}: the "
            f"recovered actions do not reproduce this episode. Re-record "
            f"with record_ckpt.py --dump-states instead of reconstructing.")

    # the trim. A FINISHER is not trimmed (loop_spine's rule); contact_cut
    # is still REPORTED, so a run that would have been chopped says so.
    packed = np.column_stack([
        np.arange(len(states), dtype=np.float64),
        np.asarray(states["origin"], np.float64),
        np.asarray(states["velocity"], np.float64),
        np.asarray(states["yaw"], np.float64)])
    cut_c, g = contact_cut(packed, args.contact_tol)
    if best["finished"]:
        cut, why = len(states) - 1, "finisher: no trim"
    else:
        cut = cut_c
        why = (f"trimmed at tick {cut} of {len(states) - 1} "
               f"(gravity step {g:g} u/tick^2)")
    spine = np.ascontiguousarray(states[:cut + 1]).copy()
    d_sp = (gf.sample(np.asarray(spine["origin"], np.float64))
            if gf is not None else None)
    secs = episode_seconds(hdr, len(spine), args.traj)
    print(f"spine {len(spine)} states = {secs:.3f}s at "
          f"{hdr.get('tick_ms')} ms ({why})")
    if d_sp is not None:
        print(f"  d {d_sp[0]:,.0f} -> {d_sp[-1]:,.0f}u "
              f"(min {d_sp.min():,.0f}u), corridor MAX "
              f"{best['corridor']:,.0f}u")
    print(f"  contact_cut would cut at {cut_c} of {len(states) - 1} "
          f"({len(states) - 1 - cut_c} ticks)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, spine)
    print(f"spine -> {args.out}")

    summary = {
        "source": "traj_to_spine", "traj": str(args.traj),
        "ckpt": str(args.ckpt), "map": str(args.map),
        "episodes": len(eps), "chosen_ep": best["ep"], "pick": args.pick,
        "chosen_seconds": best["seconds"], "chosen_ticks": best["ticks"],
        "chosen_min_d": best["min_d"], "chosen_corridor": best["corridor"],
        "chosen_finished": best["finished"],
        "any_finished": sum(r["finished"] for r in rowsum),
        "best_corridor_any": max(r["corridor"] for r in rowsum),
        "spine_len": int(len(spine)),
        "spine_seconds": round(secs, 3),
        "spine_end_d": (float(d_sp[-1]) if d_sp is not None else None),
        "spine_min_d": (float(d_sp.min()) if d_sp is not None else None),
        "trim_ticks_dropped": int(len(states) - 1 - cut),
        "contact_cut_tick": int(cut_c),
        "contact_cut_would_drop": int(len(states) - 1 - cut_c),
        "gravity_step": float(g),
        "replay_max_dev_u": round(dev, 6),
        "replay_spawn_yaw": ybest,
        "replay_end": (list(ended) if ended else None),
        "tick_ms": hdr.get("tick_ms"),
        "tick_pattern_ms": hdr.get("tick_pattern_ms"),
        "tick_phase": best["phase"],
        "episode_table": rowsum,
        "spine": str(args.out),
    }
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
        print(f"summary -> {args.summary_out}")
    return 0


def selftest():
    """Parse / rank / invert, on a recording this file makes itself."""
    import tempfile
    from surfgym import SurfCore, default_config
    from surfgym.record import record_rollout

    m = ROOT / "maps" / "surf_src_cannonball.bsp"
    if not m.exists():
        m = next((ROOT / "maps").glob("*.bsp"), None)
        if m is None:
            print("selftest: no map available, skipping the replay half")
            return 0
    core = SurfCore(str(m), default_config(num_envs=1, spawn_mode=0,
                                           max_episode_ticks=200,
                                           lidar_w=0, lidar_h=0,
                                           yaw_adaptive=1),
                    tick_ms=7.63)

    rng = np.random.default_rng(7)

    class RandPolicy:
        def act(self, obs):
            a = np.zeros((1, 6), np.int32)
            a[0, 0] = rng.integers(0, 15)
            a[0, 1] = rng.integers(0, 7)
            a[0, 2] = rng.integers(0, 3)
            a[0, 3] = rng.integers(0, 3)
            a[0, 4] = rng.integers(0, 2)
            a[0, 5] = rng.integers(0, 2)
            return a

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.jsonl"
        record_rollout(core, RandPolicy(), p, episodes=1, max_ticks=180,
                       seed=3)
        eps = load_episodes(p)
        assert len(eps) == 1, len(eps)
        hdr, rows, foot = eps[0]
        assert rows.shape[1] == 15, rows.shape
        assert hdr["tick_pattern_ms"] == [8, 8, 7], hdr
        # replay the SAME core from the SAME state: the states must match
        core2 = SurfCore(str(m), default_config(num_envs=1, spawn_mode=2,
                                                max_episode_ticks=200,
                                                lidar_w=0, lidar_h=0,
                                                yaw_adaptive=1),
                         tick_ms=7.63)
        st = np.zeros(1, dtype=core2.get_states().dtype)
        st["origin"] = rows[0, 1:4]
        st["velocity"] = rows[0, 4:7]
        st["yaw"] = rows[0, 7]
        st["pitch"] = rows[0, 12]
        st["onground"] = -1
        core2.set_spawn_pool(st)
        states, dev, ended = replay(core2, rows, st,
                                    int(hdr.get("tick_phase", 0)))
        assert len(states) == len(rows), (len(states), len(rows))
        assert dev < 0.02, f"replay deviated {dev}u"
        # the recovered hidden fields are the core's own, not defaults
        assert states.dtype.names is not None and "ducked" in states.dtype.names
        # the ranking helpers
        zones = {"end": {"mins": [rows[-1, 1] - 1, rows[-1, 2] - 1,
                                  rows[-1, 3] - 1],
                         "maxs": [rows[-1, 1] + 1, rows[-1, 2] + 1,
                                  rows[-1, 3] + 1]}}
        assert finished(rows, zones, pad=0.0)
        far = {"end": {"mins": [1e6, 1e6, 1e6], "maxs": [1e6 + 1] * 3}}
        assert not finished(rows, far, pad=0.0)
    print(f"selftest OK: {len(rows)} ticks replayed, max dev {dev:.5f}u")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
