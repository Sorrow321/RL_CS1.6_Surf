"""Build a demo spine from the beam TAS record run.

Replays runs/beam_tas/beam_best.npz open-loop on a fresh 1-env physics
core (the bit-exact replay path of tools/beam_tas.py - same sim config,
same set_state entry), captures the FULL STATE_DTYPE state at the start
of every tick, asserts the finish reproduces at the recorded tick, then
clips the last --clip-frac of TICKS so the goal-adjacent segment is
absent and saves the rest as a time-ordered STATE_DTYPE .npy - exactly
what train_fast's --demo-file expects (surfgym.respawn.DemoCurriculum:
"time-ordered STATE_DTYPE .npy demo spine", index 0 earliest). Episode
fields carried in the rows (tick, progress, ...) are re-zeroed core-side
on every spawn-pool reset, so they do not leak into training episodes.

Reports the geodesic d at the cut point and a decile profile of d along
the spine (the "spawns distributed along the line" evidence).

    python tools/build_spine.py
    python tools/build_spine.py --clip-frac 0.10 --out runs/beam_tas/spine90.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch

from surfgym.core import STATE_DTYPE
from surfgym.tick import ticks_to_secs
import beam_tas


def main():
    ap = argparse.ArgumentParser(description="demo spine from beam_best.npz")
    ap.add_argument("--npz", default=str(ROOT / "runs" / "beam_tas" /
                                         "beam_best.npz"))
    ap.add_argument("--clip-frac", type=float, default=0.10,
                    help="fraction of TICKS clipped off the goal end")
    ap.add_argument("--out", default=str(ROOT / "runs" / "beam_tas" /
                                         "spine90.npy"))
    args = ap.parse_args()

    z = np.load(args.npz)
    acts = z["acts"].astype(np.int32)
    K = int(z["act_every"])
    fin = int(z["finish_ticks"])
    spawn = np.asarray(z["spawn_state"], STATE_DTYPE)
    map_path = str(z["map"])
    ckpt = str(z["ckpt"])
    # the plan's own time base (beam_tas.tick_npz): tick_ms + the integer
    # pattern under --tick-ms. A plan searched before the flag has neither
    # and ran at 10 ms; "fin / 100" was that assumption spelled out.
    plan_tick = float(z["tick_ms"]) if "tick_ms" in z.files else 10.0
    plan_pat = ([int(v) for v in np.asarray(z["tick_pattern_ms"]).ravel()]
                if "tick_pattern_ms" in z.files else None)
    print(f"record: {fin} ticks "
          f"({ticks_to_secs(fin, plan_tick, plan_pat):.2f}s at "
          f"{plan_tick:g} ms), {len(acts)} decisions "
          f"x act_every {K}, map {map_path}")

    cfg = (torch.load(ckpt, map_location="cpu", weights_only=False)
           .get("config") or {})
    ep_cap = max(int(cfg.get("ep_ticks", 12000)), fin + 1)
    core = beam_tas.build_sim(cfg, map_path, 1, ep_cap)
    core_tick = float(getattr(core, "tick_ms", 10.0))
    if abs(core_tick - plan_tick) > 1e-6:
        raise SystemExit(
            f"{args.npz} was searched at {plan_tick:g} ms but this replay "
            f"core runs {core_tick:g} ms - the open-loop replay would meet "
            f"different physics tick by tick (the finish assert below is "
            f"the symptom). Build the core at the plan's tick "
            f"(beam_tas.build_sim(..., tick=TickClock(tick_ms))).")
    from surfgym.zones import load_zones
    zones = load_zones(core.bsp_path)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    if cfg.get("teleport_fail") or cfg.get("reward") == "race":
        core.set_teleport_fail(True)
    core.set_spawn_pool(spawn)          # the post-finish autoreset draws
    core.set_state(0, spawn)

    acts_ticks = np.repeat(acts, K, axis=0)
    states = np.zeros(fin, STATE_DTYPE)
    hit = False
    for t in range(fin):
        states[t] = core.get_states()[0]        # state at START of tick t
        _o, _r, done, _tr, _tm = core.step(
            np.ascontiguousarray(acts_ticks[t].reshape(1, 6)))
        if core.goal_hits[0]:
            hit = t + 1 == fin
            break
    assert hit, (f"replay did not reproduce the finish at tick {fin} "
                 "(goal hit early/absent) - stale npz or wrong sim config")
    print(f"replay: finish reproduced at tick {fin} - {fin} states captured")

    n_clip = int(round(args.clip_frac * fin))
    spine = states[:fin - n_clip].copy()
    print(f"clip: last {n_clip} ticks ({args.clip_frac:.0%}) dropped -> "
          f"{len(spine)} states kept (t 0..{len(spine) - 1})")

    from surfgym.goalfield import build_goal_field
    gcell = float(cfg.get("goal_cell") or cfg.get("lidar_cell") or 32.0)
    gf = build_goal_field(core, zones["end"], cell=gcell)
    d = gf.sample(spine["origin"]).astype(np.float64)
    sp = np.linalg.norm(spine["velocity"], axis=1)
    print(f"geodesic d: start {d[0]:,.0f}u  CUT POINT {d[-1]:,.0f}u  "
          f"(min along spine {d.min():,.0f}u at idx {int(d.argmin())})")
    print("d deciles along the spine (idx: d, speed):")
    for q in range(0, 11):
        i = min(len(spine) - 1, int(round(q / 10 * (len(spine) - 1))))
        print(f"  {q * 10:3d}%  idx {i:5d}  d {d[i]:9,.0f}u  "
              f"v {sp[i]:6.0f} u/s")

    np.save(args.out, spine)
    print(f"spine: {len(spine)} time-ordered STATE_DTYPE states -> "
          f"{args.out}")


if __name__ == "__main__":
    main()
