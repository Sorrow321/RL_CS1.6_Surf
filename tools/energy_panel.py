"""energy_panel.py - does the SEARCH'S sampled executor keep its momentum in the air? A generic,
map-independent bench for choosing the executor's view-noise scale (Codex 2026-09-27 20:57Z: "the
largest scale that passes a momentum-retention criterion over a policy-owned multi-map state
panel", pre-declared, never picked by which map's search wins). MEASUREMENT only.

    python tools/energy_panel.py <mover ckpt> --panel MAP:STATES.npy [MAP:STATES.npy ...]
        [--scales native,0.5,0.25,0.1,greedy] [--n 64] [--secs 2.0] [--min-speed 600]
        [--out rows.jsonl]

Per map: --n AIRBORNE states (no ground, |v| >= --min-speed) drawn once from the states file (a
search's own archive dump), the same states for every scale. Each is flown with the search's own
straight-ahead command (edge_archive.RayOperator ray 0, the level ray along the velocity) for up
to --secs; the AIR SEGMENT ends at the first contact (core telemetry), a death, or --secs. The
torch RNG is re-seeded per (map, scale), so every scale draws the SAME standard-normal noise
(z = mu + sigma * scale * eps): the scales are paired.

Reported per (map, scale): the specific-energy drift over the air segment as a fraction of the
start's KINETIC energy per second, dE / KE0 / s with E = 0.5 |v|^2 + g z (a ratio to E itself
depends on the world's z origin; Codex 20:57Z) - median and quartiles - and the air segment's
median length. A 'Y,P' scale scales yaw and pitch separately.

The pre-declared rule (2026-09-27): the executor's view scale for the search is the LARGEST scale
whose median drift is >= -1 % KE0 per second on EVERY map of the panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

G = 800.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--panel", nargs="+", required=True, help="MAP.bsp:STATES.npy pairs")
    ap.add_argument("--scales", default="native,0.5,0.25,0.1,greedy")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--min-speed", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch
    import record_ckpt
    import edge_archive as ea
    scales = [s.strip() for s in a.scales.split(",")]
    rows = []
    table = {}
    for pair in a.panel:
        mp, _, sp = pair.rpartition(":") if pair.count(":") > 1 else pair.partition(":")
        st_all = np.unique(np.load(sp))
        spd = np.linalg.norm(st_all["velocity"].astype(np.float64), axis=1)
        air = (st_all["onground"] < 0) & (spd >= float(a.min_speed))
        cand = st_all[air]
        rng = np.random.default_rng(int(a.seed))
        pick = cand[rng.choice(len(cand), size=min(int(a.n), len(cand)), replace=False)]
        n = len(pick)
        print(f"energy_panel: {Path(mp).stem}: {n} airborne states (|v| >= {a.min_speed:g}) of "
              f"{len(st_all):,} in {Path(sp).name}; |v| median "
              f"{np.median(np.linalg.norm(pick['velocity'].astype(np.float64), axis=1)):,.0f}",
              flush=True)
        for sc in scales:
            argv_b = [str(a.ckpt), "--episodes", "1", "--plan-scratch", str(n), "--ep-ticks",
                      str(int(round(a.secs * 100)) + 400), "--map", str(mp)]
            if sc != "greedy":
                argv_b += ["--stochastic"]
                if sc != "native":
                    argv_b += ["--exec-view-scale", sc.replace(":", ",")]   # 'Y:P' -> 'Y,P' 
            ctx = record_ckpt.build(argv_b)
            core = ctx.scratch.core
            fb = ctx.finish_box
            fin = 0.5 * (np.asarray(fb[0], np.float64) + np.asarray(fb[1], np.float64))
            op = ea.RayOperator(float(ctx.tick.ms), fin, n=3)
            ctx.planner = op
            fl = ea.Flyer(ctx)
            tick_s = float(ctx.tick.ms) / 1000.0
            for i in range(n):
                s2 = pick[i].copy()
                s2["tick"] = 0
                s2["stuck_ticks"] = 0
                core.set_state(i, s2)
            neutral = np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (n, 1))
            obs, _r, done0, trunc0, _ = core.step(neutral)
            obs = np.ascontiguousarray(np.asarray(obs, np.float32))
            cur = core.get_states()
            ok = ~(np.asarray(done0, bool) | np.asarray(trunc0, bool))
            lines = []
            for i in range(n):
                ln, _p = op.line_and_curve_of(cur[i]["origin"].astype(np.float64),
                                              cur[i]["velocity"].astype(np.float64),
                                              float(cur[i]["yaw"]), op.choice_nums[0], 0)
                lines.append(ln[:fl.L_MAX])
            ctx.scratch.line.set_lines(np.arange(n), lines)
            torch.manual_seed(int(a.seed))           # the same draws for every scale
            pol = ctx.scratch.make_policy(core, ctx.scratch.line)
            if fl.keys_hold:
                from surfgym.keyshold import KeysHold
                pol.keys = KeysHold(n)
                pol._keys_tick = cur["tick"].astype(np.int64) - int(getattr(pol, "_period", fl.K))
            pol._tick = 0
            v0 = cur["velocity"].astype(np.float64)
            e0 = 0.5 * np.sum(v0 ** 2, axis=1) + G * cur["origin"][:, 2].astype(np.float64)
            ke0 = 0.5 * np.sum(v0 ** 2, axis=1)
            seg_e = e0.copy()
            seg_t = np.zeros(n, np.int64)
            open_ = ok.copy()
            why = np.array(["secs"] * n, dtype=object)
            why[~ok] = "reset"
            use_touch = hasattr(core, "get_touch")
            for t in range(int(round(a.secs / tick_s))):
                acts = pol.act(obs)
                view = getattr(pol, "view", None)
                obs, _r, done, trunc, _ = (core.step(acts) if view is None
                                           else core.step(acts, view=view))
                obs = np.ascontiguousarray(np.asarray(obs, np.float32))
                ended = open_ & (np.asarray(done, bool) | np.asarray(trunc, bool))
                touched = np.zeros(n, bool)
                if use_touch:
                    cnt, _tn, _tp = core.get_touch()
                    touched = np.asarray(cnt, np.int64) > 0
                cs = core.states_view
                e = (0.5 * np.sum(cs["velocity"].astype(np.float64) ** 2, axis=1)
                     + G * cs["origin"][:, 2].astype(np.float64))
                # the air segment: every tick before the first contact / death counts
                stop = open_ & (ended | touched)
                live = open_ & ~stop
                seg_e[live] = e[live]
                seg_t[live] = t + 1
                why[stop & ended] = "death"
                why[stop & touched & ~ended] = "contact"
                open_ &= ~stop
                if not open_.any():
                    break
            dur = seg_t * tick_s
            rate = np.where(dur > 0, (seg_e - e0) / np.maximum(ke0, 1.0) / np.maximum(dur, 1e-9),
                            np.nan)
            good = ok & (seg_t >= 10)                 # at least 0.1 s of air
            r_ = rate[good]
            q = np.percentile(r_, [25, 50, 75]) if len(r_) else [np.nan] * 3
            table[(Path(mp).stem, sc)] = float(q[1])
            print(f"   scale {sc:>9s}: dE/KE0 per s median {100 * q[1]:+6.2f}% "
                  f"[{100 * q[0]:+.2f}, {100 * q[2]:+.2f}] over {int(good.sum())} air segments "
                  f"(median {np.median(dur[good]) if good.any() else 0:.2f} s; ended by contact "
                  f"{int((why == 'contact').sum())}, death {int((why == 'death').sum())})",
                  flush=True)
            for i in range(n):
                rows.append({"map": Path(mp).stem, "scale": sc, "state": i,
                             "speed0": float(np.sqrt(2 * ke0[i])), "ke0": float(ke0[i]),
                             "e0": float(e0[i]), "e_end": float(seg_e[i]),
                             "air_ticks": int(seg_t[i]), "end": str(why[i]),
                             "rate": (None if not np.isfinite(rate[i]) else float(rate[i]))})
    passing = []
    for sc in scales:
        if sc in ("native", "greedy"):
            continue
        vals = [table.get((Path(p.rpartition(":")[0] if p.count(":") > 1 else p.partition(":")[0]).stem, sc))
                for p in a.panel]
        if all(v is not None and np.isfinite(v) and v >= -0.01 for v in vals):
            passing.append(sc)
    nat = [table.get((Path(p.rpartition(":")[0] if p.count(":") > 1 else p.partition(":")[0]).stem,
                      "native")) for p in a.panel]
    if "native" in scales and all(v is not None and np.isfinite(v) and v >= -0.01 for v in nat):
        passing.append("native")
    print(f"energy_panel: scales passing the pre-declared rule (median >= -1 % KE0/s on every map): "
          f"{passing or 'none'}")
    if a.out:
        def md5(p):
            return hashlib.md5(Path(p).read_bytes()).hexdigest()
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"argv": sys.argv, "ckpt_md5": md5(a.ckpt),
                                "panel_md5": {p: md5(p.rpartition(":")[2] if p.count(":") > 1
                                                     else p.partition(":")[2])
                                              for p in a.panel}}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"   raw rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
