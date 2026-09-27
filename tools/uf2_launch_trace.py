"""uf2_launch_trace.py - WHAT does the mover do at unitfarmer2's south wall, tick by tick, next to
what the record does there? A MEASUREMENT (CLAUDE.md section 0: the record says which skill is
missing; it never trains anything).

    python tools/uf2_launch_trace.py <mover ckpt> [--t 6.0] [--secs 2.0] [--every 10] [--n 8]

The mover is spawned from the record's own state at --t (the entry bench's window,
runs/research/gate_bench/uf2_pitwindow.npy) and flies the record's own line from there (the
landing bench's `record` line). Printed every --every ticks for flight 0 (greedy) and the median
of --n sampled flights: position, |v|, horizontal speed, vz, view yaw, ground contact, held keys;
next to the record's row at the same tick (its yaw and its HL usercmd buttons).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

WIN = ROOT / "runs" / "research" / "gate_bench" / "uf2_pitwindow.npy"
REC = ROOT / "runs" / "research" / "uf2_wr" / "surf_unitfarmer2.jsonl"
T0, T1 = 2.5, 9.0


def _keys(b: int) -> str:
    return (("F" if b & 8 else "-") + ("L" if b & 512 else "-") + ("R" if b & 1024 else "-")
            + ("J" if b & 2 else "-") + ("D" if b & 4 else "-"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--t", type=float, default=6.0)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=8, help="flights (the scratch core's env count)")
    ap.add_argument("--quiet", action="store_true", help="the summary only, no per-tick table")
    ap.add_argument("--anchors", action="store_true",
                    help="also log, per tick, the vertex the OBSERVATION's fan anchors to (the "
                         "line's global nearest-vertex argmin, goals.MultiLine._anchor) next to a "
                         "LOCAL monotone anchor (nearest vertex in a 16-vertex forward window from "
                         "the previous one, the reward's ArcProgress rule): a fan that jumps "
                         "ahead of the local anchor is reading another branch of a folded line")
    ap.add_argument("--out", default=None, help="write the per-tick rows (JSON lines) here")
    a = ap.parse_args(argv)
    import torch
    import record_ckpt
    import edge_archive as ea
    from surfgym.route import resample_polyline
    from surfgym.goalprimplan import L_MAX
    win = np.load(WIN)
    j = int(round((a.t - T0) / (T1 - T0) * (len(win) - 1)))
    st = win[j].copy()
    st["tick"] = 0
    st["stuck_ticks"] = 0
    rec = []
    for ln in open(REC, encoding="utf-8"):
        if ln.startswith("["):
            rec.append(json.loads(ln)[:9])
        elif ln.startswith('{"end'):
            break
    rec = np.asarray(rec, np.float64)
    k0 = int(np.argmin(np.linalg.norm(rec[:, 1:4] - st["origin"].astype(np.float64), axis=1)))
    torch.manual_seed(int(a.seed))
    ctx = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--plan-scratch", str(int(a.n))]
                            + ([] if a.greedy else ["--stochastic"])
                            + ["--map", str(ROOT / "maps_pool" / "surf_unitfarmer2.bsp")])
    fin = np.zeros(3)
    rline, _ = resample_polyline(rec[k0:k0 + 260, 1:4], ea.RAY_SPACING)
    rline = np.asarray(rline, np.float32)

    class _Fixed:
        n_choice = 1
        commit_ticks = budget_ticks = 200
        choice_nums = np.zeros((1, 1))
        finish = fin

        def line_and_curve_of(self, origin, velocity, yaw_deg, nums, k=None):
            return rline, rline

    ctx.planner = _Fixed()
    core = ctx.scratch.core
    for i in range(core.num_envs):
        core.set_state(i, st)
    neutral = np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (core.num_envs, 1))
    obs, *_ = core.step(neutral)
    st1 = core.get_states()[0].copy()
    st1["tick"] = 0
    st1["stuck_ticks"] = 0
    obs0 = np.array(obs[0], np.float32, copy=True)
    fl = ea.Flyer(ctx)
    S, K = fl.S, fl.K
    for i in range(S):
        core.set_state(i, st1)
    fl.sc.line.set_lines(np.arange(S), [rline[:L_MAX]] * S)
    pol = fl.sc.make_policy(core, fl.sc.line)
    if fl.keys_hold:
        from surfgym.keyshold import KeysHold
        pol.keys = KeysHold(S)
        pol._keys_tick = np.zeros(S, np.int64) - int(getattr(pol, "_period", K))
    pol._tick = 0
    obs = np.ascontiguousarray(np.stack([obs0] * S).astype(np.float32))
    alive = np.ones(S, bool)
    died_at = np.full(S, -1)
    rows = []
    n_t = int(round(a.secs * 100))
    pos = np.full((n_t, S, 3), np.nan)          # every flight's position per tick (NaN once ended)
    loc = np.zeros(S, np.int64)                   # --anchors: the local monotone anchor per flight
    jumps = []                                    # (tick, flight, global - local) when > 2 vertices
    print(f"uf2_launch_trace: {Path(a.ckpt).name}, {'greedy' if a.greedy else 'sampled'}, {S} "
          f"flights on the record's line from the record's state at t {rec[k0, 0] / 100:.2f} s; "
          f"tick {float(ctx.tick.ms):.2f} ms, decisions every {K} ticks")
    print("   t(s) | MOVER flight 0: pos (x, y, z)   |v|   vh    vz   yaw  gnd keys | alive "
          "| RECORD: pos (x, y, z)   |v|   vh    vz   yaw  keys")
    for t in range(n_t):
        acts = pol.act(obs)
        view = getattr(pol, "view", None)
        sv = core.get_states()
        o, v = sv["origin"].astype(np.float64), sv["velocity"].astype(np.float64)
        a0 = np.asarray(acts)[0].tolist()
        pos[t, alive] = o[alive]
        g_i = None
        if a.anchors:
            import torch as _t
            ln = fl.sc.line
            g_i = ln._anchor(_t.as_tensor(o, dtype=_t.float32, device=ln.pts.device))[0]
            g_i = g_i.cpu().numpy().astype(np.int64)
            for i in np.flatnonzero(alive):
                w = rline[loc[i]:loc[i] + 17]
                loc[i] = loc[i] + int(np.argmin(np.linalg.norm(w - o[i][None], axis=1)))
                if g_i[i] - loc[i] > 2:
                    jumps.append((t, int(i), int(g_i[i] - loc[i])))
        rows.append({"t": t, "origin": o[0].round(1).tolist(), "velocity": v[0].round(1).tolist(),
                     "yaw": float(sv["yaw"][0]), "onground": int(sv["onground"][0]), "acts": a0,
                     "alive": int(alive.sum()),
                     "vh_med": float(np.median(np.hypot(v[alive, 0], v[alive, 1])))
                     if alive.any() else 0.0,
                     "z_med": float(np.median(o[alive, 2])) if alive.any() else 0.0})
        if t % a.every == 0 and a.anchors and g_i is not None:
            print(f"   {t / 100:4.2f} anchors flight 0: fan (global argmin) vertex {int(g_i[0]):3d} "
                  f"| local monotone vertex {int(loc[0]):3d} | line has {len(rline)} vertices")
        if t % a.every == 0 and not a.quiet:
            r = rec[min(k0 + t, len(rec) - 1)]
            vh0 = float(np.hypot(v[0, 0], v[0, 1]))
            print(f"   {t / 100:4.2f} | ({o[0, 0]:6.0f},{o[0, 1]:6.0f},{o[0, 2]:6.0f}) "
                  f"{np.linalg.norm(v[0]):5.0f} {vh0:5.0f} {v[0, 2]:5.0f} {float(sv['yaw'][0]) % 360:5.0f} "
                  f"{int(sv['onground'][0]):3d} {a0} | {int(alive.sum()):2d}/{S} med vh "
                  f"{rows[-1]['vh_med']:5.0f} z {rows[-1]['z_med']:5.0f} | ({r[1]:6.0f},{r[2]:6.0f},"
                  f"{r[3]:6.0f}) {np.linalg.norm(r[4:7]):5.0f} {np.hypot(r[4], r[5]):5.0f} {r[6]:5.0f} "
                  f"{r[7] % 360:5.0f} {_keys(int(r[8]))}")
        obs, _r, done, trunc, _term = (core.step(acts) if view is None
                                       else core.step(acts, view=view))
        ended = alive & (np.asarray(done, bool) | np.asarray(trunc, bool))
        died_at[ended] = t + 1
        alive &= ~ended
        if not alive.any():
            print(f"   all flights ended by t {(t + 1) / 100:.2f} s")
            break
    print(f"   deaths: {int((died_at >= 0).sum())}/{S}; death ticks {sorted(died_at[died_at >= 0].tolist())[:12]}")
    if a.anchors:
        fl_j = sorted({f for _t0, f, _d in jumps})
        first = {}
        for t0, f, d in jumps:
            first.setdefault(f, (t0, d))
        print(f"   ANCHORS: the fan's global anchor ran > 2 vertices ahead of the local monotone "
              f"anchor in {len(fl_j)}/{S} flights; first jump per flight (tick, vertices ahead): "
              + ", ".join(f"{f}:{first[f]}" for f in fl_j[:12])
              + (f"; largest lead {max(d for _t0, _f, d in jumps)} vertices" if jumps else ""))
    # the summary over all flights: tracking error = distance to the record's own path (the line
    # it was shown), the highest point reached (the record: z 229 at 1 s, 451 at 1.6 s, 573 at 2 s)
    path = rec[k0:k0 + 260, 1:4]
    seg_a, seg_b = path[:-1], path[1:]

    def _dist(q):
        ab = seg_b - seg_a
        tt = np.clip(np.einsum("ij,ij->i", q[None] - seg_a, ab) / np.maximum(
            np.einsum("ij,ij->i", ab, ab), 1e-9), 0.0, 1.0)
        return float(np.min(np.linalg.norm(seg_a + ab * tt[:, None] - q[None], axis=1)))
    errs = []
    for ts in (0.2, 0.4, 0.6, 0.8, 1.0):
        k = int(round(ts * 100))
        if k >= n_t:
            continue
        d = [_dist(pos[k, i]) for i in range(S) if np.isfinite(pos[k, i, 0])]
        errs.append(f"{ts:.1f} s {np.median(d):,.0f} u ({len(d)} alive)" if d else f"{ts:.1f} s -")
    zmax = np.nanmax(pos[:, :, 2], axis=0)
    alive1 = int(np.isfinite(pos[min(99, n_t - 1), :, 0]).sum())
    print(f"   SUMMARY {Path(a.ckpt).name} {'greedy' if a.greedy else 'sampled'} n={S}: alive at 1 s "
          f"{alive1}, at the end {int(alive.sum())} | tracking error to the record's path, median: "
          + "; ".join(errs) + f" | max z per flight: median {np.median(zmax):,.0f}, >= 300: "
          f"{int((zmax >= 300).sum())}, >= 450: {int((zmax >= 450).sum())} of {S}")
    if a.out:
        Path(a.out).write_text(chr(10).join(json.dumps(r) for r in rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
