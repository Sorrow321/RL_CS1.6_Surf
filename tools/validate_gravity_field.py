#!/usr/bin/env python3
"""validate_gravity_field.py - is the goal field honest about gravity?

The plain geodesic field (goal_<cell>.npz) BFSes free space as an
UNDIRECTED graph, so a voxel counts as "close to the finish" even when the
only path there is a one-way fall. On surf_src_cannonball that paints an
off-route pit as the global minimum of the shaping potential (d ~ 21.5k)
while the actual winning line reads 31k -> 107k -> unreachable: the agent
is paid to dive into the pit and punished for racing.

goalfield.build_goal_field(..., gravity_dir=True) rebuilds the same field
with a directional graph (fall and air-strafe anywhere, climb only along
geometry). This script measures whether that fixed it, from evidence only:

  (a) the field along a champion finishing route, sampled every second,
      under BOTH fields side by side;
  (b) alignment - the fraction of those seconds where the field DECREASES
      (an honest race potential is near-monotone; the raw one is not);
  (c) the reachable MINIMUM along failure trajectories under both fields,
      as the share of the whole potential a FAILING episode banks. A field
      that pays a dead-end 98% of the race is the deception, in one number.

This script NEVER bakes: it takes already-baked .npz paths. Bake with a
GPU yourself (see the module docstring of surfgym/goalfield.py).

    python tools/validate_gravity_field.py \
        --base  maps/surf_src_cannonball.goal_32.npz \
        --field maps/surf_src_cannonball.goalg_32.npz \
        --champ scratch/champ_greedy.jsonl \
        --fail  "runs/xCTL/traj_*.jsonl"

    python tools/validate_gravity_field.py --selftest    # CPU, no bake

Trajectory format (tools/record_ckpt.py): one JSON dict header per episode,
then rows [tick, x, y, z, vx, vy, vz, yaw, ...], then a footer dict. A tick
reset also splits episodes for recorders that omit the header.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from surfgym.goalfield import GoalField  # noqa: E402


# --------------------------------------------------------------------- fields

class _QuantGrid:
    """uint16 grid that scales to map units on GATHER instead of up front.

    GoalField only ever does grid[iz, iy, ix], and cannonball's field is
    671M voxels: expanding to float32 costs 2.7 GB per field and we load
    two. This keeps both resident in the 1.3 GB they occupy on disk."""

    __slots__ = ("q", "quant", "shape")

    def __init__(self, q, quant):
        self.q = q
        self.quant = float(quant)
        self.shape = q.shape

    def __getitem__(self, key):
        return self.q[key].astype(np.float32) * self.quant


def load_field(path):
    """Load a baked goal-field npz as a GoalField (no map, no bake)."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"no such field: {p}")
    z = np.load(p, allow_pickle=False)
    for k in ("grid", "quant", "mins", "cell", "reach_max"):
        if k not in z:
            raise SystemExit(f"{p.name} is not a goal field (no '{k}')")
    gf = GoalField(_QuantGrid(z["grid"], z["quant"]), z["mins"],
                   float(z["cell"]), float(z["reach_max"]))
    gf.path = p
    gf.quant = float(z["quant"])
    gf.sig = str(z["sig"]) if "sig" in z else "(none)"
    return gf


def honest_voxels(gf, chunk=32):
    """Count voxels holding a real distance (not sentinel/solid/unreachable).

    Chunked over z so a 671M-voxel compare never materializes a full
    temporary."""
    q = gf.grid.q
    thr = (gf.reach_max - 0.5 * gf.cell) / gf.quant
    n = 0
    for z0 in range(0, q.shape[0], chunk):
        n += int(np.count_nonzero(q[z0:z0 + chunk] < thr))
    return n, int(q.size)


def field_banner(name, gf):
    hon, tot = honest_voxels(gf)
    print(f"  {name:5s} {gf.path.name:38s} cell {gf.cell:g}u  "
          f"grid {gf.grid.shape[2]}x{gf.grid.shape[1]}x{gf.grid.shape[0]}  "
          f"reach_max {gf.reach_max:10.0f}u  quant {gf.quant:g}u")
    print(f"        honest voxels {hon:,} / {tot:,} ({100.0 * hon / tot:.2f}%)"
          f"   sig {gf.sig}")
    return hon


# ---------------------------------------------------------------- trajectories

def read_episodes(path):
    """-> list of episodes {src, idx, hdr, footer, tick, pos, vel}."""
    eps = []

    def new(hdr):
        e = {"src": Path(path).name, "idx": len(eps), "hdr": hdr,
             "footer": None, "rows": []}
        eps.append(e)
        return e

    cur = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                if "map" in obj or "tick_ms" in obj:
                    cur = new(obj)
                elif cur is not None:
                    cur["footer"] = obj
                continue
            if cur is None:
                cur = new({})
            elif cur["rows"] and obj[0] == 0 and cur["rows"][-1][0] != 0:
                cur = new(cur["hdr"])       # tick reset = episode boundary
            cur["rows"].append(obj)

    out = []
    for e in eps:
        if not e["rows"]:
            continue
        r = np.asarray(e["rows"], np.float64)
        e["tick"] = r[:, 0].astype(np.int64)
        e["pos"] = r[:, 1:4]
        e["vel"] = r[:, 4:7] if r.shape[1] >= 7 else np.zeros_like(r[:, 1:4])
        e["tick_ms"] = float(e["hdr"].get("tick_ms", 10.0))
        del e["rows"]
        out.append(e)
    return out


def expand(patterns):
    """PowerShell does not glob arguments to native exes - do it here."""
    files = []
    for pat in patterns or []:
        hit = sorted(glob.glob(pat))
        if not hit:
            p = Path(pat)
            if p.exists():
                hit = [str(p)]
            else:
                print(f"  (no match: {pat})")
        files.extend(hit)
    return files


def per_second(ep, period=1.0):
    """Row indices ~`period` seconds apart, always including the last."""
    stride = max(1, int(round(period * 1000.0 / ep["tick_ms"])))
    idx = list(range(0, len(ep["tick"]), stride))
    if idx[-1] != len(ep["tick"]) - 1:
        idx.append(len(ep["tick"]) - 1)
    return np.asarray(idx, np.int64)


def finished(ep, end_zone, grow=64.0):
    """True if the episode's last sample lands in the (inflated) end zone.

    The recorder's footer says "fail" for finishers whose goal box was not
    armed, so trust geometry, not the label."""
    if end_zone is None:
        return False
    mn = np.asarray(end_zone["mins"], np.float64) - grow
    mx = np.asarray(end_zone["maxs"], np.float64) + grow
    p = ep["pos"][-1]
    return bool(np.all(p >= mn) and np.all(p <= mx))


def load_end_zone(eps, override=None):
    if override:
        return json.loads(Path(override).read_text(encoding="utf-8"))["end"]
    for e in eps:
        m = e["hdr"].get("map")
        if not m:
            continue
        p = ROOT / "maps" / f"{m}.zones.json"
        if not p.exists():
            p = Path("maps") / f"{m}.zones.json"
        if p.exists():
            z = json.loads(p.read_text(encoding="utf-8"))
            if z.get("end"):
                return z["end"]
    return None


# --------------------------------------------------------------------- reports

def series(gf, pos):
    """(values, honest_mask) at `pos` under `gf`."""
    v = gf.sample(pos).astype(np.float64)
    ok = v < gf.reach_max - 0.5 * gf.cell
    return v, ok


def fmt(v, ok):
    return f"{v:10.0f}" if ok else "  UNREACH"


def align_stats(v, ok):
    """Step statistics over consecutive samples of one route."""
    dv = np.diff(v)
    both = ok[:-1] & ok[1:]
    n = len(dv)
    dec = int(np.count_nonzero(both & (dv < 0.0)))
    inc = int(np.count_nonzero(both & (dv > 0.0)))
    broken = int(np.count_nonzero(~both))
    rise = float(dv[both & (dv > 0.0)].sum()) if both.any() else 0.0
    return {"n": n, "dec": dec, "inc": inc, "broken": broken, "rise": rise,
            "dec_frac": dec / n if n else float("nan"),
            "honest_frac": float(np.count_nonzero(ok)) / len(ok),
            "vmin": float(v[ok].min()) if ok.any() else float("nan"),
            "vmax": float(v[ok].max()) if ok.any() else float("nan"),
            "v0": float(v[0]) if ok[0] else float("nan"),
            "vend": float(v[-1]) if ok[-1] else float("nan")}


def worst_climb(v, ok, t):
    """Longest/largest contiguous RISING stretch of the potential along a
    route -> (t0, t1, v0, v1, rise). This is the deception in one number:
    on a winning line the field should never climb far, and 'the field went
    31k -> 107k on the champion route' is exactly this statistic."""
    best = (0.0, 0.0, float("nan"), float("nan"), 0.0)
    i = 0
    while i < len(v) - 1:
        if ok[i] and ok[i + 1] and v[i + 1] > v[i]:
            j = i
            while j < len(v) - 1 and ok[j + 1] and v[j + 1] > v[j]:
                j += 1
            if v[j] - v[i] > best[4]:
                best = (t[i], t[j], v[i], v[j], v[j] - v[i])
            i = j
        else:
            i += 1
    return best


def climb_line(tag, v, ok, t):
    t0, t1, v0, v1, rise = worst_climb(v, ok, t)
    if rise <= 0:
        return f"    worst rising stretch ({tag}): none - fully monotone"
    return (f"    worst rising stretch ({tag}): t {t0:6.1f} -> {t1:6.1f}   "
            f"{v0:9.0f} -> {v1:9.0f}   (+{rise:.0f}u)")


def route_table(ep, idx, base, dirf):
    pos = ep["pos"][idx]
    vb, ob = series(base, pos)
    vd, od = series(dirf, pos)
    t = ep["tick"][idx] * ep["tick_ms"] / 1000.0
    print("    t(s)        x        y        z |"
          "      base     d_base |       dir      d_dir")
    print("    " + "-" * 86)
    for i in range(len(idx)):
        db = "         ." if i == 0 or not (ob[i] and ob[i - 1]) \
            else f"{vb[i] - vb[i - 1]:10.0f}"
        dd = "         ." if i == 0 or not (od[i] and od[i - 1]) \
            else f"{vd[i] - vd[i - 1]:10.0f}"
        print(f"    {t[i]:6.1f} {pos[i, 0]:8.0f} {pos[i, 1]:8.0f} "
              f"{pos[i, 2]:8.0f} |{fmt(vb[i], ob[i])} {db} |"
              f"{fmt(vd[i], od[i])} {dd}")
    return (vb, ob), (vd, od)


def summary_row(tag, st):
    return (f"    {tag:28s} {st['n']:5d} {100 * st['dec_frac']:7.1f}% "
            f"{100 * st['honest_frac']:7.1f}% {st['broken']:7d} "
            f"{st['rise']:12.0f} {st['vmin']:11.0f} {st['vmax']:11.0f} "
            f"{st['v0']:11.0f} {st['vend']:11.0f}")


SUM_HEAD = (f"    {'route':28s} {'n':>5s} {'dec%':>8s} {'honest%':>8s} "
            f"{'broken':>7s} {'total rise':>12s} {'min':>11s} {'max':>11s} "
            f"{'start':>11s} {'end':>11s}")


def run_report(args):
    base = load_field(args.base)
    dirf = load_field(args.field)
    print("=" * 100)
    print("FIELDS")
    hb = field_banner("base", base)
    hd = field_banner("dir", dirf)
    if base.grid.shape != dirf.grid.shape:
        raise SystemExit("fields have different grids - different cell/map?")
    print(f"        directional graph disconnects "
          f"{hb - hd:,} voxels ({100.0 * (hb - hd) / max(hb, 1):.2f}% of the "
          f"plain field's honest space)")

    champ = []
    for f in expand([args.champ] if args.champ else []):
        champ += read_episodes(f)
    fails = []
    for f in expand(args.fail):
        fails += read_episodes(f)
    end_zone = load_end_zone(champ + fails, args.zones)
    print(f"\n  end zone: {end_zone}")

    fin = [e for e in champ if finished(e, end_zone)]
    unfin = [e for e in champ if not finished(e, end_zone)]
    if champ and not fin:
        print("  WARNING: no champion episode ends inside the end zone - "
              "treating every champion episode as a route")
        fin = champ
        unfin = []
    # a champion file's non-finishing episodes are failures too
    fails += [(e) for e in unfin]

    stats_b, stats_d = [], []
    if fin:
        print("\n" + "=" * 100)
        print(f"(a) CHAMPION ROUTE, every {args.period:g}s "
              f"({len(fin)} finishing episode(s))")
        for n, ep in enumerate(fin):
            idx = per_second(ep, args.period)
            dur = ep["tick"][-1] * ep["tick_ms"] / 1000.0
            print(f"\n  {ep['src']} ep{ep['idx']}  {len(ep['tick'])} ticks "
                  f"({dur:.2f}s)  end {np.round(ep['pos'][-1], 1).tolist()}")
            if n < args.tables or args.all:
                (vb, ob), (vd, od) = route_table(ep, idx, base, dirf)
            else:
                pos = ep["pos"][idx]
                vb, ob = series(base, pos)
                vd, od = series(dirf, pos)
                print("    (table suppressed; --all or --tables N to show)")
            tt = ep["tick"][idx] * ep["tick_ms"] / 1000.0
            print(climb_line("base", vb, ob, tt))
            print(climb_line("dir ", vd, od, tt))
            stats_b.append((f"{ep['src']} ep{ep['idx']}", align_stats(vb, ob)))
            stats_d.append((f"{ep['src']} ep{ep['idx']}", align_stats(vd, od)))

        print("\n" + "=" * 100)
        print("(b) ALIGNMENT on the champion route "
              "(dec% = seconds where the field falls; honest% = seconds "
              "with a reachable reading)")
        for name, rows in (("BASE  (goal_)", stats_b), ("DIR   (goalg_)",
                                                        stats_d)):
            print(f"\n  {name}")
            print(SUM_HEAD)
            for tag, st in rows:
                print(summary_row(tag, st))
            agg = {k: sum(s[k] for _, s in rows)
                   for k in ("n", "dec", "inc", "broken")}
            agg["rise"] = sum(s["rise"] for _, s in rows)
            agg["dec_frac"] = agg["dec"] / agg["n"] if agg["n"] else float("nan")
            agg["honest_frac"] = float(np.mean([s["honest_frac"]
                                                for _, s in rows]))
            for k, f in (("vmin", min), ("vmax", max)):
                agg[k] = f(s[k] for _, s in rows)
            agg["v0"] = float(np.mean([s["v0"] for _, s in rows]))
            agg["vend"] = float(np.mean([s["vend"] for _, s in rows]))
            print(summary_row("ALL FINISHING EPISODES", agg))

    fail_rows = []
    if fails:
        print("\n" + "=" * 100)
        print("(c) FAILURE TRAJECTORIES - reachable minimum, i.e. how deep "
              "the basin they die in reads")
        print("    'coll%' = (start - min) / start = the share of the whole "
              "shaping potential a FAILING")
        print("    episode banks. An honest field pays a dead end little; "
              "the deceptive one pays it nearly all.")
        print(f"\n    {'episode':36s} {'n':>4s} |{'base d0':>10s}"
              f"{'base min':>10s} {'coll%':>7s} |{'dir d0':>10s}"
              f"{'dir min':>10s} {'coll%':>7s}{'h%':>7s} | argmin (base)")
        print("    " + "-" * 122)
        for ep in fails:
            idx = per_second(ep, args.period)
            pos = ep["pos"][idx]
            vb, ob = series(base, pos)
            vd, od = series(dirf, pos)
            r = {"name": f"{ep['src']} ep{ep['idx']}", "n": len(idx),
                 "b0": float(vb[0]) if ob[0] else float("nan"),
                 "d0": float(vd[0]) if od[0] else float("nan"),
                 "bmin": float(vb[ob].min()) if ob.any() else float("nan"),
                 "dmin": float(vd[od].min()) if od.any() else float("nan"),
                 "dhon": float(od.mean())}
            for a, b, k in (("b0", "bmin", "bcoll"), ("d0", "dmin", "dcoll")):
                r[k] = (100.0 * (r[a] - r[b]) / r[a]) if r[a] > 0 else \
                    float("nan")
            r["where"] = pos[int(np.argmin(np.where(ob, vb, np.inf)))] \
                if ob.any() else np.zeros(3)
            fail_rows.append(r)
            print(f"    {r['name']:36s} {r['n']:4d} |{r['b0']:10.0f}"
                  f"{r['bmin']:10.0f} {r['bcoll']:6.1f}% |{r['d0']:10.0f}"
                  f"{r['dmin']:10.0f} {r['dcoll']:6.1f}%"
                  f"{100 * r['dhon']:6.1f}% | "
                  f"{np.round(r['where'], 0).astype(int).tolist()}")

    print("\n" + "=" * 100)
    print("VERDICT")
    for name, rows, k0, kmin, kcoll in (("base", stats_b, "b0", "bmin",
                                         "bcoll"),
                                        ("dir", stats_d, "d0", "dmin",
                                         "dcoll")):
        if rows:
            n = sum(s["n"] for _, s in rows)
            print(f"  {name:5s} champion route  ({len(rows)} ep, {n} steps): "
                  f"start {np.mean([s['v0'] for _, s in rows]):9.0f}  "
                  f"end {np.mean([s['vend'] for _, s in rows]):7.0f}  "
                  f"dec {100 * sum(s['dec'] for _, s in rows) / max(n, 1):5.1f}%"
                  f"  honest "
                  f"{100 * np.mean([s['honest_frac'] for _, s in rows]):5.1f}%"
                  f"  total rise {sum(s['rise'] for _, s in rows):8.0f}u")
        if fail_rows:
            c = np.array([r[kcoll] for r in fail_rows], np.float64)
            m = np.array([r[kmin] for r in fail_rows], np.float64)
            good = np.isfinite(c)
            print(f"  {name:5s} failure routes  ({len(fail_rows)} ep): "
                  f"collected median {np.median(c[good]):5.1f}%  "
                  f"worst {np.nanmax(c):5.1f}%  "
                  f"deepest min {np.nanmin(m):8.0f}u  "
                  f"{int(np.count_nonzero(c[good] > 90.0))} ep over 90%")
    if fail_rows and stats_b and stats_d:
        cb = np.array([r["bcoll"] for r in fail_rows], np.float64)
        cd = np.array([r["dcoll"] for r in fail_rows], np.float64)
        both = np.isfinite(cb) & np.isfinite(cd)
        drop = cb[both] - cd[both]
        print(f"\n  directional field cuts the potential a failure banks by "
              f"{np.median(drop):.1f}% of the route (median), "
              f"{np.max(drop):.1f}% at best;")
        print(f"  {int(np.count_nonzero(drop > 5.0))} of "
              f"{int(both.sum())} failure episodes lose more than 5 points, "
              f"{int(np.count_nonzero(drop < -1.0))} get MORE.")
        cdec = sum(s["dec"] for _, s in stats_d) / max(
            sum(s["n"] for _, s in stats_d), 1)
        bdec = sum(s["dec"] for _, s in stats_b) / max(
            sum(s["n"] for _, s in stats_b), 1)
        chon = float(np.mean([s["honest_frac"] for _, s in stats_d]))
        print(f"  cost on the winning route: dec {100 * bdec:.1f}% -> "
              f"{100 * cdec:.1f}%, honest readings {100 * chon:.1f}% "
              f"(the route must stay reachable and near-monotone, or the "
              f"fix traded one deception for another).")
    print("=" * 100)


# -------------------------------------------------------------------- selftest

def _grid(nz, ny, nx):
    return np.zeros((nz, ny, nx), np.uint8)


def _solve(occ, seed, cell=32.0, gravity_dir=False):
    from surfgym.goalfield import _bfs_geodesic
    d, _rm, _it = _bfs_geodesic(occ, seed, cell, gravity_dir=gravity_dir,
                                device="cpu", verbose=False)
    return d.numpy()


def _seed_top(shape, z):
    s = np.zeros(shape, bool)
    s[z] = True
    return s


def selftest():
    """Directionality unit tests on tiny synthetic grids. CPU only."""
    ok = True

    def show(v):
        if isinstance(v, (bool, np.bool_)):
            return "True" if v else "False"
        return "inf" if not np.isfinite(v) else f"{float(v):.2f}"

    def check(name, got, want):
        nonlocal ok
        if isinstance(want, (bool, np.bool_)):
            good = bool(got) == bool(want)
        elif np.isfinite(want):
            good = np.isfinite(got) and abs(float(got) - float(want)) < 1e-3
        else:
            good = not np.isfinite(got)
        ok = ok and bool(good)
        print(f"    [{'PASS' if good else 'FAIL'}] {name:58s} "
              f"got {show(got):>9s}  want {show(want):>9s}")

    cell = 32.0
    NZ, NY, NX = 20, 5, 7

    # --- A: sealed pit under open air ---------------------------------
    # floor everywhere; a 1-voxel shaft at x=3 walled in to z=5; nothing
    # solid above. Climbing along the shaft walls tops out a few cells
    # above them, so the goal plane at z=18 is reachable only by FALLING.
    occ = _grid(NZ, NY, NX)
    occ[0, :, :] = 1                              # floor
    occ[1:6, 1:4, 2] = 1                          # shaft wall -x
    occ[1:6, 1:4, 4] = 1                          # shaft wall +x
    occ[1:6, 1, 3] = 1                            # shaft wall -y
    occ[1:6, 3, 3] = 1                            # shaft wall +y
    seed = _seed_top(occ.shape, 18)
    pit = (1, 2, 3)                               # bottom of the shaft
    du = _solve(occ, seed, cell, gravity_dir=False)
    dg = _solve(occ, seed, cell, gravity_dir=True)
    print("  A. sealed pit, goal 18 cells overhead through empty air")
    check("undirected: pit bottom 'reaches' the goal (the bug)",
          float(du[pit]), 17.0 * cell)
    check("gravity_dir: pit bottom cannot reach the goal",
          float(dg[pit]), float("inf"))
    check("gravity_dir: floor far from the shaft cannot reach either",
          float(dg[1, 2, 0]), float("inf"))
    check("gravity_dir: a voxel ABOVE the goal still falls into it",
          float(dg[19, 2, 3]), cell)
    check("gravity_dir: goal plane itself is zero", float(dg[18, 2, 3]), 0.0)
    check("gravity_dir: solid voxels stay unreachable",
          float(dg[0, 2, 0]), float("inf"))

    # --- B: the same pit with a climbable column next to it ------------
    # a solid column from the floor to the goal plane: every voxel beside
    # it is surface-adjacent, so the wavefront may climb the whole way.
    occ = _grid(NZ, NY, NX)
    occ[0, :, :] = 1
    occ[1:18, 2, 5] = 1                           # ramp/wall column
    seed = _seed_top(occ.shape, 18)
    dg = _solve(occ, seed, cell, gravity_dir=True)
    print("  B. solid column beside it: climbing along geometry is allowed")
    check("gravity_dir: voxel hugging the column reaches the goal",
          np.isfinite(dg[1, 2, 4]), True)
    check("gravity_dir: climb cost is >= the straight-line height",
          float(dg[1, 2, 4]) >= 17.0 * cell - 1e-6, True)
    check("gravity_dir: floor across the room reaches it too (walk, climb)",
          np.isfinite(dg[1, 2, 0]), True)
    # remove the column -> the same room must go unreachable
    occ[1:18, 2, 5] = 0
    dg2 = _solve(occ, seed, cell, gravity_dir=True)
    check("gravity_dir: delete the column and the room is cut off",
          float(dg2[1, 2, 4]), float("inf"))
    check("undirected: deleting it changes nothing (why the bug hid)",
          np.isfinite(_solve(occ, seed, cell, gravity_dir=False)[1, 2, 4]),
          True)

    # --- C: open air, weights unchanged --------------------------------
    # empty grid, SINGLE seed voxel on the floor: every path is a fall plus
    # air-strafe, so directional and undirected must agree exactly and the
    # diagonal weights must survive the masking.
    occ = _grid(NZ, NY, NX)
    seed = np.zeros(occ.shape, bool)
    seed[0, 2, 3] = True
    du = _solve(occ, seed, cell, gravity_dir=False)
    dg = _solve(occ, seed, cell, gravity_dir=True)
    print("  C. open air, goal below: falling and strafing are unrestricted")
    check("gravity_dir: straight fall costs k cells", float(dg[5, 2, 3]),
          5.0 * cell)
    check("gravity_dir: diagonal fall uses the euclidean weight",
          float(dg[5, 2, 0]), (3.0 * np.sqrt(2.0) + 2.0) * cell)
    check("gravity_dir: fall + strafe pays sqrt(3) then sqrt(1)",
          float(dg[2, 4, 6]), (2.0 * np.sqrt(3.0) + 1.0) * cell)
    check("gravity_dir == undirected when the goal is at the bottom",
          float(np.abs(du - dg).max()), 0.0)

    # --- D: open air, goal above ---------------------------------------
    seed = _seed_top(occ.shape, 19)
    dg = _solve(occ, seed, cell, gravity_dir=True)
    du = _solve(occ, seed, cell, gravity_dir=False)
    print("  D. open air, goal above: nothing below may claim to reach it")
    check("undirected: everything 'reaches' the goal", np.isfinite(du).all(),
          True)
    check("gravity_dir: nothing below the goal plane reaches it",
          bool(np.isfinite(dg[:19]).any()), False)

    print(f"\n  selftest: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description="validate a gravity-directional goal field against "
                    "recorded trajectories (never bakes)")
    ap.add_argument("--base", help="plain field npz (maps/*.goal_<cell>.npz)")
    ap.add_argument("--field", help="directional field npz "
                                    "(maps/*.goalg_<cell>.npz)")
    ap.add_argument("--champ", help="champion trajectory jsonl")
    ap.add_argument("--fail", nargs="*", default=[],
                    help="failure trajectory jsonl paths or globs")
    ap.add_argument("--zones", help="zones json (default: maps/<map>.zones.json)")
    ap.add_argument("--period", type=float, default=1.0,
                    help="sampling period in seconds (default 1)")
    ap.add_argument("--tables", type=int, default=1,
                    help="how many champion episodes get a full table")
    ap.add_argument("--all", action="store_true",
                    help="full table for every champion episode")
    ap.add_argument("--selftest", action="store_true",
                    help="run the directionality unit tests and exit")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if not (args.base and args.field):
        raise SystemExit("need --base and --field (or --selftest)")
    run_report(args)


if __name__ == "__main__":
    main()
