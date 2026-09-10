#!/usr/bin/env python3
"""visitation.py - at the place where a policy dies, has it ever VISITED the
states of the branch that survives?

M5.  Two lines are read out of trajectory files - the NAIVE one (what the
checkpoint's own greedy policy does, and where it dies) and the CORRECT one
(a line that survives that place: a demo, or one of our own finishers) - and
every sample of each is looked up in the checkpoint's own intrinsic-novelty
visit-count table (``ck["int_counts"]``).

The table's cell layout is ``RaceReward._cells`` (python/surfgym/rewards.py).
``tools/diversity_bench.py`` already reads it out of a checkpoint and this
module imports ``cell_layout`` / ``pos_cells`` / ``position_counts`` from
there rather than re-deriving anything, so the cells here are the cells the
reward paid novelty on.  ``position_counts`` MARGINALISES over the
``int_view`` yaw sectors and the ``int_speed`` horizontal-speed bins: a
reported count is "how many ticks did any env stand in this 256 u box, at any
yaw and any speed".  That is the only marginal that is comparable between two
lines that cross the same geometry at different speeds and headings.

Presentation follows ``tools/demo/wr_scan.py``: a per-line count distribution
(min / p10 / median / p90 / max, zero-cells), whole line and critical window,
plus the median ratio which is the headline.

The reservoir half reads ``ck["respawn"]["states"]`` and scores every state
with the trainer's own baked geodesic field (``maps/<stem>.goal_<cell>.npz``,
loaded straight from the cache - this NEVER calls ``build_goal_field`` and so
can never trigger a bake).

Read-only.  Nothing here writes into maps/ or runs/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import SurfCore, default_config          # noqa: E402
from surfgym.goalfield import GoalField                    # noqa: E402
from diversity_bench import (cell_layout as _db_cell_layout,   # noqa: E402
                             pos_cells, position_counts)


def cell_layout(core: SurfCore, cell: float):
    """RaceReward's layout, byte-for-byte (rewards.py ~line 988):

        mins, maxs = core.map_bounds()          # both float32
        self._mins = mins.astype(np.float64)
        self._dims = ceil((maxs[i] - mins[i]) / int_cell) + 1

    The subtraction happens in the ORIGINAL float32, and that matters:
    surf_petrus_lite's y span is 4096.0005 - (-4096), which is exactly
    8192.0 in float32 (dims 33) and 8192.000488 in float64 (dims 34).
    ``tools/diversity_bench.cell_layout`` casts mins to float64 first and so
    reports 34 - which does not divide that map's 418,176-entry table.  The
    float32 form is asserted against the checkpoint below; diversity_bench's
    form is computed alongside and the difference reported."""
    mins, maxs = core.map_bounds()
    dims = tuple(int(np.ceil((maxs[i] - mins[i]) / cell)) + 1 for i in range(3))
    return mins.astype(np.float64), dims


# ------------------------------------------------------------------ io

def load_field(bsp: Path, cell: float = 32) -> GoalField:
    """The trainer's GoalField straight out of the baked cache; never bakes."""
    p = Path(f"{Path(bsp).with_suffix('')}.goal_{cell:g}.npz")
    if not p.exists():
        raise SystemExit(f"no baked field {p} - refusing to bake")
    z = np.load(p, allow_pickle=False)
    grid = z["grid"].astype(np.float32) * float(z["quant"])
    return GoalField(grid, z["mins"], float(z["cell"]), float(z["reach_max"]))


def load_traj(path):
    """-> [(header, footer, rows ndarray)] for every episode in the file.

    Row layout (surfgym/record.py): [t, x,y,z, vx,vy,vz, yaw, buttons,
    onground, progress, reward, pitch, fwd, side]."""
    eps, rows, hdr = [], [], None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if isinstance(r, list):
                rows.append(r)
            elif isinstance(r, dict) and "end" in r:
                if rows:
                    eps.append((hdr, r, np.asarray(rows, float)))
                rows = []
            elif isinstance(r, dict):
                hdr, rows = r, []
    if rows:
        eps.append((hdr, {"end": "?"}, np.asarray(rows, float)))
    return eps


def run_cfg(ckpt: Path) -> dict:
    """The FULL training config.  The dict inside the checkpoint is a 21-key
    summary; run.json next to it carries every flag, including int_cell /
    int_view / int_speed."""
    rj = Path(ckpt).parent / "run.json"
    if rj.exists():
        d = json.loads(rj.read_text(encoding="utf-8"))
        if isinstance(d.get("config"), dict):
            return d["config"]
    return {}


# ------------------------------------------------------------------ stats

def describe(counts: np.ndarray) -> dict:
    c = np.asarray(counts, np.int64)
    if c.size == 0:
        return dict(n=0)
    return dict(n=int(c.size), zero=int((c == 0).sum()),
                min=int(c.min()),
                p10=float(np.percentile(c, 10)),
                med=float(np.median(c)),
                p90=float(np.percentile(c, 90)),
                max=int(c.max()), mean=float(c.mean()))


def row_fmt(name, d):
    if not d.get("n"):
        return f"| {name} | 0 | - | - | - | - | - | - |"
    return (f"| {name} | {d['n']} | {d['zero']} | {d['min']:,} | {d['p10']:,.0f} "
            f"| {d['med']:,.0f} | {d['p90']:,.0f} | {d['max']:,} |")


HDR = ("| set | samples | zero | min | p10 | median | p90 | max |\n"
       "|---|---|---|---|---|---|---|---|")


def cellstats(cells: np.ndarray, counts: np.ndarray):
    """-> (per-sample counts, distinct cells, per-distinct-cell counts)."""
    per_sample = counts[cells]
    uniq = np.unique(cells)
    return per_sample, uniq, counts[uniq]


def oob(xyz, mins, dims, cell):
    """How many samples fall OUTSIDE the table's index range before pos_cells
    clips them - i.e. 'zero because never visited' vs 'not addressable'."""
    p = np.asarray(xyz, np.float64).reshape(-1, 3)
    idx = np.stack([((p[:, i] - mins[i]) // cell).astype(np.int64)
                    for i in range(3)], 1)
    bad = np.zeros(len(p), bool)
    for i in range(3):
        bad |= (idx[:, i] < 0) | (idx[:, i] >= dims[i])
    return int(bad.sum())


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--map", required=True, help="absolute path to the .bsp")
    ap.add_argument("--naive", required=True, help="trajectory jsonl")
    ap.add_argument("--naive-ep", type=int, default=-1,
                    help="-1 = the DEEPEST episode (lowest geodesic d)")
    ap.add_argument("--correct", required=True, help="trajectory jsonl")
    ap.add_argument("--correct-ep", type=int, default=0)
    ap.add_argument("--d0", type=float, required=True)
    ap.add_argument("--goal-cell", type=float, default=32.0)
    ap.add_argument("--int-cell", type=float, default=256.0)
    ap.add_argument("--route", default=None,
                    help="a .route.npz; enables arc-fraction windows")
    ap.add_argument("--branch-arc-frac", type=float, default=None,
                    help="branch point as a fraction of route arc (needs "
                         "--route)")
    ap.add_argument("--branch-pos", default=None,
                    help="branch point as 'x,y,z' (used when there is no "
                         "route)")
    ap.add_argument("--window-secs", type=float, default=1.5,
                    help="critical window = this many seconds after the "
                         "branch point, on each line")
    ap.add_argument("--head-frac", type=float, default=0.30,
                    help="'first N% of the route' band, by geodesic progress")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = run_cfg(Path(a.ckpt))
    step = int(ck.get("global_step", 0))
    int_cell = float(cfg.get("int_cell") or a.int_cell)
    int_view = int(cfg.get("int_view") or 0)
    int_speed = int(cfg.get("int_speed") or 0)
    stem = Path(a.map).stem

    core = SurfCore(a.map, default_config(num_envs=1, spawn_mode=2,
                                          max_episode_ticks=12000,
                                          lidar_w=0, lidar_h=0), tick_ms=10.0)
    mins, dims = cell_layout(core, int_cell)
    _, dims_db = _db_cell_layout(core, int_cell)
    core.close()
    if dims_db != dims:
        print(f"note: diversity_bench.cell_layout says {dims_db} (float64 "
              f"mins); RaceReward's float32 form says {dims} and that is what "
              f"the table's size confirms")
    counts = position_counts(ck, stem, dims, int_view, int_speed)
    if counts is None:
        raise SystemExit("checkpoint carries no int_counts")
    field = load_field(Path(a.map), a.goal_cell)

    out = {}
    P = print
    P(f"# visitation: {a.ckpt}")
    P(f"map {stem}  step {step:,}  d0 {a.d0:,.2f}")
    P(f"int_cell {int_cell:g}  int_view {int_view}  int_speed {int_speed}  "
      f"-> {counts.size:,} position cells x {max(1,int_view)*max(1,int_speed)} "
      f"view/speed bins (marginalised over both)")
    P(f"table: {int((counts > 0).sum()):,} cells visited "
      f"({100.0*(counts > 0).mean():.2f}%), {int(counts.sum()):,} tick-visits")
    out["meta"] = dict(ckpt=a.ckpt, map=stem, step=step, d0=a.d0,
                       int_cell=int_cell, int_view=int_view,
                       int_speed=int_speed, n_pos_cells=int(counts.size),
                       cells_visited=int((counts > 0).sum()),
                       total_visits=int(counts.sum()))

    # ---- the two lines --------------------------------------------------
    def pick(path, which):
        eps = load_traj(path)
        if which >= 0:
            hdr, foot, rows = eps[which]
            k = which
        else:
            ds = [float(field.sample(e[2][:, 1:4]).min()) for e in eps]
            k = int(np.argmin(ds))
            hdr, foot, rows = eps[k]
        return k, hdr, foot, rows, len(eps)

    kn, hn, fn, naive, nn = pick(a.naive, a.naive_ep)
    kc, hc, fc, corr, nc = pick(a.correct, a.correct_ep)
    tick_n = float(hn.get("tick_ms", 10.0))
    tick_c = float(hc.get("tick_ms", 10.0))
    P(f"\nNAIVE   {Path(a.naive).name} ep {kn}/{nn}  {len(naive)} rows @ "
      f"{tick_n:g} ms = {len(naive)*tick_n/1000:.2f} s  end={fn.get('end')}")
    P(f"CORRECT {Path(a.correct).name} ep {kc}/{nc}  {len(corr)} rows @ "
      f"{tick_c:g} ms = {len(corr)*tick_c/1000:.2f} s  end={fc.get('end')}")

    lines = {}
    for nm, rows, tick in (("naive", naive, tick_n), ("correct", corr, tick_c)):
        xyz = rows[:, 1:4]
        d = field.sample(xyz).astype(np.float64)
        prog = (a.d0 - d) / a.d0
        lines[nm] = dict(rows=rows, xyz=xyz, d=d, prog=prog, tick=tick,
                         cells=pos_cells(xyz, mins, dims, int_cell))
        P(f"  {nm}: d {d.min():,.0f}..{d.max():,.0f}  geodesic progress "
          f"max {prog.max()*100:.2f}%")

    # ---- arc coordinate (cannonball) ------------------------------------
    arc_len = None
    if a.route:
        from surfgym.route import ArcProgress
        z = np.load(a.route)
        pts, spacing = z["route"].astype(np.float64), float(z["spacing"])
        arc_len = (len(pts) - 1) * spacing
        P(f"\nroute {Path(a.route).name}: {len(pts)} vertices, spacing "
          f"{spacing:g} -> {arc_len:,.0f} u; order-only window 16, "
          f"corridor 1500")
        for nm, L in lines.items():
            apr = ArcProgress(pts, spacing, corridor=1500.0, window=16)
            apr.reset(L["xyz"][:1])
            arc = np.empty(len(L["xyz"]))
            arc[0] = float(apr.arc[0])
            for k in range(1, len(L["xyz"])):
                apr.advance(L["xyz"][k:k + 1])
                arc[k] = float(apr.arc[0])
            L["arc"] = arc
            L["arcfrac"] = arc / arc_len
            P(f"  {nm}: arc max {arc.max():,.0f} u "
              f"({100*arc.max()/arc_len:.2f}%)")

    # ---- the critical window on each line -------------------------------
    def window_mask(L):
        """rows from the branch point to +window_secs (or the line's end)."""
        n = len(L["xyz"])
        if a.branch_arc_frac is not None and "arcfrac" in L:
            hit = np.flatnonzero(L["arcfrac"] >= a.branch_arc_frac)
            if not len(hit):
                return np.zeros(n, bool), -1
            i0 = int(hit[0])
        else:
            bp = np.array([float(v) for v in a.branch_pos.split(",")])
            i0 = int(np.argmin(((L["xyz"] - bp) ** 2).sum(1)))
        i1 = min(n, i0 + int(round(a.window_secs * 1000.0 / L["tick"])))
        m = np.zeros(n, bool)
        m[i0:i1] = True
        return m, i0

    res = {}
    P(f"\n## 1. whole line\n\n{HDR}")
    for nm, L in lines.items():
        ps, uq, uc = cellstats(L["cells"], counts)
        res.setdefault(nm, {})["whole_sample"] = describe(ps)
        res[nm]["whole_cell"] = describe(uc)
        res[nm]["whole_distinct"] = int(len(uq))
        res[nm]["whole_uniq"] = uq
        P(row_fmt(f"{nm} (per sample)", describe(ps)))
        P(row_fmt(f"{nm} (per distinct cell, n={len(uq)})", describe(uc)))
    sh = np.intersect1d(res["naive"]["whole_uniq"], res["correct"]["whole_uniq"])
    P(f"\ndistinct cells: naive {res['naive']['whole_distinct']}, correct "
      f"{res['correct']['whole_distinct']}, shared {len(sh)}")

    P(f"\n## 2. critical window ({a.window_secs:g} s from the branch point)"
      f"\n\n{HDR}")
    wmed = {}
    for nm, L in lines.items():
        m, i0 = window_mask(L)
        res[nm]["window_i0"] = i0
        res[nm]["window_n"] = int(m.sum())
        if not m.sum():
            P(row_fmt(f"{nm} (per sample)", dict(n=0)))
            continue
        ps, uq, uc = cellstats(L["cells"][m], counts)
        res[nm]["win_sample"] = describe(ps)
        res[nm]["win_cell"] = describe(uc)
        res[nm]["win_uniq"] = uq
        wmed[nm] = describe(ps)["med"]
        P(row_fmt(f"{nm} (per sample)", describe(ps)))
        P(row_fmt(f"{nm} (per distinct cell, n={len(uq)})", describe(uc)))
        t0 = L["rows"][i0, 0] * L["tick"] / 1000.0
        P(f"|   ^ {nm} window: rows {i0}..{i0+int(m.sum())-1}, t "
          f"{t0:.2f} s, from {np.round(L['xyz'][i0],1).tolist()} |||||||")
    if "win_uniq" in res["naive"] and "win_uniq" in res["correct"]:
        shw = np.intersect1d(res["naive"]["win_uniq"], res["correct"]["win_uniq"])
        P(f"\nwindow distinct cells: naive {len(res['naive']['win_uniq'])}, "
          f"correct {len(res['correct']['win_uniq'])}, shared {len(shw)}")
    if len(wmed) == 2 and wmed["correct"] > 0:
        P(f"\n**HEADLINE  median(naive)/median(correct) in the window = "
          f"{wmed['naive']/wmed['correct']:.2f}x**  "
          f"({wmed['naive']:,.0f} / {wmed['correct']:,.0f})")
        res["headline_ratio"] = wmed["naive"] / wmed["correct"]
    elif len(wmed) == 2:
        P(f"\n**HEADLINE  median(correct) = 0 in the window; median(naive) = "
          f"{wmed['naive']:,.0f} -> ratio infinite**")
        res["headline_ratio"] = float("inf")

    P(f"\n## 3. first {a.head_frac*100:.0f}% of the route "
      f"(geodesic progress band)\n\n{HDR}")
    for nm, L in lines.items():
        m = L["prog"] <= a.head_frac
        if not m.sum():
            P(row_fmt(nm, dict(n=0)))
            continue
        ps, uq, uc = cellstats(L["cells"][m], counts)
        res[nm]["head_sample"] = describe(ps)
        res[nm]["head_distinct"] = int(len(uq))
        P(row_fmt(f"{nm} (per sample, n_rows={int(m.sum())})", describe(ps)))
        P(row_fmt(f"{nm} (per distinct cell, n={len(uq)})", describe(uc)))

    # ---- 4. addressability ---------------------------------------------
    P("\n## 4. are the CORRECT line's zero cells never-visited or "
      "un-addressable?")
    for nm, L in lines.items():
        n_oob = oob(L["xyz"], mins, dims, int_cell)
        zc = int((counts[np.unique(L["cells"])] == 0).sum())
        P(f"  {nm}: {n_oob} of {len(L['xyz'])} samples outside the table's "
          f"index range (clipped by pos_cells); {zc} of "
          f"{len(np.unique(L['cells']))} distinct cells have count 0")
        res[nm]["oob"] = n_oob
        res[nm]["zero_cells"] = zc

    # ---- 5. reservoir ---------------------------------------------------
    P("\n## 5. respawn reservoir")
    rs = ck.get("respawn") or {}
    st = rs.get("states")
    if st is None:
        P("  no reservoir in the checkpoint")
    else:
        org = np.asarray(st["origin"], np.float64)
        dr = field.sample(org).astype(np.float64)
        dep = 100.0 / a.d0 * (a.d0 - dr)
        P(f"  {len(org):,} states (map_id {rs.get('map_id')!r})")
        P(f"  geodesic d: min {dr.min():,.0f}  p10 {np.percentile(dr,10):,.0f} "
          f" median {np.median(dr):,.0f}  max {dr.max():,.0f}")
        P(f"  min-depth (reward units, 100/d0*(d0-d)): {dep.max():.3f} "
          f"[= the DEEPEST state]; median {np.median(dep):.3f}")
        res["reservoir"] = dict(n=int(len(org)), d_min=float(dr.min()),
                                d_med=float(np.median(dr)),
                                depth_max=float(dep.max()))
        for nm, L in lines.items():
            i0 = res[nm].get("window_i0", -1)
            if i0 < 0:
                continue
            db_ = float(L["d"][i0])
            past = int((dr < db_).sum())
            P(f"  past the {nm} branch point (d < {db_:,.0f}): {past:,} "
              f"= {100.0*past/len(dr):.3f}%")
            res["reservoir"][f"past_{nm}"] = past
            res["reservoir"][f"d_branch_{nm}"] = db_

    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {k: v for k, v in res.items() if not isinstance(v, np.ndarray)},
            indent=1, default=lambda o: (o.tolist()
                                         if isinstance(o, np.ndarray)
                                         else float(o))), encoding="utf-8")
        P(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
