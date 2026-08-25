#!/usr/bin/env python3
"""Pool-wide sweep for collision-visible / vision-invisible geometry.

Per map (maps_pool/*.bsp):
  A. func_illusionary probe: 3x3 down-traces per brush model, standing hull
     vs point hull. "clip-backed" = standing stops inside the bbox where the
     point hull passes through.
  B. random-sample probe: N free voxels with SDF > FAR_U (far from anything
     vision can see); a zero-length standing-hull trace that reports
     startsolid there = invisible clip geometry (hull1 clipnodes only).
  C. behavioral scan of the final-eval traj: landings (onground 0->1) and
     hard airborne decelerations at points with SDF > FAR_U.

ASCII-only output. Writes a CSV next to this script.
"""
import sys, os, json, glob, csv, argparse, traceback
sys.path.insert(0, str(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python")))
import numpy as np
from surfgym import SurfCore, default_config
from surfgym.zones import parse_bsp

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--pool", default="C:/RL_Surf/maps_pool",
                help="dir of <map>.bsp + <map>.sdf_32.npz")
ap.add_argument("--traj-dir", default="C:/RL_Surf/runs/mmPOOL_harvest/mmPOOL",
                help="dir of traj_<step>_<map>.jsonl eval recordings")
ap.add_argument("--step", default="25268060160",
                help="eval step whose trajs to scan behaviorally")
ap.add_argument("--out", default=None)
_a = ap.parse_args()
POOL = _a.pool
TRAJ = _a.traj_dir
FINAL_STEP = _a.step
FAR_U = 80.0
NSAMP = 3000
OUT = _a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "clip_sweep.csv")

rng = np.random.default_rng(0)

def episodes_full(path):
    eps, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        if line[0] == "{":
            if cur: eps.append(np.asarray(cur, np.float64)); cur = []
            continue
        row = json.loads(line)
        if isinstance(row, list) and len(row) >= 10: cur.append(row[:10])
    if cur: eps.append(np.asarray(cur, np.float64))
    return eps

rows = []
bsps = sorted(glob.glob(os.path.join(POOL, "*.bsp")))
for k, bsp in enumerate(bsps, 1):
    stem = os.path.splitext(os.path.basename(bsp))[0]
    rec = {"map": stem, "illu_total": 0, "illu_clipbacked": 0, "illu_nonsolid": 0,
           "illu_mixed": 0, "inv_clip_pts": 0, "samp_pts": 0,
           "land_events": 0, "land_invis": 0, "decel_events": 0, "decel_invis": 0,
           "err": ""}
    try:
        core = SurfCore(bsp, default_config(num_envs=1, spawn_mode=2, lidar_w=0, lidar_h=0))
        ents, bboxes = parse_bsp(bsp)

        # --- A: illusionary probe
        for e in ents:
            if e.get("classname") != "func_illusionary": continue
            mdl = e.get("model", "")
            if not mdl.startswith("*"): continue
            mi = int(mdl[1:])
            if mi >= len(bboxes): continue
            mn = np.array(bboxes[mi][0], float); mx = np.array(bboxes[mi][1], float)
            rec["illu_total"] += 1
            hs = hp = 0
            for fx in (0.25, 0.5, 0.75):
                for fy in (0.25, 0.5, 0.75):
                    x = mn[0] + fx*(mx[0]-mn[0]); y = mn[1] + fy*(mx[1]-mn[1])
                    top = mx[2] + 200.0; bot = mn[2] - 100.0
                    for hull in (0, 2):
                        tr = core.trace([x, y, top], [x, y, bot], hull)
                        inside = mn[2] - 4 <= tr.endpos[2] <= mx[2] + 40
                        if inside:
                            if hull == 0: hs += 1
                            else: hp += 1
            if hs > 0 and hp == 0: rec["illu_clipbacked"] += 1
            elif hs == 0 and hp == 0: rec["illu_nonsolid"] += 1
            else: rec["illu_mixed"] += 1

        # --- B: random far-from-visible points, standing-hull stuck?
        sdff = os.path.join(POOL, f"{stem}.sdf_32.npz")
        sdf = mins = None; cell = 32.0
        if os.path.exists(sdff):
            z = np.load(sdff, allow_pickle=False)
            sdf = np.asarray(z["sdf"], np.float32); mins = z["mins"].astype(np.float64)
            cell = float(z["cell"])
            far = np.argwhere(sdf > FAR_U)  # (z,y,x)
            if len(far):
                pick = far[rng.choice(len(far), size=min(NSAMP, len(far)), replace=False)]
                pts = mins + (pick[:, ::-1] + 0.5) * cell
                n_stuck = 0
                for p in pts:
                    tr = core.trace(p.tolist(), p.tolist(), 0)
                    if tr.startsolid: n_stuck += 1
                rec["inv_clip_pts"] = n_stuck
                rec["samp_pts"] = len(pts)

        # --- C: behavioral scan of final traj
        def sdf_at(p):
            if sdf is None: return float("nan")
            i = np.floor((p - mins) / cell).astype(int)
            nz, ny, nx = sdf.shape
            if not (0 <= i[0] < nx and 0 <= i[1] < ny and 0 <= i[2] < nz): return float("nan")
            return float(sdf[i[2], i[1], i[0]])

        tf = os.path.join(TRAJ, f"traj_{FINAL_STEP}_{stem.replace('surf_','')}.jsonl")
        if os.path.exists(tf):
            for ep in episodes_full(tf):
                xyz = ep[:, 1:4]; v = ep[:, 4:7]; og = ep[:, 9] > 0.5
                # teleport filter: position jump > 200u in one tick
                jump = np.zeros(len(ep), bool)
                if len(ep) > 1:
                    jump[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=1) > 200.0
                land = np.where(og & ~np.roll(og, 1) & ~jump)[0]; land = land[land > 0]
                for t in land:
                    rec["land_events"] += 1
                    s = sdf_at(xyz[t])
                    if s == s and s > FAR_U: rec["land_invis"] += 1
                if len(ep) > 2:
                    sp = np.linalg.norm(v, axis=1)
                    dec = np.where((sp[:-1] - sp[1:] > 250.0) & ~og[:-1] & ~og[1:]
                                   & ~jump[1:])[0]
                    for t in dec:
                        rec["decel_events"] += 1
                        s = sdf_at(xyz[t+1])
                        if s == s and s > FAR_U: rec["decel_invis"] += 1
        del core
    except Exception as ex:
        rec["err"] = f"{type(ex).__name__}: {ex}"
        traceback.print_exc()
    rows.append(rec)
    if k % 10 == 0 or k == len(bsps):
        print(f"[{k}/{len(bsps)}] done", flush=True)

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows: w.writerow(r)

# summary
aff_a = [r for r in rows if r["illu_clipbacked"] > 0]
aff_b = [r for r in rows if r["inv_clip_pts"] > 0]
aff_c = [r for r in rows if r["land_invis"] > 0 or r["decel_invis"] > 0]
print()
print(f"maps swept: {len(rows)}  (csv: {OUT})")
print(f"A. maps with CLIP-backed func_illusionary: {len(aff_a)}")
for r in sorted(aff_a, key=lambda r: -r["illu_clipbacked"]):
    print(f"   {r['map']:28s} clipbacked {r['illu_clipbacked']:3d} / {r['illu_total']:3d} illusionary "
          f"(nonsolid {r['illu_nonsolid']}, mixed {r['illu_mixed']})")
print(f"B. maps with invisible clip volumes in open space (of {NSAMP} sampled far pts): {len(aff_b)}")
for r in sorted(aff_b, key=lambda r: -r["inv_clip_pts"]):
    print(f"   {r['map']:28s} stuck {r['inv_clip_pts']:5d} / {r['samp_pts']} pts "
          f"({100.0*r['inv_clip_pts']/max(r['samp_pts'],1):.1f} pct)")
print(f"C. maps whose final traj shows contact in SDF-free space: {len(aff_c)}")
for r in sorted(aff_c, key=lambda r: -(r["land_invis"]+r["decel_invis"])):
    print(f"   {r['map']:28s} invis landings {r['land_invis']:3d}/{r['land_events']:3d} "
          f"invis decels {r['decel_invis']:3d}/{r['decel_events']:3d}")
errs = [r for r in rows if r["err"]]
if errs:
    print("errors:")
    for r in errs: print("   ", r["map"], r["err"])
