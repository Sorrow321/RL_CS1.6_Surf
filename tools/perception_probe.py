#!/usr/bin/env python3
"""perception_probe.py - M6: what the policy actually SEES at the petrus bend.

Round 37 established WHY the agent dies at 17-22% of surf_petrus_lite in
reward and credit terms, and ruled out tolerance, the critic and the
horizon. The remaining hypothesis space is PERCEPTION and ACTION
REACHABILITY. This tool measures the perception half, with no training:

  * the SURVIVING RAMP as a voxel set, built from a reference line's own
    support (surfy voxels, 0.1 <= |n_z| <= 0.7, within --ramp-radius of
    the reference polyline over an arc band);
  * per DECISION of the dying episode: is that set inside the 120x90 FOV
    at all (an angular test, occlusion-free), how many of the 64x32 = 2048
    pixels it actually occupies once the march has run, at what depth
    values, and how its depth CONTRASTS with the wall/void beside it;
  * the same frame's --obs-potential channel on those pixels against the
    void beside them;
  * the SURF MASK the same poses would produce under --surf-mask 1
    (GpuLidar(surf_mask=True), the |n_z| second channel), which is the
    channel the depth image cannot supply;
  * a YAW SWEEP: the same decision rendered at heading(v) + offset for a
    list of offsets, so "what would it see if it looked 90 degrees left"
    is answered in pixels rather than in prose.

Every channel is built exactly as tools/record_ckpt.py builds it for the
checkpoint (LidarPotential.from_cfg on the run's own run.json), so the
depth and potential frames ARE the observation. The surf-mask lidar is a
SECOND GpuLidar over the same SDF and the same camera; its depth channel
is asserted equal to the potential lidar's, which proves the two are the
same march and the mask is the only new information.

The map path must be the MAIN checkout's (CLAUDE.md's worktree trap); a
goal-field load over 30 s is reported as a re-bake.

    python tools/perception_probe.py \\
        --traj C:/RL_Surf_pr1/runs/prRATCH/traj_2757754880.jsonl \\
        --episode best --run-json C:/RL_Surf_pr1/runs/prRATCH/run.json \\
        --map C:/RL_Surf/maps/surf_petrus_lite.bsp \\
        --route C:/RL_Surf/maps/surf_petrus_lite.wrroute.npz \\
        --ref runs/research/cornerdiag/wr/petrus_wr.jsonl --ref-episode 0 \\
        --t0 4.6 --arc-lo 17 --arc-hi 30 \\
        --out runs/research/petrusperc/m6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch                                                    # noqa: E402

from surfgym import SurfCore, default_config                    # noqa: E402
from surfgym.goalfield import build_goal_field                  # noqa: E402
from surfgym.tick import step_seconds                           # noqa: E402
from surfgym.vision import GpuLidar, LidarPotential, pick_cell   # noqa: E402
from surfgym.zones import load_zones                            # noqa: E402

sys.path.insert(0, str(ROOT / "tools" / "demo"))
from render_channels import load_episodes, vision_cells          # noqa: E402

WALL_NZ = 0.1        # field_probe's band: below this is a wall
GROUND_NZ = 0.7      # above this you WALK (pm.c's walkable threshold)


# ----------------------------------------------------------------- geometry
def arc_pct(route_pts: np.ndarray, xyz: np.ndarray):
    """Segment-projected arc along the reference route, as a percentage,
    plus the perpendicular offset. Same rule as tools/branch_table.py."""
    a, b = route_pts[:-1], route_pts[1:]
    ab = b - a
    L2 = (ab * ab).sum(1)
    seglen = np.sqrt(L2)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    out = np.empty(len(xyz))
    off = np.empty(len(xyz))
    for k, p in enumerate(xyz):
        t = np.clip(((p - a) * ab).sum(1) / L2, 0.0, 1.0)
        q = a + ab * t[:, None]
        dd = np.linalg.norm(q - p, axis=1)
        j = int(np.argmin(dd))
        out[k] = cum[j] + t[j] * seglen[j]
        off[k] = dd[j]
    return 100.0 * out / total, off, total


def ramp_voxels(nz_grid, occ, mins, cell, ref_xyz, radius):
    """The surfy voxels within `radius` of the reference polyline: the
    SURVIVING RAMP as a boolean grid in (z, y, x), plus its world centres.

    Surfy = WALL_NZ <= |n_z| <= GROUND_NZ, i.e. what the player RIDES; a
    floor he could only walk on is excluded (field_probe's split)."""
    surfy = (nz_grid >= WALL_NZ) & (nz_grid <= GROUND_NZ)
    zz, yy, xx = np.nonzero(surfy)
    cx = mins[0] + (xx + 0.5) * cell
    cy = mins[1] + (yy + 0.5) * cell
    cz = mins[2] + (zz + 0.5) * cell
    cen = np.stack([cx, cy, cz], 1)
    keep = np.zeros(len(cen), bool)
    for p in ref_xyz:
        keep |= (np.abs(cen - p) <= radius).all(1) & \
                (np.linalg.norm(cen - p, axis=1) <= radius)
    mask = np.zeros(nz_grid.shape, bool)
    mask[zz[keep], yy[keep], xx[keep]] = True
    return mask, cen[keep]


def dilate1(mask):
    """26-neighbourhood dilation by one voxel: the march's hit point lands
    on the SDF surface, up to a cell away from the solid voxel centre."""
    out = mask.copy()
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out |= np.roll(np.roll(np.roll(mask, dz, 0), dy, 1), dx, 2)
    return out


def cam_angles(eye, yaw_deg, pitch_deg, pts):
    """(delta_yaw, delta_pitch, range) of world points in the camera frame,
    in degrees. The renderer's own model: a ray is
    dx=cos(p)cos(y), dy=cos(p)sin(y), dz=sin(p) with y = yaw + yoff[col],
    p = pitch + poff[row], so a point's own azimuth/elevation ARE y and p."""
    rel = pts - eye
    r = np.linalg.norm(rel, axis=1)
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    el = np.degrees(np.arcsin(np.clip(rel[:, 2] / np.maximum(r, 1e-9),
                                      -1.0, 1.0)))
    dy = (az - yaw_deg + 180.0) % 360.0 - 180.0
    dp = el - pitch_deg
    return dy, dp, r


def pix_of(dyaw, dpitch, W, H, hfov, vfov):
    """The pixel a direction offset falls in (col from yoff, row from poff:
    col 0 is +hfov/2 = LEFT, row 0 is +vfov/2 = UP)."""
    col = W * (0.5 - dyaw / hfov) - 0.5
    row = H * (0.5 - dpitch / vfov) - 0.5
    return col, row


# ------------------------------------------------------------------- render
def hit_points(lidar, origin, yaw, pitch, duck, depth):
    """World hit point of every pixel: eye + dir * decode_depth(depth)."""
    N = origin.shape[0]
    lidar._ensure_buffers(N)
    lidar._dirs_equiangular(N, yaw, pitch, float(np.pi / 180.0))
    t = lidar.decode_depth(depth.reshape(N, lidar.H, lidar.W))
    ez = origin[:, 2] + torch.where(duck.bool(), 12.0, 17.0)
    hx = origin[:, 0].view(N, 1, 1) + lidar._dx * t
    hy = origin[:, 1].view(N, 1, 1) + lidar._dy * t
    hz = ez.view(N, 1, 1) + lidar._dz * t
    return torch.stack([hx, hy, hz], -1), t


def lookup(mask, mins, cell, pts):
    """Boolean grid lookup for a batch of world points (numpy)."""
    idx = np.floor((pts - mins) / cell).astype(np.int64)
    ok = np.all(idx >= 0, -1)
    shp = np.array(mask.shape)[::-1]           # (x, y, z)
    ok &= np.all(idx < shp, -1)
    out = np.zeros(idx.shape[:-1], bool)
    i = idx[ok]
    out[ok] = mask[i[:, 2], i[:, 1], i[:, 0]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True, help="the DYING episode's file")
    ap.add_argument("--episode", default="best")
    ap.add_argument("--run-json", required=True,
                    help="the policy's own run.json (the channels come from it)")
    ap.add_argument("--map", required=True, help="MAIN checkout .bsp")
    ap.add_argument("--route", required=True, help="the arc ruler (.npz)")
    ap.add_argument("--ref", required=True,
                    help="the SURVIVING line (jsonl), for the ramp voxel set")
    ap.add_argument("--ref-episode", type=int, default=0)
    ap.add_argument("--arc-lo", type=float, default=17.0)
    ap.add_argument("--arc-hi", type=float, default=30.0)
    ap.add_argument("--ramp-radius", type=float, default=160.0)
    ap.add_argument("--branch-arc-pct", type=float, default=17.0,
                    help="the branch, as arc %% along --route; both lines "
                         "are indexed from their OWN crossing of it")
    ap.add_argument("--ref-t0", type=float, default=None,
                    help="seconds AFTER the reference line's own branch "
                         "crossing: the start of the SURVIVING-RAMP window. "
                         "Set it past the reference's own unsupported gap "
                         "(0.52 s on petrus, Round 37 M2) so the ramp set is "
                         "the branch-specific ramp and not the shared one")
    ap.add_argument("--ref-t1", type=float, default=None)
    ap.add_argument("--t0", type=float, default=None, help="window start, s")
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--yaw-offsets", default="-90,-45,-20,-10,0,10,20,45,90",
                    help="the sweep, degrees off heading(v)")
    ap.add_argument("--sweep-at", default="",
                    help="comma-separated decision times (s) for the sweep")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.run_json).read_text(encoding="utf-8"))["config"]
    act_every = int(cfg.get("act_every") or 1)

    # ---- the dying episode -------------------------------------------
    eps, hdrs = load_episodes(Path(a.traj))
    route = np.load(a.route)["route"]
    if a.episode == "best":
        sc = [arc_pct(route, e[:, 1:4])[0].max() for e in eps]
        ep_i = int(np.argmax(sc))
        print("arc %% per episode: " + " ".join(f"ep{k} {s:.2f}"
                                                for k, s in enumerate(sc)))
    else:
        ep_i = int(a.episode)
    rows, hdr = eps[ep_i], hdrs[ep_i] or {}
    n = len(rows)
    dt = np.asarray(step_seconds(hdr, n), np.float64)
    t_s = np.concatenate([[0.0], np.cumsum(dt)[:-1]])
    pct, off, total = arc_pct(route, rows[:, 1:4])
    print(f"{Path(a.traj).name} ep{ep_i}: {n} ticks, {t_s[-1] + dt[-1]:.2f} s, "
          f"arc max {pct.max():.2f}% of {total:,.0f} u")

    # ---- the surviving line and the ramp voxel set --------------------
    reps, rhdrs = load_episodes(Path(a.ref))
    ref = reps[a.ref_episode]
    rpct, _roff, _ = arc_pct(route, ref[:, 1:4])
    rdt = np.asarray(step_seconds(rhdrs[a.ref_episode] or {}, len(ref)),
                     np.float64)
    rt = np.concatenate([[0.0], np.cumsum(rdt)[:-1]])
    if a.ref_t0 is not None or a.ref_t1 is not None:
        j0 = int(np.argmax(rpct >= a.branch_arc_pct))
        lo = rt[j0] + (a.ref_t0 if a.ref_t0 is not None else 0.0)
        hi = rt[j0] + (a.ref_t1 if a.ref_t1 is not None else 1e9)
        band = (rt >= lo) & (rt <= hi)
        print(f"reference line: branch at row {j0} (t {rt[j0]:.2f} s, arc "
              f"{rpct[j0]:.2f}%); SURVIVING-RAMP window t+{a.ref_t0}"
              f"..{a.ref_t1} s = rows {np.flatnonzero(band)[[0, -1]]}, "
              f"arc {rpct[band].min():.2f}..{rpct[band].max():.2f}%")
    else:
        band = (rpct >= a.arc_lo) & (rpct <= a.arc_hi)
        print(f"reference line: {band.sum()} samples in arc "
              f"[{a.arc_lo}, {a.arc_hi}]%")
    ref_xyz = ref[band, 1:4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = SurfCore(a.map, default_config(num_envs=1, lidar_w=0, lidar_h=0))
    stem = Path(a.map).stem
    cell, gcell = vision_cells(cfg, core, stem)
    zones = load_zones(core.bsp_path)
    t_bake = time.time()
    gf = build_goal_field(core, zones["end"], cell=gcell)
    el = time.time() - t_bake
    print(f"goal field @ cell {gcell:g} in {el:.1f}s"
          + ("   ** RE-BAKE - STOP **" if el > 30 else ""))

    from surfgym.surfmask import build_surfnz
    snz, smins = build_surfnz(core, cell)
    nzf = np.abs(snz.astype(np.int16)) / 127.0
    mins = np.asarray(smins, np.float64)
    ramp, ramp_cen = ramp_voxels(nzf, None, mins, cell, ref_xyz, a.ramp_radius)
    rampd = dilate1(ramp)
    solid = nzf > 0.0                     # any face at all -> some surface
    print(f"surviving ramp: {ramp.sum():,} surfy voxels at cell {cell:g} "
          f"({ramp_cen.shape[0]:,} centres) within {a.ramp_radius:g} u of it")

    # ---- the two lidars -----------------------------------------------
    lw, lh = int(cfg["lidar_w"]), int(cfg["lidar_h"])
    hfov = float(cfg.get("lidar_hfov") or 120.0)
    vfov = float(cfg.get("lidar_vfov") or 90.0)
    rng_u = float(cfg.get("lidar_range", 2000.0))
    near = cfg.get("lidar_near")
    pot = LidarPotential.from_cfg(cfg, gf, core, device, stem)
    L_pot = GpuLidar(core, lw, lh, hfov_deg=hfov, vfov_deg=vfov,
                     range_units=rng_u, near_range=near, cell=cell,
                     device=device, potential=pot)
    L_msk = GpuLidar(core, lw, lh, hfov_deg=hfov, vfov_deg=vfov,
                     range_units=rng_u, near_range=near, cell=cell,
                     device=device, surf_mask=True)
    print(pot.describe())

    # ---- the decisions ------------------------------------------------
    dec = np.arange(0, n, act_every)
    sel = np.ones(len(dec), bool)
    if a.t0 is not None:
        sel &= t_s[dec] >= a.t0 - 1e-9
    if a.t1 is not None:
        sel &= t_s[dec] <= a.t1 + 1e-9
    dec = dec[sel]
    print(f"{len(dec)} decisions in [{t_s[dec[0]]:.2f}, {t_s[dec[-1]]:.2f}] s")

    pos = rows[dec, 1:4].astype(np.float32)
    vel = rows[dec, 4:7].astype(np.float64)
    yaw = rows[dec, 7].astype(np.float32)
    pitch = rows[dec, 12].astype(np.float32)
    duck = ((rows[dec, 8].astype(np.int64) & 4) != 0).astype(np.int32)
    spd = np.linalg.norm(vel, axis=1)
    head = np.degrees(np.arctan2(vel[:, 1], vel[:, 0])) % 360.0
    yaw_off = (yaw - head + 180.0) % 360.0 - 180.0

    def render(o, yw, pt, dk):
        o = torch.as_tensor(o, device=device)
        yw = torch.as_tensor(np.asarray(yw, np.float32), device=device)
        pt = torch.as_tensor(np.asarray(pt, np.float32), device=device)
        dk = torch.as_tensor(np.asarray(dk, np.int32), device=device)
        ip = L_pot.render(o, yw, pt, dk)
        im = L_msk.render(o, yw, pt, dk)
        assert torch.equal(ip[..., 0], im[..., 0]), \
            "the two lidars disagree on DEPTH - not the same march"
        hp, tt = hit_points(L_pot, o, yw, pt, dk, ip[..., 0])
        return (ip[..., 0].cpu().numpy(), ip[..., 1].cpu().numpy(),
                im[..., 1].cpu().numpy(), hp.cpu().numpy().astype(np.float64),
                tt.cpu().numpy())

    depth, potc, mask, hp, tt = render(pos, yaw, pitch, duck)
    eye = pos.astype(np.float64).copy()
    eye[:, 2] += np.where(duck != 0, 12.0, 17.0)

    on_ramp = lookup(rampd, mins, cell, hp)
    is_sky = tt >= rng_u - 1.0
    on_solid = lookup(dilate1(solid), mins, cell, hp) & ~is_sky

    # the ramp's ANGULAR footprint, occlusion-free
    ang_rows = []
    mask_f = mask
    for k_frame in range(len(dec)):
        k = k_frame
        dy, dp, r = cam_angles(eye[k], float(yaw[k]), float(pitch[k]),
                               ramp_cen)
        infov = (np.abs(dy) <= hfov / 2) & (np.abs(dp) <= vfov / 2) & \
                (r <= rng_u)
        col, row = pix_of(dy, dp, lw, lh, hfov, vfov)
        pixset = {}
        if infov.any():
            ci = np.clip(np.round(col[infov]).astype(int), 0, lw - 1)
            ri = np.clip(np.round(row[infov]).astype(int), 0, lh - 1)
            rr = r[infov]
            for q in range(len(ci)):
                key = (int(ri[q]), int(ci[q]))
                if key not in pixset or rr[q] < pixset[key]:
                    pixset[key] = float(rr[q])
        e = dict(n_in_fov=int(infov.sum()), n_pix_angular=len(pixset),
                 nearest_u=float(r.min()),
                 nearest_dyaw=float(dy[int(np.argmin(r))]),
                 nearest_dpitch=float(dp[int(np.argmin(r))]),
                 min_abs_dyaw=float(np.abs(dy[r <= rng_u]).min())
                 if (r <= rng_u).any() else None)
        # OCCLUSION: at the pixels the ramp WOULD fall in, what did the
        # march actually return? A hit a cell or more NEARER than the ramp
        # itself is something in front of it.
        if pixset:
            rows_i = np.array([k[0] for k in pixset])
            cols_i = np.array([k[1] for k in pixset])
            want = np.array([pixset[k] for k in pixset])
            got = tt[k_frame][rows_i, cols_i]
            occ_m = got < want - cell
            e.update(occ_px=int(occ_m.sum()),
                     occ_frac=float(occ_m.mean()),
                     occ_want_med=float(np.median(want)),
                     occ_got_med=float(np.median(got)),
                     occ_gap_med=float(np.median(want - got)),
                     occ_by_surfy_px=int((occ_m & (
                         (mask_f[k_frame][rows_i, cols_i] >= WALL_NZ) &
                         (mask_f[k_frame][rows_i, cols_i] <= GROUND_NZ))).sum()))
        ang_rows.append(e)

    # ---- the per-decision table ---------------------------------------
    tab = []
    for k in range(len(dec)):
        rm = on_ramp[k]
        d_all = depth[k]
        t_all = tt[k]
        nb = np.zeros_like(rm)
        if rm.any():
            # the angular neighbourhood of the ramp silhouette: +-3 cols,
            # +-2 rows, minus the ramp itself. That is what "beside it"
            # means in the image the policy actually gets.
            for dr in range(-2, 3):
                for dc in range(-3, 4):
                    nb |= np.roll(np.roll(rm, dr, 0), dc, 1)
            nb &= ~rm
        row = dict(
            tick=int(rows[dec[k], 0]), t=float(t_s[dec[k]]),
            arc_pct=float(pct[dec[k]]), off_line=float(off[dec[k]]),
            d_geo=float(gf.sample(rows[dec[k]:dec[k] + 1, 1:4])[0]),
            x=float(pos[k, 0]), y=float(pos[k, 1]), z=float(pos[k, 2]),
            speed=float(spd[k]), yaw=float(yaw[k]), heading=float(head[k]),
            yaw_minus_heading=float(yaw_off[k]), pitch=float(pitch[k]),
            ramp_px=int(rm.sum()),
            ramp_px_pct=float(100.0 * rm.sum() / (lw * lh)),
            sky_px=int(is_sky[k].sum()),
            solid_px=int(on_solid[k].sum()),
            depth_mean=float(d_all.mean()), depth_std=float(d_all.std()),
        )
        row.update({f"ang_{kk}": vv for kk, vv in ang_rows[k].items()})
        if rm.any():
            row.update(
                ramp_dist_min=float(t_all[rm].min()),
                ramp_dist_med=float(np.median(t_all[rm])),
                ramp_dist_max=float(t_all[rm].max()),
                ramp_depth_mean=float(d_all[rm].mean()),
                ramp_depth_std=float(d_all[rm].std()),
                ramp_mask_mean=float(mask[k][rm].mean()),
                ramp_mask_min=float(mask[k][rm].min()),
                ramp_mask_max=float(mask[k][rm].max()),
                ramp_mask_surfy_px=int(((mask[k][rm] >= WALL_NZ) &
                                        (mask[k][rm] <= GROUND_NZ)).sum()),
                ramp_pot_mean=float(potc[k][rm].mean()),
                ramp_pot_min=float(potc[k][rm].min()),
                ramp_pot_max=float(potc[k][rm].max()),
                ramp_row_lo=int(np.nonzero(rm.any(1))[0].min()),
                ramp_row_hi=int(np.nonzero(rm.any(1))[0].max()),
                ramp_col_lo=int(np.nonzero(rm.any(0))[0].min()),
                ramp_col_hi=int(np.nonzero(rm.any(0))[0].max()),
            )
        # the NOT-RIDABLE part of the neighbourhood: wall (|n_z| < 0.1) or
        # sky. That is the set the depth channel is accused of confusing
        # the ramp with, so the contrasts below are computed against it too.
        wl = nb & ((mask[k] < WALL_NZ) | is_sky[k])
        if wl.any():
            lw2 = np.concatenate([d_all[rm], d_all[wl]]) if rm.any() \
                else d_all[wl]
            row.update(
                wall_px=int(wl.sum()),
                wall_depth_mean=float(d_all[wl].mean()),
                wall_mask_mean=float(mask[k][wl].mean()),
                wall_pot_mean=float(potc[k][wl].mean()))
            if rm.any():
                row["depth_vs_wall"] = float(d_all[rm].mean() -
                                             d_all[wl].mean())
                row["depth_vs_wall_sigma"] = float(
                    (d_all[rm].mean() - d_all[wl].mean()) / (lw2.std() + 1e-9))
                lm2 = np.concatenate([mask[k][rm], mask[k][wl]])
                row["mask_vs_wall"] = float(mask[k][rm].mean() -
                                            mask[k][wl].mean())
                row["mask_vs_wall_sigma"] = float(
                    (mask[k][rm].mean() - mask[k][wl].mean()) /
                    (lm2.std() + 1e-9))
                # the direct separability question: how many of each set
                # falls in the RIDABLE band the mask channel encodes
                row["ramp_surfy_frac"] = float(
                    ((mask[k][rm] >= WALL_NZ) &
                     (mask[k][rm] <= GROUND_NZ)).mean())
                row["wall_surfy_frac"] = float(
                    ((mask[k][wl] >= WALL_NZ) &
                     (mask[k][wl] <= GROUND_NZ)).mean())
        if nb.any():
            row.update(
                nb_px=int(nb.sum()),
                nb_depth_mean=float(d_all[nb].mean()),
                nb_depth_std=float(d_all[nb].std()),
                nb_mask_mean=float(mask[k][nb].mean()),
                nb_mask_surfy_px=int(((mask[k][nb] >= WALL_NZ) &
                                      (mask[k][nb] <= GROUND_NZ)).sum()),
                nb_pot_mean=float(potc[k][nb].mean()),
            )
        if rm.any() and nb.any():
            # the crux numbers: is the ramp separable from what is beside
            # it, in each channel, in units of the LOCAL pixel spread?
            loc = np.concatenate([d_all[rm], d_all[nb]])
            row["depth_contrast"] = float(d_all[rm].mean() - d_all[nb].mean())
            row["depth_contrast_sigma"] = float(
                (d_all[rm].mean() - d_all[nb].mean()) / (loc.std() + 1e-9))
            row["depth_contrast_over_frame_sigma"] = float(
                (d_all[rm].mean() - d_all[nb].mean()) / (d_all.std() + 1e-9))
            lm = np.concatenate([mask[k][rm], mask[k][nb]])
            row["mask_contrast"] = float(mask[k][rm].mean() -
                                         mask[k][nb].mean())
            row["mask_contrast_sigma"] = float(
                (mask[k][rm].mean() - mask[k][nb].mean()) / (lm.std() + 1e-9))
            lp = np.concatenate([potc[k][rm], potc[k][nb]])
            row["pot_contrast"] = float(potc[k][rm].mean() -
                                        potc[k][nb].mean())
            row["pot_contrast_sigma"] = float(
                (potc[k][rm].mean() - potc[k][nb].mean()) / (lp.std() + 1e-9))
        tab.append(row)

    # ---- the yaw sweep -------------------------------------------------
    offs = [float(x) for x in a.yaw_offsets.split(",") if x.strip()]
    at = [float(x) for x in a.sweep_at.split(",") if x.strip()]
    if not at:
        at = [float(t_s[dec[len(dec) // 2]])]
    sweep = []
    sweep_fr = []
    for tq in at:
        k = int(np.argmin(np.abs(t_s[dec] - tq)))
        for o in offs:
            yq = np.float32((head[k] + o) % 360.0)
            dpt, dpo, dmk, dhp, dtt = render(pos[k:k + 1], [yq],
                                             pitch[k:k + 1], duck[k:k + 1])
            rm = lookup(rampd, mins, cell, dhp)[0]
            sky = (dtt[0] >= rng_u - 1.0)
            e = dict(t=float(t_s[dec[k]]), offset_deg=o,
                     yaw=float(yq), heading=float(head[k]),
                     ramp_px=int(rm.sum()), sky_px=int(sky.sum()),
                     depth_mean=float(dpt[0].mean()),
                     mask_surfy_px=int(((dmk[0] >= WALL_NZ) &
                                        (dmk[0] <= GROUND_NZ) &
                                        ~sky).sum()),
                     pot_min=float(dpo[0].min()))
            if rm.any():
                e.update(ramp_dist_min=float(dtt[0][rm].min()),
                         ramp_mask_mean=float(dmk[0][rm].mean()),
                         ramp_col_lo=int(np.nonzero(rm.any(0))[0].min()),
                         ramp_col_hi=int(np.nonzero(rm.any(0))[0].max()))
            sweep.append(e)
            sweep_fr.append(dict(t=float(t_s[dec[k]]), off=o,
                                 depth=dpt[0], pot=dpo[0], mask=dmk[0],
                                 ramp=rm, sky=sky))

    meta = dict(traj=str(a.traj), episode=ep_i, run_json=str(a.run_json),
                map=str(a.map), route=str(a.route), ref=str(a.ref),
                arc_band=[a.arc_lo, a.arc_hi], ramp_radius=a.ramp_radius,
                cell=float(cell), lidar=[lw, lh], hfov=hfov, vfov=vfov,
                range_u=rng_u, near_u=float(near) if near else rng_u,
                act_every=act_every, obs_potential=cfg.get("obs_potential"),
                ramp_voxels=int(ramp.sum()), n_decisions=int(len(dec)),
                ep_secs=float(t_s[-1] + dt[-1]), arc_max=float(pct.max()))
    (out / "m6_table.json").write_text(json.dumps(
        dict(meta=meta, rows=tab, sweep=sweep), indent=1), encoding="utf-8")
    np.savez_compressed(out / "m6_frames.npz", depth=depth, pot=potc,
                        mask=mask, on_ramp=on_ramp, sky=is_sky,
                        t=t_s[dec], yaw=yaw, pitch=pitch, head=head,
                        pos=pos, speed=spd, arc=pct[dec],
                        sw_t=np.array([f["t"] for f in sweep_fr]),
                        sw_off=np.array([f["off"] for f in sweep_fr]),
                        sw_depth=np.stack([f["depth"] for f in sweep_fr]),
                        sw_pot=np.stack([f["pot"] for f in sweep_fr]),
                        sw_mask=np.stack([f["mask"] for f in sweep_fr]),
                        sw_ramp=np.stack([f["ramp"] for f in sweep_fr]))
    print(f"wrote {out / 'm6_table.json'} and {out / 'm6_frames.npz'}")

    # ---- the console summary -------------------------------------------
    hdrline = ("    t   arc%   spd  yaw-hdg | inFOV angpx occ%  gap_u | "
               "ramp px  dist | depth r/wall  sig | mask r/wall  sig | "
               "surfy r/wall | pot r/wall")
    print("\n" + hdrline)
    print("  " + "-" * (len(hdrline) - 2))
    for r in tab:
        print("%6.2f %6.2f %6.0f %+7.1f | %5d %5d %4.0f %6.0f | %6d %6.0f | "
              "%5.3f %5.3f %+6.2f | %5.3f %5.3f %+6.2f | %5.2f %5.2f | "
              "%+5.2f %+5.2f" % (
                  r["t"], r["arc_pct"], r["speed"], r["yaw_minus_heading"],
                  r["ang_n_in_fov"], r["ang_n_pix_angular"],
                  100.0 * r.get("ang_occ_frac", float("nan")),
                  r.get("ang_occ_gap_med", float("nan")),
                  r["ramp_px"], r.get("ramp_dist_med", float("nan")),
                  r.get("ramp_depth_mean", float("nan")),
                  r.get("wall_depth_mean", float("nan")),
                  r.get("depth_vs_wall_sigma", float("nan")),
                  r.get("ramp_mask_mean", float("nan")),
                  r.get("wall_mask_mean", float("nan")),
                  r.get("mask_vs_wall_sigma", float("nan")),
                  r.get("ramp_surfy_frac", float("nan")),
                  r.get("wall_surfy_frac", float("nan")),
                  r.get("ramp_pot_mean", float("nan")),
                  r.get("wall_pot_mean", float("nan"))))
    print("\nYAW SWEEP (offset off heading(v)):")
    print("     t   off  ramp px  sky px  surfy px  nearest u")
    for e in sweep:
        print("  %5.2f %+5.0f %8d %7d %9d %10s" % (
            e["t"], e["offset_deg"], e["ramp_px"], e["sky_px"],
            e["mask_surfy_px"],
            ("%.0f" % e["ramp_dist_min"]) if "ramp_dist_min" in e else "-"))


if __name__ == "__main__":
    main()
