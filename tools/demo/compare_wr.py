"""compare_wr.py - what does the human world record do that our finisher does
not?  Compares a parsed HLDEMO (tools/demo/parse_hldemo.py output) against
one or more of our trajectory episodes on the same map.

Everything is derived from positions, velocities and inputs plus the GoldSrc
movement laws:

  * clock      - the zone clock starts when the player crosses the start
                 curtain (zones.json "start") and stops at the finish curtain
                 ("end"); our recordings are also reported on their native
                 spawn clock.
  * arc        - both paths are projected onto the champion route line
                 (maps/<map>.route.npz) with an ORDERED projection (window of
                 --window vertices ahead, 4 behind) so a fold-back cannot be
                 credited; deciles of that arc are the ledger's segmentation.
  * contact    - a physics step is "free flight" when the velocity change is
                 exactly gravity plus PM_AirAccelerate for the recorded input
                 (wishdir from yaw + keys, addspeed = min(30 - v.wishdir,
                 airaccelerate * wishspeed * dt)).  Anything else while
                 airborne is a contact (PM_ClipVelocity ran).  Contact steps
                 are grouped into events; each event reports entry and exit
                 speed, time, height change, the surface normal implied by
                 the clip (dv is parallel to n) and, if a BSP is given, the
                 surface the map actually has there (SurfCore point trace).
  * control    - pitch / yaw-rate distributions, strafe-key cadence, wishdir
                 vs velocity angle (strafe geometry), jump / duck usage.
  * last room  - dive / ramp-1 / flight split of the finish room, the finish
                 crossing height over the finish platform, plus a floor
                 heightmap of the room traced from the BSP.

Usage:
  python tools/demo/compare_wr.py --wr runs/research/wr_demo/wr_cannonball.frames.npz \
      --ours C:/RL_Surf/runs/research/xQR32/traj_7405830144.jsonl:0 \
      --champ C:/RL_Surf/runs/sISV_par2/traj_8454144000.jsonl:8 \
      --route C:/RL_Surf/maps/surf_src_cannonball.route.npz \
      --zones C:/RL_Surf/maps/surf_src_cannonball.zones.json \
      --bsp C:/RL_Surf/maps/surf_src_cannonball.bsp --out runs/research/wr_demo

CPU only; the BSP is opened through SurfCore purely for traces (no bake).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

G = 800.0
AIR_CAP = 30.0          # PM_AirAccelerate: wishspeed projection capped at 30
AIRACCEL = 100.0
WISHSPEED = 250.0       # player maxspeed (cl_*speed clamp)
IN_JUMP, IN_DUCK = 2, 4


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_episode(path: str, ep: int) -> dict:
    rows, cur, header = [], -1, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if isinstance(r, dict) and "map" in r:
                cur = r.get("episode", cur + 1)
                if cur == ep:
                    header, rows = r, []
            elif isinstance(r, list) and cur == ep:
                rows.append(r)
            elif isinstance(r, dict) and "end" in r and cur == ep:
                break
    if not rows:
        raise ValueError("episode %d not found in %s" % (ep, path))
    a = np.asarray(rows, dtype=np.float64)
    n = len(a)
    tick = header.get("tick_ms", 10) / 1000.0
    side = np.where(a[:, 14] >= 2, 1, np.where(a[:, 14] <= 0, -1, 0))
    fwd = np.where(a[:, 13] >= 2, 1, np.where(a[:, 13] <= 0, -1, 0))
    btn = a[:, 8].astype(int)
    # the yaw that drove tick t is the post-delta yaw, i.e. row t+1's state
    yaw_cmd = np.concatenate((a[1:, 7], a[-1:, 7]))
    return {
        "name": os.path.basename(path) + ":%d" % ep,
        "t": a[:, 0] * tick, "pos": a[:, 1:4].copy(), "vel": a[:, 4:7].copy(),
        "yaw": a[:, 7].copy(), "yaw_cmd": yaw_cmd, "pitch": a[:, 12].copy(),
        "og": (a[:, 9] > 0).astype(int), "jump": (btn & IN_JUMP) > 0,
        "duck": (btn & IN_DUCK) > 0, "fwd": fwd, "side": side,
        "step_dt": np.full(n - 1, tick),
        "maxvel": float(header.get("phys", {}).get("sv_maxvelocity", 4000.0)),
        "clock": "spawn", "header": header,
    }


def load_wr(path: str, maxvel: float = 4000.0) -> dict:
    """maxvel: the per-component clamp the demo's physics actually shows
    (PM_CheckVelocity); the movevars block can disagree with it."""
    d = np.load(path)
    t = d["t"].astype(np.float64)
    ang = d["viewangles"].astype(np.float64)
    ucang = d["uc_viewangles"].astype(np.float64)
    fsu = d["uc_fsu"].astype(np.float64)
    btn = d["uc_buttons"].astype(int)
    msec = d["uc_msec"].astype(np.float64)
    # measured alignment: simorg[k+1] = simorg[k] + simvel[k+1] * msec[k]/1000,
    # so the usercmd stored in frame k drove the step k -> k+1.
    return {
        "name": os.path.basename(path), "t": t,
        "pos": d["simorg"].astype(np.float64), "vel": d["simvel"].astype(np.float64),
        "yaw": ang[:, 1] % 360.0, "yaw_cmd": ucang[:, 1] % 360.0,
        "pitch": -ang[:, 0],           # simulator sign: negative = looking down
        "og": (d["onground"] != 0).astype(int),
        "jump": (btn & IN_JUMP) > 0, "duck": (btn & IN_DUCK) > 0,
        "fwd": np.sign(fsu[:, 0]).astype(int), "side": np.sign(fsu[:, 1]).astype(int),
        "step_dt": msec[:-1] / 1000.0, "maxvel": float(maxvel), "clock": "demo",
    }


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def curtain_crossings(t, pos, zone):
    """Times at which the path crosses the y-mid-plane of a thin zone box
    while inside its x/z extent.  Returns [(time, point, index)]."""
    mins, maxs = np.asarray(zone["mins"], float), np.asarray(zone["maxs"], float)
    ymid = 0.5 * (mins[1] + maxs[1])
    y = pos[:, 1]
    out = []
    for k in range(1, len(pos)):
        if (y[k - 1] - ymid) * (y[k] - ymid) <= 0 and y[k] != y[k - 1]:
            f = (ymid - y[k - 1]) / (y[k] - y[k - 1])
            q = pos[k - 1] + f * (pos[k] - pos[k - 1])
            if mins[0] <= q[0] <= maxs[0] and mins[2] <= q[2] <= maxs[2]:
                out.append((t[k - 1] + f * (t[k] - t[k - 1]), q, k))
    return out


def finish_time(track, zone):
    """Finish-curtain crossing; if the recording ends before the crossing
    (our recorder drops the terminal state), extrapolate the last row with
    its own velocity and say so."""
    hits = curtain_crossings(track["t"], track["pos"], zone)
    if hits:
        tf, q, k = hits[0]
        return tf, q, "crossed"
    p, v = track["pos"][-1], track["vel"][-1]
    ymid = 0.5 * (zone["mins"][1] + zone["maxs"][1])
    if v[1] != 0:
        dt = (ymid - p[1]) / v[1]
        if 0 <= dt <= 0.05:
            return track["t"][-1] + dt, p + v * dt, "extrapolated %.1f ms past the last row" % (dt * 1000)
    return None, None, "no crossing"


def plane_crossing(t, pos, axis, value, direction):
    """First time the path crosses pos[axis] = value moving in `direction`."""
    a = pos[:, axis]
    for k in range(1, len(pos)):
        if direction > 0 and a[k - 1] < value <= a[k] or direction < 0 and a[k - 1] > value >= a[k]:
            f = (value - a[k - 1]) / (a[k] - a[k - 1])
            return t[k - 1] + f * (t[k] - t[k - 1]), k
    return None, None


def project_ordered(pos, route, window=24, back=4):
    """Ordered projection of a path onto a polyline: (arc, lateral distance,
    segment index, total length).  Each point may only look `back` segments
    behind and `window` ahead of the previous point's segment."""
    seg = np.diff(route, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    seglen[seglen == 0] = 1e-9
    cum = np.concatenate(([0.0], np.cumsum(seglen)))
    M = len(seg)
    arc = np.zeros(len(pos))
    dist = np.zeros(len(pos))
    sidx = np.zeros(len(pos), dtype=int)
    i = None
    for k in range(len(pos)):
        p = pos[k]
        lo, hi = (0, M) if i is None else (max(0, i - back), min(M, i + window + 1))
        a = route[lo:hi]
        s = seg[lo:hi]
        L2 = seglen[lo:hi] ** 2
        f = np.clip(np.einsum("ij,ij->i", p - a, s) / L2, 0.0, 1.0)
        q = a + f[:, None] * s
        dd = np.linalg.norm(q - p, axis=1)
        j = int(np.argmin(dd))
        i = lo + j
        arc[k] = cum[i] + f[j] * seglen[i]
        dist[k] = dd[j]
        sidx[k] = i
    return arc, dist, sidx, cum[-1]


# --------------------------------------------------------------------------
# physics: contact detection and energy
# --------------------------------------------------------------------------
def wishdir_xy(track, n):
    """Horizontal wish direction for steps 0..n-1 from yaw_cmd + keys."""
    yc = np.radians(track["yaw_cmd"][:n])
    f2 = np.stack((np.cos(yc), np.sin(yc)), axis=1)
    r2 = np.stack((np.sin(yc), -np.cos(yc)), axis=1)     # AngleVectors, roll 0
    wish = f2 * track["fwd"][:n, None] + r2 * track["side"][:n, None]
    wn = np.linalg.norm(wish, axis=1)
    wd = np.zeros_like(wish)
    ok = wn > 0
    wd[ok] = wish[ok] / wn[ok, None]
    return wd, ok


def step_analysis(track):
    """Per physics step k (state k -> k+1)."""
    v0, v1 = track["vel"][:-1], track["vel"][1:]
    dt = track["step_dt"]
    n = len(dt)
    # exact free-flight prediction: gravity + PM_AirAccelerate on the input
    wd, has_wish = wishdir_xy(track, n)
    cur = np.einsum("ij,ij->i", v0[:, :2], wd)
    add = np.clip(AIR_CAP - cur, 0.0, None)
    add = np.minimum(add, AIRACCEL * WISHSPEED * dt)
    add[~has_wish] = 0.0
    dv_air = wd * add[:, None]
    vexp = v0.copy()
    vexp[:, :2] += dv_air
    vexp[:, 2] -= G * dt
    resid = v1 - vexp
    og = track["og"]
    ground = (og[:-1] > 0) | (og[1:] > 0)
    maxvel = track["maxvel"]
    clamped = ((np.abs(v1) >= maxvel - 1.0).any(axis=1) | (np.abs(v0) >= maxvel - 1.0).any(axis=1)) & ~ground
    free = (np.linalg.norm(resid[:, :2], axis=1) <= 1.0) & (np.abs(resid[:, 2]) <= 0.75) & ~ground
    contact = ~free & ~ground & ~clamped
    # energy bookkeeping (specific, u^2/s^2).  e_g = after gravity alone,
    # e_post = actual.  Every step's non-gravity energy change (e_post - e_g)
    # is attributed to exactly one bucket, so the budget closes:
    #   free step    -> "expected": the input's PM_AirAccelerate alone
    #                   (+ gain when wishdir is perpendicular, - braking when
    #                   it points past perpendicular)
    #   contact step -> "loss": destroyed on the surface (clip + whatever the
    #                   input did in the same step; the two cannot be split)
    #   clamped step -> "clamp_loss": PM_CheckVelocity's component clamp
    #   ground step  -> "ground_change"
    vg = v0.copy()
    vg[:, 2] -= G * dt
    e_g = 0.5 * np.einsum("ij,ij->i", vg, vg)
    e_post = 0.5 * np.einsum("ij,ij->i", v1, v1)
    d_e = e_post - e_g
    expected = np.where(free, d_e, 0.0)
    loss = np.where(contact, -d_e, 0.0)
    clamp_loss = np.where(clamped, -d_e, 0.0)
    ground_change = np.where(ground, d_e, 0.0)
    nrm = np.zeros_like(resid)
    mag = np.linalg.norm(resid, axis=1)
    ok = mag > 1e-6
    nrm[ok] = resid[ok] / mag[ok, None]
    return {"contact": contact, "ground": ground, "free": free, "clamped": clamped,
            "expected": expected, "loss": loss, "clamp_loss": clamp_loss,
            "ground_change": ground_change, "normal": nrm, "resid_mag": mag,
            "speed0": np.linalg.norm(v0, axis=1), "speed1": np.linalg.norm(v1, axis=1),
            "wishdir": wd, "has_wish": has_wish}


def group_runs(mask, max_gap):
    """(start, end) index pairs of True runs, merging gaps <= max_gap."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    runs, s, p = [], idx[0], idx[0]
    for k in idx[1:]:
        if k - p > max_gap + 1:
            runs.append((s, p))
            s = k
        p = k
    runs.append((s, p))
    return runs


def contact_events(track, sa, arc, merge_gap_s=0.30):
    """Group contact steps into events (a ramp ride shows as bursts of clip
    steps as the hull rides the plane)."""
    t = track["t"]
    runs = group_runs(sa["contact"], 3)
    events = []
    for s, e in runs:
        if events and t[s] - t[events[-1]["k1"] + 1] < merge_gap_s:
            events[-1]["k1"] = e
        else:
            events.append({"k0": s, "k1": e})
    out = []
    pos = track["pos"]
    for ev in events:
        k0, k1 = ev["k0"], ev["k1"]
        steps = np.arange(k0, k1 + 1)
        c = steps[sa["contact"][steps]]
        w = sa["resid_mag"][c]
        n = (sa["normal"][c] * w[:, None]).sum(axis=0)
        n = n / (np.linalg.norm(n) + 1e-9)
        out.append({
            "k0": int(k0), "k1": int(k1), "t_in": float(t[k0]), "t_out": float(t[k1 + 1]),
            "dur": float(t[k1 + 1] - t[k0]), "contact_steps": int(len(c)),
            "contact_time": float(track["step_dt"][c].sum()),
            "v_in": float(sa["speed0"][k0]), "v_out": float(sa["speed1"][k1]),
            "z_in": float(pos[k0, 2]), "z_out": float(pos[k1 + 1, 2]),
            "z_min": float(pos[k0:k1 + 2, 2].min()),
            "arc_in": float(arc[k0]), "arc_out": float(arc[k1 + 1]),
            "e_loss": float(sa["loss"][c].sum()),
            "normal": n.tolist(), "n_z": float(n[2]),
            "mid": pos[c[len(c) // 2]].tolist(), "k_mid": int(c[len(c) // 2]),
            "vz_in": float(track["vel"][k0, 2]), "vz_out": float(track["vel"][k1 + 1, 2]),
        })
    return out


def bsp_check(core, track, events):
    """For each event, point-trace from the mid-contact origin along -n and
    record what the map has there (the hull centre sits 16-36 u off the
    plane it is riding, so a hit within ~80 u with the same normal means the
    ramp exists in this BSP)."""
    for ev in events:
        p = track["pos"][ev["k_mid"]]
        n = np.asarray(ev["normal"])
        tr = core.trace(p, p - n * 200.0, 2)
        ev["bsp_hit"] = bool(tr.fraction < 1.0 and not tr.startsolid)
        ev["bsp_dist"] = float(tr.fraction * 200.0)
        ev["bsp_normal"] = [float(v) for v in tr.normal]
        ev["bsp_angle"] = float(math.degrees(math.acos(np.clip(np.dot(n, ev["bsp_normal"]), -1, 1)))) if ev["bsp_hit"] else None
        ev["bsp_startsolid"] = bool(tr.startsolid)


# --------------------------------------------------------------------------
# control statistics
# --------------------------------------------------------------------------
def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))


def control_stats(track, sa, lo, hi):
    """Statistics over the timed window [lo, hi) of state indices."""
    sl = slice(lo, hi)
    st = slice(lo, min(hi, len(track["step_dt"])))
    dt = track["step_dt"][st]
    T = dt.sum()
    pitch = track["pitch"][sl]
    yaw = unwrap_deg(track["yaw"][sl])
    yr = np.abs(np.diff(yaw)) / dt[:len(yaw) - 1]
    side = track["side"][st]
    fwd = track["fwd"][st]
    nz = side != 0
    flips = int(np.sum((side[1:] != side[:-1]) & (side[1:] != 0) & (side[:-1] != 0)))
    presses_side = int(np.sum((side[1:] != 0) & (side[:-1] == 0)))
    holds = []
    b = 0
    for k in range(1, len(side) + 1):
        if k == len(side) or side[k] != side[b]:
            if side[b] != 0:
                holds.append(dt[b:k].sum())
            b = k
    holds = np.asarray(holds) if holds else np.zeros(1)
    # wishdir vs velocity angle on free-flight steps (strafe geometry)
    wd, ok = sa["wishdir"][st], sa["has_wish"][st]
    v = track["vel"][lo:lo + len(dt), :2]
    vn = np.linalg.norm(v, axis=1)
    m = ok & (vn > 100) & sa["free"][st]
    cosang = np.einsum("ij,ij->i", wd[m], v[m]) / vn[m]
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    exp_ = sa["expected"][st]
    airborne_free = sa["free"][st]
    gain = exp_.sum()
    gain_pos = exp_[exp_ > 0].sum()
    braking = exp_[exp_ < 0].sum()
    brake_frac = float(np.mean(exp_[airborne_free] < -1.0)) if airborne_free.any() else 0.0
    brake_mean = float(exp_[exp_ < -1.0].mean()) if (exp_ < -1.0).any() else 0.0
    loss = sa["loss"][st].sum()
    clamp_loss = sa["clamp_loss"][st].sum()
    ground_change = sa["ground_change"][st].sum()
    # wishdir-velocity angle bins over free-flight steps with a key held:
    # under-rotated (<89.5: partial gain), perfect (89.5-90.5), braking bands
    edges = [0, 85, 89.5, 90.5, 92, 95, 180]
    hist = np.histogram(ang, bins=edges)[0] / max(len(ang), 1)
    jump = track["jump"][st]
    duck = track["duck"][st]
    presses = lambda b: int(np.sum(b[1:] & ~b[:-1])) if len(b) > 1 else 0
    contact_t = dt[sa["contact"][st]].sum()
    ground_t = dt[sa["ground"][st]].sum()
    sp = np.linalg.norm(track["vel"][sl], axis=1)
    return {
        "seconds": float(T),
        "pitch_mean": float(pitch.mean()), "pitch_median": float(np.median(pitch)),
        "pitch_p10": float(np.percentile(pitch, 10)), "pitch_p90": float(np.percentile(pitch, 90)),
        "pitch_min": float(pitch.min()), "pitch_max": float(pitch.max()),
        "yawrate_mean": float(yr.mean()), "yawrate_median": float(np.median(yr)),
        "yawrate_p90": float(np.percentile(yr, 90)), "yawrate_max": float(yr.max()),
        "side_active_frac": float(np.mean(nz)), "fwd_active_frac": float(np.mean(fwd != 0)),
        "both_active_frac": float(np.mean(nz & (fwd != 0))), "none_active_frac": float(np.mean(~nz & (fwd == 0))),
        "back_active_frac": float(np.mean(fwd < 0)),
        "side_flips_per_s": float(flips / T), "side_presses_per_s": float((flips + presses_side) / T),
        "hold_median_s": float(np.median(holds)),
        "hold_mean_s": float(holds.mean()), "hold_p90_s": float(np.percentile(holds, 90)),
        "wish_vel_angle_median": float(np.median(ang)) if len(ang) else None,
        "wish_vel_angle_p10": float(np.percentile(ang, 10)) if len(ang) else None,
        "wish_vel_angle_p90": float(np.percentile(ang, 90)) if len(ang) else None,
        "wish_vel_angle_frac_in_80_100": float(np.mean((ang > 80) & (ang < 100))) if len(ang) else None,
        "wish_vel_angle_frac_over_100": float(np.mean(ang >= 100)) if len(ang) else None,
        "strafe_gain_per_s": float(gain / T), "strafe_gain_total": float(gain),
        "strafe_gain_pos": float(gain_pos), "strafe_braking": float(braking),
        "braking_step_frac": brake_frac, "braking_step_mean": brake_mean,
        "ang_lt85": float(hist[0]), "ang_85_895": float(hist[1]), "ang_895_905": float(hist[2]),
        "ang_905_92": float(hist[3]), "ang_92_95": float(hist[4]), "ang_gt95": float(hist[5]),
        "clamp_loss_total": float(clamp_loss), "ground_change_total": float(ground_change),
        "clamp_time_frac": float(dt[sa["clamped"][st]].sum() / T),
        "jump_presses": presses(jump), "jump_held_frac": float(jump.mean()),
        "duck_presses": presses(duck), "duck_held_frac": float(duck.mean()),
        "contact_time_frac": float(contact_t / T), "ground_time_frac": float(ground_t / T),
        "free_time_frac": float(1.0 - contact_t / T - ground_t / T),
        "speed_mean": float(sp.mean()), "speed_max": float(sp.max()),
        "vcomp_max": float(np.abs(track["vel"][sl]).max()),
        "contact_loss_total": float(loss),
    }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def first_passage(arc, t, grid):
    """Time at which the running-max arc first reaches each grid value."""
    am = np.maximum.accumulate(arc)
    idx = np.searchsorted(am, grid, side="left")
    out = np.full(len(grid), np.nan)
    ok = idx < len(am)
    out[ok] = t[idx[ok]]
    return out


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return ("%%.%df" % nd) % x


def floor_heightmap(core, x0, x1, y0, y1, ztop, zbot, cell=100.0):
    xs = np.arange(x0, x1 + 1, cell)
    ys = np.arange(y0, y1 + 1, cell)
    H = np.full((len(ys), len(xs)), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            tr = core.trace((x, y, ztop), (x, y, zbot), 2)
            if tr.fraction < 1.0 and not tr.startsolid:
                H[j, i] = tr.endpos[2]
    return xs, ys, H


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wr", required=True, help="<stem>.frames.npz from parse_hldemo.py")
    ap.add_argument("--ours", required=True, help="traj.jsonl:episode")
    ap.add_argument("--champ", default=None, help="traj.jsonl:episode (optional third line)")
    ap.add_argument("--route", required=True)
    ap.add_argument("--zones", required=True)
    ap.add_argument("--bsp", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--wr-label", default="WR")
    ap.add_argument("--ours-label", default="ours")
    ap.add_argument("--champ-label", default="champ")
    ap.add_argument("--room-x", default="-14700,-2000", help="finish-room x range for the heightmap")
    ap.add_argument("--room-y", default="-7500,7600")
    ap.add_argument("--room-z", default="-700,-6500", help="trace from z_top down to z_bottom")
    ap.add_argument("--room-entry-x", type=float, default=-6000.0,
                    help="x plane whose crossing (moving -x, after 80%% of the route) marks the finish-room entry")
    ap.add_argument("--wr-maxvel", type=float, default=4000.0,
                    help="per-component velocity clamp the demo physics actually shows")
    ap.add_argument("--ramp2-y", type=float, default=6000.0,
                    help="contacts with y beyond this are on the ramp under the finish wall (ramp 2)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    zones = json.load(open(args.zones))
    route = np.load(args.route)["route"].astype(np.float64)
    platform_z = float(zones["end"]["mins"][2])

    tracks = {}
    tracks[args.wr_label] = load_wr(args.wr, args.wr_maxvel)
    p, e = args.ours.rsplit(":", 1)
    tracks[args.ours_label] = load_episode(p, int(e))
    if args.champ:
        p, e = args.champ.rsplit(":", 1)
        tracks[args.champ_label] = load_episode(p, int(e))
    names = list(tracks.keys())

    core = None
    if args.bsp:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
        from surfgym.core import SurfCore, default_config
        core = SurfCore(args.bsp, default_config(num_envs=1, lidar_w=0, lidar_h=0))

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    results = {}

    # ---- clocks ----------------------------------------------------------
    say("## Timing (zone clock = start curtain -> finish curtain)")
    say("")
    say("| run | start crossing | finish crossing | zone time | native clock |")
    say("|---|---|---|---|---|")
    for name, tr in tracks.items():
        starts = curtain_crossings(tr["t"], tr["pos"], zones["start"])
        tf, qf, how = finish_time(tr, zones["end"])
        t0 = starts[-1][0] if starts else tr["t"][0]
        tr["t0"], tr["tf"], tr["finish_pos"], tr["finish_how"] = t0, tf, qf, how
        native = "%.2f s from spawn" % (tf - tr["t"][0]) if (tf is not None and tr["clock"] == "spawn") else (
            "%.2f s of playback before the curtain, %.2f s total" % (t0 - tr["t"][0], tr["t"][-1] - tr["t"][0]))
        say("| %s | %.3f s (%s) | %s (%s) | **%s s** | %s |" % (
            name, t0, "x=%.0f z=%.0f" % (starts[-1][1][0], starts[-1][1][2]) if starts else "none",
            fmt(tf, 3) + " s" if tf else "-", how,
            fmt(tf - t0, 2) if tf else "-", native))
        results[name] = {"t0": float(t0), "tf": float(tf) if tf else None, "finish_how": how,
                         "finish_pos": [float(v) for v in qf] if qf is not None else None}
    say("")

    # ---- projection, contacts --------------------------------------------
    for name, tr in tracks.items():
        arc, lat, sidx, L = project_ordered(tr["pos"], route, window=args.window)
        tr["arc"], tr["lat"], tr["L"] = arc, lat, L
        sa = step_analysis(tr)
        tr["sa"] = sa
        ev = contact_events(tr, sa, arc)
        if core is not None:
            bsp_check(core, tr, ev)
        tr["events"] = ev
        k0 = int(np.searchsorted(tr["t"], tr["t0"]))
        k1 = int(np.searchsorted(tr["t"], tr["tf"])) if tr["tf"] else len(tr["t"])
        tr["k0"], tr["k1"] = k0, k1
        tr["ctl"] = control_stats(tr, sa, k0, k1)
        tr["timed_events"] = [e for e in ev if k0 <= e["k0"] < k1]
        if core is not None:
            sub = tr["pos"][k0:k1:5]
            solid = sum(1 for q in sub if core.point_contents(q) != -1)
            tr["origins_in_solid"] = (solid, len(sub))

    L = tracks[args.wr_label]["L"]
    say("Route length %.0f u; ordered projection window %d vertices; free flight = gravity + exact PM_AirAccelerate on the recorded input." % (L, args.window))
    say("Path length flown (timed window): " + ", ".join(
        "%s %.0f u" % (name, np.linalg.norm(np.diff(tr["pos"][tr["k0"]:tr["k1"]], axis=0), axis=1).sum()) for name, tr in tracks.items()) + ".")
    for name, tr in tracks.items():
        results[name]["path_length"] = float(np.linalg.norm(np.diff(tr["pos"][tr["k0"]:tr["k1"]], axis=0), axis=1).sum())
        st = slice(tr["k0"], tr["k1"] - 1)
        air = ~tr["sa"]["ground"][st]
        say("%s: %.1f%% of airborne steps are exact free flight, %.1f%% are contacts (%d events, %d with loss > 0.1M)." % (
            name, 100 * np.mean(tr["sa"]["free"][st][air]), 100 * np.mean(tr["sa"]["contact"][st][air]),
            len(tr["timed_events"]), sum(1 for e in tr["timed_events"] if e["e_loss"] > 1e5)))
    if core is not None:
        for name, tr in tracks.items():
            s, n = tr["origins_in_solid"]
            big = [e for e in tr["timed_events"] if e["contact_time"] >= 0.05]
            hit = [e for e in big if e.get("bsp_hit") and e["bsp_dist"] <= 80]
            agree = [e for e in hit if e["bsp_angle"] is not None and e["bsp_angle"] < 20]
            say("%s in this BSP: %d of %d sampled origins inside solid; of %d events with >= 50 ms of contact, %d have a surface within 80 u along -n and %d of those with the same normal (< 20 deg)." % (
                name, s, n, len(big), len(hit), len(agree)))
    say("")

    # ---- decile split ------------------------------------------------------
    say("## Time split by route decile (zone clock; arc along the champion line; decile 10 runs to the finish crossing)")
    say("")
    grid = np.linspace(0, L, 11)
    hdr = "| decile | arc (ku) | " + " | ".join("%s split (s)" % n for n in names) + " | gap %s-%s | " % (names[1], names[0]) + \
          " | ".join("%s v (u/s)" % n for n in names) + " | " + " | ".join("%s loss (M)" % n for n in names) + " |"
    say(hdr)
    say("|" + "---|" * (hdr.count("|") - 1))
    fp = {}
    for name, tr in tracks.items():
        f = first_passage(tr["arc"], tr["t"], grid) - tr["t0"]
        f[10] = tr["tf"] - tr["t0"] if tr["tf"] else np.nan
        fp[name] = f
    dec_rows = []
    for d in range(10):
        row = {"decile": d + 1, "arc0": grid[d], "arc1": grid[d + 1]}
        for name, tr in tracks.items():
            row[name + "_t"] = fp[name][d + 1]
            row[name + "_split"] = fp[name][d + 1] - fp[name][d]
            row[name + "_loss"] = sum(e["e_loss"] for e in tr["timed_events"] if grid[d] <= e["arc_in"] < grid[d + 1] or (d == 9 and e["arc_in"] >= grid[d]))
            row[name + "_v"] = (grid[d + 1] - grid[d]) / row[name + "_split"] if row[name + "_split"] and not math.isnan(row[name + "_split"]) else float("nan")
        gap = row[names[1] + "_split"] - row[names[0] + "_split"]
        say("| %d | %.0f-%.0f | %s | %s | %s | %s |" % (
            d + 1, grid[d] / 1000, grid[d + 1] / 1000,
            " | ".join(fmt(row[n + "_split"], 2) for n in names), fmt(gap, 2),
            " | ".join(fmt(row[n + "_v"], 0) if d < 9 else "-" for n in names),
            " | ".join(fmt(row[n + "_loss"] / 1e6, 2) for n in names)))
        dec_rows.append(row)
    say("| total | 0-%.0f | %s | %s | %s | %s |" % (
        L / 1000, " | ".join(fmt(fp[n][10], 2) for n in names), fmt(fp[names[1]][10] - fp[names[0]][10], 2),
        " | ".join(fmt(tr["ctl"]["speed_mean"], 0) for tr in tracks.values()),
        " | ".join(fmt(tr["ctl"]["contact_loss_total"] / 1e6, 2) for tr in tracks.values())))
    say("")
    say("Cumulative zone-clock time at each decile boundary:")
    say("")
    say("| boundary | " + " | ".join(names) + " | %s behind %s |" % (names[1], names[0]))
    say("|---|" + "---|" * (len(names) + 1))
    for d in range(1, 11):
        say("| %d0%% | %s | %s |" % (d, " | ".join(fmt(fp[n][d], 2) for n in names), fmt(fp[names[1]][d] - fp[names[0]][d], 2)))
    say("")
    results["deciles"] = dec_rows

    # ---- contact events ----------------------------------------------------
    say("## Contact events (clip steps grouped; n_z = implied surface normal z, ramp < 0.7; bsp = distance to the map surface along -n / normal disagreement)")
    say("")
    for name, tr in tracks.items():
        ev = tr["timed_events"]
        say("### %s: %d events; contact %.1f%% of run time, free flight %.1f%%, ground %.1f%%; total clip loss %.2f M, strafe gain %.2f M" % (
            name, len(ev), tr["ctl"]["contact_time_frac"] * 100, tr["ctl"]["free_time_frac"] * 100, tr["ctl"]["ground_time_frac"] * 100,
            tr["ctl"]["contact_loss_total"] / 1e6, tr["ctl"]["strafe_gain_total"] / 1e6))
        say("")
        say("| t_in (s) | arc (ku) | dur (s) | contact (s) | v_in | v_out | vz_in | vz_out | z_in | z_out | z_min | loss (M) | n_z | bsp |")
        say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for e in ev:
            if e["contact_time"] < 0.03 and e["e_loss"] < 2e4:
                continue
            say("| %.2f | %.1f | %.2f | %.2f | %.0f | %.0f | %+.0f | %+.0f | %.0f | %.0f | %.0f | %.2f | %.2f | %s |" % (
                e["t_in"] - tr["t0"], e["arc_in"] / 1000, e["dur"], e["contact_time"], e["v_in"], e["v_out"],
                e["vz_in"], e["vz_out"], e["z_in"], e["z_out"], e["z_min"], e["e_loss"] / 1e6, e["n_z"],
                ("%.0fu/%.0fdeg" % (e["bsp_dist"], e["bsp_angle"])) if e.get("bsp_hit") else ("solid" if e.get("bsp_startsolid") else "miss")))
        say("")
    results["events"] = {name: tr["timed_events"] for name, tr in tracks.items()}

    # ---- ramp-by-ramp match (ours -> WR by arc) -----------------------------
    say("## Ramp by ramp: our contact events (>= 50 ms) matched to the nearest WR event by arc (within 2.5 ku)")
    say("")
    wr = tracks[args.wr_label]
    ours = tracks[args.ours_label]
    wr_ev = [e for e in wr["timed_events"] if e["contact_time"] >= 0.02]
    our_ev = [e for e in ours["timed_events"] if e["contact_time"] >= 0.05]
    say("| arc (ku) | ours v_in->v_out | ours contact (s) | ours dz | ours loss (M) | ours lat (u) | WR arc | WR v_in->v_out | WR contact (s) | WR dz | WR loss (M) | WR at that arc: speed / lat (u) / z - ours z | WR lead at entry (s) |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    matched, used = [], set()
    wr_am = np.maximum.accumulate(wr["arc"])
    wr_max_arc = wr_am[wr["k1"] - 1]
    for e in our_ev:
        cand = [(abs(w["arc_in"] - e["arc_in"]), i, w) for i, w in enumerate(wr_ev) if i not in used and abs(w["arc_in"] - e["arc_in"]) < 2500]
        if e["arc_in"] <= wr_max_arc:
            kw = min(int(np.searchsorted(wr_am, e["arc_in"])), len(wr["t"]) - 1)
            wr_state = "%.0f / %.0f / %+.0f" % (np.linalg.norm(wr["vel"][kw]), wr["lat"][kw], wr["pos"][kw, 2] - e["z_in"])
            lead = fmt((e["t_in"] - ours["t0"]) - (wr["t"][kw] - wr["t0"]), 2)
        else:
            wr_state, lead = "beyond WR's last projected arc", "-"
        ours_lat = ours["lat"][e["k0"]]
        if cand:
            _, i, w = min(cand)
            used.add(i)
            matched.append((e, w))
            say("| %.1f | %.0f->%.0f | %.2f | %+.0f | %.2f | %.0f | %.1f | %.0f->%.0f | %.2f | %+.0f | %.2f | %s | %s |" % (
                e["arc_in"] / 1000, e["v_in"], e["v_out"], e["contact_time"], e["z_out"] - e["z_in"], e["e_loss"] / 1e6, ours_lat,
                w["arc_in"] / 1000, w["v_in"], w["v_out"], w["contact_time"], w["z_out"] - w["z_in"], w["e_loss"] / 1e6, wr_state, lead))
        else:
            say("| %.1f | %.0f->%.0f | %.2f | %+.0f | %.2f | %.0f | none | (WR flies through) | | | | %s | %s |" % (
                e["arc_in"] / 1000, e["v_in"], e["v_out"], e["contact_time"], e["z_out"] - e["z_in"], e["e_loss"] / 1e6, ours_lat, wr_state, lead))
    unmatched_wr = [w for i, w in enumerate(wr_ev) if i not in used and w["contact_time"] >= 0.05]
    if unmatched_wr:
        say("")
        say("WR events (>= 50 ms) with no counterpart in our run: " + "; ".join(
            "arc %.1f ku (t %.1f s, %.0f->%.0f u/s, %.2f s, loss %.2f M, n_z %.2f)" % (
                w["arc_in"] / 1000, w["t_in"] - wr["t0"], w["v_in"], w["v_out"], w["contact_time"], w["e_loss"] / 1e6, w["n_z"]) for w in unmatched_wr))
    say("")
    results["matched"] = [(e, w) for e, w in matched]

    # ---- last room -----------------------------------------------------------
    say("## Finish room: dive -> ramp 1 -> flight -> finish curtain (finish platform floor z = %.0f)" % platform_z)
    say("")
    say("| run | room entry (x=%.0f) | ramp-1 first contact | ramp-1 phase (first contact -> last exit) | v at first contact | v / vz at last exit | z_min | last exit -> finish | apex z | finish z (over platform) | finish speed / vz | zone time |" % args.room_entry_x)
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    room = {}
    for name, tr in tracks.items():
        # the finish room is entered after 80% of the route; look only there
        k80 = int(np.searchsorted(np.maximum.accumulate(tr["arc"]), 0.8 * L))
        k80 = max(tr["k0"], min(k80, tr["k1"] - 1))
        te, ke = plane_crossing(tr["t"][k80:tr["k1"]], tr["pos"][k80:tr["k1"]], 0, args.room_entry_x, -1)
        if te is None:
            say("| %s | no crossing | | | | | | | | | | |" % name)
            continue
        ke += k80
        # ramp 1 = every sloped contact in the pit (well below the finish
        # platform) after the room entry; ramp 2 = contacts under the finish
        # wall (y beyond --ramp2-y)
        pit = [e for e in tr["timed_events"] if e["k0"] >= ke and e["n_z"] < 0.95 and e["contact_time"] >= 0.02
               and e["z_min"] < platform_z - 1500 and e["mid"][1] < args.ramp2_y]
        r2 = [e for e in tr["timed_events"] if e["k0"] >= ke and e["contact_time"] >= 0.02 and e["mid"][1] >= args.ramp2_y]
        if not pit:
            say("| %s | %.2f | no ramp contact after entry | | | | | | | | | |" % (name, te - tr["t0"]))
            continue
        first, last = pit[0], pit[-1]
        kx = last["k1"] + 1
        kf = min(tr["k1"], len(tr["t"]) - 1)
        apex = tr["pos"][kx:kf + 1, 2].max() if kf > kx else float("nan")
        fz = tr["finish_pos"][2] if tr["finish_pos"] is not None else float("nan")
        room[name] = {"entry": te - tr["t0"], "dive": first["t_in"] - te, "r1_in": first["t_in"] - tr["t0"],
                      "r1_x_in": first["mid"][0], "r1_x_out": tr["pos"][kx, 0], "r1_phase": last["t_out"] - first["t_in"],
                      "r1_contact": sum(e["contact_time"] for e in pit), "r1_touches": len(pit),
                      "r1_v_in": first["v_in"], "r1_v_out": last["v_out"], "r1_vz_out": last["vz_out"],
                      "z_min": min(e["z_min"] for e in pit), "r1_loss": sum(e["e_loss"] for e in pit),
                      "flight": (tr["tf"] - last["t_out"]) if tr["tf"] else None,
                      "apex": apex, "finish_z": fz, "finish_speed": float(np.linalg.norm(tr["vel"][kf])),
                      "finish_vz": float(tr["vel"][kf, 2]), "ramp2": r2}
        r = room[name]
        say("| %s | %.2f s | %.2f s (x=%.0f) | %.2f s, %d touch(es), %.2f s in contact, x %.0f -> %.0f | %.0f | %.0f / %+.0f | %.0f | %s | %.0f | %.0f (%+.0f u) | %.0f / %+.0f | %s |" % (
            name, te - tr["t0"], r["r1_in"], r["r1_x_in"], r["r1_phase"], r["r1_touches"], r["r1_contact"], r["r1_x_in"], r["r1_x_out"],
            r["r1_v_in"], r["r1_v_out"], r["r1_vz_out"], r["z_min"], fmt(r["flight"], 2) + " s", apex, fz, fz - platform_z,
            r["finish_speed"], r["finish_vz"], fmt(tr["tf"] - tr["t0"], 2) if tr["tf"] else "-"))
    say("")
    results["room"] = {n: {k: v for k, v in r.items() if k != "ramp2"} for n, r in room.items()}
    say("Ramp 2 (the ramp under the finish wall, y >= %.0f) contact:" % args.ramp2_y)
    for name, tr in tracks.items():
        if name not in room:
            continue
        late = room[name]["ramp2"]
        say("- %s: %s" % (name, "none - flies from ramp 1 straight into the finish curtain" if not late else "; ".join(
            "t %.2f s at (%.0f, %.0f, %.0f) %.2f s n_z %.2f loss %.2f M" % (e["t_in"] - tr["t0"], e["mid"][0], e["mid"][1], e["mid"][2], e["contact_time"], e["n_z"], e["e_loss"] / 1e6) for e in late)))
    say("")

    # ---- control -------------------------------------------------------------
    say("## Control (timed window)")
    say("")
    keys = [("seconds", "seconds", 2), ("speed_mean", "mean speed (u/s)", 0), ("speed_max", "max speed (u/s)", 0),
            ("vcomp_max", "max |v component| (u/s)", 0),
            ("pitch_mean", "pitch mean (deg, neg = down)", 1), ("pitch_median", "pitch median", 1),
            ("pitch_p10", "pitch p10", 1), ("pitch_p90", "pitch p90", 1), ("pitch_min", "pitch min", 1), ("pitch_max", "pitch max", 1),
            ("yawrate_median", "|yaw rate| median (deg/s)", 0), ("yawrate_mean", "|yaw rate| mean", 0),
            ("yawrate_p90", "|yaw rate| p90", 0), ("yawrate_max", "|yaw rate| max", 0),
            ("side_active_frac", "A/D held (frac of steps)", 3), ("fwd_active_frac", "W/S held", 3),
            ("both_active_frac", "W/S and A/D together", 3), ("none_active_frac", "no movement key", 3), ("back_active_frac", "S held", 3),
            ("side_flips_per_s", "direct A<->D flips per second", 2), ("side_presses_per_s", "A/D key-downs per second", 2),
            ("hold_median_s", "strafe hold median (s)", 3),
            ("hold_mean_s", "strafe hold mean (s)", 3), ("hold_p90_s", "strafe hold p90 (s)", 3),
            ("wish_vel_angle_median", "wishdir-velocity angle median (deg, free flight)", 1),
            ("wish_vel_angle_p10", "  p10", 1), ("wish_vel_angle_p90", "  p90", 1),
            ("ang_lt85", "  frac < 85 deg (under-rotated, partial gain)", 3), ("ang_85_895", "  frac 85-89.5", 3),
            ("ang_895_905", "  frac 89.5-90.5 (full gain)", 3), ("ang_905_92", "  frac 90.5-92 (mild braking)", 3),
            ("ang_92_95", "  frac 92-95 (hard braking)", 3), ("ang_gt95", "  frac > 95 (very hard braking)", 3),
            ("braking_step_mean", "  mean energy per braking step (u^2/s^2)", 0),
            ("strafe_gain_per_s", "air-strafe net energy per s (u^2/s^3)", 0),
            ("strafe_gain_total", "air-strafe net energy total", 0),
            ("strafe_gain_pos", "  of which gain (wishdir within +-0.5 deg of perpendicular)", 0),
            ("strafe_braking", "  of which braking (wishdir past perpendicular)", 0),
            ("braking_step_frac", "  frac of free-flight steps that brake", 3),
            ("contact_loss_total", "contact energy loss total", 0),
            ("clamp_loss_total", "maxvelocity clamp loss total", 0), ("clamp_time_frac", "time at the component clamp (frac)", 3),
            ("free_time_frac", "free flight (frac of time)", 3), ("contact_time_frac", "ramp/wall contact", 3),
            ("ground_time_frac", "on ground", 3),
            ("jump_presses", "jump presses", 0), ("jump_held_frac", "jump held frac", 3),
            ("duck_presses", "duck presses", 0), ("duck_held_frac", "duck held frac", 3)]
    say("| quantity | " + " | ".join(names) + " |")
    say("|---|" + "---|" * len(names))
    for k, label, nd in keys:
        say("| %s | %s |" % (label, " | ".join(fmt(tr["ctl"][k], nd) for tr in tracks.values())))
    say("")
    results["control"] = {name: tr["ctl"] for name, tr in tracks.items()}

    # ---- energy budget --------------------------------------------------------
    say("## Energy budget (specific energy, timed window; M = 1e6 u^2/s^2)")
    say("")
    say("| run | KE at start | gravity supplied | strafe gain (+) | strafe braking (-) | destroyed at ramps | maxvelocity clamp | ground | KE at finish | closure |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for name, tr in tracks.items():
        c = tr["ctl"]
        z0 = tr["pos"][tr["k0"], 2]
        kf = min(tr["k1"], len(tr["vel"]) - 1)
        zf = tr["pos"][kf, 2]
        ke0 = 0.5 * np.sum(tr["vel"][tr["k0"]] ** 2)
        kef = 0.5 * np.sum(tr["vel"][kf] ** 2)
        grav = G * (z0 - zf)
        pred = ke0 + grav + c["strafe_gain_pos"] + c["strafe_braking"] - c["contact_loss_total"] - c["clamp_loss_total"] + c["ground_change_total"]
        say("| %s | %.2f | %.2f | %+.2f | %+.2f | %.2f | %.2f | %+.2f | %.2f | %+.2f |" % (
            name, ke0 / 1e6, grav / 1e6, c["strafe_gain_pos"] / 1e6, c["strafe_braking"] / 1e6, c["contact_loss_total"] / 1e6,
            c["clamp_loss_total"] / 1e6, c["ground_change_total"] / 1e6, kef / 1e6, (kef - pred) / 1e6))
    say("")
    say("(closure = KE at finish minus the sum of the other columns; the residual is the per-step gravity discretisation, ~0.5*(g*dt)^2 per step)")
    say("")

    with open(os.path.join(args.out, "tables.md"), "w", encoding="ascii", errors="replace") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "compare_wr.json"), "w", encoding="ascii") as f:
        json.dump(clean(results), f, indent=1)

    # ---- plots -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {args.wr_label: "#1f77b4", args.ours_label: "#d62728", args.champ_label: "#7f7f7f"}

    fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for name, tr in tracks.items():
        sl = slice(tr["k0"], tr["k1"])
        sp = np.linalg.norm(tr["vel"][sl], axis=1)
        ax[0].plot(tr["arc"][sl] / 1000, sp, lw=0.8, color=colors[name], label=name)
        c = tr["sa"]["contact"][tr["k0"]:tr["k1"] - 1]
        ax[0].scatter(tr["arc"][tr["k0"]:tr["k1"] - 1][c] / 1000, sp[:-1][c], s=3, color=colors[name], alpha=0.4)
        ax[2].plot(tr["arc"][sl] / 1000, tr["pos"][sl, 2], lw=0.8, color=colors[name], label=name)
    ax[0].set_ylabel("speed (u/s)"); ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3)
    ax[0].set_title("speed vs arc along the champion line (dots = contact steps)")
    g2 = np.linspace(0, L, 400)
    base = first_passage(wr["arc"], wr["t"], g2) - wr["t0"]
    for name, tr in tracks.items():
        if name == args.wr_label:
            continue
        tt = first_passage(tr["arc"], tr["t"], g2) - tr["t0"]
        ax[1].plot(g2 / 1000, tt - base, color=colors[name], label="%s minus %s" % (name, args.wr_label))
    for d in range(1, 10):
        ax[1].axvline(grid[d] / 1000, color="k", lw=0.3, alpha=0.4)
    ax[1].set_ylabel("time behind WR (s)"); ax[1].legend(loc="upper left"); ax[1].grid(alpha=0.3)
    ax[1].set_title("cumulative gap (undefined past the WR's last projected arc: its finish line lies off the route)")
    ax[2].set_ylabel("z (u)"); ax[2].set_xlabel("arc along champion route (ku)"); ax[2].grid(alpha=0.3)
    ax2 = ax[2].twinx()
    for name, tr in tracks.items():
        sl = slice(tr["k0"], tr["k1"])
        ax2.plot(tr["arc"][sl] / 1000, tr["lat"][sl], lw=0.5, ls=":", color=colors[name])
    ax2.set_ylabel("lateral distance to route (u, dotted)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "speed_vs_arc.png"), dpi=110)
    plt.close(fig)

    # map views + finish room heightmap
    rx = [float(v) for v in args.room_x.split(",")]
    ry = [float(v) for v in args.room_y.split(",")]
    rz = [float(v) for v in args.room_z.split(",")]
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for name, tr in tracks.items():
        sl = slice(tr["k0"], tr["k1"])
        ax[0].plot(tr["pos"][sl, 0], tr["pos"][sl, 1], lw=0.8, color=colors[name], label=name)
    ax[0].plot(route[:, 0], route[:, 1], lw=0.4, color="k", alpha=0.4, label="route")
    ax[0].set_aspect("equal"); ax[0].set_title("top view (x, y), timed window"); ax[0].legend(); ax[0].grid(alpha=0.3)
    if core is not None:
        xs, ys, H = floor_heightmap(core, rx[0], rx[1], ry[0], ry[1], rz[0], rz[1])
        im = ax[1].imshow(H, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="terrain", alpha=0.75, aspect="equal")
        fig.colorbar(im, ax=ax[1], fraction=0.04, label="floor z traced from z=%.0f (u)" % rz[0])
    for name, tr in tracks.items():
        m = np.zeros(len(tr["t"]), bool)
        k85 = int(np.searchsorted(np.maximum.accumulate(tr["arc"]), 0.85 * L))
        m[max(tr["k0"], k85):tr["k1"]] = True
        m &= (tr["pos"][:, 0] > rx[0]) & (tr["pos"][:, 0] < rx[1]) & (tr["pos"][:, 1] > ry[0]) & (tr["pos"][:, 1] < ry[1])
        ax[1].plot(tr["pos"][m, 0], tr["pos"][m, 1], lw=1.2, color=colors[name], label=name)
        ax[2].plot(tr["pos"][m, 1], tr["pos"][m, 2], lw=1.2, color=colors[name], label=name)
        cm = m[:-1] & tr["sa"]["contact"]
        ax[1].scatter(tr["pos"][:-1][cm, 0], tr["pos"][:-1][cm, 1], s=6, color=colors[name], edgecolor="k", linewidth=0.3, zorder=5)
        ax[2].scatter(tr["pos"][:-1][cm, 1], tr["pos"][:-1][cm, 2], s=6, color=colors[name], edgecolor="k", linewidth=0.3, zorder=5)
        if tr["finish_pos"] is not None:
            ax[1].scatter([tr["finish_pos"][0]], [tr["finish_pos"][1]], marker="*", s=120, color=colors[name], edgecolor="k", zorder=6)
            ax[2].scatter([tr["finish_pos"][1]], [tr["finish_pos"][2]], marker="*", s=120, color=colors[name], edgecolor="k", zorder=6)
    z = zones["end"]
    ax[1].plot([z["mins"][0], z["maxs"][0]], [z["mins"][1], z["mins"][1]], color="m", lw=2, label="finish curtain")
    ax[2].plot([z["mins"][1], z["mins"][1]], [z["mins"][2], z["maxs"][2]], color="m", lw=2, label="finish curtain")
    if core is not None:
        for name, tr in tracks.items():
            if name in room:
                x0 = room[name]["r1_x_in"]
                i = int(np.argmin(np.abs(xs - x0)))
                prof = H[:, i]
                ax[2].plot(ys, prof, lw=1.0, ls="--", color=colors[name], alpha=0.8, label="floor under %s (x=%.0f)" % (name, xs[i]))
    ax[1].set_title("finish room, top view (dots = contact steps, star = finish crossing)"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8, loc="lower left")
    ax[2].set_title("finish room, side view (y, z); dashed = traced floor profile"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8, loc="lower right")
    ax[2].set_xlabel("y"); ax[2].set_ylabel("z")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "map_views.png"), dpi=110)
    plt.close(fig)

    # control distributions
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    for name, tr in tracks.items():
        sl = slice(tr["k0"], tr["k1"])
        ax[0].hist(tr["pitch"][sl], bins=60, range=(-80, 40), histtype="step", density=True, color=colors[name], label=name)
        yr = np.abs(np.diff(unwrap_deg(tr["yaw"][sl]))) / tr["step_dt"][tr["k0"]:tr["k1"] - 1]
        ax[1].hist(np.clip(yr, 0, 600), bins=60, range=(0, 600), histtype="step", density=True, color=colors[name], label=name)
        st = slice(tr["k0"], tr["k1"] - 1)
        off = {args.wr_label: 0, args.ours_label: 2.5, args.champ_label: 5}.get(name, 0)
        ax[2].plot(tr["t"][st] - tr["t0"], tr["side"][st] + off, lw=0.5, color=colors[name], label=name)
    ax[0].set_title("pitch (deg, negative = down)"); ax[0].legend(); ax[0].set_xlabel("deg")
    ax[1].set_title("|yaw rate| (deg/s), clipped at 600"); ax[1].legend(); ax[1].set_xlabel("deg/s")
    ax[2].set_title("strafe key (A=-1, none=0, D=+1), offset per run"); ax[2].set_xlabel("zone-clock time (s)"); ax[2].set_xlim(20, 30)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "control.png"), dpi=110)
    plt.close(fig)

    # ramp comparison bars
    if matched:
        fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        xs_ = np.arange(len(matched))
        ax[0].bar(xs_ - 0.2, [e["v_in"] for e, w in matched], 0.4, color=colors[args.ours_label], alpha=0.5, label="%s entry" % args.ours_label)
        ax[0].bar(xs_ + 0.2, [w["v_in"] for e, w in matched], 0.4, color=colors[args.wr_label], alpha=0.5, label="%s entry" % args.wr_label)
        ax[0].plot(xs_ - 0.2, [e["v_out"] for e, w in matched], "v", color=colors[args.ours_label], label="%s exit" % args.ours_label)
        ax[0].plot(xs_ + 0.2, [w["v_out"] for e, w in matched], "v", color=colors[args.wr_label], label="%s exit" % args.wr_label)
        ax[0].set_ylabel("speed (u/s)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        ax[0].set_title("matched contact events along the route (entry bar, exit marker)")
        ax[1].bar(xs_ - 0.2, [e["e_loss"] / 1e6 for e, w in matched], 0.4, color=colors[args.ours_label], alpha=0.5, label=args.ours_label)
        ax[1].bar(xs_ + 0.2, [w["e_loss"] / 1e6 for e, w in matched], 0.4, color=colors[args.wr_label], alpha=0.5, label=args.wr_label)
        ax[1].set_ylabel("energy destroyed (M u^2/s^2)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        ax[1].set_xticks(xs_); ax[1].set_xticklabels(["%.0f" % (e["arc_in"] / 1000) for e, w in matched], rotation=90, fontsize=7)
        ax[1].set_xlabel("event arc (ku)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "ramps.png"), dpi=110)
        plt.close(fig)

    print("wrote tables.md, compare_wr.json and PNGs to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
