#!/usr/bin/env python3
"""potential_view.py - the race potential through the policy's eyes.

The user's ask (branch contyaw-abs, 2026-09-07): "render another image, on
which you show the potential field measured at the same point, for which we
render depth. And train with 2 input channels." This is that image. For a
few states along a recorded episode it renders, side by side,

  * the depth image the policy sees (64x32 equiangular, the run's lidar);
  * the ABSOLUTE potential channel (``--obs-potential abs``): the geodesic
    distance-to-finish one field cell short of each ray's hit, divided by
    the map's start geodesic d0, clipped to [0, 1.5], 1.5 = unreachable;
  * the RELATIVE potential channel (``--obs-potential rel``): the eye's
    own geodesic minus the hit's, divided by 2,000 u, clipped to [-2, 2]:
    POSITIVE where that part of the view leads toward the goal, negative
    where it leads back, -2 = unreachable.

All three come out of one ``surfgym.vision.GpuLidar`` render with a
``LidarPotential`` attached (the same kernel the trainer runs under the
flag), so the picture IS the observation, pixel for pixel.

States: spawn, 15 s, 30 s, 45 s, 60 s and the wall entry - the last tick at
which the map pushed back (the vertical acceleration departing from the
gravity step, tools/pick_selfline.py's rule), i.e. the start of the final
free fall of a non-finishing episode.

    python tools/demo/potential_view.py \
        --traj C:/RL_Surf_base/runs/research/cyABSV/traj_8020557824.jsonl \
        --map C:/RL_Surf/maps/surf_src_cannonball.bsp --out docs/potential_view.png

The map path must be the MAIN checkout's (the caches next to it are signed
against that bsp's mtime; a worktree copy re-bakes the goal field for 30
minutes - CLAUDE.md). The lidar geometry is read from the run.json next to
the trajectory when there is one (64x32, range 11,500 / near 2,000, cell 32
on the absolute-view runs). GPU when available, else the torch fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import torch                                                   # noqa: E402

from surfgym import SurfCore, default_config                   # noqa: E402
from surfgym.goalfield import build_goal_field                 # noqa: E402
from surfgym.rewards import map_spawn_pool                     # noqa: E402
from surfgym.vision import GpuLidar, LidarPotential            # noqa: E402
from surfgym.zones import load_zones                           # noqa: E402

G = 800.0          # sv_gravity, u/s^2


def load_episode(path: Path, episode: int):
    """Rows of one episode of a traj_*.jsonl: (tick, x, y, z, vx, vy, vz,
    yaw, buttons, ..., pitch at column 12) plus its header dict."""
    header, rows, k = None, [], -1
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == "{":
                k += 1
                if k > episode:
                    break
                if k == episode:
                    header = json.loads(line)
                continue
            if k == episode:
                rows.append(json.loads(line))
    if header is None or not rows:
        raise SystemExit(f"{path}: no episode {episode}")
    return header, np.asarray(rows, np.float64)


def wall_entry_tick(a: np.ndarray, tick_ms: float) -> int:
    """The last tick at which the map pushed back: vertical acceleration
    departing from the gravity step (free flight is exactly -g dt per tick;
    any contact changes that). Returns the first tick of the final free
    fall - the state the agent left the ramp in."""
    vz = a[:, 6]
    dt = tick_ms / 1000.0
    dv = np.diff(vz)
    contact = np.abs(dv + G * dt) > 0.5
    idx = np.flatnonzero(contact)
    return int(idx[-1] + 1) if len(idx) else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traj", default="C:/RL_Surf_base/runs/research/cyABSV/"
                    "traj_8020557824.jsonl")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--map", default="C:/RL_Surf/maps/surf_src_cannonball.bsp")
    ap.add_argument("--times", default="0,15,30,45,60",
                    help="seconds into the episode (the wall entry is added)")
    ap.add_argument("--no-wall", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "potential_view.png"))
    ap.add_argument("--w", type=int, default=None)
    ap.add_argument("--h", type=int, default=None)
    args = ap.parse_args()

    traj = Path(args.traj)
    header, a = load_episode(traj, args.episode)
    tick_ms = float(header.get("tick_ms") or 10.0)
    cfg = {}
    rj = traj.parent / "run.json"
    if rj.exists():
        cfg = json.loads(rj.read_text(encoding="utf-8")).get("config", {})
    W = int(args.w or cfg.get("lidar_w") or 64)
    H = int(args.h or cfg.get("lidar_h") or 32)
    rng_u = float(cfg.get("lidar_range") or 11500.0)
    near = cfg.get("lidar_near")
    cell = float(cfg.get("lidar_cell") or 32.0)
    hfov = float(cfg.get("lidar_hfov") or 120.0)
    vfov = float(cfg.get("lidar_vfov") or 90.0)
    gcell = float(cfg.get("goal_cell") or cell)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    core = SurfCore(args.map, default_config(num_envs=1, lidar_w=0, lidar_h=0))
    zones = load_zones(core.bsp_path)
    t0 = time.time()
    gf = build_goal_field(core, zones["end"], cell=gcell, device=str(device))
    dt = time.time() - t0
    print(f"goal field @ cell {gcell:g} in {dt:.1f}s"
          + ("  ** WARNING: that smells like a RE-BAKE - wrong map "
             "path/mtime? **" if dt > 30 else ""))
    raw = map_spawn_pool(core)
    d0 = float(np.mean(gf.sample(raw["origin"])))
    print(f"start geodesic d0 {d0:,.0f} u")
    pot_abs = LidarPotential(gf, "abs", d0=d0, device=device)
    pot_rel = pot_abs.with_mode("rel")
    print(pot_abs.describe())
    print(pot_rel.describe())
    lidar = GpuLidar(core, W, H, hfov_deg=hfov, vfov_deg=vfov,
                     range_units=rng_u, near_range=near, cell=cell,
                     device=device, potential=pot_abs)

    # ---- the states -------------------------------------------------------
    ticks = a[:, 0].astype(np.int64)
    want = [(f"{s:g} s", int(round(float(s) * 1000.0 / tick_ms)))
            for s in (float(x) for x in args.times.split(",") if x.strip())]
    if not args.no_wall:
        want.append(("wall entry", wall_entry_tick(a, tick_ms)))
    states = []
    for label, tk in want:
        i = int(np.clip(np.searchsorted(ticks, tk), 0, len(a) - 1))
        states.append((label, a[i]))
    n = len(states)
    pos = np.stack([s[1][1:4] for s in states]).astype(np.float32)
    yaw = np.array([s[1][7] for s in states], np.float32)
    pitch = np.array([s[1][12] if s[1].shape[0] > 12 else 0.0 for s in states],
                     np.float32)
    duck = np.array([(int(s[1][8]) & 4) != 0 for s in states], np.int32)
    o = torch.as_tensor(pos, device=device)
    yw = torch.as_tensor(yaw, device=device)
    pt = torch.as_tensor(pitch, device=device)
    dk = torch.as_tensor(duck, device=device)
    img_abs = lidar.render(o, yw, pt, dk).cpu().numpy()           # (n,H,W,2)
    lidar.potential = pot_rel
    img_rel = lidar.render(o, yw, pt, dk).cpu().numpy()
    assert np.array_equal(img_abs[..., 0], img_rel[..., 0])       # same depth
    depth = img_abs[..., 0]
    ch_abs, ch_rel = img_abs[..., 1], img_rel[..., 1]
    d_eye = pot_abs.eye(o, dk).cpu().numpy()

    # ---- the numbers ------------------------------------------------------
    enc_max = 1.25 if (near and float(near) < rng_u) else 1.0
    print()
    print(f"{'state':>11} {'tick':>5} {'pos':>28} {'|v|':>6} {'yaw':>7} "
          f"{'pitch':>6} | {'d_eye':>8} {'d_eye/d0':>8} | "
          f"{'abs min':>7} {'max':>6} {'mean':>6} {'bad':>5} | "
          f"{'rel min':>7} {'max':>6} {'mean':>6} {'>0':>5} {'bad':>5}")
    for i, (label, r) in enumerate(states):
        v = float(np.hypot(r[4], r[5]))
        ca, cr = ch_abs[i], ch_rel[i]
        print(f"{label:>11} {int(r[0]):>5} "
              f"({r[1]:8.0f},{r[2]:8.0f},{r[3]:8.0f}) {v:6.0f} {r[7]:7.2f} "
              f"{pitch[i]:6.1f} | {d_eye[i]:8.0f} {d_eye[i] / d0:8.3f} | "
              f"{ca.min():7.3f} {ca.max():6.3f} {ca.mean():6.3f} "
              f"{(ca >= 1.5).mean():5.2f} | "
              f"{cr.min():7.3f} {cr.max():6.3f} {cr.mean():6.3f} "
              f"{(cr > 0).mean():5.2f} {(cr <= -2.0).mean():5.2f}")

    # ---- the picture ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aspect = (vfov / H) / (hfov / W)        # square ANGULAR pixels
    # four columns: the depth image; the abs channel on ITS scale (what the
    # policy sees - nearly flat within a frame, the level says where in the
    # run the agent is); the same abs frame stretched to its own range (the
    # structure the scale hides); the rel channel on its scale
    fig, axes = plt.subplots(n, 4, figsize=(19.0, 2.55 * n + 1.0),
                             squeeze=False)
    cols = (("depth image\n(0 = touching, 1 = 2,000 u, "
             f"{enc_max:g} = clear at {rng_u:,.0f} u)",
             1.0 - np.clip(depth / enc_max, 0.0, 1.0), "turbo", 0.0, 1.0),
            ("abs channel: d_hit / d0, the policy's scale\n(0 = at the "
             "finish, 1 = at the start, 1.5 = unreachable)",
             ch_abs, "viridis_r", 0.0, 1.5),
            ("abs channel, this frame's own range\n(the structure the "
             "fixed scale hides)", ch_abs, "viridis_r", None, None),
            ("rel channel: (d_eye - d_hit) / 2,000 u\n(+ = leads toward "
             "the goal, - = leads back, -2 = unreachable)",
             ch_rel, "RdYlGn", -2.0, 2.0))
    for j, (title, data, cmap, vmin, vmax) in enumerate(cols):
        for i, (label, r) in enumerate(states):
            ax = axes[i, j]
            lo = float(data[i].min()) if vmin is None else vmin
            hi = float(data[i].max()) if vmax is None else vmax
            if hi <= lo:
                hi = lo + 1e-6
            im = ax.imshow(data[i], cmap=cmap, vmin=lo, vmax=hi,
                           aspect=aspect, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                v = float(np.hypot(r[4], r[5]))
                ax.set_ylabel(f"{label}\n({r[1]:.0f}, {r[2]:.0f}, {r[3]:.0f})"
                              f"\n|v| {v:.0f} u/s  d/d0 {d_eye[i] / d0:.2f}",
                              fontsize=8)
            if i == 0:
                ax.set_title(title, fontsize=8.5)
            if vmin is None:
                ax.set_xlabel(f"range {lo:.3f} .. {hi:.3f}  "
                              f"({(hi - lo) * d0:,.0f} u)", fontsize=7.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            if j == 0:
                cb.set_ticks([0.0, 0.5, 1.0])
                cb.set_ticklabels(["clear", "", "near"])
    fig.suptitle(f"{header.get('map', '?')}  episode {args.episode} of "
                 f"{traj.name}: the depth image and the race potential "
                 f"sampled one cell short of each ray's hit  ({W}x{H}, hfov "
                 f"{hfov:g}, vfov {vfov:g}; d0 = {d0:,.0f} u)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
