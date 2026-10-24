#!/usr/bin/env python3
"""pb_perturb_spawns.py - perturbed copies of the policy's OWN greedy states before its biggest
energy loss (pbr2b4, round 2). Generic: the window is found on the policy's own greedy line as the
tick of the largest one-tick drop of the energy height h = z + |v|^2/(2g) (no map coordinate);
each state in [loss - before_s, loss - after_s] is copied K times with the origin moved by
U(-dxy, dxy) in x and y and U(0, dz) up and the horizontal velocity rotated by U(-rot, rot) deg -
the recipe's --respawn-front-perturb / --respawn-novel family (own states, moved), aimed at the
policy's own energy sink instead of its death. Copies that start inside geometry (the core reports
them stuck within 3 ticks) are dropped.

  python3 autoresearch/analysis/pb_perturb_spawns.py DUMP.npz OUT.npy [--ep 0 --k 8 --dxy 50
      --dz 40 --rot 10 --before-s 1.0 --after-s 0.1 --seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "autoresearch" / "analysis"))
import pb_line_tools as T  # noqa: E402
from surfgym.core import STATE_DTYPE, SurfCore, default_config  # noqa: E402

MAP = str(ROOT / "maps" / "surf_princessburglar_b26.bsp")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--dxy", type=float, default=50.0)
    ap.add_argument("--dz", type=float, default=40.0)
    ap.add_argument("--rot", type=float, default=10.0)
    ap.add_argument("--before-s", type=float, default=1.0)
    ap.add_argument("--after-s", type=float, default=0.1)
    ap.add_argument("--max-tick", type=int, default=0, help="search the loss only before this tick (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    e = T.eps_of(a.dump)[a.ep]
    o = e["origin"].astype(np.float64)
    v = e["velocity"].astype(np.float64)
    h = o[:, 2] + (v * v).sum(1) / 1600.0
    lim = a.max_tick if a.max_tick > 0 else len(e)
    w = int(np.argmin(np.diff(h[:lim])))
    lo, hi = max(0, w - int(a.before_s * 100)), max(0, w - int(a.after_s * 100))
    base = e[lo:hi + 1]
    print(f"largest one-tick h drop at tick {w} ({h[w]:.0f} -> {h[w + 1]:.0f}); window ticks {lo}..{hi}")
    rng = np.random.default_rng(a.seed)
    rows = np.repeat(base, a.k)
    n = len(rows)
    rows = rows.copy()
    rows["origin"][:, 0] += rng.uniform(-a.dxy, a.dxy, n).astype(np.float32)
    rows["origin"][:, 1] += rng.uniform(-a.dxy, a.dxy, n).astype(np.float32)
    rows["origin"][:, 2] += rng.uniform(0.0, a.dz, n).astype(np.float32)
    ang = np.radians(rng.uniform(-a.rot, a.rot, n))
    vx, vy = rows["velocity"][:, 0].astype(np.float64), rows["velocity"][:, 1].astype(np.float64)
    rows["velocity"][:, 0] = (vx * np.cos(ang) - vy * np.sin(ang)).astype(np.float32)
    rows["velocity"][:, 1] = (vx * np.sin(ang) + vy * np.cos(ang)).astype(np.float32)
    rows["tick"] = 0
    rows["stuck_ticks"] = 0
    # drop copies that start inside geometry: step each 3 ticks with no input, keep those that
    # neither die nor report stuck_ticks
    cfg = default_config(num_envs=n, spawn_mode=2, max_episode_ticks=6000, water_fail=1,
                         yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    cfg.phys.sv_maxvelocity = 3500.0
    core = SurfCore(MAP, cfg)
    core.set_teleport_fail(True)
    core.reset(0)
    st = np.zeros(n, STATE_DTYPE)
    for nm in STATE_DTYPE.names:
        st[nm] = rows[nm]
    for i in range(n):
        core.set_state(i, st[i:i + 1])
    act = np.zeros((n, 6), np.int32)
    act[:, 0], act[:, 1] = 7, 3
    ok = np.ones(n, bool)
    for _ in range(3):
        _, _, done, trunc, _ = core.step(act)
        ok &= ~(np.asarray(done, bool) | np.asarray(trunc, bool))
        ok &= core.get_states()["stuck_ticks"] == 0
    out = np.zeros(int(ok.sum()), STATE_DTYPE)
    for nm in STATE_DTYPE.names:
        out[nm] = rows[ok][nm]
    np.save(a.out, out)
    print(f"{len(out)} of {n} perturbed copies kept -> {a.out}")


if __name__ == "__main__":
    main()
