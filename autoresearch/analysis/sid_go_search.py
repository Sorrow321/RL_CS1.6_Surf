#!/usr/bin/env python3
"""sid_go_search.py - ANALYSIS ONLY (never training): CAN'T vs WON'T past a frontier.

Go-Explore's exploration phase with RANDOM held actions in the real simulator, rooted at a
checkpoint's OWN reservoir states (its deepest ones by goal-field distance). No policy, no demo,
no map constant: an archive of 128 u position cells x 500 u/s speed buckets keeps, per cell, the
first state that reached it; each round restarts N envs from archive cells (weight
1/sqrt(1 + times chosen), x4 for the deepest fifth by distance), drives each with random
macro-actions in the policy's own action space (velocity-frame yaw target off_warp(u), u ~ U(-1, 1);
forward / side keys uniform over {-400, 0, +400}; jump 0.1, duck 0.2; each held 4-40 ticks), and
files every live state into the archive every 10 ticks. A death (teleport_fail, as in evals) ends
an env's branch. Every archived state is reached by valid physics from a root, so the deepest
archive state says how far ANY action sequence gets from the policy's own frontier.

    python3 sid_go_search.py <ckpt> [--envs 2048] [--rounds 40] [--horizon 300] [--roots 400]
        [--out dir]

Writes <out>/archive.npy (STATE_DTYPE rows of the archive) and prints, per round, the deepest
progress reached, the archive size and where the deepest state is.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "python"))
from surfgym.core import STATE_DTYPE, SurfCore, default_config  # noqa: E402
from surfgym.goalfield import GoalField  # noqa: E402
from surfgym.view import off_warp, view_mode_code  # noqa: E402

D0 = 97768.0


def load_field(maps: Path):
    z = np.load(maps / "surf_src_sidistic.goal_32.npz", allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


FINE = False


def cell_key(p, v):
    c = np.floor(np.asarray(p, np.float64) / 128.0).astype(np.int64)
    s = np.minimum(np.floor(np.hypot(v[:, 0], v[:, 1]) / 500.0), 7).astype(np.int64)
    k = ((c[:, 0] + 512) * 1024 * 1024 * 8 + (c[:, 1] + 512) * 1024 * 8
         + (c[:, 2] + 512) * 8 + s)
    if FINE:      # + heading octant and vz sign (-, ~0, +): 24 velocity classes per cell
        h = (np.floor((np.arctan2(v[:, 1], v[:, 0]) + np.pi) / (np.pi / 4)).astype(np.int64)) % 8
        vz = np.where(v[:, 2] < -150, 0, np.where(v[:, 2] > 150, 2, 1)).astype(np.int64)
        k = k * 24 + h * 3 + vz
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=300, help="ticks per round")
    ap.add_argument("--roots", type=int, default=400, help="deepest reservoir states seeded")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maps", default=str(ROOT / "maps"))
    ap.add_argument("--out", default="go_search_out")
    ap.add_argument("--deep-bias", type=float, default=4.0,
                    help="weight multiplier for the deepest fifth of the archive by distance "
                         "(1 = pure Go-Explore count weights; a detour needs going backwards)")
    ap.add_argument("--min-prog", type=float, default=0.0,
                    help="roots: only reservoir states at >= this progress (%%)")
    ap.add_argument("--fine", action="store_true", help="cells keep heading octant + vz sign")
    ap.add_argument("--surf-p", type=float, default=0.0,
                    help="share of macro-actions that are the generic SURF primitive: one strafe "
                         "key held (A or D, fifty-fifty), no forward key, the view K x "
                         "atan(30/|v_h|) ahead of the velocity on that side (K ~ U(0.5, 2)), "
                         "held 20-80 ticks - the input that rides a ramp or air-strafes")
    a = ap.parse_args()
    global FINE
    FINE = bool(a.fine)
    import torch
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    res = ck["respawn"]["states"]
    maps = Path(a.maps)
    gf = load_field(maps)
    d = gf.sample(res["origin"].astype(np.float32)).astype(np.float64)
    okp = 100 * (D0 - d) / D0 >= a.min_prog
    order = np.flatnonzero(okp)[np.argsort(d[okp])][:a.roots]
    roots = res[order].copy()
    print(f"roots: {len(roots)} deepest reservoir states, progress "
          f"{100 * (D0 - d[order[-1]]) / D0:.2f}-{100 * (D0 - d[order[0]]) / D0:.2f}%", flush=True)
    core = SurfCore(str(maps / "surf_src_sidistic.bsp"), default_config(
        num_envs=a.envs, spawn_mode=2, max_episode_ticks=10 ** 7, water_fail=1,
        yaw_jitter_deg=0.0, sv_maxvelocity=float(cfg.get("maxvel", 4000.0)),
        yaw_adaptive=1 if cfg.get("yaw_adaptive") else 0,
        yaw_blend=float(cfg.get("yaw_blend") or 1.0), lidar_w=0, lidar_h=0,
        pitch_rate_max_deg=float(cfg.get("pitch_rate", 1.33)),
        view_mode=view_mode_code(cfg.get("view_absolute"))))
    core.set_teleport_fail(True)
    core.set_spawn_pool(roots)
    core.reset(a.seed)
    rng = np.random.default_rng(a.seed)
    # archive: key -> index into arch_rows
    arch_rows = list(roots)
    arch_d = list(d[order])
    arch_n = [0] * len(roots)
    arch_par = [-1] * len(roots)        # parent archive entry (-1 = a reservoir root)
    arch_key = {}
    for j, k in enumerate(cell_key(roots["origin"], roots["velocity"])):
        arch_key.setdefault(int(k), j)
    best_d = float(min(arch_d))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    N = a.envs
    t0 = time.time()
    for rnd in range(a.rounds):
        ad = np.asarray(arch_d)
        w = 1.0 / np.sqrt(1.0 + np.asarray(arch_n, np.float64))
        deep = ad <= np.quantile(ad, 0.2)
        w = w * np.where(deep, a.deep_bias, 1.0)
        pick = rng.choice(len(arch_rows), N, p=w / w.sum())
        for j in pick:
            arch_n[j] += 1
        for i, j in enumerate(pick):
            core.set_state(i, arch_rows[j])
        alive = np.ones(N, bool)
        acts = np.zeros((N, 6), np.int32)
        acts[:, 0], acts[:, 1] = 7, 3
        view = np.zeros((N, 2), np.float32)
        left = np.zeros(N, np.int64)
        surfing = np.zeros(N, bool)
        surf_k = np.zeros(N, np.float64)
        for t in range(a.horizon):
            nxt = left <= 0
            if nxt.any():
                m = int(nxt.sum())
                view[nxt, 0] = off_warp(rng.uniform(-1.0, 1.0, m)).astype(np.float32)
                view[nxt, 1] = 0.0
                acts[nxt, 2] = rng.integers(0, 3, m)
                acts[nxt, 3] = rng.integers(0, 3, m)
                acts[nxt, 4] = (rng.random(m) < 0.1).astype(np.int32)
                acts[nxt, 5] = (rng.random(m) < 0.2).astype(np.int32)
                left[nxt] = rng.integers(4, 41, m)
                surfing[nxt] = False
                if a.surf_p > 0.0:
                    sp = np.flatnonzero(nxt)[rng.random(m) < a.surf_p]
                    if len(sp):
                        side = rng.integers(0, 2, len(sp))               # 0 = A (left), 1 = D (right)
                        acts[sp, 3] = np.where(side == 0, 0, 2)
                        acts[sp, 2] = 1                                  # no forward key
                        acts[sp, 4] = 0
                        acts[sp, 5] = 0
                        surf_k[sp] = rng.uniform(0.5, 2.0, len(sp)) * np.where(side == 0, 1.0, -1.0)
                        surfing[sp] = True
                        left[sp] = rng.integers(20, 81, len(sp))
            if surfing.any():
                vv = core.states_view["velocity"]
                hs = np.hypot(vv[:, 0], vv[:, 1])
                lead = np.degrees(np.arctan2(30.0, np.maximum(hs, 30.0)))
                view[surfing, 0] = (surf_k[surfing] * lead[surfing]).astype(np.float32)
            left -= 1
            _, _, done, trunc, _ = core.step(acts, view=view)
            alive &= ~(np.asarray(done, bool) | np.asarray(trunc, bool))
            if t % 10 == 9 and alive.any():
                st = core.get_states()
                idx = np.flatnonzero(alive)
                sd = gf.sample(st["origin"][idx].astype(np.float32)).astype(np.float64)
                keys = cell_key(st["origin"][idx], st["velocity"][idx])
                for q, k, dd in zip(idx, keys, sd):
                    k = int(k)
                    if k not in arch_key and dd < gf.reach_max - 64.0:
                        arch_key[k] = len(arch_rows)
                        arch_rows.append(st[q].copy())
                        arch_d.append(float(dd))
                        arch_n.append(0)
                        arch_par.append(int(pick[q]))
                        if dd < best_d:
                            best_d = float(dd)
        if rnd % 50 == 49 or rnd == a.rounds - 1:
            ao = np.asarray([r["origin"] for r in arch_rows[-min(len(arch_rows), 200000):]])
            west = int(((ao[:, 0] < 1940) & (ao[:, 2] > 10000) & (ao[:, 2] < 11200)).sum())
            print(f"   cells west of the wall (x < 1940, z 10000-11200): {west}", flush=True)
        jb = int(np.argmin(arch_d))
        pb = arch_rows[jb]["origin"]
        print(f"round {rnd:3d}: archive {len(arch_rows):6d} cells, survivors {int(alive.sum()):5d}/{N}, "
              f"deepest {100 * (D0 - best_d) / D0:6.2f}% at ({pb[0]:.0f}, {pb[1]:.0f}, {pb[2]:.0f}) "
              f"({time.time() - t0:.0f}s)", flush=True)
    rows = np.asarray(arch_rows, STATE_DTYPE)
    np.save(out / "archive.npy", rows)
    np.save(out / "parent.npy", np.asarray(arch_par, np.int64))
    # the chain of archive entries from a reservoir root to the deepest state
    chain = []
    j = int(np.argmin(arch_d))
    while j >= 0 and len(chain) < 100000:
        chain.append(j)
        j = arch_par[j]
    chain = chain[::-1]
    np.save(out / "chain.npy", rows[chain])
    print("chain to the deepest state:", " -> ".join(
        f"({rows[c]['origin'][0]:.0f},{rows[c]['origin'][1]:.0f},{rows[c]['origin'][2]:.0f}) "
        f"{100 * (D0 - arch_d[c]) / D0:.2f}%" for c in chain), flush=True)
    ad = np.asarray(arch_d)
    top = np.argsort(ad)[:20]
    summary = [dict(prog=round(100 * (D0 - ad[j]) / D0, 2), pos=[round(float(x)) for x in rows[j]["origin"]],
                    vel=[round(float(x)) for x in rows[j]["velocity"]]) for j in top]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary[:10], indent=1))


if __name__ == "__main__":
    main()
