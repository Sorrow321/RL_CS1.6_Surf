#!/usr/bin/env python3
"""line_track.py - re-fly a searched line in the POLICY's action space (pbr2b2, 2026-10-24).

A Go-Explore line (tools/ge_replay.py) is produced by uniform yaw-DELTA bins held per tick, so its
view wanders (pbr1b1's line_a looks 139-176 deg backward through its flight) and nothing says a
--view-continuous --view-absolute velocity policy acting every --act-every ticks can fly it at
all. This beam search answers that: every decision it expands each beam state with every
candidate policy action - a view target offset off_warp(u) from the velocity heading (the
trainer's own warp, core view_mode 1) for --n-u values of u, times the forward/side keys (3 x 3),
duck and jump - held for act_every ticks, scores each result by its distance to the line state
act_every ticks later (position + --w-vel x velocity), keeps the --beam best and moves on. The
output is a per-tick STATE_DTYPE line flown entirely with policy-frame actions (its views are the
policy's own), plus the per-decision actions (u, keys), and the tracking error along the way:
where it explodes is where the line needs control the policy does not have.

Nothing comes from a demo or a record: the input is a line of the policy's own search.

    python tools/line_track.py --map maps/X.bsp --line autoresearch/spines/line.npy \\
        --out runs/track/line_tracked.npz [--start 0 --stop 1880 --beam 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE, SurfCore, default_config  # noqa: E402
from surfgym.view import off_warp                               # noqa: E402


def states_rw(core: SurfCore) -> np.ndarray:
    """A WRITABLE view of the core's per-env states (surf_set_state is a plain struct copy, so
    one vectorised assignment equals N set_state calls; the --pitch-fixed buffer trick)."""
    import ctypes
    from surfgym.core import SurfState
    ptr = core._lib.surf_states_ptr(core._handle())
    buf = (SurfState * core.num_envs).from_address(ptr)
    return np.frombuffer(buf, dtype=STATE_DTYPE)


def candidates(n_u: int, duck_opts, jump_opts):
    us = np.linspace(-1.0, 1.0, n_u)
    rows = []
    for u in us:
        for f in range(3):
            for s in range(3):
                for d in duck_opts:
                    for j in jump_opts:
                        rows.append((u, f, s, j, d))
    return np.array(rows, np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--line", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--act-every", type=int, default=4)
    ap.add_argument("--n-u", type=int, default=33, help="view offsets per decision (u grid)")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--w-vel", type=float, default=0.02, help="u per (u/s) in the score")
    ap.add_argument("--maxvel", type=float, default=3500.0)
    ap.add_argument("--align-start", type=int, default=1, choices=(0, 1),
                    help="start from the line state with its yaw along the velocity")
    ap.add_argument("--free-duck", action="store_true",
                    help="search duck too (default: duck follows the line's ducked flag)")
    a = ap.parse_args(argv)
    L = np.load(a.line, allow_pickle=False)
    stop = len(L) - 1 if a.stop is None else min(int(a.stop), len(L) - 1)
    K = int(a.act_every)
    jump_opts = (0, 1)
    duck_opts = (0, 1) if a.free_duck else (None,)
    C = candidates(int(a.n_u), duck_opts if a.free_duck else (0,), jump_opts)
    nc = len(C)
    B = int(a.beam)
    N = B * nc
    cfg = default_config(num_envs=N, spawn_mode=2, max_episode_ticks=1_000_000, water_fail=1,
                         yaw_jitter_deg=0.0, yaw_adaptive=1, lidar_w=0, lidar_h=0,
                         sv_maxvelocity=float(a.maxvel), view_mode=1)
    core = SurfCore(str(a.map), cfg)
    core.set_teleport_fail(True)
    core.reset(0)
    s0 = L[a.start:a.start + 1].copy()
    s0["tick"] = 0
    s0["stuck_ticks"] = 0
    if a.align_start:
        v = s0["velocity"][0]
        if np.hypot(v[0], v[1]) >= 100.0:
            s0["yaw"] = np.degrees(np.arctan2(v[1], v[0])) % 360.0
    beam = [dict(state=s0[0].copy(), err=0.0, hist=[], acts=[])]
    view = np.zeros((N, 2), np.float32)
    acts = np.zeros((N, 6), np.int32)
    acts[:, 0], acts[:, 1] = 7, 3
    errs = []
    t = int(a.start)
    while t + K <= stop:
        nb = len(beam)
        n = nb * nc
        rw = states_rw(core)
        for b in range(nb):
            rw[b * nc:(b + 1) * nc] = beam[b]["state"]
        rw["seg_hint"][:n] = np.maximum(rw["seg_hint"][:n], 0)
        cc = np.tile(C, (nb, 1))
        view[:n, 0] = off_warp(cc[:, 0])
        acts[:n, 2] = cc[:, 1]
        acts[:n, 3] = cc[:, 2]
        acts[:n, 4] = cc[:, 3]
        dead = np.zeros(N, bool)
        seg = []
        for k in range(K):
            if not a.free_duck:
                acts[:n, 5] = 1 if (int(L[t + k + 1]["oldbuttons"]) & 4) else 0
            else:
                acts[:n, 5] = cc[:, 4]
            if k > 0:
                acts[:n, 4] = 0                      # a jump is an impulse: first tick only
            _, _, done, trunc, _ = core.step(acts, view=view)
            dead |= (done | trunc) != 0
            seg.append(core.get_states()[:n].copy())
        st = seg[-1]
        tgt = L[t + K]
        e = (np.linalg.norm(st["origin"].astype(np.float64) - tgt["origin"].astype(np.float64),
                            axis=1)
             + float(a.w_vel) * np.linalg.norm(st["velocity"].astype(np.float64)
                                               - tgt["velocity"].astype(np.float64), axis=1))
        e[dead[:n]] = np.inf
        order = np.argsort(e)[:B]
        if not np.isfinite(e[order[0]]):
            print(f"t {t}: every candidate died - the track stops")
            break
        new = []
        for j in order:
            if not np.isfinite(e[j]):
                continue
            b, i = divmod(int(j), nc)
            new.append(dict(state=st[j].copy(), err=float(e[j]),
                            hist=beam[b]["hist"] + [np.array([seg[k][j] for k in range(K)])],
                            acts=beam[b]["acts"] + [C[i].copy()]))
        beam = new
        errs.append(beam[0]["err"])
        t += K
        if (t - a.start) % (25 * K) == 0:
            print(f"t {t}: best err {beam[0]['err']:.2f} (pos "
                  f"{np.linalg.norm(beam[0]['state']['origin'] - L[t]['origin']):.1f} u)",
                  flush=True)
    best = beam[0]
    states = np.concatenate([s0] + best["hist"]) if best["hist"] else s0
    np.savez(a.out, states=states, acts=np.array(best["acts"]), err=np.array(errs),
             start=int(a.start), act_every=K)
    e = np.array(errs)
    print(f"tracked {a.start}..{t} ({len(states)} ticks): err median {np.median(e):.2f}, "
          f"max {e.max():.2f} at line state {a.start + K * (1 + int(np.argmax(e)))}; -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
