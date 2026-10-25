#!/usr/bin/env python3
"""energy_waypoint.py - pick, from a Go-Explore archive, the state that reaches a place on the
policy's OWN route with the most energy to spare, and write the exact line to it (pbr3b2, 2026-10-25).

The user's map insight (autoresearch/NOTES.md, 2026-10-25): on surf_princessburglar_b26 the policy
skips the first ramp after start_run - using it runs against the geodesic field - while a human
surfs it for momentum first and arrives everywhere later with more speed. A d-ranked search or a
d-only curriculum cannot ask for that detour: the slow route reaches the same places. The generic
question this tool asks instead (nothing in it names a ramp, a box or a coordinate):

    which archived state lies ON the policy's own greedy line (within ``--near`` u of one of its
    states) and carries the most usable energy (z + (vx^2 + vy^2 + max(vz, 0)^2)/2g: downward speed
    into a floor is not credited) ABOVE what the greedy line has at that very place?

A field-opposed momentum detour shows up as exactly such a state: the same place, more energy. The
archive must come from tools/explore_phase1.py --exact-chains (reward-free, here rooted at the map's
true spawn, cells keyed on position x energy). The chain to the chosen cell is replayed exactly by
tools/ge_replay.py's rule (each link from its parent's archived state, checked at its end) and the
per-tick line is written as a STATE_DTYPE spine, clocks zeroed, views along the velocity
(tools/spine_canon_view.py's rule) - a SELF_STATES file for train_fast --demo-file with
--demo-success-energy, which asks the curriculum for the waypoint's ENERGY, not only its place.

    python tools/energy_waypoint.py --map maps/X.bsp --archive runs/ge/archive.npz \\
        --greedy runs/x/beam_frontier.npz --out runs/x/pump_line.npy
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

from surfgym.core import STATE_DTYPE                             # noqa: E402


def energy(states, g2: float = 1600.0) -> np.ndarray:
    v = np.asarray(states["velocity"], np.float64)
    return np.asarray(states["origin"], np.float64)[:, 2] + (v * v).sum(1) / g2


def usable_energy(states, g2: float = 1600.0) -> np.ndarray:
    """z + (vx^2 + vy^2 + max(vz, 0)^2) / 2g: the energy a state can still carry along - a state
    moving DOWN into a floor (a push-down trigger's victim, a landing) is not credited for the
    vertical speed it is about to lose."""
    v = np.asarray(states["velocity"], np.float64)
    vz = np.maximum(v[:, 2], 0.0)
    return (np.asarray(states["origin"], np.float64)[:, 2]
            + (v[:, 0] ** 2 + v[:, 1] ** 2 + vz ** 2) / g2)


def surplus(arch_states, greedy_states, near: float, g2: float = 1600.0, min_cos: float = -2.0):
    """Per archived state: (energy above the greedy line's at the nearest greedy state, index of
    that greedy state, distance to it); states farther than ``near`` from the line, or (min_cos)
    moving more than acos(min_cos) away from the greedy line's direction there, get -inf."""
    a = np.asarray(arch_states["origin"], np.float64)
    gpos = np.asarray(greedy_states["origin"], np.float64)
    ha, hg = usable_energy(arch_states, g2), usable_energy(greedy_states, g2)
    best_d = np.full(len(a), np.inf)
    best_j = np.zeros(len(a), np.int64)
    for j0 in range(0, len(gpos), 256):                # chunked nearest-greedy-state search
        gp = gpos[j0:j0 + 256]
        dist = np.linalg.norm(a[:, None, :] - gp[None, :, :], axis=2)
        jj = np.argmin(dist, axis=1)
        dd = dist[np.arange(len(a)), jj]
        upd = dd < best_d
        best_d[upd], best_j[upd] = dd[upd], jj[upd] + j0
    sur = ha - hg[best_j]
    sur[best_d > near] = -np.inf
    if min_cos > -1.0:
        # ON the route means moving ALONG it: same place, same way (a state crossing the line the
        # other way carries its energy somewhere the policy does not go)
        va = np.asarray(arch_states["velocity"], np.float64)
        vg = np.asarray(greedy_states["velocity"], np.float64)[best_j]
        na, ng = np.linalg.norm(va, axis=1), np.linalg.norm(vg, axis=1)
        cos = (va * vg).sum(1) / np.maximum(na * ng, 1e-9)
        sur[(ng > 50.0) & (cos < min_cos)] = -np.inf
    return sur, best_j, best_d


def replay_chain(map_path, archive_path, z, end: int, tol: float):
    """tools/ge_replay.py's rule: the chain to cell ``end`` replayed link by link from each
    parent's archived state, every link's end checked against the child's state (``tol`` u).
    -> (per-tick STATE_DTYPE rows as a list, links replayed, links in the chain)."""
    import ge_replay
    from surfgym.core import SurfCore, default_config
    meta = json.loads(Path(archive_path).with_suffix(".meta.json").read_text())
    st, parent = z["state"], z["parent"]
    chain = ge_replay.chain_to(parent, end)
    cfg = default_config(num_envs=1, spawn_mode=2, max_episode_ticks=int(meta.get("ep_ticks", 6000)),
                         water_fail=1, yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    if meta.get("maxvel") is not None:
        cfg.phys.sv_maxvelocity = float(meta["maxvel"])
    core = SurfCore(str(map_path), cfg)
    if not meta.get("keep_teleports", False):
        core.set_teleport_fail(True)
    acts, off = z["link_acts"], z["link_off"]
    line, ok = [st[chain[0]].copy()], 0
    for k in range(1, len(chain)):
        p, c = chain[k - 1], chain[k]
        la = acts[off[c]:off[c + 1]].astype(np.int32)
        core.reset(0)
        s0 = st[p:p + 1].copy()
        s0["tick"] = 0
        s0["stuck_ticks"] = 0
        core.set_state(0, s0)
        seg, dead = [], False
        for t in range(len(la)):
            _, _, done, trunc, _ = core.step(la[t][None, :])
            if (done[0] | trunc[0]) != 0:
                dead = True
                break
            seg.append(core.get_states()[0].copy())
        err = (float(np.linalg.norm(seg[-1]["origin"].astype(np.float64)
                                    - st[c]["origin"].astype(np.float64)))
               if seg and not dead else float("inf"))
        if dead or err > tol:
            print(f"link {k} ({p} -> {c}) DIVERGED ({'died' if dead else f'{err:.2f} u'}); the "
                  f"line stops at link {ok}")
            break
        line.extend(seg)
        ok = k
    return line, ok, len(chain) - 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--archive", required=True, help="explore_phase1 --exact-chains archive.npz")
    ap.add_argument("--greedy", default=None,
                    help="the policy's greedy line as STATE_DTYPE states (.npy, or a beam_tas "
                         "beam_frontier.npz from greedy-only envs)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--near", type=float, default=64.0,
                    help="u: an archived state counts as ON the greedy line within this distance "
                         "of one of its states (default 64 = one position cell)")
    ap.add_argument("--min-cos", type=float, default=0.5,
                    help="the state must move within acos(this) of the greedy line's direction at "
                         "the matched place (default 0.5 = 60 deg; -2 = off)")
    ap.add_argument("--min-surplus", type=float, default=50.0,
                    help="u of usable energy above the greedy line a waypoint must carry")
    ap.add_argument("--pick", choices=["furthest", "surplus"], default="furthest",
                    help="furthest (default): of the states carrying >= --min-surplus, the one "
                         "matched FURTHEST along the greedy line (a moving state beside the "
                         "standing spawn always has a surplus; the useful one is the deepest place "
                         "on the route still reached with energy to spare); surplus: the largest "
                         "surplus anywhere on the line")
    ap.add_argument("--prefix-archive", default=None,
                    help="a two-stage search: --archive was rooted at states of THIS archive; its "
                         "chain to the root is replayed and put in front, so the line starts at "
                         "the first archive's root (the map's true spawn)")
    ap.add_argument("--top-energy", type=int, default=0,
                    help="ROOT MODE: write the N archived states of most usable energy (anywhere - "
                         "the promise of a state, no place named) to --write-roots and exit; the "
                         "next stage explores from them (round 2's GE_pump2)")
    ap.add_argument("--write-roots", default=None)
    ap.add_argument("--tol", type=float, default=0.5)
    a = ap.parse_args(argv)
    z = np.load(a.archive, allow_pickle=False)
    if "link_acts" not in z.files:
        raise SystemExit(f"{a.archive} has no link_acts: run explore_phase1 with --exact-chains")
    st, parent = z["state"], z["parent"]
    if a.top_energy > 0:
        ue = usable_energy(st)
        idx = np.argsort(-ue)[:a.top_energy]
        roots = np.asarray(st, STATE_DTYPE)[idx].copy()
        roots["tick"] = 0
        roots["stuck_ticks"] = 0
        np.save(a.write_roots, roots)
        print(f"{len(roots)} roots of most usable energy ({ue[idx].min():,.0f} .. "
              f"{ue[idx].max():,.0f}) -> {a.write_roots}")
        return 0
    gz = (np.load(a.greedy, allow_pickle=False))
    g = gz["states"] if hasattr(gz, "files") else gz
    g = np.asarray(g, STATE_DTYPE)
    sur, gj, gd = surplus(st, g, a.near, min_cos=a.min_cos)
    ok_m = np.isfinite(sur) & (sur >= a.min_surplus)
    if not ok_m.any():
        raise SystemExit(f"no archived state on the greedy line carries {a.min_surplus:g} u of "
                         f"usable energy above it (best {np.nanmax(np.where(np.isfinite(sur), sur, np.nan)):.1f})")
    if a.pick == "furthest":
        cand = np.flatnonzero(ok_m)
        far = gj[cand].max()
        cand = cand[gj[cand] == far]
        end = int(cand[np.argmax(sur[cand])])
    else:
        end = int(np.argmax(np.where(ok_m, sur, -np.inf)))
    print(f"energy waypoint ({a.pick}, >= {a.min_surplus:g} u): cell {end} at "
          f"{np.asarray(st[end]['origin']).round(0)}, {gd[end]:.0f} u from greedy state {gj[end]} "
          f"of {len(g)}, usable energy {usable_energy(st[end:end + 1])[0]:,.0f} = the greedy's "
          f"{usable_energy(g[gj[end]:gj[end] + 1])[0]:,.0f} + {sur[end]:.0f}")
    top = np.argsort(-np.where(np.isfinite(sur), sur, -1e9))[:8]
    for i in top:
        print(f"  cell {i}: surplus {sur[i]:6.0f} u at greedy state {gj[i]} "
              f"{np.asarray(st[i]['origin']).round(0)}")
    line, ok, n_links = replay_chain(a.map, a.archive, z, end, a.tol)
    if a.prefix_archive:
        # a two-stage search: this archive's root is a state of the first archive - replay the
        # first archive's chain to it and put it in front
        z1 = np.load(a.prefix_archive, allow_pickle=False)
        r0 = np.asarray(line[0]["origin"], np.float64)
        e1 = np.linalg.norm(z1["state"]["origin"].astype(np.float64) - r0, axis=1)
        c1 = int(np.argmin(e1))
        if e1[c1] > a.tol:
            raise SystemExit(f"the chain's root is not a state of {a.prefix_archive} "
                             f"(nearest {e1[c1]:.2f} u)")
        pre, ok1, n1 = replay_chain(a.map, a.prefix_archive, z1, c1, a.tol)
        print(f"prefix: {len(pre)} states from {a.prefix_archive} (links {ok1}/{n1})")
        line = pre[:-1] + line
    out = np.array(line, dtype=STATE_DTYPE)
    out["tick"] = 0
    out["stuck_ticks"] = 0
    from surfgym.autospine import canon_view
    out, _ = canon_view(out)
    np.save(a.out, out)
    h = energy(out)
    print(f"{len(out)} states -> {a.out}; links {ok}/{n_links}; line end energy {h[-1]:,.0f}")
    return 0 if ok == n_links else 1


if __name__ == "__main__":
    raise SystemExit(main())
