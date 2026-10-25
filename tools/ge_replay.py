#!/usr/bin/env python3
"""ge_replay.py - replay a Go-Explore chain into ONE per-tick line (pbr1b1, 2026-10-23).

An archive written by tools/explore_phase1.py --exact-chains keeps, for every cell, the cell
its run was restored from (parent), that parent's state unchanged, and the run's actions from
the restore to the first entry (link_acts). Replaying the actions link by link from each
parent's archived state therefore reproduces the chain exactly; the per-tick states are the
line a backward curriculum spawns along (train_fast --demo-file). The chain end is the
archive's deepest cell on a GoalField (or --cell); nothing comes from a demo or a record.

    python tools/ge_replay.py --map maps/X.bsp --archive runs/ge/archive.npz \\
        --field maps/X.goalc_32.npz --out runs/ge/line.npy

Each link's replayed end is checked against the child's archived state (origin within
--tol u); a link that diverges is reported and the line stops there."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE, SurfCore, default_config  # noqa: E402
from surfgym.goalfield import GoalField                          # noqa: E402


def chain_to(parent: np.ndarray, i: int) -> list:
    out = []
    while i >= 0:
        out.append(int(i))
        i = int(parent[i])
    return out[::-1]


def energy_height(st, gravity: float = 800.0) -> np.ndarray:
    """h = z + |v|^2 / (2 g) of each archived state (the --energy-pot quantity)."""
    v = np.asarray(st["velocity"], np.float64).reshape(-1, 3)
    return np.asarray(st["origin"], np.float64).reshape(-1, 3)[:, 2] + (v * v).sum(1) / (2 * gravity)


def pick_order(d, h, max_d: float, k_energy: float, d0: float) -> np.ndarray:
    """--pick-max-d / --pick-energy: the cells at or past ``max_d`` on the field, best first by the
    training objective's own potential Phi = -100 d / d0 + K h / 1000 (the race reward's progress
    scale and the --energy-pot term, K its coefficient). K = 0 ranks by depth alone. Empty when no
    cell is past ``max_d``."""
    d = np.asarray(d, np.float64)
    cand = np.flatnonzero(np.isfinite(d) & (d <= float(max_d)))
    if not len(cand):
        return cand
    phi = -100.0 * d[cand] / float(d0) + float(k_energy) * np.asarray(h, np.float64)[cand] / 1000.0
    return cand[np.argsort(-phi, kind="stable")]


def survivors(map_path, meta, st, cand, runs: int, secs: float, seed: int = 1) -> np.ndarray:
    """Share of ``runs`` uniform-random rollouts from each candidate's state still alive after
    ``secs`` (actions redrawn every 20 ticks), at the archive's tick and speed cap: a dying run
    still enters new cells on its way down, and a fall into a fatal volume keeps its energy
    height until it hits, so an energy pick needs this filter (pbr3b4, 2026-10-25)."""
    from surfgym.core import ACTION_NVEC
    n_env = 2048
    per = max(1, n_env // int(runs))
    cfg = default_config(num_envs=n_env, spawn_mode=2, max_episode_ticks=100000, water_fail=1,
                         yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    if meta.get("maxvel") is not None:
        cfg.phys.sv_maxvelocity = float(meta["maxvel"])
    if meta.get("tick_ms") is not None:
        cfg.phys.msec = int(meta["tick_ms"])
    core = SurfCore(str(map_path), cfg)
    if not meta.get("keep_teleports", False):
        core.set_teleport_fail(True)
    ticks = max(1, int(round(float(secs) * 1000.0 / float(cfg.phys.msec))))
    rng = np.random.default_rng(seed)
    out = np.zeros(len(cand))
    for b in range(0, len(cand), per):
        cb = np.asarray(cand[b:b + per])
        core.reset(0)
        rows = np.repeat(cb, int(runs))
        for i, c in enumerate(rows.tolist()):
            s = np.array(st[c:c + 1], copy=True)
            s["tick"] = 0
            s["stuck_ticks"] = 0
            core.set_state(i, s)
        for i in range(len(rows), n_env):        # idle envs: a copy of the first candidate
            s = np.array(st[int(cb[0]):int(cb[0]) + 1], copy=True)
            s["tick"] = 0
            s["stuck_ticks"] = 0
            core.set_state(i, s)
        live = np.ones(n_env, bool)
        act = None
        for t in range(ticks):
            if t % 20 == 0:
                act = np.stack([rng.integers(0, n, n_env) for n in ACTION_NVEC],
                               1).astype(np.int32)
            _, _, done, trunc, _ = core.step(act)
            live &= ~(np.asarray(done, bool) | np.asarray(trunc, bool))
        out[b:b + len(cb)] = live[:len(rows)].reshape(len(cb), int(runs)).mean(1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=None,
                    help="the chain's end cell (archive index); default the deepest on the field")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--pick-max-d", type=float, default=None,
                    help="with --pick-energy: the chain end is chosen among the cells at or past "
                         "this d (a spine has to end there); none past it -> the deepest cell")
    ap.add_argument("--pick-energy", type=float, default=0.0,
                    help="> 0: among the --pick-max-d cells, the one with the highest training "
                         "potential -100 d / d0 + K h / 1000 (K = the run's --energy-pot, h the "
                         "energy height) that survives --pick-survive random rollouts, instead of "
                         "the deepest. 0 = the deepest cell (the old rule)")
    ap.add_argument("--pick-d0", type=float, default=None,
                    help="the race reward's progress scale d0 (the field at the true start)")
    ap.add_argument("--pick-survive", type=int, default=32,
                    help="uniform-random rollouts per candidate for the survival filter (0 = none)")
    ap.add_argument("--pick-survive-secs", type=float, default=1.0)
    ap.add_argument("--pick-survive-frac", type=float, default=0.25)
    ap.add_argument("--pick-check", type=int, default=512,
                    help="at most this many candidates (best first) are survival-tested")
    a = ap.parse_args(argv)
    z = np.load(a.archive, allow_pickle=False)
    if "link_acts" not in z.files:
        raise SystemExit(f"{a.archive} has no link_acts: run explore_phase1 with --exact-chains")
    meta = json.loads(Path(a.archive).with_suffix(".meta.json").read_text())
    st, parent = z["state"], z["parent"]
    acts, off = z["link_acts"], z["link_off"]
    zf = np.load(a.field, allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                  float(zf["cell"]), float(zf["reach_max"]))
    d = f.sample(st["origin"].astype(np.float64)).astype(np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    end = int(np.argmin(d)) if a.cell is None else int(a.cell)
    how = "deepest" if a.cell is None else "chosen"
    if a.cell is None and float(a.pick_energy) > 0.0 and a.pick_max_d is not None:
        # --pick-energy: the most valuable SURVIVABLE cell past the spine's bar, valued by the
        # training objective's own potential (progress + K x energy height), not depth alone - a
        # deepest chain that crashed its way there leaves the next gate without the energy
        if a.pick_d0 is None:
            raise SystemExit("--pick-energy needs --pick-d0 (the race reward's progress scale)")
        h = energy_height(st)
        order = pick_order(d, h, a.pick_max_d, a.pick_energy, a.pick_d0)[:max(1, int(a.pick_check))]
        print(f"--pick-energy {a.pick_energy:g}: {len(order)} candidate cells at or past d "
              f"{a.pick_max_d:,.0f} (deepest d {d[end]:,.0f}, energy height {h[end]:,.0f})")
        if len(order):
            ok_c = order
            if int(a.pick_survive) > 0:
                surv = np.zeros(0)
                ok_c = np.zeros(0, np.int64)
                for b in range(0, len(order), 64):
                    sb = survivors(a.map, meta, st, order[b:b + 64], int(a.pick_survive),
                                   float(a.pick_survive_secs))
                    surv = np.concatenate([surv, sb])
                    ok_c = order[:len(surv)][surv >= float(a.pick_survive_frac)]
                    if len(ok_c):
                        break
                print(f"  survival ({a.pick_survive} runs x {a.pick_survive_secs:g} s): "
                      f"{len(ok_c)} of the first {len(surv)} candidates pass "
                      f">= {a.pick_survive_frac:g}")
            if len(ok_c):
                end = int(ok_c[0])
                how = f"energy-picked (h {h[end]:,.0f})"
    chain = chain_to(parent, end)
    print(f"{how} cell {end}: d {d[end]:,.0f} at "
          f"{np.asarray(st[end]['origin']).round(0)}; chain of {len(chain)} cells")
    cfg = default_config(num_envs=1, spawn_mode=2, max_episode_ticks=int(meta.get("ep_ticks", 6000)),
                         water_fail=1, yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    if meta.get("maxvel") is not None:
        cfg.phys.sv_maxvelocity = float(meta["maxvel"])
    if meta.get("tick_ms") is not None:
        # explore_phase1 --tick-ms: the chains were found at that (whole-ms) tick
        cfg.phys.msec = int(meta["tick_ms"])
    core = SurfCore(str(a.map), cfg)
    if not meta.get("keep_teleports", False):
        core.set_teleport_fail(True)
    line = [st[chain[0]].copy()]
    ok = 0
    for k in range(1, len(chain)):
        p, c = chain[k - 1], chain[k]
        la = acts[off[c]:off[c + 1]].astype(np.int32)
        core.reset(0)
        s0 = st[p:p + 1].copy()
        s0["tick"] = 0
        s0["stuck_ticks"] = 0
        core.set_state(0, s0)
        seg = []
        dead = False
        for t in range(len(la)):
            _, _, done, trunc, _ = core.step(la[t][None, :])
            if (done[0] | trunc[0]) != 0:
                dead = True
                break
            seg.append(core.get_states()[0].copy())
        err = (float(np.linalg.norm(seg[-1]["origin"].astype(np.float64)
                                    - st[c]["origin"].astype(np.float64)))
               if seg and not dead else float("inf"))
        if dead or err > a.tol:
            print(f"link {k} ({p} -> {c}, {len(la)} ticks) DIVERGED: "
                  f"{'died' if dead else f'end off by {err:.2f} u'}; the line stops at link {ok}")
            break
        line.extend(seg)
        ok = k
    out = np.array(line, dtype=STATE_DTYPE)
    out["tick"] = 0
    out["stuck_ticks"] = 0
    np.save(a.out, out)
    dl = f.sample(out["origin"].astype(np.float64))
    print(f"{len(out)} states -> {a.out}; links {ok}/{len(chain) - 1}; line end d "
          f"{float(dl[-1]):,.0f}, best d {float(dl.min()):,.0f}")
    return 0 if ok == len(chain) - 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
