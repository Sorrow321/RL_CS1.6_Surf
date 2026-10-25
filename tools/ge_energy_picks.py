#!/usr/bin/env python3
"""ge_energy_picks.py - the explorer's most energetic states AGAINST the field (pbr3b4, 2026-10-25).

    python tools/ge_energy_picks.py ARCHIVE.npz OUT_PREFIX [--n 4] [--field F] [--min-sep 256]

From a tools/explore_phase1.py --exact-chains archive rooted at the true spawn: the cells whose
shaping-field distance is ABOVE the spawn's (behind the start - where a field-follower never
goes) ranked by energy height h = z + |v|^2 / 2g, the top --n at least --min-sep u apart. Each
pick's state is written to OUT_PREFIX_<k>.npy (a --start-state for beam_tas, where the policy's
own proposals take over from the explorer) and its exact chain from the spawn (ge_replay) to
OUT_PREFIX_<k>_chain.npy. Nothing names a place on a map: on any map this is "the energy a
start that runs against the field could collect". The user's insight that motivated it
(2026-10-25): a human surfs a field-opposed first ramp for momentum before the ramp the policy
starts on."""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def survival(a, st, cand):
    """Share of --survive-runs uniform-random rollouts alive after --survive-secs, per cell."""
    from surfgym.core import ACTION_NVEC, SurfCore, default_config
    R, N = int(a.survive_runs), 2048
    per = max(1, N // R)
    cfg = default_config(num_envs=N, spawn_mode=2, max_episode_ticks=6000, water_fail=1,
                         yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    cfg.phys.sv_maxvelocity = float(a.maxvel)
    core = SurfCore(a.map, cfg)
    core.set_teleport_fail(True)
    rng = np.random.default_rng(1)
    ticks = int(round(a.survive_secs * 100))
    out = np.zeros(len(cand))
    for b in range(0, len(cand), per):
        cb = cand[b:b + per]
        core.reset(0)
        rows = np.repeat(cb, R)
        for i, k in enumerate(rows):
            s = np.array(st[k:k + 1], copy=True)
            s["tick"] = 0
            s["stuck_ticks"] = 0
            core.set_state(i, s)
        live = np.ones(N, bool)
        for t in range(ticks):
            if t % 20 == 0:
                act = np.stack([rng.integers(0, n, N) for n in ACTION_NVEC], 1).astype(np.int32)
            _, _, done, trunc, _ = core.step(act)
            live &= ~(np.asarray(done, bool) | np.asarray(trunc, bool))
        out[b:b + len(cb)] = live[:len(rows)].reshape(len(cb), R).mean(1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive")
    ap.add_argument("out_prefix")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--field", default=str(ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz"))
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_princessburglar_b26.bsp"))
    ap.add_argument("--min-sep", type=float, default=256.0)
    ap.add_argument("--gravity", type=float, default=800.0)
    ap.add_argument("--maxvel", type=float, default=3500.0)
    ap.add_argument("--survive-secs", type=float, default=1.0)
    ap.add_argument("--survive-frac", type=float, default=0.25)
    ap.add_argument("--survive-runs", type=int, default=32)
    ap.add_argument("--max-check", type=int, default=20000,
                    help="survivability is checked on at most this many of the most energetic "
                         "candidates")
    a = ap.parse_args(argv)
    from surfgym.goalfield import GoalField
    z = np.load(a.archive, allow_pickle=False)
    st, parent = z["state"], z["parent"]
    zf = np.load(a.field, allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"], float(zf["cell"]),
                  float(zf["reach_max"]))
    o = st["origin"].astype(np.float64)
    v = st["velocity"].astype(np.float64)
    d = f.sample(o).astype(np.float64)
    ok = d < f.reach_max - 0.5 * f.cell
    roots = np.flatnonzero(parent < 0)
    d0 = float(np.min(d[roots]))
    h = o[:, 2] + (v * v).sum(axis=1) / (2.0 * a.gravity)
    # SURVIVABLE cells only: a cell reached a tick before a death has the most kinetic energy of
    # all (pbr3b4's first picks: states plunging into a fatal pit and a push-down trigger), and
    # "the explorer continued from it" is not enough (a dying run still visits new cells on the
    # way down: 0 / 1,024 random runs survived 1 s from those) - so every candidate is restored
    # --survive-runs times and stepped with uniform random actions, and kept when at least
    # --survive-frac of the runs are alive after --survive-secs
    cand = np.flatnonzero(ok & (d > d0))
    cand = cand[np.argsort(-h[cand])][:int(a.max_check)]
    alive = survival(a, st, cand)
    cand = cand[alive >= float(a.survive_frac)]
    picks = []
    for k in cand:
        if all(np.linalg.norm(o[k] - o[j]) >= a.min_sep for j in picks):
            picks.append(int(k))
        if len(picks) >= a.n:
            break
    h0 = float(h[roots].max())
    print(f"{len(cand):,} survivable cells behind the start (d > {d0:,.0f}); spawn h {h0:,.0f}")
    for i, k in enumerate(picks):
        row = np.array(st[k:k + 1], copy=True)
        row["tick"] = 0
        row["stuck_ticks"] = 0
        np.save(f"{a.out_prefix}_{i}.npy", row)
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "ge_replay.py"), "--map", a.map,
                            "--archive", a.archive, "--field", a.field, "--cell", str(k),
                            "--out", f"{a.out_prefix}_{i}_chain.npy"], capture_output=True,
                           text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"pick {i}: cell {k} h {h[k]:,.0f} (+{h[k] - h0:,.0f}) d {d[k]:,.0f} at "
              f"{np.round(o[k])} |v| {np.linalg.norm(v[k]):,.0f}; chain: {tail}")


if __name__ == "__main__":
    main()
