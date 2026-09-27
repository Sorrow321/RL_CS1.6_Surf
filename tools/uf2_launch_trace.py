"""uf2_launch_trace.py - WHAT does the mover do at unitfarmer2's south wall, tick by tick, next to
what the record does there? A MEASUREMENT (CLAUDE.md section 0: the record says which skill is
missing; it never trains anything).

    python tools/uf2_launch_trace.py <mover ckpt> [--t 6.0] [--secs 2.0] [--every 10] [--n 8]

The mover is spawned from the record's own state at --t (the entry bench's window,
runs/research/gate_bench/uf2_pitwindow.npy) and flies the record's own line from there (the
landing bench's `record` line). Printed every --every ticks for flight 0 (greedy) and the median
of --n sampled flights: position, |v|, horizontal speed, vz, view yaw, ground contact, held keys;
next to the record's row at the same tick (its yaw and its HL usercmd buttons).
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

WIN = ROOT / "runs" / "research" / "gate_bench" / "uf2_pitwindow.npy"
REC = ROOT / "runs" / "research" / "uf2_wr" / "surf_unitfarmer2.jsonl"
T0, T1 = 2.5, 9.0


def _keys(b: int) -> str:
    return (("F" if b & 8 else "-") + ("L" if b & 512 else "-") + ("R" if b & 1024 else "-")
            + ("J" if b & 2 else "-") + ("D" if b & 4 else "-"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--t", type=float, default=6.0)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the per-tick rows (JSON lines) here")
    a = ap.parse_args(argv)
    import torch
    import record_ckpt
    import edge_archive as ea
    from surfgym.route import resample_polyline
    from surfgym.goalprimplan import L_MAX
    win = np.load(WIN)
    j = int(round((a.t - T0) / (T1 - T0) * (len(win) - 1)))
    st = win[j].copy()
    st["tick"] = 0
    st["stuck_ticks"] = 0
    rec = []
    for ln in open(REC, encoding="utf-8"):
        if ln.startswith("["):
            rec.append(json.loads(ln)[:9])
        elif ln.startswith('{"end'):
            break
    rec = np.asarray(rec, np.float64)
    k0 = int(np.argmin(np.linalg.norm(rec[:, 1:4] - st["origin"].astype(np.float64), axis=1)))
    torch.manual_seed(int(a.seed))
    ctx = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--plan-scratch", "8"]
                            + ([] if a.greedy else ["--stochastic"])
                            + ["--map", str(ROOT / "maps_pool" / "surf_unitfarmer2.bsp")])
    fin = np.zeros(3)
    rline, _ = resample_polyline(rec[k0:k0 + 260, 1:4], ea.RAY_SPACING)
    rline = np.asarray(rline, np.float32)

    class _Fixed:
        n_choice = 1
        commit_ticks = budget_ticks = 200
        choice_nums = np.zeros((1, 1))
        finish = fin

        def line_and_curve_of(self, origin, velocity, yaw_deg, nums, k=None):
            return rline, rline

    ctx.planner = _Fixed()
    core = ctx.scratch.core
    for i in range(core.num_envs):
        core.set_state(i, st)
    neutral = np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (core.num_envs, 1))
    obs, *_ = core.step(neutral)
    st1 = core.get_states()[0].copy()
    st1["tick"] = 0
    st1["stuck_ticks"] = 0
    obs0 = np.array(obs[0], np.float32, copy=True)
    fl = ea.Flyer(ctx)
    S, K = fl.S, fl.K
    for i in range(S):
        core.set_state(i, st1)
    fl.sc.line.set_lines(np.arange(S), [rline[:L_MAX]] * S)
    pol = fl.sc.make_policy(core, fl.sc.line)
    if fl.keys_hold:
        from surfgym.keyshold import KeysHold
        pol.keys = KeysHold(S)
        pol._keys_tick = np.zeros(S, np.int64) - int(getattr(pol, "_period", K))
    pol._tick = 0
    obs = np.ascontiguousarray(np.stack([obs0] * S).astype(np.float32))
    alive = np.ones(S, bool)
    died_at = np.full(S, -1)
    rows = []
    n_t = int(round(a.secs * 100))
    print(f"uf2_launch_trace: {Path(a.ckpt).name}, {'greedy' if a.greedy else 'sampled'}, {S} "
          f"flights on the record's line from the record's state at t {rec[k0, 0] / 100:.2f} s; "
          f"tick {float(ctx.tick.ms):.2f} ms, decisions every {K} ticks")
    print("   t(s) | MOVER flight 0: pos (x, y, z)   |v|   vh    vz   yaw  gnd keys | alive "
          "| RECORD: pos (x, y, z)   |v|   vh    vz   yaw  keys")
    for t in range(n_t):
        acts = pol.act(obs)
        view = getattr(pol, "view", None)
        sv = core.get_states()
        o, v = sv["origin"].astype(np.float64), sv["velocity"].astype(np.float64)
        a0 = np.asarray(acts)[0].tolist()
        rows.append({"t": t, "origin": o[0].round(1).tolist(), "velocity": v[0].round(1).tolist(),
                     "yaw": float(sv["yaw"][0]), "onground": int(sv["onground"][0]), "acts": a0,
                     "alive": int(alive.sum()),
                     "vh_med": float(np.median(np.hypot(v[alive, 0], v[alive, 1])))
                     if alive.any() else 0.0,
                     "z_med": float(np.median(o[alive, 2])) if alive.any() else 0.0})
        if t % a.every == 0:
            r = rec[min(k0 + t, len(rec) - 1)]
            vh0 = float(np.hypot(v[0, 0], v[0, 1]))
            print(f"   {t / 100:4.2f} | ({o[0, 0]:6.0f},{o[0, 1]:6.0f},{o[0, 2]:6.0f}) "
                  f"{np.linalg.norm(v[0]):5.0f} {vh0:5.0f} {v[0, 2]:5.0f} {float(sv['yaw'][0]) % 360:5.0f} "
                  f"{int(sv['onground'][0]):3d} {a0} | {int(alive.sum()):2d}/{S} med vh "
                  f"{rows[-1]['vh_med']:5.0f} z {rows[-1]['z_med']:5.0f} | ({r[1]:6.0f},{r[2]:6.0f},"
                  f"{r[3]:6.0f}) {np.linalg.norm(r[4:7]):5.0f} {np.hypot(r[4], r[5]):5.0f} {r[6]:5.0f} "
                  f"{r[7] % 360:5.0f} {_keys(int(r[8]))}")
        obs, _r, done, trunc, _term = (core.step(acts) if view is None
                                       else core.step(acts, view=view))
        ended = alive & (np.asarray(done, bool) | np.asarray(trunc, bool))
        died_at[ended] = t + 1
        alive &= ~ended
        if not alive.any():
            print(f"   all flights ended by t {(t + 1) / 100:.2f} s")
            break
    print(f"   deaths: {int((died_at >= 0).sum())}/{S}; death ticks {sorted(died_at[died_at >= 0].tolist())[:12]}")
    if a.out:
        Path(a.out).write_text(chr(10).join(json.dumps(r) for r in rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
