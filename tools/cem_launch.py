#!/usr/bin/env python3
"""cem_launch.py - physics-exact CEM from the policy's OWN line, scored on the grader's max
progress (zlr2b4, 2026-11-01). Generic: no map constant, every number is a search budget.

Why. surf_za_loop's goal field is ~height (the five finish slabs are a ceiling over most of the
map), so the best progress an episode reaches is ~its highest point. The incumbent pumps its
energy height to ~4,500 u and still tops out at z ~716: it never converts speed into height.
The policy's own states with the heading turned toward the ramp launch to 75-95% (what
--respawn-front-perturb does to a reservoir row) - but those turns are not physical. This tool
asks the physical question: from the policy's own exact states (a saved greedy line, e.g.
beam_tas --save-frontier --frontier-last), is there a CONTROL sequence the engine flies that
reaches much higher progress? If so, its replay is a line made only of the policy's own prefix
and physics: a curriculum spine (train_fast --demo-file ... --self-states).

The control. Per knot (KNOT ticks): a view yaw offset from the live velocity heading IN UNITS OF
THE OPTIMAL STRAFE ANGLE atan(30/|v_h|) (the trainer's --yaw-adaptive unit - an air strafe adds
speed only within ~that angle of perpendicular, so offsets drawn in degrees mostly coast), a side
key (left / none / right) and a forward key, for --horizon seconds; then --coast seconds with no
keys and the view along the velocity. play_step = the core's usercmd physics + map triggers (a
kill / teleport ends the rollout). CEM: Gaussian per-knot offsets, categorical side keys,
Bernoulli forward keys, elite refit with smoothing.

    python tools/cem_launch.py --line runs/x/beam_frontier.npz --roots 24871,24914 \\
        --out runs/x/cem --map maps/M.bsp --field maps/M.goalc_32.npz --tick-ms 7 --maxvel 3500
    python tools/cem_launch.py --line ... --roots 24871 --spine runs/x/cem/result.json:0 \\
        --spine-out runs/x/cem/spine.npy      # the best root's line as a spine

Output: result.json (per root: own best progress over the next --own-secs on the line, the CEM
best, its tick, the control) and, with --spine, a STATE_DTYPE .npy = the line's states up to the
root + the replayed launch up to its best-progress tick (clocks zeroed, as spines are).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE, SurfCore, SurfState, default_config  # noqa: E402


def load_field(path):
    from surfgym.goalfield import GoalField
    z = np.load(path, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


class Searcher:
    def __init__(self, a):
        self.a = a
        self.core = SurfCore(str(a.map), default_config(
            num_envs=1, spawn_mode=2, water_fail=1, sv_maxvelocity=float(a.maxvel),
            max_episode_ticks=10 ** 7), tick_ms=float(a.tick_ms))
        self.core.reset()
        self.msec = int(round(float(a.tick_ms)))
        if abs(self.msec - float(a.tick_ms)) > 1e-6:
            raise SystemExit("--tick-ms must be a whole number of ms (play_step's msec)")
        self.tick = self.msec / 1000.0
        self.field = load_field(a.field)
        self.lim = float(self.field.reach_max) - 0.5 * float(self.field.cell)
        z = np.load(a.line, allow_pickle=False)
        self.states = z["states"]
        self.d0 = float(self.field.sample(np.asarray(z["spawn_state"]["origin"], np.float64)
                                          .reshape(1, 3))[0])
        self.nk = max(1, int(round(float(a.horizon) / self.tick / int(a.knot))))

    def prog(self, pos) -> float:
        d = float(self.field.sample(np.asarray(pos, np.float64).reshape(1, 3))[0])
        return -1e9 if d >= self.lim else 100.0 * (self.d0 - d) / self.d0

    def roll(self, i0, u, sd, fw, keep=False):
        """One rollout from line state i0. -> (best progress, its tick, [states] if keep)."""
        st = SurfState.from_buffer_copy(self.states[i0].tobytes())
        K = int(self.a.knot)
        n_ctl = self.nk * K
        n = n_ctl + int(round(float(self.a.coast) / self.tick))
        best, bt, out = -1e9, -1, []
        for k in range(n):
            vx, vy = st.velocity[0], st.velocity[1]
            h = math.degrees(math.atan2(vy, vx))
            if k < n_ctl:
                j = k // K
                ang = math.degrees(math.atan2(30.0, max(1.0, math.hypot(vx, vy))))
                fl = self.core.play_step(st, h + float(u[j]) * ang, 0.0, 400.0 * float(fw[j]),
                                         400.0 * (int(sd[j]) - 1), 0, self.msec)
            else:
                fl = self.core.play_step(st, h, 0.0, 0.0, 0.0, 0, self.msec)
            if fl & (self.core.PLAY_TELEPORTED | self.core.PLAY_KILLED):
                break
            if keep:
                out.append(np.frombuffer(bytes(st), dtype=STATE_DTYPE)[0].copy())
            if k % int(self.a.sample_every) == 0 or keep:
                p = self.prog((st.origin[0], st.origin[1], st.origin[2]))
                if p > best:
                    best, bt = p, k
        return best, bt, out

    def cem(self, i0, rng):
        a = self.a
        NK = self.nk
        mu, sg = np.zeros(NK), np.full(NK, float(a.sigma0))
        P, Q = np.full((NK, 3), 1.0 / 3.0), np.full(NK, float(a.fwd0))
        best = (-1e9, -1, None)
        for _ in range(int(a.iters)):
            u = np.clip(mu + sg * rng.standard_normal((a.pop, NK)), -a.umax, a.umax)
            sd = np.stack([rng.choice(3, size=a.pop, p=P[k]) for k in range(NK)], axis=1)
            fw = (rng.random((a.pop, NK)) < Q).astype(np.float64)
            res = [self.roll(i0, u[m], sd[m], fw[m]) for m in range(a.pop)]
            sc = np.array([r[0] for r in res])
            o = np.argsort(-sc)[: a.elite]
            if sc[o[0]] > best[0]:
                best = (float(sc[o[0]]), int(res[o[0]][1]),
                        (u[o[0]].copy(), sd[o[0]].copy(), fw[o[0]].copy()))
            s = float(a.smooth)
            mu = s * u[o].mean(0) + (1 - s) * mu
            sg = np.maximum(s * u[o].std(0) + (1 - s) * sg, float(a.sigma_min))
            for k in range(NK):
                c = np.bincount(sd[o, k], minlength=3) / float(a.elite)
                P[k] = np.clip(s * c + (1 - s) * P[k], 0.02, 1.0)
                P[k] /= P[k].sum()
            Q = np.clip(s * fw[o].mean(0) + (1 - s) * Q, 0.02, 0.98)
        return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--line", required=True, help="npz with per-tick 'states' + 'spawn_state' "
                    "(beam_tas --save-frontier)")
    ap.add_argument("--roots", required=True, help="comma list of line state indices")
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_za_loop.bsp"))
    ap.add_argument("--field", default=str(ROOT / "maps" / "surf_za_loop.goalc_32.npz"))
    ap.add_argument("--tick-ms", type=float, default=7.0)
    ap.add_argument("--maxvel", type=float, default=3500.0)
    ap.add_argument("--horizon", type=float, default=2.5, help="control seconds")
    ap.add_argument("--coast", type=float, default=4.0, help="hands-off seconds after it")
    ap.add_argument("--knot", type=int, default=8, help="ticks per control knot")
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--elite", type=int, default=20)
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--umax", type=float, default=40.0, help="|offset| cap, optimal-strafe units")
    ap.add_argument("--sigma0", type=float, default=2.0)
    ap.add_argument("--sigma-min", type=float, default=0.3)
    ap.add_argument("--fwd0", type=float, default=0.3)
    ap.add_argument("--smooth", type=float, default=0.7)
    ap.add_argument("--sample-every", type=int, default=4, help="field reads every N ticks")
    ap.add_argument("--own-secs", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="directory for result.json")
    ap.add_argument("--spine", default=None, metavar="RESULT_JSON:IDX",
                    help="replay entry IDX of a result.json and write the line as a spine")
    ap.add_argument("--spine-out", default=None)
    a = ap.parse_args(argv)
    S = Searcher(a)
    if a.spine:
        path, _, idx = a.spine.rpartition(":")
        r = json.loads(Path(path).read_text())[int(idx)]
        for k in ("horizon", "coast", "knot"):
            setattr(a, k, r[k])
        S.nk = len(r["u"])
        i0 = int(r["root"])
        best, bt, traj = S.roll(i0, r["u"], r["sd"], r["fw"], keep=True)
        if abs(best - float(r["cem"])) > 0.05:
            print(f"!! replay best {best:.2f}% != search {r['cem']:.2f}%")
        line = np.concatenate([S.states[:i0 + 1], np.array(traj[: bt + 1], dtype=STATE_DTYPE)])
        line = line.copy()
        line["tick"] = 0
        line["stuck_ticks"] = 0
        np.save(a.spine_out, line)
        dl = float(S.field.sample(np.asarray(line["origin"][-1], np.float64).reshape(1, 3))[0])
        print(f"spine: {len(line)} states = line[:{i0 + 1}] + {bt + 1} launch states; end d "
              f"{dl:,.0f} ({100 * (S.d0 - dl) / S.d0:.2f}%) at "
              f"({line['origin'][-1][0]:.0f}, {line['origin'][-1][1]:.0f}, "
              f"{line['origin'][-1][2]:.0f}) -> {a.spine_out}")
        return 0
    rng = np.random.default_rng(int(a.seed))
    out_dir = Path(a.out) if a.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for i0 in (int(x) for x in a.roots.split(",") if x):
        t0 = time.time()
        own_n = int(round(float(a.own_secs) / S.tick))
        o = S.states["origin"][i0:i0 + own_n].astype(np.float64)
        dd = np.asarray(S.field.sample(o), np.float64)
        own = float((100.0 * (S.d0 - dd[dd < S.lim]) / S.d0).max()) if (dd < S.lim).any() else -1e9
        best, bt, ctl = S.cem(i0, rng)
        p0 = S.states["origin"][i0]
        print(f"root {i0} ({i0 * S.tick:6.1f}s) at ({p0[0]:6.0f},{p0[1]:6.0f},{p0[2]:6.0f}) |v| "
              f"{np.linalg.norm(S.states['velocity'][i0]):5.0f}: own next {a.own_secs:g}s "
              f"{own:6.2f}% | CEM {best:6.2f}% at +{bt * S.tick:4.1f}s [{time.time() - t0:.0f}s]",
              flush=True)
        res.append({"root": i0, "t": i0 * S.tick, "own": own, "cem": best, "best_tick": bt,
                    "u": ctl[0].tolist() if ctl else None, "sd": ctl[1].tolist() if ctl else None,
                    "fw": ctl[2].tolist() if ctl else None, "horizon": float(a.horizon),
                    "coast": float(a.coast), "knot": int(a.knot)})
        if out_dir:
            (out_dir / "result.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
