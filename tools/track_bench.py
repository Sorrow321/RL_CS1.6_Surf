"""track_bench.py - how PRECISELY does a mover follow a line? A generic, map-agnostic tracking
measurement on the agent's OWN states and its OWN primitive distribution (Codex, agent bus
2026-09-27 08:57: the primary causal measurement for a tracking change must be held-out tracking on
policy-owned lines, not a record trace).

    python tools/track_bench.py <mover ckpt> <archive_states.npy> --map maps_pool/<m>.bsp
        [--n-states 64] [--k 4] [--seed 0] [--greedy] [--out rows.jsonl]

States: --n-states rows drawn (seeded) from an edge archive's state dump (edge_archive.py
--dump-states: a policy's own flights, no record). When the movers compared TRAINED on that
pool this measures training-distribution states x fresh draws, not held-out generalization. Lines: K primitive draws per state from the
MOVER'S OWN primitive config (edge_archive.PrimOperator with the checkpoint config: the same draw()
the archive and step 1 use), seeded, so two movers see the SAME states and the SAME lines, with the
same torch seed for the policy's sampling. Each (state, line) is flown once for the primitive's
2 s. Reported:
  PRIMARY: alive at the end AND the path passed within --goal-radius (192 u, the recipe's
      success sphere) of the line's end, tested on the segments between the 0.1 s samples;
  cross-track error: the flown path's distance to the drawn line with the ORDERED local
      projection (the reward's rule), per flight the mean and the max, reported SEPARATELY for
      survivors and deaths (a pooled median is death-censored);
  survivors' speed retention |v_end| / |v_0| and energy change / starting kinetic energy.
No map constant, no box, no record: the same numbers mean the same thing on every map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
G = 800.0


def _seg_dist(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """distance of point q to each segment a[i] -> b[i]"""
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", q[None] - a, ab)
                / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(a + ab * t[:, None] - q[None], axis=1)


def _ordered_errors(path: np.ndarray, line: np.ndarray, window: int = 16) -> np.ndarray:
    """Cross-track error of each path point to the line with the ORDERED LOCAL projection the
    executor's reward uses (Codex 2026-09-27: a whole-line argmin can match the wrong branch of a
    folded line): the anchor segment only moves forward, searched within +window segments of the
    previous one."""
    a, b = line[:-1], line[1:]
    k, out = 0, []
    for q in path:
        hi = min(len(a), k + window + 1)
        d = _seg_dist(q, a[k:hi], b[k:hi])
        j = int(np.argmin(d))
        k += j
        out.append(float(d[j]))
    return np.asarray(out)


def _reached(path: np.ndarray, end: np.ndarray, radius: float) -> bool:
    """did the flown path pass within radius of the end? Tested on the SEGMENTS between the
    0.1 s samples, so a fast crossing between two samples is not missed."""
    if len(path) == 0:
        return False
    if len(path) == 1:
        return bool(np.linalg.norm(path[0] - end) <= radius)
    return bool(np.min(_seg_dist(end, path[:-1], path[1:])) <= radius)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("states")
    ap.add_argument("--map", required=True)
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--goal-radius", type=float, default=192.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch
    import record_ckpt
    import edge_archive as ea
    rng = np.random.default_rng(int(a.seed))
    ctx = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--plan-scratch", "256"]
                            + ([] if a.greedy else ["--stochastic"]) + ["--map", str(a.map)])
    core = ctx.scratch.core
    st = np.unique(np.load(a.states))
    pick = st[rng.choice(len(st), size=min(int(a.n_states), len(st)), replace=False)]
    # the mover's OWN primitive distribution (its checkpoint config), seeded independently of the
    # mover so two checkpoints with the same config draw the same lines
    op = ea.PrimOperator(float(ctx.tick.ms), np.zeros(3), 1, int(a.seed),
                         cfg=getattr(ctx, "cfg", None))

    class _Fixed:
        """one fixed line per job; the lines the Flyer asked for are kept (slot order)"""
        n_choice = 1
        choice_nums = np.zeros((1, 1))
        finish = np.zeros(3)

        def __init__(self):
            self.queue, self.seen = [], []
            self.commit_ticks = self.budget_ticks = int(round(op.prim.secs * 100))

        def line_and_curve_of(self, origin, velocity, yaw_deg, nums, k=None):
            # the job's own rng (seed, job index): step 1's draw() with its loop rejection, from
            # the restored state - identical for any mover, so the lines are paired
            job = self.queue.pop(0)
            p, _l, _e, _t = op.prim.draw(origin, velocity, yaw_deg,
                                         np.random.default_rng([int(a.seed) + 7, int(job)]))
            ln, pts = op.prim.line_and_curve(origin, velocity, yaw_deg, p)
            self.seen.append(np.asarray(ln, np.float64))
            return ln, pts
    fx = _Fixed()
    ctx.planner = fx
    fl = ea.Flyer(ctx)
    arch = ea.Archive()
    neutral = np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (core.num_envs, 1))
    ids = []
    for c0 in range(0, len(pick), core.num_envs):
        part = pick[c0:c0 + core.num_envs]
        for i in range(core.num_envs):
            s2 = part[min(i, len(part) - 1)].copy()
            s2["tick"] = 0
            s2["stuck_ticks"] = 0
            core.set_state(i, s2)
        obs, *_ = core.step(neutral)
        cur = core.get_states()
        for i in range(len(part)):
            s3 = cur[i].copy()
            s3["tick"] = 0
            s3["stuck_ticks"] = 0
            ids.append(arch.add(s3, fl.fresh_keys(), np.array(obs[i], np.float32, copy=True),
                                ("S",), -1, -1, 0, 0, None))
    jobs = [(ids[i], 0) for i in range(len(ids)) for _ in range(int(a.k))]
    torch.manual_seed(int(a.seed) + 1)
    rows = []
    for c0 in range(0, len(jobs), fl.S):
        chunk = jobs[c0:c0 + fl.S]
        n = len(chunk)
        fx.queue = list(range(c0, c0 + n)) + [c0 + n - 1] * (fl.S - n)
        fx.seen = []
        res = fl.fly(chunk + [chunk[-1]] * (fl.S - n), arch, np.zeros(3), record_path=True)[:n]
        for j, ((nid, _k), r) in enumerate(zip(chunk, res)):
            line = fx.seen[j]
            s0 = arch.state[nid]
            path = np.asarray(r["path"] or [], np.float64)
            d = _ordered_errors(path, line) if len(path) else np.zeros(1)
            reach = _reached(path, line[-1], float(a.goal_radius))
            v0 = s0["velocity"].astype(np.float64)
            ke0 = 0.5 * float(np.sum(v0 ** 2))
            row = {"state": int(nid), "job": c0 + j, "alive": r["end"] is not None,
                   "err_mean": float(d.mean()), "err_max": float(d.max()), "reach": reach,
                   "v0": float(np.linalg.norm(v0))}
            if r["end"] is not None:
                v = r["end"]["velocity"].astype(np.float64)
                de = (0.5 * float(np.sum(v ** 2)) + G * float(r["end"]["origin"][2])
                      - ke0 - G * float(s0["origin"][2]))
                # speed retention and the mechanical-energy change per unit of starting kinetic
                # energy: neither depends on the map's z origin (Codex), unlike E_end / E_start
                row.update(vh=float(np.hypot(v[0], v[1])),
                           speed_ret=float(np.linalg.norm(v) / max(np.linalg.norm(v0), 1e-6)),
                           de_rel=float(de / max(ke0, 1e-6)))
            rows.append(row)
    al = [r for r in rows if r["alive"]]
    dead = [r for r in rows if not r["alive"]]
    joint = [r for r in al if r["reach"]]
    sr = np.array([r["speed_ret"] for r in al])
    dr = np.array([r["de_rel"] for r in al])
    vh = np.array([r["vh"] for r in al])
    fast = [r for r in rows if r["v0"] >= 1000.0]

    def _med(x):
        return f"{np.median(x):,.0f}" if len(x) else "-"
    print(f"track_bench: {Path(a.ckpt).name} ({'greedy' if a.greedy else 'sampled'}), {len(ids)} "
          f"states from {Path(a.states).name} x {a.k} primitive draws (the mover's own config) = "
          f"{len(rows)} flights, seed {a.seed}")
    # the PRIMARY event (Codex): alive at the end AND passed within the goal radius of the end
    print(f"   PRIMARY alive AND reached the end: {len(joint)}/{len(rows)} | alive {len(al)} | "
          f"reached {sum(r['reach'] for r in rows)} | starts at |v| >= 1,000: "
          f"{sum(1 for r in fast if r['alive'] and r['reach'])}/{len(fast)} joint, "
          f"{sum(r['alive'] for r in fast)} alive")
    # errors split by outcome: a pooled median is death-censored (an early crash keeps only its
    # on-line prefix and looks precise)
    print(f"   ordered cross-track error, per-flight mean: survivors median "
          f"{_med([r['err_mean'] for r in al])} u, dead {_med([r['err_mean'] for r in dead])} u | "
          f"per-flight max: survivors {_med([r['err_max'] for r in al])}, dead "
          f"{_med([r['err_max'] for r in dead])}")
    print(f"   survivors: speed retention median {np.median(sr) if len(sr) else float('nan'):.3f}, "
          f"energy change / starting kinetic median "
          f"{np.median(dr) if len(dr) else float('nan'):+.3f}, end vh median {_med(vh)}")
    if a.out:
        h = hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest()[:16]
        Path(a.out).write_text(chr(10).join(json.dumps(dict(r, ckpt=str(a.ckpt), sha=h,
                                                          greedy=bool(a.greedy), seed=a.seed))
                                            for r in rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
