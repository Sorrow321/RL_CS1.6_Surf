"""track_bench.py - how PRECISELY does a mover follow a line? A generic, map-agnostic tracking
measurement on the agent's OWN states and its OWN primitive distribution (Codex, agent bus
2026-09-27 08:57: the primary causal measurement for a tracking change must be held-out tracking on
policy-owned lines, not a record trace).

    python tools/track_bench.py <mover ckpt> <archive_states.npy> --map maps_pool/<m>.bsp
        [--n-states 64] [--k 4] [--seed 0] [--greedy] [--out rows.jsonl]

States: --n-states rows drawn (seeded) from an edge archive's own state dump (edge_archive.py
--dump-states: the policy's own flights, no record). Lines: K primitive draws per state from the
MOVER'S OWN primitive config (edge_archive.PrimOperator with the checkpoint config: the same draw()
the archive and step 1 use), seeded, so two movers see the SAME states and the SAME lines, with the
same torch seed for the policy's sampling. Each (state, line) is flown once for the primitive's
2 s. Reported:
  cross-track error: the distance of the flown path (every 0.1 s) to the drawn line - per flight
      the mean and the max, then the median / p90 over flights;
  completion: the path came within --goal-radius (192 u, the recipe's success sphere) of the
      line's end;
  alive at the end; energy retention E_end / E_start (|v|^2 / 2 + g z, alive flights) and
      horizontal speed at the end.
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


def _polyline_dist(q: np.ndarray, line: np.ndarray) -> float:
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", q[None] - a, ab)
                / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + ab * t[:, None] - q[None], axis=1)))


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
            d = np.array([_polyline_dist(q, line) for q in path]) if len(path) else np.zeros(1)
            reach = bool(len(path) and np.min(np.linalg.norm(path - line[-1], axis=1))
                         <= float(a.goal_radius))
            e0 = 0.5 * float(np.sum(s0["velocity"].astype(np.float64) ** 2)) \
                + G * float(s0["origin"][2])
            row = {"state": int(nid), "job": c0 + j, "alive": r["end"] is not None,
                   "err_mean": float(d.mean()), "err_max": float(d.max()), "reach": reach,
                   "v0": float(np.linalg.norm(s0["velocity"].astype(np.float64)))}
            if r["end"] is not None:
                v = r["end"]["velocity"].astype(np.float64)
                row.update(vh=float(np.hypot(v[0], v[1])),
                           e_ratio=(0.5 * float(np.sum(v ** 2)) + G * float(r["end"]["origin"][2]))
                           / max(e0, 1e-9) if e0 > 0 else None)
            rows.append(row)
    em = np.array([r["err_mean"] for r in rows])
    ex = np.array([r["err_max"] for r in rows])
    al = [r for r in rows if r["alive"]]
    er = np.array([r["e_ratio"] for r in al if r.get("e_ratio") is not None])
    vh = np.array([r["vh"] for r in al])
    fast = np.array([r["v0"] >= 1000.0 for r in rows])
    print(f"track_bench: {Path(a.ckpt).name} ({'greedy' if a.greedy else 'sampled'}), {len(ids)} own "
          f"states x {a.k} own primitive lines = {len(rows)} flights, seed {a.seed}")
    print(f"   cross-track error, per-flight mean: median {np.median(em):,.0f} u, p90 "
          f"{np.percentile(em, 90):,.0f} | per-flight max: median {np.median(ex):,.0f}, p90 "
          f"{np.percentile(ex, 90):,.0f}")
    if fast.any():
        print(f"   starts at |v| >= 1,000 ({int(fast.sum())} flights): per-flight mean error median "
              f"{np.median(em[fast]):,.0f} u; reached the end {sum(r['reach'] for r, f in zip(rows, fast) if f)}"
              f"/{int(fast.sum())}; alive {sum(r['alive'] for r, f in zip(rows, fast) if f)}")
    print(f"   reached the line's end (within {a.goal_radius:g} u): {sum(r['reach'] for r in rows)}/"
          f"{len(rows)} | alive at the end {len(al)}/{len(rows)} | energy retention median "
          f"{np.median(er) if len(er) else float('nan'):.3f} | end vh median "
          f"{np.median(vh) if len(vh) else 0:,.0f}")
    if a.out:
        h = hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest()[:16]
        Path(a.out).write_text(chr(10).join(json.dumps(dict(r, ckpt=str(a.ckpt), sha=h,
                                                          greedy=bool(a.greedy), seed=a.seed))
                                            for r in rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
