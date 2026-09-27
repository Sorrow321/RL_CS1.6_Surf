"""ramp_steer.py - can the ramp COMMAND be steered at all? From the same policy-owned states,
command different surfaces and measure where the mover's first new contact lands (the target-
contact confusion; Codex 17:17Z step 4 / Fable section 5 point 4, with LINE transport - no
training). MEASUREMENT only.

    python tools/ramp_steer.py <mover ckpt> <archive_states.npy> --ramps <ramps2.npz>
        --map maps_pool/<m>.bsp [--n-states 40] [--targets 4] [--k 16] [--seed 0]

For each of --n-states source states (drawn from an edge_archive --dump-states file: the search's
own states), the RampOperator's coast order gives the node's candidate commands; the first
--targets candidates are each commanded --k times (sampled executor, the same torch seed per
batch). A flight ends at its first new contact after departure, a death or the 6 s timeout.

Reported: P(first contact = B | command B) against P(first contact = B | another command from the
same state) - the lift, pooled over (state, B) pairs; per-rank hit rates; and how often two
commands from one state produce different first-contact distributions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("states")
    ap.add_argument("--ramps", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--n-states", type=int, default=40)
    ap.add_argument("--targets", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    import record_ckpt
    import edge_archive as ea
    from ramps import RampMap
    rng = np.random.default_rng(int(a.seed))
    torch.manual_seed(int(a.seed))
    S = int(a.targets) * int(a.k)
    ctx = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--plan-scratch", str(max(S, 64)),
                             "--stochastic", "--ep-ticks",
                             str(int(round(ea.RAMP_TIMEOUT * 100)) + 400), "--map", str(a.map)])
    core = ctx.scratch.core
    fb = ctx.finish_box
    fin = 0.5 * (np.asarray(fb[0], np.float64) + np.asarray(fb[1], np.float64))
    rm = RampMap(a.ramps)
    op = ea.RampOperator(float(ctx.tick.ms), fin, rm, int(a.targets))
    ctx.planner = op
    fl = ea.Flyer(ctx)
    fl.ramp_map = rm
    st = np.unique(np.load(a.states))
    pick = st[rng.choice(len(st), size=min(int(a.n_states), len(st)), replace=False)]
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
    op.plan(fl, arch, ids)
    rows = []          # (state, rank, commanded surface, first contact or -1 / -2 died)
    for nid in ids:
        cands = op.rank[nid][:int(a.targets)]
        jobs = [(nid, c) for c in cands for _ in range(int(a.k))]
        if not jobs:
            continue
        for c0 in range(0, len(jobs), fl.S):
            chunk = jobs[c0:c0 + fl.S]
            op.queue = [p for (p, _c) in chunk] + [chunk[-1][0]] * (fl.S - len(chunk))
            res = fl.fly(chunk + [chunk[-1]] * (fl.S - len(chunk)), arch, fin,
                         record_path=False)[:len(chunk)]
            for (p, c), r in zip(chunk, res):
                tgt = op.targets[c] if c < op.FIN else -3
                first = (-2 if (r["died"] and not r["fin"]) else
                         (int(r["hit"]) if r.get("hit") is not None else -1))
                rows.append((p, cands.index(c), tgt, first))
    rows = np.asarray(rows, np.int64)
    # the lift: P(first = B | command B) vs P(first = B | another command from the same state)
    hit_own, hit_other, n_own, n_other = 0, 0, 0, 0
    distinct, pairs = 0, 0
    for nid in ids:
        rr = rows[rows[:, 0] == nid]
        if len(rr) == 0:
            continue
        tg = np.unique(rr[:, 2])
        for b in tg:
            own = rr[rr[:, 2] == b]
            oth = rr[rr[:, 2] != b]
            hit_own += int((own[:, 3] == b).sum())
            n_own += len(own)
            hit_other += int((oth[:, 3] == b).sum())
            n_other += len(oth)
        for i_, b1 in enumerate(tg):
            for b2 in tg[i_ + 1:]:
                d1 = rr[rr[:, 2] == b1][:, 3]
                d2 = rr[rr[:, 2] == b2][:, 3]
                allv = np.union1d(d1, d2)
                h1 = np.array([(d1 == v).mean() for v in allv])
                h2 = np.array([(d2 == v).mean() for v in allv])
                pairs += 1
                distinct += int(0.5 * np.abs(h1 - h2).sum() >= 0.25)
    print(f"ramp_steer: {Path(a.ckpt).name} on {Path(a.map).stem}: {len(ids)} own states x "
          f"{a.targets} coast-ordered commands x {a.k} sampled flights = {len(rows)} flights")
    print(f"   P(first contact = B | command B) = {hit_own}/{n_own} = "
          f"{hit_own / max(1, n_own):.3f}   vs   P(first contact = B | another command, same "
          f"state) = {hit_other}/{n_other} = {hit_other / max(1, n_other):.3f}   lift "
          f"{(hit_own / max(1, n_own)) / max(1e-9, hit_other / max(1, n_other)):.2f}x")
    for rk in range(int(a.targets)):
        rr = rows[rows[:, 1] == rk]
        if len(rr):
            print(f"   rank {rk}: hit own target {int((rr[:, 3] == rr[:, 2]).sum())}/{len(rr)}, "
                  f"another surface {int(((rr[:, 3] >= 0) & (rr[:, 3] != rr[:, 2])).sum())}, "
                  f"no contact {int((rr[:, 3] == -1).sum())}, died {int((rr[:, 3] == -2).sum())}")
    print(f"   command pairs from one state whose first-contact distributions differ (total "
          f"variation >= 0.25): {distinct}/{pairs}")
    from ramps import CATS
    oth = rows[(rows[:, 3] >= 0) & (rows[:, 3] != rows[:, 2])]
    if len(oth):
        cc = np.bincount(rm.cat[oth[:, 3]], minlength=4)
        print("   'another surface' by category: " + ", ".join(f"{CATS[i]} {int(cc[i])}"
                                                            for i in range(4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
