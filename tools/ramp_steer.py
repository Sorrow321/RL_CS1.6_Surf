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
    ap.add_argument("--out", default=None, help="raw per-flight rows (.jsonl, header first)")
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
    pick_ix = rng.choice(len(st), size=min(int(a.n_states), len(st)), replace=False)
    pick = st[pick_ix]
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
        obs, _r, done, trunc, _ = core.step(neutral)
        cur = core.get_states()
        reset = np.asarray(done, bool) | np.asarray(trunc, bool)
        for i in range(len(part)):
            if reset[i]:
                continue                    # the neutral step ended it: the row is a respawn now
            s3 = cur[i].copy()
            s3["tick"] = 0
            s3["stuck_ticks"] = 0
            ids.append(arch.add(s3, fl.fresh_keys(), np.array(obs[i], np.float32, copy=True),
                                ("S",), -1, -1, 0, 0, None))
    op.plan(fl, arch, ids)

    ak = arch.token

    def clear(nid, c):
        """is the command's drawn line free of the standing hull's collisions until it arrives
        (the Hermite part)? a line through a wall is not a flyable direct transition"""
        if c >= op.FIN:
            return True                             # the finish box: no surface line to test
        st = arch.state[nid]
        op.queue = [(ak, nid)]
        _line, pts = op.line_and_curve_of(st["origin"].astype(np.float64),
                                          st["velocity"].astype(np.float64),
                                          float(st["yaw"]), op.choice_nums[c], c)
        n_arr = max(2, len(pts) - int(ea.RAMP_TAIL / 0.01))
        for i in range(0, n_arr - 5, 5):
            tr = core.trace(pts[i].tolist(), pts[i + 5].tolist(), 0)
            if tr.startsolid or tr.fraction < 1.0:
                return i + 5 >= n_arr - 5           # blocked only at the very end = arrived
        return True
    clr = {}
    rows = []          # one dict per flight: every field the estimand needs, persisted raw
    for si, nid in enumerate(ids):
        cands = op.rank[(ak, nid)][:int(a.targets)]
        jobs = [(nid, c, j) for c in cands for j in range(int(a.k))]
        for c0 in range(0, len(jobs), fl.S):
            chunk = jobs[c0:c0 + fl.S]
            pad = chunk + [chunk[-1]] * (fl.S - len(chunk))
            op.queue = [(ak, p) for (p, _c, _j) in pad]
            res = fl.fly([(p, c) for (p, c, _j) in pad], arch, fin, record_path=False)[:len(chunk)]
            for (p, c, j), r in zip(chunk, res):
                if (p, c) not in clr:
                    clr[(p, c)] = clear(p, c)
                rows.append({"state": int(pick_ix[si]), "nid": int(p), "rank": cands.index(c),
                             "cmd": int(c), "target": (int(op.targets[c]) if c < op.FIN else -3),
                             "rep": int(j), "src": r.get("src"), "src_set": r.get("src_set"),
                             # the FIRST contact is kept even when a death follows it before the
                             # next decision boundary (Codex 19:29Z): death is its own column
                             "hit": r.get("hit"), "hit_set": list(r.get("hit_set") or []),
                             "hit_tick": r.get("hit_tick"), "died": bool(r["died"]),
                             "fin": bool(r["fin"]), "ticks": int(r["ticks"]),
                             "clear": int(clr[(p, c)])})

    def label(r):
        """the flight's FIRST outcome, by the search's own rule (edge_archive.ramp_outcome): a
        contact first -> its NEW set; else the finish; else died / none"""
        ht = r["hit_tick"]
        contact = ht is not None and ht >= 0 and not (r["fin"] and ht >= r["ticks"])
        if contact:
            return "hit:" + ",".join(str(x) for x in r["hit_set"])
        if r["fin"]:
            return "fin"
        return "died" if r["died"] else "none"

    def event(r, b):
        """was flight r's first outcome B ALONE? (B = -3: the finish before any contact) - the
        same 'direct' as the search's stats, so {B, C} is not a B success"""
        return label(r) == ("fin" if b == -3 else f"hit:{b}")
    hit_own, hit_other, n_own, n_other = 0, 0, 0, 0
    distinct, pairs = 0, 0
    for nid in ids:
        rr = [r for r in rows if r["nid"] == nid]
        if not rr:
            continue
        tg = sorted({r["target"] for r in rr})
        for b in tg:
            own = [r for r in rr if r["target"] == b]
            oth = [r for r in rr if r["target"] != b]
            hit_own += sum(event(r, b) for r in own)
            n_own += len(own)
            hit_other += sum(event(r, b) for r in oth)
            n_other += len(oth)
        for i_, b1 in enumerate(tg):
            for b2 in tg[i_ + 1:]:
                d1 = [label(r) for r in rr if r["target"] == b1]
                d2 = [label(r) for r in rr if r["target"] == b2]
                allv = sorted(set(d1) | set(d2))
                h1 = np.array([d1.count(v) / len(d1) for v in allv])
                h2 = np.array([d2.count(v) / len(d2) for v in allv])
                pairs += 1
                distinct += int(0.5 * np.abs(h1 - h2).sum() >= 0.25)
    print(f"ramp_steer: {Path(a.ckpt).name} on {Path(a.map).stem}: {len(ids)} own states x "
          f"{a.targets} coast-ordered commands x {a.k} sampled flights = {len(rows)} flights")
    print(f"   P(first outcome = B | command B) = {hit_own}/{n_own} = "
          f"{hit_own / max(1, n_own):.3f}   vs   P(first outcome = B | another command, same "
          f"state) = {hit_other}/{n_other} = {hit_other / max(1, n_other):.3f}   lift "
          f"{(hit_own / max(1, n_own)) / max(1e-9, hit_other / max(1, n_other)):.2f}x")
    for rk in range(int(a.targets)):
        rr = [r for r in rows if r["rank"] == rk]
        if rr:
            own = sum(event(r, r["target"]) for r in rr)
            anoth = sum((not event(r, r["target"])) and bool(r["hit_set"]) for r in rr)
            print(f"   rank {rk}: own target {own}/{len(rr)}, another surface first {anoth}, "
                  f"no contact {sum((not r['hit_set']) and not r['fin'] for r in rr)}, "
                  f"died (any time) {sum(r['died'] for r in rr)}, finish commands "
                  f"{sum(r['target'] == -3 for r in rr)}")
    print(f"   command pairs from one state whose first-outcome distributions differ (total "
          f"variation >= 0.25): {distinct}/{pairs}")
    for cl, name in ((1, "CLEAR lines (hull-trace free until arrival)"), (0, "BLOCKED lines")):
        rr = [r for r in rows if r["clear"] == cl]
        if rr:
            own = sum(event(r, r["target"]) for r in rr)
            print(f"   {name}: {len(rr)} flights, own target {own} ({own / len(rr):.3f}), died "
                  f"{sum(r['died'] for r in rr)}")
    from ramps import CATS
    cc = np.zeros(len(CATS), np.int64)
    for r in rows:
        if r["hit_set"] and not event(r, r["target"]):
            for s_ in r["hit_set"]:
                if s_ >= 0:
                    cc[int(rm.cat[s_])] += 1
    if cc.sum():
        print("   'another surface' contacts by category: "
              + ", ".join(f"{CATS[i]} {int(cc[i])}" for i in range(len(CATS))))
    if a.out:
        import hashlib
        import json

        def md5(p):
            return hashlib.md5(Path(p).read_bytes()).hexdigest()
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"argv": sys.argv, "seed": int(a.seed),
                                "rng": "torch global, seeded once per invocation "
                                       "(no per-flight counter tape yet)",
                                "md5": {"ckpt": md5(a.ckpt), "states": md5(a.states),
                                        "ramps": md5(a.ramps), "map": md5(a.map)}}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"   raw rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
