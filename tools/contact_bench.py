"""contact_bench.py - does the primitive VOCABULARY limit ramp landings? A paired operator bench on
the agent's OWN pre-contact states (Codex, agent bus 2026-09-27 06:57).

    python tools/contact_bench.py <mover ckpt> <archive_states.npy> --map maps_pool/<m>.bsp
        [--n-states 48] [--k 4] [--seed 0] [--out rows.jsonl]

States: rows of an edge archive's own state dump (policy-owned; no record, no map box), kept when
they are AIRBORNE and a standing-hull ballistic trace (the core's own trace, gravity only) meets a
surface within 0.5 s - "about to touch something", a generic criterion. For each state K primitive
knot draws (seeded) are flown TWICE by the same mover with paired policy randomness: the knots as
turn RATES (--prim-turn rate) and as CURVATURES (curv, horizontal speed). Reported per operator:
alive after 2 s, mechanical-energy retention E_end / E_start (|v|^2 / 2 + g z, the agent's own
energy, no threshold) and horizontal speed. The mover's own primitive config otherwise.
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
G = 800.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("states")
    ap.add_argument("--map", required=True)
    ap.add_argument("--n-states", type=int, default=48)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch
    import record_ckpt
    import edge_archive as ea
    from surfgym.goalprim import PRIM_DEFAULTS, PrimitivePlanner
    from surfgym.route import resample_polyline
    rng = np.random.default_rng(int(a.seed))
    torch.manual_seed(int(a.seed))
    ctx = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--plan-scratch", "192", "--stochastic",
                             "--map", str(a.map)])
    core = ctx.scratch.core
    st = np.load(a.states)
    st = np.unique(st)                                    # the goid dump repeats rows
    air = st[np.asarray(st["onground"]) == -1]
    # the generic pre-contact filter: a ballistic hull trace meets a surface within 0.5 s
    cand = []
    for i in rng.permutation(len(air)):
        o = air[i]["origin"].astype(np.float64)
        v = air[i]["velocity"].astype(np.float64)
        p, vv, t, hit = o.copy(), v.copy(), 0.0, False
        while t < 0.5:
            q = p + vv * 0.05 + 0.5 * np.array([0, 0, -G]) * 0.05 ** 2
            tr = core.trace(p.tolist(), q.tolist(), 0)
            if tr.fraction < 1.0 and not tr.startsolid:
                hit = True
                break
            vv = vv + np.array([0, 0, -G]) * 0.05
            p, t = q, t + 0.05
        if hit:
            cand.append(air[i])
        if len(cand) >= a.n_states:
            break
    cand = np.asarray(cand)
    print(f"contact_bench: {len(air):,} airborne own states, {len(cand)} about to touch a surface "
          f"within 0.5 s (of {len(st):,} unique)")
    cfg = getattr(ctx, "cfg", None) or {}
    d = {k_: (cfg.get(k_) if cfg.get(k_) is not None else v_) for k_, v_ in PRIM_DEFAULTS.items()}
    planners = {t: PrimitivePlanner(secs=float(d["prim_secs"]), knots=int(d["prim_knots"]),
                                    side=float(d["prim_side"]), down=float(d["prim_down"]),
                                    up=float(d["prim_up"]), floor=float(d["prim_floor"]),
                                    spacing=ea.RAY_SPACING,
                                    frame=str(cfg.get("prim_frame") or "velocity"), turn=t)
                for t in ("rate", "curv")}
    draws = [[planners["rate"].sample(rng) for _ in range(a.k)] for _ in range(len(cand))]

    class _Fixed:
        """one fixed knot vector per job (set before each fly)"""
        n_choice = 1
        choice_nums = np.zeros((1, 1))
        finish = np.zeros(3)

        def __init__(self, pp):
            self.pp = pp
            self.queue = []
            self.commit_ticks = self.budget_ticks = int(round(pp.secs * 100))

        def line_and_curve_of(self, origin, velocity, yaw_deg, nums, k=None):
            p = self.queue.pop(0) if self.queue else self.queue_last
            self.queue_last = p
            return self.pp.line_and_curve(origin, velocity, yaw_deg, p)
    rows = []
    for turn in ("rate", "curv"):
        op = _Fixed(planners[turn])
        ctx.planner = op
        fl = ea.Flyer(ctx)
        arch = ea.Archive()
        ids = []
        # every start gets a real observation: set it, one engine-neutral tick, read state + obs
        neutral = np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (core.num_envs, 1))
        for c0 in range(0, len(cand), core.num_envs):
            part = cand[c0:c0 + core.num_envs]
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
        jobs = [(ids[i], 0) for i in range(len(cand)) for _ in range(a.k)]
        op.queue = [draws[i][j] for i in range(len(cand)) for j in range(a.k)]
        op.queue_last = op.queue[-1]
        torch.manual_seed(int(a.seed) + 1)            # paired policy randomness across operators
        res = []
        for c0 in range(0, len(jobs), fl.S):
            chunk = jobs[c0:c0 + fl.S]
            pad = chunk + [chunk[-1]] * (fl.S - len(chunk))
            op.queue = ([draws[j // a.k][j % a.k] for j in range(c0, c0 + len(chunk))]
                        + [draws[(c0 + len(chunk) - 1) // a.k][(c0 + len(chunk) - 1) % a.k]]
                        * (fl.S - len(chunk)))
            res += fl.fly(pad, arch, np.zeros(3), record_path=False)[:len(chunk)]
        for (nid, _k), r in zip(jobs, res):
            s0 = arch.state[nid]
            e0 = 0.5 * float(np.sum(s0["velocity"].astype(np.float64) ** 2)) + G * float(s0["origin"][2])
            if r["end"] is None:
                rows.append({"turn": turn, "alive": False})
                continue
            v = r["end"]["velocity"].astype(np.float64)
            e1 = 0.5 * float(np.sum(v ** 2)) + G * float(r["end"]["origin"][2])
            rows.append({"turn": turn, "alive": True, "vh": float(np.hypot(v[0], v[1])),
                         "e0": e0, "e1": e1})
    for turn in ("rate", "curv"):
        rr = [r for r in rows if r["turn"] == turn]
        al = [r for r in rr if r["alive"]]
        er = np.asarray([r["e1"] - r["e0"] for r in al]) if al else np.zeros(0)
        vh = np.asarray([r["vh"] for r in al]) if al else np.zeros(0)
        print(f"   {turn}: alive at 2 s {len(al)}/{len(rr)} | energy change (u^2/s^2) median "
              f"{np.median(er) if len(er) else float('nan'):,.0f}, p90 "
              f"{np.percentile(er, 90) if len(er) else float('nan'):,.0f} | vh median "
              f"{np.median(vh) if len(vh) else 0:,.0f}, p90 {np.percentile(vh, 90) if len(vh) else 0:,.0f}")
    if a.out:
        Path(a.out).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
