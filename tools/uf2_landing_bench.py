"""uf2_landing_bench.py - can a line-following mover LAND on unitfarmer2's lower A-frame and keep its
speed, when it is shown the right line?

    python tools/uf2_landing_bench.py <mover ckpt> [--t 4.0] [--n 64]

A MEASUREMENT bench (CLAUDE.md section 0: a record may be used to measure which skill a policy
lacks, never to train it). The mover is spawned from the human record's own state at time --t
(runs/research/gate_bench/uf2_pitwindow.npy, the record's dip window t 2.5-9 s, built for the
entry bench) and flies 2 s along three kinds of line:
  record : the record's own path from --t on (the landing it made), resampled at 128 u;
  coast  : straight along the current 3-D velocity (the archive's rays4 continuation);
  random : step-1 primitives (the loop archive's moves).
Reported: the speed and the horizontal speed 0.5 / 1.0 / 2.0 s later, deaths, and the record's own
numbers for comparison. If the mover keeps ~1,750 u/s on the record line, the skill exists and the
search must find the line; if not, the mover cannot land.
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--t", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true", help="the mover acts greedily (default: samples)")
    ap.add_argument("--out", default=None, help="write one JSON row per flight here")
    a = ap.parse_args(argv)
    import record_ckpt
    import edge_archive as ea
    from surfgym.route import resample_polyline
    win = np.load(WIN)
    j = int(round((a.t - T0) / (T1 - T0) * (len(win) - 1)))
    st = win[j].copy()
    st["tick"] = 0
    st["stuck_ticks"] = 0
    rec = []
    for ln in open(REC, encoding="utf-8"):
        if ln.startswith("["):
            rec.append(json.loads(ln)[:7])
        elif ln.startswith('{"end'):
            break
    rec = np.asarray(rec, np.float64)
    k0 = int(np.argmin(np.linalg.norm(rec[:, 1:4] - st["origin"].astype(np.float64), axis=1)))
    import torch
    torch.manual_seed(int(a.seed))              # the policy's sampling (Codex: pair the cohorts)
    ctx = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--plan-scratch", str(a.n)]
                            + ([] if a.greedy else ["--stochastic"])
                            + ["--map", str(ROOT / "maps_pool" / "surf_unitfarmer2.bsp")])
    fin = np.zeros(3)
    ops = {"record": None, "coast": ea.RayOperator(float(ctx.tick.ms), fin, n=4),
           "random": ea.PrimOperator(float(ctx.tick.ms), fin, 1, a.seed),
           # the SURF LINE (edge_archive.surf_curve): free flight to the surface of impact, then
           # along its tangent - from the simulator's own trace, no record input
           "surf": ea.SurfOperator(float(ctx.tick.ms), fin, 0, a.seed, ctx.scratch.core),
           # the TANGENT-APPROACH line (edge_archive.tangent_curve): arrive in the contact plane
           "tangent": ea.SurfOperator(float(ctx.tick.ms), fin, 0, a.seed, ctx.scratch.core,
                                      kind="tangent")}
    rline, _ = resample_polyline(rec[k0:k0 + 260, 1:4], ea.RAY_SPACING)
    rline = np.asarray(rline, np.float32)

    class _Fixed:
        n_choice = 1
        commit_ticks = budget_ticks = 200
        choice_nums = np.zeros((1, 1))
        finish = fin

        def line_and_curve_of(self, origin, velocity, yaw_deg, nums, k=None):
            return rline, rline

    # the start: a real state, its observation from one neutral step (the Flyer needs one)
    core = ctx.scratch.core
    for i in range(core.num_envs):
        core.set_state(i, st)
    # ENGINE NEUTRAL (Codex 2026-09-27): yaw bin 7 / pitch bin 3 = no turn, fwd 1 / side 1 = no
    # key, no jump, no duck; the earlier all-zero row turned the view 10 deg and pressed back+left
    acts = np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (core.num_envs, 1))
    obs, *_ = core.step(acts)
    st1 = core.get_states()[0].copy()
    st1["tick"] = 0
    st1["stuck_ticks"] = 0
    obs0 = np.array(obs[0], np.float32, copy=True)
    t_rec = rec[k0, 0] / 100.0
    print(f"uf2_landing_bench: {Path(a.ckpt).name} from the record's state at t {t_rec:.2f} s: "
          f"pos {np.round(st1['origin']).astype(int).tolist()}, |v| "
          f"{np.linalg.norm(st1['velocity']):,.0f}, vh {np.hypot(*st1['velocity'][:2]):,.0f}")
    for dt in (0.5, 1.0, 2.0):
        r = rec[k0 + int(dt * 100)]
        print(f"   the record {dt:.1f} s later: |v| {np.linalg.norm(r[4:7]):,.0f}, vh "
              f"{np.hypot(r[4], r[5]):,.0f}, z {r[3]:,.0f}")
    rows = []
    for name, op in ops.items():
        ctx.planner = op if op is not None else _Fixed()
        fl = ea.Flyer(ctx)
        arch = ea.Archive()
        nid = arch.add(st1, fl.fresh_keys(), obs0, ("R",), -1, -1, 0, 0, None)
        speeds = {0.5: [], 1.0: [], 2.0: []}
        deaths = 0
        # three probes per horizon: fly dt seconds (dur = dt), read the end states
        for dt in (0.5, 1.0, 2.0):
            fl.dur = int(round(dt * 100))
            jobs = [(nid, 3 if name == "coast" else 0)] * min(a.n, fl.S)
            res = fl.fly(jobs, arch, fin, record_path=False)
            for r in res:
                if r["end"] is None:
                    deaths += int(r["died"])
                    rows.append({"line": name, "dt": dt, "alive": False})
                    continue
                v = np.asarray(r["end"]["velocity"], np.float64)
                speeds[dt].append((np.linalg.norm(v), np.hypot(v[0], v[1])))
                rows.append({"line": name, "dt": dt, "alive": True,
                             "speed": float(np.linalg.norm(v)), "vh": float(np.hypot(v[0], v[1])),
                             "origin": np.round(r["end"]["origin"].astype(np.float64), 1).tolist()})
        line = []
        for dt, sv in speeds.items():
            if sv:
                sv = np.asarray(sv)
                line.append(f"{dt:.1f} s: |v| median {np.median(sv[:, 0]):,.0f} max "
                            f"{sv[:, 0].max():,.0f}, vh median {np.median(sv[:, 1]):,.0f} max "
                            f"{sv[:, 1].max():,.0f} ({len(sv)} alive)")
        n_f = min(a.n, fl.S)
        # the joint endpoint (Codex): dead flights count as failures - the share of ALL flights
        # alive AND at >= 1,400 u/s horizontal after 1 s
        ok1 = sum(1 for r in rows if r["line"] == name and r["dt"] == 1.0 and r["alive"]
                  and r["vh"] >= 1400.0)
        print(f"   {name:6s}: " + " | ".join(line) + f" | deaths {deaths} | alive AND vh >= 1,400 "
              f"at 1 s: {ok1}/{n_f}")
    if a.out:
        import hashlib
        h = hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest()[:16]
        Path(a.out).write_text(chr(10).join(json.dumps(dict(r, ckpt=str(a.ckpt), sha=h,
                                                          greedy=bool(a.greedy), seed=a.seed,
                                                          t=a.t)) for r in rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
