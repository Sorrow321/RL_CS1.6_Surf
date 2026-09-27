"""energy_panel.py - the marginal momentum cost of the search executor's YAW sampling noise, on a
frozen multi-map panel of the executor's own airborne states (Codex 2026-09-27 21:16Z protocol).
MEASUREMENT only; it chooses one executor constant and must never be tuned per map.

    python tools/energy_panel.py <mover ckpt> --panel MAP.bsp:STATES.npy [...]
        [--yaw-scales 1,0.5,0.25,0.1] [--n 48] [--horizon 0.3] [--out panel.npz]

1. ELIGIBILITY on the ACTUAL post-bootstrap state: each candidate state is restored, stepped one
   neutral tick, and kept only if it did not reset, touched nothing on that tick (the core's
   contact telemetry), is airborne (onground < 0) and its speed lies in a stratum:
   [600, 1000), [1000, 2000), [2000, inf) u/s. Up to --n per (map, stratum), drawn with a fixed
   seed; the exact post-bootstrap states and their md5s are saved.
2. PAIRED BRANCHES: every stratum's states are flown under each yaw sigma scale s AND under the
   REFERENCE yaw scale 1e-6 (the policy's mean yaw), with pitch and the categorical keys NATIVE
   and the torch RNG re-seeded per branch - identical key Gumbels and pitch draws, only the yaw
   residual differs. The command is the search's own straight-ahead line (RayOperator ray 0).
3. COMMON HORIZON H (--horizon s): per state, the reference's first contact or death before H
   CENSORS the pair; else a treatment contact / death before H is a treatment EVENT (a paired
   failure, counted); else the paired excess work is (dE_s(H) - dE_ref(H)) / KE0 / H, with
   E = 0.5 |v|^2 + 800 z (its z origin cancels in the difference).

THE RULE (declared 2026-09-27 before this panel ran): a (map, stratum) cell is DECISIVE with >= 20
uncensored pairs. Yaw scale s PASSES a cell iff the median paired excess >= -1 % KE0/s, its lower
quartile >= -3 % KE0/s, and its event fraction (events / uncensored pairs) <= 10 %. The executor's
yaw scale for the search is the LARGEST s passing every decisive cell; pitch and keys stay native;
native yaw (s = 1) passing means no change. Full greedy and the free-air physics bound are NOT the
reference (greedy also argmaxes the keys).

Output: the table, the chosen scale, and --out (.npz): per (map, stratum, scale) the per-tick
energy, contact count, death flag, yaw command and action rows, the source indices, the state
md5s, and a JSON header (argv, git HEAD, md5 of the checkpoint / states / map / core DLL).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

G = 800.0
STRATA = ((600.0, 1000.0), (1000.0, 2000.0), (2000.0, float("inf")))
REF = 1e-6
MIN_PAIRS, MED_TOL, LQ_TOL, EVENT_TOL = 20, -0.01, -0.03, 0.10


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--panel", nargs="+", required=True, help="MAP.bsp:STATES.npy pairs")
    ap.add_argument("--yaw-scales", default="1,0.5,0.25,0.1")
    ap.add_argument("--n", type=int, default=48, help="states per (map, stratum)")
    ap.add_argument("--horizon", type=float, default=0.3, help="seconds (the common horizon)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch
    import record_ckpt
    import edge_archive as ea
    scales = [float(s) for s in a.yaw_scales.split(",")]
    branches = scales + [REF]
    traces = {}
    table = {}
    header = {"argv": sys.argv, "rule": {"min_pairs": MIN_PAIRS, "median_tol": MED_TOL,
                                         "lq_tol": LQ_TOL, "event_tol": EVENT_TOL,
                                         "ref_yaw_scale": REF, "strata": STRATA[:2] + ((2000.0, None),)},
              "ckpt_md5": md5(a.ckpt), "maps": {}}
    try:
        header["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                            capture_output=True, text=True).stdout.strip()
    except Exception:                       # pragma: no cover - no git
        header["git_head"] = None
    dll = os.environ.get("SURFCORE_DLL")
    header["core_dll_md5"] = md5(dll) if dll and Path(dll).exists() else None
    for pair in a.panel:
        mp, _, sp = pair.partition(".bsp:")
        mp += ".bsp"
        st_raw = np.load(sp)
        st_all, first_ix = np.unique(st_raw, return_index=True)
        header["maps"][Path(mp).stem] = {"map_md5": md5(mp), "states": sp, "states_md5": md5(sp)}
        # 1. eligibility on the post-bootstrap state, in one pass over a big scratch core
        ctx = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--plan-scratch", "256",
                                 "--ep-ticks", str(int(round(a.horizon * 100)) + 400),
                                 "--map", str(mp), "--stochastic",
                                 "--exec-view-scale", f"{REF},1"])
        core = ctx.scratch.core
        S = core.num_envs
        tick_s = float(ctx.tick.ms) / 1000.0
        H = int(round(a.horizon / tick_s))
        rng = np.random.default_rng(int(a.seed))
        order = rng.permutation(len(st_all))
        neutral = np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (S, 1))
        elig = {k: [] for k in range(len(STRATA))}           # stratum -> [(file index, state)]
        for c0 in range(0, len(order), S):
            if all(len(v) >= int(a.n) for v in elig.values()):
                break
            idx = order[c0:c0 + S]
            for i in range(S):
                s2 = st_all[idx[min(i, len(idx) - 1)]].copy()
                s2["tick"] = 0
                s2["stuck_ticks"] = 0
                core.set_state(i, s2)
            _o, _r, done, trunc, _ = core.step(neutral)
            cnt, _tn, _tp = core.get_touch()
            cur = core.get_states()
            for i in range(len(idx)):
                if done[i] or trunc[i] or cnt[i] > 0 or cur[i]["onground"] >= 0:
                    continue
                sp_ = float(np.linalg.norm(cur[i]["velocity"].astype(np.float64)))
                for k, (lo, hi) in enumerate(STRATA):
                    if lo <= sp_ < hi and len(elig[k]) < int(a.n):
                        s3 = cur[i].copy()
                        s3["tick"] = 0
                        s3["stuck_ticks"] = 0
                        elig[k].append((int(first_ix[idx[i]]), s3))
        print(f"energy_panel: {Path(mp).stem}: eligible airborne post-bootstrap states per stratum "
              + ", ".join(f"[{lo:g}, {hi:g}) {len(elig[k])}" for k, (lo, hi) in enumerate(STRATA)),
              flush=True)
        # 2. paired branches per stratum
        for k, (lo, hi) in enumerate(STRATA):
            pool = elig[k]
            n = len(pool)
            if n == 0:
                continue
            res = {}
            for sc in branches:
                ctx_b = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--plan-scratch",
                                           str(n), "--ep-ticks", str(H + 400), "--map", str(mp),
                                           "--stochastic", "--exec-view-scale", f"{sc:g},1"])
                cb = ctx_b.scratch.core
                fb = ctx_b.finish_box
                fin = 0.5 * (np.asarray(fb[0], np.float64) + np.asarray(fb[1], np.float64))
                op = ea.RayOperator(float(ctx_b.tick.ms), fin, n=3)
                ctx_b.planner = op
                fl = ea.Flyer(ctx_b)
                for i in range(n):
                    cb.set_state(i, pool[i][1])
                # the first observation: the SAME neutral tick every branch takes
                obs, *_ = cb.step(np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (n, 1)))
                obs = np.ascontiguousarray(np.asarray(obs, np.float32))
                cur = cb.get_states()
                lines = [op.line_and_curve_of(cur[i]["origin"].astype(np.float64),
                                              cur[i]["velocity"].astype(np.float64),
                                              float(cur[i]["yaw"]), op.choice_nums[0], 0)[0]
                         for i in range(n)]
                ctx_b.scratch.line.set_lines(np.arange(n), [ln[:fl.L_MAX] for ln in lines])
                torch.manual_seed(int(a.seed))          # identical draws in every branch
                pol = ctx_b.scratch.make_policy(cb, ctx_b.scratch.line)
                if fl.keys_hold:
                    from surfgym.keyshold import KeysHold
                    pol.keys = KeysHold(n)
                    pol._keys_tick = (cur["tick"].astype(np.int64)
                                      - int(getattr(pol, "_period", fl.K)))
                pol._tick = 0
                v0 = cur["velocity"].astype(np.float64)
                ke0 = 0.5 * np.sum(v0 ** 2, axis=1)
                e0 = ke0 + G * cur["origin"][:, 2].astype(np.float64)
                E = np.zeros((H, n))
                C = np.zeros((H, n), np.int32)
                D = np.zeros((H, n), bool)
                Y = np.full((H, n), np.nan)
                A = np.zeros((H, n, 6), np.int16)
                dead = np.zeros(n, bool)
                for t in range(H):
                    acts = pol.act(obs)
                    view = getattr(pol, "view", None)
                    obs, _r, done, trunc, _ = (cb.step(acts) if view is None
                                               else cb.step(acts, view=view))
                    obs = np.ascontiguousarray(np.asarray(obs, np.float32))
                    ended = np.asarray(done, bool) | np.asarray(trunc, bool)
                    dead |= ended
                    cnt, _tn, _tp = cb.get_touch()
                    cs = cb.states_view
                    E[t] = (0.5 * np.sum(cs["velocity"].astype(np.float64) ** 2, axis=1)
                            + G * cs["origin"][:, 2].astype(np.float64))
                    C[t] = np.asarray(cnt, np.int32)
                    D[t] = dead
                    A[t] = np.asarray(acts)[:n, :6]
                    if view is not None:
                        Y[t] = np.asarray(view, np.float64).reshape(n, -1)[:, 0]
                ev = np.full(n, H, np.int64)                  # first contact / death tick (H = none)
                for i in range(n):
                    w = np.flatnonzero((C[:, i] > 0) | D[:, i])
                    if len(w):
                        ev[i] = int(w[0])
                res[sc] = {"E": E, "C": C, "D": D, "Y": Y, "A": A, "ev": ev, "e0": e0, "ke0": ke0}
                traces[(Path(mp).stem, k, sc)] = res[sc]
            ref = res[REF]
            hs = H * tick_s
            for sc in scales:
                b = res[sc]
                cen = ref["ev"] < H
                evt = (~cen) & (b["ev"] < H)
                ok = (~cen) & (~evt)
                exc = ((b["E"][H - 1] - b["e0"]) - (ref["E"][H - 1] - ref["e0"])) / np.maximum(
                    ref["ke0"], 1.0) / hs
                x = exc[ok]
                npair = int((~cen).sum())
                med = float(np.median(x)) if len(x) else float("nan")
                lq = float(np.percentile(x, 25)) if len(x) else float("nan")
                efr = float(evt.sum()) / max(1, npair)
                decisive = npair >= MIN_PAIRS
                passed = bool(decisive and med >= MED_TOL and lq >= LQ_TOL and efr <= EVENT_TOL)
                table[(Path(mp).stem, k, sc)] = {"pairs": npair, "clean": int(ok.sum()),
                                                  "events": int(evt.sum()),
                                                  "censored": int(cen.sum()), "median": med,
                                                  "lq": lq, "event_frac": efr,
                                                  "decisive": decisive, "pass": passed}
                print(f"   [{lo:g}, {hi:g}) yaw x{sc:g}: {npair} uncensored pairs "
                      f"({int(cen.sum())} censored by the reference), treatment events "
                      f"{int(evt.sum())} ({100 * efr:.0f}%), paired excess median "
                      f"{100 * med:+.2f}% KE0/s, lower quartile {100 * lq:+.2f}%"
                      f" -> {'PASS' if passed else ('fail' if decisive else 'not decisive')}",
                      flush=True)
    cells = {(m, k) for (m, k, _s) in table}
    chosen = None
    for sc in sorted(scales, reverse=True):
        dec = [table[(m, k, sc)] for (m, k) in cells if table[(m, k, sc)]["decisive"]]
        if dec and all(c["pass"] for c in dec):
            chosen = sc
            break
    print(f"energy_panel: decisive cells {sum(1 for (m, k) in cells if table[(m, k, scales[0])]['decisive'])}"
          f" of {len(cells)}; the rule chooses yaw scale "
          f"{'x%g' % chosen if chosen is not None else 'NONE (no scale passes every decisive cell)'}"
          f" (pitch and keys native)")
    if a.out:
        blob = {}
        for (m, k, sc), r in traces.items():
            for key_, arr in r.items():
                blob[f"{m}|{k}|{sc:g}|{key_}"] = arr
        header["table"] = {f"{m}|{k}|{sc:g}": v for (m, k, sc), v in table.items()}
        header["chosen_yaw_scale"] = chosen
        np.savez_compressed(a.out, header=np.array(json.dumps(header)), **blob)
        print(f"   traces -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
