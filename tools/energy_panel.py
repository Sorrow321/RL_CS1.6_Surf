"""energy_panel.py (v3) - the marginal momentum cost of the search executor's YAW sampling noise,
on a frozen multi-map panel (Codex 2026-09-27 21:16Z / 21:35Z protocol). MEASUREMENT only; it
chooses ONE executor constant for every map and must never be tuned per map.

    python tools/energy_panel.py <mover ckpt> --panel MAP.bsp:STATES.npy [...]
        [--yaw-scales 1,0.5,0.25,0.1] [--n 48] [--out panel.npz]

FIXTURE (declared): a STANDARDIZED physics/controller bench, not a search-node continuation. Each
candidate is an original zero-clocked SurfState from a frozen collector's dump; before every batch
the core is reset (a clean hidden state: PmPersist, triggers), the candidate is restored and ONE
neutral tick is taken (the bootstrap); the executor starts from that tick's observation with
fresh held keys. Every branch replays exactly this from the ORIGINAL candidate and must reproduce
the eligibility pass's post-bootstrap state bit for bit (origin + velocity hash), else the state
is dropped from that comparison and counted.

ELIGIBILITY (post-bootstrap): no reset, no contact on the bootstrap tick (core telemetry),
airborne (onground < 0), HORIZONTAL speed in a stratum [600, 1000), [1000, 2000), [2000, inf)
(the air wish is horizontal). Up to --n per (map, stratum), a fixed seed.

BRANCHES: yaw sigma x s for each s, and the REFERENCE yaw x1e-6 (the policy's mean yaw); pitch and
the categorical keys NATIVE; torch re-seeded per branch, so the first decision's key Gumbels, pitch
draw and observation are identical - only the yaw residual differs.

THE DECISIVE HORIZON is ONE controller interval (act_every ticks: 40 ms for prim1). Per state:
the reference's contact or death inside it CENSORS the pair; a treatment death inside it is a
DEATH event; a treatment contact inside it is a CONTACT (reported, excluded from the energy
statistic, not a failure); otherwise the paired excess work is
    (dE_s - dE_ref) / KE_h0 / interval,   E = 0.5 |v|^2 + 800 z,   KE_h0 = 0.5 |v_h|^2
(yaw moves only the horizontal velocity in the air, so the gravity terms cancel in the pair).
The same statistic at 0.3 s is reported as the CLOSED-LOOP effect (7-8 decisions; not decisive).

THE RULE (declared 2026-09-27, committed before this version ran):
  * a (map, stratum) cell with >= 20 eligible states is POPULATED; a populated cell is DECISIVE
    iff >= 20 pairs are uncensored AND at most 50 % are censored;
  * the panel is CONCLUSIVE only if every populated cell is decisive and every map has at least
    one decisive cell - otherwise the verdict is INCONCLUSIVE and nothing is chosen;
  * yaw scale s PASSES a decisive cell iff the median excess >= -1 % KE_h0/s, its lower quartile
    >= -3 % KE_h0/s, and its death fraction (deaths / uncensored pairs) <= 10 %;
  * the chosen yaw scale is the LARGEST s passing every decisive cell (s = 1 passing: no change);
    pitch and keys stay native.

Output: the table, the verdict, and --out (.npz): per cell the original source indices, the
original and expected post-bootstrap states, their hashes; per branch the start-state hashes and
match flags, per-tick energy, contact counts, death flags, the full view vector (yaw, pitch) and
the action rows; a JSON header (argv, git HEAD, rule, fixture, md5 of checkpoint / states / map /
core DLL).
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
POP, MIN_PAIRS, MAX_CENS = 20, 20, 0.50
MED_TOL, LQ_TOL, DEATH_TOL = -0.01, -0.03, 0.10
LONG_SECS = 0.3


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def st_hash(st):
    return hashlib.md5(np.ascontiguousarray(st["origin"]).tobytes()
                       + np.ascontiguousarray(st["velocity"]).tobytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--panel", nargs="+", required=True, help="MAP.bsp:STATES.npy pairs")
    ap.add_argument("--yaw-scales", default="1,0.5,0.25,0.1")
    ap.add_argument("--n", type=int, default=48, help="states per (map, stratum)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch
    import record_ckpt
    import edge_archive as ea
    scales = [float(s) for s in a.yaw_scales.split(",")]
    branches = scales + [REF]
    header = {"argv": sys.argv, "version": 3,
              "rule": {"populated": POP, "min_pairs": MIN_PAIRS, "max_censored": MAX_CENS,
                       "median_tol": MED_TOL, "lq_tol": LQ_TOL, "death_tol": DEATH_TOL,
                       "ref_yaw_scale": REF, "strata_horizontal": [[600, 1000], [1000, 2000],
                                                                   [2000, None]]},
              "fixture": "standardized: core reset + restore original candidate + one neutral "
                         "bootstrap tick; fresh held keys; not a search-node continuation",
              "ckpt_md5": md5(a.ckpt), "maps": {}}
    try:
        header["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                            capture_output=True, text=True).stdout.strip()
    except Exception:                       # pragma: no cover - no git
        header["git_head"] = None
    dll = os.environ.get("SURFCORE_DLL")
    header["core_dll_md5"] = md5(dll) if dll and Path(dll).exists() else None
    blob, table = {}, {}
    neutral1 = np.array([7, 3, 1, 1, 0, 0], np.int32)
    for pair in a.panel:
        mp, _, sp = pair.partition(".bsp:")
        mp += ".bsp"
        mname = Path(mp).stem
        st_raw = np.load(sp)
        st_all, first_ix = np.unique(st_raw, return_index=True)
        header["maps"][mname] = {"map_md5": md5(mp), "states": sp, "states_md5": md5(sp)}
        ctx = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--plan-scratch", "256",
                                 "--ep-ticks", "1000", "--map", str(mp), "--stochastic",
                                 "--exec-view-scale", f"{REF},1"])
        core = ctx.scratch.core
        S = core.num_envs
        tick_s = float(ctx.tick.ms) / 1000.0
        fb0 = ctx.finish_box
        ctx.planner = ea.RayOperator(float(ctx.tick.ms), 0.5 * (np.asarray(fb0[0], np.float64)
                                                             + np.asarray(fb0[1], np.float64)), n=3)
        K = int(ea.Flyer(ctx).K)                      # one controller interval, in ticks
        HL = int(round(LONG_SECS / tick_s))
        rng = np.random.default_rng(int(a.seed))
        order = rng.permutation(len(st_all))
        elig = {k: [] for k in range(len(STRATA))}   # stratum -> [(file ix, original, post)]
        for c0 in range(0, len(order), S):
            if all(len(v) >= int(a.n) for v in elig.values()):
                break
            idx = order[c0:c0 + S]
            core.reset(int(a.seed))                   # a clean hidden state for the batch
            origs = []
            for i in range(S):
                s2 = st_all[idx[min(i, len(idx) - 1)]].copy()
                s2["tick"] = 0
                s2["stuck_ticks"] = 0
                origs.append(s2)
                core.set_state(i, s2)
            _o, _r, done, trunc, _ = core.step(np.tile(neutral1, (S, 1)))
            cnt, _tn, _tp = core.get_touch()
            cur = core.get_states()
            for i in range(len(idx)):
                if done[i] or trunc[i] or cnt[i] > 0 or cur[i]["onground"] >= 0:
                    continue
                vh = float(np.hypot(cur[i]["velocity"][0], cur[i]["velocity"][1]))
                for k, (lo, hi) in enumerate(STRATA):
                    if lo <= vh < hi and len(elig[k]) < int(a.n):
                        elig[k].append((int(first_ix[idx[i]]), origs[i], cur[i].copy()))
        print(f"energy_panel v3: {mname}: eligible per horizontal-speed stratum "
              + ", ".join(f"[{lo:g}, {hi:g}) {len(elig[k])}" for k, (lo, hi) in enumerate(STRATA))
              + f"; decisive horizon {K} ticks ({K * tick_s * 1000:.0f} ms)", flush=True)
        for k, (lo, hi) in enumerate(STRATA):
            pool = elig[k]
            n = len(pool)
            if n == 0:
                continue
            cell = f"{mname}|{k}"
            blob[cell + "|src_ix"] = np.array([p[0] for p in pool], np.int64)
            blob[cell + "|orig"] = np.stack([p[1] for p in pool])
            blob[cell + "|post"] = np.stack([p[2] for p in pool])
            exp_hash = [st_hash(p[2]) for p in pool]
            res = {}
            for sc in branches:
                cb_ctx = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--plan-scratch",
                                            str(n), "--ep-ticks", str(HL + 400), "--map", str(mp),
                                            "--stochastic", "--exec-view-scale", f"{sc:g},1"])
                cb = cb_ctx.scratch.core
                fb = cb_ctx.finish_box
                fin = 0.5 * (np.asarray(fb[0], np.float64) + np.asarray(fb[1], np.float64))
                op = ea.RayOperator(float(cb_ctx.tick.ms), fin, n=3)
                cb_ctx.planner = op
                fl = ea.Flyer(cb_ctx)
                cb.reset(int(a.seed))
                for i in range(n):
                    cb.set_state(i, pool[i][1])
                obs, _r, d0, t0, _ = cb.step(np.tile(neutral1, (n, 1)))     # THE bootstrap tick
                obs = np.ascontiguousarray(np.asarray(obs, np.float32))
                c0_, _tn, _tp = cb.get_touch()
                cur = cb.get_states()
                match = np.array([(not d0[i]) and (not t0[i]) and c0_[i] == 0
                                  and st_hash(cur[i]) == exp_hash[i] for i in range(n)])
                lines = [op.line_and_curve_of(cur[i]["origin"].astype(np.float64),
                                              cur[i]["velocity"].astype(np.float64),
                                              float(cur[i]["yaw"]), op.choice_nums[0], 0)[0]
                         for i in range(n)]
                cb_ctx.scratch.line.set_lines(np.arange(n), [ln[:fl.L_MAX] for ln in lines])
                torch.manual_seed(int(a.seed))            # identical draws in every branch
                pol = cb_ctx.scratch.make_policy(cb, cb_ctx.scratch.line)
                if fl.keys_hold:
                    from surfgym.keyshold import KeysHold
                    pol.keys = KeysHold(n)
                    pol._keys_tick = (cur["tick"].astype(np.int64)
                                      - int(getattr(pol, "_period", fl.K)))
                pol._tick = 0
                v0 = cur["velocity"].astype(np.float64)
                e0 = 0.5 * np.sum(v0 ** 2, axis=1) + G * cur["origin"][:, 2].astype(np.float64)
                keh0 = 0.5 * (v0[:, 0] ** 2 + v0[:, 1] ** 2)
                E = np.zeros((HL, n))
                C = np.zeros((HL, n), np.int32)
                D = np.zeros((HL, n), bool)
                VW = np.full((HL, n, 3), np.nan)
                A = np.zeros((HL, n, 6), np.int16)
                dead = np.zeros(n, bool)
                for t in range(HL):
                    acts = pol.act(obs)
                    view = getattr(pol, "view", None)
                    obs, _r, done, trunc, _ = (cb.step(acts) if view is None
                                               else cb.step(acts, view=view))
                    obs = np.ascontiguousarray(np.asarray(obs, np.float32))
                    dead |= np.asarray(done, bool) | np.asarray(trunc, bool)
                    cnt, _tn, _tp = cb.get_touch()
                    cs = cb.states_view
                    E[t] = (0.5 * np.sum(cs["velocity"].astype(np.float64) ** 2, axis=1)
                            + G * cs["origin"][:, 2].astype(np.float64))
                    C[t] = np.asarray(cnt, np.int32)
                    D[t] = dead
                    A[t] = np.asarray(acts)[:n, :6]
                    if view is not None:
                        vv = np.asarray(view, np.float64).reshape(n, -1)
                        VW[t, :, :vv.shape[1]] = vv[:, :3]
                res[sc] = {"E": E, "C": C, "D": D, "VW": VW, "A": A, "e0": e0, "keh0": keh0,
                           "match": match}
                for key_, arr in res[sc].items():
                    blob[f"{cell}|{sc:g}|{key_}"] = arr
                blob[f"{cell}|{sc:g}|start_hash"] = np.array([st_hash(cur[i]) for i in range(n)])
            ref = res[REF]

            def stats(sc, H):
                b = res[sc]
                ok_start = ref["match"] & b["match"]
                r_ev = ((ref["C"][:H] > 0) | ref["D"][:H]).any(axis=0)
                cen = ok_start & r_ev
                pair = ok_start & ~r_ev
                death = pair & b["D"][:H].any(axis=0)
                contact = pair & ~death & (b["C"][:H] > 0).any(axis=0)
                clean = pair & ~death & ~contact
                hs = H * tick_s
                exc = ((b["E"][H - 1] - b["e0"]) - (ref["E"][H - 1] - ref["e0"])) / np.maximum(
                    ref["keh0"], 1.0) / hs
                x = exc[clean]
                npair = int(pair.sum())
                return {"eligible": n, "start_mismatch": int((~ok_start).sum()),
                        "censored": int(cen.sum()), "pairs": npair, "deaths": int(death.sum()),
                        "contacts": int(contact.sum()), "clean": int(clean.sum()),
                        "median": float(np.median(x)) if len(x) else float("nan"),
                        "lq": float(np.percentile(x, 25)) if len(x) else float("nan"),
                        "death_frac": float(death.sum()) / max(1, npair)}
            for sc in scales:
                s1 = stats(sc, K)
                sl = stats(sc, HL)
                populated = n >= POP
                decisive = bool(populated and s1["pairs"] >= MIN_PAIRS
                                and s1["censored"] <= MAX_CENS * max(1, n - s1["start_mismatch"]))
                passed = bool(decisive and s1["median"] >= MED_TOL and s1["lq"] >= LQ_TOL
                              and s1["death_frac"] <= DEATH_TOL)
                table[(mname, k, sc)] = dict(s1, populated=populated, decisive=decisive,
                                             passed=passed, closed_loop_0p3=sl)
                print(f"   [{lo:g}, {hi:g}) yaw x{sc:g}: one interval: {s1['pairs']} pairs "
                      f"({s1['censored']} censored, {s1['start_mismatch']} start mismatches), "
                      f"deaths {s1['deaths']}, contacts {s1['contacts']}, excess median "
                      f"{100 * s1['median']:+.2f}% KE_h0/s LQ {100 * s1['lq']:+.2f}% -> "
                      f"{'PASS' if passed else ('fail' if decisive else ('NOT DECISIVE' if populated else 'unpopulated'))}"
                      f" | 0.3 s closed loop: median {100 * sl['median']:+.2f}% LQ "
                      f"{100 * sl['lq']:+.2f}% over {sl['clean']} clean", flush=True)
    # EVERY panel map, including one with no eligible state at all (it has no cell in the
    # table): a map without a decisive cell makes the panel INCONCLUSIVE (v3's first run checked
    # only the maps present in the table and so passed the two edgeflow maps vacuously)
    maps_ = sorted(header["maps"])
    cells = sorted({(m, k) for (m, k, _s) in table})
    s0 = scales[0]
    bad_cells = [(m, k) for (m, k) in cells
                 if table[(m, k, s0)]["populated"] and not table[(m, k, s0)]["decisive"]]
    maps_ok = all(any(table[(m, k, s0)]["decisive"] for (mm, k) in cells if mm == m) for m in maps_)
    conclusive = (not bad_cells) and maps_ok
    chosen = None
    if conclusive:
        for sc in sorted(scales, reverse=True):
            if all(table[(m, k, sc)]["passed"] for (m, k) in cells if table[(m, k, sc)]["decisive"]):
                chosen = sc
                break
    verdict = ("CONCLUSIVE" if conclusive else "INCONCLUSIVE") + (
        f": yaw x{chosen:g}" if chosen is not None else ": nothing chosen")
    print(f"energy_panel v3: {verdict}; populated-but-not-decisive cells {bad_cells or 'none'}; "
          f"every map has a decisive cell: {maps_ok}")
    if a.out:
        header["table"] = {f"{m}|{k}|{sc:g}": v for (m, k, sc), v in table.items()}
        header["verdict"] = verdict
        header["chosen_yaw_scale"] = chosen
        np.savez_compressed(a.out, header=np.array(json.dumps(header)), **blob)
        print(f"   traces -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
