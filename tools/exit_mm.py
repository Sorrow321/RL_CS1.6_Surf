"""exit_mm.py - the expert loop (tools/exit_local.py: search -> distil -> train -> gate) for ONE
shared policy over a --maps run (the user, 2026-11-21, after all seven maps finished: "Try to
implement the expert loop for multi map scenario. I see that timers are improving, but quite
slowly. It feels like expert loop does better job at improving timers").

Round r (OUT/round_<r>/):

  1. PLAN    every map: tools/beam_tas.py waves from the incumbent (the policy's own proposals,
             a greedy floor, the K fastest finishers kept), --plan-par maps at a time; a wave is
             distilled only if it crossed and its fastest line is not slower than its own greedy
             + --distil-max-over-greedy (exit_local.keep_wave) - or, when that greedy did not
             finish, than the incumbent's own PACE on the map (the median finish of its latest
             test): a map the policy finishes 7 times in 9 must not learn a line slower than the
             runs it already flies (mmx_E1 round 0: a 138 s sidistic line against ~113 s);
  2. DISTIL  every map with a kept wave: tools/plan_to_bc.py -> <map>/bc.npz + <map>/spine.npy
             (+ spine.keys.npy under --keys-hold);
  3. TRAIN   ONE warm resume of the incumbent over all its maps with the run's own recipe tokens
             (--recipe-file: the launch's token list) plus: --bc-file tag=bc.npz,... (each map's
             planner rows, rendered by that map, an equal share of every BC batch) and
             --resume-spines tag=spine.npy,... (those maps' curricula restart on the planner's
             line with a FIXED full window: spawns uniform along the whole line, 10% at the map
             start, as exit_local pins it); maps without a kept wave keep their held spines;
  4. EVAL    every map: tools/record_ckpt.py greedy episodes of the challenger AND the incumbent
             on the same spawns (--gate-evals evals of --eval-eps, --eval-par maps at a time);
  5. GATE    per map the EXPECTED time over the shared spawns, a DNF charged the map's cap (the
             seed's slowest finish x --cap-mult, fixed at round 0); the challenger becomes the
             incumbent only if the mean over maps of (challenger / incumbent) expected time is
             below 1 AND no map finishes more than --gate-finish-tol of its episodes fewer.

Round 0 first evaluates the seed on every map (the caps). One JSON line per round in
OUT/summary.jsonl; the incumbent is OUT/best_mean.pt; OUT/STOP ends the loop after the round.

Every line the trainer sees is the POLICY'S OWN (the planner proposes from the checkpoint's own
distribution): the BC files and spines are policy-derived (CLAUDE.md section 0; SELF_STATES=1).
The same flags and constants on every map (section 0b): the per-map numbers (caps, search
horizons) are the policy's own times, never read off a map.

    python tools/exit_mm.py --ckpt runs_ckpt.pt --recipe-file runs/x/recipe.json --out runs/mmx \
        --rounds 12 --deadline-h 10
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from exit_local import (ckpt_step, episode_times, eval_tick_ms, finish_times,  # noqa: E402
                        gate_seed, keep_wave, log)

PY = sys.executable


def set_flag(tokens, flag, *values):
    """``tokens`` with every ``flag`` and its values (the tokens up to the next one that starts
    with "--") removed, then ``flag`` + ``values`` appended (argparse's last-wins would do for a
    plain value, but not for a flag whose old value another flag refuses). ``values`` (None,) =
    removed only; () = a bare switch."""
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] == flag:
            i += 1
            while i < len(tokens) and not str(tokens[i]).startswith("--"):
                i += 1
            continue
        out.append(tokens[i])
        i += 1
    if not (values and values[0] is None):
        out += [flag, *[str(v) for v in values]]
    return out


def ckpt_maps(path: Path):
    """(map stems, tags) of a --maps checkpoint (its config's maps; the trainer's tag rule)"""
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from surfgym.mapfleet import map_tags
    cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
    maps = list(cfg.get("maps") or [])
    if not maps:
        raise SystemExit(f"{path}: not a --maps checkpoint (tools/exit_local.py is the one-map loop)")
    return maps, map_tags(maps)


def run_many(jobs, par: int, timeout: float):
    """jobs = [(key, cmd, log_path, env_extra)], at most ``par`` at once -> {key: returncode}"""
    rcs, live, queue = {}, [], list(jobs)
    t_end = time.time() + timeout
    while queue or live:
        while queue and len(live) < max(1, int(par)):
            key, cmd, lp, envx = queue.pop(0)
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            env.update(envx or {})
            f = open(lp, "w", encoding="utf-8", errors="replace")
            p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), env=env)
            live.append((key, p, f))
        time.sleep(2.0)
        for item in list(live):
            key, p, f = item
            rc = p.poll()
            if rc is None and time.time() > t_end:
                p.kill()
                rc = p.wait()
            if rc is not None:
                f.close()
                rcs[key] = rc
                live.remove(item)
    return rcs


def threads(par: int) -> dict:
    """the env of a job that runs ``par`` at once: the core's batch step is OpenMP-parallel
    over every core (src/env.c), so each of ``par`` jobs gets its share, not all of them"""
    n = str(max(1, (os.cpu_count() or 2) // max(1, int(par))))
    return {"OMP_NUM_THREADS": n, "NUMBA_NUM_THREADS": n}


def eval_jobs(ck: Path, maps, rdir: Path, stem: str, eps: int, caps_ticks, seeds, par: int):
    """record_ckpt jobs: every map x every gate seed -> files <map>/<stem>_<k>.jsonl"""
    jobs = []
    for m in maps:
        md = rdir / Path(m).stem
        md.mkdir(parents=True, exist_ok=True)
        for k, sd in enumerate(seeds):
            out = md / f"{stem}_{k}.jsonl"
            jobs.append(((m, k), [PY, "-u", ROOT / "tools" / "record_ckpt.py", ck,
                                  "--map", ROOT / "maps" / f"{Path(m).stem}.bsp",
                                  "--episodes", eps, "--ep-ticks", caps_ticks[m],
                                  "--seed", sd, "--out", out],
                         md / f"{stem}_{k}.log", threads(par)))
    return jobs


def map_times(rdir: Path, m: str, stem: str, n_evals: int):
    """every episode of a map's gate evals in file order: finish seconds or None"""
    md = rdir / Path(m).stem
    out = []
    for k in range(n_evals):
        f = md / f"{stem}_{k}.jsonl"
        out += episode_times(f) if f.exists() else []
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if not n else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def expected(times, dnf_s: float):
    return (sum(dnf_s if t is None else t for t in times) / len(times)) if times else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="the seed: a --maps checkpoint (policy-derived)")
    ap.add_argument("--recipe-file", required=True,
                    help="JSON list: the run's launch tokens (box.py's 'flags' of its launch: the "
                         "recipe with its --envs and per-map --maxvel), resumed every round")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--deadline-h", type=float, default=10.0)
    ap.add_argument("--round-h", type=float, default=1.5,
                    help="a round is not started when it would end past the deadline")
    # PLAN
    ap.add_argument("--waves", type=int, default=2, help="search waves per map per round")
    ap.add_argument("--plan-envs", type=int, default=2048)
    ap.add_argument("--greedy-envs", type=int, default=128)
    ap.add_argument("--resample", type=int, default=25)
    ap.add_argument("--score", default="dh")
    ap.add_argument("--keep", type=int, default=16)
    ap.add_argument("--plan-args", default="--allow-nonfinisher --score-energy 30",
                    help="appended to every beam_tas call (one string)")
    ap.add_argument("--plan-par", type=int, default=2, help="maps planned at once")
    ap.add_argument("--plan-mult", type=float, default=1.5,
                    help="a map's search horizon: its incumbent's slowest finish x this")
    ap.add_argument("--distil-max-over-greedy", type=float, default=0.0)
    ap.add_argument("--distil-args", default="--no-value-target")
    # TRAIN
    ap.add_argument("--train-steps", type=float, default=1e9, help="FLEET steps per round")
    ap.add_argument("--bc-coef", type=float, default=0.5)
    ap.add_argument("--bc-coef-final", type=float, default=0.5)
    ap.add_argument("--bc-target", default="dist")
    ap.add_argument("--bc-value-coef", type=float, default=0.0)
    ap.add_argument("--train-args", default="",
                    help="more trainer tokens for every round (one string; last wins)")
    # EVAL / GATE
    ap.add_argument("--eval-eps", type=int, default=9)
    ap.add_argument("--gate-evals", type=int, default=2)
    ap.add_argument("--eval-par", type=int, default=4, help="recordings at once")
    ap.add_argument("--cap-mult", type=float, default=1.5,
                    help="a map's eval cap and DNF charge: the seed's slowest finish x this")
    ap.add_argument("--gate-finish-tol", type=float, default=0.1,
                    help="a map may finish at most this share of its gate episodes fewer")
    args = ap.parse_args()

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "exit_mm.log", "a", encoding="utf-8")
    t_start = time.time()
    deadline = t_start + args.deadline_h * 3600.0
    recipe = [str(t) for t in json.loads(Path(args.recipe_file).read_text(encoding="utf-8"))]
    seed = Path(args.ckpt).resolve()
    maps, tags = ckpt_maps(seed)
    tag_of = dict(zip(maps, tags))
    state_p = out / "state.json"
    if state_p.exists():
        # a restart: the incumbent and the caps of the run so far
        st = json.loads(state_p.read_text(encoding="utf-8"))
        inc, caps = Path(st["incumbent"]), {m: float(st["caps"][m]) for m in maps}
        tick_ms = float(st["tick_ms"])
        pace = {m: st.get("pace", {}).get(m) for m in maps}
        log(fh, f"restart: incumbent {inc}, caps {caps}")
    else:
        inc, caps, tick_ms, pace = seed, None, None, {}
    log(fh, f"seed {seed} (step {ckpt_step(seed):,}); maps {', '.join(tags)}; deadline in "
            f"{args.deadline_h:g} h")
    plan_extra, distil_extra = args.plan_args.split(), args.distil_args.split()

    if caps is None:
        # round 0's caps: the seed's own times on every map (finished or not)
        r0 = out / "seed_eval"
        r0.mkdir(exist_ok=True)
        big = {m: 40000 for m in maps}
        rcs = run_many(eval_jobs(seed, maps, r0, "seed", args.eval_eps, big, [gate_seed(0, 0)],
                                 args.eval_par), args.eval_par, 3 * 3600)
        caps = {}
        for m in maps:
            f = r0 / Path(m).stem / "seed_0.jsonl"
            fins = finish_times(f)[0] if f.exists() else []
            tick_ms = tick_ms or (eval_tick_ms(f) if f.exists() else None)
            if not fins:
                raise SystemExit(f"the seed finishes no episode on {m} (rc {rcs.get((m, 0))}): "
                                 "the loop improves times of maps the policy already finishes")
            caps[m] = round(max(fins) * float(args.cap_mult), 2)
            pace[m] = round(median(fins), 3)
            log(fh, f"seed {tag_of[m]}: {len(fins)} finish(es), best {min(fins):.2f} s, slowest "
                    f"{max(fins):.2f} s -> cap {caps[m]:.2f} s")
        state_p.write_text(json.dumps({"incumbent": str(inc), "caps": caps,
                                       "tick_ms": tick_ms, "pace": pace}, indent=1),
                           encoding="utf-8")

    for r in range(args.start_round, args.start_round + args.rounds):
        if (out / "STOP").exists():
            log(fh, f"round {r}: STOP file - stopping")
            break
        if time.time() + args.round_h * 3600.0 > deadline:
            log(fh, f"round {r}: would cross the deadline - stopping")
            break
        rdir = out / f"round_{r}"
        rdir.mkdir(exist_ok=True)
        row = {"round": r, "ckpt_in": str(inc), "t0": time.strftime("%H:%M:%S")}
        caps_ticks = {m: int(round(caps[m] * 1000.0 / tick_ms)) + 1 for m in maps}
        plan_ticks = {m: int(round(caps[m] / args.cap_mult * args.plan_mult * 1000.0 / tick_ms))
                      for m in maps}

        # 1. PLAN -------------------------------------------------------------------------------
        t0 = time.time()
        jobs = []
        for m in maps:
            md = rdir / Path(m).stem
            md.mkdir(parents=True, exist_ok=True)
            for w in range(args.waves):
                jobs.append(((m, w), [PY, "-u", ROOT / "tools" / "beam_tas.py", inc,
                                      "--map", ROOT / "maps" / f"{Path(m).stem}.bsp",
                                      "--envs", args.plan_envs, "--greedy-envs", args.greedy_envs,
                                      "--score", args.score, "--resample-every", args.resample,
                                      "--greedy-eps", 1, "--max-ticks", plan_ticks[m],
                                      "--keep-finishers", args.keep, "--route-file", "none",
                                      "--torch-seed", 1000 * r + w, "--seed", 1 + 100 * r + w,
                                      "--out-dir", md / f"wave_{w}", *plan_extra],
                             md / f"wave_{w}.log", threads(args.plan_par)))
        rcs = run_many(jobs, args.plan_par, 3 * 3600)
        plan, kept = {}, {}
        for m in maps:
            md, ws = rdir / Path(m).stem, []
            for w in range(args.waves):
                s = {}
                sp = md / f"wave_{w}" / "summary.json"
                if sp.exists():
                    s = json.loads(sp.read_text(encoding="utf-8"))
                ok = (rcs.get((m, w)) == 0 and s.get("crossed")
                      and (md / f"wave_{w}" / "beam_best.npz").exists())
                # the bar: the wave's own greedy, or the incumbent's pace when that greedy failed
                bar = s.get("greedy_s") if s.get("greedy_s") is not None else pace.get(m)
                keep = bool(ok and keep_wave(s.get("best_s"), bar, args.distil_max_over_greedy))
                ws.append({"wave": w, "rc": rcs.get((m, w)), "crossed": bool(s.get("crossed")),
                           "best_s": s.get("best_s"), "greedy_s": s.get("greedy_s"),
                           "bar_s": bar, "distilled": keep})
                if keep:
                    kept.setdefault(m, []).append(md / f"wave_{w}" / "beam_best.npz")
            plan[tag_of[m]] = ws
            best = min([x["best_s"] for x in ws if x.get("best_s")], default=None)
            log(fh, f"round {r} plan {tag_of[m]}: best {best} s; waves "
                    + ", ".join(f"{x['best_s']} vs greedy {x['greedy_s']}"
                                + ("" if x["greedy_s"] is not None else f" (pace {x['bar_s']})")
                                + ("" if x["distilled"] else " (not kept)") for x in ws))
        row["plan"] = {"maps": plan, "wall_s": round(time.time() - t0, 1)}
        if not kept:
            log(fh, f"round {r}: no map has a kept wave - the next round plans again")
            row["stopped"] = "nothing kept"
            with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            continue

        # 2. DISTIL -----------------------------------------------------------------------------
        import numpy as np
        sys.path.insert(0, str(ROOT / "python"))
        from surfgym.bc import merge_bc_files
        jobs = []
        for m, npzs in kept.items():
            md = rdir / Path(m).stem
            for i, z in enumerate(npzs):
                jobs.append(((m, i), [PY, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan", z,
                                      "--ckpt", inc, "--out", md / f"bc_{i}.npz",
                                      "--spine", md / f"spine_{i}.npy",
                                      "--map", ROOT / "maps" / f"{Path(m).stem}.bsp",
                                      *distil_extra],
                             md / f"distil_{i}.log", threads(args.plan_par)))
        rcs = run_many(jobs, args.plan_par, 3600)
        bcs = {}
        for m, npzs in kept.items():
            md = rdir / Path(m).stem
            idx = [i for i in range(len(npzs)) if rcs.get((m, i)) == 0
                   and (md / f"bc_{i}.npz").exists() and (md / f"spine_{i}.npy").exists()]
            if len(idx) < len(npzs):
                log(fh, f"round {r}: distil failed on {len(npzs) - len(idx)} of {tag_of[m]}'s "
                        f"waves - see {md}/distil_*.log")
            if not idx:
                continue
            merge_bc_files([md / f"bc_{i}.npz" for i in idx], md / "bc.npz")
            np.save(md / "spine.npy", np.concatenate([np.load(md / f"spine_{i}.npy")
                                                      for i in idx]))
            kf = [md / f"spine_{i}.keys.npy" for i in idx]
            if all(k.exists() for k in kf):
                # the spines' held keys, concatenated in the same order
                np.save(md / "spine.keys.npy", np.concatenate([np.load(k) for k in kf]))
            bcs[m] = md
        if not bcs:
            row["stopped"] = "distil failed"
            with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            continue
        row["distilled"] = [tag_of[m] for m in bcs]

        # 3. TRAIN ------------------------------------------------------------------------------
        t0 = time.time()
        run_name = f"{out.name}_r{r}"
        shutil.rmtree(ROOT / "runs" / run_name, ignore_errors=True)
        win = max(int(np.load(d / "spine.npy", mmap_mode="r").shape[0]) for d in bcs.values())
        tok = list(recipe)
        for flag, vals in (("--demo-window", (win,)), ("--demo-rate", (2.0,)),
                           ("--demo-min-ep", ("1e9",)), ("--demo-grow", (0,)),
                           ("--demo-grow-max", (None,)), ("--demo-front-frac", (0,)),
                           ("--demo-success-bonus", (None,)),
                           ("--demo-success-bonus-energy-only", (None,)),
                           ("--record-every", ("1e12",)), ("--ckpt-every", ("1e12",))):
            tok = set_flag(tok, flag, *vals)
        steps = ckpt_step(inc) + int(float(args.train_steps))
        cmd = [PY, "-u", ROOT / "python" / "train_fast.py", "--ckpt", inc, "--run", run_name,
               "--steps", steps, "--no-eval-at-start", "--eval-greedy-only", *tok,
               "--bc-file", ",".join(f"{tag_of[m]}={d / 'bc.npz'}" for m, d in bcs.items()),
               "--bc-coef", args.bc_coef, "--bc-coef-final", args.bc_coef_final,
               "--bc-steps", args.train_steps, "--bc-target", args.bc_target,
               "--bc-value-coef", args.bc_value_coef,
               "--resume-spines", ",".join(f"{tag_of[m]}={d / 'spine.npy'}"
                                           for m, d in bcs.items()),
               *args.train_args.split()]
        rcs = run_many([("train", cmd, rdir / "train.log", {"SELF_STATES": "1"})], 1, 6 * 3600)
        final = ROOT / "runs" / run_name / "ckpt_final.pt"
        row["train"] = {"run": run_name, "rc": rcs.get("train"),
                        "wall_s": round(time.time() - t0, 1)}
        bl = ROOT / "runs" / run_name / "bc_log.csv"
        if bl.exists():
            lines = bl.read_text(encoding="utf-8").strip().splitlines()
            if len(lines) > 2:
                row["train"]["bc_first"], row["train"]["bc_last"] = lines[1], lines[-1]
        if rcs.get("train") != 0 or not final.exists():
            log(fh, f"round {r}: train failed (rc {rcs.get('train')}) - see {rdir / 'train.log'}")
            row["stopped"] = "train failed"
            with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            break
        log(fh, f"round {r}: trained {args.train_steps:g} fleet steps in "
                f"{row['train']['wall_s']:.0f} s on {len(bcs)} map(s)' lines")

        # 4. EVAL: the challenger and the incumbent on the same spawns ----------------------------
        seeds = [gate_seed(r, k) for k in range(args.gate_evals)]
        run_many(eval_jobs(final, maps, rdir, "eval", args.eval_eps, caps_ticks, seeds,
                           args.eval_par)
                 + eval_jobs(inc, maps, rdir, "eval_incumbent", args.eval_eps, caps_ticks, seeds,
                             args.eval_par), args.eval_par, 3 * 3600)

        # 5. GATE -------------------------------------------------------------------------------
        per, ratios, worse = {}, [], []
        for m in maps:
            ci = map_times(rdir, m, "eval", args.gate_evals)
            ii = map_times(rdir, m, "eval_incumbent", args.gate_evals)
            n = min(len(ci), len(ii))
            ec, ei = expected(ci[:n], caps[m]), expected(ii[:n], caps[m])
            fc = sum(t is not None for t in ci[:n])
            fi = sum(t is not None for t in ii[:n])
            tol = max(1, int(round(args.gate_finish_tol * n)))
            fin_c = [t for t in ci[:n] if t is not None]
            per[tag_of[m]] = {"episodes": n, "finishes": [fc, fi],
                              "expected": [None if ec is None else round(ec, 3),
                                           None if ei is None else round(ei, 3)],
                              "best": round(min(fin_c), 3) if fin_c else None}
            if ec is None or ei is None or n == 0:
                worse.append(f"{tag_of[m]}: no eval")
                continue
            ratios.append(ec / ei)
            if fc < fi - tol:
                worse.append(f"{tag_of[m]}: {fc}/{n} finishes vs {fi}/{n}")
            log(fh, f"round {r} gate {tag_of[m]}: {fc}/{n} vs {fi}/{n} finishes, expected "
                    f"{ec:.2f} vs {ei:.2f} s (cap {caps[m]:.2f}), best "
                    f"{per[tag_of[m]]['best']}")
        mean_ratio = (sum(ratios) / len(ratios)) if ratios else None
        accept = mean_ratio is not None and mean_ratio < 1.0 and not worse
        for m in maps:
            tm = [t for t in map_times(rdir, m, "eval" if accept else "eval_incumbent",
                                       args.gate_evals) if t is not None]
            if tm:
                pace[m] = round(median(tm), 3)
        row["gate"] = {"maps": per, "mean_ratio": None if mean_ratio is None
                       else round(mean_ratio, 4), "worse": worse,
                       "verdict": "accepted" if accept else "rejected"}
        if accept:
            shutil.copyfile(final, out / "best_mean.pt")
            inc = out / "best_mean.pt"
            state_p.write_text(json.dumps({"incumbent": str(inc), "caps": caps,
                                           "tick_ms": tick_ms, "pace": pace}, indent=1),
                               encoding="utf-8")
            log(fh, f"round {r}: gate ACCEPTED (mean expected ratio {mean_ratio:.4f}) -> {inc}")
        else:
            log(fh, f"round {r}: gate rejected (mean expected ratio "
                    f"{'n/a' if mean_ratio is None else f'{mean_ratio:.4f}'}"
                    + (f"; {'; '.join(worse)}" if worse else "") + f") - the next round plans "
                    f"from {inc}")
        row["pace"] = dict(pace)
        if not accept:
            state_p.write_text(json.dumps({"incumbent": str(inc), "caps": caps,
                                           "tick_ms": tick_ms, "pace": pace}, indent=1),
                               encoding="utf-8")
        with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    log(fh, f"done: incumbent {inc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
