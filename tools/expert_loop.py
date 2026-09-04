#!/usr/bin/env python3
"""expert_loop.py - AlphaZero-style expert iteration for the surf racer.

Each round: the current policy PROPOSES (greedy evals from the map start),
the beam planner IMPROVES the line by search in the real simulator
(tools/beam_tas.py), and the policy is trained to IMITATE the planner's
(state, action) pairs on top of its ordinary PPO objective, spawning along
the planner's line. The improvement compounds instead of being thrown away.

    round r (runs/<name>/round_<r>/):
      1. EVAL     tools/record_ckpt.py: E greedy map-start episodes of
                  policy_r -> finish times (spawn basis; the ledger's number)
      2. PLAN     tools/beam_tas.py waves (torch seeds 0,1,...) until the
                  planning budget is spent; the fastest crossing wins ->
                  plan/best/beam_best.npz (the winner + the K fastest
                  distinct finishing lineages, replay-verified)
      3. DISTIL   tools/plan_to_bc.py: replay the kept lines -> bc.npz
                  (one row per decision: state, core scalars, latch, action)
                  + spine.npy (the best line's per-tick states)
      4. TRAIN    tools/run_arm.sh ARM_RESUME=1: WARM resume of policy_r for
                  --train-steps with --bc-file bc.npz (the BC term inside
                  every PPO minibatch step, --bc-coef decaying linearly to
                  --bc-coef-final) and --demo-file spine.npy (the reservoir
                  share of every spawn pool drawn uniformly from the
                  planner's line, the xLOOP/xDEMO-validated demo flags)
                  -> train/ckpt_final.pt = policy_{r+1}
      5. EVAL     policy_{r+1} (reused as round r+1's step 1)

Every subprocess is logged under the round directory; every path handed to
a subprocess is absolute (the map ALWAYS from the main checkout: a worktree
copy re-bakes every cache). The machine-readable result is
runs/<name>/expert_summary.jsonl, one line per round: the seed's greedy
time, the planner's time, the distilled policy's greedy time, and the
wall clock of every phase. A completed round (round.json with done=true) is
skipped on restart, so a crashed overnight loop resumes where it stopped.

A seed with a --quantiles critic (runs/research/xQR32) is collapsed to the
scalar head first (tools/ckpt_qr_to_scalar.py: exact for the value, actor
untouched) so the mainline trainer can resume it.

    python tools/expert_loop.py C:/RL_Surf/runs/research/xQR32/xQR32_final.pt \
        --name exit --rounds 10 --train-steps 3e8 --plan-budget 600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
PY = sys.executable
DEF_MAP = "C:/RL_Surf/maps/surf_src_cannonball.bsp"
FINISH_PAD = 64.0          # eval_honesty's finish tolerance (loop_spine.py)


def log(fh, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_bash():
    for p in (r"C:\Program Files\Git\bin\bash.exe",
              r"C:\Program Files\Git\usr\bin\bash.exe"):
        if os.path.exists(p):
            return p
    return "bash"


def make_shim(root: Path) -> str:
    """run_arm.sh invokes `python3`, which Windows python installs do not
    provide (tools/loop_driver.py's shim, verbatim)."""
    d = root / "bin"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "python3"
    with open(p, "w", encoding="ascii", newline="\n") as fh:
        fh.write('#!/bin/bash\nexec "%s" "$@"\n' % PY.replace("\\", "/"))
    os.chmod(p, 0o755)
    return str(d)


def run(cmd, log_path, timeout, cwd=None, env=None):
    """Run a subprocess with stdout+stderr to log_path. -> returncode."""
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    if env:
        e.update(env)
    with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
        fh.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        fh.flush()
        try:
            r = subprocess.run([str(c) for c in cmd], cwd=cwd or str(ROOT),
                               env=e, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            fh.write(f"\n!! TIMEOUT after {timeout}s\n")
            return -9
    return r.returncode


def find_trainer(run_name):
    me = os.getpid()
    out = []
    for p in psutil.process_iter(["cmdline"]):
        try:
            cl = p.info["cmdline"] or []
            if (p.pid != me
                    and any("train_fast.py" in str(c) for c in cl)
                    and any(str(c) == run_name for c in cl)):
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def kill_all(procs):
    tokill = []
    for p in procs:
        try:
            tokill += p.children(recursive=True)
        except psutil.Error:
            pass
        tokill.append(p)
    for p in tokill:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(tokill, timeout=20)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------
def prepare_seed(seed_ckpt: Path, root: Path, fh) -> Path:
    """A quantile-critic seed is collapsed to a scalar head first."""
    import torch
    ck = torch.load(seed_ckpt, map_location="cpu", weights_only=False)
    nq = int(ck["policy"]["value_head.weight"].shape[0])
    cfg = ck.get("config") or {}
    log(fh, f"seed {seed_ckpt} md5 {md5(seed_ckpt)} step "
            f"{int(ck.get('global_step', 0)):,} value head rows {nq} "
            f"act_every {cfg.get('act_every')} race_latch {cfg.get('race_latch')}"
            f" obs_reward {cfg.get('obs_reward')}")
    del ck
    if nq == 1:
        return seed_ckpt
    dst = root / "seed_scalar.pt"
    if not dst.exists():
        rc = run([PY, "-u", ROOT / "tools" / "ckpt_qr_to_scalar.py",
                  seed_ckpt, dst], root / "seed_convert.log", 600)
        if rc != 0 or not dst.exists():
            raise SystemExit("seed conversion failed (see seed_convert.log)")
    log(fh, f"seed: {nq}-quantile critic collapsed -> {dst} md5 {md5(dst)}")
    return dst


def eval_policy(ckpt: Path, rdir: Path, map_path: str, episodes: int,
                seed: int, fh, tag: str) -> dict:
    """E greedy map-start episodes -> finish times (spawn basis)."""
    traj = rdir / f"{tag}.jsonl"
    states = rdir / f"{tag}_states.npz"
    t0 = time.time()
    rc = run([PY, "-u", ROOT / "tools" / "record_ckpt.py", ckpt,
              "--map", map_path, "--episodes", episodes, "--seed", seed,
              "--out", traj, "--dump-states", states],
             rdir / f"{tag}.log", 3600)
    if rc != 0 or not states.exists():
        raise SystemExit(f"eval of {ckpt} failed (rc={rc}, see {tag}.log)")
    from surfgym.zones import load_zones
    zones = load_zones(map_path)
    lo = np.asarray(zones["end"]["mins"], np.float64) - FINISH_PAD
    hi = np.asarray(zones["end"]["maxs"], np.float64) + FINISH_PAD
    z = np.load(states, allow_pickle=False)
    times, ends = [], []
    for k in sorted(z.files):
        e = z[k]
        o = np.asarray(e["origin"], np.float64)
        fin = bool(np.any(np.all((o >= lo) & (o <= hi), axis=1)))
        ends.append(int(len(e)))
        if fin:
            times.append(len(e) / 100.0)
    res = {"ckpt": str(ckpt), "episodes": int(len(ends)),
           "finishes": int(len(times)),
           "best_s": (min(times) if times else None),
           "mean_s": (float(np.mean(times)) if times else None),
           "times_s": times, "episode_ticks": ends,
           "wall_s": round(time.time() - t0, 1)}
    (rdir / f"{tag}.json").write_text(json.dumps(res, indent=2),
                                      encoding="utf-8")
    log(fh, f"{tag}: {res['finishes']}/{res['episodes']} finished, best "
            f"{res['best_s']} s, mean {res['mean_s']} s ({res['wall_s']}s)")
    return res


def plan(ckpt: Path, rdir: Path, map_path: str, args, fh) -> dict:
    """beam_tas waves until the budget is spent; the fastest crossing wins."""
    pdir = rdir / "plan"
    pdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    waves = []
    w = 0
    while True:
        wdir = pdir / f"wave_{w}"
        cmd = [PY, "-u", ROOT / "tools" / "beam_tas.py", ckpt,
               "--map", map_path, "--envs", args.plan_envs,
               "--resample-every", args.plan_resample,
               "--elite-frac", args.plan_elite,
               "--greedy-envs", args.plan_greedy_envs,
               "--score", args.plan_score, "--v-switch", args.plan_v_switch,
               "--torch-seed", w, "--seed", args.plan_seed,
               "--greedy-eps", 3, "--keep-finishers", args.keep_finishers,
               "--allow-nonfinisher", "--log-every", 25,
               "--max-ticks", args.plan_max_ticks,
               "--out-dir", wdir]
        tw = time.time()
        rc = run(cmd, pdir / f"wave_{w}.log", max(600, args.plan_budget * 3))
        row = {"wave": w, "rc": rc, "wall_s": round(time.time() - tw, 1)}
        sfile = wdir / "summary.json"
        if sfile.exists():
            s = json.loads(sfile.read_text(encoding="utf-8"))
            row.update(crossed=bool(s.get("crossed")),
                       best_ticks=s.get("best_ticks"), best_s=s.get("best_s"),
                       greedy_ticks=s.get("greedy_ticks"),
                       finishes=s.get("finishes"),
                       search_wall_s=s.get("search_wall_s"),
                       env_steps_per_s=s.get("env_steps_per_s"))
        else:
            row["crossed"] = False
        waves.append(row)
        log(fh, f"plan wave {w}: rc={rc} crossed={row.get('crossed')} "
                f"best={row.get('best_s')} s greedy="
                f"{(row.get('greedy_ticks') or 0) / 100:.2f} s "
                f"finishes={row.get('finishes')} ({row['wall_s']}s)")
        w += 1
        spent = time.time() - t0
        if w >= args.plan_max_waves:
            break
        if w >= args.plan_min_waves and spent + row["wall_s"] > args.plan_budget:
            break
    ok = [x for x in waves if x.get("crossed") and x.get("best_ticks")]
    res = {"waves": waves, "wall_s": round(time.time() - t0, 1)}
    if not ok:
        res["best"] = None
        (pdir / "plan.json").write_text(json.dumps(res, indent=2),
                                        encoding="utf-8")
        return res
    best = min(ok, key=lambda x: (x["best_ticks"], x["wave"]))
    res["best"] = best
    res["best_npz"] = str(pdir / f"wave_{best['wave']}" / "beam_best.npz")
    # every crossing wave's kept finishers pool into the BC rows (same
    # spawn, different --torch-seed): fastest wave first
    res["npz_all"] = [str(pdir / f"wave_{x['wave']}" / "beam_best.npz")
                      for x in sorted(ok, key=lambda x: (x["best_ticks"],
                                                         x["wave"]))]
    res["greedy_gate_ticks"] = best.get("greedy_ticks")
    (pdir / "plan.json").write_text(json.dumps(res, indent=2),
                                    encoding="utf-8")
    log(fh, f"plan: best wave {best['wave']} {best['best_s']} s "
            f"(gate greedy {(best.get('greedy_ticks') or 0) / 100:.2f} s), "
            f"{len(ok)}/{len(waves)} waves crossed, {res['wall_s']}s")
    return res


def distil(plan_npzs: list, ckpt: Path, rdir: Path, map_path: str, args, fh):
    t0 = time.time()
    bc = rdir / "bc.npz"
    spine = rdir / "spine.npy"
    summ = rdir / "bc_summary.json"
    rc = run([PY, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan",
              *plan_npzs,
              "--ckpt", ckpt, "--out", bc, "--spine", spine, "--map", map_path,
              "--lines", args.bc_lines,
              "--line-weight-decay", args.line_weight_decay,
              "--summary-out", summ], rdir / "distil.log", 1800)
    if rc != 0 or not summ.exists():
        raise SystemExit(f"plan_to_bc failed (rc={rc}, see distil.log)")
    meta = json.loads(summ.read_text(encoding="utf-8"))
    meta["wall_s"] = round(time.time() - t0, 1)
    log(fh, f"distil: {meta['rows']:,} rows from {meta['lines']} line(s), "
            f"spine {meta.get('spine_len')} states ({meta['wall_s']}s)")
    return bc, spine, meta


def train(ckpt: Path, rdir: Path, run_name: str, bc: Path, spine: Path,
          spine_len: int, map_path: str, args, shim: str, fh) -> tuple:
    """Warm resume through run_arm.sh (ARM_RESUME=1). -> (ckpt_final, info)"""
    t0 = time.time()
    tdir = ROOT / "runs" / run_name
    flags = ["--map", map_path,
             "--seed", args.train_seed_base + int(rdir.name.split("_")[-1]),
             "--no-eval-at-start", "--record-every", "1e12",
             "--ckpt-every", "1e12",
             "--bc-file", bc, "--bc-coef", args.bc_coef,
             "--bc-coef-final", args.bc_coef_final,
             "--bc-steps", args.train_steps, "--bc-batch", args.bc_batch]
    if not args.no_demo:
        flags += ["--demo-file", spine, "--demo-window", spine_len,
                  "--demo-rate", "2.0", "--demo-min-ep", "1e9"]
    if args.train_envs:
        flags += ["--envs", args.train_envs]
    flags += list(args.train_extra or [])
    env = {"PATH": shim + os.pathsep + os.environ.get("PATH", ""),
           "ARM_RESUME": "1", "CKPT": str(ckpt).replace("\\", "/"),
           "BUDGET": str(int(float(args.train_steps))),
           "RECORD_EVERY": "1e12", "EVAL_EPS": str(args.episodes),
           "PYTHONIOENCODING": "utf-8"}
    if find_trainer(run_name):
        log(fh, f"stale trainer for {run_name}: killing")
        kill_all(find_trainer(run_name))
        time.sleep(5)
    cmd = [find_bash(), str(ROOT / "tools" / "run_arm.sh"), run_name] + \
        [str(f) for f in flags]
    rc = run(cmd, rdir / "train_launch.log", 900, env=env)
    log(fh, f"train: launcher rc={rc} (runs/{run_name}_launch.txt)")
    csvf = tdir / "progress.csv"
    last_note = 0.0
    while True:
        procs = find_trainer(run_name)
        now = time.time()
        if not procs:
            break
        if now - t0 > args.train_wall:
            kill_all(procs)
            log(fh, f"train: WALL {args.train_wall:.0f}s hit, killed")
            break
        try:
            stale = now - os.path.getmtime(csvf)
        except OSError:
            stale = 0.0 if now - t0 < args.train_grace else now - t0
        if now - t0 > args.train_grace and stale > args.train_stall:
            kill_all(procs)
            log(fh, f"train: STALLED (csv stale {stale:.0f}s), killed")
            break
        if now - last_note > 600:
            last_note = now
            step = None
            try:
                import csv as _csv
                with open(csvf, encoding="utf-8", errors="replace") as f:
                    rows = list(_csv.DictReader(f))
                if rows:
                    step = int(rows[-1].get("time/total_timesteps") or 0)
            except OSError:
                pass
            log(fh, f"train: t+{(now - t0) / 60:.1f} min step={step}")
        time.sleep(30)
    final = None
    for cand in ("ckpt_final.pt", "ckpt_latest.pt"):
        if (tdir / cand).exists():
            final = tdir / cand
            break
    info = {"run": run_name, "wall_s": round(time.time() - t0, 1),
            "ckpt": None if final is None else str(final),
            "launcher_rc": rc}
    if final is None:
        raise SystemExit(f"train: no checkpoint in {tdir}")
    # the BC diagnostics the trainer wrote (last line)
    bl = tdir / "bc_log.csv"
    if bl.exists():
        lines = bl.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > 1:
            info["bc_last"] = lines[-1]
            info["bc_first"] = lines[1]
    log(fh, f"train: done in {info['wall_s']}s -> {final} "
            f"(bc first {info.get('bc_first')} last {info.get('bc_last')})")
    return final, info


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="expert iteration driver")
    ap.add_argument("seed_ckpt")
    ap.add_argument("--name", default="exit")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--train-steps", type=float, default=3e8,
                    help="env steps of warm PPO+BC per round")
    ap.add_argument("--plan-budget", type=float, default=600.0,
                    help="seconds of planner waves per round (at least "
                         "--plan-min-waves waves run regardless)")
    ap.add_argument("--plan-min-waves", type=int, default=1)
    ap.add_argument("--plan-max-waves", type=int, default=64)
    ap.add_argument("--plan-envs", type=int, default=2048)
    ap.add_argument("--plan-resample", type=int, default=25,
                    help="beam_tas --resample-every (round 27: R=25 best)")
    ap.add_argument("--plan-elite", type=float, default=0.25)
    ap.add_argument("--plan-greedy-envs", type=int, default=64,
                    help="beam_tas --greedy-envs: greedy continuations of "
                         "the elites (env 0 = the untouched greedy line), so "
                         "a wave can never lose to the policy's own mode")
    ap.add_argument("--plan-score", default="dv",
                    choices=["d", "route", "v", "dv"],
                    help="beam_tas --score; 'dv' = geodesic d until the "
                         "frontier is within --plan-v-switch of the goal, "
                         "then the checkpoint's critic (on xQR32: 76.65 s "
                         "vs 76.79 s for plain d, vs no crossing at all "
                         "without greedy envs)")
    ap.add_argument("--plan-v-switch", type=float, default=20000.0)
    ap.add_argument("--plan-seed", type=int, default=0,
                    help="beam_tas spawn seed (its greedy gate's spawn)")
    ap.add_argument("--plan-max-ticks", type=int, default=12000)
    ap.add_argument("--keep-finishers", type=int, default=8)
    ap.add_argument("--bc-lines", type=int, default=16,
                    help="distil at most this many of the fastest distinct "
                         "finishing lines pooled over the crossing waves "
                         "(0 = all)")
    ap.add_argument("--line-weight-decay", type=float, default=0.0)
    ap.add_argument("--bc-coef", type=float, default=0.5)
    ap.add_argument("--bc-coef-final", type=float, default=0.0)
    ap.add_argument("--bc-batch", type=int, default=2048)
    ap.add_argument("--no-demo", action="store_true",
                    help="do NOT spawn along the planner's line (BC only)")
    ap.add_argument("--episodes", type=int, default=9,
                    help="greedy map-start eval episodes per policy")
    ap.add_argument("--eval-seed", type=int, default=777)
    ap.add_argument("--map", default=DEF_MAP)
    ap.add_argument("--train-envs", type=int, default=0,
                    help="override the checkpoint's --envs (smoke only)")
    ap.add_argument("--train-seed-base", type=int, default=100)
    ap.add_argument("--train-wall", type=float, default=4 * 3600.0)
    ap.add_argument("--train-grace", type=float, default=1200.0)
    ap.add_argument("--train-stall", type=float, default=900.0)
    ap.add_argument("--train-extra", nargs=argparse.REMAINDER,
                    help="everything after this goes to the trainer verbatim")
    args = ap.parse_args()

    root = ROOT / "runs" / args.name
    root.mkdir(parents=True, exist_ok=True)
    fh = open(root / "driver.log", "a", encoding="ascii", errors="replace")
    map_path = str(Path(args.map).resolve()).replace("\\", "/")
    if not Path(map_path).exists():
        raise SystemExit(f"map not found: {map_path}")
    if "RL_Surf_" in map_path and "/maps/" in map_path:
        log(fh, f"WARNING: {map_path} is inside a worktree - its caches "
                "re-bake on mtime; use the main checkout's map")
    shim = make_shim(root)
    summary = root / "expert_summary.jsonl"
    log(fh, f"=== {args.name}: {args.rounds} rounds x {args.train_steps:.3g} "
            f"steps, planner budget {args.plan_budget:.0f}s, bc coef "
            f"{args.bc_coef}->{args.bc_coef_final} batch {args.bc_batch}, "
            f"demo {'off' if args.no_demo else 'on'}, map {map_path}")
    policy = prepare_seed(Path(args.seed_ckpt).resolve(), root, fh)
    prev_eval = None

    for r in range(args.rounds):
        rdir = root / f"round_{r}"
        rdir.mkdir(parents=True, exist_ok=True)
        done_f = rdir / "round.json"
        if done_f.exists():
            info = json.loads(done_f.read_text(encoding="utf-8"))
            if info.get("done") and info.get("policy_out") \
                    and Path(info["policy_out"]).exists():
                policy = Path(info["policy_out"])
                prev_eval = info.get("eval_out")
                log(fh, f"round {r}: RESUME - complete, policy {policy}")
                continue
        t_round = time.time()
        log(fh, f"=== round {r}: policy_in {policy} md5 {md5(policy)}")
        # 1. the proposer's own greedy time (the previous round's eval_out)
        if prev_eval is not None and prev_eval.get("ckpt") == str(policy):
            ev_in = prev_eval
            log(fh, f"eval_in: reused round {r - 1}'s eval_out "
                    f"(best {ev_in['best_s']} s)")
        else:
            ev_in = eval_policy(policy, rdir, map_path, args.episodes,
                                args.eval_seed, fh, "eval_in")
        # 2. the planner
        pl = plan(policy, rdir, map_path, args, fh)
        if pl.get("best") is None:
            log(fh, f"round {r}: PLANNER CROSSED NOTHING - loop STOPPED")
            (rdir / "round.json").write_text(json.dumps(
                {"done": False, "why": "planner crossed nothing",
                 "eval_in": ev_in, "plan": pl}, indent=2), encoding="utf-8")
            break
        # 3. rows + spine
        bc, spine, bmeta = distil(pl["npz_all"], policy, rdir, map_path,
                                  args, fh)
        # 4. warm PPO + BC along the line
        run_name = f"{args.name}/round_{r}/train"
        nxt, tinfo = train(policy, rdir, run_name, bc, spine,
                           int(bmeta.get("spine_len") or 1), map_path, args,
                           shim, fh)
        # 5. the distilled policy's greedy time
        ev_out = eval_policy(nxt, rdir, map_path, args.episodes,
                             args.eval_seed, fh, "eval_out")
        row = {"round": r, "policy_in": str(policy), "policy_in_md5": md5(policy),
               "greedy_in_best_s": ev_in["best_s"],
               "greedy_in_mean_s": ev_in["mean_s"],
               "greedy_in_finishes": f"{ev_in['finishes']}/{ev_in['episodes']}",
               "planner_best_s": pl["best"]["best_s"],
               "planner_gate_greedy_s": ((pl.get("greedy_gate_ticks") or 0)
                                         / 100.0),
               "planner_waves": len(pl["waves"]),
               "planner_wall_s": pl["wall_s"],
               "bc_rows": bmeta["rows"], "bc_lines": bmeta["lines"],
               "spine_len": bmeta.get("spine_len"),
               "distil_wall_s": bmeta["wall_s"],
               "train_steps": float(args.train_steps),
               "train_wall_s": tinfo["wall_s"],
               "bc_first": tinfo.get("bc_first"), "bc_last": tinfo.get("bc_last"),
               "policy_out": str(nxt), "policy_out_md5": md5(nxt),
               "greedy_out_best_s": ev_out["best_s"],
               "greedy_out_mean_s": ev_out["mean_s"],
               "greedy_out_finishes": f"{ev_out['finishes']}/{ev_out['episodes']}",
               "eval_wall_s": ev_in["wall_s"] + ev_out["wall_s"],
               "round_wall_s": round(time.time() - t_round, 1)}
        with open(summary, "a", encoding="ascii", errors="replace") as sf:
            sf.write(json.dumps(row) + "\n")
        (rdir / "round.json").write_text(json.dumps(
            {"done": True, "policy_out": str(nxt), "eval_in": ev_in,
             "eval_out": ev_out, "plan": pl, "bc": bmeta, "train": tinfo,
             "summary": row}, indent=2), encoding="utf-8")
        log(fh, f"round {r} SUMMARY: greedy_in {ev_in['best_s']} s -> "
                f"planner {pl['best']['best_s']} s -> greedy_out "
                f"{ev_out['best_s']} s ({ev_out['finishes']}/"
                f"{ev_out['episodes']} finish); wall {row['round_wall_s']}s")
        policy, prev_eval = nxt, ev_out
    log(fh, f"=== {args.name} finished")
    (root / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
