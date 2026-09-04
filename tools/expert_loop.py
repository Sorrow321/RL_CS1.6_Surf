#!/usr/bin/env python3
"""expert_loop.py - AlphaZero-style expert iteration for the surf racer.

Each round: the current policy PROPOSES (greedy evals from the map start),
the beam planner IMPROVES the line by search in the real simulator
(tools/beam_tas.py), and the policy is trained to IMITATE the planner's
(state, action) pairs on top of its ordinary PPO objective, spawning along
the planner's line. The improvement compounds instead of being thrown away.

    round r (runs/<name>/round_<r>/):
      0. SCRATCH  (--seed scratch, round 0 only) tools/run_arm.sh SCRATCH=1:
                  the from-scratch baseline argument set plus the LOOP
                  experiment's start-state flags (--respawn-margin 2
                  --respawn-binned 1 --respawn-bins 128 --eval-stall 1),
                  plain geodesic race reward, --scratch-steps of PPO
                  -> scratch/ckpt_final.pt = policy_0
      1. EVAL     tools/record_ckpt.py: E greedy map-start episodes of
                  policy_r -> finish times (spawn basis; the ledger's number)
                  AND tools/eval_honesty.py's order-only corridor progress
                  (MAX / mean over the episodes, window 16) - the metric
                  that can see a non-finishing policy move
      2. PLAN     tools/beam_tas.py waves (torch seeds 0,1,...) until the
                  planning budget is spent. --objective finish: the fastest
                  crossing wins. --objective progress/auto: the furthest
                  order-only arc wins (ties by time), lineages that die
                  keep their best arc, no crossing needed -> plan/wave_*/
                  beam_best.npz (the K best lineages, replay-verified)
      3. DISTIL   tools/plan_to_bc.py: replay the kept lines -> bc.npz
                  (one row per decision: state, core scalars, latch, action)
                  + spine.npy (the best line's states). A non-finishing
                  line is TRIMMED at its last map contact first (the fall
                  is not a demo) and the spine is spaced uniformly along
                  the line's path, so spawns are uniform over the track
      4. TRAIN    tools/run_arm.sh ARM_RESUME=1: WARM resume of policy_r for
                  --train-steps with --bc-file bc.npz (the BC term inside
                  every PPO minibatch step, --bc-coef decaying linearly to
                  --bc-coef-final) and --demo-file spine.npy (the reservoir
                  share of every spawn pool drawn uniformly from the
                  planner's line, the xLOOP/xDEMO-validated demo flags)
                  -> train/ckpt_final.pt = policy_{r+1}
      5. EVAL     policy_{r+1} (reused as round r+1's step 1)

--objective auto (default) plans for PROGRESS until a round's planner has
crossed the finish (or the policy finishes greedily), then for FINISH TIME,
the proven finisher configuration (dv score, greedy envs). Under auto the
progress planner still ranks any finisher first, and a finish-mode planner
that crosses nothing falls back to a progress plan in the same round
instead of stopping the loop. The objective in force is logged per round.

Every subprocess is logged under the round directory; every path handed to
a subprocess is absolute (the map ALWAYS from the main checkout on Windows:
a worktree copy re-bakes every cache; on a rented Linux box pass --map
/root/RL_Surf/maps/<map>.bsp). The machine-readable result is
runs/<name>/expert_summary.jsonl, one line per round (SUMMARY_KEYS): the
seed's greedy time and corridor, the planner's time or best arc, the
distilled policy's greedy time and corridor, and the wall clock of every
phase. A completed round (round.json with done=true) is skipped on
restart, so a crashed overnight loop resumes where it stopped.

A seed with a --quantiles critic (runs/research/xQR32) is collapsed to the
scalar head first (tools/ckpt_qr_to_scalar.py: exact for the value, actor
untouched) so the mainline trainer can resume it.

    python tools/expert_loop.py C:/RL_Surf/runs/research/xQR32/xQR32_final.pt \
        --name exit --rounds 10 --train-steps 3e8 --plan-budget 600
    python3 tools/expert_loop.py scratch --name exit_scratch --rounds 12 \
        --map /root/RL_Surf/maps/surf_src_cannonball.bsp \
        --scratch-steps 1.5e9 --train-steps 3e8 --plan-budget 600
    python tools/expert_loop.py scratch --name dry --dry-run   # CPU, minutes
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
sys.path.insert(0, str(ROOT / "tools"))
import expert_dagger   # noqa: E402  DAgger relabel phase (--dagger-k; off = unchanged)
PY = sys.executable
WIN = os.name == "nt"
_MAIN_MAP = Path("C:/RL_Surf/maps/surf_src_cannonball.bsp")
DEF_MAP = str(_MAIN_MAP if (WIN and _MAIN_MAP.exists())
              else ROOT / "maps" / "surf_src_cannonball.bsp")
FINISH_PAD = 64.0          # eval_honesty's finish tolerance (loop_spine.py)
# the LOOP experiment's start-state flags on top of run_arm.sh's SCRATCH
# argument set (plain geodesic race reward, no goal-fan conditioning):
# 2 s harvest margin (round 18: the reservoir reaches the frontier),
# uniform over 128 goal-distance bins, and evals under training's stall rule
SCRATCH_FLAGS = ["--respawn-margin", "2", "--respawn-binned", "1",
                 "--respawn-bins", "128", "--eval-stall", "1"]
# The trainer overrides that make a CPU dry run take minutes. SPLIT in two,
# because half of them cannot be applied to a warm resume:
#   * the BUDGET half changes no tensor - a smaller rollout, fewer epochs,
#     shorter episodes - so it is safe from any seed;
#   * --emb / --hidden are the ARCHITECTURE, and passing them to a resume
#     builds a different network than the checkpoint's. That is not a
#     hypothetical: `--dry-run` from runs/research/xENT131 (emb 512,
#     hidden 448) died in the train phase with "--route cannot warm-start
#     this checkpoint: pi.0.weight is (448, 524), i.e. 449 route-side
#     columns over a 75-wide trunk" - a route-block message for a run with
#     no route, because 524 = 11 scalars + 512 conv + 1 latch against this
#     run's 11 + 64. The trainer now names the real cause
#     (train_fast.check_arch_matches); the dry run stops causing it.
# A --seed scratch dry run still shrinks the net, because there is no
# checkpoint to disagree with and the tiny trunk is most of the speed-up.
DRY_BUDGET_FLAGS = ["--n-steps", "8", "--minibatches", "2", "--epochs", "1",
                    "--ep-ticks", "3000"]
DRY_SCRATCH_ARCH = ["--emb", "64", "--hidden", "64"]
DRY_TRAIN_FLAGS = DRY_SCRATCH_ARCH + DRY_BUDGET_FLAGS
EXTRA_ENV = {}             # main(): CUDA_VISIBLE_DEVICES=-1 under --cpu

SUMMARY_KEYS = (
    "round", "objective_mode", "objective", "policy_in", "policy_in_md5",
    "scratch_steps", "scratch_wall_s",
    "greedy_in_best_s", "greedy_in_mean_s", "greedy_in_finishes",
    "greedy_in_corridor_max", "greedy_in_corridor_mean",
    "planner_objective", "planner_crossed", "planner_best_s",
    "planner_best_arc", "planner_best_arc_pct", "planner_best_arc_s",
    "planner_finishes", "planner_gate_greedy_s", "planner_waves",
    "planner_kept_lines", "planner_fallback", "planner_wall_s",
    "bc_rows", "bc_lines", "bc_finishers", "bc_best_arc",
    "spine_len", "spine_ticks", "spine_trim_ticks", "spine_spacing",
    "distil_wall_s",
    "train_steps", "train_wall_s", "bc_first", "bc_last",
    "policy_out", "policy_out_md5",
    "greedy_out_best_s", "greedy_out_mean_s", "greedy_out_finishes",
    "greedy_out_corridor_max", "greedy_out_corridor_mean",
    "eval_wall_s", "round_wall_s")


def log(fh, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def tick_secs(ticks, tick_ms=None, pattern=None) -> float:
    """Planner / recorder ticks -> seconds at the tick the FILE says (a
    beam_tas summary.json's tick_ms + tick_pattern_ms, a trajectory
    header's); 10 ms when it says nothing, which is every file that
    predates --tick-ms. Never divide a tick count by 100 here."""
    from surfgym.tick import ticks_to_secs
    return ticks_to_secs(int(ticks or 0),
                         10.0 if tick_ms is None else float(tick_ms), pattern)


def traj_headers(traj_path):
    """The per-episode headers of a record_rollout .jsonl (the recorder's
    time base: tick_ms, and tick_pattern_ms + tick_phase under a pattern),
    in file order; [] when the file has none."""
    from surfgym.route import episodes_from_traj
    try:
        _eps, hdrs = episodes_from_traj(traj_path, with_headers=True)
    except (OSError, ValueError):
        return []
    return [h for h in hdrs if h]


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_bash():
    if not WIN:
        return "bash"
    for p in (r"C:\Program Files\Git\bin\bash.exe",
              r"C:\Program Files\Git\usr\bin\bash.exe"):
        if os.path.exists(p):
            return p
    return "bash"


def make_shim(root: Path) -> str:
    """run_arm.sh invokes `python3`, which Windows python installs do not
    provide (tools/loop_driver.py's shim, verbatim). A Linux box has it."""
    if not WIN:
        return ""
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
    e.update(EXTRA_ENV)
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
# pure helpers (unit-tested)
# --------------------------------------------------------------------------
def next_objective(mode: str, crossed_before: bool = False,
                   eval_finishes: int = 0) -> str:
    """The beam_tas --objective for a round.

    'finish' and 'progress' are what they say. 'auto' plans for PROGRESS
    (passing beam_tas 'auto', so a lineage that happens to finish still
    ranks first, by time) until a planner has crossed the finish line or
    the policy's greedy evals finish; from then on it plans for FINISH
    TIME, the proven finisher configuration."""
    if mode in ("finish", "progress"):
        return mode
    if mode != "auto":
        raise ValueError(f"unknown objective mode {mode!r}")
    return "finish" if (bool(crossed_before)
                        or int(eval_finishes or 0) > 0) else "auto"


def summary_row(**kw) -> dict:
    """One expert_summary.jsonl line: every SUMMARY_KEYS key, in order,
    None where a phase did not run. An unknown key is a bug, not a new
    column."""
    bad = sorted(set(kw) - set(SUMMARY_KEYS))
    if bad:
        raise KeyError(f"summary_row: unknown keys {bad}")
    return {k: kw.get(k) for k in SUMMARY_KEYS}


def select_waves(waves, objective: str):
    """Rank a round's planner waves. -> (result, best, ranked) where result
    is 'finish' (some wave crossed: the fastest crossing wins, ties by wave
    index), 'progress' (no crossing, objective progress/auto: the furthest
    best arc wins, ties by the EARLIER arc tick, then wave index) or None
    (nothing usable). ``ranked`` lists every usable wave best first -
    crossings by time, then arc waves by arc - for pooling into the BC
    rows. A wave row is beam_tas's summary.json subset that plan() keeps:
    crossed, best_ticks, best_arc, best_arc_tick, kept_lines, wave."""
    crossed = [x for x in waves if x.get("crossed") and x.get("best_ticks")]
    arcs = [x for x in waves if not x.get("crossed")
            and x.get("best_arc") is not None and (x.get("kept_lines") or 0)]
    by_time = sorted(crossed, key=lambda x: (x["best_ticks"], x["wave"]))
    by_arc = sorted(arcs, key=lambda x: (-float(x["best_arc"]),
                                         int(x.get("best_arc_tick") or 0),
                                         x["wave"]))
    if by_time:
        return "finish", by_time[0], by_time + by_arc
    if by_arc and objective != "finish":
        return "progress", by_arc[0], by_arc
    return None, None, []


def honesty_scores(traj_path, pts, spacing, corridor: float, window: int):
    """tools/eval_honesty.py's order-only corridor progress of every
    episode in a record_ckpt trajectory (its own functions, same numbers
    the CLI prints under --order-only WINDOW). -> list of floats (u)."""
    from eval_honesty import corridor_progress_ordered
    from surfgym.route import episodes_from_traj
    out = []
    for ep in episodes_from_traj(traj_path):
        xyz = np.asarray(ep[:, 1:4], np.float32)
        q, _off = corridor_progress_ordered(xyz, pts, spacing, corridor,
                                            window)
        out.append(float(q))
    return out


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


def eval_policy(ckpt: Path, rdir: Path, map_path: str, args, fh,
                tag: str, route) -> dict:
    """E greedy map-start episodes -> finish times (spawn basis) and the
    order-only corridor progress (eval_honesty) of every episode."""
    traj = rdir / f"{tag}.jsonl"
    states = rdir / f"{tag}_states.npz"
    t0 = time.time()
    cmd = [PY, "-u", ROOT / "tools" / "record_ckpt.py", ckpt,
           "--map", map_path, "--episodes", args.episodes,
           "--seed", args.eval_seed, "--out", traj, "--dump-states", states]
    rc = run(cmd, rdir / f"{tag}.log", 3600)
    if rc != 0 or not states.exists():
        raise SystemExit(f"eval of {ckpt} failed (rc={rc}, see {tag}.log)")
    from surfgym.zones import load_zones
    zones = load_zones(map_path)
    lo = np.asarray(zones["end"]["mins"], np.float64) - FINISH_PAD
    hi = np.asarray(zones["end"]["maxs"], np.float64) + FINISH_PAD
    z = np.load(states, allow_pickle=False)
    # the recording's time base, from its own headers (record_rollout
    # writes tick_ms, and the pattern + phase under --tick-ms); episode i
    # of the state dump is episode i of the trajectory
    from surfgym.tick import episode_seconds
    hdrs = traj_headers(traj)
    keys = sorted(z.files)
    per_ep = hdrs if len(hdrs) == len(keys) else []
    h0 = hdrs[0] if hdrs else {}
    times, ends = [], []
    for i, k in enumerate(keys):
        e = z[k]
        o = np.asarray(e["origin"], np.float64)
        fin = bool(np.any(np.all((o >= lo) & (o <= hi), axis=1)))
        ends.append(int(len(e)))
        if fin:
            if per_ep and per_ep[i].get("tick_ms") is not None:
                times.append(episode_seconds(per_ep[i], len(e)))
            else:
                times.append(tick_secs(len(e), h0.get("tick_ms"),
                                       h0.get("tick_pattern_ms")))
    pts, spacing = route
    corr = honesty_scores(traj, pts, spacing, args.corridor,
                          args.order_window)
    res = {"ckpt": str(ckpt), "episodes": int(len(ends)),
           "finishes": int(len(times)),
           "best_s": (min(times) if times else None),
           "mean_s": (float(np.mean(times)) if times else None),
           "times_s": times, "episode_ticks": ends,
           "corridor": corr,
           "corridor_max": (float(max(corr)) if corr else None),
           "corridor_mean": (float(np.mean(corr)) if corr else None),
           "corridor_window": int(args.order_window),
           "route_len": float((len(pts) - 1) * spacing),
           "wall_s": round(time.time() - t0, 1)}
    (rdir / f"{tag}.json").write_text(json.dumps(res, indent=2),
                                      encoding="utf-8")
    log(fh, f"{tag}: {res['finishes']}/{res['episodes']} finished, best "
            f"{res['best_s']} s, mean {res['mean_s']} s; order-only "
            f"corridor MAX {res['corridor_max']} mean "
            f"{None if res['corridor_mean'] is None else round(res['corridor_mean'])}"
            f" of {res['route_len']:,.0f}u ({res['wall_s']}s)")
    return res


def plan(ckpt: Path, rdir: Path, map_path: str, objective: str, args, fh,
         tag: str = "plan") -> dict:
    """beam_tas waves until the budget is spent. objective 'finish': the
    fastest crossing wins; 'progress'/'auto': the furthest best arc wins
    (ties by the tick it was reached at), finishers first under auto."""
    pdir = rdir / tag
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
               "--objective", objective,
               "--route-file", args.route, "--corridor", args.corridor,
               "--arc-window", args.order_window,
               "--arc-quant", args.arc_quant, "--arc-bank", args.arc_bank,
               "--contact-tol", args.contact_tol,
               "--out-dir", wdir]
        if objective != "finish":
            # the gate episode only supplies a finisher's matched greedy
            # time; a progress search needs the spawn state alone
            cmd.append("--skip-gate")
        tw = time.time()
        rc = run(cmd, pdir / f"wave_{w}.log", max(600, args.plan_budget * 3))
        row = {"wave": w, "rc": rc, "wall_s": round(time.time() - tw, 1),
               "crossed": False}
        sfile = wdir / "summary.json"
        if sfile.exists():
            s = json.loads(sfile.read_text(encoding="utf-8"))
            row.update(crossed=bool(s.get("crossed")),
                       best_ticks=s.get("best_ticks"), best_s=s.get("best_s"),
                       tick_ms=s.get("tick_ms"),
                       tick_pattern_ms=s.get("tick_pattern_ms"),
                       greedy_ticks=s.get("greedy_ticks"),
                       finishes=s.get("finishes"),
                       best_arc=s.get("best_arc"),
                       best_arc_tick=s.get("best_arc_tick"),
                       arc_pct=s.get("arc_pct"),
                       kept_lines=s.get("kept_lines"),
                       diverged_lines=s.get("diverged_lines"),
                       search_wall_s=s.get("search_wall_s"),
                       env_steps_per_s=s.get("env_steps_per_s"))
            if row["crossed"] and row.get("kept_lines") is None:
                row["kept_lines"] = 1
        waves.append(row)
        log(fh, f"{tag} wave {w}: rc={rc} crossed={row.get('crossed')} "
                f"best={row.get('best_s')} s arc={row.get('best_arc')} "
                f"({row.get('arc_pct')}%) at "
                f"{tick_secs(row.get('best_arc_tick'), row.get('tick_ms'), row.get('tick_pattern_ms')):.2f} s "
                f"finishes={row.get('finishes')} kept={row.get('kept_lines')}"
                f" ({row['wall_s']}s)")
        w += 1
        spent = time.time() - t0
        if w >= args.plan_max_waves:
            break
        if w >= args.plan_min_waves and spent + row["wall_s"] > args.plan_budget:
            break
    res = {"waves": waves, "objective": objective,
           "wall_s": round(time.time() - t0, 1), "best": None,
           "result": None}
    result, best, ranked = select_waves(waves, objective)
    if result is None:
        (pdir / "plan.json").write_text(json.dumps(res, indent=2),
                                        encoding="utf-8")
        log(fh, f"{tag}: NOTHING usable from {len(waves)} wave(s) "
                f"(objective {objective}), {res['wall_s']}s")
        return res
    crossed = [x for x in ranked if x.get("crossed")]
    arcs = [x for x in ranked if not x.get("crossed")]
    res["result"] = result
    res["best"] = best
    res["best_npz"] = str(pdir / f"wave_{best['wave']}" / "beam_best.npz")
    # every wave's kept lineages pool into the BC rows (same spawn,
    # different --torch-seed): plan_to_bc ranks them again as one set
    res["npz_all"] = [str(pdir / f"wave_{x['wave']}" / "beam_best.npz")
                      for x in ranked]
    res["greedy_gate_ticks"] = best.get("greedy_ticks")
    res["tick_ms"] = best.get("tick_ms")
    res["tick_pattern_ms"] = best.get("tick_pattern_ms")
    res["finishes"] = int(sum(int(x.get("finishes") or 0) for x in waves))
    (pdir / "plan.json").write_text(json.dumps(res, indent=2),
                                    encoding="utf-8")
    if res["result"] == "finish":
        log(fh, f"{tag}: best wave {best['wave']} {best['best_s']} s "
                f"(gate greedy {tick_secs(best.get('greedy_ticks'), best.get('tick_ms'), best.get('tick_pattern_ms')):.2f} s), "
                f"{len(crossed)}/{len(waves)} waves crossed, {res['wall_s']}s")
    else:
        log(fh, f"{tag}: best wave {best['wave']} arc {best['best_arc']:,.0f}u "
                f"({best.get('arc_pct') or 0:.1f}%) at "
                f"{tick_secs(best.get('best_arc_tick'), best.get('tick_ms'), best.get('tick_pattern_ms')):.2f} s, "
                f"{len(arcs)}/{len(waves)} waves kept lines, "
                f"{res['finishes']} finishes, {res['wall_s']}s")
    return res


def distil(plan_npzs: list, ckpt: Path, rdir: Path, map_path: str,
           spine_spacing: float, args, fh):
    t0 = time.time()
    bc = rdir / "bc.npz"
    spine = rdir / "spine.npy"
    summ = rdir / "bc_summary.json"
    rc = run([PY, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan",
              *plan_npzs,
              "--ckpt", ckpt, "--out", bc, "--spine", spine, "--map", map_path,
              "--lines", args.bc_lines,
              "--line-weight-decay", args.line_weight_decay,
              "--route", args.route, "--corridor", args.corridor,
              "--arc-window", args.order_window,
              "--contact-tol", args.contact_tol,
              "--spine-spacing", spine_spacing,
              "--summary-out", summ], rdir / "distil.log", 1800)
    if rc != 0 or not summ.exists():
        raise SystemExit(f"plan_to_bc failed (rc={rc}, see distil.log)")
    meta = json.loads(summ.read_text(encoding="utf-8"))
    meta["wall_s"] = round(time.time() - t0, 1)
    log(fh, f"distil: {meta['rows']:,} rows from {meta['lines']} line(s) "
            f"({meta.get('finishers')} finisher(s), best arc "
            f"{meta.get('best_arc')}), spine {meta.get('spine_len')} states "
            f"of {meta.get('spine_ticks')} ticks (trimmed "
            f"{meta.get('spine_trim_ticks')} ticks of fall) "
            f"({meta['wall_s']}s)")
    return bc, spine, meta


def wait_trainer(run_name: str, tdir: Path, t0: float, args, fh, tag: str):
    """Poll the detached trainer until it exits, the wall cap hits, or its
    progress.csv goes stale (the run_arm/loop_driver rule)."""
    csvf = tdir / "progress.csv"
    last_note = 0.0
    while True:
        procs = find_trainer(run_name)
        now = time.time()
        if not procs:
            break
        if now - t0 > args.train_wall:
            kill_all(procs)
            log(fh, f"{tag}: WALL {args.train_wall:.0f}s hit, killed")
            break
        try:
            stale = now - os.path.getmtime(csvf)
        except OSError:
            stale = 0.0 if now - t0 < args.train_grace else now - t0
        if now - t0 > args.train_grace and stale > args.train_stall:
            kill_all(procs)
            log(fh, f"{tag}: STALLED (csv stale {stale:.0f}s), killed")
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
            log(fh, f"{tag}: t+{(now - t0) / 60:.1f} min step={step}")
        time.sleep(5 if args.dry_run else 30)
    for cand in ("ckpt_final.pt", "ckpt_latest.pt"):
        if (tdir / cand).exists():
            return tdir / cand
    return None


def launch_arm(run_name: str, flags: list, env: dict, rdir: Path, fh,
               tag: str, shim: str):
    """tools/run_arm.sh <run_name> flags..., detached; -> launcher rc."""
    e = {"PATH": shim + os.pathsep + os.environ.get("PATH", "") if shim
         else os.environ.get("PATH", ""),
         "PYTHONIOENCODING": "utf-8"}
    e.update(env)
    if find_trainer(run_name):
        log(fh, f"stale trainer for {run_name}: killing")
        kill_all(find_trainer(run_name))
        time.sleep(5)
    cmd = [find_bash(), str(ROOT / "tools" / "run_arm.sh"), run_name] + \
        [str(f) for f in flags]
    rc = run(cmd, rdir / f"{tag}_launch.log", 900, env=e)
    log(fh, f"{tag}: launcher rc={rc} (runs/{run_name}_launch.txt; a "
            f"non-zero rc with a checkpoint is a short run outrunning "
            f"the liveness probe)")
    return rc


def train_scratch(rdir: Path, run_name: str, map_path: str, args, shim: str,
                  fh) -> tuple:
    """Round 0 of --seed scratch: the SCRATCH branch of run_arm.sh (the
    complete from-scratch argument set) + SCRATCH_FLAGS, --scratch-steps.
    -> (ckpt_final, info)."""
    t0 = time.time()
    tdir = ROOT / "runs" / run_name
    flags = list(SCRATCH_FLAGS) + [
        "--seed", args.train_seed_base, "--no-eval-at-start",
        "--record-every", "1e12", "--ckpt-every", "1e12"]
    if args.train_envs:
        flags += ["--envs", args.train_envs]
    flags += list(args.train_extra or [])
    env = {"SCRATCH": "1", "MAP": map_path,
           "BUDGET": str(int(float(args.scratch_steps))),
           "RECORD_EVERY": "1e12", "EVAL_EPS": str(args.episodes)}
    rc = launch_arm(run_name, flags, env, rdir, fh, "scratch", shim)
    final = wait_trainer(run_name, tdir, t0, args, fh, "scratch")
    info = {"run": run_name, "wall_s": round(time.time() - t0, 1),
            "ckpt": None if final is None else str(final),
            "launcher_rc": rc, "steps": float(args.scratch_steps),
            "flags": [str(f) for f in flags]}
    if final is None:
        raise SystemExit(f"scratch: no checkpoint in {tdir} (rc={rc}, see "
                         f"scratch_launch.log and runs/{run_name}_launch.txt)")
    log(fh, f"scratch: {args.scratch_steps:.3g} steps from nothing in "
            f"{info['wall_s']}s -> {final}")
    return final, info


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
    env = {"ARM_RESUME": "1", "CKPT": str(ckpt).replace("\\", "/"),
           "BUDGET": str(int(float(args.train_steps))),
           "RECORD_EVERY": "1e12", "EVAL_EPS": str(args.episodes)}
    rc = launch_arm(run_name, flags, env, rdir, fh, "train", shim)
    final = wait_trainer(run_name, tdir, t0, args, fh, "train")
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
def build_parser():
    ap = argparse.ArgumentParser(description="expert iteration driver")
    ap.add_argument("seed_ckpt",
                    help="a race checkpoint to improve, or 'scratch' to "
                         "train round 0's policy from nothing first")
    ap.add_argument("--name", default="exit")
    ap.add_argument("--rounds", type=int, default=None,
                    help="planner rounds (default 10; 1 under --dry-run)")
    ap.add_argument("--objective", choices=["auto", "progress", "finish"],
                    default="auto",
                    help="what the planner optimises: 'auto' = progress "
                         "(order-only corridor arc) until a round's planner "
                         "finishes or the policy finishes greedily, then "
                         "finish time; 'progress' / 'finish' fixed")
    ap.add_argument("--train-steps", type=float, default=3e8,
                    help="env steps of warm PPO+BC per round")
    ap.add_argument("--scratch-steps", type=float, default=None,
                    help="--seed scratch: env steps of round 0's from-"
                         "scratch PPO (default: --train-steps)")
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
                         "without greedy envs). Under a progress objective "
                         "this is the tie-break inside an arc bin")
    ap.add_argument("--plan-v-switch", type=float, default=20000.0)
    ap.add_argument("--plan-seed", type=int, default=0,
                    help="beam_tas spawn seed (its greedy gate's spawn)")
    ap.add_argument("--plan-max-ticks", type=int, default=None,  # 12000
                    help="beam_tas --max-ticks (default 12000). None means "
                         "'not given', which is what lets --dry-run shorten "
                         "it to 500 WITHOUT overriding a caller who asked "
                         "for a full-length window")
    ap.add_argument("--arc-quant", type=float, default=0.0,
                    help="beam_tas --arc-quant: elite arc bins (u) so the "
                         "--plan-score value decides inside a bin; 0 = exact")
    ap.add_argument("--arc-bank", choices=["contact", "raw"],
                    default="contact",
                    help="beam_tas --arc-bank: credit a lineage's arc only "
                         "at map-contact ticks (default: a fall earns "
                         "nothing, matching the distiller's trim) or at "
                         "every live tick (raw)")
    ap.add_argument("--keep-finishers", "--keep-lines", type=int, default=8,
                    dest="keep_finishers",
                    help="lineages each wave keeps (finishers first)")
    ap.add_argument("--bc-lines", type=int, default=16,
                    help="distil at most this many of the best distinct "
                         "lines pooled over the round's waves (0 = all)")
    ap.add_argument("--line-weight-decay", type=float, default=0.0)
    ap.add_argument("--bc-coef", type=float, default=0.5)
    ap.add_argument("--bc-coef-final", type=float, default=0.0)
    ap.add_argument("--bc-batch", type=int, default=2048)
    ap.add_argument("--no-demo", action="store_true",
                    help="do NOT spawn along the planner's line (BC only)")
    ap.add_argument("--spine-spacing", type=float, default=None,
                    help="plan_to_bc --spine-spacing: spine states this "
                         "many u of travel apart (default: 64 under a "
                         "progress plan, 0 = every tick under finish)")
    ap.add_argument("--contact-tol", type=float, default=1.0,
                    help="plan_to_bc --contact-tol (last-contact trim)")
    ap.add_argument("--episodes", type=int, default=9,
                    help="greedy map-start eval episodes per policy")
    ap.add_argument("--eval-seed", type=int, default=777)
    ap.add_argument("--map", default=DEF_MAP)
    ap.add_argument("--route", default=None,
                    help="route .npz for the order-only corridor (planner "
                         "objective + eval honesty); default <map>.route.npz")
    ap.add_argument("--corridor", type=float, default=1500.0)
    ap.add_argument("--order-window", type=int, default=16,
                    help="eval_honesty --order-only window (route vertices)")
    ap.add_argument("--train-envs", type=int, default=0,
                    help="override the checkpoint's --envs (smoke only)")
    ap.add_argument("--train-seed-base", type=int, default=100)
    ap.add_argument("--train-wall", type=float, default=4 * 3600.0)
    ap.add_argument("--train-grace", type=float, default=1200.0)
    ap.add_argument("--train-stall", type=float, default=900.0)
    ap.add_argument("--cpu", action="store_true",
                    help="hide CUDA from every subprocess "
                         "(CUDA_VISIBLE_DEVICES=-1): the wiring test")
    ap.add_argument("--dry-run", action="store_true",
                    help="every phase on CPU with tiny budgets: 64 envs, a "
                         "5-second planner window, 2 lineages, ~8k train "
                         "steps, 2 eval episodes, 1 round")
    ap.add_argument("--train-extra", nargs=argparse.REMAINDER,
                    help="everything after this goes to the trainer verbatim")
    expert_dagger.add_args(ap)
    return ap


PLAN_MAX_TICKS = 12000     # --plan-max-ticks' default, resolved in main()


def apply_dry_run(args):
    """The tiny budgets of --dry-run (CPU). Returns args."""
    args.cpu = True
    args.plan_envs = 64
    # 5 s of planning window unless the caller ASKED for a longer one. A
    # pre-flight from a real finisher wants that option: at 500 ticks every
    # kept lineage is still alive when the window ends, so no line reaches a
    # terminal, plan_to_bc's `zmask` is 0 everywhere and --bc-value-coef is
    # an (announced) no-op - which exercises the guard but not the term.
    if args.plan_max_ticks is None:
        args.plan_max_ticks = 500
    args.plan_resample = 10
    args.plan_greedy_envs = 4
    args.plan_budget = 0.0
    args.plan_min_waves = 1
    args.plan_max_waves = 1
    args.keep_finishers = 2
    args.bc_lines = 2
    args.episodes = 2
    args.train_steps = 8192.0
    args.scratch_steps = 8192.0
    args.train_envs = 64
    args.bc_batch = 64
    # a WARM seed keeps its own --emb/--hidden: they are tensor shapes, and
    # the trainer refuses a resume that disagrees with them (the comment on
    # DRY_TRAIN_FLAGS has the failure this prevents). Only a scratch dry run
    # gets the tiny net.
    warm = str(getattr(args, "seed_ckpt", "scratch")) != "scratch"
    dry = list(DRY_BUDGET_FLAGS if warm else DRY_TRAIN_FLAGS)
    args.train_extra = dry + list(args.train_extra or [])
    if args.rounds is None:
        args.rounds = 1
    return args


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        apply_dry_run(args)
    if args.rounds is None:
        args.rounds = 10
    if args.plan_max_ticks is None:
        args.plan_max_ticks = PLAN_MAX_TICKS
    if args.scratch_steps is None:
        args.scratch_steps = args.train_steps
    if args.cpu:
        EXTRA_ENV["CUDA_VISIBLE_DEVICES"] = "-1"

    root = ROOT / "runs" / args.name
    root.mkdir(parents=True, exist_ok=True)
    fh = open(root / "driver.log", "a", encoding="ascii", errors="replace")
    map_path = str(Path(args.map).resolve()).replace("\\", "/")
    if not Path(map_path).exists():
        raise SystemExit(f"map not found: {map_path}")
    if "RL_Surf_" in map_path and "/maps/" in map_path:
        log(fh, f"WARNING: {map_path} is inside a worktree - its caches "
                "re-bake on mtime; use the main checkout's map")
    if args.route is None:
        args.route = str(Path(map_path).with_suffix(".route.npz"))
    args.route = str(Path(args.route).resolve()).replace("\\", "/")
    if not Path(args.route).exists():
        raise SystemExit(f"route file not found: {args.route} (the order-"
                         "only corridor needs it; tools/build_route.py)")
    from eval_honesty import load_route
    route = load_route(Path(args.route))
    shim = make_shim(root)
    summary = root / "expert_summary.jsonl"
    scratch = str(args.seed_ckpt).lower() == "scratch"
    log(fh, f"=== {args.name}: {args.rounds} rounds x {args.train_steps:.3g} "
            f"steps, planner budget {args.plan_budget:.0f}s, objective "
            f"{args.objective}, bc coef {args.bc_coef}->{args.bc_coef_final} "
            f"batch {args.bc_batch}, demo {'off' if args.no_demo else 'on'}, "
            f"map {map_path}, route {args.route}"
            + (f", SCRATCH round 0 {args.scratch_steps:.3g} steps" if scratch
               else "")
            + (" [CPU]" if args.cpu else "")
            + (" [DRY RUN]" if args.dry_run else ""))
    policy = None if scratch else prepare_seed(Path(args.seed_ckpt).resolve(),
                                               root, fh)
    prev_eval = None
    crossed_before = False        # --objective auto: has a planner crossed?

    for r in range(args.rounds):
        rdir = root / f"round_{r}"
        rdir.mkdir(parents=True, exist_ok=True)
        done_f = rdir / "round.json"
        if done_f.exists():
            info = json.loads(done_f.read_text(encoding="utf-8"))
            if info.get("done") and info.get("policy_out") \
                    and Path(info["policy_out"]).exists():
                policy = Path(info["policy_out"])
                prev_eval = info.get("eval_in_next") or info.get("eval_out")
                crossed_before = bool(info.get("crossed_before"))
                log(fh, f"round {r}: RESUME - complete, policy {policy}, "
                        f"crossed_before {crossed_before}")
                continue
        t_round = time.time()
        sinfo = None
        if policy is None:
            # --seed scratch: round 0 trains policy_0 from nothing first
            sdir = rdir / "scratch"
            if (sdir / "ckpt_final.pt").exists():
                policy = sdir / "ckpt_final.pt"
                sinfo = {"wall_s": None, "steps": float(args.scratch_steps),
                         "resumed": True}
                log(fh, f"round {r}: scratch policy already trained: {policy}")
            else:
                log(fh, f"=== round {r}: SCRATCH training "
                        f"{args.scratch_steps:.3g} steps "
                        f"({' '.join(SCRATCH_FLAGS)})")
                policy, sinfo = train_scratch(
                    rdir, f"{args.name}/round_{r}/scratch", map_path, args,
                    shim, fh)
        log(fh, f"=== round {r}: policy_in {policy} md5 {md5(policy)}")
        # 1. the proposer's own greedy time (the previous round's eval_out)
        if prev_eval is not None and prev_eval.get("ckpt") == str(policy):
            ev_in = prev_eval
            log(fh, f"eval_in: reused round {r - 1}'s eval_out "
                    f"(best {ev_in['best_s']} s, corridor MAX "
                    f"{ev_in.get('corridor_max')})")
        else:
            ev_in = eval_policy(policy, rdir, map_path, args, fh, "eval_in",
                                route)
        # 2. the planner, under the objective in force
        objective = next_objective(args.objective, crossed_before,
                                   ev_in["finishes"])
        log(fh, f"round {r}: objective in force {objective} (mode "
                f"{args.objective}, crossed_before {crossed_before}, "
                f"greedy finishes {ev_in['finishes']})")
        pl = plan(policy, rdir, map_path, objective, args, fh)
        fallback = False
        if pl.get("best") is None and objective == "finish" \
                and args.objective == "auto":
            # a finish planner that crosses nothing on a policy that used
            # to finish: plan for progress in the same round instead of
            # stopping the loop
            log(fh, f"round {r}: finish planner crossed nothing - FALLING "
                    "BACK to a progress plan")
            fallback = True
            objective = "auto"
            pl = plan(policy, rdir, map_path, objective, args, fh,
                      tag="plan_fallback")
        if pl.get("best") is None:
            log(fh, f"round {r}: PLANNER PRODUCED NOTHING - loop STOPPED")
            (rdir / "round.json").write_text(json.dumps(
                {"done": False, "why": "planner produced nothing",
                 "objective": objective, "eval_in": ev_in, "plan": pl},
                indent=2), encoding="utf-8")
            break
        if pl["result"] == "finish":
            crossed_before = True
        # 3. rows + spine (trimmed at the last contact if not a finisher)
        spacing = args.spine_spacing
        if spacing is None:
            spacing = 0.0 if pl["result"] == "finish" else 64.0
        bc, spine, bmeta = distil(pl["npz_all"], policy, rdir, map_path,
                                  spacing, args, fh)
        # 3b. --dagger-k: relabel the policy's OWN states (default off)
        bc, bmeta = expert_dagger.maybe_relabel(args, policy, rdir, map_path,
                                                bc, spine, bmeta, fh, run, log)
        # 4. warm PPO + BC along the line
        run_name = f"{args.name}/round_{r}/train"
        nxt, tinfo = train(policy, rdir, run_name, bc, spine,
                           int(bmeta.get("spine_len") or 1), map_path, args,
                           shim, fh)
        # 5. the distilled policy's greedy time and corridor
        ev_out = eval_policy(nxt, rdir, map_path, args, fh, "eval_out", route)
        best = pl["best"]
        row = summary_row(
            round=r, objective_mode=args.objective, objective=objective,
            policy_in=str(policy), policy_in_md5=md5(policy),
            scratch_steps=(None if sinfo is None else sinfo.get("steps")),
            scratch_wall_s=(None if sinfo is None else sinfo.get("wall_s")),
            greedy_in_best_s=ev_in["best_s"], greedy_in_mean_s=ev_in["mean_s"],
            greedy_in_finishes=f"{ev_in['finishes']}/{ev_in['episodes']}",
            greedy_in_corridor_max=ev_in.get("corridor_max"),
            greedy_in_corridor_mean=ev_in.get("corridor_mean"),
            planner_objective=pl["objective"], planner_crossed=pl["result"] == "finish",
            planner_best_s=best.get("best_s"),
            planner_best_arc=best.get("best_arc"),
            planner_best_arc_pct=best.get("arc_pct"),
            planner_best_arc_s=(None if best.get("best_arc_tick") is None
                                else tick_secs(best["best_arc_tick"],
                                               best.get("tick_ms"),
                                               best.get("tick_pattern_ms"))),
            planner_finishes=pl.get("finishes"),
            planner_gate_greedy_s=tick_secs(pl.get("greedy_gate_ticks"),
                                            pl.get("tick_ms"),
                                            pl.get("tick_pattern_ms")),
            planner_waves=len(pl["waves"]),
            planner_kept_lines=int(sum(int(x.get("kept_lines") or 0)
                                       for x in pl["waves"])),
            planner_fallback=fallback, planner_wall_s=pl["wall_s"],
            bc_rows=bmeta["rows"], bc_lines=bmeta["lines"],
            bc_finishers=bmeta.get("finishers"),
            bc_best_arc=bmeta.get("best_arc"),
            spine_len=bmeta.get("spine_len"), spine_ticks=bmeta.get("spine_ticks"),
            spine_trim_ticks=bmeta.get("spine_trim_ticks"),
            spine_spacing=bmeta.get("spine_spacing"),
            distil_wall_s=bmeta["wall_s"],
            train_steps=float(args.train_steps), train_wall_s=tinfo["wall_s"],
            bc_first=tinfo.get("bc_first"), bc_last=tinfo.get("bc_last"),
            policy_out=str(nxt), policy_out_md5=md5(nxt),
            greedy_out_best_s=ev_out["best_s"], greedy_out_mean_s=ev_out["mean_s"],
            greedy_out_finishes=f"{ev_out['finishes']}/{ev_out['episodes']}",
            greedy_out_corridor_max=ev_out.get("corridor_max"),
            greedy_out_corridor_mean=ev_out.get("corridor_mean"),
            eval_wall_s=ev_in["wall_s"] + ev_out["wall_s"],
            round_wall_s=round(time.time() - t_round, 1))
        with open(summary, "a", encoding="ascii", errors="replace") as sf:
            sf.write(json.dumps(row) + "\n")
        (rdir / "round.json").write_text(json.dumps(
            {"done": True, "policy_out": str(nxt), "eval_in": ev_in,
             "eval_out": ev_out, "plan": pl, "bc": bmeta, "train": tinfo,
             "scratch": sinfo, "objective": objective,
             "crossed_before": crossed_before, "summary": row},
            indent=2), encoding="utf-8")
        log(fh, f"round {r} SUMMARY [{objective}]: greedy_in "
                f"{ev_in['best_s']} s / corridor {ev_in.get('corridor_max')} "
                f"-> planner "
                + (f"{best.get('best_s')} s" if pl["result"] == "finish"
                   else f"arc {best.get('best_arc')} ({best.get('arc_pct')}%)"
                        f" at {row['planner_best_arc_s']} s")
                + f" -> greedy_out {ev_out['best_s']} s / corridor "
                f"{ev_out.get('corridor_max')} ({ev_out['finishes']}/"
                f"{ev_out['episodes']} finish); wall {row['round_wall_s']}s")
        policy, prev_eval = nxt, ev_out
    log(fh, f"=== {args.name} finished")
    (root / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
