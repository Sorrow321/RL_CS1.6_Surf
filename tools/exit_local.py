"""exit_local.py - a minimal LOCAL expert-iteration loop (AlphaZero-style: search -> distil ->
train -> evaluate) for one single-map race checkpoint on this Windows box.

tools/expert_loop.py is the original, built around tools/run_arm.sh on the rented Linux boxes;
this driver calls the same three tools directly (the user, 2026-10-04, skate_laby: "implement
this alpha zero style expert loop ... I think it can help"):

  1. PLAN    tools/beam_tas.py waves (torch seeds 0, 1, ...) from the current checkpoint: a
             population of the policy's own proposals, the best lineages cloned over the rest
             every --resample decisions, a --greedy-envs floor, the K fastest finishers kept;
  2. DISTIL  tools/plan_to_bc.py over every crossing wave -> bc.npz (the planner's decisions:
             policy-space targets, held-key columns, search distribution, line returns) and the
             best line's per-tick spine;
  3. TRAIN   a warm resume of the checkpoint: PPO + the BC term (coef --bc-coef -> 0 over the
             round, --bc-target dist, --bc-value-coef) + spawns uniform along the spine (a fixed
             full window, as expert_loop pins it);
  4. EVAL    tools/record_ckpt.py greedy episodes of the round's checkpoint from the map start.

With --gate 1 (default) the next round plans from the GATED best checkpoint - AlphaGo Zero's
evaluator: a round's policy becomes the new base only when every eval episode finished and their
mean beats the best mean so far (<out>/best_mean.pt); otherwise the next round restarts from the
best. --gate 0 always continues from the latest (the plain order; on skate_laby it let one bad
round, 73.71 -> 75.34 s mean, seed the next). The best single eval run is copied to <out>/best.pt.
One JSON line per round in <out>/summary.jsonl.

Every input the trainer sees is the POLICY'S OWN (the planner only ever proposes from the
checkpoint's distribution), so the BC file and the spine are declared SELF_STATES=1 (CLAUDE.md
section 0) - never point --ckpt at a demo-trained checkpoint.

Usage:
    python tools/exit_local.py --ckpt runs/X/ckpt_best.pt --map maps/skate_laby.bsp \\
        --out runs/skEXIT --rounds 20 --deadline-h 7.5
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
PY = sys.executable
NO_WINDOW = 0x08000000 if os.name == "nt" else 0     # never pop a console (the user's rule)


def run(cmd, log_path: Path, env_extra=None, timeout=None) -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.update(env_extra or {})
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                           cwd=str(ROOT), env=env, creationflags=NO_WINDOW, timeout=timeout)
    return p.returncode


def log(fh, msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def ckpt_step(path: Path) -> int:
    import torch
    return int(torch.load(path, map_location="cpu", weights_only=False)["global_step"])


def finish_times(traj: Path):
    """-> (finish seconds list, episodes) of a record_ckpt trajectory: an episode that ended
    'done' finished; its time is its tick count at the header's tick."""
    out, n, hdr, rows = [], 0, None, 0
    for line in open(traj, encoding="utf-8"):
        o = json.loads(line)
        if isinstance(o, dict) and "map" in o:
            hdr, rows = o, 0
        elif isinstance(o, list):
            rows += 1
        elif isinstance(o, dict) and "end" in o:
            n += 1
            if o["end"] == "done":
                out.append(rows * float(hdr.get("tick_ms", 10.0)) / 1000.0)
    return out, n


def episode_times(traj: Path):
    """-> one entry per episode in file order: its finish seconds, or None if it did not finish.
    Two record_ckpt runs on the same --seed are aligned episode by episode (same spawns)."""
    out, hdr, rows = [], None, 0
    for line in open(traj, encoding="utf-8"):
        o = json.loads(line)
        if isinstance(o, dict) and "map" in o:
            hdr, rows = o, 0
        elif isinstance(o, list):
            rows += 1
        elif isinstance(o, dict) and "end" in o:
            out.append(rows * float(hdr.get("tick_ms", 10.0)) / 1000.0 if o["end"] == "done"
                       else None)
    return out


def eval_tick_ms(traj: Path) -> float:
    """The tick of a record_ckpt eval file: its first header's tick_ms (refused when absent)."""
    for line in open(traj, encoding="utf-8"):
        o = json.loads(line)
        if isinstance(o, dict) and "map" in o:
            if "tick_ms" not in o:
                raise SystemExit(f"{traj}: header without tick_ms - not assuming 10 ms")
            return float(o["tick_ms"])
    raise SystemExit(f"{traj}: no header")


def expected_gate(ci, ii, dnf_s: float):
    """--gate-rule expected: the mean episode time over the shared spawns, a DNF charged dnf_s.
    ci / ii are episode_times() lists aligned spawn by spawn -> (challenger, incumbent, accept)."""
    both = list(zip(ci, ii))
    if not both:
        return None, None, False
    ec = sum(dnf_s if a is None else a for a, _ in both) / len(both)
    ei = sum(dnf_s if b is None else b for _, b in both) / len(both)
    return ec, ei, ec < ei


def gate_seed(r: int, k: int) -> int:
    """The spawn seed of the k-th gate eval of round r, the same for the challenger and the
    incumbent's rematch: 7000 + r for the first (as it always was), then 20000 + 100 r + k."""
    return 7000 + r if k == 0 else 20000 + 100 * r + k


def gate_files(rdir: Path, n: int, incumbent: bool = False):
    """The gate's eval files of one round: eval.jsonl (eval_incumbent.jsonl) as before, then
    eval_<k>.jsonl (eval_incumbent_<k>.jsonl) for k = 1 .. n-1."""
    stem = "eval_incumbent" if incumbent else "eval"
    return [rdir / f"{stem}.jsonl"] + [rdir / f"{stem}_{k}.jsonl" for k in range(1, max(1, n))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the seed checkpoint (policy-derived only)")
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--deadline-h", type=float, default=8.0,
                    help="no new round starts after this many hours")
    ap.add_argument("--round-h", type=float, default=0.6,
                    help="expected hours per round (a round that would cross the deadline "
                         "is not started)")
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--plan-envs", type=int, default=2048)
    ap.add_argument("--greedy-envs", type=int, default=64)
    ap.add_argument("--resample", type=int, default=25)
    ap.add_argument("--score", default="d")
    ap.add_argument("--keep", type=int, default=16)
    ap.add_argument("--spawn-per-wave", type=int, default=0, choices=(0, 1),
                    help="1: each planner wave starts from its own spawn seed (the evals' start "
                         "jitter); 0: every wave from spawn seed 0")
    ap.add_argument("--max-ticks", type=int, default=23478)
    ap.add_argument("--train-steps", type=float, default=3e8)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bc-coef", type=float, default=0.5)
    ap.add_argument("--bc-value-coef", type=float, default=0.25)
    ap.add_argument("--bc-target", default="dist")
    ap.add_argument("--eval-eps", type=int, default=9)
    ap.add_argument("--gate", type=int, default=1, choices=(0, 1),
                    help="1: plan each round from the gated best checkpoint (all eval episodes "
                         "finished and the best mean so far); 0: always the latest")
    ap.add_argument("--seed-mean", type=float, default=float("inf"),
                    help="the seed's known eval mean (s): the gate's starting bar")
    ap.add_argument("--gate-rematch", type=int, default=0, choices=(0, 1),
                    help="1: every round re-evaluates the INCUMBENT on the same spawn seeds as the "
                         "challenger (record_ckpt --seed 7000+round for both) and gates on that "
                         "paired mean - AlphaGo Zero's evaluator plays both on fresh games each "
                         "time. 0: gate against the incumbent's own earlier mean, which after a "
                         "selection is optimistically biased (skate_laby rounds 15-16: 73.77 and "
                         "73.88 s rejected against round 14's 73.71)")
    ap.add_argument("--target-mean", type=float, default=None,
                    help="stop once a round is ACCEPTED with an eval mean at or below this (s); "
                         "skate_laby 2026-10-04: the user's goal 'a 1 s margin on the WR' = 73.88")
    ap.add_argument("--target-best", type=float, default=None,
                    help="stop as soon as ANY greedy eval run (the challenger's, or the "
                         "incumbent's rematch) finishes at or below this (s); its checkpoint is "
                         "copied to <out>/target_run.pt. skate_laby 2026-10-04, the user: a 1 s "
                         "margin on the WR, 'just one run is enough' = 73.88")
    ap.add_argument("--lr-halve-after", type=int, default=0,
                    help="after this many gate rejections IN A ROW, halve the training lr (down to "
                         "--lr-min) and count again; 0 = off. The manual rule of skate_laby rounds "
                         "19-24: lr 5e-5 kept breaking a polished incumbent, 2.5e-5 did not")
    ap.add_argument("--lr-min", type=float, default=1.25e-5)
    ap.add_argument("--bc-history", type=int, default=0,
                    help="dataset aggregation (DAgger's D <- D u D_i; AlphaZero's replay window): "
                         "train each round on its own planner lines PLUS those of the previous N "
                         "rounds (<out>/round_<k>/bc.npz + spine.npy, each round's own file, never "
                         "a pool); 0 = this round's lines only")
    ap.add_argument("--start-round", type=int, default=0,
                    help="the first round's number (directory and run names continue from it)")
    ap.add_argument("--plan-args", default="",
                    help="extra tools/beam_tas.py flags for every wave, one string (sidistic "
                         "2026-10-23: '--allow-nonfinisher --v-switch 5000', with --score dv)")
    ap.add_argument("--distil-args", default="",
                    help="extra tools/plan_to_bc.py flags, one string (e.g. '--no-value-target' "
                         "for a lineage whose reward plan_to_bc's line reward does not mirror)")
    ap.add_argument("--gate-rule", choices=("mean", "score", "paired", "expected"),
                    default="mean",
                    help="mean: every eval episode finished and the mean beats the bar (as "
                         "before). score: the GRADER's number (--grader: best progress + 10 x "
                         "finish fraction + the time term) of the challenger beats the "
                         "incumbent's on the same spawns (needs --gate-rematch 1) - finishing "
                         "more often first, then faster, for a seed that does not finish yet. "
                         "expected: the mean episode time over the same spawns, a DNF charged "
                         "--gate-dnf-s - pace and reliability at one exchange rate")
    ap.add_argument("--gate-dnf-s", type=float, default=None,
                    help="--gate-rule expected: the seconds a DNF costs (default: the eval cap, "
                         "--max-ticks at the eval file's tick). 'paired' trades up to "
                         "--gate-finish-tol finishes for pace and never pays a challenger for "
                         "finishing MORE, so the accepted policies' reliability can only drift "
                         "down (pbx 2026-10-30: r3 50/54 at a 66.39 s mean -> r10 64/81 at 65.67 "
                         "s, the expected time 67.0 -> 67.6 s; round 11 lost on +0.62 s with 24 "
                         "finishes to the incumbent's 19)")
    ap.add_argument("--gate-finish-tol", type=float, default=0.1,
                    help="--gate-rule paired: the challenger may finish this fraction of the eval "
                         "episodes FEWER than the incumbent (at least 1) and still be accepted when "
                         "it is faster on the spawns both finished. 'paired' is speed first with a "
                         "reliability floor; 'score' is the grader, where one more finisher of 9 "
                         "outweighs 37 s of best time (pbx 2026-10-28: round 23 replaced a 0.1 s "
                         "faster incumbent on one extra finish and the next planner lines lost 0.5 s)")
    ap.add_argument("--grader", default="C:/RL_Surf/autoresearch/score.py",
                    help="--gate-rule score: the score.py whose score_file grades an eval")
    ap.add_argument("--gate-evals", type=int, default=1,
                    help="evals per side per round: the challenger's eval and the incumbent's "
                         "rematch are each N record_ckpt runs of --eval-eps episodes on the same N "
                         "spawn seeds (gate_seed), and --gate-rule score gates on the MEAN grader "
                         "score over them (finishes and the paired time over all of them). The "
                         "grader counts only its first score_episodes (9) episodes of a file, so a "
                         "longer --eval-eps does not reduce the gate's noise; more files do (pbx "
                         "2026-10-28: one finisher of 9 is 1.11 points, a second of best time "
                         "0.03). 1 = one eval per side, as before")
    ap.add_argument("--demo-keys", choices=("auto", "off"), default="auto",
                    help="auto: when plan_to_bc wrote the spine's held keys (<spine>.keys.npy, a "
                         "--keys-hold checkpoint), train with --demo-keys so a spawn on the "
                         "spine starts holding the keys the line held there instead of neutral "
                         "(pbx 2026-10-28); off: never (the command line before the option)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra train_fast flags (everything after --extra)")
    args = ap.parse_args()
    import shlex
    plan_extra = shlex.split(args.plan_args)
    distil_extra = shlex.split(args.distil_args)
    grade = None
    if args.gate_rule in ("paired", "expected") and not (args.gate and args.gate_rematch):
        raise SystemExit(f"--gate-rule {args.gate_rule} compares on the same spawns: needs "
                         "--gate 1 --gate-rematch 1")
    if args.gate_dnf_s is not None and args.gate_rule != "expected":
        raise SystemExit("--gate-dnf-s is the DNF charge of --gate-rule expected")
    if args.gate_evals > 1 and not (args.gate and args.gate_rematch):
        raise SystemExit("--gate-evals N > 1 compares on shared spawns: needs --gate 1 "
                         "--gate-rematch 1")
    if args.gate_rule == "score":
        if not (args.gate and args.gate_rematch):
            raise SystemExit("--gate-rule score compares on the same spawns: needs --gate 1 "
                             "--gate-rematch 1")
        import importlib.util
        _spec = importlib.util.spec_from_file_location("exit_grader", args.grader)
        _grader = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_grader)

        def grade(p):
            s = _grader.score_file(Path(p)) if Path(p).exists() else {}
            return s.get("score")

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "exit_local.log", "a", encoding="utf-8")
    t_start = time.time()
    deadline = t_start + args.deadline_h * 3600.0
    cur = Path(args.ckpt).resolve()
    best_s, best_ck = float("inf"), None
    best_mean, best_mean_ck = float(args.seed_mean), cur
    lr_now, rejects_in_row, target_hit = float(args.lr), 0, False
    log(fh, f"seed {cur} (step {ckpt_step(cur):,}); deadline in {args.deadline_h:g} h")
    for r in range(args.start_round, args.start_round + args.rounds):
        if time.time() + args.round_h * 3600.0 > deadline:
            log(fh, f"round {r}: would cross the deadline - stopping")
            break
        rdir = out / f"round_{r}"
        rdir.mkdir(exist_ok=True)
        row = {"round": r, "ckpt_in": str(cur), "t0": time.strftime("%H:%M:%S")}
        # 1. PLAN ------------------------------------------------------------------------------
        t0 = time.time()
        npzs, waves = [], []
        for w in range(args.waves):
            wdir = rdir / f"wave_{w}"
            rc = run([PY, "-u", ROOT / "tools" / "beam_tas.py", cur, "--map", args.map,
                      "--envs", args.plan_envs, "--greedy-envs", args.greedy_envs,
                      "--score", args.score, "--resample-every", args.resample,
                      "--greedy-eps", 1, "--max-ticks", args.max_ticks,
                      "--keep-finishers", args.keep, "--route-file", "none",
                      "--torch-seed", 1000 * r + w,
                      # --spawn-per-wave: each wave plans from its own spawn draw, so the BC lines
                      # cover the start jitter the evals see (a wave of one spawn tunes one start)
                      "--seed", (1 + 100 * r + w) if args.spawn_per_wave else 0,
                      "--out-dir", wdir, *plan_extra],
                     rdir / f"wave_{w}.log", timeout=3600)
            s = {}
            if (wdir / "summary.json").exists():
                s = json.loads((wdir / "summary.json").read_text(encoding="utf-8"))
            waves.append({"wave": w, "rc": rc, "crossed": bool(s.get("crossed")),
                          "best_s": s.get("best_s"), "greedy_s": s.get("greedy_s")})
            if rc == 0 and s.get("crossed") and (wdir / "beam_best.npz").exists():
                npzs.append(wdir / "beam_best.npz")
            log(fh, f"round {r} wave {w}: rc {rc}, crossed {s.get('crossed')}, best "
                    f"{s.get('best_s')} s (greedy {s.get('greedy_s')} s)")
        row["plan"] = {"waves": waves, "wall_s": round(time.time() - t0, 1),
                       "best_s": min([x["best_s"] for x in waves if x.get("best_s")],
                                     default=None)}
        # the round's fastest search line, kept for the viewer (beam_best.jsonl is its open-loop
        # replay, bit-exact by beam_tas' own assert)
        fast = [x for x in waves if x.get("best_s")]
        if fast:
            bw = min(fast, key=lambda x: x["best_s"])
            src = rdir / f"wave_{bw['wave']}" / "beam_best.jsonl"
            if src.exists():
                shutil.copyfile(src, rdir / "best_line.jsonl")
                row["plan"]["best_line"] = str(rdir / "best_line.jsonl")
        if (args.target_best is not None and row["plan"]["best_s"] is not None
                and row["plan"]["best_s"] <= args.target_best):
            # the user, 2026-10-23 (sidistic): "if our search finds it, it's enough"
            row["target_reached"] = {"by": "planner", "run_s": row["plan"]["best_s"]}
            log(fh, f"round {r}: TARGET REACHED by the planner - a search line of "
                    f"{row['plan']['best_s']:.2f} s <= {args.target_best:g} s -> "
                    f"{rdir / 'best_line.jsonl'}")
            with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            break
        if not npzs:
            log(fh, f"round {r}: no wave crossed - stopping")
            row["stopped"] = "no crossing"
            with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            break
        # 2. DISTIL ----------------------------------------------------------------------------
        bc, spine = rdir / "bc.npz", rdir / "spine.npy"
        if args.spawn_per_wave and len(npzs) > 1:
            # plan_to_bc replays every line of ONE file from one spawn: one file per wave's spawn,
            # then merged (surfgym.bc.merge_bc_files); the spines are concatenated (the fixed
            # full window draws spawns uniformly over all of them)
            import numpy as np
            sys.path.insert(0, str(ROOT / "python"))
            from surfgym.bc import merge_bc_files
            parts, spines, skeys, rc = [], [], [], 0
            for i, z in enumerate(npzs):
                bi, si = rdir / f"bc_{i}.npz", rdir / f"spine_{i}.npy"
                rc = run([PY, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan", z, "--ckpt", cur,
                          "--out", bi, "--spine", si, "--map", args.map, *distil_extra],
                         rdir / f"distil_{i}.log", timeout=3600)
                if rc != 0 or not bi.exists():
                    break
                parts.append(bi)
                spines.append(np.load(si))
                ki = si.with_suffix(".keys.npy")
                skeys.append(np.load(ki) if ki.exists() else None)
            if rc == 0 and parts:
                merge_bc_files(parts, bc)
                np.save(spine, np.concatenate(spines))
                if skeys and all(k is not None for k in skeys):
                    # the spines' held keys, concatenated in the same order
                    np.save(spine.with_suffix(".keys.npy"), np.concatenate(skeys))
        else:
            rc = run([PY, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan", *npzs, "--ckpt", cur,
                      "--out", bc, "--spine", spine, "--map", args.map, *distil_extra],
                     rdir / "distil.log", timeout=3600)
        if rc != 0 or not bc.exists():
            log(fh, f"round {r}: distil failed (rc {rc}) - see {rdir / 'distil.log'}")
            row["stopped"] = "distil failed"
            with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            break
        import numpy as np
        bc_train, spine_train = bc, spine
        if args.bc_history > 0:
            hist = [out / f"round_{k}" for k in range(r - 1, r - 1 - args.bc_history, -1)]
            hist = [h for h in hist if (h / "bc.npz").exists() and (h / "spine.npy").exists()]
            if hist:
                sys.path.insert(0, str(ROOT / "python"))
                from surfgym.bc import merge_bc_files
                bc_train, spine_train = rdir / "bc_pool.npz", rdir / "spine_pool.npy"
                pmeta = merge_bc_files([bc] + [h / "bc.npz" for h in hist], bc_train)
                np.save(spine_train, np.concatenate([np.load(spine)]
                                                    + [np.load(h / "spine.npy") for h in hist]))
                _kf = [spine.with_suffix(".keys.npy")] + [h / "spine.keys.npy" for h in hist]
                if all(k.exists() for k in _kf):
                    np.save(spine_train.with_suffix(".keys.npy"),
                            np.concatenate([np.load(k) for k in _kf]))
                row["bc_pool"] = [h.name for h in hist]
                log(fh, f"round {r}: BC pool = this round + {[h.name for h in hist]} "
                        f"({pmeta.get('lines')} lines)")
        spine_len = int(np.load(spine_train).shape[0])
        # --demo-keys auto: the spine's held keys, when plan_to_bc wrote them (keys-hold)
        keys_train = spine_train.with_suffix(".keys.npy")
        demo_keys = (["--demo-keys", keys_train]
                     if args.demo_keys == "auto" and keys_train.exists() else [])
        # 3. TRAIN -----------------------------------------------------------------------------
        t0 = time.time()
        run_name = f"{out.name}_r{r}"
        shutil.rmtree(ROOT / "runs" / run_name, ignore_errors=True)
        steps = ckpt_step(cur) + int(float(args.train_steps))
        rc = run([PY, "-u", ROOT / "python" / "train_fast.py", "--map", args.map,
                  "--ckpt", cur, "--run", run_name, "--steps", steps, "--envs", args.envs,
                  "--no-eval-at-start", "--record-every", "1e12", "--ckpt-every", "1e12",
                  "--lr", lr_now,
                  "--bc-file", bc_train, "--bc-coef", args.bc_coef, "--bc-coef-final", 0,
                  "--bc-steps", args.train_steps, "--bc-target", args.bc_target,
                  "--bc-value-coef", args.bc_value_coef,
                  "--demo-file", spine_train, "--demo-window", spine_len, "--demo-rate", "2.0",
                  "--demo-min-ep", "1e9", "--demo-grow", "0", *demo_keys, *args.extra],
                 rdir / "train.log", env_extra={"SELF_STATES": "1"}, timeout=4 * 3600)
        final = ROOT / "runs" / run_name / "ckpt_final.pt"
        if bc_train != bc:
            bc_train.unlink(missing_ok=True)        # derivable from the rounds' own files
        row["train"] = {"run": run_name, "rc": rc, "wall_s": round(time.time() - t0, 1)}
        bl = ROOT / "runs" / run_name / "bc_log.csv"
        if bl.exists():
            lines = bl.read_text(encoding="utf-8").strip().splitlines()
            if len(lines) > 2:
                row["train"]["bc_first"], row["train"]["bc_last"] = lines[1], lines[-1]
        if rc != 0 or not final.exists():
            log(fh, f"round {r}: train failed (rc {rc}) - see {rdir / 'train.log'}")
            row["stopped"] = "train failed"
            with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            break
        # 4. EVAL ------------------------------------------------------------------------------
        ev = rdir / "eval.jsonl"
        ev_seed = ["--seed", 7000 + r] if args.gate_rematch else []
        run([PY, "-u", ROOT / "tools" / "record_ckpt.py", final, "--map", args.map,
             "--episodes", args.eval_eps, "--ep-ticks", args.max_ticks, "--out", ev, *ev_seed],
            rdir / "eval.log", timeout=3600)
        # --gate-evals N: N - 1 more evals on their own shared seeds (gate_seed)
        evs = gate_files(rdir, args.gate_evals)
        for k, e in enumerate(evs[1:], start=1):
            run([PY, "-u", ROOT / "tools" / "record_ckpt.py", final, "--map", args.map,
                 "--episodes", args.eval_eps, "--ep-ticks", args.max_ticks, "--out", e,
                 "--seed", gate_seed(r, k)], rdir / f"eval_{k}.log", timeout=3600)
        fins, n = [], 0
        for e in evs:
            _f, _n = finish_times(e) if e.exists() else ([], 0)
            fins, n = fins + _f, n + _n
        row["eval"] = {"finishes": len(fins), "episodes": n,
                       "best_s": round(min(fins), 3) if fins else None,
                       "mean_s": round(sum(fins) / len(fins), 3) if fins else None}
        log(fh, f"round {r}: planner best {row['plan']['best_s']} s; policy after training: "
                f"{len(fins)}/{n} finish, best {row['eval']['best_s']} s, mean "
                f"{row['eval']['mean_s']} s (train {row['train']['wall_s']:.0f} s)")
        if fins and min(fins) < best_s:
            best_s, best_ck = min(fins), out / "best.pt"
            shutil.copyfile(final, best_ck)
            row["new_best"] = True
            log(fh, f"round {r}: NEW BEST policy {best_s:.2f} s -> {best_ck}")
        if args.target_best is not None and fins and min(fins) <= args.target_best:
            target_hit = True
            shutil.copyfile(final, out / "target_run.pt")
            row["target_reached"] = {"by": "challenger", "run_s": round(min(fins), 3)}
            log(fh, f"round {r}: TARGET REACHED - a challenger run of {min(fins):.2f} s <= "
                    f"{args.target_best:g} s -> {out / 'target_run.pt'}")
        nxt = final
        if args.gate:
            mean = (sum(fins) / len(fins)) if (fins and len(fins) == n) else float("inf")
            bar = best_mean
            if args.gate_rematch:
                evi = rdir / "eval_incumbent.jsonl"
                run([PY, "-u", ROOT / "tools" / "record_ckpt.py", best_mean_ck, "--map", args.map,
                     "--episodes", args.eval_eps, "--ep-ticks", args.max_ticks, "--out", evi,
                     *ev_seed], rdir / "eval_incumbent.log", timeout=3600)
                evis = gate_files(rdir, args.gate_evals, incumbent=True)
                for k, e in enumerate(evis[1:], start=1):
                    run([PY, "-u", ROOT / "tools" / "record_ckpt.py", best_mean_ck, "--map",
                         args.map, "--episodes", args.eval_eps, "--ep-ticks", args.max_ticks,
                         "--out", e, "--seed", gate_seed(r, k)],
                        rdir / f"eval_incumbent_{k}.log", timeout=3600)
                fi, ni = [], 0
                for e in evis:
                    _f, _n = finish_times(e) if e.exists() else ([], 0)
                    fi, ni = fi + _f, ni + _n
                bar = (sum(fi) / len(fi)) if (fi and len(fi) == ni) else float("inf")
                row["eval_incumbent"] = {"ckpt": str(best_mean_ck), "finishes": len(fi),
                                         "episodes": ni, "mean_s": round(bar, 3) if fi else None}
                log(fh, f"round {r}: incumbent on the same spawns: {len(fi)}/{ni} finish, mean "
                        f"{bar:.3f} s")
                if (args.target_best is not None and not target_hit and fi
                        and min(fi) <= args.target_best):
                    target_hit = True
                    shutil.copyfile(best_mean_ck, out / "target_run.pt")
                    row["target_reached"] = {"by": "incumbent", "run_s": round(min(fi), 3)}
                    log(fh, f"round {r}: TARGET REACHED - an incumbent run of {min(fi):.2f} s <= "
                            f"{args.target_best:g} s -> {out / 'target_run.pt'}")
            accept = mean < bar
            if args.gate_rematch:
                # the PAIRED rule (skate_laby skEXIT3 round 6, 2026-10-04): one incumbent crash
                # set its mean to inf and handed the gate to a challenger 0.33 s SLOWER on the 8
                # spawns both finished. Compare on the spawns both finished; the challenger must
                # still finish all of its own.
                ci = [t for e in evs for t in (episode_times(e) if e.exists() else [])]
                ii = [t for e in evis for t in (episode_times(e) if e.exists() else [])]
                pairs = [(a, b) for a, b in zip(ci, ii) if a is not None and b is not None]
                paired = (sum(a - b for a, b in pairs) / len(pairs)) if pairs else None
                row["paired_s"] = round(paired, 3) if paired is not None else None
                accept = mean != float("inf") and (paired < 0.0 if pairs else True)
                log(fh, f"round {r}: paired on {len(pairs)} shared finishes: "
                        f"{'n/a' if paired is None else f'{paired:+.3f} s'} (challenger - incumbent)")
                if args.gate_rule == "paired":
                    # speed first with a reliability floor: faster on the shared finishes, and no
                    # more than --gate-finish-tol of the episodes fewer finishes
                    tol = max(1, int(round(args.gate_finish_tol * max(n, ni))))
                    accept = bool(pairs) and paired < 0.0 and len(fins) >= len(fi) - tol
                    row["paired_rule"] = {"tol": tol, "finishes": [len(fins), len(fi)]}
                if args.gate_rule == "expected":
                    # pace and reliability at one exchange rate: one more finish is worth the
                    # DNF charge minus its time (--gate-dnf-s)
                    dnf = (args.gate_dnf_s if args.gate_dnf_s is not None
                           else args.max_ticks * eval_tick_ms(evs[0]) / 1000.0)
                    ec, ei, accept = expected_gate(ci, ii, dnf)
                    row["expected"] = {"challenger": None if ec is None else round(ec, 3),
                                       "incumbent": None if ei is None else round(ei, 3),
                                       "dnf_s": round(dnf, 3), "spawns": min(len(ci), len(ii)),
                                       "finishes": [len(fins), len(fi)]}
                    log(fh, f"round {r}: expected time on {min(len(ci), len(ii))} shared spawns "
                            f"(DNF = {dnf:.2f} s): {row['expected']['challenger']} s "
                            f"(challenger) vs {row['expected']['incumbent']} s (incumbent)")
                if args.gate_rule == "score":
                    # the grader on the same spawns: more finishes first, then the faster best
                    # finish, then the deeper best progress (sidistic 2026-10-23: the seed
                    # finishes 0/9 greedy, so the mean rule above could never accept)
                    if args.gate_evals > 1:
                        # --gate-evals: the mean grader score over the N shared-seed evals (a
                        # file the grader cannot read makes that side's score None)
                        def _mean_grade(files):
                            g = [grade(e) for e in files]
                            return (None if any(x is None for x in g)
                                    else round(sum(g) / len(g), 3))
                        sc_c, sc_i = _mean_grade(evs), _mean_grade(evis)
                        row["score_evals"] = {"challenger": [grade(e) for e in evs],
                                              "incumbent": [grade(e) for e in evis]}
                    else:
                        sc_c, sc_i = grade(ev), grade(evi)
                    row["score"] = {"challenger": sc_c, "incumbent": sc_i}
                    accept = sc_c is not None and (sc_i is None or sc_c > sc_i)
                    log(fh, f"round {r}: grader {sc_c} (challenger) vs {sc_i} (incumbent) "
                            "on the same spawns")
            if accept:
                best_mean, best_mean_ck = mean, out / "best_mean.pt"
                shutil.copyfile(final, best_mean_ck)
                row["gate"] = "accepted"
                rejects_in_row = 0
                why = (f"grader {row['score']['challenger']} vs {row['score']['incumbent']}, "
                       f"{len(fins)}/{n} finished" if args.gate_rule == "score"
                       else f"paired {row.get('paired_s')} s, {len(fins)}/{n} vs {len(fi)}/{ni} "
                            f"finished" if args.gate_rule == "paired"
                       else f"expected {row['expected']['challenger']} s vs "
                            f"{row['expected']['incumbent']} s, {len(fins)}/{n} vs {len(fi)}/{ni} "
                            f"finished" if args.gate_rule == "expected"
                       else f"mean {mean:.2f} s vs {bar:.2f} s, all {n} finished")
                log(fh, f"round {r}: gate ACCEPTED ({why}) -> {best_mean_ck}")
                if args.target_mean is not None and mean <= args.target_mean:
                    target_hit = True
                    row["target_reached"] = True
                    log(fh, f"round {r}: TARGET REACHED (mean {mean:.2f} s <= "
                            f"{args.target_mean:g} s) - stopping")
            else:
                nxt = best_mean_ck
                row["gate"] = "rejected"
                rejects_in_row += 1
                why = (f"grader {row['score']['challenger']} vs {row['score']['incumbent']}"
                       if args.gate_rule == "score"
                       else f"paired {row.get('paired_s')} s, {len(fins)}/{n} vs {len(fi)}/{ni} "
                            f"finished" if args.gate_rule == "paired"
                       else f"expected {row['expected']['challenger']} s vs "
                            f"{row['expected']['incumbent']} s, {len(fins)}/{n} vs {len(fi)}/{ni} "
                            f"finished" if args.gate_rule == "expected"
                       else f"mean {mean:.2f} s vs {bar:.2f} s")
                log(fh, f"round {r}: gate rejected ({why}) - the next round plans from "
                        f"{best_mean_ck}")
                if (args.lr_halve_after and rejects_in_row >= args.lr_halve_after
                        and lr_now > args.lr_min):
                    lr_now, rejects_in_row = max(args.lr_min, lr_now / 2.0), 0
                    log(fh, f"round {r}: {args.lr_halve_after} rejections in a row - lr -> "
                            f"{lr_now:g}")
        row["lr"] = lr_now
        with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        cur = nxt
        if target_hit:
            break
    log(fh, f"done: best policy run {best_s if best_ck else None} s ({best_ck}); best mean "
            f"{best_mean:.2f} s ({best_mean_ck})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
