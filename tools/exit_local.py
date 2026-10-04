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
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra train_fast flags (everything after --extra)")
    args = ap.parse_args()

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
                      "--out-dir", wdir],
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
            parts, spines, rc = [], [], 0
            for i, z in enumerate(npzs):
                bi, si = rdir / f"bc_{i}.npz", rdir / f"spine_{i}.npy"
                rc = run([PY, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan", z, "--ckpt", cur,
                          "--out", bi, "--spine", si, "--map", args.map],
                         rdir / f"distil_{i}.log", timeout=3600)
                if rc != 0 or not bi.exists():
                    break
                parts.append(bi)
                spines.append(np.load(si))
            if rc == 0 and parts:
                merge_bc_files(parts, bc)
                np.save(spine, np.concatenate(spines))
        else:
            rc = run([PY, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan", *npzs, "--ckpt", cur,
                      "--out", bc, "--spine", spine, "--map", args.map],
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
                row["bc_pool"] = [h.name for h in hist]
                log(fh, f"round {r}: BC pool = this round + {[h.name for h in hist]} "
                        f"({pmeta.get('lines')} lines)")
        spine_len = int(np.load(spine_train).shape[0])
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
                  "--demo-min-ep", "1e9", "--demo-grow", "0", *args.extra],
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
        fins, n = finish_times(ev) if ev.exists() else ([], 0)
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
                fi, ni = finish_times(evi) if evi.exists() else ([], 0)
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
            if mean < bar:
                best_mean, best_mean_ck = mean, out / "best_mean.pt"
                shutil.copyfile(final, best_mean_ck)
                row["gate"] = "accepted"
                rejects_in_row = 0
                log(fh, f"round {r}: gate ACCEPTED (mean {mean:.2f} s vs {bar:.2f} s, all {n} "
                        f"finished) -> {best_mean_ck}")
                if args.target_mean is not None and mean <= args.target_mean:
                    target_hit = True
                    row["target_reached"] = True
                    log(fh, f"round {r}: TARGET REACHED (mean {mean:.2f} s <= "
                            f"{args.target_mean:g} s) - stopping")
            else:
                nxt = best_mean_ck
                row["gate"] = "rejected"
                rejects_in_row += 1
                log(fh, f"round {r}: gate rejected (mean {mean:.2f} s vs {bar:.2f} s)"
                        f" - the next round plans from {best_mean_ck}")
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
