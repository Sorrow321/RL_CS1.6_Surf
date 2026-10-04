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

The next round plans from the round's checkpoint (always - the AlphaZero order); the best eval
overall is copied to <out>/best.pt. One JSON line per round in <out>/summary.jsonl.

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
    ap.add_argument("--max-ticks", type=int, default=23478)
    ap.add_argument("--train-steps", type=float, default=3e8)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bc-coef", type=float, default=0.5)
    ap.add_argument("--bc-value-coef", type=float, default=0.25)
    ap.add_argument("--bc-target", default="dist")
    ap.add_argument("--eval-eps", type=int, default=5)
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
    log(fh, f"seed {cur} (step {ckpt_step(cur):,}); deadline in {args.deadline_h:g} h")
    for r in range(args.rounds):
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
                      "--torch-seed", 1000 * r + w, "--seed", 0, "--out-dir", wdir],
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
        spine_len = int(np.load(spine).shape[0])
        # 3. TRAIN -----------------------------------------------------------------------------
        t0 = time.time()
        run_name = f"{out.name}_r{r}"
        shutil.rmtree(ROOT / "runs" / run_name, ignore_errors=True)
        steps = ckpt_step(cur) + int(float(args.train_steps))
        rc = run([PY, "-u", ROOT / "python" / "train_fast.py", "--map", args.map,
                  "--ckpt", cur, "--run", run_name, "--steps", steps, "--envs", args.envs,
                  "--no-eval-at-start", "--record-every", "1e12", "--ckpt-every", "1e12",
                  "--lr", args.lr,
                  "--bc-file", bc, "--bc-coef", args.bc_coef, "--bc-coef-final", 0,
                  "--bc-steps", args.train_steps, "--bc-target", args.bc_target,
                  "--bc-value-coef", args.bc_value_coef,
                  "--demo-file", spine, "--demo-window", spine_len, "--demo-rate", "2.0",
                  "--demo-min-ep", "1e9", "--demo-grow", "0", *args.extra],
                 rdir / "train.log", env_extra={"SELF_STATES": "1"}, timeout=4 * 3600)
        final = ROOT / "runs" / run_name / "ckpt_final.pt"
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
        run([PY, "-u", ROOT / "tools" / "record_ckpt.py", final, "--map", args.map,
             "--episodes", args.eval_eps, "--ep-ticks", args.max_ticks, "--out", ev],
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
        with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        cur = final
    log(fh, f"done: best policy {best_s if best_ck else None} s ({best_ck})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
