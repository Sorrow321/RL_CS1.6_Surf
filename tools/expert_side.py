"""expert_side.py - the expert loop's PLAN + DISTIL halves BESIDE one training run (uf2r2b3).

tools/exit_local.py runs search -> distil -> train -> eval as separate processes, so each round
is a fresh trainer launch with its evals off. On an autoresearch box the grader counts only the
TRAINER's own greedy evals (runs/<run>/traj_<step>.jsonl), and every relaunch costs the record
gate and a torch.compile. This driver keeps ONE trainer running (train_fast --bc-file F
--bc-reload --spawn-states S) and, every round, beside it:

  1. SNAPSHOT  runs/<run>/ckpt_latest.pt -> <out>/round_<r>/ckpt.pt (the policy as it is now);
  2. PLAN      tools/beam_tas.py waves from the TRUE start on that snapshot (the policy's own
               proposals, the leaders by goal-field d cloned every --resample decisions, a
               --greedy-envs floor, the --keep fastest distinct finishers kept; each wave on its
               own spawn draw, so the lines cover the start jitter the evals see);
  3. DISTIL    tools/plan_to_bc.py over every crossing wave -> one BC file (merged) and the
               fastest wave's per-tick line;
  4. HAND ON   os.replace of F (the trainer re-reads it: --bc-reload) and of S (the trainer
               re-reads a --spawn-states file whose mtime moved), so the run distils the
               planner's newest lines and spawns along the fastest one.

A round whose planner does not finish leaves both files as they were. Nothing here is read off a
map, a record or a demo: the planner proposes from the checkpoint's own distribution, so F and S
are the policy's own (CLAUDE.md section 0: declare them with SELF_STATES=1 / box.py launch
--self-states). One JSON line per round in <out>/summary.jsonl.

    nohup python3 -u tools/expert_side.py --run uf2r2b3_x --bc runs/uf2r2b3_x_side/bc.npz \\
        --states runs/uf2r2b3_x_side/line_states.npy --out runs/uf2r2b3_x_side --pid <trainer pid> \\
        > runs/uf2r2b3_x_side.txt 2>&1 &

--once runs a single round without a trainer (the files the launch needs before it starts).
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


def run(cmd, log_path: Path, timeout=None) -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        try:
            p = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                               cwd=str(ROOT), env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            f.write(f"\n!! timeout after {timeout} s\n")
            return 124
    return p.returncode


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def ckpt_step(path: Path) -> int:
    import torch
    return int(torch.load(path, map_location="cpu", weights_only=False)["global_step"])


def snapshot(src: Path, dst: Path) -> bool:
    """Copy a checkpoint the trainer may be rewriting: copy, then accept only a copy that loads
    (the trainer writes ckpt_latest through a temp file, but a copy can still race it)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        try:
            shutil.copyfile(src, dst)
            ckpt_step(dst)
            return True
        except Exception:                            # noqa: BLE001 - retry a torn copy
            time.sleep(5)
    return False


def replace_atomic(src: Path, dst: Path):
    """dst := src by one rename on the same filesystem (a reader sees the old or the new file,
    never a partial one)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp" + dst.suffix)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def one_round(a, r: int, ckpt: Path, rdir: Path) -> dict:
    row = {"round": r, "t": time.time(), "ckpt_step": ckpt_step(ckpt)}
    npzs, waves = [], []
    t0 = time.time()
    for w in range(a.waves):
        wdir = rdir / f"wave_{w}"
        cmd = [PY, "-u", ROOT / "tools" / "beam_tas.py", ckpt, "--map", a.map,
               "--envs", a.envs, "--greedy-envs", a.greedy_envs, "--score", a.score,
               "--resample-every", a.resample, "--greedy-eps", 1, "--max-ticks", a.max_ticks,
               "--keep-finishers", a.keep, "--route-file", "none",
               "--torch-seed", 1000 * r + w + a.seed_base, "--seed", 1 + 100 * r + w + a.seed_base,
               "--out-dir", wdir, *a.plan_args.split()]
        rc = run(cmd, rdir / f"wave_{w}.log", timeout=a.wave_timeout)
        s = {}
        if (wdir / "summary.json").exists():
            s = json.loads((wdir / "summary.json").read_text(encoding="utf-8"))
        waves.append({"wave": w, "rc": rc, "crossed": bool(s.get("crossed")),
                      "best_s": s.get("best_s"), "greedy_s": s.get("greedy_s")})
        if rc == 0 and s.get("crossed") and (wdir / "beam_best.npz").exists():
            npzs.append((s.get("best_s") or 1e9, wdir / "beam_best.npz"))
        log(f"round {r} wave {w}: rc {rc} crossed {s.get('crossed')} best {s.get('best_s')} s "
            f"(greedy {s.get('greedy_s')} s)")
    row["plan"] = {"waves": waves, "wall_s": round(time.time() - t0, 1)}
    if not npzs:
        row["handed_on"] = False
        return row
    npzs.sort(key=lambda x: x[0])
    parts, best_spine = [], None
    sys.path.insert(0, str(ROOT / "python"))
    from surfgym.bc import merge_bc_files
    for i, (best_s, z) in enumerate(npzs):
        bi, si = rdir / f"bc_{i}.npz", rdir / f"spine_{i}.npy"
        rc = run([PY, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan", z, "--ckpt", ckpt,
                  "--out", bi, "--spine", si, "--map", a.map, *a.distil_args.split()],
                 rdir / f"distil_{i}.log", timeout=1800)
        if rc != 0 or not bi.exists():
            log(f"round {r}: distil {i} failed (rc {rc})")
            continue
        parts.append(bi)
        if best_spine is None and si.exists():
            best_spine = si
    if not parts:
        row["handed_on"] = False
        return row
    bc = rdir / "bc.npz"
    meta = merge_bc_files(parts, bc) if len(parts) > 1 else None
    if len(parts) == 1:
        shutil.copyfile(parts[0], bc)
    replace_atomic(bc, Path(a.bc))
    if best_spine is not None and a.states:
        replace_atomic(best_spine, Path(a.states))
    row.update(handed_on=True, best_s=float(npzs[0][0]), waves_distilled=len(parts),
               lines=(int(meta.get("lines", 0)) if isinstance(meta, dict) else None))
    log(f"round {r}: handed on {len(parts)} wave(s), best line {npzs[0][0]} s -> {a.bc}"
        + (f", {a.states}" if a.states else ""))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=None, help="the trainer's run (runs/<run>/ckpt_latest.pt)")
    ap.add_argument("--ckpt", default=None, help="--once: plan on this checkpoint instead")
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_unitfarmer2.bsp"))
    ap.add_argument("--bc", required=True, help="the trainer's --bc-file (replaced atomically)")
    ap.add_argument("--states", default=None,
                    help="the trainer's --spawn-states file (the fastest line's states)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pid", type=int, default=0, help="stop when this trainer pid is gone")
    ap.add_argument("--once", action="store_true", help="one round, then exit")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--every-min", type=float, default=12.0,
                    help="minimum wall minutes between round starts")
    ap.add_argument("--min-step-gap", type=float, default=3e7,
                    help="plan again only when ckpt_latest moved this many env steps")
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--greedy-envs", type=int, default=128)
    ap.add_argument("--keep", type=int, default=16)
    ap.add_argument("--resample", type=int, default=25)
    ap.add_argument("--score", default="d")
    ap.add_argument("--max-ticks", type=int, default=7000)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--start-step", type=float, default=-1.0,
                    help="the step the files were last planned at (a --once round before the "
                         "launch): the loop plans again only --min-step-gap past it")
    ap.add_argument("--wave-timeout", type=float, default=1200.0)
    ap.add_argument("--plan-args", default="", help="extra beam_tas flags, one string")
    ap.add_argument("--distil-args", default="--no-value-target",
                    help="extra plan_to_bc flags, one string")
    ap.add_argument("--deadline", type=float, default=0.0, help="unix time to stop at (0 = none)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    last_step = int(a.start_step)
    # a loop started after a --once round into the same --out continues its numbering
    r = sum(1 for d in out.glob("round_*") if d.is_dir())
    r_end = r + a.rounds
    while r < r_end:
        if a.deadline and time.time() > a.deadline:
            log("deadline reached")
            break
        if not alive(a.pid):
            log(f"trainer pid {a.pid} gone: stop")
            break
        t_start = time.time()
        rdir = out / f"round_{r}"
        src = Path(a.ckpt) if a.ckpt else ROOT / "runs" / a.run / "ckpt_latest.pt"
        if not src.exists():
            time.sleep(30)
            continue
        ck = rdir / "ckpt.pt"
        if not snapshot(src, ck):
            log(f"round {r}: could not copy {src}")
            time.sleep(60)
            continue
        st = ckpt_step(ck)
        if not a.once and last_step >= 0 and st - last_step < a.min_step_gap:
            shutil.rmtree(rdir, ignore_errors=True)
            time.sleep(60)
            continue
        log(f"round {r}: planning on {src} at step {st:,}")
        row = one_round(a, r, ck, rdir)
        with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        last_step = st
        try:
            ck.unlink()                              # 60 MB per round; the run keeps its own
        except OSError:
            pass
        r += 1
        if a.once:
            break
        wait = a.every_min * 60.0 - (time.time() - t_start)
        while wait > 0 and alive(a.pid):
            time.sleep(min(30.0, wait))
            wait -= 30.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
