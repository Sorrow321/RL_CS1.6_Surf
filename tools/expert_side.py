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


def wave_cmd(a, ckpt: Path, r: int, w: int, wdir: Path, extra=()):
    return [PY, "-u", ROOT / "tools" / "beam_tas.py", ckpt, "--map", a.map,
            "--envs", a.envs, "--greedy-envs", a.greedy_envs, "--score", a.score,
            "--resample-every", a.resample, "--greedy-eps", 1, "--max-ticks", a.max_ticks,
            "--keep-finishers", a.keep, "--route-file", "none", "--save-frontier",
            "--torch-seed", 1000 * r + w + a.seed_base, "--seed", 1 + 100 * r + w + a.seed_base,
            "--out-dir", wdir, *a.plan_args.split(), *[str(x) for x in extra]]


def run_wave(a, ckpt: Path, r: int, w: int, rdir: Path, extra=(), tag="plain") -> dict:
    wdir = rdir / f"wave_{w}"
    rc = run(wave_cmd(a, ckpt, r, w, wdir, extra), rdir / f"wave_{w}.log",
             timeout=a.wave_timeout)
    s = {}
    if (wdir / "summary.json").exists():
        s = json.loads((wdir / "summary.json").read_text(encoding="utf-8"))
    out = {"wave": w, "kind": tag, "rc": rc, "crossed": bool(s.get("crossed")),
           "best_s": s.get("best_s"), "greedy_s": s.get("greedy_s"),
           "npz": str(wdir / "beam_best.npz"), "frontier": str(wdir / "beam_frontier.npz")}
    if tag != "plain":
        out["branch"] = list(extra)
    log(f"round {r} wave {w} ({tag}): rc {rc} crossed {s.get('crossed')} best "
        f"{s.get('best_s')} s (greedy {s.get('greedy_s')} s)")
    return out


def line_plateau_fork(a, frontier: Path, rng) -> tuple:
    """The fork tick for a branched wave: inside the first half of the LONGEST record plateau of
    a finishing line (its per-tick states, beam_tas --save-frontier) on --field, drawn uniformly.
    -> (tick, (start, end)) or (None, None) when the line has no plateau of --branch-secs."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "python"))
    from surfgym.goalfield import GoalField
    from surfgym.greedy_spawn import record_plateaus
    z = np.load(frontier, allow_pickle=False)
    st = z["states"]
    zf = np.load(a.field, allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                  float(zf["cell"]), float(zf["reach_max"]))
    d = np.asarray(f.sample(st["origin"].astype(np.float64)), np.float64)
    d = np.where(d < f.reach_max - 0.5 * f.cell, d, np.inf)
    tick_ms = float(json.loads((frontier.parent / "summary.json").read_text())["tick_ms"])
    pl = record_plateaus(d, 1000.0 / tick_ms, a.branch_secs)
    if not pl:
        return None, None
    s0, e0 = max(pl, key=lambda x: x[1] - x[0])
    return int(s0 + rng.uniform(0.0, 0.5) * (e0 - s0)), (int(s0), int(e0))


def one_round(a, r: int, ckpt: Path, rdir: Path) -> dict:
    import numpy as np
    row = {"round": r, "t": time.time(), "ckpt_step": ckpt_step(ckpt)}
    waves = []
    t0 = time.time()
    for w in range(a.waves):
        waves.append(run_wave(a, ckpt, r, w, rdir))
    # --branch-waves: forks at the fastest plain line's longest record plateau (its own time
    # sink: an extra lap, a loop), half the population holding yaw offsets around the policy's
    # own target for --branch-hold ticks - a line the policy's narrow proposals never emit
    plain = [x for x in waves if x["rc"] == 0 and x["crossed"] and x["best_s"]]
    if a.branch_waves > 0 and plain:
        fast = min(plain, key=lambda x: x["best_s"])
        rng = np.random.default_rng(7919 * r + a.seed_base)
        for k in range(a.branch_waves):
            try:
                tick, pl = line_plateau_fork(a, Path(fast["frontier"]), rng)
            except Exception as e:                   # noqa: BLE001 - a missing frontier file
                log(f"round {r}: no fork tick ({e!r})")
                break
            if tick is None:
                log(f"round {r}: the fastest line has no record plateau >= {a.branch_secs:g} s")
                break
            extra = ["--branch-at", f"t{tick}:{int(a.envs) // 2}", "--branch-view-only",
                     "--branch-jitter", int(a.branch_jitter), "--branch-hold", int(a.branch_hold),
                     "--branch-burst", int(a.branch_burst), "--branch-protect",
                     int(a.branch_protect)]
            x = run_wave(a, ckpt, r, a.waves + k, rdir, extra, tag="branch")
            x["plateau"] = pl
            waves.append(x)
    row["plan"] = {"waves": waves, "wall_s": round(time.time() - t0, 1)}
    ok = [x for x in waves if x["rc"] == 0 and x["crossed"] and x["best_s"]
          and Path(x["npz"]).exists()]
    if not ok:
        row["handed_on"] = False
        return row
    ok.sort(key=lambda x: x["best_s"])
    best = float(ok[0]["best_s"])
    # --wave-tol: distil only the waves within this fraction of the round's fastest line, so a
    # much faster line (one found by a fork) is not diluted by the old route's waves
    keep = [x for x in ok if x["best_s"] <= best * (1.0 + a.wave_tol)]
    parts, best_spine = [], None
    sys.path.insert(0, str(ROOT / "python"))
    from surfgym.bc import merge_bc_files
    for i, x in enumerate(keep):
        bi, si = rdir / f"bc_{i}.npz", rdir / f"spine_{i}.npy"
        rc = run([PY, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan", x["npz"], "--ckpt", ckpt,
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
        ks = best_spine.with_suffix(".keys.npy")
        if ks.exists():
            replace_atomic(ks, Path(a.states).with_suffix(".keys.npy"))
    row.update(handed_on=True, best_s=best, best_kind=ok[0]["kind"],
               waves_distilled=len(parts),
               lines=(int(meta.get("lines", 0)) if isinstance(meta, dict) else None))
    log(f"round {r}: handed on {len(parts)} wave(s), best line {best} s ({ok[0]['kind']}) -> "
        f"{a.bc}" + (f", {a.states}" if a.states else ""))
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
    ap.add_argument("--field", default=None,
                    help="the goal field for the record plateaus (default: the map's "
                         "no-corner-cut field <map>.goalc_32.npz, else <map>.goal_32.npz)")
    ap.add_argument("--branch-waves", type=int, default=0,
                    help="per round, this many extra waves forked at the fastest plain line's "
                         "longest record plateau (0 = off)")
    ap.add_argument("--branch-secs", type=float, default=1.5,
                    help="--branch-waves: the shortest record plateau that counts (s)")
    ap.add_argument("--branch-jitter", type=int, default=30,
                    help="--branch-waves: the held yaw offset U(-J, J) deg on the policy's target")
    ap.add_argument("--branch-hold", type=int, default=50, help="--branch-waves: ticks per offset")
    ap.add_argument("--branch-burst", type=int, default=300, help="--branch-waves: burst ticks")
    ap.add_argument("--branch-protect", type=int, default=4,
                    help="--branch-waves: generations the forked lineages cannot be culled")
    ap.add_argument("--wave-tol", type=float, default=0.02,
                    help="distil only the waves whose fastest line is within this fraction of "
                         "the round's fastest")
    ap.add_argument("--start-step", type=float, default=-1.0,
                    help="the step the files were last planned at (a --once round before the "
                         "launch): the loop plans again only --min-step-gap past it")
    ap.add_argument("--wave-timeout", type=float, default=1200.0)
    ap.add_argument("--plan-args", default="", help="extra beam_tas flags, one string")
    ap.add_argument("--distil-args", default="--no-value-target",
                    help="extra plan_to_bc flags, one string")
    ap.add_argument("--deadline", type=float, default=0.0, help="unix time to stop at (0 = none)")
    a = ap.parse_args(argv)
    if a.field is None:
        mp = Path(a.map)
        for cand in (mp.with_suffix(".goalc_32.npz"), mp.with_suffix(".goal_32.npz")):
            if cand.exists():
                a.field = str(cand)
                break
    if a.branch_waves > 0 and not a.field:
        raise SystemExit("--branch-waves needs a goal field (--field)")
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
