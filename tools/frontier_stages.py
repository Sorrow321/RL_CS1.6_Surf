"""frontier_stages.py - return-then-explore at the policy's OWN frontier, refreshed while it trains.

A --ramp-sequence run with --spawn-states FILE re-reads FILE whenever its mtime changes (every 20
iterations). This loop keeps FILE at the run's current frontier (CLAUDE.md 0b: an archive of the
policy's own states; section 0: never a human record):

  1. on every new checkpoint (and at most every --every-min): record --episodes stochastic
     episodes of it from the map start with --dump-states (tools/record_ckpt.py);
  2. the frontier is the deepest list position those episodes TOUCHED (k_max, 0-based);
  3. cut the policy's own states --leads seconds before its touches at list positions
     k_max - 1 and k_max (tools/stage_states.cut), each stage resampled to the same row count so
     the frontier is not drowned by the stage before it;
  4. write FILE atomically (tmp + os.replace), with seq_k (where each state starts the list) and
     source self:<run>.

The rule reads nothing off the map or a record - one rule, the same constants on every map. It is
the automatic form of the hand-cut uf2_stage23 / uf2_stage34 files (2026-10-07: own-state practice
before the deepest touches took uf2R3P from S19 to S20 in 9/9 greedy evals; the control did not).

    python3 tools/frontier_stages.py --run runs/uf2R3P2 --map maps_pool/surf_unitfarmer2.bsp \\
        --out runs/research/uf2_stage/uf2_stage34_r3p.npz --pid-file runs/uf2R3P2.pid
    python3 tools/frontier_stages.py --once --ckpt X.pt --map ... --out F.npz --source self:R
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from stage_states import cut, episode_touches  # noqa: E402


def frontier_stages(touches):
    """the deepest 0-based list position any episode touched and the stages to cut before:
    (k_max, [k_max - 1, k_max] clipped at 0); (None, []) when nothing was touched"""
    deep = [len(t) for t in touches if t]
    if not deep:
        return None, []
    k_max = max(deep) - 1
    return k_max, [k for k in (k_max - 1, k_max) if k >= 0]


def balance(states, ks, rng):
    """every stage resampled (with replacement) to the largest stage's row count"""
    ks = np.asarray(ks, np.int64)
    n = max(int((ks == k).sum()) for k in np.unique(ks))
    rows, out_k = [], []
    for k in np.unique(ks):
        idx = np.flatnonzero(ks == k)
        pick = idx if len(idx) == n else np.concatenate(
            [idx, rng.choice(idx, n - len(idx), replace=True)])
        rows.append(states[pick])
        out_k.append(np.full(len(pick), int(k), np.int64))
    return np.concatenate(rows), np.concatenate(out_k)


def jitter_speed(states, lo, hi, rng):
    """--vel-scale LO HI: every state's velocity x U(LO, HI) (Florensa-style start-state
    perturbation: the frontier is also met FASTER than the policy reaches it today, so its value
    can tell what more energy there is worth). 1 1 = the exact states"""
    if lo == 1.0 and hi == 1.0:
        return states
    out = states.copy()
    out["velocity"] = out["velocity"] * rng.uniform(lo, hi, len(out)).astype(np.float32)[:, None]
    return out


def build(ckpt, map_path, out, source, episodes, leads, ep_ticks, work, python, seed,
          vel_scale=(1.0, 1.0)):
    """record `ckpt`, cut its frontier, write `out` atomically -> a one-line report (or None)"""
    work.mkdir(parents=True, exist_ok=True)
    ck = work / "frontier_ckpt.pt"
    shutil.copyfile(ckpt, ck)                   # never read a checkpoint while it is rewritten
    traj, dump = work / "frontier_rec.jsonl", work / "frontier_rec_states.npz"
    for p in (traj, dump):
        if p.exists():
            p.unlink()
    cmd = [python, str(ROOT / "tools" / "record_ckpt.py"), str(ck), "--map", str(map_path),
           "--episodes", str(episodes), "--ep-ticks", str(ep_ticks), "--stochastic",
           "--dump-states", str(dump), "--out", str(traj)]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0 or not traj.exists() or not dump.exists():
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        print(f"[{time.strftime('%H:%M:%S')}] record failed rc {r.returncode}: {' | '.join(tail)}",
              flush=True)
        return None
    touches, tick_ms = episode_touches(traj)
    z = np.load(dump, allow_pickle=False)
    eps = [z[k] for k in sorted(z.files)]
    if len(eps) != len(touches):
        print(f"{len(eps)} dumped episodes for {len(touches)} recorded ones - skipped", flush=True)
        return None
    k_max, stages = frontier_stages(touches)
    hist = {}
    for t in touches:
        hist[len(t)] = hist.get(len(t), 0) + 1
    if k_max is None:
        return f"no touches in {len(touches)} episodes - {out} kept"
    st, ks, _rep = cut(touches, eps, stages, leads, tick_ms or 10.0)
    if st is None:
        return f"frontier k={k_max}: nothing to cut - {out} kept"
    rng = np.random.default_rng(seed)
    st, ks = balance(st, ks, rng)
    st = jitter_speed(st, float(vel_scale[0]), float(vel_scale[1]), rng)
    tmp = Path(str(out) + ".tmp.npz")
    np.savez(tmp, states=st, seq_k=ks, source=str(source),
             vel_scale=np.asarray(vel_scale, np.float32))
    os.replace(tmp, out)
    per = {int(k): int((ks == k).sum()) for k in np.unique(ks)}
    vs = "" if tuple(vel_scale) == (1.0, 1.0) else f", velocity x U{tuple(vel_scale)}"
    return (f"touches per episode {dict(sorted(hist.items()))} -> frontier k={k_max}, "
            f"stages {per}{vs} -> {out}")


def ckpt_step(path):
    try:
        import torch
        c = torch.load(path, map_location="cpu", weights_only=False)
        return int(c.get("step") or c.get("global_step") or -1)
    except Exception:                            # noqa: BLE001 - a step is only for the log
        return -1


def alive(pid_file):
    """is the trainer in pid_file running? NEVER os.kill(pid, 0) on Windows: there any signal but
    the two console events is TerminateProcess - the probe would kill the trainer"""
    try:
        pid = int(Path(pid_file).read_text().strip())
    except (OSError, ValueError):
        return False
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True,
                           text=True)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="the run directory (its ckpt_latest.pt is read)")
    ap.add_argument("--ckpt", default=None, help="--once: this checkpoint instead of --run's")
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True, help="the run's --spawn-states file")
    ap.add_argument("--source", default=None, help="provenance (default self:<run name>)")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--leads", type=float, nargs="+", default=[0.3, 0.5, 0.8])
    ap.add_argument("--ep-ticks", type=int, default=6000)
    ap.add_argument("--every-min", type=float, default=10.0)
    ap.add_argument("--vel-scale", type=float, nargs=2, default=[1.0, 1.0],
                    help="multiply every cut state's velocity by U(LO, HI) (1 1 = off)")
    ap.add_argument("--pid-file", default=None, help="exit when this trainer is gone")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)
    if args.ckpt is None and args.run is None:
        raise SystemExit("--run or --ckpt is needed")
    name = Path(args.run).name if args.run else Path(args.ckpt).stem
    source = args.source or f"self:{name}"
    if not source.startswith("self:"):
        raise SystemExit("--source must be self:<run> (CLAUDE.md section 0)")
    # the recorder's scratch: next to the run, or (--once without --run) next to --out
    work = (Path(args.run) if args.run else Path(args.out).parent) / "frontier_work"
    out = Path(args.out)
    if args.once:
        ck = Path(args.ckpt) if args.ckpt else Path(args.run) / "ckpt_latest.pt"
        rep = build(ck, args.map, out, source, args.episodes, args.leads, args.ep_ticks, work,
                    args.python, 0, args.vel_scale)
        print(f"[{time.strftime('%H:%M:%S')}] step {ckpt_step(ck):,}: {rep}", flush=True)
        return
    ck = Path(args.run) / "ckpt_latest.pt"
    last_mt, last_t, n = None, 0.0, 0
    print(f"[{time.strftime('%H:%M:%S')}] frontier loop on {ck} -> {out} ({args.episodes} "
          f"episodes, leads {args.leads}, at most every {args.every_min:g} min)", flush=True)
    while True:
        if args.pid_file and not alive(args.pid_file):
            print(f"[{time.strftime('%H:%M:%S')}] trainer gone - frontier loop exits", flush=True)
            return
        try:
            mt = ck.stat().st_mtime
        except OSError:
            mt = None
        if mt is not None and mt != last_mt and time.time() - last_t >= args.every_min * 60.0:
            time.sleep(20)                       # let the trainer finish writing it
            n += 1
            rep = build(ck, args.map, out, source, args.episodes, args.leads, args.ep_ticks,
                        work, args.python, n, args.vel_scale)
            print(f"[{time.strftime('%H:%M:%S')}] step {ckpt_step(work / 'frontier_ckpt.pt'):,}:"
                  f" {rep}", flush=True)
            last_mt, last_t = mt, time.time()
        time.sleep(60)


if __name__ == "__main__":
    main()
