#!/usr/bin/env python3
"""plan_waves.py - the PLAN + DISTIL half of tools/exit_local.py, without its training step
(uf2r2b4, 2026-10-31).

exit_local.py trains each round with the trainer's evals OFF (--record-every 1e12) and gates
on record_ckpt evals. The autoresearch grader counts only the TRAINER's own greedy evals
(runs/<run>/traj_<step>.jsonl), so the training half has to be an ordinary launch
(autoresearch/box.py launch --resume ... -- --bc-file ...) with evals on. This script does
the rest, with exit_local's exact commands:

  1. PLAN    --waves beam_tas.py waves from --ckpt (torch seed --seed-base + w; with
             --spawn-per-wave each wave from its own spawn draw), --greedy-envs floor,
             --keep the K fastest distinct finishers;
  2. DISTIL  plan_to_bc.py over the crossing waves -> <out>/bc.npz (the planner's decisions)
             and <out>/spine.npy (+ spine.keys.npy under --keys-hold): one file per spawn,
             merged (surfgym.bc.merge_bc_files), spines concatenated.

Every input is the POLICY'S OWN (the planner proposes only from the checkpoint's
distribution), so the BC file and the spine are SELF_STATES data (CLAUDE.md section 0) - never
point --ckpt at a demo-trained checkpoint. Nothing here names a place: the same flags run on
any map. The last stdout line is ``PLAN_WAVES {json}``.

    python tools/plan_waves.py --ckpt runs_ckpt.pt --map maps/M.bsp --out runs/beam/x1 \\
        --waves 3 --spawn-per-wave 1
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd, log_path: Path, timeout=None) -> int:
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                           cwd=str(ROOT), timeout=timeout)
    return p.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--plan-envs", type=int, default=4096)
    ap.add_argument("--greedy-envs", type=int, default=64)
    ap.add_argument("--resample", type=int, default=25)
    ap.add_argument("--score", default="d")
    ap.add_argument("--keep", type=int, default=16)
    ap.add_argument("--max-ticks", type=int, default=6500)
    ap.add_argument("--seed-base", type=int, default=0,
                    help="wave w plans with torch seed seed_base + w (and, with "
                         "--spawn-per-wave, spawn seed 1 + seed_base + w)")
    ap.add_argument("--spawn-per-wave", type=int, default=1, choices=(0, 1),
                    help="1: each wave plans from its own spawn draw (the evals' start jitter); "
                         "0: every wave from spawn seed 0")
    ap.add_argument("--plan-args", default="", help="extra beam_tas.py flags, one string")
    ap.add_argument("--distil-args", default="", help="extra plan_to_bc.py flags, one string")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    plan_extra, distil_extra = shlex.split(a.plan_args), shlex.split(a.distil_args)
    t0 = time.time()
    npzs, waves = [], []
    for w in range(a.waves):
        wdir = out / f"wave_{w}"
        seed = (1 + a.seed_base + w) if a.spawn_per_wave else 0
        rc = run([PY, "-u", ROOT / "tools" / "beam_tas.py", a.ckpt, "--map", a.map,
                  "--envs", a.plan_envs, "--greedy-envs", a.greedy_envs, "--score", a.score,
                  "--resample-every", a.resample, "--greedy-eps", 1, "--max-ticks", a.max_ticks,
                  "--keep-finishers", a.keep, "--route-file", "none",
                  "--torch-seed", a.seed_base + w, "--seed", seed, "--out-dir", wdir,
                  *plan_extra], out / f"wave_{w}.log", timeout=3600)
        s = {}
        if (wdir / "summary.json").exists():
            s = json.loads((wdir / "summary.json").read_text(encoding="utf-8"))
        waves.append({"wave": w, "rc": rc, "seed": seed, "crossed": bool(s.get("crossed")),
                      "best_s": s.get("best_s"), "greedy_s": s.get("greedy_s")})
        if rc == 0 and s.get("crossed") and (wdir / "beam_best.npz").exists():
            npzs.append(wdir / "beam_best.npz")
        print(f"wave {w}: rc {rc} seed {seed} crossed {s.get('crossed')} best "
              f"{s.get('best_s')} s (greedy {s.get('greedy_s')} s)", flush=True)
    res = {"out": str(out), "waves": waves, "plan_wall_s": round(time.time() - t0, 1)}
    if not npzs:
        res["stopped"] = "no crossing"
        print("PLAN_WAVES " + json.dumps(res))
        return 1
    bc, spine = out / "bc.npz", out / "spine.npy"
    if a.spawn_per_wave and len(npzs) > 1:
        sys.path.insert(0, str(ROOT / "python"))
        from surfgym.bc import merge_bc_files
        parts, spines, skeys, rc = [], [], [], 0
        for i, z in enumerate(npzs):
            bi, si = out / f"bc_{i}.npz", out / f"spine_{i}.npy"
            rc = run([PY, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan", z, "--ckpt", a.ckpt,
                      "--out", bi, "--spine", si, "--map", a.map, *distil_extra],
                     out / f"distil_{i}.log", timeout=3600)
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
                np.save(spine.with_suffix(".keys.npy"), np.concatenate(skeys))
    else:
        rc = run([PY, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan", *npzs, "--ckpt", a.ckpt,
                  "--out", bc, "--spine", spine, "--map", a.map, *distil_extra],
                 out / "distil.log", timeout=3600)
    if rc != 0 or not bc.exists() or not spine.exists():
        res["stopped"] = f"distil failed (rc {rc})"
        print("PLAN_WAVES " + json.dumps(res))
        return 1
    res.update({"bc": str(bc), "spine": str(spine), "spine_len": int(np.load(spine).shape[0]),
                "spine_keys": str(spine.with_suffix(".keys.npy"))
                if spine.with_suffix(".keys.npy").exists() else None,
                "best_s": min([x["best_s"] for x in waves if x.get("best_s")], default=None),
                "wall_s": round(time.time() - t0, 1)})
    (out / "plan_waves.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("PLAN_WAVES " + json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
