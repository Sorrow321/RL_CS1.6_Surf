#!/usr/bin/env python3
"""band_bench.py - how often does a checkpoint get from a band of spine states to the spine's
success line, under which sampling? A MEASUREMENT (CLAUDE.md section 0: benches never train).

For each band [a, b] of a STATE_DTYPE spine and each sampling mode, record_ckpt.py spawns the
checkpoint from a uniform draw of the band's states (N episodes) and every episode is scored on
the grader's field (autoresearch/config.json "field"): SUCCESS when its min d along the episode
reaches the success line (--d-success, or d(spine end) + --success-d like train_fast's
--demo-success-d). Prints one table: band x mode -> successes / N, the median and best progress.

    python tools/band_bench.py --ckpt runs_ckpt.pt --spine autoresearch/spines/pbr1b1_line_a.npy \
        --bands 1230:1234,1235:1239,1240:1244 --episodes 48 --success-d 100 \
        --modes greedy,stoch,keys2,keys4,view4,all2 --out-dir runs/bench

Modes: greedy (argmax), stoch (the policy's own sampling), keysT (--exec-keys-temp T),
viewS (--exec-view-scale S), yawS (--exec-view-scale S,1), allT (--exec-temp T).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "autoresearch"))


def mode_flags(m: str) -> list[str]:
    if m == "greedy":
        return []
    if m == "stoch":
        return ["--stochastic"]
    for pre, flag in (("keys", "--exec-keys-temp"), ("all", "--exec-temp")):
        if m.startswith(pre):
            return ["--stochastic", flag, m[len(pre):]]
    if m.startswith("view"):
        return ["--stochastic", "--exec-view-scale", m[4:]]
    if m.startswith("yaw"):
        return ["--stochastic", "--exec-view-scale", f"{m[3:]},1"]
    raise SystemExit(f"unknown mode {m!r}")


def spine_keys(spine_path: str, keys: str | None, n: int):
    """The held keys of a spine's rows (int32, n x 3), or None: --keys none, or no
    <spine>.keys.npy beside the spine when --keys is not given."""
    if keys is not None and str(keys).lower() == "none":
        return None
    if keys is None:
        kp = Path(str(spine_path)[:-4] + ".keys.npy" if str(spine_path).endswith(".npy")
                  else str(spine_path) + ".keys.npy")
        if not kp.exists():
            return None
        keys = str(kp)
    k = np.asarray(np.load(keys), np.int32)
    if k.shape != (n, 3):
        raise SystemExit(f"--keys {keys}: shape {k.shape}, the spine has {n} rows of 3 keys")
    return k


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spine", required=True)
    ap.add_argument("--bands", required=True, help="a:b[,a:b...] inclusive spine indices")
    ap.add_argument("--modes", default="greedy,stoch")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--ep-ticks", type=int, default=1000)
    ap.add_argument("--success-d", type=float, default=100.0,
                    help="success line = d(spine end) + this (train_fast --demo-success-d)")
    ap.add_argument("--d-success", type=float, default=None, help="an absolute success line")
    ap.add_argument("--out-dir", default="runs/bench")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keys", default=None,
                    help="the spine's held keys (one row of 3 per state; default: <spine>.keys.npy "
                         "beside it when it exists, 'none' = neutral): a band spawn holds its row's "
                         "keys (record_ckpt --spawn-keys), as train_fast --demo-keys does")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="passed to record_ckpt verbatim")
    a = ap.parse_args(argv)

    import score as grader
    f = grader.field()
    spine = np.load(a.spine)
    keys = spine_keys(a.spine, a.keys, len(spine))
    d_end = float(f.sample(spine["origin"][-1:].astype(np.float64))[0])
    line = a.d_success if a.d_success is not None else d_end + a.success_d
    d0 = float(grader.START_D)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"spine {a.spine}: {len(spine)} states, d(end) {d_end:,.0f}; success line d <= "
          f"{line:,.0f} ({100 * (d0 - line) / d0:.2f}%)"
          + ("" if keys is None else "; spawns hold the rows' keys"), flush=True)
    rows = []
    for band in a.bands.split(","):
        lo, hi = (int(x) for x in band.split(":"))
        bf = out / f"band_{lo}_{hi}.npy"
        np.save(bf, spine[lo:hi + 1])
        kf = out / f"band_{lo}_{hi}.keys.npy"
        if keys is not None:
            np.save(kf, keys[lo:hi + 1])
        for m in a.modes.split(","):
            name = f"{Path(a.ckpt).stem}_{lo}_{hi}_{m}"
            tj = out / f"{name}.jsonl"
            cmd = [sys.executable, str(ROOT / "tools" / "record_ckpt.py"), a.ckpt,
                   "--spawn-states", str(bf), "--episodes", str(a.episodes),
                   "--ep-ticks", str(a.ep_ticks), "--out", str(tj), "--seed", str(a.seed)]
            if keys is not None:
                cmd += ["--spawn-keys", str(kf)]
            cmd += mode_flags(m) + list(a.extra)
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", cwd=str(ROOT))
            if r.returncode != 0 or not tj.exists():
                tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-12:])
                print(f"== {name}: record_ckpt FAILED\n{tail}", flush=True)
                continue
            eps = grader.episodes(tj)
            dmins, ends = [], []
            for hdr, rr, tr in eps:
                d = f.sample(rr[:, 1:4]).astype(np.float64)
                ok = d < f.reach_max - 0.5 * f.cell
                dmins.append(float(d[ok].min()) if ok.any() else float("inf"))
                ends.append(tr.get("end"))
            dm = np.asarray(dmins)
            succ = int((dm <= line).sum())
            prog = 100 * (d0 - dm) / d0
            row = {"band": [lo, hi], "mode": m, "n": len(eps), "succ": succ,
                   "prog_med": round(float(np.median(prog)), 3),
                   "prog_max": round(float(prog.max()), 3),
                   "done": int(sum(e == "done" for e in ends)), "secs": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"== band [{lo},{hi}] {m:>7}: {succ:3d}/{len(eps)} pass  progress median "
                  f"{row['prog_med']:.2f}% max {row['prog_max']:.2f}%  ({row['secs']} s)",
                  flush=True)
    (out / f"band_bench_{int(time.time())}.json").write_text(json.dumps(
        {"ckpt": a.ckpt, "spine": a.spine, "line": line, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
