#!/usr/bin/env python3
"""uf2_record_pit_bench.py - MEASUREMENT ONLY (CLAUDE.md section 0: a diagnostic bench that spawns a
policy from demo states to measure which skill it lacks; never a spawn file, a spine or a target).

WON'T or CAN'T, for the start pit's extra lap: place the policy (greedy) in the WR demo's own states
along its single clockwise pit lap and climb, and report whether it gets out of the pit (best
progress past --line %) and when, against the record's own exit from the same state. Writes the
states to a scratch directory only (one file per moment, never under the repo).

    python autoresearch/analysis/uf2_record_pit_bench.py CKPT OUT_DIR [t1,t2,...]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "autoresearch"))
import score as S  # noqa: E402
from surfgym.core import STATE_DTYPE  # noqa: E402

WR = Path("/home/steam/ar_uf2/wr/surf_unitfarmer2_wr.frames.npz")


def rec_state(z, t):
    i = int(np.argmin(np.abs(z["t"] - t)))
    st = np.zeros(1, STATE_DTYPE)
    st["origin"][0] = z["simorg"][i]
    st["velocity"][0] = z["simvel"][i]
    st["yaw"][0] = float(z["viewangles"][i][1])
    st["pitch"][0] = -float(z["viewangles"][i][0])       # simulator convention: negative = down
    st["onground"][0] = 0 if int(z["onground"][i]) else -1
    st["ducked"][0] = int((int(z["uc_buttons"][i]) & 4) != 0)
    return st, i


def main():
    ck, out = sys.argv[1], Path(sys.argv[2])
    ts = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else
                             "3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0").split(",")]
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(WR, allow_pickle=False)
    f = S.field()
    # the record's own first arrival at 5% from each moment (its clock)
    d_rec = f.sample(z["simorg"].astype(np.float64)).astype(np.float64)
    prog_rec = 100.0 * (S.START_D - d_rec) / S.START_D
    for t in ts:
        st, i = rec_state(z, t)
        p = out / f"rec_{t:.1f}.npy"
        np.save(p, st)
        hit = np.flatnonzero((prog_rec >= 5.0) & (np.arange(len(prog_rec)) > i))
        rec_exit = (z["t"][hit[0]] - z["t"][i]) if len(hit) else float("nan")
        traj = out / f"pol_{t:.1f}.jsonl"
        r = subprocess.run([sys.executable, str(REPO / "tools" / "record_ckpt.py"), ck,
                            "--episodes", "1", "--spawn-states", str(p), "--ep-ticks", "2000",
                            "--out", str(traj)], capture_output=True, text=True, cwd=str(REPO))
        if r.returncode != 0 or not traj.exists():
            print(f"t {t}: record_ckpt failed: {r.stdout[-300:]} {r.stderr[-300:]}")
            continue
        eps = S.episodes(traj)
        h, rows, tr = eps[0]
        tick = float(h.get("tick_ms", 10.0)) / 1000.0
        pos, vel = rows[:, 1:4], rows[:, 4:7]
        d = f.sample(pos).astype(np.float64)
        prog = 100.0 * (S.START_D - d) / S.START_D
        ph = np.flatnonzero(prog >= 5.0)
        E0 = float(st["origin"][0][2] + (st["velocity"][0] ** 2).sum() / 1600.0)
        print(f"rec t {t:4.1f}: pos {np.round(st['origin'][0])} |v| {np.linalg.norm(st['velocity'][0]):5.0f} "
              f"E {E0:5.0f} | record out (5%) in {rec_exit:5.2f} s | policy: "
              + (f"out in {ph[0] * tick:5.2f} s" if len(ph) else "NOT out")
              + f", best {prog.max():5.2f}% , end {tr.get('end')} after {len(rows) * tick:5.2f} s")


if __name__ == "__main__":
    main()
