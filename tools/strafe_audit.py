"""strafe_audit.py - how much of the engine's air acceleration is the
policy actually collecting?

At sv_airaccelerate 100 the air impulse is saturated (accelspeed 250 >>
the 30u projection cap), so per tick

    mu = max(0, 30 - v.wishdir),   |v'|^2 = |v|^2 + 900 - (v.wishdir)^2

Two consequences the policy has to satisfy simultaneously: the ONLY
speed-maximizing wishdir is exactly perpendicular to velocity, and the
engine applies nothing at all once v.wishdir >= 30 - i.e. the usable
window is +-arcsin(30/|v|), which is +-0.52 deg at 3000 u/s. This script
reconstructs wishdir from the recorded (yaw, fmove, smove) and reports
what fraction of the 900/tick ceiling the run actually banked.

Reading the output: "capture" is the honest number. It can be NEGATIVE -
past perpendicular the same impulse brakes, so a policy that over-turns
loses speed it would have kept by doing nothing.

Usage: python tools/strafe_audit.py <traj.jsonl> [--ep N] [--label NAME]
"""
import argparse
import json

import numpy as np

AIR_CAP = 30.0
MAX_DV2 = AIR_CAP ** 2


def load_episodes(path):
    eps, rows = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, dict) and "map" in row:
                rows = []
            elif isinstance(row, list):
                rows.append(row)
            elif isinstance(row, dict) and "end" in row:
                if rows:
                    eps.append(np.asarray(rows, dtype=np.float64))
                rows = []
    return eps


def audit(a):
    """(stats dict) for one episode array."""
    v, yaw = a[:, 4:7], np.radians(a[:, 7])
    fb, sb = a[:, 13].astype(int), a[:, 14].astype(int)
    fmove = np.where(fb <= 0, -400.0, np.where(fb >= 2, 400.0, 0.0))
    smove = np.where(sb <= 0, -400.0, np.where(sb >= 2, 400.0, 0.0))
    cy, sy = np.cos(yaw), np.sin(yaw)
    # AngleVectors with roll=0: forward=(cos,sin,0), right=(sin,-cos,0)
    wx, wy = cy * fmove + sy * smove, sy * fmove - cy * smove
    n = np.hypot(wx, wy)
    ok = n > 1e-6
    wdx = np.where(ok, wx / np.maximum(n, 1e-9), 0.0)
    wdy = np.where(ok, wy / np.maximum(n, 1e-9), 0.0)
    vh = np.hypot(v[:, 0], v[:, 1])
    c = v[:, 0] * wdx + v[:, 1] * wdy            # v . wishdir
    theta = np.degrees(np.arccos(np.clip(c / np.maximum(vh, 1e-9), -1, 1)))
    acc = ok & (c < AIR_CAP)                     # engine accelerates only here
    dv2 = np.where(acc, MAX_DV2 - c ** 2, 0.0)   # realized growth of |v_h|^2
    m = slice(0, len(a) - 1)
    nt = len(a) - 1
    return {
        "ticks": nt,
        "mean_speed": float(vh.mean()),
        "peak_speed": float(vh.max()),
        "capture": float(dv2[m].sum() / (MAX_DV2 * nt)),
        "in_window": float(acc[m].mean()),
        "abs_theta_err_med": float(np.median(np.abs(theta[m] - 90.0))),
        "within5deg": float(np.mean(np.abs(theta[m] - 90.0) < 5.0)),
        "mean_turn": float(np.abs(np.diff((a[:, 7] + 540.0) % 360.0 - 180.0)).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--ep", type=int, default=None, help="1-based; default all")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    eps = load_episodes(args.traj)
    sel = [eps[args.ep - 1]] if args.ep else eps
    if not sel:
        raise SystemExit(f"no episodes in {args.traj}")
    st = [audit(e) for e in sel]
    w = np.array([s["ticks"] for s in st], dtype=np.float64)

    def wm(key):
        return float(np.average([s[key] for s in st], weights=w))

    tag = args.label or args.traj
    print(f"{tag}  ({len(sel)} episodes, {int(w.sum()):,} ticks)")
    print(f"  air-accel capture   {wm('capture') * 100:8.1f} %  "
          f"of the 900/tick ceiling")
    print(f"  ticks in window     {wm('in_window') * 100:8.1f} %  "
          f"(engine applies nothing outside)")
    print(f"  median |theta-90|   {wm('abs_theta_err_med'):8.2f} deg   "
          f"within 5 deg: {wm('within5deg') * 100:.1f}%")
    print(f"  mean speed          {wm('mean_speed'):8,.0f} u/s   "
          f"peak {max(s['peak_speed'] for s in st):,.0f}")
    print(f"  mean |yaw delta|    {wm('mean_turn'):8.3f} deg/tick")


if __name__ == "__main__":
    main()
