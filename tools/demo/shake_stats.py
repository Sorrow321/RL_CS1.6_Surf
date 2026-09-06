"""shake_stats.py - where does the camera shake come from?

The first-person video is rendered from the recording's per-tick yaw and
pitch, so the camera's motion IS the recorded view-angle sequence. This
compares that sequence, tick by tick at the same 130 Hz, between our
recordings (record_ckpt / expert-loop eval trajectories) and the human
record (parse_hldemo frames), per axis:

  rate         |d angle| per tick (deg/tick) - how fast the view turns
  reversals/s  sign changes of the per-tick delta - back-and-forth flicks
  jerk         |d2 angle| per tick (deg/tick^2) - changes of turn rate,
               what a viewer feels as shake
  HF power     share of the turn-rate signal's power above 5 Hz / 10 Hz
  hold step    jerk AT decision boundaries (every act_every ticks) vs inside
               a hold - the staircase of a held bin

Usage:
    python tools/demo/shake_stats.py --wr runs/wr_demo/wr_cannonball.frames.npz \
        --traj LABEL=path.jsonl[:episode] [--traj ...] [--act-every 4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HZ = 1000.0 / 7.6667


def load_traj(path, ep=None):
    """-> list of (t, yaw, pitch) arrays per episode (all, or just `ep`)."""
    eps, rows = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if isinstance(r, dict) and "map" in r:
            rows = []
        elif isinstance(r, list):
            rows.append(r)
        elif isinstance(r, dict) and "end" in r:
            a = np.asarray(rows, np.float64)
            eps.append(dict(t=a[:, 0], yaw=a[:, 7], pitch=a[:, 12] if a.shape[1] > 12 else np.zeros(len(a)),
                            end=r.get("end"), n=len(a)))
            rows = []
    if ep is not None:
        return [eps[ep]]
    return eps


def load_record(path):
    d = np.load(path)
    yaw = np.rad2deg(np.unwrap(np.deg2rad(d["uc_viewangles"][:, 1])))
    pitch = -d["uc_viewangles"][:, 0]
    t = d["t"] - 1.81
    m = (t >= 0.0) & (t <= 68.6)
    return dict(t=t[m], yaw=yaw[m], pitch=pitch[m], n=int(m.sum()))


def stats(angle, k=4, eps=0.02):
    a = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(angle, np.float64))))
    d1 = np.diff(a)
    d2 = np.diff(d1)
    n_s = len(d1) / HZ
    sg = np.sign(np.where(np.abs(d1) < eps, 0.0, d1))
    nz = sg[sg != 0]
    rev = int(np.sum(nz[1:] * nz[:-1] < 0))
    # spectrum of the turn rate
    x = d1 - d1.mean()
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / HZ)
    tot = P[1:].sum() if P[1:].sum() > 0 else 1.0
    hf5 = P[f > 5.0].sum() / tot
    hf10 = P[f > 10.0].sum() / tot
    # decision-boundary structure: jerk at t % k == 0 vs elsewhere
    # the recording's tick 0 need not sit on a decision boundary: take the
    # phase whose ticks carry the most jerk as the boundary phase
    idx = np.arange(len(d2))
    per = [np.abs(d2[(idx % k) == ph]).mean() for ph in range(k)]
    jerk_b = float(max(per))
    jerk_i = float(np.mean([v for j, v in enumerate(per) if j != int(np.argmax(per))])) if k > 1 else np.nan
    return dict(rate_p50=np.median(np.abs(d1)), rate_p90=np.percentile(np.abs(d1), 90),
                rev_per_s=rev / n_s, jerk=np.abs(d2).mean(), jerk_p95=np.percentile(np.abs(d2), 95),
                hf5=hf5, hf10=hf10, jerk_b=jerk_b, jerk_i=jerk_i, rng=a.max() - a.min(), std=a.std())


def fmt_row(label, s):
    return (f"| {label} | {s['rate_p50']:.3f} / {s['rate_p90']:.3f} | {s['rev_per_s']:.1f} | {s['jerk']:.3f} / {s['jerk_p95']:.3f} | "
            f"{100*s['hf5']:.0f}% / {100*s['hf10']:.0f}% | {s['jerk_b']:.3f} vs {s['jerk_i']:.3f} | {s['rng']:.0f} / {s['std']:.1f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr", required=True)
    ap.add_argument("--traj", action="append", default=[], help="LABEL=path[:episode]")
    ap.add_argument("--act-every", type=int, default=4)
    ap.add_argument("--only-finished", action="store_true")
    args = ap.parse_args()
    k = args.act_every
    srcs = [("human record", [load_record(args.wr)])]
    for spec in args.traj:
        label, _, rest = spec.partition("=")
        path, _, ep = rest.rpartition(":")
        if not path or not ep.isdigit():
            path, ep = rest, None
        eps = load_traj(path, int(ep) if ep is not None else None)
        if args.only_finished:
            eps = [e for e in eps if e["n"] > 8000]
        srcs.append((label, eps))
    hdr = ("| source | yaw: |rate| p50 / p90 (deg/tick) | reversals/s | jerk mean / p95 (deg/tick^2) | HF power >5 Hz / >10 Hz | "
           "jerk at decision boundary vs inside hold | range / std (deg) |")
    for axis in ("yaw", "pitch"):
        print(f"\n### {axis}\n")
        print(hdr.replace("yaw:", axis + ":"))
        print("|---|---|---|---|---|---|---|")
        for label, eps in srcs:
            ss = [stats(e[axis], k) for e in eps]
            m = {key: float(np.nanmean([s[key] for s in ss])) for key in ss[0]}
            print(fmt_row(f"{label} ({len(eps)} ep)", m))
    # what a viewer sees: the per-frame angular step of the camera at 100 fps output
    print("\nfor reference: 1 deg/tick at 130 Hz is 130 deg/s; a 90-deg FOV frame is 640 px wide at 720p, so 0.1 deg/tick "
          "of jerk is ~1 px of image slip per frame changing sign - visible as shake.")


if __name__ == "__main__":
    main()
