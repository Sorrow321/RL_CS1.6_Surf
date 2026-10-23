#!/usr/bin/env python3
"""pb_where.py - where each greedy eval episode got to, read against a reference line.

ANALYSIS ONLY (CLAUDE.md section 0): the reference line is typically the human record of the
map, and it is used here as a ruler - which part of the course an episode reached - never as
a training input. Per episode: the end label, the duration, the honest progress on the grader's
field (score.py), and for the episode's FURTHEST point (min d): its position, speed, the nearest
reference-line row (expressed as reference-line time) and the distance to it.

    python tools/pb_where.py --ref /home/steam/ar_pb/incumbent/wr/pb_wr.jsonl runs/x/traj_*.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
sys.path.insert(0, str(ROOT / "python"))
import score as S  # noqa: E402


def load_ref(path):
    hdr, rows = None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            if hdr is None:
                hdr = json.loads(line)
            continue
        rows.append(json.loads(line))
    r = np.asarray(rows, np.float64)
    return hdr or {}, r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", required=True, help="reference line (record_ckpt jsonl)")
    ap.add_argument("--all", action="store_true", help="every episode, not just the first 9")
    a = ap.parse_args(argv)
    f = S.field()
    rh, rr = load_ref(a.ref)
    rdt = float(rh.get("tick_ms", 10.0)) / 1000.0
    rp = rr[:, 1:4]
    rd = f.sample(rp).astype(np.float64)
    rok = rd < f.reach_max - 0.5 * f.cell
    # the reference line's own clock starts at its first on-field row
    i0 = int(np.argmax(rok))
    ref_idx = np.where(rok)[0]
    for fn in a.files:
        eps = S.episodes(Path(fn))
        if not a.all:
            eps = S.first_episodes(eps, S.SCORE_EPISODES)
        print(f"== {fn}: {len(eps)} episodes")
        for hdr, r, tr in eps:
            p = r[:, 1:4]
            v = r[:, 4:7]
            d = f.sample(p).astype(np.float64)
            ok = d < f.reach_max - 0.5 * f.cell
            d0 = float(d[0])
            j = int(np.argmin(np.where(ok, d, np.inf)))
            prog = 100.0 * (d0 - d[j]) / d0
            secs = len(r) * float(hdr.get("tick_ms", 10.0)) / 1000.0
            # nearest reference row to the furthest point
            dist = np.linalg.norm(rp[ref_idx] - p[j], axis=1)
            k = ref_idx[int(np.argmin(dist))]
            spd = float(np.linalg.norm(v[j]))
            spd_end = float(np.linalg.norm(v[-1]))
            print(f"  ep {hdr.get('episode')}: {tr.get('end'):5s} {secs:6.1f}s  prog {prog:5.1f}%"
                  f"  far@{j * float(hdr.get('tick_ms', 10.0)) / 1000:5.1f}s"
                  f" ({p[j, 0]:6.0f},{p[j, 1]:6.0f},{p[j, 2]:6.0f}) |v| {spd:5.0f}"
                  f"  ref t {(k - i0) * rdt:5.1f}s (off {dist.min():5.0f}u, ref |v| "
                  f"{np.linalg.norm(rr[k, 4:7]):5.0f})  end ({p[-1, 0]:6.0f},{p[-1, 1]:6.0f},"
                  f"{p[-1, 2]:6.0f}) |v| {spd_end:5.0f}")


if __name__ == "__main__":
    main()
