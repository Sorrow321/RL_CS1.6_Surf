"""stage_states.py - the policy's OWN states cut just before its own contact with a target of a
--ramp-sequence, each with the sequence index it starts at (seq_k) - the file --spawn-states
takes (.npz {states, seq_k}) to start a share of the training episodes part way along the list:
return-then-explore at the transition the run is stuck on (CLAUDE.md 0b: an archive of the
policy's own states; section 0: never a human record).

From one record_ckpt.py recording made with --dump-states (its .jsonl gives each episode's
--ramp-touch touches [row, target]; the .npz the full per-tick states, aligned row for row):
for every episode and every chosen k, the state LEAD seconds (several leads spread the cut) before
the episode's k-th touch, with seq_k = k. An episode that never made its k-th touch gives none.

    python tools/stage_states.py --traj rec.jsonl --states rec_states.npz --stages 2 \\
        --leads 0.3 0.5 0.8 --out runs/research/uf2_stage2.npz --source self:uf2SEQ_WRdiagS17U
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def episode_touches(traj):
    """-> [touches per episode] ([row, target] lists) and the header's tick_ms"""
    out, tick_ms = [], None
    for line in open(traj, encoding="utf-8"):
        o = json.loads(line)
        if isinstance(o, dict) and "map" in o:
            tick_ms = float(o.get("tick_ms", 10.0))
        elif isinstance(o, dict) and "end" in o:
            out.append(list(((o.get("targets") or {}).get("touches")) or []))
    return out, tick_ms


def cut(touches, eps, stages, leads, tick_ms):
    """-> (states, seq_k, report): per episode / stage / lead the state LEAD before touch k"""
    rows, ks, rep = [], [], []
    for e, (tch, arr) in enumerate(zip(touches, eps)):
        for k in stages:
            if k >= len(tch):
                continue
            r = int(tch[k][0])
            for lead in leads:
                j = r - int(round(lead * 1000.0 / tick_ms))
                if 0 <= j < len(arr):
                    rows.append(arr[j])
                    ks.append(int(k))
                    rep.append((e, k, int(tch[k][1]), float(lead), j))
    if not rows:
        return None, None, rep
    return np.array(rows, dtype=eps[0].dtype), np.asarray(ks, np.int64), rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True, help="the recording's .jsonl (with touches)")
    ap.add_argument("--states", required=True, help="its --dump-states .npz")
    ap.add_argument("--stages", type=int, nargs="+", required=True,
                    help="sequence indices k to cut before (the k-th touch, 0-based)")
    ap.add_argument("--leads", type=float, nargs="+", default=[0.5],
                    help="seconds before the touch (several spread the cut)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", required=True,
                    help="provenance: self:<run> (the policy's own recording)")
    args = ap.parse_args(argv)
    if not str(args.source).startswith("self:"):
        raise SystemExit("--source must be self:<run>: these are spawn states for TRAINING, "
                         "and only the policy's own recordings may supply them (CLAUDE.md "
                         "section 0)")
    touches, tick_ms = episode_touches(args.traj)
    z = np.load(args.states, allow_pickle=False)
    eps = [z[k] for k in sorted(z.files)]
    if len(eps) != len(touches):
        raise SystemExit(f"{len(eps)} dumped episodes for {len(touches)} recorded ones")
    st, ks, rep = cut(touches, eps, args.stages, args.leads, tick_ms or 10.0)
    if st is None:
        raise SystemExit("no episode reached a chosen stage: nothing to cut")
    np.savez(args.out, states=st, seq_k=ks, source=str(args.source))
    for k in sorted(set(int(x) for x in ks)):
        n = int((ks == k).sum())
        tg = sorted({t for _, kk, t, _, _ in rep if kk == k})
        print(f"stage k={k} (target {tg}): {n} states")
    print(f"{len(st)} own states -> {args.out} (source {args.source})")


if __name__ == "__main__":
    main()
