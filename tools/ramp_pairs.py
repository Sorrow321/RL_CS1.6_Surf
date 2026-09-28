"""ramp_pairs.py - a finisher's ramp sequence as training PAIRS (the user, 2026-09-28).

"Take finisher lines for some maps, extract sequences of ramps and just train on it statically
(just taking sequences of 2) ... so that the model gets more examples to learn the pattern 'see
this => do this'."

From a recorded run that finishes the map (the FULL per-tick states of record_ckpt --dump-states,
or a human record's frames), with the map's ramp vocabulary (tools/ramps_mesh.py):

1. the validated contact surface at every tick (RampVocab.contact_of), its target surfaces only;
2. the ORDERED sequence of distinct targets the run rides: a new element each time the contact
   moves to another target surface, touches under MIN_TICKS dropped, a surface repeated only
   after another one (a ramp left and landed on again stays twice);
3. the PAIRS (r_i, r_(i+1)), each with a spawn STATE: the run's own state LEAD s before its first
   contact with r_i (the map start for the first pair) - so a training episode starts in the air
   heading for r_i with the finisher's own speed, is shown [r_i, r_(i+1)], and succeeds once it
   has passed r_i and entered r_(i+1).

    python tools/ramp_pairs.py --vocab runs/research/ramps_mesh_surf_src_utopia.npz \\
        --bsp maps_pool/surf_src_utopia.bsp --states runs/research/utopia_finisher_states.npz \\
        --out runs/research/pairs/utopia_pairs.npz --source self:jt3ANCHU

Output npz: states (P,) STATE_DTYPE, t1 (P,), t2 (P,), t_spawn (P,) seconds into the run,
sequence (the ordered targets), map, source (provenance: 'self:<run>' or 'demo:<record>').
A demo source marks the pairs demo-derived (CLAUDE.md section 0 - a labelled diagnostic).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

MIN_TICKS = 3          # a contact shorter than this is a graze, not a ride
LEAD = 0.5             # s: a pair's spawn is the run's state this long before its first contact


def contact_runs(voc, org, duck, tick_s):
    """[(surface, first tick, last tick)] of the target contacts, in order"""
    cs = np.asarray(voc.contact_of(org, duck), np.int64)
    targets = set(int(k) for k in voc.tp)
    runs, cur = [], None
    for k, s in enumerate(cs):
        s = int(s)
        if s not in targets:
            continue
        if cur is not None and cur[0] == s and k - cur[2] <= max(1, int(0.1 / tick_s)):
            cur[2] = k
            continue
        if cur is not None:
            runs.append(cur)
        cur = [s, k, k]
    if cur is not None:
        runs.append(cur)
    return [r for r in runs if r[2] - r[1] + 1 >= MIN_TICKS]


def sequence(runs, voc):
    """the ordered distinct targets: consecutive runs on the same PIECE merge"""
    seq = []
    for s, k0, k1 in runs:
        if seq and voc.piece[s] == voc.piece[seq[-1][0]]:
            seq[-1][2] = k1
            continue
        seq.append([s, k0, k1])
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--bsp", required=True)
    ap.add_argument("--states", default=None, help="record_ckpt --dump-states npz (ep_0000, ...)")
    ap.add_argument("--episode", default="ep_0000")
    ap.add_argument("--wr-frames", default=None, help="a human record's frames npz (demo source)")
    ap.add_argument("--t-max", type=float, default=None, help="only the first T s of the run")
    ap.add_argument("--tick-ms", type=float, default=10.0)
    ap.add_argument("--source", required=True, help="provenance: self:<run> or demo:<record>")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from surfgym.rampvocab import RampVocab
    from surfgym.core import STATE_DTYPE
    voc = RampVocab(args.vocab, args.bsp)
    if args.states:
        st = np.load(args.states)[args.episode]
        tick_s = args.tick_ms / 1000.0
        org = st["origin"].astype(np.float64)
        duck = st["ducked"].astype(np.int64)
    elif args.wr_frames:
        z = np.load(args.wr_frames)
        t = z["t"]
        tick_s = float(np.mean(np.diff(t)))
        org = z["simorg"].astype(np.float64)
        vel = z["simvel"].astype(np.float64)
        duck = ((z["uc_buttons"] & 4) != 0).astype(np.int64)
        st = np.zeros(len(org), STATE_DTYPE)
        st["origin"] = org
        st["velocity"] = vel
        st["yaw"] = z["viewangles"][:, 1]
        st["pitch"] = z["viewangles"][:, 0]
        st["ducked"] = duck
        st["onground"] = z["onground"]
    else:
        raise SystemExit("--states or --wr-frames")
    n = len(org) if args.t_max is None else min(len(org), int(args.t_max / tick_s))
    runs = contact_runs(voc, org[:n], duck[:n], tick_s)
    seq = sequence(runs, voc)
    if len(seq) < 2:
        raise SystemExit(f"only {len(seq)} target(s) ridden - no pair")
    lead = int(round(LEAD / tick_s))
    rows, t1, t2, ts = [], [], [], []
    for i in range(len(seq) - 1):
        k = 0 if i == 0 else max(seq[i - 1][2] + 1, seq[i][1] - lead)
        rows.append(st[k])
        t1.append(seq[i][0])
        t2.append(seq[i + 1][0])
        ts.append(k * tick_s)
    states = np.array(rows, dtype=st.dtype)
    if "tick" in states.dtype.names:
        states["tick"] = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, states=states, t1=np.asarray(t1, np.int64), t2=np.asarray(t2, np.int64),
             t_spawn=np.asarray(ts, np.float64),
             sequence=np.asarray([s for s, _a, _b in seq], np.int64), map=Path(args.bsp).stem,
             source=str(args.source))
    print(f"{Path(args.bsp).stem}: {len(seq)} targets ridden in order "
          f"{[s for s, _a, _b in seq]} -> {len(t1)} pairs (source {args.source}) -> {out}")
    for i in range(len(t1)):
        print(f"  pair {i}: [{t1[i]}, {t2[i]}] spawn t {ts[i]:.2f} s at "
              f"{states['origin'][i].round(0)} speed {np.linalg.norm(states['velocity'][i]):.0f}")


if __name__ == "__main__":
    main()
