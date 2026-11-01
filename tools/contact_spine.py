#!/usr/bin/env python3
"""contact_spine.py - a curriculum spine from a policy's OWN greedy line + a Go-Explore exact chain
rooted on it, cut at the chain's deepest CONTACT tick (zlr2b1, 2026-11-01).

tools/ge_spine.py cuts at the deepest state on the goal field. On a map whose field flies through
open air that is a ballistic apex, so the spine would end in mid-air over the void. This cuts at
the deepest tick the map HOLDS (walkable ground, or a surfable face pushing vz back up - the
--ratchet-contact test), the only kind of progress --ratchet-contact pays.

  base   : a STATE_DTYPE line from the true start (a record_ckpt --dump-states episode, key
           ep_NNNN of an .npz, or a .npy) - the policy's own states
  chain  : tools/ge_replay.py's line; its first state is a root taken from the base
  output : base[:root] + chain (the chain's view re-aimed along the velocity,
           tools/spine_canon_view.py), cut at the deepest contact tick; tick/stuck_ticks zeroed

    python tools/contact_spine.py runs/x/states.npz:ep_0000 runs/x/line.npy out.npy \\
        --field maps/X.goalc_32.npz --tick-ms 7 [--d0 6361.8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from contact_roots import contact_mask          # noqa: E402
from spine_canon_view import canon_view         # noqa: E402


def load_line(spec: str) -> np.ndarray:
    path, _, key = spec.partition(":")
    z = np.load(path)
    if hasattr(z, "files"):
        return z[key or sorted(z.files)[0]]
    return z


def main(argv=None):
    from surfgym.core import STATE_DTYPE
    from surfgym.goalfield import load_goal_field
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("chain")
    ap.add_argument("out")
    ap.add_argument("--field", required=True)
    ap.add_argument("--tick-ms", type=float, required=True)
    ap.add_argument("--d0", type=float, default=None)
    a = ap.parse_args(argv)
    base = np.asarray(load_line(a.base), STATE_DTYPE)
    gl = np.asarray(np.load(a.chain), STATE_DTYPE)
    o = gl[0]["origin"].astype(np.float64)
    dist = np.linalg.norm(base["origin"].astype(np.float64) - o, axis=1)
    j = int(np.argmin(dist))
    if dist[j] > 1.0:
        raise SystemExit(f"the chain's root is {dist[j]:.2f} u from the nearest base state: "
                         "not rooted on this line")
    rest, m = canon_view(gl[1:])
    sp = np.concatenate([base[:j], gl[:1], np.asarray(rest, STATE_DTYPE)])
    f = load_goal_field(a.field)
    d = f.sample(sp["origin"].astype(np.float64)).astype(np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    d0 = float(a.d0) if a.d0 else float(d[0])
    c = contact_mask(sp, a.tick_ms)
    dc = np.where(c, d, np.inf)
    e = int(np.argmin(dc))
    sp = np.array(sp[:e + 1], copy=True)
    sp["tick"] = 0
    sp["stuck_ticks"] = 0
    np.save(a.out, sp)
    base_c = np.where(contact_mask(base[:j + 1], a.tick_ms),
                      f.sample(base[:j + 1]["origin"].astype(np.float64)), np.inf)
    info = {"root_index": j, "n": int(len(sp)), "secs": round(len(sp) * a.tick_ms / 1000.0, 2),
            "end_d": float(d[e]), "end_pct": round(100.0 * (d0 - d[e]) / d0, 2),
            "end_origin": np.asarray(sp[-1]["origin"]).round(0).tolist(),
            "base_contact_best_pct": round(100.0 * (d0 - float(np.min(base_c))) / d0, 2),
            "chain_len": int(len(gl)), "canon_rows": int(m.sum()),
            "md5": hashlib.md5(open(a.out, "rb").read()).hexdigest()}
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
