#!/usr/bin/env python3
"""splice_roll.py - cut a stitched spine at one of its rows and continue it with a policy rollout (pbr9b3).

tools/stitch_spine.py appends a roll segment at the line's LAST state, and its ge segment takes a
whole Go-Explore chain (the archive's link actions are looked up for the chain's end cell, so a chain
cut short loses its keys). When the policy must take over INSIDE the chain (round 9: the explorer's
chain past princessburglar's 84% platform keeps 400 u more energy than the policy's line, but its last
states plunge into the lower level's floor, and the policy finishes only from the chain's states
before that), stitch the whole chain with --no-cut, then:

    python tools/splice_roll.py --spine S.npy --row J --roll line.npz --roll-jsonl beam_best.jsonl \\
        --field F.npz --out OUT.npy

keeps S[:J+1] (+ keys), checks that the rollout starts at S[J] (beam_tas --start-state on that very
state; tools/beam_best_states.py writes its per-tick states), appends the rollout's states and held
keys, and cuts the result at its best d like stitch_spine. Every state is the policy's own or the run's
own reward-free search: no demo, no record (CLAUDE.md section 0)."""
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

from surfgym.core import STATE_DTYPE  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spine", required=True, help="a stitched spine (its .keys.npy beside it)")
    ap.add_argument("--row", type=int, required=True, help="the spine row the rollout starts at")
    ap.add_argument("--roll", required=True, help="the rollout's states (npz, first array)")
    ap.add_argument("--roll-jsonl", required=True, help="the rollout's trajectory (held keys)")
    ap.add_argument("--field", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-cut", action="store_true", help="keep the line past its best d")
    a = ap.parse_args(argv)
    from auto_spine import field_d, load_field
    from stitch_spine import roll_keys
    sp = np.asarray(np.load(a.spine), STATE_DTYPE)
    kp = Path(a.spine).with_suffix(".keys.npy")
    keys = np.asarray(np.load(kp), np.int32)
    if keys.shape != (len(sp), 3):
        raise SystemExit(f"{kp}: {keys.shape} keys for {len(sp)} states")
    j = int(a.row)
    if not 0 <= j < len(sp):
        raise SystemExit(f"--row {j} outside the spine (0..{len(sp) - 1})")
    z = np.load(a.roll, allow_pickle=False)
    ep = np.asarray(z[sorted(z.files)[0]], STATE_DTYPE)
    err = float(np.linalg.norm(np.asarray(ep[0]["origin"], np.float64)
                               - np.asarray(sp[j]["origin"], np.float64)))
    verr = float(np.linalg.norm(np.asarray(ep[0]["velocity"], np.float64)
                                - np.asarray(sp[j]["velocity"], np.float64)))
    if err > 1.0 or verr > 1.0:
        raise SystemExit(f"{a.roll}: the rollout does not start at spine row {j} "
                         f"({err:.2f} u, {verr:.2f} u/s apart)")
    rk = roll_keys(a.roll_jsonl, len(ep))
    line = np.concatenate([sp[:j + 1], ep[1:]])
    kk = np.concatenate([keys[:j + 1], rk[1:]])
    field = load_field(a.field)
    d = field_d(field, line)
    e = len(line) - 1 if a.no_cut else int(np.argmin(d))
    line, kk = np.array(line[:e + 1], copy=True), kk[:e + 1]
    line["tick"] = 0
    line["stuck_ticks"] = 0
    out = Path(a.out)
    np.save(out, line)
    ko = out.with_suffix(".keys.npy")
    np.save(ko, np.asarray(kk, np.int32))
    res = {"n": int(len(line)), "end_d": float(d[e]), "spine": a.spine, "row": j,
           "roll": a.roll, "roll_jsonl": a.roll_jsonl, "roll_n": int(len(ep)),
           "md5": hashlib.md5(out.read_bytes()).hexdigest(),
           "keys_md5": hashlib.md5(ko.read_bytes()).hexdigest()}
    out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
