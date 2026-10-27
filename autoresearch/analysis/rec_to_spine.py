#!/usr/bin/env python3
"""rec_to_spine.py - one episode of a record_ckpt --dump-states recording as a curriculum spine (pbr9b1).

record_ckpt.py --dump-states writes the full per-tick STATE_DTYPE states of every recorded episode
(npz keys ep_0000, ...) beside its .jsonl. This writes episode E as a spine (.npy, tick and
stuck_ticks zeroed) and its held keys (.keys.npy: fwd col 13, side col 14, duck = buttons & 4 of
the jsonl row before each state; state 0 neutral) - stitch_spine.py's `spine:NPY:KEYS` first
segment and train_fast's --demo-file / --demo-keys. The recording is the policy's own greedy line:
no demo, no record data (CLAUDE.md section 0).

    python autoresearch/analysis/rec_to_spine.py --npz g0_states.npz --jsonl g0.jsonl --ep 1 --out g0e1.npy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

NEUTRAL = (1, 1, 0)


def episode_rows(jsonl, ep: int):
    rows, cur, hdr, k = None, [], None, -1
    for line in open(jsonl, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None:
                if k == ep:
                    return cur, hdr, d
                hdr, cur = None, []
            else:
                hdr, cur = d, []
                k += 1
            continue
        cur.append(json.loads(line))
    raise SystemExit(f"{jsonl}: no episode {ep}")


def main(argv=None) -> int:
    from surfgym.core import STATE_DTYPE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--rows", default=None, help="LO:HI - keep rows [LO, HI) only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    z = np.load(a.npz, allow_pickle=False)
    st = np.asarray(z[f"ep_{a.ep:04d}"], STATE_DTYPE)
    rows, hdr, tr = episode_rows(a.jsonl, a.ep)
    n = len(st)
    keys = np.empty((n, 3), np.int32)
    keys[:] = NEUTRAL
    for t in range(1, min(n, len(rows) + 1)):
        r = rows[t - 1]
        keys[t] = (int(r[13]), int(r[14]), 1 if int(r[8]) & 4 else 0)
    if a.rows:
        lo, hi = (int(x) for x in a.rows.split(":"))
        st, keys = st[lo:hi], keys[lo:hi]
    st = np.array(st, copy=True)
    st["tick"] = 0
    st["stuck_ticks"] = 0
    out = Path(a.out)
    np.save(out, st)
    np.save(out.with_suffix(".keys.npy"), keys)
    print(json.dumps({"n": int(len(st)), "end": tr.get("end"), "tick_ms": hdr.get("tick_ms"),
                      "md5": hashlib.md5(out.read_bytes()).hexdigest(),
                      "keys_md5": hashlib.md5(out.with_suffix(".keys.npy").read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
