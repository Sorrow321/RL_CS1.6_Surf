#!/usr/bin/env python3
"""stitch_spine.py - one curriculum spine (+ held keys) from alternating search segments (pbr5b2).

The search -> curriculum loop (tools/auto_spine.py) splices ONE Go-Explore chain onto the
policy's beam line. At princessburglar's 56.6% shaft that is not enough (autoresearch/NOTES.md,
round 5): the explorer's archive holds states with the energy the climb needs, the explorer's own
chains waste it before the climb, and the policy, flown from those states, keeps more of it but
misses the shaft. So the line is built from alternating segments, each the policy's or the
explorer's own exact trajectory, each starting at the previous one's state:

    beam  : the policy's beam line from the true start (beam_tas --save-frontier npz)
    ge    : an exact Go-Explore chain (explore_phase1 --exact-chains, ge_replay) whose root is a
            state of the line so far (matched by position; a boosted root keeps its velocity)
    roll  : the policy's own greedy rollout from the line's last state (record_ckpt
            --dump-states npz + its .jsonl for the held keys)

    python tools/stitch_spine.py --out runs/x/spine.npy --field runs/x/field.npz \\
        beam:runs/k01/beam/beam_frontier.npz \\
        ge:runs/k01/ge/archive.npz:runs/x/chainA.npy \\
        roll:runs/x/rollB.npz:runs/x/rollB.jsonl \\
        ge:runs/x/geC/archive.npz:runs/x/chainC.npy

A ge segment's chain file is ge_replay's --out for the archive (its own --cell choice); the
chain's views go along the velocity and its keys are translated with them (auto_spine's rule).
The spine is cut at its best d on the field. Writes the spine, <out>.keys.npy (fwd, side, duck;
engine bins, for train_fast --demo-keys) and <out>.json (the segments). Every segment is the
policy's own or the run's own reward-free search: no demo, no record (CLAUDE.md section 0)."""
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

NEUTRAL = (1, 1, 0)


def chain_keys_cell(archive, cell: int, n_states: int):
    """(n_states - 1, 3) held keys of the chain to ``cell`` (auto_spine.chain_keys for any cell)."""
    from ge_replay import chain_to
    z = np.load(archive, allow_pickle=False)
    ch = chain_to(z["parent"], int(cell))
    seq = np.concatenate([z["link_acts"][z["link_off"][c]:z["link_off"][c + 1]]
                          for c in ch[1:]]) if len(ch) > 1 else np.zeros((0, 6), np.int64)
    if len(seq) != n_states - 1:
        return None
    return np.asarray(seq[:, [2, 3, 5]], np.int32)


def chain_cell(archive, chain):
    """The archive cell whose state is the chain's last state (ge_replay's chosen cell)."""
    z = np.load(archive, allow_pickle=False)
    o = np.asarray(z["state"]["origin"], np.float64)
    e = np.asarray(chain[-1]["origin"], np.float64)
    return int(np.argmin(np.linalg.norm(o - e, axis=1)))


def roll_keys(jsonl, n: int):
    """(n, 3) held keys of a recorded rollout's states: state t+1 was reached under row t's
    engine keys (fwd col 13, side col 14, duck = buttons & 4); state 0 holds neutral."""
    rows = []
    for line in open(jsonl, encoding="utf-8"):
        line = line.strip()
        if line and line[0] == "[":
            rows.append(json.loads(line))
        elif line.startswith("{") and '"end"' in line and rows:
            break
    out = np.empty((n, 3), np.int32)
    out[:] = NEUTRAL
    for t in range(1, min(n, len(rows) + 1)):
        r = rows[t - 1]
        out[t] = (int(r[13]), int(r[14]), 1 if int(r[8]) & 4 else 0)
    return out


def splice_at(line, keys, chain):
    """-> (j, err): the line index whose position matches the chain's first state."""
    ro = np.asarray(chain[0]["origin"], np.float64)
    dist = np.linalg.norm(np.asarray(line["origin"], np.float64) - ro, axis=1)
    j = int(np.argmin(dist))
    return j, float(dist[j])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("segments", nargs="+", help="beam:NPZ | ge:ARCHIVE:CHAIN.npy | roll:NPZ:JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--no-cut", action="store_true", help="keep the line past its best d")
    a = ap.parse_args(argv)
    from auto_spine import field_d, keys_from_acts, load_field
    from spine_canon_keys import translate
    from spine_canon_view import canon_view
    line = keys = None
    info = []
    for seg in a.segments:
        kind, _, rest = seg.partition(":")
        if kind == "beam":
            z = np.load(rest, allow_pickle=False)
            line = np.asarray(z["states"], STATE_DTYPE)
            keys = (keys_from_acts(z["acts"], int(z["act_every"]), len(line))
                    if "acts" in z.files else np.tile(np.int32(NEUTRAL), (len(line), 1)))
            info.append({"seg": "beam", "file": rest, "n": int(len(line))})
            continue
        if line is None:
            raise SystemExit("the first segment must be the beam line")
        if kind == "ge":
            arch, _, cf = rest.partition(":")
            chain = np.asarray(np.load(cf), STATE_DTYPE)
            j, err = splice_at(line, keys, chain)
            if err > 1.0:
                raise SystemExit(f"{cf}: its root matches no state of the line (nearest {err:.1f} u)")
            cell = chain_cell(arch, chain)
            ck = chain_keys_cell(arch, cell, len(chain))
            cv, _m = canon_view(chain[1:])
            if ck is None:
                ck = np.tile(np.int32(NEUTRAL), (len(chain) - 1, 1))
            else:
                v = np.asarray(chain["velocity"][1:], np.float64)
                hd = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
                fast = np.hypot(v[:, 0], v[:, 1]) >= 100.0
                ck = ck.copy()
                ck[fast] = translate(ck[fast], np.asarray(chain["yaw"][1:], np.float64)[fast],
                                     hd[fast])
            vr = float(np.linalg.norm(np.asarray(chain[0]["velocity"], np.float64))
                       / max(1e-6, float(np.linalg.norm(np.asarray(line[j]["velocity"],
                                                                     np.float64)))))
            line = np.concatenate([line[:j + 1], np.asarray(cv, STATE_DTYPE)])
            keys = np.concatenate([keys[:j + 1], np.asarray(ck, np.int32)])
            info.append({"seg": "ge", "archive": arch, "chain": cf, "cell": cell, "root": j,
                         "root_boost": round(vr, 3), "n": int(len(chain))})
        elif kind == "roll":
            npz, _, jl = rest.partition(":")
            z = np.load(npz, allow_pickle=False)
            ep = np.asarray(z[sorted(z.files)[0]], STATE_DTYPE)
            err = float(np.linalg.norm(np.asarray(ep[0]["origin"], np.float64)
                                       - np.asarray(line[-1]["origin"], np.float64)))
            if err > 1.0:
                raise SystemExit(f"{npz}: the rollout does not start at the line's end ({err:.1f} u)")
            rk = roll_keys(jl, len(ep))
            line = np.concatenate([line, ep[1:]])
            keys = np.concatenate([keys, rk[1:]])
            info.append({"seg": "roll", "file": npz, "n": int(len(ep))})
        else:
            raise SystemExit(f"unknown segment kind {kind!r}")
    field = load_field(a.field)
    d = field_d(field, line)
    e = len(line) - 1 if a.no_cut else int(np.argmin(d))
    line, keys = np.array(line[:e + 1], copy=True), keys[:e + 1]
    line["tick"] = 0
    line["stuck_ticks"] = 0
    out = Path(a.out)
    np.save(out, line)
    kp = out.with_suffix(".keys.npy")
    np.save(kp, np.asarray(keys, np.int32))
    res = {"n": int(len(line)), "end_d": float(d[e]), "segments": info,
           "md5": hashlib.md5(out.read_bytes()).hexdigest(),
           "keys_md5": hashlib.md5(kp.read_bytes()).hexdigest()}
    out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
