#!/usr/bin/env python3
"""pb_line_tools.py - own-line helpers for the GE + backward-curriculum loop (pbr2b4, round 2).

Generic: nothing here reads a map coordinate, a box or a demo. Every input is the policy's own
greedy recording (record_ckpt --dump-states) or a reward-free Go-Explore archive rooted at it.

  roots   <dump.npz> <out.npy> [--ep K] [--window-s S] [--stride N]
          the greedy episode's states from (best-progress tick - S seconds) to the best tick
          (the policy's own frontier approach) as --roots-spine for explore_phase1
  profile <dump.npz>   per episode: best progress on the grader field, its tick, the end
  join    <dump.npz> <chain_line.npy> <out.npy> [--ep K]
          prefix the replayed chain (tools/ge_replay.py) with the greedy episode's own states
          up to the chain's root, so the spine starts at the TRUE start (index 0)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "autoresearch"))

from surfgym.core import STATE_DTYPE  # noqa: E402


def field():
    import score as S
    return S.field()


def prog(f, st):
    d = f.sample(st["origin"].astype(np.float64)).astype(np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.nan
    return 100.0 * (57149.0 - d) / 57149.0


def eps_of(path):
    z = np.load(path, allow_pickle=False)
    return [z[k] for k in sorted(z.files)]


def cmd_profile(a):
    f = field()
    for k, e in enumerate(eps_of(a.dump)):
        p = prog(f, e)
        i = int(np.nanargmax(p))
        o = e["origin"][i]
        v = e["velocity"][i]
        print(f"ep {k}: {len(e)} ticks, best {p[i]:.3f}% at tick {i} ({o[0]:.0f},{o[1]:.0f},{o[2]:.0f})"
              f" |v| {np.linalg.norm(v):.0f}; end ({e['origin'][-1][0]:.0f},{e['origin'][-1][1]:.0f},"
              f"{e['origin'][-1][2]:.0f})")


def cmd_roots(a):
    f = field()
    e = eps_of(a.dump)[a.ep]
    p = prog(f, e)
    i = int(np.nanargmax(p))
    lo = max(0, i - int(round(a.window_s * 100)))
    r = e[lo:i + 1:a.stride].copy()
    out = np.zeros(len(r), STATE_DTYPE)
    for n in STATE_DTYPE.names:
        out[n] = r[n]
    out["tick"] = 0
    out["stuck_ticks"] = 0
    np.save(a.out, out)
    print(f"{len(out)} roots = ep {a.ep} ticks {lo}..{i} stride {a.stride} (best {p[i]:.3f}%) -> {a.out}")


def cmd_join(a):
    f = field()
    e = eps_of(a.dump)[a.ep]
    line = np.load(a.line, allow_pickle=False)
    o0 = line[0]["origin"].astype(np.float64)
    dist = np.linalg.norm(e["origin"].astype(np.float64) - o0, axis=1)
    j = int(np.argmin(dist))
    if dist[j] > 0.01:
        raise SystemExit(f"the chain root is not a state of episode {a.ep} (closest {dist[j]:.3f} u)")
    pre = np.zeros(j, STATE_DTYPE)
    for n in STATE_DTYPE.names:
        pre[n] = e[:j][n]
    out = np.concatenate([pre, line.astype(STATE_DTYPE)])
    out["tick"] = 0
    out["stuck_ticks"] = 0
    np.save(a.out, out)
    p = prog(f, out)
    print(f"joined: greedy prefix {j} ticks + chain {len(line)} = {len(out)} states -> {a.out}; "
          f"end {p[-1]:.3f}%, best {np.nanmax(p):.3f}% at {int(np.nanargmax(p))}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile"); p.add_argument("dump"); p.set_defaults(fn=cmd_profile)
    p = sub.add_parser("roots"); p.add_argument("dump"); p.add_argument("out")
    p.add_argument("--ep", type=int, default=0); p.add_argument("--window-s", type=float, default=4.0)
    p.add_argument("--stride", type=int, default=2); p.set_defaults(fn=cmd_roots)
    p = sub.add_parser("join"); p.add_argument("dump"); p.add_argument("line"); p.add_argument("out")
    p.add_argument("--ep", type=int, default=0); p.set_defaults(fn=cmd_join)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
