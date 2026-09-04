#!/usr/bin/env python3
"""pick_selfline.py - pick and trim ONE non-finishing episode to build a
reference line from the agent's OWN play.

Round 19's xARC and xAUTO both paid arc length along a line extracted from a
recording of a FINISHER, so neither licenses any claim about autonomous
discovery. The next question is REACH: does a monotone progress coordinate
that covers only the part of the map the agent already flies still remove the
barrier? Building that line means taking a recording from a checkpoint that
has never finished - and such a recording ENDS IN A FALL.

That fall is not a detail. Measured on surf_src_cannonball's stuck checkpoint
(270 recorded greedy episodes across xROUTE / xSP / xNECTO / xCONTACT), every
episode leaves the ramp at ~88.6% of the route and then free-falls 1.8-6.3 s
into the pit. Resampling such an episode raw gives a line whose last few
thousand units point DOWN, and the reward then pays for falling: replaying the
recordings through ``surfgym.route.ArcProgress`` on an untrimmed line, the
checkpoint's own fallers reach **99.3%** of it while episodes that actually
FINISH the map reach only **96.9%**. A line built that way is a second
treatment, not a truncation.

So the tail has to be cut, and the cut must not smuggle in the answer. Three
rules were measured on the real recordings; two fail:

* **the champion route as a ruler** (cut at the episode's furthest corridor
  progress along the champion line) - works, but injects champion knowledge
  into the one number this experiment is about;
* **the map's own geodesic goal field** (cut at the episode's minimum distance
  to goal) - lands 0.55-1.31 s INSIDE the fall, because the field's deceptive
  basin is goal-adjacent airspace below the ramp. That is the same defect that
  made the arc reward necessary in the first place (ledger round 18), showing
  up again in the only map-derived selector available;
* **consensus across the checkpoint's own episodes** (cut where an episode
  stops agreeing with its siblings) - fails because the failure is not
  idiosyncratic. One checkpoint played greedily falls the SAME way every time,
  so the falls corroborate each other: at a 256 u median-agreement radius the
  rule dropped 0.05 s of a 2.2 s tail.

What survives is **the last tick at which the map pushed back**. Between
surface contacts a Source player is a projectile: vertical velocity falls by
exactly the gravity constant every tick. A tick whose vertical acceleration
DEPARTS from that is a tick where geometry acted - the player was on a ramp.
The failure here is a fall into an empty pit, so the last such tick is the end
of the part of the track that physically exists for this policy. It needs no
champion, no route, no goal field and no map file - only the recording and the
fact that gravity is constant.

Measured on this checkpoint's 270 recorded episodes, the rule is sharp: the
five best all cut within **25 u** of the same physical point, 1.1-1.3 k units
short of where they leave the ramp, and it ranks the same episode first that
the champion ruler does. On a champion's own finishing episodes it cuts
~4.2 s / 10.9 k units before the finish, which is not an error - that is a
genuine unbroken flight, and it is the reason this rule is offered for
TRUNCATING a failure, not for building a complete route.

    python tools/pick_selfline.py --out runs/xSELF/source.jsonl \\
        "runs/research/xROUTE/traj_*.jsonl" "runs/research/xSP/traj_*.jsonl"

Then the line itself comes from the normal tool, which records the truncation:

    python tools/build_route.py --allow-unfinished \\
        --out maps/<map>.self.route.npz runs/xSELF/source.jsonl

--selftest runs the ranking and the trim on synthetic episodes, no files.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from surfgym.route import episodes_from_traj  # noqa: E402
from surfgym.tick import episode_seconds  # noqa: E402  (the recording's tick)


def path_length(xyz) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(xyz, np.float64), axis=0),
                                axis=1).sum())


def contact_cut(ep, tol: float = 1.0) -> tuple:
    """-> (last tick the map pushed back, the gravity step it was measured
    against), from ``[tick, x, y, z, vx, vy, vz, yaw]`` rows.

    Between contacts ``vz`` falls by exactly one gravity step per tick, so the
    ticks where ``diff(vz)`` departs from that step by more than ``tol`` are
    exactly the ticks where geometry acted. The gravity step is taken as the
    MEDIAN of ``diff(vz)`` rather than read from a config: a recording of a
    policy that ends in a fall is overwhelmingly ballistic, and taking it from
    the data means the rule needs no engine constant and survives a map or
    server that changes it.

    The cut is the contact tick itself, kept: the line should end where the
    track ended, not one tick into the air. An episode with no ballistic ticks
    at all keeps the whole thing, which is the conservative direction.
    """
    ep = np.asarray(ep, np.float64)
    if len(ep) < 3:
        return len(ep) - 1, 0.0
    dvz = np.diff(ep[:, 6])
    g = float(np.median(dvz))
    hit = np.flatnonzero(np.abs(dvz - g) > float(tol))
    return (int(hit[-1]) + 1 if len(hit) else len(ep) - 1), g


def tick_keys(hdr):
    """The tick fields a file written FROM ``hdr`` must carry: ``tick_ms``,
    plus ``tick_pattern_ms`` / ``tick_phase`` under a --tick-ms pattern, so
    the line keeps the time base it was flown at instead of a 10 ms
    assumption. A header with no tick is the 10 ms reference - every
    recording made before the flag ran at it."""
    if not hdr or hdr.get("tick_ms") is None:
        return {"tick_ms": 10}
    out = {"tick_ms": hdr["tick_ms"]}
    pat = hdr.get("tick_pattern_ms")
    if pat:
        out["tick_pattern_ms"] = [int(v) for v in pat]
        out["tick_phase"] = int(hdr.get("tick_phase", 0))
    return out


def write_episode(path, ep, hdr=None, note=""):
    """One episode in the tools/record_ckpt.py .jsonl format: header dict,
    ``[tick, x, y, z, vx, vy, vz, yaw]`` rows, footer dict. ``hdr`` is the
    SOURCE episode's header, whose tick fields are copied through
    (:func:`tick_keys`); None writes the 10 ms reference."""
    ep = np.asarray(ep, np.float64)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"episode": 0, **tick_keys(hdr),
                             "source": note}) + "\n")
        for row in ep:
            fh.write("[" + ",".join(f"{v:.6g}" for v in row[:8]) + "]\n")
        fh.write(json.dumps({"ticks": int(len(ep)),
                             "path": round(path_length(ep[:, 1:4]), 1)}) + "\n")
    return p


def synth(n=400, last_contact=300, g=-8.0, bounce_every=50):
    """A synthetic episode: periodic surface contacts up to ``last_contact``,
    pure free fall afterwards. Rows are the recorder's 8 columns."""
    t = np.arange(n, dtype=np.float64)
    vz = np.empty(n)
    v = 0.0
    for k in range(n):
        vz[k] = v
        v += g
        if k < last_contact and (k + 1) % bounce_every == 0:
            v = 0.0                                     # the ramp pushed back
    z = np.concatenate(([0.0], np.cumsum(vz[:-1])))
    x = t * 10.0
    y = np.zeros(n)
    return np.stack([t, x, y, z, np.full(n, 10.0), y, vz, y], 1)


def selftest():
    """Ranking and trimming on synthetic episodes - no files, no map."""
    ep = synth()
    cut, g = contact_cut(ep)
    assert abs(g - (-8.0)) < 1e-9, g            # gravity recovered from data
    assert cut == 300, cut                      # the last contact, kept
    # an episode that keeps touching down is cut at its LAST touchdown, which
    # for a 400-tick episode bouncing every 50 ticks is tick 350
    assert contact_cut(synth(last_contact=400))[0] == 350
    # an episode that is ballistic throughout is not trimmed at all
    assert contact_cut(synth(last_contact=0))[0] == 399
    # a longer tail does not move the cut
    assert contact_cut(synth(n=900))[0] == 300
    # degenerate inputs do not raise
    assert contact_cut(np.zeros((1, 8)))[0] == 0
    # path length is monotone in the prefix
    L = [path_length(ep[:k, 1:4]) for k in (10, 100, 400)]
    assert L[0] < L[1] < L[2], L
    # the writer round-trips through the reader build_route.py uses
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = write_episode(Path(td) / "e.jsonl", ep[:cut + 1], note="x")
        back = episodes_from_traj(f)
        assert len(back) == 1, len(back)
        assert back[0].shape == (cut + 1, 8), back[0].shape
        assert np.allclose(back[0][:, 1:4], ep[:cut + 1, 1:4], rtol=1e-5)
    # the ranking the CLI uses inverts the raw one: an episode that leaves the
    # track early but then falls a long way has the LONGER raw path and the
    # SHORTER trimmed path, which is the whole point of trimming before ranking
    early = synth(n=900, last_contact=100)
    assert path_length(early[:, 1:4]) > path_length(ep[:, 1:4])
    ce = contact_cut(early)[0]
    assert ce == 100 and path_length(early[:ce + 1, 1:4]) \
        < path_length(ep[:cut + 1, 1:4])
    print("selftest OK: 11 checks")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", nargs="*", help="trajectory .jsonl files or globs")
    ap.add_argument("--out", default=None, help="output one-episode .jsonl")
    ap.add_argument("--contact-tol", type=float, default=1.0,
                    help="how far diff(vz) may sit from the gravity step and "
                         "still count as free fall, in map units per tick")
    ap.add_argument("--no-trim", action="store_true",
                    help="rank and write the raw episode. Measured on this "
                         "map: the checkpoint's own FALLERS then reach 99.3%% "
                         "of the line while episodes that FINISH reach 96.9%%, "
                         "i.e. the line pays for falling")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.traj or not a.out:
        ap.error("need trajectory files plus --out (or --selftest)")

    files = []
    for pat in a.traj:
        files.extend(sorted(glob.glob(pat))
                     or ([pat] if Path(pat).exists() else []))
    if not files:
        raise SystemExit("no trajectory files matched")
    eps = []
    for f in files:
        e_list, h_list = episodes_from_traj(f, with_headers=True)
        for i, (e, h) in enumerate(zip(e_list, h_list)):
            eps.append((Path(f).name, i, e, h))
    if not eps:
        raise SystemExit("no episodes found")
    raw = np.array([path_length(e[:, 1:4]) for _, _, e, _ in eps])
    print(f"{len(eps)} episodes in {len(files)} file(s); "
          f"raw path length {raw.min():,.0f}..{raw.max():,.0f}u")

    # EVERY episode is trimmed first and ranked on what survives: an episode
    # that leaves the track early and then falls a long way has the longest
    # raw path and one of the shortest trimmed ones.
    best, best_len, rows, grav = None, -1.0, [], []
    for j, (name, i, e, h) in enumerate(eps):
        cut, g = ((len(e) - 1, 0.0) if a.no_trim
                  else contact_cut(e, a.contact_tol))
        grav.append(g)
        L = path_length(e[:cut + 1, 1:4])
        rows.append((L, name, i, cut, len(e) - 1, raw[j]))
        if L > best_len:
            best, best_len = (name, i, e, cut, h), L
    rows.sort(reverse=True)
    print(f"  gravity step recovered from diff(vz): median {np.median(grav):g}, "
          f"spread {np.min(grav):g}..{np.max(grav):g} u/tick^2")
    print(f"\nall {len(rows)} episodes ranked by TRIMMED path length "
          f"({'NO TRIM' if a.no_trim else 'last surface contact'}):")
    print(f"  {'trimmed':>10s} {'raw':>10s} {'cut':>6s} {'ticks':>6s}  file / ep")
    for L, name, i, cut, n, r0 in rows[:10]:
        print(f"  {L:10,.0f} {r0:10,.0f} {cut:6d} {n:6d}  {name} ep {i}")

    name, i, e, cut, hdr = best
    trimmed = e[:cut + 1]
    note = (f"{name} ep {i}" if a.no_trim else
            f"{name} ep {i} trimmed at tick {cut} of {len(e) - 1}, the last "
            f"tick whose vertical acceleration departs from free fall")
    out = write_episode(a.out, trimmed, hdr=hdr, note=note)
    # the dropped tail in SECONDS at the recording's own tick: the exact
    # per-tick sum between the cut and the end (a 0.01 s literal read a
    # 7.667 ms pattern 30% long), and the written line keeps that tick
    what = f"{name} ep {i}"
    tail_s = (episode_seconds(hdr, len(e) - 1, what)
              - episode_seconds(hdr, cut, what))
    print(f"\nchosen: {note}")
    print(f"  raw path {path_length(e[:, 1:4]):,.0f}u -> trimmed "
          f"{best_len:,.0f}u (dropped {path_length(e[:, 1:4]) - best_len:,.0f}u"
          f", {tail_s:.2f}s of tail)")
    print(f"  start {np.round(trimmed[0, 1:4], 1).tolist()}  "
          f"end {np.round(trimmed[-1, 1:4], 1).tolist()}")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
