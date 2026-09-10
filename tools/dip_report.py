#!/usr/bin/env python3
"""dip_report.py - the potential-vs-time trace, and the DIPS in it.

The user's framing (2026-09-10): "draw a plot where on X you have time and on
Y the potential; follow the agent trajectory and at each time write down the
potential at his point. In a hard moment you see dips down. Compare
cannonball's dips along the part the agent passes properly, the dip on the
finisher line at the end, and petrus (winner/loser)."

So this walks a recorded trajectory, samples the run's OWN geodesic goal
field at every DECISION (rows subsampled by ``act_every`` - the reward is
paid per decision, not per physics tick), and reports two things:

  * the CURVE, in two normalisations:
        raw     d0 - d          map units of geodesic ground banked
        reward  (d0 - d) * 100/d0    reward units; the whole map is worth
                100 on ANY map, so cannonball (d0 198,380) and petrus
                (d0 35,637) are directly comparable. This is the axis the
                agent actually optimises.
    Progress is UP and a setback is a DIP, by construction.

  * the DIPS, under the definition shared with the online ``dip/`` metric
    (python/surfgym/dipmeter.py - this file IMPORTS that enumerator, so the
    offline plot and the online metric cannot drift):
        b_t         running MINIMUM of d inside the episode, reset at every
                    episode start
        depth_t     (d_t - b_t) * scale,  scale = 100/d0, reward units, >= 0
        a DIP       a maximal run of depth > 0
        SURVIVED    depth returns to exactly 0 (a new record is set)
        FAILED      the episode ends while depth > 0

Two derived numbers are the point of the exercise:

  * the largest dip a policy has demonstrably SURVIVED anywhere on its map -
    the tolerance it currently has;
  * the depth and duration of the dip it FAILS on - the tolerance the map
    is asking for.

Each dip also carries ``gae_w`` = (gamma^act_every * lambda)^k at k = the
dip's duration in decisions: the share of a payoff arriving at the far end
of the dip that survives the trainer's own GAE weighting. That is what turns
"how big a dip" into "how big a dip can the optimiser see across".

Nothing here needs a GPU or the simulator - it reads the baked goal field
next to the .bsp and the recorded positions. Always pass an ABSOLUTE
main-checkout map path (CLAUDE.md's worktree bake trap); this tool only ever
np.loads an existing cache and never calls build_goal_field, so it cannot
trigger a bake, but the cache must be the one the run trained against.

    python tools/dip_report.py --spec tools/dip_specs/round37.json \
        --out runs/research/cornerdiag/dips
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.goalfield import GoalField  # noqa: E402
from surfgym.dipmeter import enumerate_dips as _enumerate_dips  # noqa: E402
from surfgym.dipmeter import dip_depths as _dip_depths          # noqa: E402


# ------------------------------------------------------------------ io

def load_field(bsp: Path, cell: int = 32) -> GoalField:
    """The trainer's own GoalField, straight out of the baked cache.

    Never calls build_goal_field, so it cannot bake. Raises if the cache is
    missing rather than silently building one."""
    f = Path(bsp).with_suffix("")
    p = Path(f"{f}.goal_{cell:g}.npz")
    if not p.exists():
        raise SystemExit(f"no baked field {p} - refusing to bake")
    z = np.load(p, allow_pickle=False)
    grid = z["grid"].astype(np.float32) * float(z["quant"])
    return GoalField(grid, z["mins"], float(z["cell"]), float(z["reach_max"]))


def load_traj(path):
    """-> [(footer, header, rows ndarray)] for every episode in the file."""
    eps, rows, hdr = [], [], None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if isinstance(r, dict) and "map" in r:
                hdr, rows = r, []
            elif isinstance(r, list):
                rows.append(r)
            elif isinstance(r, dict) and "end" in r:
                if rows:
                    eps.append((r, hdr, np.asarray(rows, float)))
                rows = []
    if rows:
        eps.append(({"end": "open", "ticks": len(rows)}, hdr,
                    np.asarray(rows, float)))
    return eps


# ------------------------------------------------------- the definition

def enumerate_dips(d, scale, dt, ended_in_episode=True):
    """DIPS in a per-decision geodesic series, in this tool's dict form.

    The DEFINITION is not here: it is ``surfgym.dipmeter.enumerate_dips``,
    the same function the trainer's online ``dip/*`` metric accumulates
    with, so the offline plot and the online diagnostic cannot drift. This
    wrapper only re-shapes its ``(i0, i1, depth, secs, survived)`` tuples
    into the dicts the report and the figures carry, and returns the
    instantaneous depth series alongside.
    """
    d = np.asarray(d, np.float64)
    depth = np.asarray(_dip_depths(d, float(scale)), np.float64)
    dips = []
    for i0, i1, dep, secs, survived in _enumerate_dips(
            d, float(scale), float(dt), ended_in_dip=bool(ended_in_episode)):
        seg = depth[i0:i1]
        dips.append(dict(
            i0=int(i0), i1=int(i1),               # [i0, i1)
            depth=float(dep), secs=float(secs),
            decisions=int(i1 - i0),
            survived=bool(survived),
            d_at_start=float(d[i0]),
            arg_deepest=int(i0 + int(np.argmax(seg))),
        ))
    return dips, depth


def gae_weight(secs, gamma_tick, act_every, lam, tick_ms):
    """(gamma^act_every * lam)^k for k = the dip's length in decisions - the
    direct GAE weight a TD residual that far ahead gets (docs/credit_diag.md).
    """
    k = secs / (act_every * tick_ms / 1000.0)
    g = (float(gamma_tick) ** int(act_every)) * float(lam)
    return float(g ** k)


def discount_weight(secs, gamma_tick, tick_ms):
    """gamma^(ticks) - the pure discount over the same delay."""
    return float(float(gamma_tick) ** (secs * 1000.0 / float(tick_ms)))


# ------------------------------------------------------------- driver

def trace(name, bsp, traj, episode, d0, act_every, tick_ms, cell=32,
          gamma_tick=0.9995, lam=0.95, time_pen_tick=0.005, finished=None,
          trim_finish=False):
    field = load_field(Path(bsp), cell)
    eps = load_traj(traj)
    if episode == "all":
        picks = list(range(len(eps)))
    elif episode == "best":
        # deepest = smallest geodesic minimum reached
        mins = []
        for _f, _h, a in eps:
            mins.append(float(field.sample(a[:, 1:4]).min()))
        picks = [int(np.argmin(mins))]
    elif episode == "longest":
        picks = [int(np.argmax([len(a) for _f, _h, a in eps]))]
    else:
        picks = [int(episode)]

    scale = 100.0 / float(d0)
    out = []
    for k in picks:
        foot, hdr, a = eps[k]
        tm = float(hdr.get("tick_ms", tick_ms)) if hdr else tick_ms
        sub = a[::act_every]
        d = field.sample(sub[:, 1:4]).astype(np.float64)
        if trim_finish and (d <= 150.0).any():
            # a RECORDING can keep running after the line (the human demo
            # walks around the finish room), and the rise afterwards is not
            # a dip the racer ever paid. Cut at the finish, using the same
            # d <= 150 test train_fast.eval_finish_times uses.
            k_end = int(np.argmax(d <= 150.0)) + 1
            sub, d = sub[:k_end], d[:k_end]
        dt = act_every * tm / 1000.0
        t = np.arange(len(d)) * dt
        ended = str(foot.get("end", "")) in ("fail", "done", "trunc")
        dips, depth = enumerate_dips(d, scale, dt, ended_in_episode=ended)
        for dp in dips:
            dp["gae_w"] = gae_weight(dp["secs"], gamma_tick, act_every, lam, tm)
            dp["disc_w"] = discount_weight(dp["secs"], gamma_tick, tm)
            dp["t0"] = float(dp["i0"] * dt)
            dp["t1"] = float(dp["i1"] * dt)
            dp["depth_pct_d0"] = dp["depth"]      # reward units == % of d0
            dp["depth_units"] = dp["depth"] / scale
        spd = np.linalg.norm(sub[:, 4:7], axis=1)
        out.append(dict(
            name=name, episode=int(k), end=str(foot.get("end", "")),
            ticks=int(foot.get("ticks", len(a))),
            secs=float(len(a) * tm / 1000.0),
            tick_ms=tm, act_every=int(act_every), d0=float(d0),
            scale=scale, gamma_tick=float(gamma_tick), gae_lambda=float(lam),
            time_pen_tick=float(time_pen_tick),
            t=t.tolist(), d=d.tolist(),
            banked_reward=((float(d0) - d) * scale).tolist(),
            depth=depth.tolist(),
            speed=spd.tolist(),
            xyz=sub[:, 1:4].tolist(),
            dips=dips,
            d_min=float(d.min()), d_start=float(d[0]), d_end=float(d[-1]),
            finished=finished,
        ))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True,
                    help="JSON list of curve specs (see the round37 example)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    specs = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    allc = []
    for s in specs:
        got = trace(**s)
        allc.extend(got)
        for c in got:
            n_s = sum(1 for x in c["dips"] if x["survived"] is True)
            surv = [x["depth"] for x in c["dips"] if x["survived"] is True]
            fail = [x for x in c["dips"] if x["survived"] is False]
            print(f"{c['name']:24s} ep{c['episode']:<2d} {c['end']:5s} "
                  f"{c['secs']:6.2f}s  dips {len(c['dips']):3d} "
                  f"(survived {n_s:3d})  max survived depth "
                  f"{(max(surv) if surv else 0.0):7.3f}  "
                  f"failed depth "
                  f"{(fail[-1]['depth'] if fail else 0.0):7.3f} over "
                  f"{(fail[-1]['secs'] if fail else 0.0):5.2f}s")
    (outdir / "curves.json").write_text(json.dumps(allc), encoding="utf-8")
    print(f"\nwrote {outdir / 'curves.json'}  ({len(allc)} curves)")


if __name__ == "__main__":
    main()
