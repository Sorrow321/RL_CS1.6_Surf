"""pick_dfloor.py - measure `--race-dfloor` from a policy's OWN recordings.

    python tools/pick_dfloor.py runs/research/x*/traj_*.jsonl \
        --field maps/surf_src_cannonball.goal_32.npz

The number this prints is the potential floor for `--race-dfloor`:

    d_floor = min over the policy's episodes of  d( last tick the map pushed
                                                    back )

i.e. **the closest to the finish this policy has ever been while it was still
surfing rather than falling**, measured in the geodesic field's own units.

Why that and not the plain minimum of `d`. Every episode of a non-finishing
policy ends in a fall, and on this map the field's low-`d` shell reaches down
into lethal airspace - the voxel BFS believes the player can glide ~8,700 u
level across open air from route vertex 1600 - so the plain minimum is a
point *inside the fall* (2,303 u on surf_src_cannonball) and a floor there
would clamp nothing that matters. The last contact separates the two: between
surface contacts a Source player is a projectile and `vz` falls by exactly one
gravity step per tick, so the ticks where `diff(vz)` departs from that step are
exactly the ticks where geometry acted. Same criterion, and the same eight
lines, as `tools/pick_selfline.py` uses to trim a self-built reference line -
which is the point: the floor lands on the same state that trim lands on.

Champion-free by construction. The inputs are the policy's own trajectory
recordings, the map's own distance field, and the constancy of gravity. No
route, no demo, no finisher, no human line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.goalfield import GoalField            # noqa: E402
from surfgym.route import episodes_from_traj       # noqa: E402


def contact_cut(ep: np.ndarray, tol: float = 1.0) -> int:
    """Index of the last tick the map pushed back, from
    ``[tick, x, y, z, vx, vy, vz, yaw]`` rows.

    The gravity step is the MEDIAN of ``diff(vz)`` rather than a config
    constant: a recording is enough to recover it, which keeps the rule
    portable to a map or a build whose gravity differs.
    """
    dvz = np.diff(ep[:, 6])
    g = float(np.median(dvz))
    hit = np.flatnonzero(np.abs(dvz - g) > float(tol))
    # dvz[i] spans samples i -> i+1, so the state AT the contact is i+1; the
    # contact tick is kept, exactly as tools/pick_selfline.py keeps it
    return int(hit[-1]) + 1 if len(hit) else len(ep) - 1


def load_field(path: Path) -> GoalField:
    z = np.load(path)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]),
                     z["mins"], float(z["cell"]), float(z["reach_max"]))


def measure(files, field, tol: float = 1.0):
    """-> (per-episode d at last contact, per-episode raw min d)."""
    cut_d, raw_d = [], []
    for p in files:
        for ep in episodes_from_traj(p):
            d = field.sample(ep[:, 1:4])
            if not np.isfinite(d).any():
                continue
            cut_d.append(float(d[contact_cut(ep, tol)]))
            raw_d.append(float(np.nanmin(d)))
    return np.asarray(cut_d), np.asarray(raw_d)


def _selftest() -> None:
    """Gravity recovered from the data, and the cut is the last contact."""
    def synth(n=400, last_contact=300, g=-8.0, bounce_every=50):
        t = np.arange(n, dtype=np.float64)
        vz = np.empty(n)
        v = 0.0
        for k in range(n):
            vz[k] = v
            if k < last_contact and (k + 1) % bounce_every == 0:
                v = 0.0                              # the ramp pushed back
            else:
                v += g
        z = np.concatenate(([0.0], np.cumsum(vz[:-1])))
        y = np.zeros(n)
        return np.stack([t, t, y, z, np.full(n, 10.0), y, vz, y], 1)

    assert contact_cut(synth()) == 300, contact_cut(synth())
    assert contact_cut(synth(last_contact=400)) == 350
    assert contact_cut(synth(last_contact=0)) == 399     # never in contact
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", nargs="*", help="record_rollout .jsonl files")
    ap.add_argument("--field", default="maps/surf_src_cannonball.goal_32.npz")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="u/tick of vz departure that counts as contact")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    files = [Path(f) for f in a.traj]
    if not files:
        raise SystemExit("no trajectory files given")
    fp = Path(a.field)
    if not fp.exists():
        fp = ROOT / a.field
    field = load_field(fp)
    cut, raw = measure(files, field, a.tol)
    if not len(cut):
        raise SystemExit("no usable episodes")
    print(f"episodes            {len(cut)}")
    print(f"raw min d ever      {raw.min():10,.1f}   "
          f"(inside the fall - NOT the floor)")
    print(f"d at last contact   min {cut.min():10,.1f}   "
          f"p5 {np.percentile(cut, 5):,.1f}   "
          f"median {np.median(cut):,.1f}")
    print(f"\n--race-dfloor {cut.min():.0f}")


if __name__ == "__main__":
    main()
