"""make_physics_codebook.py - a behavior-chunk codebook CONSTRUCTED from
engine math, zero trajectories, zero learned information.

For a genuinely from-scratch chunked run: every code is a hand-derived
primitive over the yaw-adaptive action space, where yaw index k means
"k x the analytic optimal-strafe turn rate arctan(30/|v|)" (src/env.c
K_BINS; the air-accel law from docs/research-litsurvey.md section 5).
Behaviors, not statistics:

  - hold-strafe: k in {0, +-0.25, +-0.5, +-1, +-2} with the matching
    strafe key (A with left, D with right - wishdir geometry, not data),
    fwd neutral. k=+-1 is "surf this ramp optimally".
  - drive: forward held, side neutral, k in {0, +-0.5}.
  - jump variants: the same primitives with jump pressed on the first
    decision only (takeoff), or held (bhop-style).
  - coast: everything neutral.

Output matches build_action_codebook.py's npz schema so the trainer
integration loads either interchangeably. Occupancy is uniform (no data
to estimate a prior from - that is the point).

    python tools/make_physics_codebook.py --out runs/PHYS_H10.npz
"""
import argparse

import numpy as np

# head index conventions (python/surfgym/core.py, src/env.c)
NVEC = (15, 7, 3, 3, 2, 2)            # yaw, pitch, fwd, side, jump, duck
YAW_BINS = [-10.0, -7.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0,
            0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0]   # = K_BINS under adaptive
PITCH_NEUTRAL = 3
FWD = {"back": 0, "none": 1, "fwd": 2}
SIDE = {"left": 0, "none": 1, "right": 2}


def yaw_idx(k):
    return YAW_BINS.index(k)


def code(h, yaw_k, side, fwd="none", jump="none"):
    """One (h, 6) chunk holding a primitive; jump: none|tap|hold."""
    c = np.zeros((h, 6), np.int8)
    c[:, 0] = yaw_idx(yaw_k)
    c[:, 1] = PITCH_NEUTRAL
    c[:, 2] = FWD[fwd]
    c[:, 3] = SIDE[side]
    if jump == "tap":
        c[0, 4] = 1
    elif jump == "hold":
        c[:, 4] = 1
    return c


def build(h):
    codes = []
    strafe_ks = [0.25, 0.5, 1.0, 2.0]
    # hold-strafe left/right at each commitment level, +-jump variants
    for k in strafe_ks:
        for jump in ("none", "tap"):
            codes.append(code(h, -k, "left", jump=jump))
            codes.append(code(h, +k, "right", jump=jump))
    # counter-strafe (turn against the held key - exit/entry corrections)
    for k in (0.5, 1.0):
        codes.append(code(h, +k, "left"))
        codes.append(code(h, -k, "right"))
    # straight-line: coast / drive / drive+tap / bhop hold
    for jump in ("none", "tap", "hold"):
        codes.append(code(h, 0.0, "none", jump=jump))
        codes.append(code(h, 0.0, "none", "fwd", jump=jump))
    # gentle steering while driving
    for k in (0.25, 0.5):
        codes.append(code(h, -k, "none", "fwd"))
        codes.append(code(h, +k, "none", "fwd"))
    # pure side-hold without turning (ramp-riding at fixed heading)
    codes.append(code(h, 0.0, "left"))
    codes.append(code(h, 0.0, "right"))
    # half-chunk switches: carve left then right (and mirror) - S-turns
    for k in (0.5, 1.0):
        c = np.concatenate([code(h - h // 2, -k, "left"),
                            code(h // 2, +k, "right")])
        codes.append(c)
        codes.append(np.concatenate([code(h - h // 2, +k, "right"),
                                     code(h // 2, -k, "left")]))
    # takeoff-then-carve: tap jump, then commit to a hard strafe
    for k in (1.0, 2.0):
        c = code(h, -k, "left")
        c[0, 4] = 1
        codes.append(c)
        c = code(h, +k, "right")
        c[0, 4] = 1
        codes.append(c)
    return np.stack(codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--out", default="runs/PHYS_H10.npz")
    args = ap.parse_args()
    cb = build(args.horizon)
    k, h, _ = cb.shape
    cent = cb.reshape(k, -1).astype(np.float32)
    np.savez(args.out,
             codebook=cb, centroids=cent,
             occupancy=np.full(k, 1, np.int64),        # uniform prior
             intra_var=np.zeros(k, np.float32),
             horizon=np.int64(h), k=np.int64(k),
             nvec=np.array(NVEC, np.int64),
             yaw_bins=np.array(YAW_BINS, np.float64),
             pitch_bins=np.array([-10, -5, -2, 0, 2, 5, 10], np.float64),
             yaw_encode="deg", weights=np.ones(6, np.float32),
             quant_mse=np.float32(0.0), total_var=np.float32(0.0),
             n_windows=np.int64(0),
             source="CONSTRUCTED from engine math - no trajectories")
    print(f"physics codebook: {k} codes x {h} decisions -> {args.out}")


if __name__ == "__main__":
    main()
