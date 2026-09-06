"""view.py - the CONTINUOUS view command (--view-continuous), numpy side.

The discrete policy steers with two categorical heads: 15 yaw bins and 7
pitch bins (surfcore.h). Bins are separate classes to a softmax - "+0.5 is
the same distance from +0.6 as it is from -10", every neighbourhood has to
be learned - and the human record's turn rate falls between two of our bins
42% of the time (tools/demo/wr_scan.py). Under the flag the two heads are
squashed Gaussians and the core receives a float command per env
(``SurfCore.step(acts, view=...)`` -> ``surf_step_view``):

    view[:, 0] = K       the yaw command. Under --yaw-adaptive the core turns
                         the view by K * atan(30/|v_h|) deg per tick - the
                         same formula as the bins, whose table K_BINS is now
                         a continuous number; with fixed bins K is in units
                         where +-20 is the outermost bin (K/2 deg at the
                         reference rate). Clamped in the core exactly where
                         the bins are.
    view[:, 1] = pitch   deg per tick, clamped to the outermost pitch bin.

THE POLICY'S PARAMETERISATION (train_fast.py). Each head draws a pre-tanh
z ~ N(mu(s), sigma) with a state-independent sigma, squashes it, u =
tanh(z), and the env applies a deterministic map of u:

    K      = warp(u) = 20 * sign(u) * (exp(alpha |u|) - 1) / (exp(alpha) - 1)
    pitch  = u * pitch_rate_max_deg

with alpha = 2 ln 19 = 5.889, so that u = +-0.5 -> K = +-1 (the analytic
optimal-strafe multiple, the single most useful command at every speed) and
u = +-1 -> K = +-20 (the old outermost bin). Resolution is fine near zero -
dK/du = 0.33 at u = 0 - and coarse at the extremes, where the bins were
coarse too. z is THE action PPO scores: the log-density of z under the
Gaussian, no Jacobian (the squash and the warp are part of the environment's
deterministic response, like the bin table was).

This module holds the numpy half (targets for the BC rows, the transplant,
the planner's records); the torch half lives next to the Policy in
train_fast.py and must agree with it - tests/python/test_view_continuous.py
pins the two against each other.
"""
from __future__ import annotations

import numpy as np

from .core import PITCH_BINS, YAW_BINS

__all__ = ["K_BINS", "K_MAX", "WARP_ALPHA", "U_CLIP", "LOG_STD_INIT",
           "warp", "warp_inv", "z_from_u", "u_from_z",
           "bin_to_view", "view_from_z", "z_from_view",
           "bin_view_moments"]

#: Mirrors src/env.c K_BINS exactly (the C table is the authority): a yaw
#: bin under --yaw-adaptive is this multiple of atan(30/|v_h|) per tick.
K_BINS = np.array([-20.0, -8.0, -3.0, -1.5, -1.0, -0.75, -0.5, 0.0,
                   0.5, 0.75, 1.0, 1.5, 3.0, 8.0, 20.0], dtype=np.float32)
K_MAX = 20.0                       # warp(+-1): the old outermost bin
WARP_ALPHA = 2.0 * np.log(19.0)    # warp(+-0.5) = +-1 exactly
#: |u| is clipped here before atanh when a TARGET is built from a physical
#: value (a bin at the ceiling maps to u = 1, whose atanh is infinite)
U_CLIP = 0.999
#: sigma of the pre-tanh Gaussian at init (log 0.3); the transplant
#: overrides it from the categorical spread
LOG_STD_INIT = float(np.log(0.3))


def warp(u):
    """u in [-1, 1] -> K in [-20, 20]. Odd, monotone, warp(0.5) = 1."""
    u = np.asarray(u, np.float64)
    a = WARP_ALPHA
    return (K_MAX * np.sign(u) * np.expm1(a * np.abs(u)) / np.expm1(a))


def warp_inv(k):
    """K in [-20, 20] -> u in [-1, 1] (the inverse of warp, clipped)."""
    k = np.clip(np.asarray(k, np.float64), -K_MAX, K_MAX)
    a = WARP_ALPHA
    return np.sign(k) * np.log1p(np.abs(k) / K_MAX * np.expm1(a)) / a


def z_from_u(u, clip: float = U_CLIP):
    """u -> pre-tanh z, |u| clipped to `clip` so the target is finite."""
    return np.arctanh(np.clip(np.asarray(u, np.float64), -clip, clip))


def u_from_z(z):
    return np.tanh(np.asarray(z, np.float64))


def bin_to_view(acts, yaw_adaptive: bool, pitch_rate_max_deg: float):
    """(n, 6) discrete actions -> (n, 2) float32 view commands that the
    continuous core path applies BIT-IDENTICALLY to the bins (src/env.c
    surf_yaw_delta_cont / surf_pitch_delta_cont):

    * yaw: K_BINS[b] under --yaw-adaptive, else 2 * YAW_BINS[b] (the core
      halves it and scales like the bins);
    * pitch: PITCH_BINS[b] * (pitch_rate_max_deg / 10), computed in float32
      in the core's own operation order.
    """
    a = np.asarray(acts, np.int64).reshape(-1, 6)
    yb = np.clip(a[:, 0], 0, 14)
    pb = np.clip(a[:, 1], 0, 6)
    if yaw_adaptive:
        k = K_BINS[yb]
    else:
        k = (np.float32(2.0) * YAW_BINS[yb]).astype(np.float32)
    scale = np.float32(pitch_rate_max_deg) / np.float32(10.0)
    pd = (PITCH_BINS[pb] * scale).astype(np.float32)
    out = np.empty((len(a), 2), np.float32)
    out[:, 0] = k
    out[:, 1] = pd
    return out


def view_from_z(z, pitch_rate_max_deg: float):
    """(n, 2) pre-tanh z -> (n, 2) float32 view command (what the env gets):
    K = warp(tanh(z_yaw)), pitch = tanh(z_pitch) * pitch_rate_max_deg."""
    z = np.asarray(z, np.float64).reshape(-1, 2)
    u = np.tanh(z)
    out = np.empty((len(z), 2), np.float32)
    out[:, 0] = warp(u[:, 0])
    out[:, 1] = u[:, 1] * float(pitch_rate_max_deg)
    return out


def z_from_view(view, pitch_rate_max_deg: float, clip: float = U_CLIP):
    """(n, 2) physical view (K, pitch deg/tick) -> (n, 2) float32 target z:
    z_yaw = atanh(clip(warp_inv(K))), z_pitch = atanh(clip(pitch / rate)).
    A frozen gaze (rate 0) gets z_pitch = 0."""
    v = np.asarray(view, np.float64).reshape(-1, 2)
    zy = z_from_u(warp_inv(v[:, 0]), clip)
    rate = float(pitch_rate_max_deg)
    zp = (z_from_u(v[:, 1] / rate, clip) if rate > 0.0
          else np.zeros(len(v), np.float64))
    return np.stack([zy, zp], 1).astype(np.float32)


def bin_view_moments(probs, yaw_adaptive: bool, pitch_rate_max_deg: float,
                     clip: float = U_CLIP):
    """Per-head categorical distributions over the yaw / pitch BINS
    ((n, 6, NPAD) probs, HeadPacker layout) -> the mean and std of the
    bins' target z under those distributions: (n, 2) mu, (n, 2) std, float32.
    The moment-matched Gaussian target a --bc-target dist row uses when the
    file carries a bin distribution and no per-copy z (and what the
    transplant uses to seed log_std)."""
    p = np.asarray(probs, np.float64)
    n = len(p)
    zy_bins = z_from_u(warp_inv(K_BINS if yaw_adaptive
                                else 2.0 * YAW_BINS.astype(np.float64)), clip)
    rate = float(pitch_rate_max_deg)
    zp_bins = (z_from_u(PITCH_BINS.astype(np.float64) / 10.0, clip)
               if rate > 0.0 else np.zeros(7))
    py = p[:, 0, :15]
    pp = p[:, 1, :7]
    py = py / np.maximum(py.sum(-1, keepdims=True), 1e-12)
    pp = pp / np.maximum(pp.sum(-1, keepdims=True), 1e-12)
    mu = np.empty((n, 2))
    sd = np.empty((n, 2))
    mu[:, 0] = (py * zy_bins[None, :]).sum(-1)
    sd[:, 0] = np.sqrt(np.maximum((py * (zy_bins[None, :] - mu[:, :1]) ** 2)
                                  .sum(-1), 0.0))
    mu[:, 1] = (pp * zp_bins[None, :]).sum(-1)
    sd[:, 1] = np.sqrt(np.maximum((pp * (zp_bins[None, :] - mu[:, 1:2]) ** 2)
                                  .sum(-1), 0.0))
    return mu.astype(np.float32), sd.astype(np.float32)
