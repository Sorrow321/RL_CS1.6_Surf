"""Build a discrete BEHAVIOR-CHUNK codebook from recorded trajectories.

VQ-BeT / BeT-style (arXiv 2206.11251, 2403.03181): cluster fixed-length
windows of H consecutive DECISIONS into K codes with k-means, then let a
high-level policy emit one index per H decisions and a FROZEN lookup
decoder expand it back into H x 6 primitive actions.  This script builds
that decoder.  See docs/action-chunks-design.md.

    # survey what the corpus holds, no clustering
    python tools/build_action_codebook.py --audit

    # the real thing
    python tools/build_action_codebook.py \
        --runs "C:/RL_Surf/runs/race_cannonball" --k 128 --horizon 10 \
        --out codebook_H10_K128.npz

    # correctness
    python tools/build_action_codebook.py --selftest


WHY THIS WORKS WITHOUT A GPU ROLLOUT DUMP
-----------------------------------------
The trainer never writes an (obs, action) tape, but the recorder does write
a trajectory whose 15 columns are ENOUGH to recover every action exactly.
Per surfgym/record.py the row is

    [t, x,y,z, vx,vy,vz, yaw, buttons, onground, progress, reward, pitch,
     fwd, side]

and line ``t`` holds the PRE-step state s_t paired with the action a_t
applied during tick t.  Four of the six action heads are literally in the
row:

    a[2] fwd   = col 13        a[4] jump = bool(col 8 & SURF_IN_JUMP=2)
    a[3] side  = col 14        a[5] duck = bool(col 8 & SURF_IN_DUCK=4)

The other two are RATES the core integrates into the state before the
physics step (src/env.c:549 ``st->yaw = wrap_yaw(st->yaw + yd)``,
src/env.c:551 ``st->pitch += pd``), so

    a[0] yaw   = argmin_b | wrap180(yaw[t+1] - yaw[t]) - yaw_delta(b) |
    a[1] pitch = argmin_b | pitch[t+1] - pitch[t]      - PITCH_BINS[b]*s |

is an EXACT inversion, not an estimate: the bin ladder is a set of 15
distinct values and the observed delta lands on one of them.  These are
derived actions, not pseudo-actions.  ``--audit`` prints the snap residual
so the claim is checked on every file rather than assumed; see also the
champion-trajectory reconstruction check under --validate.

Known non-invertible cases, all detected and masked rather than guessed:
  * pitch saturation at the +30 / -70 clamp (src/env.c:556) censors dp;
  * --yaw-adaptive runs use a velocity-dependent ladder (src/env.c:69),
    handled by rebuilding the 15 candidate deltas per tick from (vx,vy) --
    but in the |yd| > yaw_rate_max clip region several k collapse to the
    same delta, so those ticks are ambiguous and get masked;
  * the last tick of an episode has no successor row.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---- action semantics, mirrored from src/env.c:46-64 / surfgym.core --------
# Must match; a divergence here silently mislabels every chunk.
YAW_BINS = np.array([-10.0, -7.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0,
                     0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0], np.float64)
K_BINS = np.array([-20.0, -8.0, -3.0, -1.5, -1.0, -0.75, -0.5, 0.0,
                   0.5, 0.75, 1.0, 1.5, 3.0, 8.0, 20.0], np.float64)   # yaw_adaptive
PITCH_BINS = np.array([-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0], np.float64)
NVEC = (15, 7, 3, 3, 2, 2)          # yaw, pitch, fwd, side, jump, duck
YAW_ZERO, PITCH_ZERO = 7, 3         # index of the 0-deg bin in each ladder
SURF_IN_JUMP, SURF_IN_DUCK = 2, 4

# jsonl column indices (surfgym/record.py)
C_T, C_X, C_Y, C_Z = 0, 1, 2, 3
C_VX, C_VY, C_VZ = 4, 5, 6
C_YAW, C_BTN, C_ONGND, C_PROG, C_REW, C_PITCH, C_FWD, C_SIDE = 7, 8, 9, 10, 11, 12, 13, 14
NCOL = 15

# The air-accel gain window (docs/research-results.md "the action space
# cannot express optimal strafing"): at sv_airaccelerate 100 the only
# speed-increasing wishdir is within +-arcsin(30/|v|) of perpendicular,
# i.e. +-0.52 deg/tick at 3000 u/s and +-3.44 deg/tick at 500 u/s.  A chunk
# whose yaw column sits outside that band for most of its length is a chunk
# that brakes.  Reported per cluster so the codebook can be judged on the
# mechanism, not just on cluster purity.
GAIN_WIN_LO, GAIN_WIN_HI = 0.52, 3.44


def wrap180(d):
    """Signed shortest angular difference; the core wraps yaw to [0,360)."""
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# 1. reading trajectories
# ---------------------------------------------------------------------------

def read_episodes(path, max_rows=None):
    """Yield (header_dict, rows (T,15) float64) per episode in a traj jsonl."""
    hdr, rows = None, []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except ValueError:
                continue                      # a torn last line on a live run
            if isinstance(o, dict):
                if "end" in o:                # trailer: episode closes
                    if rows:
                        yield hdr, np.asarray(rows, np.float64)
                    rows = []
                else:                         # header: episode opens
                    if rows:                  # file truncated mid-episode
                        yield hdr, np.asarray(rows, np.float64)
                    hdr, rows = o, []
                continue
            if len(o) != NCOL:
                continue                      # pre-ABI-6 recording, skip
            rows.append(o)
            if max_rows and len(rows) >= max_rows:
                yield hdr, np.asarray(rows, np.float64)
                rows = []
    if rows:
        yield hdr, np.asarray(rows, np.float64)


def run_config(traj_path):
    """The run.json beside a traj file, or {}. Carries act_every / yaw mode."""
    rj = Path(traj_path).parent / "run.json"
    if not rj.exists():
        return {}
    try:
        d = json.loads(rj.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return d.get("config", d)


# ---------------------------------------------------------------------------
# 2. exact action inversion  (see module docstring)
# ---------------------------------------------------------------------------

def yaw_candidates(rows, yaw_rate=10.0, yaw_adaptive=False):
    """(T-1, 15) per-tick candidate yaw deltas for each yaw bin.

    Stock: YAW_BINS * (yaw_rate/10), constant.  Adaptive (src/env.c:69):
    K_BINS * atan(30/|v_h|) clipped to +-yaw_rate, so it depends on the
    tick's own horizontal speed -- which the row carries in cols 4,5.
    """
    n = len(rows) - 1
    if not yaw_adaptive:
        return np.repeat((YAW_BINS * (yaw_rate / 10.0))[None, :], n, axis=0)
    vh = np.hypot(rows[:n, C_VX], rows[:n, C_VY])
    vh = np.maximum(vh, 1.0)
    w = np.degrees(np.arctan(30.0 / vh))                     # env.c:75
    return np.clip(K_BINS[None, :] * w[:, None], -yaw_rate, yaw_rate)


def derive_actions(rows, yaw_rate=10.0, pitch_rate=10.0, yaw_adaptive=False,
                   tol_yaw=0.02, tol_pitch=0.02):
    """Recover the (T-1, 6) per-tick action indices from one episode's rows.

    Returns (act, valid, resid) where ``valid`` (T-1,) is False on ticks
    whose yaw/pitch bin could not be identified unambiguously, and
    ``resid`` (T-1, 2) holds the yaw and pitch snap residuals in degrees.
    """
    T = len(rows)
    if T < 2:
        z = np.zeros((0, 6), np.int64)
        return z, np.zeros(0, bool), np.zeros((0, 2))
    n = T - 1
    act = np.zeros((n, 6), np.int64)
    valid = np.ones(n, bool)

    dy = wrap180(rows[1:, C_YAW] - rows[:n, C_YAW])
    cand = yaw_candidates(rows, yaw_rate, yaw_adaptive)
    err = np.abs(cand - dy[:, None])
    act[:, 0] = np.argmin(err, axis=1)
    ry = err[np.arange(n), act[:, 0]]
    valid &= ry <= tol_yaw
    if yaw_adaptive:
        # in the clip region many k give the same delta: refuse to name one
        best = cand[np.arange(n), act[:, 0]]
        ties = (np.abs(cand - best[:, None]) <= tol_yaw).sum(axis=1)
        valid &= ties == 1

    dp = rows[1:, C_PITCH] - rows[:n, C_PITCH]
    pc = PITCH_BINS * (pitch_rate / 10.0)
    perr = np.abs(pc[None, :] - dp[:, None])
    act[:, 1] = np.argmin(perr, axis=1)
    rp = perr[np.arange(n), act[:, 1]]
    # the clamp (env.c:556) censors dp at the rails; the row is fine, the
    # PITCH action is simply not observable there
    sat = (rows[1:, C_PITCH] >= 29.99) | (rows[1:, C_PITCH] <= -69.99)
    ok_p = (rp <= max(tol_pitch, 0.011)) | sat
    act[sat, 1] = PITCH_ZERO
    rp = np.where(sat, np.nan, rp)     # a censored tick has no residual
    valid &= ok_p

    act[:, 2] = np.clip(rows[:n, C_FWD], 0, 2).astype(np.int64)
    act[:, 3] = np.clip(rows[:n, C_SIDE], 0, 2).astype(np.int64)
    btn = rows[:n, C_BTN].astype(np.int64)
    act[:, 4] = (btn & SURF_IN_JUMP) != 0
    act[:, 5] = (btn & SURF_IN_DUCK) != 0
    return act, valid, np.stack([ry, rp], axis=1)


DETECT_COLS = (0, 2, 3, 4, 5)     # yaw, fwd, side, jump, duck -- NOT pitch:
# pitch is censored at the +30/-70 clamp (env.c:556), and a block straddling
# the clamp looks non-constant even when the decision was held.


def _block_const_frac(act, valid, K, phase):
    a = act[phase:]
    v = valid[phase:]
    m = (len(a) // K) * K
    if m == 0:
        return 0.0
    blocks = a[:m][:, DETECT_COLS].reshape(-1, K, len(DETECT_COLS))
    use = v[:m].reshape(-1, K).all(axis=1)
    if not use.any():
        return 0.0
    const = (blocks == blocks[:, :1, :]).all(axis=(1, 2))
    return float(const[use].mean())


def detect_act_every(act, valid, candidates=(1, 2, 3, 4, 5, 6, 9)):
    """Largest (K, phase) for which every aligned K-block holds one action.

    The trainer repeats a decision for K physics ticks (train_fast.py:2184
    ``for _j in range(K)``), so a per-tick tape is a per-decision tape
    upsampled K times.  Recovering K is what lets chunks be counted in
    DECISIONS instead of ticks.

    PHASE matters: _TorchPolicyBase.act (train_fast.py:549) counts ticks in
    ``self._tick``, which record_rollout never resets between episodes, so
    episode e starts mid-decision at phase (ticks so far) % K.  Assuming
    phase 0 silently misreads a clean act_every=3 tape as act_every=1 on
    two thirds of the episodes in a multi-episode recording.
    """
    best, best_frac, best_ph = 1, 1.0, 0
    scores = {}
    n = len(act)
    for K in candidates:
        if n < K * 8:
            continue
        fr, ph = max(((_block_const_frac(act, valid, K, p), p)
                      for p in range(K)))
        scores[K] = fr
        if fr >= 0.999 and K > best:
            best, best_frac, best_ph = K, fr, ph
    return best, best_frac, scores, best_ph


def decisions_from_ticks(act, valid, K, phase=0):
    """Downsample a per-tick tape to one row per decision (stride K)."""
    a, v = act[phase:], valid[phase:]
    n = (len(a) // K) * K
    if n == 0:
        return np.zeros((0, 6), np.int64), np.zeros(0, bool)
    d_act = a[:n:K]
    # a decision is usable only if every tick it covers inverted cleanly
    d_valid = v[:n].reshape(-1, K).all(axis=1)
    return d_act, d_valid


def detect_pitch_rate(rows, default=10.0):
    """Recover cfg.pitch_rate_max_deg from the pitch column alone.

    pitch_rate is a per-RUN flag (train_fast.py:842 --pitch-rate) that scales
    PITCH_BINS, and run.json is not always next to a trajectory (the
    champion recording in the scratchpad has none).  The observed |dp| set
    is {0, .2, .5, 1} x rate/... , so each observed magnitude divided by a
    bin magnitude is a candidate rate; score them all by snap residual.
    """
    if len(rows) < 3:
        return default
    p = rows[:, C_PITCH]
    dp = np.diff(p)
    sat = (p[1:] >= 29.99) | (p[1:] <= -69.99)
    dp = dp[~sat]
    nz = np.abs(dp[np.abs(dp) > 5e-3])
    if len(nz) < 8:
        return default
    obs = np.unique(np.round(nz, 2))
    cands = sorted({round(float(v) * 10.0 / b, 4)
                    for v in obs for b in (10.0, 5.0, 2.0)
                    if 0.01 <= v * 10.0 / b <= 200.0})
    best, best_err = default, np.inf
    for r in cands:
        e = np.abs(PITCH_BINS[None, :] * (r / 10.0) - dp[:, None]).min(1)
        # tolerance is the recorder's 2-dp rounding on both endpoints
        err = float((e > 0.011).mean())
        if err < best_err - 1e-9 or (abs(err - best_err) < 1e-9 and r < best):
            best, best_err = r, err
    return best if best_err < 0.02 else default


# ---------------------------------------------------------------------------
# 3. windows and the clustering metric
# ---------------------------------------------------------------------------

def make_windows(d_act, d_valid, horizon, stride):
    """(W, H, 6) chunks that never straddle an invalid decision.

    Episode boundaries never appear inside `d_act` -- read_episodes hands
    one episode at a time -- so an intact window is also a window that
    never crosses a done.
    """
    n = len(d_act)
    if n < horizon:
        return np.zeros((0, horizon, 6), np.int64)
    starts = np.arange(0, n - horizon + 1, stride)
    if len(starts) == 0:
        return np.zeros((0, horizon, 6), np.int64)
    idx = starts[:, None] + np.arange(horizon)[None, :]
    keep = d_valid[idx].all(axis=1)
    return d_act[idx[keep]]


def yaw_feature(yb, mode):
    """Map a yaw bin index to a scalar the Euclidean metric can respect.

    The ladder is ORDINAL and geometric ({0, .25, .5, 1, 2, 4, 7, 10}), so
    one-hot (= Hamming) is wrong: it makes 'trim left 0.25' as far from
    'trim left 0.5' as from 'slam right 10'.
      deg  : the raw degrees -- physically honest, but the seven fine bins
             around zero collapse into one cluster and the codebook spends
             all its resolution on hard turns.
      log  : signed log1p(|deg|/0.25), normalised -- roughly uniform over
             the ladder, so a code can distinguish a 0.25 trim from a 0.5
             trim.  That distinction IS the air-accel gain window
             (+-0.52 deg at 3000 u/s), so this is the default.
      rank : (b - 7)/7 -- exactly uniform, ignores physical magnitude.
    """
    deg = YAW_BINS[yb]
    if mode == "deg":
        return deg / 10.0
    if mode == "rank":
        return (yb - YAW_ZERO) / float(YAW_ZERO)
    s = np.sign(deg) * np.log1p(np.abs(deg) / 0.25)
    return s / math.log1p(10.0 / 0.25)


# per-channel weights in the clustering metric.
#   yaw   dominant: it is the steering channel and the one the ledger's
#         strafe audit blames for the -7.9% air-accel capture.
#   pitch ZERO by default: src/env.c:580 passes 0.0 to pm_tick, so pitch is
#         a LIDAR-AIM action with no effect on the physics whatsoever.
#         Chunking a sensor-aiming action open-loop would blind the agent
#         for H decisions; it stays on a live per-decision head.
#   duck  half: it is inert unless enable_duck and rarely decisive on surf.
DEFAULT_W = dict(yaw=2.0, pitch=0.0, fwd=1.0, side=1.0, jump=1.0, duck=0.5)


def featurise(win, yaw_mode="log", w=None):
    """(W, H, 6) action indices -> (W, H*6) float32 clustering features."""
    w = dict(DEFAULT_W) if w is None else w
    W, H, _ = win.shape
    f = np.zeros((W, H, 6), np.float32)
    f[:, :, 0] = yaw_feature(win[:, :, 0], yaw_mode) * w["yaw"]
    f[:, :, 1] = ((win[:, :, 1] - PITCH_ZERO) / float(PITCH_ZERO)) * w["pitch"]
    f[:, :, 2] = (win[:, :, 2] - 1.0) * w["fwd"]
    f[:, :, 3] = (win[:, :, 3] - 1.0) * w["side"]
    f[:, :, 4] = win[:, :, 4] * w["jump"]
    f[:, :, 5] = win[:, :, 5] * w["duck"]
    return f.reshape(W, H * 6)


def snap_to_actions(cent, H, yaw_mode="log", w=None, pitch_bin=None):
    """Continuous centroid -> the nearest LEGAL (H, 6) action sequence.

    The decoder must emit action INDICES, so every centroid is projected
    back onto the discrete grid.  This is the frozen decoder table.
    """
    w = dict(DEFAULT_W) if w is None else w
    K = len(cent)
    c = cent.reshape(K, H, 6)
    out = np.zeros((K, H, 6), np.int64)
    if w["yaw"] > 0:
        lut = yaw_feature(np.arange(15), yaw_mode) * w["yaw"]
        out[:, :, 0] = np.argmin(np.abs(c[:, :, 0, None] - lut), axis=-1)
    else:
        out[:, :, 0] = YAW_ZERO
    if w["pitch"] > 0:
        lut = ((np.arange(7) - PITCH_ZERO) / float(PITCH_ZERO)) * w["pitch"]
        out[:, :, 1] = np.argmin(np.abs(c[:, :, 1, None] - lut), axis=-1)
    else:
        out[:, :, 1] = PITCH_ZERO if pitch_bin is None else pitch_bin
    for j, key in ((2, "fwd"), (3, "side")):
        lut = (np.arange(3) - 1.0) * w[key]
        out[:, :, j] = np.argmin(np.abs(c[:, :, j, None] - lut), axis=-1)
    for j, key in ((4, "jump"), (5, "duck")):
        out[:, :, j] = (c[:, :, j] > 0.5 * w[key]).astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# 4. k-means (sklearn if available, numpy minibatch otherwise)
# ---------------------------------------------------------------------------

def kmeans_np(X, K, iters=60, batch=8192, seed=0, verbose=False):
    """Minibatch k-means with k-means++ seeding and dead-code re-seeding.

    Dead-code re-seeding is not a nicety: VQ-BeT reports codebook collapse
    as the failure mode of discrete latents, and a dead code in a policy's
    action set is a logit the policy can never learn anything about.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    # k-means++ on a subsample (the full pairwise pass is the slow part)
    sub = X[rng.choice(n, min(n, 20000), replace=False)]
    C = np.empty((K, X.shape[1]), np.float32)
    C[0] = sub[rng.integers(len(sub))]
    d2 = ((sub - C[0]) ** 2).sum(1)
    for i in range(1, K):
        p = d2 / max(d2.sum(), 1e-12)
        C[i] = sub[rng.choice(len(sub), p=p)]
        d2 = np.minimum(d2, ((sub - C[i]) ** 2).sum(1))
    counts = np.zeros(K, np.int64)
    for it in range(iters):
        idx = rng.choice(n, min(n, batch), replace=False)
        B = X[idx]
        # gemm form, not the (batch, K, D) broadcast: at K=128, D=60,
        # batch=8192 the broadcast allocates ~250 MB per iteration
        a = np.argmin((B ** 2).sum(1)[:, None] - 2.0 * B @ C.T
                      + (C ** 2).sum(1)[None, :], axis=1)
        for k in np.unique(a):
            m = a == k
            c = int(m.sum())
            counts[k] += c
            eta = c / float(counts[k])
            C[k] += eta * (B[m].mean(0) - C[k])
        if it and it % 10 == 0:
            dead = np.flatnonzero(counts == 0)
            if len(dead):
                C[dead] = X[rng.choice(n, len(dead), replace=False)]
                if verbose:
                    print(f"  iter {it}: re-seeded {len(dead)} dead codes")
    return C


def assign(X, C, chunk=20000):
    lab = np.empty(len(X), np.int64)
    d = np.empty(len(X), np.float64)
    cn = (C ** 2).sum(1)
    for s in range(0, len(X), chunk):
        B = X[s:s + chunk]
        dd = (B ** 2).sum(1)[:, None] - 2.0 * B @ C.T + cn[None, :]
        lab[s:s + chunk] = np.argmin(dd, axis=1)
        d[s:s + chunk] = np.maximum(dd[np.arange(len(B)), lab[s:s + chunk]], 0.0)
    return lab, d


def fit_kmeans(X, K, seed=0, verbose=False, prefer_sklearn=True):
    """sklearn if it imports AND runs, else the numpy minibatch above.

    The fallback is not decoration: this box's sklearn 1.2.2 raises inside
    threadpoolctl on MiniBatchKMeans.fit, so a bare `except ImportError`
    would have taken the whole tool down.
    """
    if prefer_sklearn:
        try:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=5,
                                 batch_size=4096, max_iter=200,
                                 reassignment_ratio=0.02)
            km.fit(X)
            return (km.cluster_centers_.astype(np.float32),
                    "sklearn.MiniBatchKMeans")
        except Exception as exc:
            if verbose:
                print(f"  sklearn unavailable ({type(exc).__name__}: {exc}) "
                      f"-- numpy minibatch k-means")
    return (kmeans_np(X, K, iters=max(60, 4 * K), seed=seed, verbose=verbose),
            "numpy minibatch k-means")


# ---------------------------------------------------------------------------
# 5. human-readable decoding
# ---------------------------------------------------------------------------

_FWD = {0: "S", 1: "-", 2: "W"}
_SIDE = {0: "A", 1: "-", 2: "D"}
# src/env.c:243-254 builds the ego frame as forward=(cos yaw, sin yaw),
# "y' = left", i.e. yaw increases COUNTER-CLOCKWISE.  A POSITIVE yaw bin is
# therefore a LEFT turn and a negative one a RIGHT turn -- the opposite of
# the naive reading, and the sign the ledger's strafe audit already
# measured ("holding D turns the velocity heading clockwise, mean sign
# -0.61, and the view follows at -0.57").


def turn_dir(deg):
    return "left" if deg > 0 else "right"


def decision_token(a):
    y = YAW_BINS[a[0]]
    t = ("yaw0" if a[0] == YAW_ZERO
         else f"{'R' if y < 0 else 'L'}{abs(y):g}")
    t += "/" + _FWD[int(a[2])] + _SIDE[int(a[3])]
    if a[4]:
        t += "+JUMP"
    if a[5]:
        t += "+DUCK"
    return t


def rle(seq):
    """Run-length compress a decoded chunk so a 10-decision code is one line."""
    out, cur, n = [], None, 0
    for s in seq:
        if s == cur:
            n += 1
        else:
            if cur is not None:
                out.append(cur + (f" x{n}" if n > 1 else ""))
            cur, n = s, 1
    if cur is not None:
        out.append(cur + (f" x{n}" if n > 1 else ""))
    return " | ".join(out)


def gain_window(speed):
    """Half-width of the air-accel gain window, in deg per TICK.

    arcsin(30/|v_h|): outside it the engine's addspeed is <= 0 and air
    acceleration applies nothing.  +-0.52 deg at 3000 u/s, +-3.44 at 500.
    """
    v = np.maximum(np.asarray(speed, np.float64), 30.0)
    return np.degrees(np.arcsin(np.clip(30.0 / v, -1.0, 1.0)))


def describe(seq, speed=None):
    """One English phrase for an (H, 6) chunk -- the 'slide left side of
    ramp / takeoff / hard right turn' the design asks the codebook to name.

    ``speed`` (u/s) turns the crude fixed band into the real, speed-dependent
    gain window; without it the fixed [0.52, 3.44] band is used.
    """
    deg = YAW_BINS[seq[:, 0]]
    net, mag = float(deg.sum()), float(np.abs(deg).mean())
    H = len(seq)
    sgn = np.sign(deg[deg != 0])
    coherent = len(sgn) > 0 and abs(sgn.mean()) > 0.8
    side, fwd = seq[:, 3], seq[:, 2]
    parts = []
    if mag < 1e-9:
        parts.append("yaw held straight")
    elif mag <= 0.5 and coherent:
        parts.append(f"fine {turn_dir(net)} trim ({mag:.2f} deg/tick)")
    elif mag <= 0.5:
        parts.append(f"fine yaw dither ({mag:.2f} deg/tick)")
    elif coherent and abs(net) >= 20:
        hard = "hard" if mag >= 4 else "steady"
        parts.append(f"{hard} {turn_dir(net)} turn "
                     f"({abs(net):.1f} deg of view over {H} dec)")
    elif coherent:
        parts.append(f"{turn_dir(net)} turn ({abs(net):.1f} deg)")
    else:
        parts.append(f"yaw sawtooth (|.|={mag:.2f}, net {net:+.1f})")
    if (side == side[0]).all() and side[0] != 1:
        parts.append(f"hold {_SIDE[int(side[0])]}")
    elif (side != 1).mean() > 0.5:
        parts.append("strafe-key swaps")
    else:
        parts.append("no strafe key")
    if (fwd == 2).all():
        parts.append("W held")
    elif (fwd == 0).any():
        parts.append("S used")
    if seq[:, 4].any():
        parts.append(f"jump x{int(seq[:, 4].sum())}"
                     + (" (lead)" if seq[0, 4] else ""))
    if seq[:, 5].mean() > 0.5:
        parts.append("ducked")
    if speed is None:
        inwin = float(((np.abs(deg) >= GAIN_WIN_LO)
                       & (np.abs(deg) <= GAIN_WIN_HI)).mean())
        parts.append(f"in fixed gain band {inwin * 100:.0f}%")
    else:
        w = float(gain_window(speed))
        over = mag / max(w, 1e-9)
        parts.append(f"gain window at {float(speed):.0f} u/s is +-{w:.2f} "
                     f"deg/tick -> this code turns {over:.1f}x it")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# 6. corpus ingest
# ---------------------------------------------------------------------------

def gather(paths, horizon, stride, max_windows, yaw_mode, weights,
           force_act_every=None, verbose=True, collect_state=True):
    """Walk trajectory files -> windows (+ per-window state context)."""
    wins, ctx, stats = [], [], dict(
        files=0, episodes=0, ticks=0, decisions=0, invalid=0,
        resid_yaw=[], resid_pitch=[], act_every={}, skipped=[])
    total = 0
    for p in paths:
        cfg = run_config(p)
        yaw_adaptive = bool(cfg.get("yaw_adaptive"))
        cfg_pr = cfg.get("pitch_rate")
        Kcfg = force_act_every or cfg.get("act_every")
        nf = 0
        for hdr, rows in read_episodes(p):
            if len(rows) < horizon * 3 + 2:
                continue
            # --fix-pitch (0.0) and the unset default (-1.0 -> 10.0) both need
            # the fallback; with no run.json at all, read it off the data
            pitch_rate = (float(cfg_pr) if cfg_pr not in (None, -1.0, 0.0)
                          else detect_pitch_rate(rows))
            act, valid, resid = derive_actions(
                rows, yaw_rate=10.0, pitch_rate=pitch_rate,
                yaw_adaptive=yaw_adaptive)
            if len(act) == 0:
                continue
            if valid.mean() < 0.5:
                stats["skipped"].append((str(p), float(valid.mean())))
                continue
            if Kcfg:
                K = int(Kcfg)
                ph = max(range(K), key=lambda q: _block_const_frac(
                    act, valid, K, q)) if K > 1 else 0
            else:
                K, _fr, _sc, ph = detect_act_every(act, valid)
            stats["act_every"][K] = stats["act_every"].get(K, 0) + 1
            d_act, d_valid = decisions_from_ticks(act, valid, K, ph)
            w = make_windows(d_act, d_valid, horizon, stride)
            if len(w):
                wins.append(w)
                if collect_state:
                    starts = np.arange(0, len(d_act) - horizon + 1, stride)
                    keepm = d_valid[starts[:, None]
                                    + np.arange(horizon)[None, :]].all(axis=1)
                    ti = ph + starts[keepm] * K
                    sp = np.hypot(rows[ti, C_VX], rows[ti, C_VY])
                    ctx.append(np.stack([sp, rows[ti, C_VZ],
                                         rows[ti, C_ONGND],
                                         rows[ti, C_Z]], axis=1))
                total += len(w)
            stats["episodes"] += 1
            stats["ticks"] += len(rows)
            stats["decisions"] += len(d_act)
            stats["invalid"] += int((~d_valid).sum())
            stats["resid_yaw"].append(np.abs(resid[valid, 0]).max()
                                      if valid.any() else 0.0)
            rp = resid[valid, 1]
            rp = rp[~np.isnan(rp)]            # clamp-censored ticks excluded
            stats["resid_pitch"].append(float(np.abs(rp).max()) if len(rp) else 0.0)
            nf += 1
        stats["files"] += 1
        if verbose and stats["files"] % 100 == 0:
            print(f"  {stats['files']} files, {total:,} windows", flush=True)
        if max_windows and total >= max_windows:
            break
    if not wins:
        return (np.zeros((0, horizon, 6), np.int64),
                np.zeros((0, 4), np.float64), stats)
    W = np.concatenate(wins)
    C = np.concatenate(ctx) if ctx else np.zeros((len(W), 4))
    if max_windows and len(W) > max_windows:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(W), max_windows, replace=False)
        W, C = W[sel], C[sel]
    return W, C, stats


def find_trajs(roots, limit=None):
    out = []
    for r in roots:
        p = Path(r)
        if p.is_file():
            out.append(p)
        else:
            out.extend(sorted(p.glob("**/traj_*.jsonl")))
    if limit:
        # spread the sample across runs instead of taking one run's tail
        rng = np.random.default_rng(0)
        if len(out) > limit:
            out = [out[i] for i in sorted(rng.choice(len(out), limit,
                                                     replace=False))]
    return out


# ---------------------------------------------------------------------------
# 7. validation: reconstruct yaw from the derived bins
# ---------------------------------------------------------------------------

def validate_file(path, pitch_rate=None, yaw_adaptive=None, max_ep=3):
    """Reconstruct yaw (and pitch) from the DERIVED bins and diff against
    the recorded columns.  Zero error means the inversion is exact."""
    cfg = run_config(path)
    if yaw_adaptive is None:
        yaw_adaptive = bool(cfg.get("yaw_adaptive"))
    src = "argument"
    if pitch_rate is None:
        pr = cfg.get("pitch_rate")
        if pr not in (None, -1.0, 0.0):
            pitch_rate, src = float(pr), "run.json"
        else:
            pitch_rate, src = None, "auto-detected from the pitch column"
    print(f"\n=== VALIDATE {path}")
    print(f"    yaw_adaptive={yaw_adaptive}, yaw_rate_max_deg=10 "
          f"(train_fast.py exposes no --yaw-rate, so it is always the core "
          f"default); pitch_rate {src}")
    nep = 0
    for hdr, rows in read_episodes(path):
        nep += 1
        pr = detect_pitch_rate(rows) if pitch_rate is None else pitch_rate
        act, valid, resid = derive_actions(
            rows, pitch_rate=pr, yaw_adaptive=yaw_adaptive)
        n = len(act)
        cand = yaw_candidates(rows, 10.0, yaw_adaptive)
        yd = cand[np.arange(n), act[:, 0]]
        # integrate the DERIVED bins forward from the first recorded yaw and
        # compare against the recorded yaw column, tick by tick
        rec = np.cumsum(np.concatenate([[rows[0, C_YAW]], yd])) % 360.0
        err = np.abs(wrap180(rec - rows[:n + 1, C_YAW]))
        pd_ = PITCH_BINS[act[:, 1]] * (pr / 10.0)
        prec = np.clip(np.cumsum(np.concatenate([[rows[0, C_PITCH]], pd_])),
                       -70.0, 30.0)
        perr = np.abs(prec - rows[:n + 1, C_PITCH])
        K, frac, scores, ph = detect_act_every(act, valid)
        ok_y = np.abs(resid[:, 0]) <= 0.02
        sat = (rows[1:, C_PITCH] >= 29.99) | (rows[1:, C_PITCH] <= -69.99)
        ok_p = (np.abs(np.nan_to_num(resid[:, 1])) <= 0.011) | sat
        print(f"  episode {nep - 1}: {len(rows):,} ticks, pitch_rate={pr:g}")
        print(f"    yaw   bins identified on {100 * ok_y.mean():.3f}% of ticks,"
              f" worst snap residual {np.abs(resid[ok_y, 0]).max():.6f} deg")
        print(f"    yaw   INTEGRATED RECONSTRUCTION vs recorded column: "
              f"mean {err.mean():.6f} deg, max {err.max():.6f} deg "
              f"over {n:,} ticks")
        print(f"    pitch bins identified on {100 * ok_p.mean():.3f}% of ticks"
              f" ({int(sat.sum())} clamp-censored), worst snap residual "
              f"{np.nanmax(np.abs(resid[ok_p & ~sat, 1])):.6f} deg")
        # one-step is the honest pitch measure: integrating derived bins
        # cannot recover the trajectory once the clamp censors them, because
        # a censored tick's true bin is genuinely unknowable from the row
        p1 = np.abs(np.clip(rows[:n, C_PITCH] + pd_, -70.0, 30.0)
                    - rows[1:n + 1, C_PITCH])
        print(f"    pitch ONE-STEP reconstruction: mean {p1.mean():.6f} deg, "
              f"max {p1.max():.6f} deg   (integrated: mean {perr.mean():.4f}, "
              f"max {perr.max():.4f} -- drift is the clamp, see above)")
        print(f"    fully invertible ticks (yaw AND pitch): "
              f"{100 * valid.mean():.3f}%")
        print(f"    act_every = {K} at phase {ph}   block-constancy "
              + " ".join(f"[{k}:{v:.4f}]" for k, v in scores.items()))
        u, c = np.unique(act[ok_y, 0], return_counts=True)
        print("    yaw bin histogram: "
              + " ".join(f"{YAW_BINS[b]:g}:{n_}" for b, n_ in zip(u, c)))
        if nep >= max_ep:
            break
    return nep


# ---------------------------------------------------------------------------
# 8. selftest
# ---------------------------------------------------------------------------

def _synth_rows(act, pitch_rate=1.33, yaw0=291.36, yaw_adaptive=False,
                speed=1200.0):
    """Forward-simulate the core's yaw/pitch integration into 15-col rows.

    Mirrors src/env.c:549-557 exactly, including the 2-decimal rounding the
    recorder applies, so the inversion is tested against the real pipeline
    rather than against itself.
    """
    n = len(act)
    rows = np.zeros((n + 1, NCOL))
    yaw, pitch = yaw0, -10.0
    th = np.linspace(0.0, 2.0, n + 1)
    for t in range(n + 1):
        rows[t, C_T] = t
        rows[t, C_VX] = speed * math.cos(th[t])
        rows[t, C_VY] = speed * math.sin(th[t])
        rows[t, C_YAW] = round(yaw, 2)
        rows[t, C_PITCH] = round(pitch, 2)
        if t == n:
            break
        a = act[t]
        if yaw_adaptive:
            vh = max(math.hypot(rows[t, C_VX], rows[t, C_VY]), 1.0)
            w = math.degrees(math.atan(30.0 / vh))
            yd = min(10.0, max(-10.0, K_BINS[a[0]] * w))
        else:
            yd = YAW_BINS[a[0]]
        yaw = (yaw + yd) % 360.0
        pitch = min(30.0, max(-70.0, pitch + PITCH_BINS[a[1]] * pitch_rate / 10.0))
        rows[t, C_FWD], rows[t, C_SIDE] = a[2], a[3]
        rows[t, C_BTN] = (SURF_IN_JUMP if a[4] else 0) | (SURF_IN_DUCK if a[5] else 0)
    # the recorder writes the PRE-step yaw on line t, which is what the loop
    # above stored; re-round the whole column the way _round() does
    rows[:, C_YAW] = np.round(rows[:, C_YAW], 2)
    rows[:, C_PITCH] = np.round(rows[:, C_PITCH], 2)
    return rows


def selftest():
    rng = np.random.default_rng(7)
    fails = []

    def check(name, cond, extra=""):
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    print("SELFTEST 1 -- exact action inversion, stock yaw ladder")
    a = np.stack([rng.integers(0, n, 900) for n in NVEC], axis=1)
    rows = _synth_rows(a, pitch_rate=1.33)
    d, v, r = derive_actions(rows, pitch_rate=1.33)
    check("all ticks invertible", v.all(), f"{100 * v.mean():.2f}%")
    check("yaw   bins exact", (d[v, 0] == a[v, 0]).all())
    ok_p = d[:, 1] == a[:, 1]
    sat = (rows[1:, C_PITCH] >= 29.99) | (rows[1:, C_PITCH] <= -69.99)
    check("pitch bins exact off the clamp", ok_p[~sat].all(),
          f"{int(sat.sum())} clamped ticks masked")
    check("fwd/side/jump/duck exact", (d[:, 2:] == a[:, 2:]).all())
    check("yaw snap residual < 1e-6", float(np.abs(r[v, 0]).max()) < 1e-6,
          f"max {np.abs(r[v, 0]).max():.2e}")

    print("SELFTEST 2 -- yaw wrap across 0/360")
    a2 = np.zeros((60, 6), np.int64)
    a2[:, 0] = 0                     # -10 deg every tick: crosses 0 twice
    a2[:, 1] = PITCH_ZERO
    a2[:, 2:4] = 1
    rows2 = _synth_rows(a2, yaw0=5.0)
    d2, v2, _ = derive_actions(rows2)
    check("wrap-safe inversion", v2.all() and (d2[:, 0] == 0).all(),
          f"yaw spans {rows2[:, C_YAW].min():.1f}..{rows2[:, C_YAW].max():.1f}")

    print("SELFTEST 3 -- yaw_adaptive inversion")
    a3 = np.stack([rng.integers(0, n, 400) for n in NVEC], axis=1)
    rows3 = _synth_rows(a3, pitch_rate=1.33, yaw_adaptive=True, speed=900.0)
    d3, v3, _ = derive_actions(rows3, pitch_rate=1.33, yaw_adaptive=True)
    check("adaptive: unambiguous ticks recovered", (d3[v3, 0] == a3[v3, 0]).all(),
          f"{100 * v3.mean():.1f}% unambiguous (clip region masked)")

    print("SELFTEST 4 -- act_every detection and decision downsampling")
    for K in (1, 3, 4, 6):
        base = np.stack([rng.integers(0, n, 240) for n in NVEC], axis=1)
        rep = np.repeat(base, K, axis=0)
        rows4 = _synth_rows(rep, pitch_rate=1.33)
        d4, v4, _ = derive_actions(rows4, pitch_rate=1.33)
        det, frac, _sc, ph = detect_act_every(d4, v4)
        da, dv = decisions_from_ticks(d4, v4, K, ph)
        # the last decision loses its final tick to the (T-1) inversion;
        # pitch is excluded for the same reason detect_act_every excludes it
        m = min(len(da), len(base) - 1)
        same = (da[:m][:, DETECT_COLS] == base[:m][:, DETECT_COLS]).all()
        check(f"act_every={K} detected & downsampled",
              det == K and ph == 0 and same, f"detected {det} phase {ph}")

    print("SELFTEST 4b -- act_every detection at a nonzero PHASE")
    # record_rollout never resets _TorchPolicyBase._tick between episodes
    # (train_fast.py:549), so real episode 2+ starts mid-decision
    for K, want_ph in ((3, 2), (4, 1)):
        base = np.stack([rng.integers(0, n, 200) for n in NVEC], axis=1)
        rep = np.repeat(base, K, axis=0)[K - want_ph:]
        rows4 = _synth_rows(rep, pitch_rate=1.33)
        d4, v4, _ = derive_actions(rows4, pitch_rate=1.33)
        det, _f, _sc, ph = detect_act_every(d4, v4)
        da, _dv = decisions_from_ticks(d4, v4, det, ph)
        m = min(len(da), len(base) - 2)
        same = (da[:m][:, DETECT_COLS] == base[1:1 + m][:, DETECT_COLS]).all()
        check(f"act_every={K} recovered at phase {want_ph}",
              det == K and ph == want_ph and same,
              f"detected K={det} phase={ph}")

    print("SELFTEST 4c -- pitch_rate recovered from the pitch column alone")
    for pr in (1.33, 10.0, 2.5):
        a5 = np.stack([rng.integers(0, n, 600) for n in NVEC], axis=1)
        rows5 = _synth_rows(a5, pitch_rate=pr)
        got = detect_pitch_rate(rows5)
        # the pitch column is written to 2 dp, so the recoverable precision
        # is 0.01 deg; what has to be exact is the BIN it identifies
        d5, v5, _ = derive_actions(rows5, pitch_rate=got)
        sat5 = (rows5[1:, C_PITCH] >= 29.99) | (rows5[1:, C_PITCH] <= -69.99)
        check(f"pitch_rate={pr:g} detected to recorder precision",
              abs(got - pr) <= 0.02 and (d5[~sat5, 1] == a5[~sat5, 1]).all(),
              f"got {got:g}, bins exact off the clamp")

    print("SELFTEST 5 -- windowing never straddles an invalid decision")
    dv = np.ones(50, bool)
    dv[23] = False
    da = np.tile(np.arange(50)[:, None], (1, 6)) % 2
    w = make_windows(da, dv, horizon=10, stride=1)
    check("invalid decision excluded from every window", len(w) == 31,
          f"{len(w)} windows (41 possible, 10 touch index 23)")
    w0 = make_windows(da, np.ones(50, bool), horizon=10, stride=10)
    check("non-overlapping stride", len(w0) == 5, f"{len(w0)} windows")
    check("short episode -> no windows",
          len(make_windows(da[:4], dv[:4], 10, 1)) == 0)

    print("SELFTEST 6 -- k-means recovers planted behavior modes")
    H = 10
    protos = []
    p = np.zeros((H, 6), np.int64); p[:, 0] = 3; p[:, 1] = PITCH_ZERO
    p[:, 2] = 2; p[:, 3] = 0; protos.append(p)                 # hard left + A
    p = np.zeros((H, 6), np.int64); p[:, 0] = 11; p[:, 1] = PITCH_ZERO
    p[:, 2] = 2; p[:, 3] = 2; protos.append(p)                 # hard right + D
    p = np.zeros((H, 6), np.int64); p[:, 0] = YAW_ZERO; p[:, 1] = PITCH_ZERO
    p[:, 2] = 2; p[:, 3] = 1; p[0, 4] = 1; protos.append(p)    # straight + jump
    lab_true, X = [], []
    for i, pr in enumerate(protos):
        for _ in range(400):
            q = pr.copy()
            j = rng.integers(0, H, 2)                          # 20% jitter
            q[j, 0] = np.clip(q[j, 0] + rng.integers(-1, 2, len(j)), 0, 14)
            X.append(q); lab_true.append(i)
    X = np.asarray(X); lab_true = np.asarray(lab_true)
    F = featurise(X)
    C, _ = fit_kmeans(F, 3, seed=0)
    lab, _d = assign(F, C)
    pur = 0
    for k in range(3):
        m = lab == k
        if m.any():
            pur += np.bincount(lab_true[m]).max()
    check("3 planted modes recovered", pur / len(X) > 0.99,
          f"purity {pur / len(X):.4f}")

    print("SELFTEST 7 -- centroid snap is a legal action sequence")
    snap = snap_to_actions(C, H)
    legal = all((snap[:, :, j] >= 0).all() and (snap[:, :, j] < NVEC[j]).all()
                for j in range(6))
    check("snapped codebook in range", legal and snap.shape == (3, H, 6))
    lut = [featurise(snap[k:k + 1])[0] for k in range(3)]
    err = float(np.mean([np.linalg.norm(lut[k] - C[k]) for k in range(3)]))
    check("snap error small vs centroid", err < 0.5 * math.sqrt(H),
          f"mean L2 {err:.4f}")

    print("SELFTEST 8 -- yaw feature respects the ordinal ladder")
    f = yaw_feature(np.arange(15), "log")
    check("monotone in bin index", (np.diff(f) > 0).all())
    check("0.25 vs 0.5 separated more than 1/50 of the range",
          abs(f[8] - f[9]) > (f[14] - f[0]) / 50.0,
          f"|f(.25)-f(.5)| = {abs(f[8] - f[9]):.4f}, range {f[14] - f[0]:.4f}")
    fd = yaw_feature(np.arange(15), "deg")
    ratio = abs(f[8] - f[9]) / abs(fd[8] - fd[9])
    check("'deg' mode collapses the fine bins (why 'log' is the default)",
          ratio > 4.0,
          f"log separates 0.25 from 0.5 by {ratio:.1f}x what 'deg' does "
          f"({abs(f[8] - f[9]):.4f} vs {abs(fd[8] - fd[9]):.4f}); the "
          f"air-accel gain window lives in exactly that gap")

    print("SELFTEST 9 -- episode splitting in read_episodes")
    import tempfile, os
    fd_, tmp = tempfile.mkstemp(suffix=".jsonl"); os.close(fd_)
    with open(tmp, "w", encoding="utf-8") as fh:
        for ep in range(3):
            fh.write(json.dumps({"map": "m", "tick_ms": 10, "episode": ep}) + "\n")
            for t in range(7 + ep):
                fh.write(json.dumps([t] + [0.0] * 14) + "\n")
            fh.write(json.dumps({"end": "fail", "ticks": 7 + ep}) + "\n")
    eps = list(read_episodes(tmp))
    os.unlink(tmp)
    check("3 episodes, correct lengths",
          [len(r) for _h, r in eps] == [7, 8, 9], f"{[len(r) for _h, r in eps]}")

    print()
    if fails:
        print(f"SELFTEST FAILED: {len(fails)} check(s): {fails}")
        return 1
    print("SELFTEST PASSED (9 groups)")
    return 0


# ---------------------------------------------------------------------------
# 9. main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=["C:/RL_Surf/runs"],
                    help="run dirs or traj_*.jsonl files to ingest")
    ap.add_argument("--max-files", type=int, default=120)
    ap.add_argument("--max-windows", type=int, default=200000)
    ap.add_argument("--horizon", type=int, default=10,
                    help="chunk length in DECISIONS (act_every ticks each)")
    ap.add_argument("--stride", type=int, default=None,
                    help="window stride in decisions (default = horizon, "
                         "i.e. non-overlapping, matching how the chunks will "
                         "actually be executed)")
    ap.add_argument("--k", type=int, default=128, help="codebook size")
    ap.add_argument("--yaw-encode", choices=["log", "deg", "rank"],
                    default="log")
    ap.add_argument("--pitch-weight", type=float, default=None,
                    help="clustering weight for the pitch head (default 0: "
                         "pitch is lidar-aim only, src/env.c:580)")
    ap.add_argument("--act-every", type=int, default=None,
                    help="override; default reads run.json, else detects")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=8,
                    help="clusters to decode in full")
    ap.add_argument("--nearest", type=int, default=10,
                    help="nearest-to-centroid members to print per cluster")
    ap.add_argument("--out", default=None, help="codebook .npz path")
    ap.add_argument("--validate", nargs="*", default=None,
                    help="traj files to run the yaw-reconstruction check on")
    ap.add_argument("--audit", action="store_true",
                    help="ingest + report, no clustering")
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="run dirs held out for a cross-MAP transfer check")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.validate is not None:
        for p in args.validate:
            validate_file(p)
        if args.audit:
            return 0

    W_ = dict(DEFAULT_W)
    if args.pitch_weight is not None:
        W_["pitch"] = args.pitch_weight
    stride = args.stride if args.stride is not None else args.horizon
    files = find_trajs(args.runs, args.max_files)
    print(f"\n=== INGEST  {len(files)} trajectory files, horizon="
          f"{args.horizon} decisions, stride={stride}")
    t0 = time.perf_counter()
    win, ctx, st = gather(files, args.horizon, stride, args.max_windows,
                          args.yaw_encode, W_, args.act_every)
    print(f"  files {st['files']}  episodes {st['episodes']:,}  "
          f"ticks {st['ticks']:,}  decisions {st['decisions']:,}")
    print(f"  act_every seen: {dict(sorted(st['act_every'].items()))}")
    if st["resid_yaw"]:
        print(f"  worst yaw snap residual over all episodes:   "
              f"{max(st['resid_yaw']):.6f} deg")
        print(f"  worst pitch snap residual over all episodes: "
              f"{max(st['resid_pitch']):.6f} deg")
    print(f"  non-invertible decisions: {st['invalid']:,} "
          f"({100.0 * st['invalid'] / max(st['decisions'], 1):.4f}%)")
    if st["skipped"]:
        print(f"  files skipped (inversion failed): {len(st['skipped'])}")
    print(f"  windows: {len(win):,}   ({time.perf_counter() - t0:.1f}s)")
    if len(win) < args.k * 4:
        print("  too few windows to fit a codebook -- widen --runs")
        return 1

    print(f"\n=== ACTION MARGINALS over {len(win) * args.horizon:,} decisions")
    flat = win.reshape(-1, 6)
    yb = np.bincount(flat[:, 0], minlength=15)
    print("  yaw  " + "  ".join(
        f"{YAW_BINS[i]:g}:{100.0 * yb[i] / yb.sum():.1f}%" for i in range(15)))
    for j, nm, lab in ((2, "fwd ", _FWD), (3, "side", _SIDE)):
        c = np.bincount(flat[:, j], minlength=3)
        print(f"  {nm} " + "  ".join(
            f"{lab[i]}:{100.0 * c[i] / c.sum():.1f}%" for i in range(3)))
    print(f"  jump {100.0 * flat[:, 4].mean():.2f}%   "
          f"duck {100.0 * flat[:, 5].mean():.2f}%")
    dg = np.abs(YAW_BINS[flat[:, 0]])
    print(f"  |yaw| inside the FIXED gain band "
          f"[{GAIN_WIN_LO}, {GAIN_WIN_HI}] deg/tick: "
          f"{100.0 * ((dg >= GAIN_WIN_LO) & (dg <= GAIN_WIN_HI)).mean():.1f}%")
    if len(ctx) == len(win):
        # the real, speed-dependent window, using each window's own entry
        # speed for all H of its decisions (an approximation: speed drifts
        # inside a 300 ms chunk, but not by the order of magnitude that
        # would change the verdict)
        w = np.repeat(gain_window(ctx[:, 0]), args.horizon)
        print(f"  |yaw| inside the REAL gain window arcsin(30/|v|) at each "
              f"window's own speed: {100.0 * (dg <= w).mean():.1f}%")
        print(f"    (median speed {np.median(ctx[:, 0]):.0f} u/s -> window "
              f"+-{float(gain_window(np.median(ctx[:, 0]))):.2f} deg/tick; "
              f"median |yaw| commanded {np.median(dg):.2f} deg/tick -> "
              f"over-turn {np.median(dg) / float(gain_window(np.median(ctx[:, 0]))):.1f}x)")
    # strafe-key / turn-direction pairing: env.c's yaw is CCW-positive, so
    # the speed-gaining pairing is A with a LEFT (+) turn and D with RIGHT (-)
    yd = YAW_BINS[flat[:, 0]]
    held = flat[:, 3] != 1
    paired = ((flat[:, 3] == 0) & (yd > 0)) | ((flat[:, 3] == 2) & (yd < 0))
    print(f"  strafe pairing (A with +yaw / D with -yaw) on held-key "
          f"decisions: {100.0 * paired[held].mean():.1f}%  "
          f"(ledger's own tick-level audit of the champion: 71.8%)")

    # how much of the raw action space these H-decision windows actually use
    uniq = len(np.unique(win.reshape(len(win), -1), axis=0))
    space = math.log10(15 * 3 * 3 * 2 * 2) * args.horizon
    print(f"  distinct windows: {uniq:,} of {len(win):,} sampled; the raw "
          f"space is 10^{space:.0f}")

    if args.audit:
        return 0

    print(f"\n=== KMEANS  K={args.k}")
    F = featurise(win, args.yaw_encode, W_)
    t0 = time.perf_counter()
    C, backend = fit_kmeans(F, args.k, seed=args.seed, verbose=True)
    lab, d2 = assign(F, C)
    print(f"  backend {backend}, {time.perf_counter() - t0:.1f}s")
    occ = np.bincount(lab, minlength=args.k)
    p = occ / occ.sum()
    nz = p[p > 0]
    perp = float(np.exp(-(nz * np.log(nz)).sum()))
    print(f"  occupancy: min {occ.min()}  p10 {np.percentile(occ, 10):.0f}  "
          f"median {np.median(occ):.0f}  p90 {np.percentile(occ, 90):.0f}  "
          f"max {occ.max()}")
    print(f"  dead codes (0 members): {int((occ == 0).sum())} / {args.k}")
    print(f"  code entropy {-(nz * np.log2(nz)).sum():.2f} bits of "
          f"{math.log2(args.k):.2f}   perplexity {perp:.1f} "
          f"({100 * perp / args.k:.0f}% of K)")
    tot_var = float(((F - F.mean(0)) ** 2).sum(1).mean())
    wss = float(d2.mean())
    print(f"  intra-cluster MSE {wss:.4f}  vs total variance {tot_var:.4f}  "
          f"-> {100 * (1 - wss / tot_var):.1f}% of action variance explained")
    per = np.array([d2[lab == k].mean() if occ[k] else np.nan
                    for k in range(args.k)])
    print(f"  per-cluster intra variance: min {np.nanmin(per):.4f}  "
          f"median {np.nanmedian(per):.4f}  max {np.nanmax(per):.4f}")

    table = snap_to_actions(C, args.horizon, args.yaw_encode, W_)
    # the snapped table is the DECODER; measure what snapping costs
    snap_err = float(np.mean(np.linalg.norm(
        featurise(table, args.yaw_encode, W_) - C, axis=1)))
    print(f"  centroid -> legal-action snap: mean L2 {snap_err:.4f}")
    # and the round-trip quantisation error a chunked policy actually pays
    lut = featurise(table, args.yaw_encode, W_)
    qerr = float(np.mean(((F - lut[lab]) ** 2).sum(1)))
    print(f"  QUANTISATION MSE of the frozen decoder: {qerr:.4f} "
          f"({100 * qerr / tot_var:.1f}% of total action variance)")

    # WHERE the quantisation error sits decides the follow-up: a codebook
    # that gets the key pattern right and the yaw magnitude wrong is the
    # case BeT/VQ-BeT answer with a per-step offset head.
    dec = table[lab]
    print("  decoder-vs-truth per-head disagreement: " + "  ".join(
        f"{nm} {100 * float((dec[:, :, j] != win[:, :, j]).mean()):.1f}%"
        for j, nm in enumerate(("yaw", "pitch", "fwd", "side", "jump", "duck"))))
    yt, yq = YAW_BINS[win[:, :, 0]], YAW_BINS[dec[:, :, 0]]
    print(f"  yaw: sign agreement {100 * float((np.sign(yt) == np.sign(yq)).mean()):.1f}%"
          f", mean |error| {float(np.abs(yt - yq).mean()):.3f} deg/tick")
    if len(ctx) == len(win):
        gw = gain_window(np.repeat(ctx[:, 0], args.horizon))
        print(f"  decisions inside the real gain window: decoder "
              f"{100 * float((np.abs(yq.ravel()) <= gw).mean()):.1f}%  vs  "
              f"data {100 * float((np.abs(yt.ravel()) <= gw).mean()):.1f}% "
              f"-- quantisation low-passes the over-turning the strafe audit "
              f"blamed for the -7.9% air-accel capture")

    print(f"\n=== TOP {args.top} CLUSTERS BY OCCUPANCY")
    order = np.argsort(-occ)[:args.top]
    for rank, k in enumerate(order):
        m = np.flatnonzero(lab == k)
        near = m[np.argsort(d2[m])[:args.nearest]]
        have_ctx = len(ctx) == len(win)
        sp = ctx[m, 0] if have_ctx else np.zeros(1)
        og = ctx[m, 2] if have_ctx else np.zeros(1)
        msp = float(np.median(sp)) if have_ctx else None
        print(f"\n--- code {k:3d}   rank {rank + 1}   {occ[k]:,} members "
              f"({100 * p[k]:.2f}%)   intra-var {per[k]:.4f}")
        print(f"    centroid  : {describe(table[k], msp)}")
        print(f"    decoded   : {rle([decision_token(a) for a in table[k]])}")
        print(f"    state ctx : speed {sp.mean():7.0f} u/s "
              f"(p10 {np.percentile(sp, 10):.0f} / p90 "
              f"{np.percentile(sp, 90):.0f}), onground {100 * og.mean():.0f}%")
        print(f"    {args.nearest} nearest real windows:")
        for i in near:
            print(f"      d={math.sqrt(d2[i]):.3f}  "
                  f"{rle([decision_token(a) for a in win[i]])}")

    if args.holdout:
        hf = find_trajs(args.holdout, args.max_files)
        hw, _hc, hst = gather(hf, args.horizon, stride, 40000,
                              args.yaw_encode, W_, args.act_every,
                              verbose=False, collect_state=False)
        if len(hw):
            HF = featurise(hw, args.yaw_encode, W_)
            hl, hd = assign(HF, C)
            htot = float(((HF - HF.mean(0)) ** 2).sum(1).mean())
            hq = float(np.mean(((HF - lut[hl]) ** 2).sum(1)))
            print(f"\n=== TRANSFER TO HELD-OUT RUNS  ({len(hw):,} windows "
                  f"from {len(hf)} files)")
            print(f"  variance explained: {100 * (1 - hd.mean() / htot):.1f}% "
                  f"held-out vs {100 * (1 - wss / tot_var):.1f}% in-sample")
            print(f"  decoder QMSE:       {100 * hq / htot:.1f}% held-out vs "
                  f"{100 * qerr / tot_var:.1f}% in-sample "
                  f"(each against its OWN action variance)")
            # does the held-out data USE the same repertoire, or crowd into a
            # corner of it?  JS over the code-usage histograms answers that
            # independently of reconstruction error.
            oi = occ / occ.sum()
            oh = np.bincount(hl, minlength=args.k) / len(hl)
            mid = 0.5 * (oi + oh)

            def _kl(a, b):
                s = a > 0
                return float((a[s] * np.log2(a[s] / b[s])).sum())
            print(f"  code-usage Jensen-Shannon divergence: "
                  f"{0.5 * _kl(oi, mid) + 0.5 * _kl(oh, mid):.3f} bits "
                  f"(0 = same repertoire, 1 = disjoint); "
                  f"{int((oh == 0).sum())} / {args.k} codes unused held-out")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out,
                 codebook=table.astype(np.int8),          # (K, H, 6) DECODER
                 centroids=C.astype(np.float32),          # (K, H*6) continuous
                 occupancy=occ.astype(np.int64),
                 intra_var=per.astype(np.float32),
                 horizon=np.int64(args.horizon),
                 k=np.int64(args.k),
                 nvec=np.asarray(NVEC, np.int64),
                 yaw_bins=YAW_BINS, pitch_bins=PITCH_BINS,
                 yaw_encode=np.asarray(args.yaw_encode),
                 weights=np.asarray([W_[x] for x in
                                     ("yaw", "pitch", "fwd", "side",
                                      "jump", "duck")], np.float32),
                 quant_mse=np.float32(qerr),
                 total_var=np.float32(tot_var),
                 n_windows=np.int64(len(win)),
                 source=np.asarray(json.dumps(
                     {"runs": [str(x) for x in args.runs],
                      "files": st["files"], "episodes": st["episodes"],
                      "act_every": {str(a): b for a, b in st["act_every"].items()},
                      "stride": stride})))
        print(f"\nwrote {out}  "
              f"(codebook {table.shape} int8, centroids {C.shape} f32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
