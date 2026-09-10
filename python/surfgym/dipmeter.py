"""dipmeter.py - how big a temporary SETBACK the policy is willing to tolerate.

The race shaping is built on the geodesic distance-to-finish ``d``.  Round 19
and round 30 both turned on one question that ``race/eval_progress`` cannot
ask: when the agent has to give ground - fly back UP a ramp, leave the low-``d``
shell, take the outer line - does it hold on long enough to get the ground
back, or does it die inside the setback?  ``--race-ratchet`` already maintains
exactly the machinery needed to answer it (``_rec``, ``ratchet_gap``), but it
is a TREATMENT and only exists when the flag is on.  This module is the same
bookkeeping as a pure DIAGNOSTIC: it is on for every arm, it changes no
reward, no observation and no RNG draw, and it is defined identically online
(``DipMeter``, inside :class:`surfgym.rewards.RaceReward`) and offline
(:func:`enumerate_dips`, over a recorded trajectory).

Definitions - PINNED, because the online accumulator and the offline
trajectory analysis must produce the same numbers or neither is evidence:

``b_t``
    the running MINIMUM of ``d`` within the CURRENT episode.  It is reset to
    the current ``d`` at every episode start - spawn, reservoir respawn, demo
    respawn, random respawn, a truncation reset: any reset that restarts the
    episode.  This is exactly ``--race-ratchet``'s ``b``, computed whether or
    not that flag is on.

``depth_t``
    ``(d_t - b_t) * scale`` with ``scale`` the reward's OWN shaping scale
    (``100/d0 * --race-shaping``).  Units are REWARD UNITS: the whole map is
    worth 100 on any map, so a depth of 4.24 means "this setback costs as
    much shaping as 4.24% of the map".  ``depth_t >= 0`` always, and it is 0
    exactly at a new record.

DIP
    a maximal contiguous stretch of REWARD CALLS (decisions, not physics
    ticks) with ``depth > 0``.  Its DEPTH is the maximum ``depth_t`` inside
    it; its DURATION is ``(number of calls) * act_every * tick_ms / 1000``
    seconds, from the run's own tick base - never a hard-coded 100 Hz.

SURVIVED / FAILED
    a dip SURVIVES when ``depth`` returns to exactly 0, i.e. the episode set
    a new record and got the ground back.  It FAILS when the episode ends
    (any terminal: fail, done, truncation, stall-kill, respawn) while
    ``depth > 0``.  A stretch that is still open at the end of a WINDOW that
    is not an episode end is neither - it is unresolved and is not reported.

The index convention, which is what makes the two implementations comparable:
sample 0 of a sequence is the episode's own START (the spawn), so
``depth_0 = 0`` by construction and the spawn call is never inside a dip.
Online that sample is the ``on_reset`` / autoreset distance; offline it is
row 0 of the episode, subsampled at ``act_every``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["enumerate_dips", "dip_depths", "terminal_depth", "DipMeter",
           "DIP_KEYS", "dip_summary", "empty_dip_raw", "merge_dip_raw",
           "dips_from_episode"]

# the CSV / dict keys, in the order train_fast writes them
DIP_KEYS = ("max_survived_depth", "p90_survived_depth", "max_survived_secs",
            "survived_per_ep", "fail_depth", "fail_secs", "fail_frac",
            "p50_term_depth", "p90_term_depth")


# --------------------------------------------------------------------------
# the OFFLINE reference: one episode, one array of d
# --------------------------------------------------------------------------
def dip_depths(d, scale) -> np.ndarray:
    """``depth_t = (d_t - min_{s<=t} d_s) * scale`` over ONE episode.

    ``d`` is the per-DECISION geodesic distance, sample 0 being the episode's
    own start.  float64 throughout: this is a diagnostic, and the online
    accumulator uses the same dtype so the two agree exactly.
    """
    d = np.asarray(d, np.float64).reshape(-1)
    if d.size == 0:
        return np.zeros(0, np.float64)
    return (d - np.minimum.accumulate(d)) * float(scale)


def enumerate_dips(d, scale, dt, ended_in_dip: bool = True) -> list:
    """Every dip in ONE episode's ``d`` sequence.

    Parameters
    ----------
    d
        per-DECISION geodesic distance-to-finish, sample 0 = the episode's
        own start (spawn / respawn).  A recorded trajectory is per PHYSICS
        TICK, so subsample it by ``act_every`` before calling this - the
        reward, and therefore a dip, is evaluated once per decision.
    scale
        the reward's own shaping scale, ``100 / d0 * --race-shaping``.
    dt
        seconds per decision, ``act_every * tick_ms / 1000``.
    ended_in_dip
        True when the sequence ends at a real episode TERMINAL, so a dip
        still open on the last sample FAILED.  False when the sequence is
        merely a window that was cut (the trailing open dip is unresolved
        and is dropped, being neither survived nor failed).

    Returns
    -------
    list of ``(start_idx, end_idx, depth, secs, survived)``
        HALF-OPEN index range ``[start_idx, end_idx)`` into ``d``; ``depth``
        is the maximum ``depth_t`` inside it (reward units); ``secs`` is
        ``(end_idx - start_idx) * dt``; ``survived`` is True when the dip was
        closed by a new record.
    """
    depth = dip_depths(d, scale)
    n = depth.size
    inside = depth > 0.0
    out = []
    i = 0
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j < n and inside[j]:
            j += 1
        survived = j < n                      # closed by a new record
        if not survived and not ended_in_dip:
            break                             # unresolved trailing stretch
        out.append((i, j, float(depth[i:j].max()),
                    float(j - i) * float(dt), bool(survived)))
        i = j
    return out


def terminal_depth(d, scale) -> float:
    """``depth`` on the episode's LAST sample - the depth at which it died."""
    dep = dip_depths(d, scale)
    return float(dep[-1]) if dep.size else 0.0


# --------------------------------------------------------------------------
# the ONLINE accumulator: N envs, one reward call at a time
# --------------------------------------------------------------------------
def empty_dip_raw() -> dict:
    """The shape :meth:`DipMeter.pop_raw` returns with nothing recorded."""
    z = np.zeros(0, np.float64)
    return {"surv_depth": z, "surv_secs": z, "fail_depth": z,
            "fail_secs": z, "term_depth": z, "n_ended": 0}


class DipMeter:
    """Per-env dip bookkeeping, one vectorised update per reward call.

    LOGGING ONLY.  It reads ``d``, the ``ended`` mask, the scale and the
    decision period; it writes nothing any other code reads, draws from no
    RNG and allocates only its own arrays.

    The call contract mirrors :meth:`RaceReward.__call__`'s own: on a call
    where ``ended[i]`` is set, ``d[i]`` is ALREADY the next episode's spawn
    (the core autoresets before the reward is computed), so the dying
    episode's last state is the PREVIOUS call's - which is why the depth at
    termination is the depth this meter is already carrying, and why the
    ended rows restart their record at ``d[i]`` rather than folding it in.
    """

    def __init__(self, n: int) -> None:
        n = int(n)
        self.b: np.ndarray | None = None      # running min of d, per env
        self.depth = np.zeros(n, np.float64)  # depth as of the last call
        self.run_max = np.zeros(n, np.float64)   # max depth in the OPEN dip
        self.run_n = np.zeros(n, np.int64)       # calls in the OPEN dip
        self._surv_depth: list = []
        self._surv_secs: list = []
        self._fail_depth: list = []
        self._fail_secs: list = []
        self._term_depth: list = []
        self._n_ended = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self, d) -> None:
        """Anchor every env at ``d`` (RaceReward.on_reset)."""
        d = np.asarray(d, np.float64)
        if self.b is None or self.b.shape != d.shape:
            self.depth = np.zeros(d.shape, np.float64)
            self.run_max = np.zeros(d.shape, np.float64)
            self.run_n = np.zeros(d.shape, np.int64)
        self.b = d.copy()
        self.depth[:] = 0.0
        self.run_max[:] = 0.0
        self.run_n[:] = 0

    def update(self, d, ended, scale: float, dt: float) -> None:
        """One reward call.  ``dt`` = ``act_every * tick_ms / 1000`` as of
        THIS call (``--tick-ms-schedule`` moves it; a dip is charged the
        period in force when it closed, which is a diagnostic-level
        approximation and exact whenever the tick is constant)."""
        d = np.asarray(d, np.float64)
        ended = np.asarray(ended, bool)
        if self.b is None:
            self.start(d)
            return
        scale = float(scale)
        dt = float(dt)
        # 1. close out the episodes that ended.  Their last live state is
        #    the previous call's, so self.depth / run_* still describe it.
        ei = np.flatnonzero(ended)
        if ei.size:
            self._term_depth.append(self.depth[ei].copy())
            self._n_ended += int(ei.size)
            fi = ei[self.run_n[ei] > 0]
            if fi.size:
                self._fail_depth.append(self.run_max[fi].copy())
                self._fail_secs.append(self.run_n[fi].astype(np.float64) * dt)
            # the new episode's record starts at ITS OWN spawn
            self.b[ei] = d[ei]
            self.depth[ei] = 0.0
            self.run_max[ei] = 0.0
            self.run_n[ei] = 0
        # 2. advance the live envs
        li = np.flatnonzero(~ended)
        if not li.size:
            return
        bl = np.minimum(self.b[li], d[li])
        self.b[li] = bl
        dep = (d[li] - bl) * scale
        self.depth[li] = dep
        inside = dep > 0.0
        # a dip that was open and is now back at a record SURVIVED
        closed = li[(~inside) & (self.run_n[li] > 0)]
        if closed.size:
            self._surv_depth.append(self.run_max[closed].copy())
            self._surv_secs.append(self.run_n[closed].astype(np.float64) * dt)
            self.run_max[closed] = 0.0
            self.run_n[closed] = 0
        oi = li[inside]
        if oi.size:
            self.run_n[oi] += 1
            self.run_max[oi] = np.maximum(self.run_max[oi], dep[inside])

    # -- read-out ----------------------------------------------------------
    def pop_raw(self) -> dict:
        """Drain the window: raw arrays, so pooling over MAPS (and, in
        principle, over ranks) is a concatenation rather than a mean of
        means.  Percentiles are computed once, at the end, over the pool."""
        def cat(parts):
            return (np.concatenate(parts) if parts
                    else np.zeros(0, np.float64))
        out = {"surv_depth": cat(self._surv_depth),
               "surv_secs": cat(self._surv_secs),
               "fail_depth": cat(self._fail_depth),
               "fail_secs": cat(self._fail_secs),
               "term_depth": cat(self._term_depth),
               "n_ended": self._n_ended}
        self._surv_depth.clear()
        self._surv_secs.clear()
        self._fail_depth.clear()
        self._fail_secs.clear()
        self._term_depth.clear()
        self._n_ended = 0
        return out


def merge_dip_raw(parts) -> dict:
    """Pool several :meth:`DipMeter.pop_raw` dicts (one per map slot)."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return empty_dip_raw()
    out = {}
    for k in ("surv_depth", "surv_secs", "fail_depth", "fail_secs",
              "term_depth"):
        out[k] = np.concatenate([np.asarray(p[k], np.float64) for p in parts])
    out["n_ended"] = int(sum(int(p["n_ended"]) for p in parts))
    return out


def dip_summary(raw) -> dict:
    """The nine ``dip/*`` numbers from a (pooled) raw dict.

    NaN wherever the window holds nothing to compute from - the trainer
    writes NaN as a blank CSV cell, the same as every other absent metric.
    """
    nan = float("nan")
    sd = np.asarray(raw["surv_depth"], np.float64)
    ss = np.asarray(raw["surv_secs"], np.float64)
    fd = np.asarray(raw["fail_depth"], np.float64)
    fs = np.asarray(raw["fail_secs"], np.float64)
    td = np.asarray(raw["term_depth"], np.float64)
    n_ep = int(raw["n_ended"])
    return {
        "max_survived_depth": float(sd.max()) if sd.size else nan,
        "p90_survived_depth": (float(np.percentile(sd, 90))
                               if sd.size else nan),
        "max_survived_secs": float(ss.max()) if ss.size else nan,
        # per ENDED episode - the same denominator fail_frac uses, so the
        # two columns are read against each other without a rescale
        "survived_per_ep": (sd.size / n_ep) if n_ep else nan,
        "fail_depth": float(fd.mean()) if fd.size else nan,
        "fail_secs": float(fs.mean()) if fs.size else nan,
        "fail_frac": (fd.size / n_ep) if n_ep else nan,
        "p50_term_depth": float(np.percentile(td, 50)) if td.size else nan,
        "p90_term_depth": float(np.percentile(td, 90)) if td.size else nan,
    }


def dips_from_episode(d, scale, dt, ended_in_dip: bool = True) -> dict:
    """The offline path packed into the same raw dict the online one drains,
    so one episode's arrays can be pooled with (or checked against) the
    trainer's.  ``d`` is per DECISION, sample 0 = the episode's start."""
    dips = enumerate_dips(d, scale, dt, ended_in_dip=ended_in_dip)
    sd = np.array([x[2] for x in dips if x[4]], np.float64)
    ss = np.array([x[3] for x in dips if x[4]], np.float64)
    fd = np.array([x[2] for x in dips if not x[4]], np.float64)
    fs = np.array([x[3] for x in dips if not x[4]], np.float64)
    # an UNRESOLVED window (ended_in_dip=False) contributes no ended episode
    # and therefore no termination depth - it did not terminate
    td = (np.array([terminal_depth(d, scale)], np.float64) if ended_in_dip
          else np.zeros(0, np.float64))
    return {"surv_depth": sd, "surv_secs": ss, "fail_depth": fd,
            "fail_secs": fs, "term_depth": td,
            "n_ended": 1 if ended_in_dip else 0}
