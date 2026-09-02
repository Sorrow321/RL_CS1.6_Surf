"""goalarc.py - per-env arc-length progress: ArcProgress with the LINE moved
into the batch.

``route.ArcProgress`` pays "how far along the route have you got, counting
only samples inside a corridor of it, advancing in order" for ONE line shared
by every env. Goal-conditioned training does not work that way: every env
gets its own goal and its own reference line to it (``goals.MultiLine``),
redrawn on every reset, so the reward-side coordinate has to be per-env too.
This is that object - the same rules, the same arithmetic, the line index
moved into the batch dimension.

The rules are ArcProgress's, and each one exists for the reason recorded
there:

* **local window.** The anchor may move at most ``window`` vertices per call,
  so an off-route flight cannot walk the coordinate down the track and cash
  it on re-entry.
* **corridor gate.** A sample farther than ``corridor`` from the polyline
  pays nothing and does not move the anchor - leaving the line freezes the
  clock rather than charging for it.
* **signed delta inside the corridor**, so the term telescopes to
  ``arc_end - arc_spawn`` over an episode and hovering nets exactly zero.

What is genuinely new here is the padding, and it is the whole risk. ``pts``
is (N, l_max, 3) with only the first ``length[n]`` rows of row n valid; the
rest is the previous line's geometry or a repeat of this line's endpoint.
Every index into it is therefore clamped or masked PER ENV, exactly as
``MultiLine`` has to do - miss one and an env with a short line reads a
neighbour's leftovers, which looks like a perfectly plausible route and is
silent.

**Why the +inf mask is the same object as ArcProgress's edge padding.**
ArcProgress precomputes its candidate window over an edge-padded copy of the
line, so window slot ``k`` is ``pts[clip(c + k - w, 0, L-1)]``; here the
window is gathered per env and slot ``k`` is masked to +inf when
``c + k - w`` falls outside ``[0, length-1]``. Those two searches always
return the SAME anchor:

* every padded slot repeats ``pts[0]`` or ``pts[L-1]``, and both of those are
  also present as a VALID slot whenever the padding is active, so the
  minimum value itself is identical;
* the head duplicates sit at SMALLER ``k`` than the valid copy, so numpy's
  first-minimum argmin prefers them - but ``i = clip(c + k - w, 0, L-1)``
  then clamps both answers to 0. The tail duplicates sit at LARGER ``k`` and
  lose the tie outright;
* a NaN in the line propagates the same way, for the same reason.

So this module is a per-env generalization, not a redefinition, and the
equivalence is asserted row-by-row against ArcProgress in
tests/python/test_goalarc.py rather than argued only here.

Two paths, as in route.py and goalfield.py: the numpy body of
:meth:`MultiArcProgress._advance_np` is THE REFERENCE, and
:func:`_build_fast_advance` fuses it into one numba pass doing the identical
float32/float64 arithmetic in the identical order. ``SURFGYM_NO_NUMBA=1``
forces the reference with no code edit. The identity is bitwise, not
approximate, because with ``--race-arc`` the arc coordinate IS the reward: a
fast path that merely rounded differently would fork the shaping from every
number in the ledger while looking exactly like an experimental effect.

numpy only at import time (numba is optional and imported inside the
builder); nothing here touches torch or a GPU.
"""
from __future__ import annotations

import os

import numpy as np

from .route import DEFAULT_SPACING

__all__ = ["MultiArcProgress"]

# Rows per chunk in the GLOBAL (re-anchor) search. That search is the only
# place a (rows, l_max, 3) float64 temporary appears; at the training shape
# (2,048 envs x 768 vertices) one unchunked pass is a 37 MB allocation on the
# reset path for no reason. 256 rows bounds it at ~4.7 MB and costs nothing.
_RESET_CHUNK = 256


def _build_fast_advance():
    """One fused numba pass replacing ``_advance_np``'s ~20 numpy passes.

    Structure, dtypes and order of operations are lifted from
    ``route._build_fast_advance`` - see that docstring for why each one is
    load-bearing. The three that decide bit-identity, restated because they
    are easy to "clean up" and impossible to notice afterwards:

    * **three-term sums are LEFT-TO-RIGHT**: ``(x*x + y*y) + z*z`` is what
      ``np.sum(..., -1)`` does for n = 3 and is NOT the same float32 number
      as ``x*x + (y*y + z*z)``.
    * **argmin takes the FIRST minimum, and a NaN is minimal** - numpy's
      ``FLOAT_argmin`` returns the index of the first NaN it meets and stops.
    * **``np.clip`` propagates NaN**, and ``dd <= corridor**2`` compares in
      FLOAT32 because a float32 array against a Python float stays float32;
      ``corr2`` is pre-rounded to float32 by the caller.

    The one addition over the single-line kernel is that ``length`` is read
    per row: the vertex window is masked to +inf outside ``[0, length-1]``
    (where ArcProgress reads its edge-padded copy - identical answers, see
    the module docstring) and both segment indices clamp at ``length - 2``.

    ``error_model="numpy"`` is not decoration: numba's default model raises
    ZeroDivisionError on ``0.0 / 0.0``, and a hand-built line can carry a
    zero-length segment where numpy quietly yields NaN and the caller's
    ``d1 <= d2`` / corridor tests then reject the row.

    Single-threaded on purpose - the batch is a few thousand rows of ~200
    flops, under the fork/join cost of a ``prange``, and a thread pool here
    would take CPU from the trainer mid-rollout.
    """
    try:
        from numba import njit
    except Exception:
        return None

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _seg1(px, py, pz, pts, r, s, spacing):
        """``_seg`` for ONE point against env ``r``'s segment [s, s+1]."""
        ax = pts[r, s, 0]
        ay = pts[r, s, 1]
        az = pts[r, s, 2]
        abx = pts[r, s + 1, 0] - ax
        aby = pts[r, s + 1, 1] - ay
        abz = pts[r, s + 1, 2] - az
        pax = px - ax
        pay = py - ay
        paz = pz - az
        num = ((pax * abx) + (pay * aby)) + (paz * abz)
        den = ((abx * abx) + (aby * aby)) + (abz * abz)
        t = num / den
        if t == t:                          # np.clip leaves NaN alone
            if not (t > np.float32(0.0)):   # max(t, 0.0), -0.0 -> +0.0
                t = np.float32(0.0)
            if not (t < np.float32(1.0)):   # min(t, 1.0)
                t = np.float32(1.0)
        dqx = pax - t * abx
        dqy = pay - t * aby
        dqz = paz - t * abz
        d = ((dqx * dqx) + (dqy * dqy)) + (dqz * dqz)
        return (np.float64(s) + np.float64(t)) * spacing, d

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _f(p, pts, length, idx, arc, spacing, corr2, window):
        n = p.shape[0]
        m = 2 * window + 1
        delta = np.empty(n, np.float64)
        inside = np.empty(n, np.bool_)
        arc_out = np.empty(n, np.float64)
        idx_out = np.empty(n, np.int64)
        far = np.float32(np.inf)
        for r in range(n):
            px = p[r, 0]
            py = p[r, 1]
            pz = p[r, 2]
            c = idx[r]
            lmax = np.int64(length[r]) - 1
            smax = lmax - 1
            # windowed nearest VALID vertex == the masked d2.argmin(1) of the
            # reference; slots off either end of the line are +inf
            j = c - window
            if j < 0 or j > lmax:
                best = far
            else:
                dx = pts[r, j, 0] - px
                dy = pts[r, j, 1] - py
                dz = pts[r, j, 2] - pz
                best = ((dx * dx) + (dy * dy)) + (dz * dz)
            bi = 0
            if best == best:                # a NaN at k = 0 is already argmin
                for k in range(1, m):
                    j = c + k - window
                    if j < 0 or j > lmax:
                        v = far
                    else:
                        dx = pts[r, j, 0] - px
                        dy = pts[r, j, 1] - py
                        dz = pts[r, j, 2] - pz
                        v = ((dx * dx) + (dy * dy)) + (dz * dz)
                    if v != v:              # numpy: NaN is minimal, stop here
                        bi = k
                        break
                    if v < best:            # strict: FIRST minimum wins ties
                        best = v
                        bi = k
            i = c + bi - window
            if i < 0:
                i = 0
            elif i > lmax:
                i = lmax
            s1 = i - 1                      # np.clip(i - 1, 0, length - 2)
            if s1 < 0:
                s1 = 0
            elif s1 > smax:
                s1 = smax
            s2 = i                          # np.clip(i,     0, length - 2)
            if s2 < 0:
                s2 = 0
            elif s2 > smax:
                s2 = smax
            a1, d1 = _seg1(px, py, pz, pts, r, s1, spacing)
            a2, d2 = _seg1(px, py, pz, pts, r, s2, spacing)
            if d1 <= d2:                    # np.where(d1 <= d2, ...)
                a = a1
                dd = d1
            else:
                a = a2
                dd = d2
            if dd <= corr2:
                inside[r] = True
                delta[r] = a - arc[r]
                arc_out[r] = a
                # _index() only INSIDE the branch. The reference computes it
                # for every row and throws it away outside, which on a
                # degenerate segment casts a NaN to int64 (numpy warns and
                # yields INT64_MIN; in LLVM that is undefined). A NaN arc
                # always carries a NaN dd, so it can never be inside - the
                # discarded value is unreachable and the results still match.
                jj = np.int64(np.rint(a / spacing))
                if jj < 0:
                    jj = 0
                elif jj > lmax:
                    jj = lmax
                idx_out[r] = jj
            else:
                inside[r] = False
                delta[r] = 0.0
                arc_out[r] = arc[r]
                idx_out[r] = idx[r]
        return delta, inside, arc_out, idx_out

    return _f


# SURFGYM_NO_NUMBA=1 forces the numpy reference, the same switch route.py and
# goalfield.py use. The fast path is asserted bit-identical by
# tests/python/test_goalarc.py, but "is this the optimizer or the reward?"
# has to be answerable on a rented box with no code edit.
_FAST_ADVANCE = None if os.environ.get("SURFGYM_NO_NUMBA") == "1" \
    else _build_fast_advance()


class MultiArcProgress:
    """Order-only arc-length progress along N reference lines, one per env.

    ``pts`` is (n_envs, l_max, 3) float32 with the first ``length[n]`` rows of
    row n valid, matching ``goals.MultiLine`` exactly so the reward and the
    observation can be fed from the same numpy lines. ``arc`` (float64) and
    ``idx`` (int64) are the per-env anchor, and :meth:`advance` is the
    per-tick call.

    Every env starts on a placeholder line - two vertices ``spacing`` apart
    on +z at the map origin - so that nothing is NaN and nothing is
    degenerate before the first :meth:`set_lines`. ``MultiLine`` uses a
    zero-length placeholder because its featurizer clamps the tangent norm;
    this class divides by the segment length with no clamp (that is what
    makes it bit-identical to ArcProgress), so the placeholder has to be a
    real segment.
    """

    def __init__(self, n_envs, l_max: int = 768,
                 spacing: float = DEFAULT_SPACING, corridor: float = 1500.0,
                 window: int = 16):
        self.n_envs = int(n_envs)
        self.l_max = int(l_max)
        if self.n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {self.n_envs}")
        if self.l_max < 2:
            raise ValueError(f"l_max must be >= 2, got {self.l_max}")
        self.spacing = float(spacing)
        self.corridor = float(corridor)
        self.window = max(1, int(window))
        self.pts = np.zeros((self.n_envs, self.l_max, 3), np.float32)
        self.pts[:, 1:, 2] = np.float32(self.spacing)
        self.length = np.full(self.n_envs, 2, np.int32)
        self.arc = np.zeros(self.n_envs, np.float64)
        self.idx = np.zeros(self.n_envs, np.int64)
        # held once: the reset path rebuilds nothing per call and the advance
        # path allocates only the window it actually reads
        self._rows = np.arange(self.n_envs, dtype=np.int64)
        self._col = np.arange(self.l_max, dtype=np.int64)
        self._k = np.arange(2 * self.window + 1, dtype=np.int64)

    # ----------------------------------------------------------------- build
    def set_lines(self, idx, lines, origin=None) -> None:
        """Install ``lines[k]`` into env ``idx[k]`` and re-anchor those envs.

        Same contract as ``MultiLine.set_lines``: the lines are numpy (L, 3),
        ``2 <= L <= l_max``, and ALREADY resampled at ``self.spacing`` - the
        arc coordinate is ``vertex index * spacing`` and is simply wrong on a
        line that is not uniform, so resampling is the caller's job
        (``goals.chord_line``, ``goals.segment_line``,
        ``route.resample_polyline``). The pad is the line's own last point
        repeated, not zeros, so that a missed clamp reads as "the env stares
        at its goal" instead of "the env flies at the map origin".

        A new line invalidates the old anchor - the arc coordinate of the
        line that was just thrown away means nothing on the new one - so the
        touched envs are re-anchored and nobody else is touched:

        * ``origin=None``: anchor at the START of the new line (arc 0,
          idx 0). That is the right answer for a line BUILT from the player's
          current position (a chord to a fresh goal starts under the
          player's feet) and it needs no position passed in.
        * ``origin`` given: a GLOBAL nearest-point search on the new line,
          i.e. :meth:`reset` restricted to these envs. Shape (n_envs, 3) is
          read as the whole batch and indexed with ``idx``; otherwise it must
          be (len(idx), 3) and is matched to ``idx`` positionally.
        """
        idx = np.asarray(idx, np.int64).reshape(-1)
        lines = list(lines)
        if len(idx) != len(lines):
            raise ValueError(f"set_lines: {len(idx)} indices but "
                             f"{len(lines)} lines")
        if len(idx) == 0:
            return
        if idx.min() < 0 or idx.max() >= self.n_envs:
            raise ValueError(f"set_lines: env index out of range "
                             f"[0, {self.n_envs}): min {idx.min()}, "
                             f"max {idx.max()}")
        for k, ln in enumerate(lines):
            a = np.ascontiguousarray(np.asarray(ln, np.float32))
            if a.ndim != 2 or a.shape[1] != 3:
                raise ValueError(f"set_lines: line {k} has shape {a.shape}, "
                                 f"want (L, 3)")
            if len(a) < 2 or len(a) > self.l_max:
                raise ValueError(f"set_lines: line {k} has L={len(a)}, need "
                                 f"2 <= L <= l_max={self.l_max}")
            e = int(idx[k])
            self.pts[e, :len(a)] = a
            self.pts[e, len(a):] = a[-1]
            self.length[e] = len(a)
        if origin is None:
            self.arc[idx] = 0.0
            self.idx[idx] = 0
            return
        o = np.asarray(origin, np.float64)
        if o.ndim != 2 or o.shape[1] != 3:
            raise ValueError(f"set_lines: origin has shape {o.shape}, want "
                             f"(n_envs, 3) or ({len(idx)}, 3)")
        if len(o) == self.n_envs:
            o = o[idx]
        elif len(o) != len(idx):
            raise ValueError(f"set_lines: origin has {len(o)} rows, want "
                             f"{self.n_envs} or {len(idx)}")
        arc = self._locate(idx, np.ascontiguousarray(o))
        self.arc[idx] = arc
        self.idx[idx] = self._index(arc, idx)

    def describe(self) -> str:
        lo = float((int(self.length.min()) - 1) * self.spacing)
        hi = float((int(self.length.max()) - 1) * self.spacing)
        return (f"multiarc {self.n_envs} envs x <= {self.l_max} pts @ "
                f"{self.spacing:g}u (arc {lo:,.0f}-{hi:,.0f}u), corridor "
                f"{self.corridor:g}u, window +/-{self.window} "
                f"({self.window * self.spacing:,.0f}u), "
                f"{'numba' if _FAST_ADVANCE is not None else 'numpy'}")

    def total_arc(self):
        """(N,) float64 arc length of each env's own line, in map units."""
        return (self.length.astype(np.float64) - 1.0) * self.spacing

    # -------------------------------------------------------------- geometry
    def _seg(self, p, seg, rows):
        """Nearest point on env ``rows``' segments [seg, seg+1].

        -> (arc float64, dist^2 float32). Every operation is the dtype and
        the order ``ArcProgress._seg`` uses; the only change is that the
        vertices are gathered per env.
        """
        a = self.pts[rows, seg]
        ab = self.pts[rows, seg + 1] - a
        pa = p - a
        t = (pa * ab).sum(1) / (ab * ab).sum(1)
        np.clip(t, 0.0, 1.0, out=t)
        dq = pa - t[:, None] * ab
        return (seg + t.astype(np.float64)) * self.spacing, (dq * dq).sum(1)

    def _refine(self, p, i, rows):
        """Vertex index -> continuous arc, checking BOTH adjacent segments.

        Snapping to the vertex would quantize the reward at ``spacing``,
        which is larger than one decision of travel and therefore most of
        the signal.
        """
        last = self.length[rows] - 2
        a1, d1 = self._seg(p, np.clip(i - 1, 0, last), rows)
        a2, d2 = self._seg(p, np.clip(i, 0, last), rows)
        take = d1 <= d2
        return np.where(take, a1, a2), np.where(take, d1, d2)

    def _index(self, arc, rows):
        return np.clip(np.rint(arc / self.spacing).astype(np.int64), 0,
                       self.length[rows] - 1)

    def _locate(self, rows, p64):
        """GLOBAL nearest point on each row's OWN line -> arc. Anchoring only.

        Chunked over rows, and the invalid tail is +inf so a padded vertex -
        which is the previous line's geometry, anywhere at all - can never be
        picked. Masking AFTER the distance is computed also swallows a NaN
        left in the tail, which an arithmetic mask would not.

        The difference is taken directly in float64 rather than through
        ArcProgress's ``|R|^2 - 2 p.R`` expansion: there is no shared matmul
        to hoist ``|R|^2`` out of when every env has a different line, and
        the direct form is the more accurate of the two.
        """
        p32 = np.ascontiguousarray(p64, np.float32)
        i0 = np.empty(len(rows), np.int64)
        for s in range(0, len(rows), _RESET_CHUNK):
            r = rows[s:s + _RESET_CHUNK]
            d = self.pts[r].astype(np.float64) \
                - p64[s:s + _RESET_CHUNK, None, :]
            d2 = (d * d).sum(-1)
            d2[self._col[None, :] >= self.length[r][:, None]] = np.inf
            i0[s:s + _RESET_CHUNK] = d2.argmin(1)
        return self._refine(p32, i0, rows)[0]

    # ----------------------------------------------------------------- state
    def reset(self, origin, mask=None) -> None:
        """(Re-)anchor the (masked) envs with a GLOBAL search over their own
        line - spawns and respawns relocate the player arbitrarily and a
        local window cannot follow that."""
        p64 = np.asarray(origin, np.float64)
        if p64.shape != (self.n_envs, 3):
            raise ValueError(f"reset: origin has shape {p64.shape}, want "
                             f"({self.n_envs}, 3)")
        rows = self._rows if mask is None \
            else np.flatnonzero(np.asarray(mask, bool))
        if len(rows) == 0:
            return
        arc = self._locate(rows, np.ascontiguousarray(p64[rows]))
        self.arc[rows] = arc
        self.idx[rows] = self._index(arc, rows)

    def advance(self, origin):
        """-> (delta (N,) float64, inside (N,) bool).

        ``delta`` is 0 outside the corridor and the anchor is frozen there,
        so no off-route excursion can be cashed; inside it is the signed
        change in arc, which telescopes over an episode.

        :meth:`_advance_np` is THE REFERENCE. The numba kernel computes the
        identical float32/float64 arithmetic in the identical order and is
        asserted bit-identical - returns AND the (arc, idx) state - by
        tests/python/test_goalarc.py. Anything unexpected (a foreign dtype, a
        non-contiguous block, ``SURFGYM_NO_NUMBA=1``, no numba) falls back to
        the reference silently.
        """
        p = np.ascontiguousarray(origin, np.float32)
        if p.shape != (self.n_envs, 3):
            raise ValueError(f"advance: origin has shape {p.shape}, want "
                             f"({self.n_envs}, 3)")
        if _FAST_ADVANCE is not None and self.pts.dtype == np.float32 \
                and self.pts.flags.c_contiguous \
                and self.length.dtype == np.int32 \
                and self.arc.dtype == np.float64 \
                and self.idx.dtype == np.int64:
            # corr2 is rounded to float32 HERE because `dd <= corridor**2` in
            # the reference compares in float32 (a float32 array against a
            # Python float), and it is not cached so that a caller may
            # retune self.corridor between calls
            delta, inside, arc, idx = _FAST_ADVANCE(
                p, self.pts, self.length, self.idx, self.arc, self.spacing,
                np.float32(self.corridor * self.corridor),
                np.int64(self.window))
            self.arc = arc
            self.idx = idx
            return delta, inside
        return self._advance_np(p)

    def _advance_np(self, p):
        """The numpy reference: ``ArcProgress.advance``'s body, per env.

        The candidate window is gathered rather than precomputed - the
        single-line class can afford one (L, 2w+1, 3) strided view of an
        edge-padded copy, but per env that is (N, l_max, 2w+1, 3) = 623 MB at
        the training shape. Slots off either end of the line are masked to
        +inf, which returns the same anchor as the edge padding does (see the
        module docstring).
        """
        w = self.window
        rows = self._rows
        j = self.idx[:, None] + self._k[None, :] - w
        valid = (j >= 0) & (j < self.length[:, None])
        cand = self.pts[rows[:, None], np.clip(j, 0, self.l_max - 1)]
        d2 = ((cand - p[:, None, :]) ** 2).sum(-1)
        d2 = np.where(valid, d2, np.float32(np.inf))
        i = np.clip(self.idx + d2.argmin(1) - w, 0, self.length - 1)
        arc, dd = self._refine(p, i, rows)
        inside = dd <= self.corridor * self.corridor
        delta = np.where(inside, arc - self.arc, 0.0)
        self.arc = np.where(inside, arc, self.arc)
        self.idx = np.where(inside, self._index(arc, rows), self.idx)
        return delta, inside
