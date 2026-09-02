"""MultiArcProgress must BE ArcProgress, once per env, to the last bit.

Two separate identities are asserted here and they fail in different ways:

1. **Against ArcProgress.** The per-env class is a generalization of the
   single-line one, not a second implementation of the same idea. If they
   disagree, then "arc progress" means two different things depending on
   whether the run is goal-conditioned, and no corridor MAX in the ledger is
   comparable across that boundary. Every test below drives a real
   ``ArcProgress`` on the same points and compares the returns AND the
   internal (arc, idx) state after every step - the class is a state machine
   (the anchor freezes outside the corridor, the window clamps how far it may
   move), so a one-shot comparison would prove nothing.

2. **numpy reference vs numba kernel.** With --race-arc the arc coordinate IS
   the reward, integrated over ~700k row updates a second; a fast path that
   were merely close would fork the shaping from every number already
   recorded while looking exactly like an experimental effect.

**No dtype alignment is needed anywhere.** Both classes return ``delta``
float64 and ``inside`` bool, and hold ``arc`` float64 / ``idx`` int64, so the
comparisons are byte-exact rather than ``atol``-exact. The float32 half of
the arithmetic (the window search and the segment projection) is float32 in
BOTH classes, in the same order, which is what makes that possible; the
dtypes themselves are asserted, so a future widening cannot slip past as a
silent tolerance.

The one deliberate exception, and the reason the reset origins below are
never ambiguous: ``ArcProgress.locate`` finds the globally nearest vertex
through the ``|R|^2 - 2 p.R`` expansion (one BLAS matmul over the shared
line), while ``MultiArcProgress._locate`` takes the difference directly in
float64 (there is no shared matmul when every env has its own line). Both are
float64 and the direct form is the more accurate, but they round differently,
so an EXACT tie between two candidate vertices could resolve either way.
Every re-anchor here is therefore done at a point whose nearest vertex wins
by hundreds of units - which is what a spawn is - and the equality stays
exact.
"""
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import surfgym.goalarc as ga                                  # noqa: E402
from surfgym.goalarc import MultiArcProgress                  # noqa: E402
from surfgym.route import ArcProgress, resample_polyline      # noqa: E402

SPACING = 128.0
CORRIDOR = 1500.0
WINDOW = 16

# Force BOTH paths regardless of the environment. SURFGYM_NO_NUMBA=1 nulls the
# module-level dispatcher at import time, and a bit-identity test that quietly
# compared numpy against numpy would be worse than no test at all.
_FAST = ga._FAST_ADVANCE if ga._FAST_ADVANCE is not None \
    else ga._build_fast_advance()
_needs_numba = pytest.mark.skipif(_FAST is None, reason="numba not installed")

# every equivalence test runs on the numpy reference AND on the kernel: the
# claim is about the CLASS, and a bug in either path is a bug in the class
_BOTH = pytest.mark.parametrize(
    "fast", [pytest.param(False, id="numpy"),
             pytest.param(True, id="numba", marks=_needs_numba)])


@contextmanager
def _path(fast: bool):
    saved = ga._FAST_ADVANCE
    ga._FAST_ADVANCE = _FAST if fast else None
    try:
        yield
    finally:
        ga._FAST_ADVANCE = saved


# ---------------------------------------------------------------- geometry
def _helix(n=400, radius=1500.0, rise=500.0, spacing=SPACING):
    """``n`` vertices of a 3D helix at constant ``spacing``.

    Successive turns are ``rise`` apart - well INSIDE a 1,500 u corridor - so
    the globally nearest vertex is regularly a whole turn (~74 vertices) away
    from the anchor, which is exactly the excursion the +/-window clamp
    exists to refuse. A straight line would never exercise it.
    """
    per_turn = float(np.hypot(2.0 * np.pi * radius, rise))
    turns = n * spacing / per_turn + 2.0
    th = np.linspace(0.0, 2.0 * np.pi * turns, int(400 * turns) + 2)
    raw = np.stack([radius * np.cos(th), radius * np.sin(th),
                    rise * th / (2.0 * np.pi)], 1)
    pts, _ = resample_polyline(raw, spacing)
    assert len(pts) >= n, f"helix gave {len(pts)} vertices, wanted {n}"
    return np.ascontiguousarray(pts[:n], np.float64)


def _straight(n=200, spacing=SPACING):
    p = np.zeros((n, 3), np.float64)
    p[:, 0] = np.arange(n) * spacing
    return p


def _stair(n=200, spacing=SPACING):
    """A right-angle staircase in z=0, legs of exactly ``spacing``.

    Every square's centre is equidistant to THREE vertices and to BOTH
    adjacent segments, in exact binary arithmetic - the argmin tie-break and
    the ``d1 <= d2`` refine tie-break decide the answer there, and they
    decide it differently (the two segments have different arcs).
    """
    p = np.zeros((n, 3), np.float64)
    x = y = 0.0
    for i in range(n):
        p[i] = (x, y, 0.0)
        if i % 2 == 0:
            x += spacing
        else:
            y += spacing
    return p


def _wiggle(rng, n, step=900.0, spacing=SPACING):
    """``n`` vertices of a resampled random walk - no symmetry at all."""
    k = int(n * spacing / step) + 8
    raw = np.cumsum(rng.normal(0.0, step, (k, 3)), 0)
    pts, _ = resample_polyline(raw, spacing)
    assert len(pts) >= n, f"wiggle gave {len(pts)} vertices, wanted {n}"
    return np.ascontiguousarray(pts[:n], np.float64)


# ------------------------------------------------------------- the harness
def _identical(a, b, what):
    """Exact equality, one notch STRONGER than np.array_equal.

    A degenerate segment gives t = 0/0 = NaN and that NaN reaches the ``arc``
    state, where ``array_equal`` calls two identical NaNs unequal; a byte
    compare holds there, and also separates +0.0 from -0.0.
    """
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} vs {b.dtype}"
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    assert a.tobytes() == b.tobytes(), f"{what} diverged (bitwise)"
    if a.dtype.kind != "f" or not np.isnan(a).any():
        assert np.array_equal(a, b), f"{what} diverged"


def _same_state(ref, mp, where, rows=None):
    """``ref`` may be an ArcProgress or another MultiArcProgress."""
    assert ref.arc.dtype == mp.arc.dtype == np.float64
    assert ref.idx.dtype == mp.idx.dtype == np.int64
    if rows is None:
        _identical(ref.arc, mp.arc, f"arc after {where}")
        _identical(ref.idx, mp.idx, f"idx after {where}")
    else:
        _identical(ref.arc[rows], mp.arc[rows], f"arc after {where}")
        _identical(ref.idx[rows], mp.idx[rows], f"idx after {where}")


def _returns(d_ref, i_ref, d_mp, i_mp, where):
    assert d_ref.dtype == d_mp.dtype == np.float64
    assert i_ref.dtype == i_mp.dtype == np.bool_
    _identical(d_ref, d_mp, f"delta at {where}")
    _identical(i_ref, i_mp, f"inside at {where}")


def _spawn(lines, rng, jitter=40.0):
    """Unambiguous re-anchor points: a vertex plus a jitter far smaller than
    the gap to any other candidate. See the module docstring for why the
    global searches are allowed to round differently."""
    q = np.stack([ln[rng.integers(0, len(ln))] for ln in lines])
    return np.ascontiguousarray(q + rng.normal(0.0, jitter, q.shape),
                                np.float64)


def _walk(lines, rng, steps, near=400.0, far=6000.0, p_far=0.35):
    """Per-env sequences that drift FORWARD along each env's own line, with a
    third of the rows thrown well outside the corridor - so the paying branch
    and the frozen-anchor branch are both exercised on every step."""
    n = len(lines)
    lens = np.array([len(ln) for ln in lines])
    v = (rng.random(n) * lens).astype(np.int64)
    out = []
    for _ in range(steps):
        v = np.clip(v + rng.integers(-4, 14, n), 0, lens - 1)
        q = np.stack([lines[i][v[i]] for i in range(n)]).astype(np.float64)
        q += rng.normal(0.0, near, (n, 3))
        gone = rng.random(n) < p_far
        q[gone] += rng.normal(0.0, far, (int(gone.sum()), 3))
        out.append(np.ascontiguousarray(q, np.float64))
    return out


def _multi(lines, l_max=None, corridor=CORRIDOR, window=WINDOW):
    l_max = l_max or max(len(ln) for ln in lines) + 37
    mp = MultiArcProgress(len(lines), l_max=l_max, spacing=SPACING,
                          corridor=corridor, window=window)
    mp.set_lines(np.arange(len(lines)), lines)
    return mp


# ------------------------------------------- 1. one shared line == ArcProgress
@_BOTH
def test_matches_one_arcprogress_when_every_env_holds_the_same_line(fast):
    """N envs, one line, driven with the same origins as a single
    ArcProgress. Identical deltas, identical inside, identical state."""
    pts = _helix(400)
    n = 96
    rng = np.random.default_rng(0)
    ap = ArcProgress(pts, SPACING, corridor=CORRIDOR, window=WINDOW)
    mp = _multi([pts] * n, l_max=512)
    assert mp.total_arc().dtype == np.float64
    assert np.all(mp.total_arc() == (len(pts) - 1) * SPACING)
    assert np.all(mp.total_arc() == ap.length)

    spawn = _spawn([pts] * n, rng)
    ap.reset(spawn)
    with _path(fast):
        mp.reset(spawn)
    _same_state(ap, mp, "reset")

    seen_in = seen_out = seen_freeze = False
    for k, q in enumerate(_walk([pts] * n, rng, 40)):
        before = mp.arc.copy()
        d_a, i_a = ap.advance(q)
        with _path(fast):
            d_m, i_m = mp.advance(q)
        _returns(d_a, i_a, d_m, i_m, f"step {k}")
        _same_state(ap, mp, f"step {k}")
        seen_in |= bool(i_m.any())
        seen_out |= bool((~i_m).any())
        if (~i_m).any():
            seen_freeze |= bool(np.array_equal(mp.arc[~i_m], before[~i_m])
                                and np.all(d_m[~i_m] == 0.0))
    # a sequence that never left the corridor would pass without testing it
    assert seen_in and seen_out and seen_freeze


@_BOTH
def test_matches_arcprogress_across_masked_resets_mid_sequence(fast):
    """Respawns re-anchor a SUBSET of rows with a global search while the
    others keep advancing - the state-machine half of the identity."""
    pts = _helix(300)
    n = 64
    rng = np.random.default_rng(5)
    ap = ArcProgress(pts, SPACING, corridor=CORRIDOR, window=WINDOW)
    mp = _multi([pts] * n)
    spawn = _spawn([pts] * n, rng)
    ap.reset(spawn)
    with _path(fast):
        mp.reset(spawn)
    for k, q in enumerate(_walk([pts] * n, rng, 24)):
        if k in (7, 8, 15):
            mask = rng.random(n) < 0.4
            assert mask.any()
            s = _spawn([pts] * n, rng)
            ap.reset(s, mask)
            with _path(fast):
                mp.reset(s, mask)
            _same_state(ap, mp, f"masked reset {k}")
        d_a, i_a = ap.advance(q)
        with _path(fast):
            d_m, i_m = mp.advance(q)
        _returns(d_a, i_a, d_m, i_m, f"masked step {k}")
        _same_state(ap, mp, f"masked step {k}")


# ------------------------------------------------- 2. one line PER env
@_BOTH
def test_every_env_matches_its_own_single_line_arcprogress(fast):
    """Eight different lines of five different LENGTHS, round-robin over the
    envs. Env i is compared against an ArcProgress holding exactly line i,
    which is the only way to catch a per-env index that silently reads the
    neighbouring row of the padded block."""
    rng = np.random.default_rng(11)
    pool = [_helix(400), _helix(233, radius=900.0, rise=1200.0),
            _straight(150), _stair(120), _wiggle(rng, 300),
            _helix(88, radius=2600.0, rise=200.0), _wiggle(rng, 140),
            _straight(64)]
    n = 24
    lines = [pool[i % len(pool)] for i in range(n)]
    assert len({len(ln) for ln in lines}) >= 5
    mp = _multi(lines)
    aps = [ArcProgress(ln, SPACING, corridor=CORRIDOR, window=WINDOW)
           for ln in lines]
    assert np.array_equal(mp.total_arc(),
                          np.array([a.length for a in aps], np.float64))
    assert np.array_equal(mp.length,
                          np.array([len(ln) for ln in lines], np.int32))

    spawn = _spawn(lines, rng)
    for i, ap in enumerate(aps):
        ap.reset(spawn[i:i + 1])
    with _path(fast):
        mp.reset(spawn)
    for i, ap in enumerate(aps):
        _identical(ap.arc, mp.arc[i:i + 1], f"env {i} arc after reset")
        _identical(ap.idx, mp.idx[i:i + 1], f"env {i} idx after reset")

    for k, q in enumerate(_walk(lines, rng, 30)):
        with _path(fast):
            d_m, i_m = mp.advance(q)
        for i, ap in enumerate(aps):
            d_a, i_a = ap.advance(q[i:i + 1])
            _returns(d_a, i_a, d_m[i:i + 1], i_m[i:i + 1], f"env {i} @ {k}")
            _identical(ap.arc, mp.arc[i:i + 1], f"env {i} arc @ {k}")
            _identical(ap.idx, mp.idx[i:i + 1], f"env {i} idx @ {k}")


# ------------------------------------------------------------- 3. the padding
@_BOTH
def test_garbage_beyond_length_never_influences_the_result(fast):
    """The padded tail of a short line is the previous line's geometry - it
    can sit anywhere, including nearer the player than any real vertex. Poison
    it with the query point itself (distance 0, which wins every argmin) and
    with NaN (which numpy's argmin treats as MINIMAL and returns immediately),
    then require bit-identical output to an unpoisoned twin.

    The anchors sit at the END of each line on purpose, so the +/-window
    search really does reach into the pad; that is asserted, or the test
    would pass by never looking."""
    rng = np.random.default_rng(23)
    lines = [_helix(120), _helix(64, radius=900.0), _straight(48),
             _wiggle(rng, 90), _stair(40), _helix(31, rise=1500.0),
             _straight(200), _wiggle(rng, 55)]
    clean = _multi(lines, l_max=256)
    dirty = _multi(lines, l_max=256)
    n = len(lines)

    # anchor at the last vertex of each line, then query just short of it
    ends = np.stack([ln[-1] for ln in lines]).astype(np.float64)
    q = ends + rng.normal(0.0, 300.0, ends.shape)
    for mp in (clean, dirty):
        with _path(fast):
            mp.reset(ends)
    assert np.array_equal(clean.idx, clean.length.astype(np.int64) - 1)
    assert np.all(clean.idx + WINDOW >= clean.length), \
        "the window never reaches the pad - the test proves nothing"

    for i in range(n):
        L = int(dirty.length[i])
        dirty.pts[i, L:] = q[i].astype(np.float32)   # distance ~0
        dirty.pts[i, L + 1] = np.nan                 # numpy argmin: minimal
        dirty.pts[i, min(L + 2, dirty.l_max - 1)] = -1e15
    assert not np.array_equal(clean.pts, dirty.pts)

    for k in range(8):
        qq = np.ascontiguousarray(q + rng.normal(0.0, 200.0, q.shape),
                                  np.float64)
        with _path(fast):
            d_c, i_c = clean.advance(qq)
            d_d, i_d = dirty.advance(qq)
        _returns(d_c, i_c, d_d, i_d, f"poisoned step {k}")
        _same_state(clean, dirty, f"poisoned step {k}")
    # and the GLOBAL search (reset) must mask the same tail
    s = _spawn(lines, rng)
    with _path(fast):
        clean.reset(s)
        dirty.reset(s)
    _same_state(clean, dirty, "poisoned reset")


# --------------------------------------------------------- 4. corridor freeze
@_BOTH
def test_far_off_the_line_pays_zero_and_freezes_the_anchor(fast):
    """Outside the corridor: delta is exactly 0.0, arc and idx do not move,
    and the row recovers when it comes back - which is what stops an
    off-route excursion being cashed on re-entry."""
    lines = [_helix(200), _straight(200), _wiggle(np.random.default_rng(2),
                                                  200)]
    mp = _multi(lines)
    ap = [ArcProgress(ln, SPACING, corridor=CORRIDOR, window=WINDOW)
          for ln in lines]
    on = np.stack([ln[60] for ln in lines]).astype(np.float64)
    with _path(fast):
        mp.reset(on)
    for i, a in enumerate(ap):
        a.reset(on[i:i + 1])
    arc0, idx0 = mp.arc.copy(), mp.idx.copy()

    off = on + np.array([0.0, 0.0, 9.0 * CORRIDOR])
    for k in range(5):
        with _path(fast):
            d, ins = mp.advance(off)
        assert not ins.any(), "9x the corridor is not inside it"
        assert np.array_equal(d, np.zeros(len(lines)))
        assert np.array_equal(mp.arc, arc0) and np.array_equal(mp.idx, idx0)
        for i, a in enumerate(ap):
            d_a, i_a = a.advance(off[i:i + 1])
            _returns(d_a, i_a, d[i:i + 1], ins[i:i + 1], f"off {k} env {i}")

    # back on the line one vertex further: the frozen anchor pays the step it
    # actually made, not the excursion
    back = np.stack([ln[61] for ln in lines]).astype(np.float64)
    with _path(fast):
        d, ins = mp.advance(back)
    assert ins.all()
    assert np.allclose(d, SPACING, atol=1e-3)
    for i, a in enumerate(ap):
        d_a, i_a = a.advance(back[i:i + 1])
        _returns(d_a, i_a, d[i:i + 1], ins[i:i + 1], f"back env {i}")


# ------------------------------------------------------------ 5. window clamp
@_BOTH
def test_a_jump_past_the_window_is_clamped_exactly_like_arcprogress(fast):
    """One turn of the helix is ~74 vertices and only ~500 u away in space,
    so a player who teleports a turn ahead is INSIDE the corridor of a vertex
    74 ahead of the anchor. A global search would hand him 74 * 128 = 9,472 u
    of free arc; the +/-16 window must refuse it, and must refuse it the same
    way ArcProgress does."""
    pts = _helix(400)
    n = 32
    per_turn = int(round(float(np.hypot(2.0 * np.pi * 1500.0, 500.0))
                         / SPACING))
    assert per_turn > WINDOW * 2
    v = np.arange(60, 60 + n)
    ap = ArcProgress(pts, SPACING, corridor=CORRIDOR, window=WINDOW)
    mp = _multi([pts] * n, l_max=512)
    here = np.ascontiguousarray(pts[v], np.float64)
    ap.reset(here)
    with _path(fast):
        mp.reset(here)
    assert np.array_equal(mp.idx, v)
    before_arc, before_idx = mp.arc.copy(), mp.idx.copy()

    # the jump really is inside the corridor of a vertex a whole turn ahead
    ahead = np.ascontiguousarray(pts[v + per_turn], np.float64)
    assert np.all(np.linalg.norm(ahead - here, axis=1) < CORRIDOR)

    d_a, i_a = ap.advance(ahead)
    with _path(fast):
        d_m, i_m = mp.advance(ahead)
    _returns(d_a, i_a, d_m, i_m, "one-turn jump")
    _same_state(ap, mp, "one-turn jump")
    # the anchor moved by at most the window (+1 for the refine's rounding)
    assert np.all(np.abs(mp.idx - before_idx) <= WINDOW + 1)
    assert np.all(np.abs(mp.arc - before_arc) <= (WINDOW + 1) * SPACING)
    assert np.all(np.abs(d_m) <= (WINDOW + 1) * SPACING)
    assert float(np.abs(d_m).max()) < 0.5 * per_turn * SPACING

    # and a legal step of exactly `window` vertices IS paid, so the clamp is
    # not just freezing everything
    step = np.ascontiguousarray(pts[v + WINDOW], np.float64)
    ap.reset(here)
    with _path(fast):
        mp.reset(here)
        d_m, i_m = mp.advance(step)
    d_a, i_a = ap.advance(step)
    _returns(d_a, i_a, d_m, i_m, "legal window step")
    assert i_m.all()
    assert np.allclose(d_m, WINDOW * SPACING, atol=1e-3)


# --------------------------------------------------------------- 6. set_lines
@_BOTH
def test_set_lines_mid_run_re_anchors_only_the_given_envs(fast):
    """A new line invalidates that env's anchor and NOBODY else's. The
    untouched envs must keep advancing bit-identically to a twin that never
    saw the call, and the replaced envs must behave like a fresh ArcProgress
    on the new line."""
    rng = np.random.default_rng(31)
    n = 16
    old = _helix(300)
    new = _wiggle(rng, 220)
    keep = _multi([old] * n, l_max=512)
    mp = _multi([old] * n, l_max=512)
    spawn = _spawn([old] * n, rng)
    with _path(fast):
        keep.reset(spawn)
        mp.reset(spawn)
    qs = _walk([old] * n, rng, 12)
    for k, q in enumerate(qs[:6]):
        with _path(fast):
            keep.advance(q)
            mp.advance(q)
    _same_state(keep, mp, "before set_lines")

    hit = np.array([1, 4, 5, 11])
    rest = np.setdiff1d(np.arange(n), hit)
    where = np.ascontiguousarray(new[np.array([0, 40, 41, 199])], np.float64)
    mp.set_lines(hit, [new] * len(hit), origin=where)

    # untouched envs: line, length and anchor all exactly as they were
    assert np.array_equal(keep.pts[rest], mp.pts[rest])
    assert np.array_equal(keep.length[rest], mp.length[rest])
    _same_state(keep, mp, "after set_lines", rows=rest)
    # touched envs: the new line, and the anchor a fresh ArcProgress gives
    assert np.array_equal(mp.length[hit],
                          np.full(len(hit), len(new), np.int32))
    ref = ArcProgress(new, SPACING, corridor=CORRIDOR, window=WINDOW)
    ref.reset(where)
    _identical(ref.arc, mp.arc[hit], "re-anchored arc")
    _identical(ref.idx, mp.idx[hit], "re-anchored idx")

    # keep driving: the untouched envs still track their twin exactly, and
    # the replaced ones track the fresh ArcProgress
    for k, q in enumerate(qs[6:]):
        q = np.ascontiguousarray(q, np.float64)
        q[hit] = new[np.clip(np.array([2, 42, 45, 198]) + k, 0,
                             len(new) - 1)]
        with _path(fast):
            keep.advance(q)
            d_m, i_m = mp.advance(q)
        _same_state(keep, mp, f"after set_lines step {k}", rows=rest)
        d_r, i_r = ref.advance(q[hit])
        _returns(d_r, i_r, d_m[hit], i_m[hit], f"replaced env step {k}")
        _identical(ref.arc, mp.arc[hit], f"replaced arc step {k}")
        _identical(ref.idx, mp.idx[hit], f"replaced idx step {k}")


@_BOTH
def test_set_lines_without_an_origin_anchors_at_the_start_of_the_new_line(
        fast):
    """The no-origin form is for a line BUILT from the player's position (a
    chord to a fresh goal starts under his feet), so arc 0 / idx 0."""
    rng = np.random.default_rng(37)
    n = 8
    lines = [_helix(200)] * n
    mp = _multi(lines, l_max=400)
    with _path(fast):
        mp.reset(_spawn(lines, rng))
    assert float(mp.arc.max()) > 0.0
    hit = np.array([0, 3, 7])
    rest = np.setdiff1d(np.arange(n), hit)
    before_arc = mp.arc.copy()
    new = _wiggle(rng, 120)
    mp.set_lines(hit, [new] * len(hit))
    assert np.array_equal(mp.arc[hit], np.zeros(len(hit)))
    assert np.array_equal(mp.idx[hit], np.zeros(len(hit), np.int64))
    assert np.array_equal(mp.arc[rest], before_arc[rest])
    # the first advance from the line's start pays forward motion
    q = np.repeat(new[0][None], n, 0).astype(np.float64)
    q[hit] = new[6]
    with _path(fast):
        d, ins = mp.advance(q)
    assert ins[hit].all()
    assert np.allclose(d[hit], 6 * SPACING, atol=1e-3)


def test_set_lines_rejects_what_multiline_rejects():
    """Same contract as MultiLine.set_lines, including the failures."""
    mp = MultiArcProgress(4, l_max=32, spacing=SPACING)
    good = _straight(8)
    with pytest.raises(ValueError):
        mp.set_lines([0, 1], [good])                    # count mismatch
    with pytest.raises(ValueError):
        mp.set_lines([4], [good])                       # env out of range
    with pytest.raises(ValueError):
        mp.set_lines([0], [good[:1]])                   # L < 2
    with pytest.raises(ValueError):
        mp.set_lines([0], [_straight(33)])              # L > l_max
    with pytest.raises(ValueError):
        mp.set_lines([0], [good[:, :2]])                # not (L, 3)
    with pytest.raises(ValueError):
        mp.set_lines([0], [good], origin=np.zeros((3, 3)))
    mp.set_lines([0], [good])                           # and the happy path
    assert int(mp.length[0]) == 8
    with pytest.raises(ValueError):
        mp.advance(np.zeros((3, 3)))                    # wrong row count
    with pytest.raises(ValueError):
        mp.reset(np.zeros((5, 3)))


# ------------------------------------------------- 7. numpy vs numba identity
@_needs_numba
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_the_two_paths_are_bit_identical_over_a_random_fuzz():
    """A few thousand row-updates over random lines, random lengths, exact
    ties and a degenerate zero-length segment, comparing the numpy reference
    against the kernel after every step - returns AND state."""
    rng = np.random.default_rng(101)
    n, steps = 128, 40
    updates = 0
    for trial in range(3):
        lines, dead = [], {}
        for i in range(n):
            kind = i % 4
            if kind == 0:
                lines.append(_wiggle(rng, int(rng.integers(40, 260))))
            elif kind == 1:
                lines.append(_helix(int(rng.integers(30, 300)),
                                    radius=float(rng.uniform(600, 2600)),
                                    rise=float(rng.uniform(200, 1500))))
            elif kind == 2:
                lines.append(_stair(int(rng.integers(20, 120))))
            else:
                ln = _straight(int(rng.integers(20, 200)))
                # a duplicate vertex survives a hand-built line: t = 0/0 is
                # NaN in numpy and would be ZeroDivisionError under numba's
                # default error model
                m = len(ln) // 2
                ln[m] = ln[m - 1]
                dead[i] = m
                lines.append(ln)
        ref = _multi(lines, l_max=320)
        fast = _multi(lines, l_max=320)
        spawn = _spawn(lines, rng)
        # anchor the degenerate rows ON the vertex before the duplicate: the
        # zero-length segment is then the SECOND one _refine tries, `d1 <= d2`
        # is False against its NaN, and the NaN reaches the arc state - which
        # is the case that must round identically on both paths
        for i, m in dead.items():
            spawn[i] = lines[i][m - 1]
        with _path(False):
            ref.reset(spawn)
        with _path(True):
            fast.reset(spawn)
        _same_state(ref, fast, f"trial {trial} reset")
        assert np.isnan(ref.arc).any(), "no degenerate segment was hit"

        qs = _walk(lines, rng, steps, p_far=0.3)
        for k, q in enumerate(qs):
            if k in (9, 21):                     # respawn a subset
                mask = rng.random(n) < 0.35
                s = _spawn(lines, rng)
                with _path(False):
                    ref.reset(s, mask)
                with _path(True):
                    fast.reset(s, mask)
                _same_state(ref, fast, f"trial {trial} masked reset {k}")
            if k % 7 == 3:                       # exact ties: on the vertex
                q = np.stack([lines[i][min(k, len(lines[i]) - 1)]
                              for i in range(n)]).astype(np.float64)
            if k % 7 == 5:                       # exact ties: the bisector
                q = np.stack([0.5 * (lines[i][min(k, len(lines[i]) - 2)]
                                     + lines[i][min(k + 1,
                                                    len(lines[i]) - 1)])
                              for i in range(n)]).astype(np.float64)
            q = np.ascontiguousarray(q, np.float64)
            with _path(False):
                d_r, i_r = ref.advance(q)
            with _path(True):
                d_f, i_f = fast.advance(q)
            _returns(d_r, i_r, d_f, i_f, f"trial {trial} step {k}")
            _same_state(ref, fast, f"trial {trial} step {k}")
            updates += n
    assert updates >= 3000, updates


@_needs_numba
def test_the_corridor_boundary_decides_in_float32_on_both_paths():
    """dist == corridor is INSIDE (`<=`), one ulp out is not, and the compare
    happens in float32 in both paths. 1,500 and 1,500^2 are exact in
    float32, so this really is the boundary."""
    pts = _straight(64)
    c = np.float32(CORRIDOR)
    ys = np.array([np.nextafter(c, np.float32(0.0)), c,
                   np.nextafter(c, np.float32(1e30))], np.float32)
    lines = [pts, pts, pts]
    v = np.arange(20, 23)
    q = np.ascontiguousarray(pts[v], np.float64)
    q[:, 1] = ys.astype(np.float64)
    outs = []
    for use_fast in (False, True):
        mp = _multi(lines)
        with _path(use_fast):
            mp.reset(np.ascontiguousarray(pts[v], np.float64))
            outs.append(mp.advance(q))
    assert list(outs[0][1]) == [True, True, False]
    _returns(outs[0][0], outs[0][1], outs[1][0], outs[1][1], "boundary")


@_needs_numba
@pytest.mark.parametrize("poison", ["pts", "length", "arc", "idx"])
def test_a_foreign_dtype_declines_the_kernel_silently(poison):
    """The guard, not the kernel: a poisoned dtype must fall back to the
    reference without raising and keep the return contract. It cannot be a
    value comparison - float64 segment math is not float32 segment math."""
    lines = [_helix(120), _straight(90)]
    mp = _multi(lines)
    rng = np.random.default_rng(9)
    q = np.stack([ln[30] for ln in lines]).astype(np.float64)
    mp.reset(q)
    calls = []

    def spy(*a, **kw):
        calls.append(1)
        return _FAST(*a, **kw)

    saved = ga._FAST_ADVANCE
    ga._FAST_ADVANCE = spy
    try:
        mp.advance(q + rng.normal(0.0, 100.0, q.shape))
        assert calls == [1], "the kernel is supposed to run by default"
        wrong = {"pts": np.float64, "length": np.int64, "arc": np.float32,
                 "idx": np.int32}[poison]
        setattr(mp, poison, getattr(mp, poison).astype(wrong))
        delta, inside = mp.advance(q + rng.normal(0.0, 100.0, q.shape))
    finally:
        ga._FAST_ADVANCE = saved
    assert calls == [1], f"{poison} as {wrong.__name__} entered the kernel"
    assert delta.shape == (2,) and inside.shape == (2,)
    assert inside.dtype == np.bool_


def test_the_numpy_reference_still_runs_with_the_fast_path_disabled():
    """SURFGYM_NO_NUMBA=1 must not break the module, only slow it down."""
    lines = [_helix(120), _straight(90), _stair(40)]
    mp = _multi(lines)
    with _path(False):
        mp.reset(np.stack([ln[10] for ln in lines]).astype(np.float64))
        delta, inside = mp.advance(
            np.stack([ln[18] for ln in lines]).astype(np.float64))
    assert delta.shape == (3,) and inside.shape == (3,)
    assert bool(inside.all())
    assert np.allclose(delta, 8 * SPACING, atol=1e-3)
    assert "numpy" in _multi(lines).describe() or _FAST is not None


# ---------------------------------------------------------------- 8. the cost
@_needs_numba
def test_micro_benchmark_at_the_training_batch(capsys):
    """Not asserted - the ledger wants a number. N = 2,048 envs, l_max = 768
    vertices per env (98,304 u of line at 128 u), window 16: the shape a
    goal-conditioned --race-arc run pays every physics tick."""
    n, l_max, steps, reps = 2048, 768, 8, 30
    rng = np.random.default_rng(17)
    base = _helix(l_max)
    lens = rng.integers(l_max // 3, l_max + 1, n)
    lines = [base[:L] for L in lens]
    qs = _walk(lines, rng, steps, p_far=0.15)
    spawn = _spawn(lines, rng)

    def bench(use_fast):
        mp = _multi(lines, l_max=l_max)
        with _path(use_fast):
            mp.reset(spawn)
            for k in range(4):                     # warm up / JIT compile
                mp.advance(qs[k])
            best = float("inf")
            for _ in range(3):
                t0 = time.perf_counter()
                for k in range(reps):
                    mp.advance(qs[k % steps])
                best = min(best, (time.perf_counter() - t0) / reps)
        return best

    ref_s = bench(False)
    fast_s = bench(True)
    msg = (f"\nmultiarc advance N={n} l_max={l_max} window={WINDOW}: "
           f"numpy {ref_s * 1e6:8.1f} us/call, "
           f"numba {fast_s * 1e6:8.1f} us/call  "
           f"({ref_s / fast_s:.2f}x, "
           f"{n / fast_s / 1e6:.1f}M row-updates/s)")
    with capsys.disabled():
        print(msg)
