"""ArcProgress's numba fast path must be BIT-IDENTICAL to the numpy reference.

Why bit-identity and not allclose: with --race-arc the arc coordinate IS the
reward. Every checkpoint, every corridor MAX in the ledger and every cross-arm
comparison assumes one definition of "how far along the route". A fast path
that were merely close would silently fork the shaping from every result
already recorded, and - because the arc term is integrated over ~700k row
updates a second - the divergence would compound into something that looks
exactly like an experimental effect.

`advance` is also stateful, which the goalfield test's pure `sample` is not:
the anchor freezes outside the corridor and the window clamps how far it may
move per call, so a one-shot comparison proves nothing. Everything below runs
a SEQUENCE on two independent ArcProgress objects fed identical inputs - one
pinned to the numpy body, one to the kernel - and compares the returns AND the
internal (arc, idx) state after every single step.

The reference is the numpy body of ArcProgress.advance; the fast path is
selected by route._FAST_ADVANCE, so setting that to None inside a test gives
the reference on the same object.
"""
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import surfgym.route as rt                                   # noqa: E402
from surfgym.route import ArcProgress, resample_polyline     # noqa: E402

SPACING = 128.0
CORRIDOR = 1500.0
WINDOW = 16

# Force BOTH paths regardless of the environment. SURFGYM_NO_NUMBA=1 nulls the
# module-level dispatcher at import time, and a bit-identity test that quietly
# compared numpy against numpy would be worse than no test at all.
_FAST = rt._FAST_ADVANCE if rt._FAST_ADVANCE is not None \
    else rt._build_fast_advance()
_needs_numba = pytest.mark.skipif(_FAST is None, reason="numba not installed")


@contextmanager
def _path(fast: bool):
    saved = rt._FAST_ADVANCE
    rt._FAST_ADVANCE = _FAST if fast else None
    try:
        yield
    finally:
        rt._FAST_ADVANCE = saved


# ---------------------------------------------------------------- geometry
def _spiral(n=400, radius=1500.0, rise=500.0, spacing=SPACING):
    """``n`` vertices of a 3D helix at constant ``spacing``.

    Successive turns are ``rise`` apart - well INSIDE a 1,500 u corridor - so
    the globally nearest vertex is regularly a whole turn away from the
    anchor, which is exactly the excursion the +/-window clamp exists to
    refuse. A straight line would never exercise it.
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

    Every square's centre is equidistant to THREE vertices and equidistant to
    BOTH adjacent segments, in exact binary arithmetic - the argmin tie-break
    and the ``d1 <= d2`` refine tie-break decide the answer there, and they
    decide it differently (the two segments have different arcs). Constructed
    on powers of two so the ties are bit-exact in float32, not almost-ties.
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


# ------------------------------------------------------------- the harness
def _pair(pts, corridor=CORRIDOR, window=WINDOW, spacing=SPACING):
    return (ArcProgress(pts, spacing, corridor=corridor, window=window),
            ArcProgress(pts, spacing, corridor=corridor, window=window))


def _identical(a, b, what):
    """Exact equality, one notch STRONGER than np.array_equal.

    A degenerate segment gives t = 0/0 = NaN and that NaN reaches the ``arc``
    state, where ``array_equal`` calls two identical NaNs unequal; a byte
    compare holds there, and also separates +0.0 from -0.0, which
    ``array_equal`` does not. Both are asserted where both apply, so this
    can never be weaker than the plain check.
    """
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} vs {b.dtype}"
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    assert a.tobytes() == b.tobytes(), f"{what} diverged (bitwise)"
    if a.dtype.kind != "f" or not np.isnan(a).any():
        assert np.array_equal(a, b), f"{what} diverged"


def _same_state(ref, fast, where):
    assert ref.arc.dtype == fast.arc.dtype == np.float64
    assert ref.idx.dtype == fast.idx.dtype == np.int64
    _identical(ref.arc, fast.arc, f"arc after {where}")
    _identical(ref.idx, fast.idx, f"idx after {where}")


def _reset_both(ref, fast, p, mask=None, where="reset"):
    with _path(False):
        ref.reset(p, mask)
    with _path(True):
        fast.reset(p, mask)
    _same_state(ref, fast, where)


def _advance_both(ref, fast, q, where):
    with _path(False):
        d_ref, i_ref = ref.advance(q)
    with _path(True):
        d_fast, i_fast = fast.advance(q)
    assert d_ref.dtype == d_fast.dtype == np.float64
    assert i_ref.dtype == i_fast.dtype == np.bool_
    _identical(d_ref, d_fast, f"delta at {where}")
    _identical(i_ref, i_fast, f"inside at {where}")
    _same_state(ref, fast, where)
    return d_ref, i_ref


def _walk(pts, rng, n, steps, near=400.0, far=6000.0, p_far=0.35):
    """A sequence that keeps most rows near the line and throws the rest well
    outside the corridor, so the paying branch and the frozen-anchor branch
    are both exercised on every step."""
    L = len(pts)
    v = rng.integers(0, L, n)
    out = []
    for _ in range(steps):
        v = np.clip(v + rng.integers(-4, 14, n), 0, L - 1)
        q = pts[v] + rng.normal(0.0, near, (n, 3))
        gone = rng.random(n) < p_far
        q[gone] += rng.normal(0.0, far, (int(gone.sum()), 3))
        out.append(np.ascontiguousarray(q, np.float64))
    return out


# ----------------------------------------------------------- the sequences
@_needs_numba
def test_bit_identical_over_a_random_sequence_on_a_spiral():
    """The main event: a few thousand query points per step, in and out of
    the corridor, over a line whose own turns sit inside the corridor."""
    pts = _spiral(400)
    rng = np.random.default_rng(0)
    ref, fast = _pair(pts)
    qs = _walk(pts, rng, 1500, 40)
    _reset_both(ref, fast, qs[0])
    seen_in = seen_out = seen_freeze = False
    clamped = 0
    for k, q in enumerate(qs):
        before_arc, before_idx = ref.arc.copy(), ref.idx.copy()
        delta, inside = _advance_both(ref, fast, q, f"spiral step {k}")
        seen_in |= bool(inside.any())
        seen_out |= bool((~inside).any())
        frozen = ~inside
        if frozen.any():
            seen_freeze |= bool(np.array_equal(ref.arc[frozen],
                                               before_arc[frozen])
                                and np.all(delta[frozen] == 0.0))
        clamped += int(np.sum(np.abs(ref.idx - before_idx) == WINDOW))
    # guards: a sequence that never left the corridor, or never hit the
    # window clamp, would pass this test without testing anything
    assert seen_in and seen_out and seen_freeze
    assert clamped > 0, "the window clamp was never reached"


@_needs_numba
def test_bit_identical_across_a_masked_reset_mid_sequence():
    """Respawns re-anchor a SUBSET of rows with a global search while the
    others keep advancing. The two paths must stay locked together across
    that, which is the state-machine half of the identity."""
    pts = _spiral(400)
    rng = np.random.default_rng(5)
    ref, fast = _pair(pts)
    qs = _walk(pts, rng, 512, 24)
    _reset_both(ref, fast, qs[0])
    for k, q in enumerate(qs):
        if k in (7, 8, 15):
            mask = rng.random(len(q)) < 0.4
            assert mask.any()
            _reset_both(ref, fast, q, mask, where=f"masked reset {k}")
        _advance_both(ref, fast, q, f"masked step {k}")
    # and a full (maskless) re-anchor still agrees
    _reset_both(ref, fast, qs[-1], where="full reset")
    _advance_both(ref, fast, qs[0], "after full reset")


@_needs_numba
def test_bit_identical_when_the_window_argmin_is_an_exact_tie():
    """numpy's argmin returns the FIRST minimum. Every query here sits on the
    perpendicular bisector of two adjacent vertices, so the two smallest
    window entries are bit-equal float32 and the tie-break IS the answer."""
    pts = _straight(200)
    ref, fast = _pair(pts)
    v = np.arange(40, 160)
    anchor = pts[v]
    _reset_both(ref, fast, anchor)
    assert np.array_equal(ref.idx, v)
    q = anchor.copy()
    q[:, 0] += 0.5 * SPACING                 # exactly between v and v+1
    q[:, 1] = 300.0
    # the tie is real, not almost: the reference window has TWO minima
    p32 = np.ascontiguousarray(q, np.float32)
    d2 = ((ref._win[ref.idx] - p32[:, None, :]) ** 2).sum(-1)
    assert np.all((d2 == d2.min(1, keepdims=True)).sum(1) == 2)
    _advance_both(ref, fast, q, "vertex bisector")


@_needs_numba
def test_bit_identical_on_a_staircase_of_three_way_ties():
    """Square centres on a right-angle staircase: three vertices equidistant
    AND both adjacent segments equidistant, exactly. Neither tie-break may
    drift."""
    pts = _stair(160)
    ref, fast = _pair(pts, corridor=CORRIDOR, window=8)
    v = np.arange(20, 140)
    _reset_both(ref, fast, pts[v])
    q = 0.5 * (pts[v] + pts[v + 2])          # the square's centre
    p32 = np.ascontiguousarray(q, np.float32)
    d2 = ((ref._win[ref.idx] - p32[:, None, :]) ** 2).sum(-1)
    assert np.all((d2 == d2.min(1, keepdims=True)).sum(1) >= 2)
    _advance_both(ref, fast, q, "staircase centre")
    # walk the ties for a while so the tie-break also drives the STATE
    for k in range(12):
        _advance_both(ref, fast, 0.5 * (pts[v + k] + pts[v + k + 2]),
                      f"staircase centre {k}")


@_needs_numba
def test_bit_identical_at_and_beyond_both_ends_of_the_line():
    """The clamps at vertex 0 and L-1 (and the L-2 segment clamp) are three
    separate branches in the kernel; run every one of them."""
    pts = _straight(64)
    lo, hi = pts[0], pts[-1]
    q = np.array([
        lo, hi,                                        # exactly on the ends
        lo - [1.0, 0.0, 0.0], hi + [1.0, 0.0, 0.0],    # one unit past
        lo - [5000.0, 0.0, 0.0], hi + [5000.0, 0.0, 0.0],   # far past
        lo + [0.0, 1400.0, 0.0], hi + [0.0, 1400.0, 0.0],   # lateral, inside
        lo - [64.0, 0.0, 0.0], hi + [64.0, 0.0, 0.0],  # half a spacing past
    ], np.float64)
    ref, fast = _pair(pts)
    _reset_both(ref, fast, q)
    for k in range(6):
        _advance_both(ref, fast, q, f"ends {k}")
        # drag the anchor to the far end and run the same points again, so
        # both the "window hangs off the front" and "off the back" pads run
        _reset_both(ref, fast, np.repeat(hi[None], len(q), 0),
                    where=f"ends re-anchor {k}")
        _advance_both(ref, fast, q, f"ends from the tail {k}")
        _reset_both(ref, fast, np.repeat(lo[None], len(q), 0),
                    where=f"ends re-anchor lo {k}")


@_needs_numba
def test_bit_identical_exactly_on_the_corridor_boundary():
    """dist == corridor decides inside/outside on a `<=` in FLOAT32. 1,500
    and 1,500^2 are both exact in float32, so this really is the boundary and
    not a nearby float."""
    pts = _straight(64)
    ref, fast = _pair(pts, corridor=CORRIDOR)
    c = np.float32(CORRIDOR)
    ys = np.array([np.nextafter(c, np.float32(0.0)), c,
                   np.nextafter(c, np.float32(1e30))], np.float32)
    v = np.arange(20, 23)
    q = pts[v].copy()
    q[:, 1] = ys.astype(np.float64)
    _reset_both(ref, fast, pts[v])
    _, inside = _advance_both(ref, fast, q, "corridor boundary")
    # the boundary itself is INSIDE (<=), one ulp out is not
    assert list(inside) == [True, True, False]
    # and the same three distances again from a frozen anchor
    _advance_both(ref, fast, q, "corridor boundary, repeat")


@_needs_numba
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_bit_identical_with_a_degenerate_zero_length_segment():
    """Duplicate vertices make t = 0/0, which numpy answers with NaN (and a
    RuntimeWarning) and then propagates through clip, so the row fails both
    `d1 <= d2` and the corridor test and pays nothing.

    This case found a live bug: numba's DEFAULT error_model raises
    ZeroDivisionError there, i.e. a route not built through
    resample_polyline (which dedups) would have crashed a rollout on the
    fast path and been fine on the reference. Hence error_model="numpy".
    """
    pts = _straight(64)
    pts[30] = pts[29]                        # a stationary sample survives
    ref, fast = _pair(pts)
    rng = np.random.default_rng(3)
    v = np.arange(24, 40)
    _reset_both(ref, fast, pts[v])           # exactly on the vertices
    assert np.isnan(ref.arc).any(), "the degenerate segment produced no NaN"
    for k in range(10):
        q = pts[v] + rng.normal(0.0, 600.0, (16, 3))
        _advance_both(ref, fast, q, f"degenerate {k}")
        _advance_both(ref, fast, pts[v], f"degenerate on-vertex {k}")


@_needs_numba
@pytest.mark.parametrize("poison", ["_win", "_p32", "arc", "idx"])
def test_a_foreign_dtype_declines_the_kernel_silently(poison):
    """The guard, not the kernel. Poisoning a dtype also changes what the
    NUMPY body computes (float64 segment math is not float32 segment math),
    so this cannot be a value comparison - what it checks is that the kernel
    is not entered, nothing raises, and the returns keep their contract."""
    pts = _spiral(120)
    ap = ArcProgress(pts, SPACING, corridor=CORRIDOR, window=WINDOW)
    rng = np.random.default_rng(9)
    q = [pts[rng.integers(0, 120, 32)] + rng.normal(0.0, 300.0, (32, 3))
         for _ in range(3)]
    ap.reset(q[0])

    calls = []

    def spy(*a, **kw):
        calls.append(1)
        return _FAST(*a, **kw)

    saved = rt._FAST_ADVANCE
    rt._FAST_ADVANCE = spy
    try:
        ap.advance(q[1])
        assert calls == [1], "the kernel is supposed to run by default"
        wrong = {"_win": np.float64, "_p32": np.float64,
                 "arc": np.float32, "idx": np.int32}[poison]
        setattr(ap, poison, getattr(ap, poison).astype(wrong))
        delta, inside = ap.advance(q[2])
    finally:
        rt._FAST_ADVANCE = saved
    assert calls == [1], f"{poison} as {wrong.__name__} entered the kernel"
    assert delta.shape == (32,) and inside.shape == (32,)
    assert inside.dtype == np.bool_


def test_the_numpy_reference_still_runs_with_the_fast_path_disabled():
    """Runs with or without numba: SURFGYM_NO_NUMBA=1 must not break the
    module, only slow it down."""
    pts = _spiral(120)
    ap = ArcProgress(pts, SPACING, corridor=CORRIDOR, window=WINDOW)
    with _path(False):
        ap.reset(pts[:16])
        delta, inside = ap.advance(pts[8:24])
    assert delta.shape == (16,) and inside.shape == (16,)
    assert bool(inside.all())
    assert float(delta.min()) > 0.0


# ---------------------------------------------------------------- the cost
@_needs_numba
def test_micro_benchmark_at_the_training_batch(capsys):
    """Not asserted - the ledger wants a number. N = 2,048 envs, L = 1,600
    vertices (204,800 u of route at 128 u), window 16: the shape --race-arc
    actually pays on a training box."""
    n, steps, reps = 2048, 16, 60
    pts = _spiral(1600)
    rng = np.random.default_rng(17)
    qs = _walk(pts, rng, n, steps, p_far=0.15)

    def bench(use_fast):
        ap = ArcProgress(pts, SPACING, corridor=CORRIDOR, window=WINDOW)
        with _path(use_fast):
            ap.reset(qs[0])
            for k in range(4):                    # warm up / JIT compile
                ap.advance(qs[k])
            best = float("inf")
            for _ in range(3):
                t0 = time.perf_counter()
                for k in range(reps):
                    ap.advance(qs[k % steps])
                best = min(best, (time.perf_counter() - t0) / reps)
        return best

    ref_s = bench(False)
    fast_s = bench(True)
    msg = (f"\narc advance N={n} L={len(pts)} window={WINDOW}: "
           f"numpy {ref_s * 1e6:8.1f} us/call, "
           f"numba {fast_s * 1e6:8.1f} us/call  "
           f"({ref_s / fast_s:.2f}x)")
    with capsys.disabled():
        print(msg)
