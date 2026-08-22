"""The harvest margin: what a shorter one does, and what it must NOT do.

Round 18 established that the start-state family is capped by the reservoir's
CONTENTS, not by its sampling rule - the final descent kills within ~1.5 s of
being entered while `--respawn-margin` discards everything within 10 s of an
episode's end, so the wall's states essentially cannot be harvested. The
`xMARGIN` arm reduces that margin and nothing else, so these pin the three
properties the arm's honesty rests on:

  * the control's margin is a pinned constant (10 s) and a buffer built with
    it harvests exactly what it always did;
  * a shorter margin's harvest is a strict SUPERSET of the long margin's -
    which is why no explicit "mixed distribution" knob is needed: the change
    is an addition to the tail of an otherwise unchanged distribution, not a
    replacement of it;
  * the SAMPLING rule is untouched - identical reservoir contents and seed
    produce bit-identical spawn pools whatever margin the buffer was built
    with. (Round 15's `wMARGIN` is unusable because it moved a second thing;
    this pins that this one does not.)

Plus the reservoir-depth diagnostic `depth_report`, which for this arm is the
primary measurement: the arm is only meaningful if the low bins actually gain
states.

    python -m pytest tests/python/test_respawn_margin.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                       # noqa: E402
from surfgym.respawn import RespawnBuffer, depth_report    # noqa: E402


def _run_episode(rb, n_ticks, envs=1):
    """Drive one episode of n_ticks through the buffer; states carry their
    tick in origin[0] so the harvest is identifiable afterwards."""
    states = np.zeros(envs, dtype=STATE_DTYPE)
    for t in range(1, n_ticks + 1):
        states["origin"] = np.tile([float(t), 0.0, 0.0], (envs, 1))
        rb.observe(states, np.full(envs, t == n_ticks))
    return sorted(float(x) for x in rb._store[:rb.size]["origin"][:, 0])


# --------------------------------------------------------------------------
# the control constant
# --------------------------------------------------------------------------
def test_the_control_margin_is_a_pinned_ten_seconds():
    """Every baseline number in CLAUDE.md was measured at 10 s. If this
    default ever moves, the control curve stops applying and the ledger's
    comparisons are silently wrong."""
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert re.search(r"args\.respawn_margin\s*=\s*10\.0", src), \
        "train_fast's --respawn-margin fallback is no longer 10.0 seconds"
    arm = (ROOT / "tools" / "run_arm.sh").read_text(encoding="utf-8")
    assert '"respawn_margin": 10.0' in arm, \
        "run_arm.sh's pinned baseline config no longer says margin 10 s"


def test_control_margin_harvest_is_unchanged():
    """margin_ticks=1000 (the control), 3,000-tick episode, snapshots every
    100 -> exactly ticks 100..2000."""
    rb = RespawnBuffer(1, reservoir=1000, margin_ticks=1000, snap_every=100,
                       map_id="t")
    got = _run_episode(rb, 3000)
    assert got == [float(t) for t in range(100, 2001, 100)]


# --------------------------------------------------------------------------
# what a 2 s margin buys, and what it deliberately still refuses
# --------------------------------------------------------------------------
def test_two_second_margin_admits_the_ramp_and_excludes_the_freefall():
    """The measured wall: the descent's deepest ON-ROUTE point is reached
    ~1.3-1.5 s before the episode ends (the rest is free-fall). A 200-tick
    margin must admit a snapshot from BEFORE that fall and none from inside
    it."""
    n = 3000
    fall_starts = n - 150            # 1.5 s of free-fall
    rb = RespawnBuffer(1, reservoir=1000, margin_ticks=200, snap_every=100,
                       map_id="t")
    got = _run_episode(rb, n)
    assert got[-1] == 2800.0, f"deepest harvested tick {got[-1]}, want 2800"
    assert got[-1] < fall_starts, "a 2 s margin must not harvest the fall"
    # and the control cannot reach anywhere near it
    rb10 = RespawnBuffer(1, reservoir=1000, margin_ticks=1000, snap_every=100,
                         map_id="t")
    assert _run_episode(rb10, n)[-1] == 2000.0
    assert got[-1] - 2000.0 == 800.0, "8 s of track the control never sees"


def test_short_margin_harvest_is_a_strict_superset_of_the_long_one():
    """Why this arm needs no explicit mixture knob: reducing the margin ADDS
    the episode tail to an otherwise identical harvest. The old distribution
    survives intact, so the reservoir is already a mixture."""
    for n in (900, 1500, 2600, 3000, 7200):
        long_ = set(_run_episode(
            RespawnBuffer(1, reservoir=1000, margin_ticks=1000,
                          snap_every=100, map_id="t"), n))
        short = set(_run_episode(
            RespawnBuffer(1, reservoir=1000, margin_ticks=200,
                          snap_every=100, map_id="t"), n))
        assert long_ <= short, f"n={n}: margin 2 s dropped states margin 10 s kept"
        if n > 1200:
            # exactly the 8 s the control threw away, no more and no less
            assert len(short) - len(long_) == 8, f"n={n}: {len(short)-len(long_)}"


def test_the_added_tail_is_a_known_share_of_the_harvest():
    """The dose, stated so the ledger cannot round it. At the measured
    training episode length (2,600-3,200 ticks) a 2 s margin adds 8 snapshots
    to a 16-22 snapshot harvest - about a third of it - and a 3 s margin adds
    7, which is within three points of the same dose while stopping 1,920 u
    short of the wall. The depth argument picks the value, not the dose."""
    for n, want2, want3 in ((2600, 8 / 24, 7 / 23), (3200, 8 / 30, 7 / 29)):
        base = len(_run_episode(RespawnBuffer(
            1, reservoir=1000, margin_ticks=1000, snap_every=100,
            map_id="t"), n))
        for margin, want in ((200, want2), (300, want3)):
            short = len(_run_episode(RespawnBuffer(
                1, reservoir=1000, margin_ticks=margin, snap_every=100,
                map_id="t"), n))
            assert abs((short - base) / short - want) < 1e-9, \
                f"n={n} margin={margin}: added share {(short - base) / short}"


def test_a_margin_shorter_than_one_snapshot_still_harvests_nothing_extra():
    """Snapshots exist only every snap_every ticks, so the margin is
    quantized: below one snapshot interval it buys nothing more."""
    a = _run_episode(RespawnBuffer(1, reservoir=1000, margin_ticks=100,
                                   snap_every=100, map_id="t"), 3000)
    b = _run_episode(RespawnBuffer(1, reservoir=1000, margin_ticks=1,
                                   snap_every=100, map_id="t"), 3000)
    assert a == b == [float(t) for t in range(100, 2901, 100)]


def test_short_episodes_still_contribute_nothing_under_a_short_margin():
    rb = RespawnBuffer(1, reservoir=100, margin_ticks=200, snap_every=100,
                       map_id="t")
    assert _run_episode(rb, 250) == []


def test_stagnant_states_are_still_excluded_at_a_short_margin():
    """The stall regime is guarded by the stagnant mask, NOT by the margin -
    so shortening the margin must not start admitting stalled states."""
    rb = RespawnBuffer(1, reservoir=100, margin_ticks=200, snap_every=100,
                       map_id="t")
    states = np.zeros(1, dtype=STATE_DTYPE)
    stag = np.array([True])
    for t in range(1, 3001):
        rb.observe(states, np.array([t == 3000]), stagnant=stag)
    assert rb.size == 0


# --------------------------------------------------------------------------
# the sampling rule is NOT what this arm changes
# --------------------------------------------------------------------------
def test_margin_does_not_change_the_sampler():
    """Same reservoir contents, same seed, different margin -> bit-identical
    spawn pools. The arm changes what is harvested, never how it is drawn."""
    pools = []
    for margin in (1000, 200):
        rb = RespawnBuffer(1, reservoir=1000, margin_ticks=margin,
                           snap_every=100, map_id="t", seed=7)
        # bypass the harvest: load the SAME states into both buffers
        rows = np.zeros(400, dtype=STATE_DTYPE)
        rows["origin"] = np.stack(
            [np.arange(400, dtype=np.float32), np.zeros(400, np.float32),
             np.zeros(400, np.float32)], 1)
        rows["velocity"] = 100.0
        for r in rows:
            rb._push(r)
        start = np.zeros(8, dtype=STATE_DTYPE)
        pools.append(rb.build_pool(start, pool_size=256, fresh_frac=0.1))
    a, b = pools
    assert a.dtype == b.dtype and len(a) == len(b)
    for name in STATE_DTYPE.names:
        assert np.array_equal(a[name], b[name]), f"sampler moved on {name!r}"


# --------------------------------------------------------------------------
# the depth diagnostic - the arm's primary measurement
# --------------------------------------------------------------------------
def test_depth_report_bins_from_the_finish():
    d = np.array([0.0, 100.0, 5_000.0, 5_500.0, 9_999.0])
    txt = depth_report(d, dmax=10_000.0, bins=10)
    counts = [int(x) for x in txt.splitlines()[1].split(": ")[1].split()]
    assert len(counts) == 10 and sum(counts) == 5
    assert counts[0] == 2, "bin 0 must be the slice nearest the FINISH"
    assert counts[5] == 2 and counts[9] == 1


def test_depth_report_states_min_and_mean():
    d = np.array([10.0, 20.0, 30.0, 40.0])
    txt = depth_report(d, dmax=100.0, bins=4)
    assert "min 10" in txt and "mean 25" in txt and "(4 states)" in txt


def test_depth_report_pool_line_is_optional_and_counts_deep_draws():
    d = np.arange(1, 101, dtype=float) * 100.0
    assert "start pool" not in depth_report(d, dmax=10_000.0, bins=16)
    pool = np.array([100.0, 200.0, 5_000.0, 9_000.0])   # 2 of 4 in bins 0-1
    txt = depth_report(d, dmax=10_000.0, bins=16, pool_d=pool)
    assert "start pool d: min 100" in txt and "mean 3,575" in txt
    assert "50.00% in bins 0-1" in txt


def test_depth_report_survives_an_empty_reservoir():
    assert depth_report(np.array([]), dmax=1.0) == "reservoir d: empty"


def test_depth_report_is_ascii_only():
    """The user's console is cp1251; a non-ASCII glyph renders as '?'."""
    txt = depth_report(np.arange(50.0), dmax=100.0, bins=8,
                       pool_d=np.arange(10.0))
    txt.encode("ascii")
