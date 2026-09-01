"""goal-conditioned RL: the per-env featurizer must BE RouteLine.

``MultiLine`` is a second copy of the observation every trained checkpoint
reads. If it drifts from ``RouteLine`` - a clamp applied per batch instead of
per env, a mask forgotten, a yaw sign flipped - the goal-conditioned arm is
not comparable to anything in the ledger, and nothing about the training
curve would say so. So the equivalence is asserted numerically here, env by
env, against the original.

The other half is the padding. A dense (N, l_max, 3) block means every index
into it has to be clamped by that env's OWN length, and a missed clamp reads
the tail: plausible geometry, silently wrong, only for the envs whose line is
short. ``test_padded_tail_is_never_read`` puts a decoy at distance zero in
the tail, which wins any unmasked argmin outright.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.goals import (AirSampler, GoalStats, KCurriculum,   # noqa: E402
                           MultiLine, SphereGoals, chord_line, segment_line)
from surfgym.route import RouteLine, resample_polyline           # noqa: E402

SPACING = 128.0


def helix(turns=4.0, radius=2000.0, drop=6000.0, n=1200):
    """A 3D line that curves in every axis, so a broadcast bug shows up."""
    th = np.linspace(0.0, turns * 2.0 * np.pi, n)
    xyz = np.stack([radius * np.cos(th), radius * np.sin(th),
                    np.linspace(0.0, -drop, n)], 1)
    pts, _ = resample_polyline(xyz, SPACING)
    return pts


def probe(n, seed):
    """Random ego states: speeds straddling the floor, points far off line."""
    rng = np.random.default_rng(seed)
    o = torch.tensor(rng.uniform(-4000.0, 4000.0, (n, 3)), dtype=torch.float32)
    o[: n // 8] += torch.tensor([20000.0, -15000.0, 9000.0])   # far off route
    y = torch.tensor(rng.uniform(-720.0, 720.0, n), dtype=torch.float32)
    s = torch.tensor(np.concatenate([rng.uniform(0.0, 500.0, n // 2),
                                     rng.uniform(500.0, 3500.0, n - n // 2)]),
                     dtype=torch.float32)
    return o, y, s


# --------------------------------------------------------- RouteLine parity
def test_multiline_matches_routeline_for_a_shared_line():
    pts = helix()
    r = RouteLine(pts, SPACING)
    n = 64
    ml = MultiLine(n, l_max=len(pts) + 32, spacing=SPACING)
    ml.set_lines(np.arange(n), [pts] * n)
    assert ml.n_features == r.n_features == 27
    assert "27 features" in ml.describe()

    o, y, s = probe(n, 7)
    got = ml.features(o, y, s)
    assert got.shape == (n, ml.n_features)
    assert torch.allclose(got, r.features(o, y, s), atol=1e-5)
    assert float(ml.total_arc()[0]) == (len(pts) - 1) * SPACING


def test_multiline_per_env_lines_match_their_own_routeline():
    """Different line per env, different ego state per env, one at a time."""
    lines = [helix(),
             chord_line(np.array([0.0, 0.0, 0.0]),
                        np.array([9000.0, -3000.0, -4000.0]), SPACING),
             np.array([[100.0, 200.0, -50.0], [1400.0, 900.0, -700.0]],
                      np.float32)]
    n = 12
    ml = MultiLine(n, l_max=max(len(ln) for ln in lines) + 5, spacing=SPACING)
    ml.set_lines(np.arange(n), [lines[i % 3] for i in range(n)])
    o, y, s = probe(n, 11)
    got = ml.features(o, y, s)
    for i in range(n):
        one = RouteLine(lines[i % 3], SPACING)
        want = one.features(o[i:i + 1], y[i:i + 1], s[i:i + 1])
        assert torch.allclose(got[i:i + 1], want, atol=1e-5), \
            f"env {i} does not read its own line"
    arcs = ml.total_arc().numpy()
    assert arcs[2] == SPACING and arcs[0] > arcs[1] > arcs[2]


def test_padded_tail_is_never_read():
    short = chord_line(np.zeros(3), np.array([2000.0, 0.0, 0.0]), SPACING)
    n = 4
    ml = MultiLine(n, l_max=64, spacing=SPACING)
    ml.set_lines(np.arange(n), [short] * n)
    assert int(ml.length[0]) == len(short) < 64

    o = torch.tensor([[500.0, 300.0, -100.0], [0.0, 0.0, 0.0],
                      [1900.0, -20.0, 5.0], [-900.0, 900.0, 900.0]])
    y = torch.tensor([0.0, 45.0, -90.0, 181.0])
    s = torch.tensor([0.0, 900.0, 2500.0, 1200.0])
    want = ml.features(o, y, s)
    want_arc = ml.arc_position(o)

    tail = int(ml.length[0])
    ml.pts[:, tail:] = o[:, None, :]          # a decoy AT the query point
    assert torch.equal(ml.features(o, y, s), want)
    assert torch.equal(ml.arc_position(o), want_arc)
    ml.pts[:, tail:] = float("nan")           # and one that poisons any read
    assert torch.equal(ml.features(o, y, s), want)


def test_set_lines_reports_the_bad_length():
    ml = MultiLine(2, l_max=8, spacing=SPACING)
    with pytest.raises(ValueError, match="L=1"):
        ml.set_lines(np.array([0]), [np.zeros((1, 3), np.float32)])
    with pytest.raises(ValueError, match="L=9"):
        ml.set_lines(np.array([0]), [np.zeros((9, 3), np.float32)])
    ml.set_lines(np.zeros(0, np.int64), [])   # a reset that touched nothing
    assert int(ml.length[0]) == 2


# -------------------------------------------------------------------- lines
def test_chord_line_geometry():
    a = np.array([-1000.0, 500.0, 200.0])
    b = np.array([3000.0, -700.0, -1800.0])
    ln = chord_line(a, b, SPACING)
    assert ln.dtype == np.float32 and ln.ndim == 2 and ln.shape[1] == 3
    assert len(ln) >= 2
    assert np.allclose(ln[0], a, atol=1e-3)
    assert np.allclose(ln[-1], b, atol=1e-3)
    step = np.linalg.norm(np.diff(ln.astype(np.float64), axis=0), axis=1)
    assert np.allclose(step, step[0], atol=1e-3)         # constant arc length
    assert abs(step[0] - SPACING) <= 0.5 * SPACING       # near the request
    total = float(np.linalg.norm(b - a))
    t = (np.linalg.norm(ln - a, axis=1) / total)[:, None]
    assert np.allclose(ln, a + t * (b - a), atol=1e-2)   # collinear


def test_chord_line_degenerate_is_still_a_line():
    a = np.array([12.0, -3.0, 4.0])
    ln = chord_line(a, a.copy(), SPACING)
    assert ln.shape == (2, 3) and np.isfinite(ln).all()
    assert np.allclose(ln[0], a)
    assert np.linalg.norm(ln[1] - ln[0]) > 1e-6
    ml = MultiLine(1, l_max=8, spacing=SPACING)
    ml.set_lines(np.array([0]), [ln])
    f = ml.features(torch.zeros(1, 3), torch.zeros(1), torch.zeros(1))
    assert torch.isfinite(f).all()


def corner_count(line, deg=1.0):
    d = np.diff(line.astype(np.float64), axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    cos = np.clip((d[:-1] * d[1:]).sum(1), -1.0, 1.0)
    return int((np.degrees(np.arccos(cos)) > deg).sum())


def test_segment_line_drops_noise_and_keeps_the_shape():
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 1.0, 600)[:, None]
    base = np.concatenate([t * np.array([8000.0, 0.0, 0.0]),
                           np.array([8000.0, 0.0, 0.0])
                           + t * np.array([0.0, 6000.0, -3000.0])])
    noisy = base + rng.normal(0.0, 40.0, base.shape)      # jitter well < eps
    out = segment_line(noisy, SPACING, rdp_eps=512.0)

    assert out.dtype == np.float32 and out.shape[1] == 3
    assert np.allclose(out[0], noisy[0], atol=1e-2)       # endpoints kept
    assert np.allclose(out[-1], noisy[-1], atol=1e-2)
    # 1,200 noisy vertices collapse to an L: the only turn left is the corner
    # (two of them in the resampled line - the chord that straddles it)
    assert corner_count(out) <= 4
    step = np.linalg.norm(np.diff(out.astype(np.float64), axis=0), axis=1)
    assert step.max() <= SPACING * 1.05                   # resampled at 128 u
    assert step.min() >= SPACING * 0.5                    # corner cutting only
    arc_in = float(np.linalg.norm(np.diff(noisy, axis=0), axis=1).sum())
    assert float(step.sum()) <= arc_in

    straight = (np.linspace(0.0, 12000.0, 600)[:, None]
                * np.array([0.6, 0.8, 0.0]) + rng.normal(0.0, 40.0, (600, 3)))
    flat = segment_line(straight, SPACING, rdp_eps=512.0)
    assert corner_count(flat) == 0
    fstep = np.linalg.norm(np.diff(flat.astype(np.float64), axis=0), axis=1)
    assert np.allclose(fstep, fstep[0], atol=1e-2)


def test_segment_line_keeps_what_is_bigger_than_eps():
    """The tolerance is the whole contract: a 2,000 u detour is the route, a
    100 u one is noise, and eps is where the line between them is drawn."""
    leg = np.linspace(0.0, 4000.0, 200)[:, None] * np.array([1.0, 0.0, 0.0])
    tail = (np.linspace(5000.0, 9000.0, 200)[:, None]
            * np.array([1.0, 0.0, 0.0]))

    def with_bump(h):
        return np.concatenate([leg, np.array([[4000.0, 0.0, 0.0],
                                              [4500.0, h, 0.0],
                                              [5000.0, 0.0, 0.0]]), tail])

    kept = segment_line(with_bump(2000.0), SPACING, rdp_eps=512.0)
    assert np.linalg.norm(kept - np.array([4500.0, 2000.0, 0.0]),
                          axis=1).min() <= SPACING
    gone = segment_line(with_bump(100.0), SPACING, rdp_eps=512.0)
    assert float(np.abs(gone[:, 1]).max()) < 100.0


# --------------------------------------------------------------------- arc
def test_arc_position_is_distance_along_a_chord():
    """|b - a| is exactly 50 spacings, so the resampled step is exactly 128 u
    and the arc coordinate is directly comparable to a distance."""
    a = np.array([-1000.0, 500.0, 200.0])
    b = a + np.array([2304.0, 3072.0, -5120.0])            # norm 6,400 exactly
    ln = chord_line(a, b, SPACING)
    assert len(ln) == 51
    u = (b - a) / np.linalg.norm(b - a)

    want = np.arange(50) * SPACING + 32.0        # first quarter of a segment
    ml = MultiLine(len(want), l_max=128, spacing=SPACING)
    ml.set_lines(np.arange(len(want)), [ln] * len(want))
    pos = torch.tensor(a + want[:, None] * u, dtype=torch.float32)
    got = ml.arc_position(pos)
    assert got.dtype == torch.float32 and got.shape == (len(want),)
    assert np.allclose(got.numpy(), want, atol=1.0)

    # Everywhere else the coordinate is RouteLine's: the tangent refinement
    # looks only forward from the nearest vertex, so a point past the middle
    # of a segment snaps up to the next one. Bounded by spacing/2, and shared
    # with the fan by construction - the two must not disagree.
    dense = np.linspace(0.0, 6400.0, 401)
    md = MultiLine(len(dense), l_max=128, spacing=SPACING)
    md.set_lines(np.arange(len(dense)), [ln] * len(dense))
    err = md.arc_position(torch.tensor(a + dense[:, None] * u,
                                       dtype=torch.float32)).numpy() - dense
    assert err.min() > -1e-2 and err.max() <= SPACING / 2 + 1e-2


# ------------------------------------------------------------------- goals
def test_sphere_goals_hit():
    g = SphereGoals(5, radius=192.0)
    assert not g.hit(np.zeros((5, 3))).any()             # nothing set yet
    centers = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0],
                        [0.0, -500.0, 300.0]])
    g.set(np.array([0, 1, 2]), centers)
    o = np.zeros((5, 3), np.float32)
    o[0] = [100.0, 0.0, 0.0]                             # inside
    o[1] = [1192.0, 0.0, 0.0]                            # exactly on the shell
    o[2] = [0.0, -500.0, 493.0]                          # 1 u outside
    hit = g.hit(o)
    assert hit.dtype == bool and hit.shape == (5,)
    assert hit[0] and hit[1] and not hit[2]
    assert not hit[3] and not hit[4]                     # inactive envs

    g.active[3] = True                                   # active, centre NaN
    assert not g.hit(o)[3]
    g.clear(np.array([0]))
    assert not g.hit(o)[0]
    g.set(np.array([2]), centers[2:3], radius=1000.0)    # per-env radius
    assert g.hit(o)[2]


def test_air_sampler_respects_the_predicates():
    mins, maxs = np.full(3, -1000.0), np.full(3, 1000.0)
    rng = np.random.default_rng(0)
    s = AirSampler(mins, maxs, lambda p: p[:, 2] > 0.0)
    p = s.sample(200, rng)
    assert p.shape == (200, 3) and p.dtype == np.float32
    assert (p[:, 2] > 0.0).all()
    assert (p >= mins).all() and (p <= maxs).all()

    s2 = AirSampler(mins, maxs, lambda q: q[:, 2] > 0.0,
                    exclude_fn=lambda q: q[:, 0] > 0.0)
    q = s2.sample(100, rng)
    assert (q[:, 2] > 0.0).all() and (q[:, 0] <= 0.0).all()
    assert s.sample(0, rng).shape == (0, 3)


def test_air_sampler_raises_rather_than_returning_short():
    rng = np.random.default_rng(0)
    dead = AirSampler(np.full(3, -1.0), np.full(3, 1.0),
                      lambda p: np.zeros(len(p), bool), max_tries=3)
    with pytest.raises(RuntimeError, match="only 0 of 4"):
        dead.sample(4, rng)


def test_air_sampler_near_stays_in_the_shell():
    box = np.full(3, 1e5)
    s = AirSampler(-box, box, lambda p: np.ones(len(p), bool))
    anchor = np.array([100.0, -200.0, 300.0])
    p = s.sample_near(500, anchor, 1000.0, 4000.0, np.random.default_rng(1))
    d = np.linalg.norm(p.astype(np.float64) - anchor, axis=1)
    assert p.shape == (500, 3)
    assert d.min() >= 1000.0 - 1e-1 and d.max() <= 4000.0 + 1e-1
    assert d.max() - d.min() > 2000.0                    # the whole shell


# -------------------------------------------------------------- curriculum
def test_k_curriculum_moves_with_the_frontier():
    c = KCurriculum(k_min=1.0, k_max=5.0, step=2.0, min_episodes=8,
                    lo=0.10, hi=0.50)
    d = c.draw(1000, np.random.default_rng(0))
    assert d.dtype == np.float64 and d.shape == (1000,)
    assert (d >= 1.0).all() and (d <= 5.0).all()

    # only the top third votes: 50 easy wins move nothing and are not counted
    for _ in range(50):
        c.note(1.5, True)
    assert c.update() == 5.0 and c.state()["k_n"] == 0
    # too few frontier episodes to read: hold, and keep accumulating
    c.note(4.5, True)
    assert c.update() == 5.0 and c.state()["k_n"] == 1
    for _ in range(7):
        c.note(4.5, True)
    assert c.update() == 7.0                             # rate 1.0 > hi
    # k_updates counts BAND EVALUATIONS, so the two that held for want of
    # episodes do not appear in it - this is the first one that read anything
    assert c.state()["k_n"] == 0 and c.state()["k_updates"] == 1

    # failures at the new frontier (top third of [1, 7] starts at 5.0)
    c.note(np.full(8, 6.0), np.zeros(8, bool))
    assert c.update() == 5.0
    # in between: nothing moves
    c.note(np.full(8, 4.5), np.array([1, 1, 0, 0, 0, 0, 0, 0], bool))
    assert c.update() == 5.0                             # rate 0.25, lo..hi


def test_k_curriculum_respects_cap_and_floor():
    c = KCurriculum(k_min=1.0, k_max=5.0, k_cap=6.0, k_floor=2.0, step=2.0,
                    min_episodes=4)
    for _ in range(6):
        c.note(np.full(4, 100.0), np.ones(4, bool))
        c.update()
    assert c.k_max == 6.0
    for _ in range(6):
        c.note(np.full(4, 100.0), np.zeros(4, bool))
        c.update()
    assert c.k_max == 2.0
    d = c.draw(200, np.random.default_rng(2))
    assert (d >= 1.0).all() and (d <= 2.0).all()


# ------------------------------------------------------------------- stats
def test_goal_stats_pop():
    g = GoalStats()
    empty = g.pop()
    assert empty["n"] == 0
    assert np.isnan(empty["success_rate"]) and np.isnan(empty["ticks_mean"])
    assert empty["kind"]["air"]["n"] == 0
    assert np.isnan(empty["kind"]["air"]["success_rate"])
    assert list(empty["k_bins"]["n"]) == [0, 0, 0, 0, 0]

    g.note(1.0, "achieved", True, 60)
    g.note(3.0, "air", False, 400)
    g.note(12.0, "finish", True, 900)
    g.note(30.0, "air", True, 1200)
    assert len(g) == 4
    d = g.pop()
    assert d["n"] == 4
    assert d["success_rate"] == pytest.approx(0.75)
    assert d["ticks_mean"] == pytest.approx((60.0 + 900.0 + 1200.0) / 3.0)
    assert d["kind"]["air"] == {"n": 2, "success_rate": pytest.approx(0.5)}
    assert d["kind"]["achieved"]["n"] == 1
    assert d["kind"]["finish"]["success_rate"] == pytest.approx(1.0)
    assert list(d["k_bins"]["n"]) == [1, 1, 0, 1, 1]   # 0-2 2-5 5-10 10-20 20+
    assert d["k_bins"]["labels"][2] == "5-10"
    assert np.isnan(d["k_bins"]["success_rate"][2])       # no data, not zero
    assert d["k_bins"]["success_rate"][1] == pytest.approx(0.0)

    assert g.pop()["n"] == 0                             # cleared by pop
    with pytest.raises(ValueError, match="unknown goal kind"):
        g.note(1.0, "bogus", True, 1)
