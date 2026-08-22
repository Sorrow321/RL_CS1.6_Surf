"""Necto / RLGym difficulty-weighted respawn draws (--respawn-difficulty).

Necto's replay state setter (Rolv-Arild/Necto, training/state.py,
``NectoReplaySetter.generate_probabilities``) is exactly two lines::

    weights = 1 + 10 * (ball_heights + player_heights.sum(-1)) / CEILING_Z
    return weights / weights.sum()

i.e. a PER-STATE linear weight on a normalized difficulty statistic, which
multiplies the replay pool's own density rather than flattening it. The
transplant here keeps the form and swaps the statistic for the state's
distance-bin FAILURE RATE (episodes started in the bin that ended without
ever improving on their start distance) - the one difficulty measure that is
still graded when the win rate is identically zero, which is what made round
16's Florensa arm degenerate to uniform.

What these pin:

  * the realized oversampling of the hardest bin over the easiest is
    1 + k per stored state (k = 10 -> 11x), on a synthetic reservoir;
  * the weight MULTIPLIES visitation density (a bin with 10x the states
    still gets ~10x the draws at equal difficulty) - it is not the
    bin-flattening the older --respawn-mode machinery does;
  * it needs no wins;
  * a flat/absent failure signal degrades to uniform instead of amplifying
    noise into an 11x curriculum;
  * FLAG OFF is bit-identical to the pre-existing sampler, on both the
    control path (no dist_fn) and the binned path, over successive draws
    (so the RNG stream is untouched, not merely the first pool);
  * RaceReward's episode-best latch, which supplies the failure signal,
    cannot influence any reward it returns.

    python -m pytest tests/python -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                       # noqa: E402
from surfgym.goalfield import EuclidField                  # noqa: E402
from surfgym.respawn import RespawnBuffer                  # noqa: E402
from surfgym.rewards import RaceReward                     # noqa: E402

K = 10.0            # Necto's constant
BINS = 16
DMAX = 1600.0       # -> 100u bins


def _rows(xs):
    rows = np.zeros(len(xs), STATE_DTYPE)
    rows["origin"][:, 0] = xs
    rows["velocity"][:, 0] = 100.0
    return rows


def _dist(o):
    return np.asarray(o)[:, 0]


def _buf(difficulty=K, bins=BINS, dist_valid_max=None, seed=23):
    return RespawnBuffer(4, reservoir=200_000, map_id="m", seed=seed,
                         dist_fn=_dist, dist_max=DMAX, bins=bins,
                         dist_valid_max=dist_valid_max,
                         difficulty=difficulty)


def _fill(rb, per_bin, bins=BINS):
    """`per_bin` states in the centre of each bin (or a per-bin count list)."""
    if np.isscalar(per_bin):
        per_bin = [int(per_bin)] * bins
    xs = np.concatenate([np.full(int(c), (b + 0.5) * DMAX / bins)
                         for b, c in enumerate(per_bin)])
    for k, row in enumerate(_rows(xs)):
        rb._push(row, float(xs[k]))
    return xs


def _set_rates(rb, rates, eps=200.0):
    """Plant a decayed failure-rate per bin directly (deterministic)."""
    rb.bin_ep[:] = eps
    rb.bin_fail[:] = np.asarray(rates, np.float64) * eps


def _draw(rb, n=200_000):
    """Reservoir draws only, as their distance-bin index."""
    n_fresh = max(1, int(round(n * 0.001)))
    pool = rb.build_pool(_rows(np.zeros(8)), pool_size=n, fresh_frac=0.001,
                         vel_scale=(1.0, 1.0), pitch_jitter=0.0)
    d = pool["origin"][:, 0][n_fresh:]           # concatenate([fresh, re])
    assert len(d) == n - n_fresh
    return np.clip(np.digitize(d, np.linspace(0.0, DMAX, BINS + 1)) - 1,
                   0, BINS - 1)


# --------------------------------------------------------------------------
# the weighting itself
# --------------------------------------------------------------------------
def test_hardest_bin_is_oversampled_one_plus_k_per_state():
    rb = _buf()
    _fill(rb, 500)                                    # equal population
    rates = np.linspace(0.05, 0.95, BINS)             # graded difficulty
    _set_rates(rb, rates)
    which = _draw(rb)
    drawn = np.bincount(which, minlength=BINS).astype(float)
    dens = drawn / 500.0                              # draws per stored state
    ratio = dens[BINS - 1] / dens[0]                  # hardest / easiest
    assert 1.0 + K == 11.0
    assert abs(ratio - 11.0) < 0.4, (ratio, dens)
    # and the whole curve is the linear 1 + k*normalized, not just its ends
    want = 1.0 + K * (rates - rates.min()) / (rates.max() - rates.min())
    assert np.allclose(dens / dens[0], want, rtol=0.05), dens / dens[0]
    # the buffer's own diagnostic must agree (it is what the run logs, and
    # what tells a null result apart from "the weighting never bit")
    assert abs(rb.diff_ratio - ratio) < 1e-6
    assert "11.0x realized" in rb.last_info or "10." in rb.last_info


def test_weight_multiplies_visitation_density_not_flattens_it():
    # equal difficulty, 10x the population -> 10x the draws. The older
    # --respawn-mode machinery flattens over bins; Necto's weight does not,
    # and that difference is the whole reason this is a one-factor arm.
    rb = _buf()
    counts = [0] * BINS
    counts[2], counts[9] = 200, 2000
    _fill(rb, counts)
    _set_rates(rb, np.full(BINS, 0.5))                # flat -> no difficulty
    drawn = np.bincount(_draw(rb), minlength=BINS).astype(float)
    assert abs(drawn[9] / drawn[2] - 10.0) < 0.5, drawn[[2, 9]]


def test_difficulty_is_defined_with_zero_wins():
    # round 16's Florensa arm degenerated because its statistic needed wins.
    rb = _buf()
    _fill(rb, 500)
    bins = np.repeat(np.arange(BINS), 400)
    rng = np.random.default_rng(0)
    p = np.linspace(0.05, 0.95, BINS)[bins]
    rb.note_outcomes(bins, np.zeros(len(bins), bool),
                     rng.random(len(bins)) < p)
    assert rb.bin_win.sum() == 0.0                    # no win ever
    dens = np.bincount(_draw(rb), minlength=BINS) / 500.0
    assert dens[BINS - 1] / dens[0] > 6.0, dens       # weighting still bites


def test_flat_failure_rates_degrade_to_uniform():
    rb = _buf()
    _fill(rb, 500)
    _set_rates(rb, np.full(BINS, 0.87))               # no spread at all
    dens = np.bincount(_draw(rb), minlength=BINS) / 500.0
    assert dens.max() / dens.min() < 1.1, dens


def test_unevaluated_bins_take_the_neutral_weight():
    rb = _buf()
    _fill(rb, 500)
    _set_rates(rb, np.linspace(0.05, 0.95, BINS))
    rb.bin_ep[7] = 1.0                                # below fail_min_ep
    rb.bin_fail[7] = 1.0
    D, rate, ev = rb._bin_difficulty()
    assert not ev[7] and ev.sum() == BINS - 1
    assert abs(D[7] - D[ev].mean()) < 1e-9            # neutral, never starved
    assert 0.0 < D[7] < 1.0


def test_sentinel_states_never_drawn_by_the_necto_path():
    rb = _buf(dist_valid_max=DMAX + 100.0)
    _fill(rb, 200)
    xs = np.full(300, DMAX + 500.0)                   # field sentinels
    for k, row in enumerate(_rows(xs)):
        rb._push(row, float(xs[k]))
    _set_rates(rb, np.linspace(0.05, 0.95, BINS))
    pool = rb.build_pool(_rows(np.zeros(8)), pool_size=4096, fresh_frac=0.1,
                         vel_scale=(1.0, 1.0), pitch_jitter=0.0)
    assert (pool["origin"][:, 0] > DMAX + 100.0).sum() == 0


def test_failure_rate_ema_tracks_the_fed_rate():
    rb = _buf()
    rng = np.random.default_rng(4)
    for _ in range(4000):
        b = np.array([3, 11])
        rb.note_outcomes(b, np.zeros(2, bool),
                         rng.random(2) < np.array([0.2, 0.9]))
    rate = np.divide(rb.bin_fail, rb.bin_ep, out=np.zeros(BINS),
                     where=rb.bin_ep > 0)
    assert abs(rate[3] - 0.2) < 0.12 and abs(rate[11] - 0.9) < 0.12


# --------------------------------------------------------------------------
# flag OFF must be the untouched code path
# --------------------------------------------------------------------------
def _reference_build_pool(rb, start_pool, pool_size, fresh_frac,
                          vel_scale, pitch_jitter):
    """The sampler as it stood before --respawn-difficulty existed, copied
    verbatim: same operations, same order, same RNG draws."""
    n_fresh = max(1, int(round(pool_size * fresh_frac)))
    if rb._size == 0:
        return start_pool
    n_re = pool_size - n_fresh
    idx = (rb._binned_pick(n_re) if rb._d is not None
           else rb.rng.integers(0, rb._size, n_re))
    re = rb._store[idx].copy()
    scale = rb.rng.uniform(vel_scale[0], vel_scale[1],
                           n_re).astype(np.float32)
    re["velocity"] = re["velocity"] * scale[:, None]
    re["pitch"] = np.clip(re["pitch"] + rb.rng.uniform(
        -pitch_jitter, pitch_jitter, n_re).astype(np.float32), -70.0, 30.0)
    fresh = start_pool[rb.rng.integers(0, len(start_pool), n_fresh)]
    return np.concatenate([fresh, re])


def _identical_over_successive_pools(make):
    a, b = make(), make()
    start = _rows(np.arange(8, dtype=float))
    for _ in range(5):     # successive calls: the RNG stream must not drift
        pa = a.build_pool(start, pool_size=1024, fresh_frac=0.1,
                          vel_scale=(1.0, 1.5), pitch_jitter=5.0)
        pb = _reference_build_pool(b, start, 1024, 0.1, (1.0, 1.5), 5.0)
        assert pa.tobytes() == pb.tobytes()


def test_flag_off_bit_identical_on_the_control_path():
    # the control (ckpt sOBSR2: respawn_binned 0, no mode) has NO dist_fn:
    # uniform-over-states, the branch --respawn-difficulty must not perturb
    def make():
        rb = RespawnBuffer(4, reservoir=5000, map_id="m", seed=23)
        for row in _rows(np.linspace(1.0, 1599.0, 3000)):
            rb._push(row)
        return rb
    _identical_over_successive_pools(make)


def test_flag_off_bit_identical_on_the_binned_path():
    def make():
        rb = RespawnBuffer(4, reservoir=5000, map_id="m", seed=23,
                           dist_fn=_dist, dist_max=DMAX, bins=BINS,
                           difficulty=0.0)
        xs = np.linspace(1.0, 1599.0, 3000)
        for k, row in enumerate(_rows(xs)):
            rb._push(row, float(xs[k]))
        return rb
    _identical_over_successive_pools(make)


def test_note_outcomes_without_fails_keeps_old_semantics():
    rng = np.random.default_rng(11)
    rb = _buf(difficulty=0.0)
    ep = np.zeros(BINS)
    win = np.zeros(BINS)
    for _ in range(200):
        bins = rng.integers(-1, BINS, 6)
        wins = rng.random(6) < 0.3
        rb.note_outcomes(bins, wins)
        for b, w in zip(bins[bins >= 0], wins[bins >= 0]):
            ep[b] = ep[b] * 0.99 + 1.0
            win[b] = win[b] * 0.99 + float(w)
    assert np.array_equal(rb.bin_ep, ep) and np.array_equal(rb.bin_win, win)
    assert not rb.bin_fail.any()


def test_difficulty_refuses_to_stack_with_a_mode():
    for bad in ("goex", "florensa", "backward"):
        try:
            RespawnBuffer(4, map_id="m", dist_fn=_dist, dist_max=DMAX,
                          mode=bad, difficulty=K)
        except ValueError as e:
            assert "two start-state curricula" in str(e)
        else:                                    # pragma: no cover
            raise AssertionError(f"{bad} + difficulty was accepted")


# --------------------------------------------------------------------------
# the failure signal's source: RaceReward's episode-best latch
# --------------------------------------------------------------------------
class FakeCore:
    def __init__(self, n):
        self.num_envs = n
        self.states_view = np.zeros(n, dtype=STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)

    def map_bounds(self):
        return (np.full(3, -2e4, np.float32), np.full(3, 2e4, np.float32))


def _race(n):
    # EuclidField to a goal at the origin: d == |pos|, so a scripted walk
    # has an exactly known best-ever d
    return RaceReward(EuclidField({"mins": np.zeros(3), "maxs": np.zeros(3)}),
                      scale=1.0, int_coef=0.0)


def test_episode_best_latch_cannot_change_any_reward():
    n = 3
    good, poisoned = _race(n), _race(n)
    cg, cp = FakeCore(n), FakeCore(n)
    rng = np.random.default_rng(5)
    zero = np.zeros(n, np.float32)
    for t in range(400):
        pos = rng.uniform(-3000.0, 3000.0, (n, 3)).astype(np.float32)
        done = rng.random(n) < 0.03
        trunc = rng.random(n) < 0.01
        for c in (cg, cp):
            c.states_view["origin"] = pos
            c.goal_hits[:] = 0
        rg = good(None, None, None, zero, done, trunc, cg)
        # poison the latch: if any reward read it, this diverges
        if poisoned._ep_best is not None:
            poisoned._ep_best[:] = np.nan
        rp = poisoned(None, None, None, zero, done, trunc, cp)
        assert rg.tobytes() == rp.tobytes(), t


def test_last_episode_best_is_the_episode_minimum():
    r = _race(1)
    c = FakeCore(1)
    zero = np.zeros(1, np.float32)
    off = np.zeros(1, bool)
    walk = [3000.0, 2000.0, 900.0, 1500.0, 2500.0]
    for x in walk:
        c.states_view["origin"] = np.array([[x, 0.0, 0.0]], np.float32)
        r(None, None, None, zero, off, off, c)
    c.states_view["origin"] = np.array([[7000.0, 0.0, 0.0]], np.float32)
    r(None, None, None, zero, np.ones(1, bool), off, c)   # died; autoreset
    assert abs(float(r.last_episode_best()[0]) - 900.0) < 1e-3
    # the new episode is armed from the fresh spawn, not the old minimum
    assert abs(float(r._best[0]) - 7000.0) < 1e-3
