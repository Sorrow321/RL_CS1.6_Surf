"""``--respawn-greedy F``: spawn a share of the training episodes on the policy's OWN
greedy line, in the seconds before each place it dies.

Why. A run is judged on GREEDY evals from the true start, and the greedy policy (the
mean view, the argmax keys) is never executed in training: the behaviour policy samples
at the unstuck temperature, and the reservoir is harvested from those stochastic
episodes. So the states the greedy line reaches just before it dies can be states that
training seldom visits with that position AND velocity, and nothing corrects the action
there. On sidistic (round 2) the stochastic policy kept crossing the CP2 pit (training
ceiling 29.76%) while every greedy eval for ~1.5B steps dived into it (27.4%). DAgger's
point (Ross, Gordon, Bagnell 2011): train on the states the DEPLOYED policy visits.

What. Every greedy eval's episodes are captured through the eval's ``on_tick`` hook (the
eval core's pre-step STATE_DTYPE snapshot of env 0, exactly what ``record_ckpt.py
--dump-states`` writes). For each episode that ended in a FAIL, the states from ``hi``
down to ``lo`` seconds before its last tick are kept (default 5.0 .. 0.5 s; the last
half second is mostly past saving); and for a FAIL or a truncated episode whose best
field progress came more than ``hi`` s before its end (a line that got stuck - evals
never stall-kill - or lost ground long before dying), the ``hi`` seconds up to that
best tick. Each iteration ``frac`` of the pool's NON-start rows
- the reservoir TAIL, so the map-start rows at the head and the novel rows right after
them are untouched - are replaced by uniform draws from those states, clocks zeroed
(``tick``, ``stuck_ticks``) as ``--spawn-states`` does. The pool is rebuilt at every
eval, so it follows the greedy line wherever it now dies; an eval with no failed
episode empties it (nothing to correct).

Generic. The policy's own states from the true start (CLAUDE.md section 0 allows
"windows cut from its own greedy line"), two constants in seconds, no map quantity
(section 0b). Episodes spawned from these rows are REAL episodes (harvested into the
reservoir, they move the novel ceiling): they are states the policy reached on its own
from the map start.
"""
from __future__ import annotations

import numpy as np

from .core import STATE_DTYPE


def as_truncated(summaries, idx) -> list:
    """``summaries`` (record_rollout's per-episode trailers) with the episodes at ``idx``
    marked ``end = "trunc"``. --eval-patience ends a greedy eval episode that stopped
    gaining as a FAIL (core.force_fail); for the pool that episode is the STUCK line it
    would have been without the rule (a truncated episode: only its best-progress anchor
    counts), not a death with decisions to correct in the seconds before it."""
    idx = set(int(i) for i in idx)
    return [dict(s, end="trunc") if i in idx else s for i, s in enumerate(summaries)]


def last_contact(ep, tail: int, tol: float = 1.0) -> int:
    """The index of the last state the map pushed back on, for an episode whose end is a
    ballistic flight: between contacts vz changes by one gravity step per tick, taken as the
    median vz step of the last ``tail`` ticks (tools/pick_selfline.contact_cut's rule, its step
    read off the final fall instead of the whole episode, which on a surf line is mostly ramp
    contact). A last ``tail`` that is not ballistic gives the last index (no flight)."""
    return last_contact_vz(np.asarray(ep["velocity"], np.float64).reshape(-1, 3)[:, 2],
                           tail, tol)


def last_contact_vz(vz, tail: int, tol: float = 1.0) -> int:
    """:func:`last_contact` on a per-tick vertical velocity array."""
    vz = np.asarray(vz, np.float64)
    n = len(vz)
    if n < 3 or tail < 2:
        return n - 1
    dvz = np.diff(vz)
    t = dvz[-min(int(tail), len(dvz)):]
    g = float(np.median(t))
    if np.abs(t - g).max() > tol or g >= 0.0:
        return n - 1
    hit = np.flatnonzero(np.abs(dvz - g) > tol)
    return int(hit[-1]) + 1 if len(hit) else 0


G_ENERGY = 800.0           # sv_gravity: h = z + |v|^2 / (2 g)


def energy_losses(ep, stop: int, per_s: float, secs: float, n: int, win: float = 0.1,
                  gap: float = 0.5) -> list:
    """[(tick, loss)] of the ``n`` largest losses of an episode's energy height over ``win``
    seconds, at least ``gap`` s apart, in its last ``secs`` s up to ``stop`` (tools/auto_spine's
    energy_roots rule); the tick is the largest one-tick loss inside each window."""
    o = np.asarray(ep["origin"], np.float64).reshape(-1, 3)
    v = np.asarray(ep["velocity"], np.float64).reshape(-1, 3)
    h = o[:, 2] + (v * v).sum(axis=1) / (2.0 * G_ENERGY)
    stop = int(min(max(int(stop), 1), len(h) - 1))
    w = max(1, int(round(win * per_s)))
    lo = max(0, stop - int(round(secs * per_s)))
    if stop - lo <= w:
        return []
    t = np.arange(lo, stop - w + 1)
    dh = h[t + w] - h[t]
    out, picked = [], []
    for tt in t[np.argsort(dh, kind="stable")]:
        if len(picked) >= int(n) or h[tt + w] - h[tt] >= 0.0:
            break
        if all(abs(int(tt) - p) >= gap * per_s for p in picked):
            picked.append(int(tt))
            seg = np.diff(h[tt:tt + w + 1])
            m = int(tt + int(np.argmin(seg)))
            out.append((m, float(h[m + 1] - h[m])))
    return out


class GreedyLinePool:
    def __init__(self, frac: float, window=(0.5, 5.0), seed: int = 0, stuck: int = 0,
                 turn: float = 0.0, speed=None, contact: bool = False, energy: int = 0,
                 energy_secs: float = 12.0):
        frac = float(frac)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"--respawn-greedy F needs 0 < F < 1, got {frac!r}")
        lo, hi = (float(v) for v in window)
        if not 0.0 <= lo < hi:
            raise ValueError(f"--respawn-greedy-window LO HI needs 0 <= LO < HI, got {lo!r} {hi!r}")
        if int(stuck) < 0:
            raise ValueError(f"--respawn-greedy-stuck N needs N >= 0, got {stuck!r}")
        if not 0.0 <= float(turn) <= 180.0:
            raise ValueError(f"--respawn-greedy-turn T needs 0 <= T <= 180, got {turn!r}")
        if speed is not None:
            speed = tuple(float(v) for v in speed)
            if not (len(speed) == 2 and 0.0 < speed[0] <= speed[1]):
                raise ValueError(f"--respawn-greedy-speed LO HI needs 0 < LO <= HI, got {speed!r}")
        self.frac = frac
        self.lo, self.hi = lo, hi
        # --respawn-greedy-stuck N: also N evenly spaced states of each episode's STALLED
        # stretch (its best-progress tick to its end) - where the deployed line wanders once it
        # stops gaining. 0 = off: the pool is the two anchors' windows, as before
        self.stuck = int(stuck)
        # --respawn-greedy-turn T / --respawn-greedy-speed LO HI: each mixed row's horizontal
        # velocity and view yaw turned together by U(-T, T) deg and its velocity scaled by
        # U(LO, HI) - the policy's own frontier states, nudged in heading and energy. Their
        # own RNG, so the rows DRAWN are exactly the unperturbed ones; off = no draw at all
        self.turn = float(turn)
        self.speed = speed
        # --respawn-greedy-contact: a FAIL that ends in a ballistic flight longer than LO is
        # anchored at its LAST MAP CONTACT (the launch) instead of its last tick - after the
        # last contact nothing but air strafing acts, so the decisions that led to the death
        # are the ones before it. Off = the death anchor, as before
        self.contact = bool(contact)
        # --respawn-greedy-energy N (pbr5b2, 2026-10-25): also anchor each episode at the N
        # largest losses of its energy height h = z + |v|^2 / (2 g) in its last ENERGY_SECS s
        # before its stop (its best-progress tick, else its end) - the HI..LO seconds before each
        # crash, where a line spends the energy a later climb needs (a princessburglar descent
        # lost ~1,200 u to three landings that the line seconds later could not make up). With
        # --respawn-greedy-temp those spawns explore there. 0 = off: no energy anchor
        if int(energy) < 0:
            raise ValueError(f"--respawn-greedy-energy N needs N >= 0, got {energy!r}")
        if not float(energy_secs) > 0.0:
            raise ValueError(f"--respawn-greedy-energy-secs S needs S > 0, got {energy_secs!r}")
        self.energy = int(energy)
        self.energy_secs = float(energy_secs)
        self._prng = np.random.default_rng(int(seed) + 7717)
        self.rng = np.random.default_rng(int(seed))
        self.rows = np.zeros(0, STATE_DTYPE)
        self.info = {"eps": 0, "fail": 0, "deaths_s": []}
        self.mixed = 0
        self._eps: list = []
        self._cur: list = []
        self._keys, self._keys_of = set(), None

    @property
    def size(self) -> int:
        return int(len(self.rows))

    # -- capture -------------------------------------------------------------
    def hook(self, n_envs: int = 0):
        """A fresh ``on_tick(t, states, rewards, done, trunc)`` for ONE eval recording:
        it keeps env 0's pre-step snapshot every tick and closes an episode on done or
        trunc (record_rollout's own rule), so episode i aligns with trailer i.

        ``n_envs`` > 0: the recording runs its episodes side by side
        (record_rollout_parallel) - env i's FIRST episode is episode i, captured until
        it ends; the env's later ticks (the core autoresets it in place) are ignored."""
        self._eps, self._cur = [], []
        if n_envs > 0:
            cur = [[] for _ in range(int(n_envs))]
            eps = [None] * int(n_envs)
            live = np.ones(int(n_envs), bool)
            self._par = (eps, cur)

            def on_tick_par(t, states, rewards, done, trunc):
                for i in np.flatnonzero(live):
                    cur[i].append(states[i].copy())
                    if bool(done[i]) or bool(trunc[i]):
                        eps[i] = np.array(cur[i], dtype=STATE_DTYPE)
                        cur[i] = []
                        live[i] = False
            return on_tick_par
        self._par = None

        def on_tick(t, states, rewards, done, trunc):
            self._cur.append(states[0].copy())
            if bool(done[0]) or bool(trunc[0]):
                self._eps.append(np.array(self._cur, dtype=STATE_DTYPE))
                self._cur = []
        return on_tick

    def build(self, summaries, tick_ms: float, d_fn=None) -> dict:
        """Rebuild the pool from the recording just captured. ``summaries`` is
        record_rollout's return (one trailer per episode, in order); ``tick_ms`` the eval
        core's realised mean tick. -> the build's stats.

        Two anchors per episode that did not finish:
        * a FAIL's last tick: the states HI..LO s before it (the decisions that led to
          the death; the last LO s are mostly past saving);
        * with ``d_fn`` (origins (n, 3) -> field distance, non-finite = off the field),
          the tick of the episode's BEST progress (its minimum d) when that lies more than
          HI s before the end: the states HI..0 s before it. That is where a line that
          got STUCK (a truncated episode: evals never stall-kill) or that lost ground
          long before dying stopped making progress."""
        par = getattr(self, "_par", None)
        if par is not None:
            # side by side: episode i is env i's first episode (one still running when the
            # recording stopped is what it captured so far)
            eps = [e if e is not None else np.array(c, dtype=STATE_DTYPE)
                   for e, c in zip(*par)]
            self._par = None
        else:
            eps = list(self._eps)
            if self._cur:                      # an episode cut by the recording budget
                eps.append(np.array(self._cur, dtype=STATE_DTYPE))
        self._eps, self._cur = [], []
        per_s = 1000.0 / float(tick_ms)
        lo_t, hi_t = int(round(self.lo * per_s)), int(round(self.hi * per_s))
        keep, deaths, stops, launches, losses, n_used = [], [], [], [], [], 0
        for i, ep in enumerate(eps):
            end = summaries[i].get("end") if i < len(summaries) else None
            n = len(ep)
            if end not in ("fail", "trunc") or not n:
                continue
            idx = []
            c = last_contact(ep, lo_t) if (self.contact and end == "fail") else n - 1
            if self.contact and end == "fail" and n - 1 - c > lo_t:
                # the launch and the HI - LO seconds before it
                idx.append(np.arange(max(0, c - (hi_t - lo_t)), c + 1))
                deaths.append(round(n / per_s, 2))
                launches.append(round(c / per_s, 2))
            elif end == "fail" and n - lo_t > max(0, n - hi_t):
                idx.append(np.arange(max(0, n - hi_t), n - lo_t))
                deaths.append(round(n / per_s, 2))
            stop = n - 1
            if d_fn is not None:
                d = np.asarray(d_fn(np.asarray(ep["origin"], np.float64)), np.float64)
                d = np.where(np.isfinite(d), d, np.inf)
                if np.isfinite(d).any():
                    tb = int(np.argmin(d))
                    stop = max(1, tb)
                    if tb < n - hi_t and tb > 0:
                        idx.append(np.arange(max(0, tb - hi_t), tb + 1))
                        stops.append(round(tb / per_s, 2))
                        if self.stuck > 0:
                            # the stalled stretch after the stop, evenly spaced
                            idx.append(np.unique(np.linspace(
                                tb, n - 1, num=min(self.stuck, n - tb)).astype(np.int64)))
            if self.energy > 0:
                for m, dh in energy_losses(ep, stop, per_s, self.energy_secs, self.energy):
                    if m - lo_t > max(0, m - hi_t):
                        idx.append(np.arange(max(0, m - hi_t), m - lo_t))
                        losses.append((round(m / per_s, 2), round(dh)))
            if idx:
                keep.append(ep[np.unique(np.concatenate(idx))])
                n_used += 1
        if keep:
            rows = np.concatenate(keep).astype(STATE_DTYPE, copy=True)
            rows["tick"] = 0
            rows["stuck_ticks"] = 0
        else:
            rows = np.zeros(0, STATE_DTYPE)
        self.rows = rows
        self.info = {"eps": len(eps), "fail": n_used, "deaths_s": deaths, "stops_s": stops}
        if self.contact:
            self.info["launches_s"] = launches
        if self.energy:
            self.info["energy_losses"] = losses
        return dict(self.info, rows=self.size)

    # -- the pool ------------------------------------------------------------
    def mix(self, pool: np.ndarray, n_fresh: int) -> np.ndarray:
        """Overwrite ``frac`` of the pool's non-start rows (the TAIL of
        ``[fresh (n_fresh) | novel | reservoir]``; the env resets by uniform pool draw, so
        counts are the probabilities) with uniform draws from the greedy-line states."""
        n = int(round((len(pool) - int(n_fresh)) * self.frac))
        if n <= 0 or not self.size:
            self.mixed = 0
            return pool
        out = np.array(pool, dtype=STATE_DTYPE, copy=True)
        out[len(out) - n:] = self.perturb(self.rows[self.rng.integers(0, self.size, n)])
        self.mixed = n
        return out

    def is_row(self, origins) -> np.ndarray:
        """Whether each origin is one of the pool's states (origins rounded to 0.1 u; a mixed
        row keeps its state's origin, --respawn-greedy-turn / -speed move only the velocity
        and the view) - how the trainer tells a greedy-line spawn from the rest of the pool
        (--respawn-greedy-temp)."""
        if self._keys_of is not self.rows:
            self._keys = {tuple(np.round(np.asarray(o, np.float64), 1))
                          for o in np.asarray(self.rows["origin"])}
            self._keys_of = self.rows
        return np.array([tuple(np.round(np.asarray(o, np.float64), 1)) in self._keys
                         for o in np.asarray(origins)], bool)

    def perturb(self, rows: np.ndarray) -> np.ndarray:
        """``rows`` with --respawn-greedy-turn / --respawn-greedy-speed applied (a copy);
        the rows themselves when both are off."""
        if self.turn <= 0.0 and self.speed is None:
            return rows
        rows = np.array(rows, dtype=STATE_DTYPE, copy=True)
        n = len(rows)
        v = rows["velocity"].astype(np.float64)
        if self.turn > 0.0:
            a = self._prng.uniform(-self.turn, self.turn, n)
            c, s_ = np.cos(np.radians(a)), np.sin(np.radians(a))
            vx, vy = v[:, 0].copy(), v[:, 1].copy()
            v[:, 0] = c * vx - s_ * vy
            v[:, 1] = s_ * vx + c * vy
            rows["yaw"] = (rows["yaw"].astype(np.float64) + a).astype(np.float32)
        if self.speed is not None:
            v *= self._prng.uniform(self.speed[0], self.speed[1], n)[:, None]
        rows["velocity"] = v.astype(np.float32)
        return rows

    def line(self) -> str:
        d, b = self.info["deaths_s"], self.info.get("stops_s", [])
        la = self.info.get("launches_s")
        return ("greedy-line pool: {:,} own states from {}/{} failed greedy episodes "
                "(deaths at {} s, {:g}..{:g} s before each; best-progress stops at {} s), "
                "{:.0%} of the non-start pool"
                .format(self.size, self.info["fail"], self.info["eps"],
                        ", ".join(f"{v:.1f}" for v in d) if d else "-", self.hi, self.lo,
                        ", ".join(f"{v:.1f}" for v in b) if b else "-", self.frac)
                + ("" if la is None else "; fatal flights anchored at their last contact ({} s)"
                   .format(", ".join(f"{v:.1f}" for v in la) if la else "none"))
                + ("" if not self.energy else "; energy losses anchored ({})".format(
                    ", ".join(f"{du:,} u at {s:.1f} s" for s, du in self.info.get(
                        "energy_losses", [])) or "none")))

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> dict:
        return {"rows": self.rows.copy(), "info": dict(self.info)}

    def load_state_dict(self, st) -> bool:
        rows = st.get("rows") if isinstance(st, dict) else None
        if rows is None or getattr(rows, "dtype", None) != STATE_DTYPE:
            return False
        self.rows = np.array(rows, dtype=STATE_DTYPE, copy=True)
        self.info = dict(st.get("info") or self.info)
        return True
