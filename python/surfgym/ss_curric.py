"""--spawn-states-curric: a reverse curriculum over a --spawn-states file (generic).

Florensa et al., "Reverse Curriculum Generation for Reinforcement Learning" (CoRL 2017,
arXiv 1707.05300): train from the starts whose success rate is intermediate - the ones just
behind the frontier of what the policy can do - and let that set move as the policy improves.
Here the starts are the rows of a --spawn-states file (the policy's own states, or a search's
own states rooted at the true start), and an episode started from row i SUCCEEDS when its
progress record (the ratchet's best field distance) gets ``gain`` map units past the row's
own distance. ``gain`` is a fraction of the true start's distance d0, so the same constant
means the same thing on every map; nothing here knows where a map is hard.

Each row keeps decayed counts (episodes, successes); its success estimate is
p = (successes + 0.5) / (episodes + 1) (a row never tried reads 0.5, the band's centre), and
rows are drawn with weight max(p (1 - p), floor): intermediate rows dominate, mastered rows
(p -> 1) and hopeless ones (p -> 0) are still drawn at the floor so the estimate can move.

The C reset draws uniformly from the pool and never reports an index, so realized spawns are
re-identified by matching their origins against the rows (the trainer copies rows exactly).
"""
from __future__ import annotations

import numpy as np


class SpawnStateCurriculum:
    """mode "band" (the default): Florensa's band weights above. mode "backward": the
    Salimans & Chen backward algorithm (arXiv 1812.03381) - Go-Explore's own phase 2 - over the
    rows ordered by progress, deepest first: a window of D rows starts at the deep end and
    slides one step S toward the start whenever the window's success rate (the same per-row
    success) reaches ``rate`` over at least ``min_ep`` episodes, and slides back when it falls
    below rate / 2; ``front`` of the draws come from the window, the rest uniformly from every
    row it has already passed (so the downstream part is not forgotten). The window parks where
    the route is hardest for the current policy and concentrates the practice there, which the
    band weights cannot: a row that always fails sits at their floor."""

    def __init__(self, states, field, d0: float, gain_frac: float,
                 decay: float = 0.99, floor: float = 0.02, mode: str = "band",
                 rate: float = 0.2, min_ep: float = 30.0, front: float = 2.0 / 3.0) -> None:
        if not gain_frac > 0.0:
            raise ValueError("gain_frac must be > 0")
        if mode not in ("band", "backward"):
            raise ValueError(f"mode must be band or backward, got {mode!r}")
        self.field = field
        self.d0 = float(d0)
        self.gain = float(gain_frac) * self.d0
        self.decay = float(decay)
        self.floor = float(floor)
        self.mode = str(mode)
        self.rate = float(rate)
        self.min_ep = float(min_ep)
        self.front = float(front)
        self.n = 0
        self.ep = np.zeros(0, np.float64)
        self.win = np.zeros(0, np.float64)
        self.noted = 0
        self.tau = 0
        self.advances = 0
        self.set_states(states)

    @staticmethod
    def _keys(origins) -> list:
        return [tuple(np.round(np.asarray(o, np.float64), 1)) for o in origins]

    def set_states(self, states) -> None:
        """(Re)index the rows; a row whose origin was already known keeps its counts."""
        old = ({k: (self.ep[i], self.win[i]) for k, i in self._key.items()}
               if self.n else {})
        self.S = np.asarray(states)
        self.n = len(self.S)
        org = self.S["origin"].astype(np.float64)
        self.spawn_d = self.field.sample(org).astype(np.float64)
        self._key = {}
        for i, k in enumerate(self._keys(org)):
            self._key.setdefault(k, i)          # duplicate rows share the first index
        self.ep = np.zeros(self.n, np.float64)
        self.win = np.zeros(self.n, np.float64)
        for k, i in self._key.items():
            if k in old:
                self.ep[i], self.win[i] = old[k]
        # --backward: rows by progress, deepest first (canonical index of each), the window size
        # D = max(16, n / 32) and its step S = max(4, D / 4); a re-read keeps tau in range
        canon = np.array([self._key[k] for k in self._keys(org)], np.int64)
        prog = self.d0 - self.spawn_d
        uniq = np.unique(canon)
        self.order = uniq[np.argsort(-prog[uniq], kind="stable")]
        self.D = max(16, len(self.order) // 32)
        self.step_n = max(4, self.D // 4)
        self.tau = min(getattr(self, "tau", 0), max(0, len(self.order) - self.D))

    def p(self) -> np.ndarray:
        return (self.win + 0.5) / (self.ep + 1.0)

    def weights(self) -> np.ndarray:
        p = self.p()
        w = np.maximum(p * (1.0 - p), self.floor)
        # duplicate rows of one origin share its weight (counts live on the first index)
        canon = np.array([self._key[k] for k in self._keys(self.S["origin"])], np.int64)
        return w[canon]

    def _window(self) -> np.ndarray:
        return self.order[self.tau:self.tau + self.D]

    def step(self) -> None:
        """--backward: slide the window by its own success (call once per pool build)."""
        if self.mode != "backward":
            return
        w = self._window()
        ep, win = float(self.ep[w].sum()), float(self.win[w].sum())
        if ep < self.min_ep:
            return
        r = win / ep
        last = max(0, len(self.order) - self.D)
        if r >= self.rate and self.tau < last:
            self.tau = min(last, self.tau + self.step_n)
            self.advances += 1
        elif r < 0.5 * self.rate and self.tau > 0:
            self.tau = max(0, self.tau - self.step_n)
            self.advances -= 1

    def draw(self, rng, n: int) -> np.ndarray:
        if self.mode == "backward":
            self.step()
            n = int(n)
            k = int(round(n * self.front))
            passed = self.order[:self.tau + self.D]
            idx = np.concatenate([rng.choice(self._window(), k),
                                  rng.choice(passed, n - k)])
            return self.S[idx].copy()
        w = self.weights()
        idx = rng.choice(self.n, int(n), p=w / w.sum())
        return self.S[idx].copy()

    def match(self, origins) -> np.ndarray:
        """Row index of each spawn origin, -1 when it is not a row."""
        return np.array([self._key.get(k, -1) for k in self._keys(origins)], np.int64)

    def note(self, idxs, best_d) -> None:
        """Ended episodes started from rows ``idxs`` (>= 0) with progress record ``best_d``."""
        idxs = np.asarray(idxs, np.int64)
        best_d = np.asarray(best_d, np.float64)
        for i, b in zip(idxs, best_d):
            if i < 0:
                continue
            ok = float(self.spawn_d[i] - b >= self.gain)
            self.ep[i] = self.ep[i] * self.decay + 1.0
            self.win[i] = self.win[i] * self.decay + ok
            self.noted += 1

    def line(self) -> str:
        tried = self.ep > 0.5
        p = self.p()
        band = tried & (p > 0.1) & (p < 0.9)
        w = self.weights()
        prog = self.d0 - self.spawn_d
        top = np.argsort(-w)[: max(1, self.n // 10)]
        if self.mode == "backward":
            win = self._window()
            ep_w = float(self.ep[win].sum())
            r = float(self.win[win].sum()) / ep_w if ep_w > 0 else float("nan")
            return (f"ss-curric backward: {len(self.order)} rows, window {self.tau}..{self.tau + self.D - 1} "
                    f"(D {self.D}, S {self.step_n}) at progress {prog[win].min():,.0f}..{prog[win].max():,.0f}u, "
                    f"window success {r:.2f} over {ep_w:.0f} ep, net advances {self.advances}; "
                    f"gain {self.gain:,.0f}u; {self.noted:,} episodes noted")
        return (f"ss-curric: {self.n} rows, {int(tried.sum())} tried, {int(band.sum())} in "
                f"(0.1, 0.9), {int((tried & (p >= 0.9)).sum())} mastered, "
                f"{int((tried & (p <= 0.1)).sum())} failing; gain {self.gain:,.0f}u; "
                f"top-decile rows progress med {np.median(prog[top]):,.0f}u; "
                f"{self.noted:,} episodes noted")
