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
    def __init__(self, states, field, d0: float, gain_frac: float,
                 decay: float = 0.99, floor: float = 0.02) -> None:
        if not gain_frac > 0.0:
            raise ValueError("gain_frac must be > 0")
        self.field = field
        self.d0 = float(d0)
        self.gain = float(gain_frac) * self.d0
        self.decay = float(decay)
        self.floor = float(floor)
        self.n = 0
        self.ep = np.zeros(0, np.float64)
        self.win = np.zeros(0, np.float64)
        self.noted = 0
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

    def p(self) -> np.ndarray:
        return (self.win + 0.5) / (self.ep + 1.0)

    def weights(self) -> np.ndarray:
        p = self.p()
        w = np.maximum(p * (1.0 - p), self.floor)
        # duplicate rows of one origin share its weight (counts live on the first index)
        canon = np.array([self._key[k] for k in self._keys(self.S["origin"])], np.int64)
        return w[canon]

    def draw(self, rng, n: int) -> np.ndarray:
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
        return (f"ss-curric: {self.n} rows, {int(tried.sum())} tried, {int(band.sum())} in "
                f"(0.1, 0.9), {int((tried & (p >= 0.9)).sum())} mastered, "
                f"{int((tried & (p <= 0.1)).sum())} failing; gain {self.gain:,.0f}u; "
                f"top-decile rows progress med {np.median(prog[top]):,.0f}u; "
                f"{self.noted:,} episodes noted")
