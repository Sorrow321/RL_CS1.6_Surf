"""Greedy-tail spawn states: the policy's OWN greedy line, just before it fails.

The trainer's periodic greedy eval from the true start is the line the score is read from; where
it fails is the frontier of the deployed policy. ``TailCapture`` is a ``record_rollout`` on_tick
hook that keeps env 0's full core states per episode; ``cut_tails`` keeps, for every FAILED
episode, the states of the last ``secs`` seconds before its end (to ``skip`` seconds before it,
every ``every`` ticks, slower than ``min_speed`` dropped). ``train_fast.py --greedy-tail-frac``
draws a share of every training spawn pool from them, refreshed at every eval; the same cut runs
offline in ``tools/greedy_tail_states.py`` on a ``record_ckpt.py --dump-states`` recording.

Generic by construction: no map quantity, only a tail length, a share and a speed floor. And
champion-free (CLAUDE.md section 0): the states are the policy's own greedy recording.
"""
from __future__ import annotations

import numpy as np

#: a reward this large on the last tick is the finish bonus (record.py's own rule)
DONE_BONUS_MIN = 25.0


class TailCapture:
    """``on_tick(t, states, rewards, done, trunc)`` hook for ``record_rollout``: env 0's
    pre-step state every tick, cut into episodes. ``episodes`` holds ``(states, end)`` with
    end in {"fail", "done", "trunc"}; an episode still open when the recording stops is
    dropped (it has no end to be a tail of)."""

    def __init__(self):
        self.episodes: list[tuple[np.ndarray, str]] = []
        self._cur: list = []

    def __call__(self, t, states, rewards, done, trunc):
        self._cur.append(states[0].copy())
        if bool(done[0]) or bool(trunc[0]):
            if bool(trunc[0]):
                end = "trunc"
            elif float(rewards[0]) >= DONE_BONUS_MIN:
                end = "done"
            else:
                end = "fail"
            self.episodes.append((np.array(self._cur, dtype=states.dtype), end))
            self._cur = []


def cut_tails(episodes, hz: float, secs: float = 3.0, skip: float = 0.25, every: int = 5,
              max_secs: float = 120.0, min_speed: float = 0.0):
    """``episodes``: ``[(states (n,) STATE_DTYPE, end)]``. Returns ``(rows, notes)``: the
    concatenated tail states of every FAILED episode no longer than ``max_secs`` (a loiter's
    end is not a failure point) and the per-episode notes. Empty rows when none qualifies."""
    rows, notes = [], []
    for i, (st, end) in enumerate(episodes):
        n = len(st)
        if end != "fail":
            notes.append(f"ep {i}: end {end} - skipped")
            continue
        if n / hz > max_secs:
            notes.append(f"ep {i}: {n / hz:.1f} s > {max_secs:g} s - skipped")
            continue
        hi = max(0, n - int(round(skip * hz)))
        lo = max(0, n - int(round(secs * hz)))
        if hi <= lo:
            notes.append(f"ep {i}: too short")
            continue
        keep = st[lo:hi:max(1, int(every))]
        if min_speed > 0.0:
            v = keep["velocity"].astype(np.float64)
            keep = keep[np.sqrt((v * v).sum(1)) >= min_speed]
        rows.append(keep)
        notes.append(f"ep {i}: {n / hz:.2f} s, kept ticks {lo}..{hi} every {every} ({len(keep)} rows)")
    if not rows:
        return np.empty(0), notes
    return np.concatenate(rows), notes
