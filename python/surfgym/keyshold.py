"""``--keys-hold``: the movement keys become HELD STATE, not per-decision presses.

Motivation (round 32, branch ``keyshold``). The continuous *absolute* view
(``--view-continuous --view-absolute velocity``) works because the policy
commands **where the view IS**, not how much to turn it this tick; the agent
no longer has to re-issue the same delta 25 times a second to hold a heading.
The A/D/W/S/Ctrl keys still work the old way: every decision re-presses (or
re-releases) each key from scratch, and the measured consequence is chatter -
A/D flips per second are 0.42 on the human world record, 7.01 on the exitABS
r9 policy and 18.91 at ``--act-every 1``. A key that has to be re-decided
every 40 ms is a key that gets dropped by accident.

This module applies the absolute-view logic to the keys. For the **fwd**,
**side** and **duck** heads only (jump is left exactly as it is - a jump is a
genuine impulse, not a held state), the head gains a **"keep" bin at index 0**:

    bin 0        keep whatever this env is currently holding
    bins 1..n    the existing absolute values, in their existing order

so a 3-bin head becomes 4 bins and the 2-bin duck head becomes 3. The held
value is carried per env across decisions and resolved HERE, in Python: the
core receives exactly the absolute action row it receives today, so there is
no ABI change and no C edit. The policy also OBSERVES what it is holding
(``features``: fwd one-hot 3 + side one-hot 3 + duck 1 = 7 columns), because
the reward now depends on episode history through the held state and a policy
that could not see that state would be acting in a partially observed MDP -
the same argument ``--race-latch``'s flag column makes.

Episode starts (spawn, autoreset, reservoir respawn, stall kill, demo start)
reset the held state to NEUTRAL - fwd none, side none, duck off - because the
engine resets the buttons there too.

``boot`` is the ``latch_boot`` pattern: the held state as of the LAST resolve,
kept so the truncation bootstrap can build the terminal row's key columns
after the autoreset has already moved the live state on to the next episode's
spawn.
"""

from __future__ import annotations

import numpy as np

__all__ = ["KeysHold", "N_FEATURES", "NEUTRAL", "HELD_HEADS", "KEEP",
           "nvec_with_keep"]

#: the "keep" bin, index 0 of every widened head
KEEP = 0

#: head indices into NVEC that gain a keep bin: fwd, side, duck (NOT jump)
HELD_HEADS = (2, 3, 5)

#: neutral held value per held head, in ENGINE bins: a[2]/a[3] are
#: ``{-400, 0, +400}[i]`` (surfcore.h), so 1 is "no key"; duck 0 is "not
#: ducking".  This is what the engine has at every spawn.
NEUTRAL = (1, 1, 0)

#: one-hot(fwd, 3) + one-hot(side, 3) + duck(1)
N_FEATURES = 7


def nvec_with_keep(nvec):
    """``(15, 7, 3, 3, 2, 2)`` -> ``(15, 7, 4, 4, 2, 3)``.

    One extra bin on each held head; yaw, pitch and jump are untouched, and
    ``max(nvec)`` is unchanged at 15 so the padded (NACT, NPAD) logit table
    keeps its shape.
    """
    out = list(int(v) for v in nvec)
    for h in HELD_HEADS:
        out[h] += 1
    return tuple(out)


class KeysHold:
    """Per-env held state for the three held keys.

    ``state`` and ``boot`` are ``(n, 3)`` int32 in ENGINE bins, columns in
    ``HELD_HEADS`` order (fwd, side, duck).
    """

    n_features = N_FEATURES

    def __init__(self, n: int):
        self.n = int(n)
        self.state = np.empty((self.n, 3), np.int32)
        self.boot = np.empty((self.n, 3), np.int32)
        self._feat = np.zeros((self.n, N_FEATURES), np.float32)
        self.reset(None)
        self.boot[:] = self.state

    # ---------------------------------------------------------------- reset
    def reset(self, mask=None) -> None:
        """Collapse the held state to NEUTRAL.

        ``mask`` None = every env (construction / a fresh eval); otherwise a
        boolean (n,) mask of the envs whose episode just started.  This is
        the ``obs_aux.reset(ended_acc)`` contract exactly: called once per
        decision, after the truncation bootstrap has read ``boot`` and before
        the observation the fresh episode's first decision reads is built.
        """
        nz = np.asarray(NEUTRAL, np.int32)
        if mask is None:
            self.state[:] = nz
            return
        m = np.asarray(mask, bool)
        if m.any():
            self.state[m] = nz

    # -------------------------------------------------------------- resolve
    def resolve(self, act) -> np.ndarray:
        """POLICY-space action row -> ENGINE-space, IN PLACE.

        ``act`` is ``(n, NACT)`` integer.  For each held head, bin 0 means
        "keep" and is replaced by this env's held value; bins ``1..n`` are the
        engine's own bins shifted up by one, so they are written back as
        ``bin - 1`` and become the new held value.  Every other head (yaw,
        pitch, jump) is passed through untouched.

        Returns ``act`` so callers can chain.  ``boot`` is snapshotted here:
        after this call it is the state the NEXT observation will show, which
        is what ``V(s_T)`` needs at a truncation.
        """
        a = act
        if a.shape[0] != self.n:
            raise ValueError(f"KeysHold sized for {self.n} envs, got "
                             f"{a.shape[0]}")
        for c, h in enumerate(HELD_HEADS):
            col = a[:, h]
            keep = col == KEEP
            new = np.where(keep, self.state[:, c], col - 1)
            self.state[:, c] = new
            col[:] = new
        self.boot[:] = self.state
        return a

    # ------------------------------------------------------------- features
    def features(self, out=None) -> np.ndarray:
        """``(n, 7)`` float32: one-hot(fwd) | one-hot(side) | duck.

        The state as of the decision the policy is being asked to make - i.e.
        the state its action will modify.
        """
        return self._encode(self.state, out)

    def boot_features(self, idx=None) -> np.ndarray:
        """``features`` of the ONE-CALL-OLD copy, optionally for rows ``idx``.

        The truncation bootstrap's terminal row: ``state`` has already been
        collapsed by the fresh spawn, ``boot`` still holds what the terminal
        state was holding.
        """
        src = self.boot if idx is None else self.boot[idx]
        return self._encode(src, None)

    @staticmethod
    def _encode(src, out):
        n = src.shape[0]
        if out is None:
            out = np.zeros((n, N_FEATURES), np.float32)
        else:
            out[:] = 0.0
        r = np.arange(n)
        out[r, np.clip(src[:, 0], 0, 2)] = 1.0            # fwd one-hot
        out[r, 3 + np.clip(src[:, 1], 0, 2)] = 1.0        # side one-hot
        out[:, 6] = src[:, 2].astype(np.float32)          # duck 0/1
        return out

    # ------------------------------------------------------------ reporting
    def describe(self) -> str:
        return (f"--keys-hold: fwd/side/duck carry a KEEP bin at index 0 "
                f"(heads {HELD_HEADS} widened by one); the held state is "
                f"{N_FEATURES} observation columns and resets to "
                f"{NEUTRAL} at every episode start")
