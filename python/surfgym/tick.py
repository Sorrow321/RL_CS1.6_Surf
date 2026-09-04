"""tick.py - the physics tick length as a first-class quantity (``--tick-ms``).

GoldSrc's air-accelerate impulse saturates at 30 u/s PER FRAME
(``src/pm.c`` PM_AirAccelerate: wishspd capped at 30 while
``accelspeed = airaccelerate * wishspeed * frametime`` = 320 at our
settings), so the acceleration available to a strafer is proportional to
the FRAME RATE. The cannonball world-record demo runs its usercmd frames at
7 / 8 ms (mean 7.63 ms = 131 fps): 31% more air-accelerate steps per second
than our fixed 10 ms (100 Hz) tick. This module is everything the rest of
the code needs to run the core at another tick and stay comparable in
SECONDS:

* :func:`tick_pattern` - the integer millisecond sequence that realises a
  non-integer tick (the C core steps in whole milliseconds; ``SurfPhys.msec``
  is an int32). 7.63 -> ``[8, 8, 7]`` = 7.667 ms, the SHORTEST repeating
  pattern whose mean is within 0.05 ms of the request.
* :class:`TickClock` - the conversions. Every per-tick constant of the
  trainer (gamma, --time-pen, --stall-secs, the respawn margin, snap_every,
  --goal-kmin/kmax, --finish-tref, the pitch/yaw rates) is DEFINED at the
  10 ms reference tick and converted here, so an arm at 131 Hz means the
  same thing per second as the 100 Hz control. At ``tick_ms == 10.0`` every
  conversion returns the legacy expression bit for bit (``secs * 100.0``,
  ``gamma``, ``value``), which is what keeps the default path byte-identical.
* :func:`episode_seconds` / :func:`header_tick_ms` - the recorder's time
  base. A trajectory header carries ``tick_ms`` (mean, ms) and, for a
  multi-element pattern, ``tick_pattern_ms`` + ``tick_phase`` (the pattern
  index at the episode's first row) so an episode's duration is exact. A
  header WITHOUT ``tick_ms`` raises: a tool that cannot know the tick must
  refuse rather than mis-time (CLAUDE.md, the honest-metric rule).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

REFERENCE_TICK_MS = 10.0     # every per-tick constant is defined at this tick
MIN_TICK_MS = 1.0
MAX_TICK_MS = 50.0           # engine chops >50 ms usercmds; env.c clamps too
PATTERN_TOL_MS = 0.05        # "within 0.05 ms of the request"
MAX_PATTERN_LEN = 64


def tick_pattern(tick_ms: float, tol_ms: float = PATTERN_TOL_MS) -> List[int]:
    """Shortest repeating integer-ms sequence whose mean is within ``tol_ms``
    of ``tick_ms``.

    An integer request maps to itself (``[10]``). Otherwise, with
    ``f = floor(tick_ms)`` and ``c = f + 1``, a pattern of length ``n`` with
    ``k`` long ticks has mean ``f + k / n``; the smallest ``n`` with some
    ``k`` inside the tolerance wins, and the ``k`` long ticks are spread as
    evenly as possible over the ``n`` slots (Bresenham), long ticks first,
    so 7.63 -> ``[8, 8, 7]`` (mean 7.667 ms, 130.4 Hz).
    """
    t = float(tick_ms)
    if not math.isfinite(t) or t < MIN_TICK_MS or t > MAX_TICK_MS:
        raise ValueError(f"tick_ms must be in [{MIN_TICK_MS:g}, "
                         f"{MAX_TICK_MS:g}] ms, got {tick_ms!r}")
    if abs(t - round(t)) < 1e-9:
        return [int(round(t))]
    f = int(math.floor(t))
    c = f + 1
    for n in range(1, MAX_PATTERN_LEN + 1):
        # k long ticks out of n: pick the k whose mean is closest
        k = int(round((t - f) * n))
        k = min(max(k, 0), n)
        if abs(f + k / n - t) <= tol_ms + 1e-12:
            # Bresenham spread of the k long ticks over n slots, long ticks
            # first: slot i is long when ceil((i+1)k/n) - ceil(ik/n) == 1,
            # so 7.63 -> [8, 8, 7] and 7.6 -> [8, 8, 7, 8, 7]
            pat = [c if (math.ceil((i + 1) * k / n) - math.ceil(i * k / n)) == 1
                   else f for i in range(n)]
            assert pat.count(c) == k and len(pat) == n
            return pat
    raise ValueError(f"no integer-ms pattern of length <= {MAX_PATTERN_LEN} "
                     f"lands within {tol_ms:g} ms of {tick_ms:g} ms")


def pattern_mean_ms(pattern: Sequence[int]) -> float:
    if not pattern:
        raise ValueError("empty tick pattern")
    return float(sum(int(v) for v in pattern)) / len(pattern)


class TickClock:
    """The realised tick and every seconds<->ticks conversion the trainer
    makes. ``requested_ms`` is what the flag said (7.63); ``ms`` is what the
    pattern delivers (7.667) and is the value every conversion uses, so the
    seconds are exact for the physics actually run.
    """

    def __init__(self, tick_ms: float = REFERENCE_TICK_MS) -> None:
        self.requested_ms = float(tick_ms)
        self.pattern = tick_pattern(self.requested_ms)
        self.ms = pattern_mean_ms(self.pattern)
        self.is_reference = (len(self.pattern) == 1
                             and self.pattern[0] == int(REFERENCE_TICK_MS))

    # -- identity ----------------------------------------------------------
    @property
    def hz(self) -> float:
        return 1000.0 / self.ms

    @property
    def scale(self) -> float:
        """ticks-per-10ms-tick ratio: how much of a reference tick one real
        tick is (0.7667 at 7.667 ms). Exactly 1.0 at the reference."""
        return 1.0 if self.is_reference else self.ms / REFERENCE_TICK_MS

    def describe(self) -> str:
        if self.is_reference:
            return "tick 10 ms (100 Hz, the reference; every constant as before)"
        pat = ",".join(str(v) for v in self.pattern)
        return (f"tick {self.requested_ms:g} ms requested -> pattern "
                f"[{pat}] ms = {self.ms:.4f} ms ({self.hz:.1f} Hz), "
                f"{self.hz / 100.0:.3f}x the physics steps per second of "
                f"the 10 ms reference")

    # -- conversions -------------------------------------------------------
    def secs_to_ticks(self, secs: float, rounding: str = "trunc") -> int:
        """Seconds -> ticks. At the reference tick this is exactly the legacy
        ``secs * 100.0`` (then ``int()`` or ``round()`` as the call site
        always did), so the 10 ms path stays bit-identical."""
        if self.is_reference:
            raw = float(secs) * 100.0
        else:
            raw = float(secs) * 1000.0 / self.ms
        if rounding == "trunc":
            return int(raw)
        if rounding == "round":
            return int(round(raw))
        raise ValueError(f"rounding must be 'trunc' or 'round', got {rounding!r}")

    def ticks_to_secs(self, ticks: float) -> float:
        if self.is_reference:
            return float(ticks) / 100.0
        return float(ticks) * self.ms / 1000.0

    def gamma(self, gamma_ref: float) -> float:
        """Per-tick discount with the SAME horizon in seconds:
        ``gamma_ref ** (tick_ms / 10)``. 0.9995 at 10 ms (2,000 ticks =
        20.0 s) becomes 0.999617 at 7.667 ms (2,609 ticks = 20.0 s)."""
        if self.is_reference:
            return float(gamma_ref)
        return float(gamma_ref) ** self.scale

    def per_tick(self, value_ref: float) -> float:
        """A per-tick amount (time penalty, speed bonus, a deg-per-tick view
        rate) defined at 10 ms, rescaled so the per-SECOND amount is
        unchanged: ``value * tick_ms / 10``."""
        if self.is_reference:
            return float(value_ref)
        return float(value_ref) * self.scale

    def to_dict(self) -> Dict[str, Any]:
        return {"tick_ms": self.requested_ms, "tick_ms_eff": self.ms,
                "tick_pattern_ms": list(self.pattern)}


# ---------------------------------------------------------------------------
# trajectory headers: the recorder's time base
# ---------------------------------------------------------------------------

def header_tick_ms(header: Optional[Dict[str, Any]], what: str = "trajectory") -> float:
    """The mean tick, ms, from a record_rollout episode header. Raises when
    the header is missing or has no ``tick_ms``: a tool that cannot know the
    tick must refuse loudly rather than assume 10 ms."""
    if not header or header.get("tick_ms") is None:
        raise ValueError(
            f"{what}: episode header has no 'tick_ms' - cannot time this "
            f"recording (record_rollout writes it; pass the tick explicitly "
            f"if the tool offers a flag, never assume 10 ms)")
    t = float(header["tick_ms"])
    if not (MIN_TICK_MS <= t <= MAX_TICK_MS):
        raise ValueError(f"{what}: header tick_ms {t!r} is out of range")
    return t


def episode_seconds(header: Optional[Dict[str, Any]], n_rows: int,
                    what: str = "trajectory") -> float:
    """Duration of ``n_rows`` recorded ticks. Exact when the header carries
    ``tick_pattern_ms`` + ``tick_phase`` (a repeating pattern, phase = index
    of the first row's tick); otherwise ``n_rows * tick_ms``."""
    n = int(n_rows)
    pat = header.get("tick_pattern_ms") if header else None
    if pat:
        pat = [int(v) for v in pat]
        L = len(pat)
        phase = int((header or {}).get("tick_phase", 0)) % L
        full, rem = divmod(n, L)
        total = full * sum(pat)
        for i in range(rem):
            total += pat[(phase + i) % L]
        return total / 1000.0
    return n * header_tick_ms(header, what) / 1000.0


def step_seconds(header: Optional[Dict[str, Any]], n_rows: int,
                 what: str = "trajectory") -> List[float]:
    """Per-row tick durations (s), length ``n_rows`` - the exact ``dt`` of
    every recorded step under a repeating pattern, uniform otherwise."""
    n = int(n_rows)
    pat = header.get("tick_pattern_ms") if header else None
    if pat:
        pat = [int(v) for v in pat]
        L = len(pat)
        phase = int((header or {}).get("tick_phase", 0)) % L
        return [pat[(phase + i) % L] / 1000.0 for i in range(n)]
    dt = header_tick_ms(header, what) / 1000.0
    return [dt] * n


def ticks_to_secs(ticks, tick_ms: float = REFERENCE_TICK_MS,
                  pattern: Optional[Sequence[int]] = None,
                  phase: int = 0) -> float:
    """Ticks counted from an episode start -> seconds. The legacy
    ``ticks / 100.0`` bit for bit at the reference tick; EXACT under a
    repeating pattern (each tick's own ms summed from ``phase``: 9,645
    ticks of [8, 8, 7] = 73.945 s, where 9,645 x 7.667 would be off by
    up to one tick); ``ticks * tick_ms / 1000`` for any other plain tick.
    The planner (tools/beam_tas.py) and every reader of its output
    (plan_to_bc, expert_loop) time finish ticks with this."""
    t = float(tick_ms)
    pat = [int(v) for v in pattern] if pattern is not None else []
    if len(pat) <= 1:
        if t == REFERENCE_TICK_MS:
            return float(ticks) / 100.0
        return float(ticks) * t / 1000.0
    return episode_seconds({"tick_ms": t, "tick_pattern_ms": pat,
                            "tick_phase": int(phase)}, int(ticks))


def header_fields(tick_ms_eff: float, pattern: Sequence[int],
                  phase: int) -> Dict[str, Any]:
    """The tick keys the recorder writes into every episode header. An
    integer tick writes the plain int it always did (``"tick_ms": 10``) so
    the 10 ms header is byte-identical; a pattern adds its members."""
    pat = [int(v) for v in pattern]
    if len(pat) == 1:
        return {"tick_ms": int(pat[0])}
    return {"tick_ms": round(float(tick_ms_eff), 6),
            "tick_pattern_ms": pat, "tick_phase": int(phase) % len(pat)}
