"""goalprim.py - MOTION PRIMITIVES as the executor's plans (``--goal-planner prim``, step 1).

A primitive is a short curve that leaves the agent ALONG its velocity (along the view direction
below a speed floor). Its whole shape is two turn-rate functions of time - how fast the heading
turns sideways (+ = left) and up (+ = up) - each given at K knots spread evenly over the primitive
and joined by the polynomial through them (K = 3: quadratic, 6 numbers). The world-record analysis
(tools/primitive_coverage.py, ledger 2026-09-25) found 6 numbers redraw 85-88% of 2-second record
route pieces within 128 u and all of our edgeflow finishers' within 64 u. The curve is traced at
max(speed, floor) for ``secs`` seconds, so it scales with speed like the lookahead points.
``--prim-frame level`` lays it on the velocity's PROJECTION onto the horizontal plane instead: it
starts level along the horizontal heading and is traced at the horizontal speed, so the same
numbers draw the same shape whether the agent is climbing or falling.

Step 1 learns nothing here: every number is drawn uniformly from fixed, generic ranges (the same on
every map), and only the EXECUTOR learns to follow them - one primitive per episode, started from
the map spawn or from states it reached before (the reservoir), rewarded for progress along the
curve plus the bonus of reaching its end. The per-bin success tables say which parts of the space
it masters; step 2 puts a learned distribution (a mixture) over the same numbers.
"""
from __future__ import annotations

import numpy as np

from .goalplan import Plan

PRIM_DEFAULTS = {"prim_secs": 2.0, "prim_knots": 3, "prim_side": 180.0, "prim_down": 120.0,
                 "prim_up": 90.0, "prim_floor": 300.0}
PRIM_SEED_OFFSET = 7331
DT = 0.01
SPEED_DIR_MIN = 50.0          # u/s: below this the curve starts along the VIEW direction
PRIM_FRAMES = ("velocity", "level")               # --prim-frame
SIDE_BINS = (0.0, 45.0, 90.0, 135.0, 1e9)         # |mean sideways rate|, deg/s
VERT_BINS = (-1e9, -30.0, 30.0, 1e9)              # mean vertical rate, deg/s


def rate_profile(knots, secs, t):
    """Knot values (K,) at times 0 .. secs, evenly spaced -> the polynomial through them at t."""
    k = np.asarray(knots, np.float64)
    if len(k) == 1:
        return np.full_like(t, k[0])
    tk = np.linspace(0.0, secs, len(k))
    return np.polyval(np.polyfit(tk, k, len(k) - 1), t)


def curve(origin, velocity, yaw_deg, params, secs, knots, floor, flat=False,
          frame="velocity"):
    """-> (P, 3) float64 points of the primitive from ``origin``. ``params`` = K sideways knots
    then K vertical knots (deg/s)."""
    o = np.asarray(origin, np.float64).reshape(3)
    v = np.asarray(velocity, np.float64).reshape(3)
    vh = float(np.hypot(v[0], v[1]))
    sp = float(np.linalg.norm(v))
    if frame == "level":
        # --prim-frame level (the user, 2026-09-26): the curve is laid on the velocity's
        # projection onto the horizontal plane - it leaves LEVEL along the horizontal heading (the
        # view yaw below SPEED_DIR_MIN of horizontal speed, as goalprimplan.motion_frame) and is
        # traced at the horizontal speed. Whether the agent is climbing or falling no longer
        # changes what the numbers draw; the vertical rates bend it up or down from level
        yaw0 = (np.arctan2(v[1], v[0]) if vh >= SPEED_DIR_MIN
                else np.radians(float(yaw_deg)))
        pitch0 = 0.0
        sp = vh
    elif sp >= SPEED_DIR_MIN:
        yaw0 = np.arctan2(v[1], v[0])
        pitch0 = np.arctan2(v[2], max(vh, 1e-6))
    else:
        yaw0 = np.radians(float(yaw_deg))
        pitch0 = 0.0
    if flat:
        # --prim-flat: a HORIZONTAL curve - no initial pitch, no vertical turn rate; it bends
        # sideways only, at the height it starts from
        pitch0 = 0.0
    speed = max(sp, float(floor))
    t = np.arange(0.0, secs + 1e-9, DT)
    p = np.asarray(params, np.float64)
    wh = np.radians(rate_profile(p[:knots], secs, t))
    wv = (np.zeros_like(t) if flat
          else np.radians(rate_profile(p[knots:2 * knots], secs, t)))
    yaw = yaw0 + np.concatenate(([0.0], np.cumsum(0.5 * (wh[1:] + wh[:-1]) * DT)))
    pitch = np.clip(pitch0 + np.concatenate(([0.0], np.cumsum(0.5 * (wv[1:] + wv[:-1]) * DT))),
                    np.radians(-85.0), np.radians(85.0))
    d = np.stack([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], 1)
    step = 0.5 * (d[1:] + d[:-1]) * speed * DT
    return o + np.vstack([np.zeros(3), np.cumsum(step, 0)])


class PrimitivePlanner:
    """Step 1's planner: a uniform draw over the primitive numbers, and the curve they give.
    The goal system calls :meth:`goal` once per spawn (goalsys.GoalSystem._planned_goal)."""

    primitive = True
    fin = None
    finish_center = None
    n_rand = 0

    def __init__(self, secs=2.0, knots=3, side=180.0, down=120.0, up=90.0, floor=300.0,
                 spacing=128.0, n_envs=1, radius=192.0, flat=False, frame="velocity"):
        self.secs, self.knots = float(secs), int(knots)
        # a primitive is DRAWN AGAIN when its end sphere could be entered before the curve is
        # mostly done - a tight turn at the floor speed loops back onto its own start (180 deg/s
        # at 300 u/s is a 95 u radius) and would pay the completion bonus for standing still
        self.radius = float(radius)
        self.redraws = 0
        self.side, self.down, self.up = float(side), float(down), float(up)
        self.floor, self.spacing = float(floor), float(spacing)
        self.flat = bool(flat)          # --prim-flat: horizontal curves only
        self.frame = str(frame)         # --prim-frame: velocity (the default) or level
        if self.frame not in PRIM_FRAMES:
            raise ValueError(f"primitive frame {self.frame!r}: one of {PRIM_FRAMES}")
        if self.knots < 1 or self.secs <= 0.0:
            raise ValueError("primitive: need >= 1 knot and a positive duration")
        self.params = np.zeros((int(n_envs), 2 * self.knots), np.float32)
        self.bin_n = np.zeros((len(SIDE_BINS) - 1, len(VERT_BINS) - 1), np.int64)
        self.bin_ok = np.zeros_like(self.bin_n)

    @property
    def n_numbers(self) -> int:
        return 2 * self.knots

    def sample(self, rng) -> np.ndarray:
        return np.concatenate([rng.uniform(-self.side, self.side, self.knots),
                               rng.uniform(-self.down, self.up, self.knots)])

    def line_and_curve(self, origin, velocity, yaw_deg, params):
        """-> (line resampled at the fan spacing, the raw curve: one point per DT = 10 ms tick,
        i.e. where the primitive wants the agent k ticks after it starts)."""
        from .route import resample_polyline
        pts = curve(origin, velocity, yaw_deg, params, self.secs, self.knots, self.floor,
                    flat=self.flat, frame=self.frame)
        line, _total = resample_polyline(pts, self.spacing)
        if len(line) < 2:
            line = np.vstack([pts[0], pts[-1]]).astype(np.float32)
        return np.asarray(line, np.float32), pts

    def line_of(self, origin, velocity, yaw_deg, params):
        from .route import resample_polyline
        pts = curve(origin, velocity, yaw_deg, params, self.secs, self.knots, self.floor,
                    flat=self.flat, frame=self.frame)
        line, total = resample_polyline(pts, self.spacing)
        if len(line) < 2:
            line = np.vstack([pts[0], pts[-1]]).astype(np.float32)
        return np.asarray(line, np.float32), pts[-1].copy(), float(total)

    def draw(self, origin, velocity, yaw_deg, rng, tries: int = 20):
        """-> (numbers, line, end, length): a uniform draw whose end sphere (``radius``) cannot be
        entered from any point more than two radii of PATH before the end (a loop back onto
        itself), redrawn up to ``tries`` times."""
        for _ in range(int(tries)):
            p = self.sample(rng)
            pts = curve(origin, velocity, yaw_deg, p, self.secs, self.knots, self.floor,
                        flat=self.flat, frame=self.frame)
            s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
            early = pts[s < s[-1] - 2.0 * self.radius]
            if (len(early) == 0 or float(np.min(np.linalg.norm(early - pts[-1], axis=1)))
                    >= self.radius):
                break
            self.redraws += 1
        line, end, total = self.line_of(origin, velocity, yaw_deg, p)
        return p, line, end, total

    def goal(self, i, origin, velocity, yaw_deg, rng) -> Plan:
        """Env ``i`` spawned at (origin, velocity, yaw): draw its primitive -> a Plan whose line is
        the curve (resampled at the fan spacing) and whose goal is the curve's end."""
        p, line, end, total = self.draw(origin, velocity, yaw_deg, rng)
        self.params[int(i)] = p
        return Plan(line=line, goal=end, length=total, target=-1, finish=False,
                    start=-1, raw=line.astype(np.float64))

    # ------------------------------------------------------------------ stats
    def _bin(self, p):
        k = self.knots
        s = abs(float(np.mean(p[:k])))
        v = float(np.mean(p[k:2 * k]))
        return (int(np.searchsorted(SIDE_BINS, s, side="right") - 1),
                int(np.searchsorted(VERT_BINS, v, side="right") - 1))

    def note(self, i, success: bool) -> None:
        a, b = self._bin(self.params[int(i)])
        self.bin_n[a, b] += 1
        self.bin_ok[a, b] += int(bool(success))

    def table(self, reset: bool = True) -> str:
        """Success by |mean sideways rate| (rows) x mean vertical rate (columns)."""
        rows = ["0-45", "45-90", "90-135", "135+"]
        cols = ["down", "level", "up"]
        out = ["primitive success (|mean sideways| deg/s x mean vertical): "
               + "  ".join(f"{c:>12s}" for c in cols)]
        for a, r in enumerate(rows):
            cells = []
            for b in range(len(cols)):
                n = int(self.bin_n[a, b])
                cells.append(f"{(100.0 * self.bin_ok[a, b] / n if n else float('nan')):5.1f}% of {n:>6d}")
            out.append(f"   {r:>7s}: " + "  ".join(cells))
        if reset:
            self.bin_n[:] = 0
            self.bin_ok[:] = 0
        return "\n".join(out)

    def describe(self) -> str:
        return (("[FLAT: horizontal curves only] " if self.flat else "")
                + ("[LEVEL frame: curves leave level along the horizontal velocity, traced at "
                   "the horizontal speed] " if self.frame == "level" else "")
                + f"goals: MOTION PRIMITIVES (--goal-planner prim, step 1) - every spawn draws "
                f"{self.n_numbers} numbers uniformly (sideways turn rate at {self.knots} knots in "
                f"[-{self.side:g}, {self.side:g}] deg/s, vertical in [-{self.down:g}, {self.up:g}]), "
                f"a {self.secs:g} s curve along the velocity at max(speed, {self.floor:g} u/s); "
                f"success = reaching its end")


def make_prim_hooks(planner: PrimitivePlanner, core, ev: dict, *, line=None, ball=None,
                    radius: float = 192.0, rng=None):
    """(episode_meta, on_tick) for ``record_rollout`` on a core whose env 0 is recorded - the
    primitive planner's eval, shared by the trainer and tools/record_ckpt.py. Each episode draws a
    primitive (seeded ``rng``) from env 0's spawn state; entering the sphere of ``radius`` at its
    end force-fails env 0 and counts a success. ``ev`` is the caller's tally dict."""
    rng = rng if rng is not None else np.random.default_rng(0)
    ev.update({"n": 0, "succ": 0, "pending": False, "center": None, "ticks": [], "dists": [],
               "t0": 0, "box": False, "prim": []})
    P = planner

    def episode_meta(ep):
        sv = core.states_view
        o = sv["origin"][0].astype(np.float64)
        v = sv["velocity"][0].astype(np.float64)
        yaw = float(sv["yaw"][0])
        p, ln, end, total = P.draw(o, v, yaw, rng)
        if line is not None:
            line.set_lines(np.array([0]), [ln])
        if ball is not None:
            ball.set_goals([0], [end])
        ev["center"], ev["radius"], ev["pending"] = end, float(radius), False
        ev["n"] += 1
        ev["dists"].append(total)
        ev["prim"].append([float(x) for x in p])
        return {"goal": {"center": [float(x) for x in end], "radius": float(radius)},
                "line": [[float(x) for x in q] for q in ln],
                "plan": {"planner": "prim", "numbers": [round(float(x), 1) for x in p],
                         "length": round(total, 1)}}

    def on_tick(t, states, rewards, done, trunc):
        if bool(done[0]) or bool(trunc[0]):
            if ev["pending"]:
                ev["succ"] += 1
                ev["ticks"].append(t - ev["t0"])
            ev["pending"] = False
            ev["t0"] = t + 1
            return
        if ev["pending"] or ev["center"] is None:
            return
        o = core.states_view["origin"][0].astype(np.float64)
        if np.linalg.norm(o - ev["center"]) <= float(ev["radius"]):
            m = np.zeros(core.num_envs, np.uint8)
            m[0] = 1
            core.force_fail(m)
            ev["pending"] = True

    return episode_meta, on_tick
