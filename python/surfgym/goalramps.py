"""goalramps.py - the RAMP-WINDOW executor task (--goal-planner ramps; the user, 2026-09-28): the
executor is shown its next target surfaces as an IMAGE CHANNEL (surfgym.targetmask) and a LINE
(the fan) through them, and rides them one after another in one long episode.

* WINDOW: per env the next target T1 and the one after it T2 (surfaces of a surfgym.rampvocab
  vocabulary, or the finish box). A target is ELIGIBLE only when it lies closer to the finish
  than the agent - its geodesic distance (the map's goal field, the median over its contact
  origins) is PROGRESS_DELTA below the agent's, and T2's below T1's: the consecutive ramps down
  the geodesic potential (the user, 2026-09-28; without it a standing spawn's ballistic arc is a
  vertical drop and its closest targets were the start's own floors), beyond the WHOLE previous
  target (below its lowest part: the facing side of the same chute is not a next target) - and
  by at most what the
  horizon can cover: along any flown path the geodesic distance falls no faster than the path
  is long, so a target more than (speed + g * horizon + SPEED_MARGIN) * horizon below the agent
  cannot be the next one (without the cap utopia's chain jumped S27 -> S84, 85k u of geodesic
  across a wall the ballistic arc ignores; the finisher's consecutive ramps step at most 13.5k).
  Among the eligible, T1 is
  drawn among the top-k by closest approach of the spawn's ballistic arc (the source surface
  excluded) and T2 is the closest to the arc launched off T1's ride; the line is
  window_line([T1, T2]).
* CAPTURE and TAKEOFF come from collision telemetry (SurfCore.get_touch -> RampVocab.classify),
  never from positions: the first contact with T1 is its CAPTURE (the line is rebuilt as the ride
  along T1 then T2); DEPART_TICKS contact-free ticks after it are its TAKEOFF, and the window
  shifts (T2 -> T1, a new T2 from the new T1's ride). A contact with T2 before T1 skips T1.
* CHANNEL values, continuous (targetmask.takeoff_fade_values restricted to three slots): T1 = 1,
  T2 = 0.5; at a takeoff a FADE cross-fade runs - the ramp just left 1 -> 0, the new T1 0.5 -> 1,
  the new T2 0 -> 0.5 - so the channel never jumps at a touch or a takeoff.
* The reward is the goal-arc reward along the current line (GoalSystem / MultiArcProgress), its
  per-episode bank KEPT across window shifts (a new line re-anchors the arc at zero instantaneous
  reward; it does not mint a fresh shaping budget - Codex 23:16Z).

window_line is the ramp operator's line (tools/edge_archive.py RampOperator delegates here): one
line through a window of targets - per target an arrival into its plane (a cubic Hermite curve off
the current velocity, arriving tangentially RAMP_PRESS u inside its contact plane at the point of
the target closest to the ballistic path) and a RIDE along it at the arrival height to its far
edge; the finish box ends the line; past the last ride, RIDE_PAST u of lookahead.
"""
from __future__ import annotations

import numpy as np

# the ramp operator's constants (tools/edge_archive.py uses these values; one set for every map)
RAY_FLOOR = 300.0        # u/s: a line is laid at max(speed, this)
RAY_SPACING = 128.0      # u: the line's vertex spacing (the fan's)
RAMP_COAST = 6.0         # s: the ballistic path a target's arrival point is chosen on
RIDE_PAST = 256.0        # u: a ride line runs this far past the target's far edge
RAMP_PRESS = 16.0        # u: a line arrives this far inside the target's contact plane
DEPART_TICKS = 10        # contact-free physics ticks after a capture = the TAKEOFF

FIN = -2                 # the finish box as a target (surfgym.targetmask.FIN)
NONE = -1
RAMP_DEFAULTS = {"ramp_topk": 2, "ramp_horizon": 3.0, "ramp_fade": 0.3}
PROGRESS_DELTA = 250.0   # u: a target must lie this much closer to the finish (geodesic) than
                         # the agent (or than the previous target) to be eligible
SPEED_MARGIN = 300.0     # u/s added to the speed bound of the reach cap (air-strafe gain)


def find_goal_field(bsp):
    """the map's cached GEODESIC goal field (<map>.goal_<cell>.npz beside the .bsp) -> path, or None"""
    from pathlib import Path as _P
    b = _P(bsp)
    c = [q for q in sorted(b.parent.glob(f"{b.stem}.goal_*.npz"))
         if q.stem.split(".goal_")[-1].isdigit()]
    return str(c[0]) if c else None


def ride_dir(nb, toward):
    """a surface's in-plane HORIZONTAL direction (its level line), signed along `toward`; a floor
    (no level line) rides along `toward` projected into its plane"""
    lv = np.cross(nb, np.array([0.0, 0.0, 1.0]))
    if float(np.linalg.norm(lv)) < 1e-3:
        lv = toward - float(toward @ nb) * nb
    lv = lv / max(float(np.linalg.norm(lv)), 1e-9)
    return lv if float(lv @ toward) >= 0.0 else -lv


def window_line(tp, tn, tt, finish, origin, velocity, ks, *, fin, riding_first=False, coast=None,
                gravity=800.0, dt_path=0.05, line_cap=768, ray_floor=RAY_FLOOR,
                ray_spacing=RAY_SPACING, ramp_coast=RAMP_COAST, ramp_press=RAMP_PRESS,
                ride_past=RIDE_PAST, choose_next=None, chosen=None):
    """ONE line through a WINDOW of targets ks = [k0, k1, ...] (tp / tn / tt: per target its
    contact origins, their normals and a cKDTree over them; `fin` = the id that means the finish
    box). Per target: an arrival into its plane (k0 from the node's real coast when given, a
    later one from a ballistic launch off the previous ride) and a RIDE along it at the arrival
    height to its far edge; riding_first: the flight is ON k0 already and the line starts with its
    ride from here. The finish is a target like a ramp: the line arrives in its box and ends.
    A None in ks is chosen ON THE WAY by choose_next(launch point, launch velocity, previous k)
    - the next target from where the previous ride ends, in the same pass - and every resolved
    id is appended to `chosen` (a list) when one is given.
    -> (resampled line float32, raw points)."""
    from .route import resample_polyline
    o = np.asarray(origin, np.float64).reshape(3)
    v = np.asarray(velocity, np.float64).reshape(3)
    g = np.array([0.0, 0.0, -gravity])
    finish = np.asarray(finish, np.float64)
    segs = [o[None]]
    cur_p, cur_v = o, v
    prev_k = None
    for j, k in enumerate(ks):
        if k is None:
            k = choose_next(cur_p, cur_v, prev_k)
            if k is None or k == NONE:
                break
        k = int(k)
        prev_k = k
        if chosen is not None:
            chosen.append(k)
        spd = max(float(np.linalg.norm(cur_v)), ray_floor)
        if j == 0 and riding_first and k != fin:
            # already riding k0: its level line from here to its far edge
            dq, iq = tt[k].query(cur_p[None], k=1)
            nb = tn[k][int(iq[0])]
            lv = ride_dir(nb, cur_v / max(float(np.linalg.norm(cur_v)), 1e-9))
            ext = float(((tp[k] - cur_p[None]) @ lv).max())
            ss = np.arange(1, max(2, int(max(ext, 0.0) / (spd * 0.01))) + 1) * spd * 0.01
            segs.append(cur_p[None] + lv[None] * ss[:, None])
            cur_p, cur_v = segs[-1][-1], lv * spd
            continue
        if j == 0 and coast is not None:
            path, pvel = coast
        else:
            ts = np.arange(0.0, ramp_coast, dt_path)
            path = cur_p[None] + cur_v[None] * ts[:, None] + 0.5 * g[None] * ts[:, None] ** 2
            pvel = cur_v[None] + g[None] * ts[:, None]
        if k == fin:
            jj = int(np.argmin(np.linalg.norm(path - finish[None], axis=1)))
            pb, nb = finish, None
        else:
            dq, iq = tt[k].query(path, k=1)
            jj = int(np.argmin(dq))
            pb, nb = tp[k][int(iq[jj])], tn[k][int(iq[jj])]
        tc = max(0.2, jj * dt_path, float(np.linalg.norm(pb - cur_p)) / spd)
        vc = pvel[min(jj, len(pvel) - 1)]
        sp2 = max(float(np.linalg.norm(vc)), ray_floor)
        if nb is not None:
            u = vc - float(vc @ nb) * nb
            if np.linalg.norm(u) < 1e-3:
                u = (pb - cur_p) - float((pb - cur_p) @ nb) * nb
            end = pb - nb * ramp_press
        else:
            u = pb - cur_p
            end = pb
        un = u / max(float(np.linalg.norm(u)), 1e-6)
        vv = cur_v if np.linalg.norm(cur_v) >= 1.0 else un * ray_floor
        ss = np.linspace(0.0, 1.0, max(2, int(np.ceil(tc / 0.01))) + 1)
        h00 = 2 * ss ** 3 - 3 * ss ** 2 + 1
        h10 = ss ** 3 - 2 * ss ** 2 + ss
        h01 = -2 * ss ** 3 + 3 * ss ** 2
        h11 = ss ** 3 - ss ** 2
        herm = (h00[:, None] * cur_p[None] + h10[:, None] * (vv * tc)[None]
                + h01[:, None] * end[None] + h11[:, None] * (un * sp2 * tc)[None])
        segs.append(herm[1:])
        if nb is None:
            cur_p, cur_v = end, un * sp2
            break                                  # the finish: the line ends in its box
        lv = ride_dir(nb, un)
        ext = float(((tp[k] - end[None]) @ lv).max())
        rs = np.arange(1, max(2, int(max(ext, 0.0) / (sp2 * 0.01))) + 1) * sp2 * 0.01
        segs.append(end[None] + lv[None] * rs[:, None])
        cur_p, cur_v = segs[-1][-1], lv * sp2
    else:
        # past the last ride: ride_past of lookahead along the launch direction
        d = cur_v / max(float(np.linalg.norm(cur_v)), 1e-9)
        segs.append(cur_p[None] + d[None] * np.linspace(ride_past / 8, ride_past, 8)[:, None])
    pts = np.vstack(segs)
    line, _total = resample_polyline(pts, ray_spacing)
    if len(line) > line_cap:
        line = line[:line_cap]                     # the far end of the window is lookahead
    return np.asarray(line, np.float32), pts


class RampWindows:
    """per env the target window [T1, T2] of the ramp-window task, driven by collision telemetry.

    spawn(idx, origin, velocity, ducked) draws the windows of freshly spawned envs and returns
    their lines; on_tick(ids, origin, velocity, ended) consumes one physics tick's classified
    contacts (RampVocab.classify) and returns the envs whose line changed + their new lines;
    slots(idx) gives the channel's (ids, values) per env. deterministic=True (the eval) takes the
    closest candidate instead of a uniform draw among the top-k."""

    def __init__(self, vocab, n_envs: int, finish_box, tick_ms: float, *, topk: int = 4,
                 horizon: float = 3.0, fade: float = 0.3, gravity: float = 800.0,
                 line_cap: int = 768, rng=None, deterministic: bool = False, goal_field=None):
        from scipy.spatial import cKDTree
        self.voc = vocab
        # the geodesic potential that orders the targets (None = no progress filter)
        self.gf = goal_field
        self.N = int(n_envs)
        self.tick_ms = float(tick_ms)
        self.topk = max(1, int(topk))
        self.horizon = float(horizon)
        self.fade_ticks = max(1.0, float(fade) * 1000.0 / self.tick_ms)
        self.depart = int(DEPART_TICKS)
        self.gravity = float(gravity)
        self.line_cap = int(line_cap)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.deterministic = bool(deterministic)
        if isinstance(finish_box, dict):
            finish_box = (finish_box["mins"], finish_box["maxs"])
        self.fin_lo = np.asarray(finish_box[0], np.float64)
        self.fin_hi = np.asarray(finish_box[1], np.float64)
        self.finish = 0.5 * (self.fin_lo + self.fin_hi)
        self.dt_path = 5.0 * self.tick_ms / 1000.0
        self.tp, self.tn = vocab.tp, vocab.tn
        self.tt = {k: cKDTree(v) for k, v in self.tp.items()}
        # per target a bounding sphere of its origins: the arc query is best-first over these
        # lower bounds with EXACT per-target distances (one tree of every target's origins
        # returns the k nearest points, which all belong to the big surface under the arc -
        # the source itself - and hides every other target)
        self._ids = np.asarray(sorted(self.tp), np.int64)
        self._cen = (np.stack([self.tp[int(k)].mean(0) for k in self._ids])
                     if len(self._ids) else np.zeros((0, 3)))
        self._rad = (np.asarray([float(np.linalg.norm(self.tp[int(k)] - self._cen[j], axis=1).max())
                                 for j, k in enumerate(self._ids)])
                     if len(self._ids) else np.zeros(0))
        # per target its geodesic distance to the finish: the median over its contact origins,
        # and its LOWEST part (10th percentile) - the next target must lie beyond all of it
        self.d_surf = {}
        self.d_low = {}
        if self.gf is not None:
            for k in self._ids:
                d = np.asarray(self.gf.sample(self.tp[int(k)]), np.float64)
                d = d[d < self.gf.reach_max]
                self.d_surf[int(k)] = float(np.median(d)) if len(d) else np.inf
                self.d_low[int(k)] = float(np.percentile(d, 10)) if len(d) else np.inf
        n = self.N
        self.t1 = np.full(n, NONE, np.int64)
        self.t2 = np.full(n, NONE, np.int64)
        self.prev = np.full(n, NONE, np.int64)
        self.tau = np.full(n, 1 << 30, np.int64)      # ticks since the last takeoff
        self.captured = np.zeros(n, bool)              # T1 touched, not yet left
        self.free = np.zeros(n, np.int64)              # contact-free ticks since T1's last touch
        self.source = np.full(n, NONE, np.int64)       # the surface an env spawned on
        self.n_capt = np.zeros(n, np.int64)            # takeoffs (= completed rides) this episode
        self.n_skip = np.zeros(n, np.int64)
        self.stats = {"episodes": 0, "rides": 0, "skips": 0, "fin": 0, "ride_hist": {}}

    # ------------------------------------------------------------------ targets
    def _arc(self, p, v, secs):
        ts = np.arange(0.0, secs, 0.1)          # 0.1 s steps: ~350 u apart at surf speed
        g = np.array([0.0, 0.0, -self.gravity])
        return p[None] + v[None] * ts[:, None] + 0.5 * g[None] * ts[:, None] ** 2

    def _d_at(self, p):
        """the geodesic distance to the finish at p (inf without a field / off it)"""
        if self.gf is None:
            return np.inf
        d = float(self.gf.sample(np.asarray(p, np.float64)[None])[0])
        return d if d < self.gf.reach_max else np.inf

    def _reach(self, v):
        """u of geodesic the horizon can cover from speed |v| at most (the reach cap)"""
        return ((float(np.linalg.norm(v)) + self.gravity * self.horizon + SPEED_MARGIN)
                * self.horizon)

    def _candidates(self, p, v, exclude, d_max=np.inf, d_min=-np.inf):
        """ELIGIBLE targets (geodesic distance in [d_min, d_max)) by their closest approach to the
        ballistic arc from (p, v) within the horizon: -> [(dist, id)] sorted; the finish box is
        a candidate like a target and always eligible"""
        path = self._arc(p, v, self.horizon)
        need = self.topk + 1
        out = []
        if len(self._ids):
            lb = np.maximum(0.0, np.linalg.norm(path[:, None, :] - self._cen[None], axis=2).min(0)
                            - self._rad)
            for j in np.argsort(lb):
                s = int(self._ids[j])
                if s in exclude:
                    continue
                if self.d_surf and not (d_min <= self.d_surf.get(s, np.inf) < d_max):
                    continue
                if len(out) >= need and lb[j] > out[need - 1][0]:
                    break                          # no later target can beat the k-th best
                dq, _iq = self.tt[s].query(path, k=1)
                out.append((float(dq.min()), s))
                out.sort()
        # the finish box: the arc's closest approach to the box (0 inside it)
        q = np.clip(path, self.fin_lo[None], self.fin_hi[None])
        out.append((float(np.linalg.norm(path - q, axis=1).min()), FIN))
        return sorted(out)

    def _pick(self, cands):
        if not cands:
            return FIN
        if self.deterministic:
            return int(cands[0][1])
        k = min(self.topk, len(cands))
        return int(cands[int(self.rng.integers(k))][1])

    def _next_fn(self, exclude):
        """choose_next for window_line: the target closest to the arc launched at the end of the
        previous ride (the previous target and `exclude` never)"""
        def f(cp, cv, prev):
            cp = np.asarray(cp, np.float64)
            cv = np.asarray(cv, np.float64)
            d_max, d_min = np.inf, -np.inf
            if self.d_surf:
                # BEYOND the previous target (below its lowest part: its facing sibling on the
                # same chute is not a next target) and below the launch point; within the reach
                # cap of it
                d_ref = min(self.d_low.get(int(prev), np.inf) if prev is not None else np.inf,
                            self._d_at(cp))
                if np.isfinite(d_ref):
                    d_max = d_ref - PROGRESS_DELTA
                    d_min = d_ref - self._reach(cv)
            c = self._candidates(cp, cv, exclude | ({int(prev)} if prev is not None else set()),
                                 d_max=d_max, d_min=d_min)
            return int(c[0][1]) if c else FIN
        return f

    def _window(self, p, v, t1, exclude, riding=False):
        """ONE pass: the line through T1 and a T2 chosen where T1's ride ends -> (line, t2)"""
        if t1 == FIN:
            ln, _raw = window_line(self.tp, self.tn, self.tt, self.finish, p, v, [FIN], fin=FIN,
                                   gravity=self.gravity, dt_path=self.dt_path,
                                   line_cap=self.line_cap)
            return ln, NONE
        got = []
        ln, _raw = window_line(self.tp, self.tn, self.tt, self.finish, p, v, [t1, None], fin=FIN,
                               riding_first=riding, gravity=self.gravity, dt_path=self.dt_path,
                               line_cap=self.line_cap, choose_next=self._next_fn(exclude),
                               chosen=got)
        return ln, (got[1] if len(got) > 1 else NONE)

    def _after(self, s, p, v, exclude):
        """the target after s: the closest to the ballistic arc launched off s's ride (the end of
        window_line([s]) and its direction)"""
        if s == FIN:
            return NONE
        ln, raw = window_line(self.tp, self.tn, self.tt, self.finish, p, v, [s], fin=FIN,
                              gravity=self.gravity, dt_path=self.dt_path, line_cap=self.line_cap)
        pe = raw[-1]
        d = raw[-1] - raw[-2] if len(raw) >= 2 else v
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        spd = max(float(np.linalg.norm(v)), RAY_FLOOR)
        c = self._candidates(pe, d * spd, exclude | {s})
        return int(c[0][1]) if c else FIN

    def _line(self, i, p, v, riding):
        ks = [int(self.t1[i])] + ([int(self.t2[i])] if self.t2[i] != NONE else [])
        ln, _raw = window_line(self.tp, self.tn, self.tt, self.finish, p, v, ks, fin=FIN,
                               riding_first=riding, gravity=self.gravity, dt_path=self.dt_path,
                               line_cap=self.line_cap)
        return ln

    # ------------------------------------------------------------------ spawn
    def spawn(self, idx, origin, velocity, source=None):
        """fresh episodes for envs idx: draw T1 / T2 from their spawn states -> list of lines.
        `source` (len(idx),) = the surface each spawn is on (never drawn as T1), NONE if none"""
        idx = np.asarray(idx, np.int64).reshape(-1)
        origin = np.asarray(origin, np.float64).reshape(-1, 3)
        velocity = np.asarray(velocity, np.float64).reshape(-1, 3)
        src = (np.full(len(idx), NONE) if source is None
               else np.asarray(source, np.int64).reshape(-1))
        lines = []
        for n, i in enumerate(idx):
            p, v = origin[n], velocity[n]
            ex = {int(src[n])} if src[n] != NONE else set()
            # a slow spawn (standing on the start, say) would draw its arc as a vertical drop:
            # lay it along the geodesic field's descent direction at RAY_FLOOR instead
            va = v
            if self.gf is not None and float(np.linalg.norm(v[:2])) < RAY_FLOOR:
                yaw = np.radians(float(self.gf.descent_yaw(p[None])[0]))
                va = np.array([np.cos(yaw) * RAY_FLOOR, np.sin(yaw) * RAY_FLOOR, v[2]])
            d0 = self._d_at(p)
            if src[n] != NONE and int(src[n]) in self.d_low:
                d0 = min(d0, self.d_low[int(src[n])])      # beyond the surface it spawned on
            t1 = self._pick(self._candidates(p, va, ex, d_max=d0 - PROGRESS_DELTA,
                                             d_min=d0 - self._reach(va)))
            ln, t2 = self._window(p, v, t1, ex)
            self.t1[i] = t1
            self.t2[i] = t2
            self.prev[i] = NONE
            self.tau[i] = 1 << 30
            self.captured[i] = False
            self.free[i] = 0
            self.source[i] = src[n]
            self.n_capt[i] = 0
            self.n_skip[i] = 0
            lines.append(ln)
        return lines

    # ------------------------------------------------------------------ tick
    def on_tick(self, ids, origin, velocity, ended):
        """one physics tick: ids (N, T) classified contacts; ended (N,) bool (those rows are
        settled and skipped - they respawn through spawn()). -> (idx of envs whose line changed,
        their lines)"""
        ids = np.asarray(ids)
        ended = np.asarray(ended, bool)
        self.tau += 1
        live = ~ended
        on1 = (ids == self.t1[:, None]).any(1) & live & (self.t1 >= 0)
        on2 = (ids == self.t2[:, None]).any(1) & live & (self.t2 >= 0)
        changed = {}
        # a contact with T2 before T1 is captured: T1 is skipped
        for i in np.flatnonzero(on2 & ~on1 & ~self.captured):
            self.n_skip[i] += 1
            self.stats["skips"] += 1
            changed[int(i)] = self._shift(i, origin[i], velocity[i], riding=True)
        on1 = (ids == self.t1[:, None]).any(1) & live & (self.t1 >= 0)
        # the first contact with T1: its CAPTURE - the line becomes the ride along T1 then T2
        cap = on1 & ~self.captured
        for i in np.flatnonzero(cap):
            self.captured[i] = True
            if int(i) not in changed:
                changed[int(i)] = None             # the ride line, built below
        self.free[on1] = 0
        self.free[self.captured & ~on1 & live] += 1
        # DEPART_TICKS contact-free ticks after a capture: the TAKEOFF - the window shifts
        for i in np.flatnonzero(self.captured & live & (self.free >= self.depart)):
            self.n_capt[i] += 1
            self.stats["rides"] += 1
            changed[int(i)] = self._shift(i, origin[i], velocity[i], riding=False)
        idx = np.asarray(sorted(changed), np.int64)
        lines = [changed[int(i)] if changed[int(i)] is not None
                 else self._line(int(i), origin[i], velocity[i], True) for i in idx]
        return idx, lines

    def _shift(self, i, p, v, riding):
        """T1 done (left, or skipped): T2 becomes T1, a new T2 chosen where its ride ends - one
        pass that also lays the new line -> the line"""
        self.prev[i] = self.t1[i]
        self.t1[i] = self.t2[i] if self.t2[i] != NONE else FIN
        ln, t2 = self._window(p, v, int(self.t1[i]), {int(self.prev[i])}, riding=bool(riding))
        self.t2[i] = t2
        self.tau[i] = 0
        self.captured[i] = bool(riding)
        self.free[i] = 0
        return ln

    def settle(self, idx, finished):
        """episodes of envs idx ended (finished: reached the finish box) - the stats"""
        for i, f in zip(np.asarray(idx, np.int64).reshape(-1),
                        np.asarray(finished, bool).reshape(-1)):
            self.stats["episodes"] += 1
            self.stats["fin"] += int(f)
            r = int(self.n_capt[i])
            self.stats["ride_hist"][r] = self.stats["ride_hist"].get(r, 0) + 1

    def pop_stats(self) -> dict:
        s = self.stats
        self.stats = {"episodes": 0, "rides": 0, "skips": 0, "fin": 0, "ride_hist": {}}
        return s

    # ------------------------------------------------------------------ channel
    def slots(self, idx=None):
        """(ids (n, 3), values (n, 3)): [the target just left, T1, T2] with the continuous takeoff
        cross-fade (1 -> 0, 0.5 -> 1, 0 -> 0.5 over the fade)"""
        sl = slice(None) if idx is None else np.asarray(idx, np.int64)
        f = np.minimum(1.0, self.tau[sl] / self.fade_ticks)
        ids = np.stack([self.prev[sl], self.t1[sl], self.t2[sl]], axis=1)
        vals = np.stack([np.where(self.prev[sl] != NONE, 1.0 - f, 0.0),
                         np.where(self.t1[sl] != NONE, 0.5 + 0.5 * f, 0.0),
                         np.where(self.t2[sl] != NONE, 0.5 * f, 0.0)], axis=1)
        vals[ids == NONE] = 0.0
        return ids, vals.astype(np.float32)

    def describe(self) -> str:
        return (f"ramp windows: {self.N} envs, targets "
                + ("ORDERED by the geodesic potential (eligible = closer to the finish by "
                   f"{PROGRESS_DELTA:g} u than the agent / the previous target), "
                   if self.d_surf else "NOT ordered (no goal field), ")
                + f"T1 drawn {'closest' if self.deterministic else f'among the top {self.topk}'} "
                f"targets by closest approach of the {self.horizon:g} s ballistic arc, T2 off T1's "
                f"ride; capture = first telemetry contact, takeoff = {self.depart} contact-free "
                f"ticks; channel fade {self.fade_ticks * self.tick_ms / 1000.0:g} s")


class RampPlanner:
    """--goal-planner ramps: the GoalSystem's planner for the ramp-window task - the vocabulary,
    the training fleet's windows (random among the top-k) and the eval's (one env, the closest
    candidate every time)."""

    ramps = True
    primitive = False

    def __init__(self, vocab, n_envs: int, finish_box, tick_ms: float, *, topk: int, horizon: float,
                 fade: float, gravity: float = 800.0, line_cap: int = 768, seed: int = 0,
                 goal_field=None):
        self.vocab = vocab
        kw = dict(topk=topk, horizon=horizon, fade=fade, gravity=gravity, line_cap=line_cap,
                  goal_field=goal_field)
        self.windows = RampWindows(vocab, n_envs, finish_box, tick_ms,
                                   rng=np.random.default_rng(int(seed)), **kw)
        self.eval_windows = RampWindows(vocab, 1, finish_box, tick_ms, deterministic=True, **kw)
        self.finish_center = self.windows.finish
        self.fin = None

    def describe(self) -> str:
        return self.vocab.describe() + "\n" + self.windows.describe()


class TargetLidar:
    """A lidar with the target channel appended (the GoalBallLidar pattern): channels = the
    wrapped lidar's + 1; render() draws the per-env window slots of `windows` (RampWindows) through
    `tm` (surfgym.targetmask.TargetMask, no occlusion, max-combine). mode "off" renders the same
    channel as zeros - the control arm with the identical architecture."""

    def __init__(self, lidar, tm, windows=None, mode: str = "live"):
        if mode not in ("live", "off"):
            raise ValueError(f"TargetLidar: mode live|off, got {mode!r}")
        if getattr(lidar, "pinhole", False):
            raise ValueError("TargetLidar mirrors the equiangular camera; --pinhole is separate")
        self.lidar, self.tm, self.windows, self.mode = lidar, tm, windows, mode
        self.base = int(getattr(lidar, "channels", 1))
        self.channels = self.base + 1
        self.W, self.H = int(lidar.W), int(lidar.H)
        self.device = lidar.device
        self.near, self.range = float(lidar.near), float(lidar.range)
        self.yoff, self.poff = lidar.yoff, lidar.poff
        self.pinhole = False
        self.takes_idx = True          # MapFleet.render_rows passes a subset batch's env rows
        self.potential = getattr(lidar, "potential", None)
        self.vision_clip = bool(getattr(lidar, "vision_clip", False))

    def __getattr__(self, name):
        # everything else (decode_depth, _ensure_buffers, cell, ...) is the wrapped lidar's
        return getattr(self.__dict__["lidar"], name)

    def describe(self) -> str:
        return (f"target channel ({self.mode}): the window's surfaces drawn without occlusion, "
                f"after the lidar's {self.base} channel(s) -> in_ch {self.channels}")

    def render(self, origin, yaw_deg, pitch_deg, ducked, idx=None, **kw):
        import torch
        img = self.lidar.render(origin, yaw_deg, pitch_deg, ducked, **kw)
        if img.dim() == 3:
            img = img.unsqueeze(-1)
        n = origin.shape[0]
        if self.mode == "off" or self.windows is None:
            ch = torch.zeros((n, self.H, self.W), dtype=img.dtype, device=img.device)
        else:
            ids, vals = self.windows.slots(idx)
            if len(ids) != n:
                raise ValueError(f"TargetLidar: {n} poses but {len(ids)} windows (pass idx for "
                                 "a subset batch)")
            self.tm.set_slots(n, ids, vals, combine="max")
            ch = self.tm.render(self.lidar, origin, yaw_deg, pitch_deg, ducked).to(img.dtype)
        return torch.cat([img, ch.unsqueeze(-1)], dim=-1)


def make_ramp_hooks(windows, vocab, core, ev: dict, *, line=None):
    """(episode_meta, on_tick) for record_rollout on a core whose env 0 is recorded - the
    ramp-window eval, shared by the trainer and tools/record_ckpt.py. Each episode draws env 0's
    window from its spawn state (windows is a 1-env RampWindows, deterministic for the headline);
    every tick reads env 0's collision telemetry, advances the window and keeps `line` (the eval
    fan) on it. ev tallies episodes, rides, skips and finishes."""
    ev.update({"n": 0, "succ": 0, "rides": [], "skips": 0, "chain": []})
    # the window as the policy SAW it, for the POV render (tools/render_pov.py --targets): one
    # event per change, [row, prev, T1, T2, tau at that row] - the channel's values at row k are
    # the fade of tau + (k - row) (slot_values); rows are the episode's own tick index
    rec = {"t0": None, "events": []}

    def _snap(row):
        e = [int(row), int(windows.prev[0]), int(windows.t1[0]), int(windows.t2[0]),
             int(min(windows.tau[0], 1 << 20))]
        if not rec["events"] or rec["events"][-1][1:4] != e[1:4]:
            rec["events"].append(e)

    def _src():
        sv = core.states_view
        return vocab.contact_of(sv["origin"][:1].astype(np.float64),
                                sv["ducked"][:1].astype(np.int64))

    def episode_meta(ep):
        sv = core.states_view
        o = sv["origin"][0].astype(np.float64)
        v = sv["velocity"][0].astype(np.float64)
        ln = windows.spawn([0], o[None], v[None], source=_src())[0]
        rec["t0"] = None
        rec["events"] = []
        _snap(0)
        if line is not None:
            line.set_lines(np.array([0]), [ln])
        ev["n"] += 1
        ev["chain"].append([int(windows.t1[0])])
        thin = ln[:: max(1, len(ln) // 64)]
        return {"line": [[float(x) for x in q] for q in thin],
                "plan": {"planner": "ramps", "t1": int(windows.t1[0]), "t2": int(windows.t2[0])}}

    def on_tick(t, states, rewards, done, trunc):
        if rec["t0"] is None:
            rec["t0"] = int(t)
        ended = bool(done[0]) or bool(trunc[0])
        if ended:
            fin = bool(np.asarray(core.goal_hits, bool)[0])
            ev["succ"] += int(fin)
            ev["rides"].append(int(windows.n_capt[0]))
            ev["skips"] += int(windows.n_skip[0])
            windows.settle([0], [fin])
            return
        cnt, nrm, pts = core.get_touch()
        sv = core.states_view
        ids = vocab.classify(cnt[:1], nrm[:1], pts[:1], sv["ducked"][:1])
        idx, lines = windows.on_tick(ids, sv["origin"][:1].astype(np.float64),
                                     sv["velocity"][:1].astype(np.float64), np.zeros(1, bool))
        if len(idx):
            if ev["chain"]:
                ev["chain"][-1].append(int(windows.t1[0]))
            if line is not None:
                line.set_lines(idx, lines)
            # the policy sees the new window from the NEXT row on
            _snap(int(t) - int(rec["t0"]) + 1)

    def episode_end(ep):
        return {"targets": {"events": list(rec["events"]),
                            "fade_ticks": float(windows.fade_ticks)}}

    episode_meta.episode_end = episode_end
    return episode_meta, on_tick


def slot_values(events, n_rows, fade_ticks):
    """a recorded episode's window events (make_ramp_hooks' trailer) -> per row the channel's
    three slots: (ids (n, 3) int64, values (n, 3) float32) - exactly RampWindows.slots"""
    ids = np.full((n_rows, 3), NONE, np.int64)
    vals = np.zeros((n_rows, 3), np.float32)
    ev = sorted(events, key=lambda e: e[0])
    for j, (row, prev, t1, t2, tau0) in enumerate(ev):
        lo = max(0, int(row))
        hi = n_rows if j + 1 == len(ev) else min(n_rows, int(ev[j + 1][0]))
        if hi <= lo:
            continue
        k = np.arange(lo, hi)
        f = np.minimum(1.0, (float(tau0) + (k - lo)) / max(float(fade_ticks), 1.0))
        ids[lo:hi] = (prev, t1, t2)
        vals[lo:hi, 0] = (1.0 - f) if prev != NONE else 0.0
        vals[lo:hi, 1] = (0.5 + 0.5 * f) if t1 != NONE else 0.0
        vals[lo:hi, 2] = (0.5 * f) if t2 != NONE else 0.0
    return ids, vals
