"""goalramps.py - the RAMP-WINDOW executor task (--goal-planner ramps; the user, 2026-09-28): the
executor is shown its next target surfaces as an IMAGE CHANNEL (surfgym.targetmask) and a LINE
(the fan) through them, and rides them one after another in one long episode.

* WINDOW: per env the next target T1 and the one after it T2 (surfaces of a surfgym.rampvocab
  vocabulary, or the finish box). A target is ELIGIBLE only when it lies closer to the finish
  than the agent - its geodesic distance (the map's goal field, the median over its contact
  origins) is PROGRESS_DELTA below the agent's, and T2's below T1's: the consecutive ramps down
  the geodesic potential (the user, 2026-09-28; without it a standing spawn's ballistic arc is a
  vertical drop and its closest targets were the start's own floors), beyond the WHOLE previous
  piece (below its lowest part: the facing side of the same chute is not a next target) - and
  by at most what the
  horizon can cover: along any flown path the geodesic distance falls no faster than the path
  is long, so a target more than (speed + g * horizon + SPEED_MARGIN) * horizon below the agent
  cannot be the next one (without the cap utopia's chain jumped S27 -> S84, 85k u of geodesic
  across a wall the ballistic arc ignores; the finisher's consecutive ramps step at most 13.5k).
  The choice is per PIECE (RampVocab.piece: target surfaces joined at an edge - a wedge's two
  sides and its end caps): among the eligible, T1 is drawn among the top-k pieces by closest
  approach of the spawn's ballistic arc (the piece it spawned on excluded) and T2 is the closest
  piece to the arc launched off T1's ride (T1's piece excluded); a piece enters the window as its
  RIDE surface - the one running furthest along the travel direction, less the arc's distance to
  it - so a wedge's side, never its end cap (without pieces utopia's chain went S28 -> S30 -> S31:
  the end caps of two wedges, and the line off S30's ride turned the agent 90 deg off the route);
  the line is window_line([T1, T2]). A standing spawn's window is laid along the same descent
  direction as its T1 draw (with the spawn's own zero velocity the ride along T1 had no direction
  and ran back toward the start).
* ENTER and PASS are BOXES (the user, 2026-09-28: "put a bounding box just around each ramp ...
  enter the bounding box and exit the bounding box ... that's what we consider a finished ramp"):
  per piece a box in its own frame (its main axis, the horizontal normal to it, z) around every
  validated contact origin of its surfaces, BOX_MARGIN wider on every side. Being inside T1's box
  ENTERS T1 (the line is rebuilt as the ride along T1 then T2 from there); leaving it after that
  PASSES T1 and the window shifts (T2 -> T1, a new T2 from the new T1's ride); being inside T2's
  box before T1 was entered SKIPS T1. Positions only - no collision telemetry, no departure
  count, no hop test: a hop stays in the box (the utopia finisher: 98% of its ride ticks inside
  with no margin, all 49 of its hops inside with this one, the box left a median 4 ticks after
  its last contact). Replaces the telemetry capture / takeoff, 2.5 s of a 7 s training iteration.
* OFF-TARGET (--ramp-offtarget-pen, the user: "penalize for surfing on things that it's not
  supposed to surf"): a tick with a contact on a RAMP-like plane (normal z in RAMP_NZ, the
  extractor's ramp category) outside the boxes of the pieces the env may ride - T1, T2, and the
  one it spawned on until its first pass - is off-target (RampWindows.offtarget); going back to
  a piece already passed is off-target.
* CHANNEL values, continuous (targetmask.takeoff_fade_values restricted to three slots): T1 = 1,
  T2 = 0.5; at a takeoff a FADE cross-fade runs - the ramp just left 1 -> 0, the new T1 0.5 -> 1,
  the new T2 0 -> 0.5 - so the channel never jumps at a touch or a takeoff. Each fade starts from
  the value its surface shows at the shift, so a second shift inside a fade does not jump either
  (only the oldest surface, dropped from the three slots, goes to zero at once).
* The reward (--ramp-reward): "arc" = the goal-arc reward along the current line (GoalSystem /
  MultiArcProgress), its per-episode bank KEPT across window shifts (a new line re-anchors the arc
  at zero instantaneous reward; it does not mint a fresh shaping budget - Codex 23:16Z); "pass"
  (the user, 2026-09-28: "plus one for one ramp fully passed ... impossible to farm reward by
  just sliding on the same thing") = +1 per WINDOW SHIFT and nothing else - a target ridden and
  left, or skipped for the one after it, the moment a new target enters the channel. A piece
  pays once: the next target always lies beyond the whole previous piece on the geodesic.
  RampWindows.tick_pass holds this tick's shifts per env.

window_line is the ramp operator's line (tools/edge_archive.py RampOperator delegates here): one
line through a window of targets - per target an arrival into its plane (a cubic Hermite curve off
the current velocity, arriving tangentially RAMP_PRESS u inside its contact plane at the point of
the target closest to the ballistic path) and a RIDE along it at the arrival height to its far
edge; the finish box ends the line; past the last ride, RIDE_PAST u of lookahead.
"""
from __future__ import annotations

import os

import numpy as np

from . import rampfast as _rf

# the ramp operator's constants (tools/edge_archive.py uses these values; one set for every map)
RAY_FLOOR = 300.0        # u/s: a line is laid at max(speed, this)
RAY_SPACING = 128.0      # u: the line's vertex spacing (the fan's)
RAMP_COAST = 6.0         # s: the ballistic path a target's arrival point is chosen on
RIDE_PAST = 256.0        # u: a ride line runs this far past the target's far edge
RAMP_PRESS = 16.0        # u: a line arrives this far inside the target's contact plane
BOX_MARGIN = 48.0        # u: a piece's box reaches this far past its validated contact origins
                         # (rampvocab.LATERAL_MIN, the vocabulary's own contact radius)
RAMP_NZ = (0.02, 0.7)    # a RAMP-like contact plane: tools/ramps_mesh.py's category rule

FIN = -2                 # the finish box as a target (surfgym.targetmask.FIN)
NONE = -1
RAMP_DEFAULTS = {"ramp_topk": 2, "ramp_horizon": 3.0, "ramp_fade": 0.3, "ramp_reward": "arc",
                 "ramp_offtarget_pen": 0.0, "ramp_obs_pass": 0, "target_views": 1,
                 "ramp_exit_bonus": 0.0}
# --target-views 6: the target channel's five extra directions after the view's own, as (name, yaw
# offset deg, pitch deg, horizontal span deg or None = the lidar's own). Back / left / right are
# level; up / down look straight up / down with a 180 deg span, so the six views cover the whole
# sphere around the agent (the level views cover +-vfov/2 about the horizon, the two poles the
# caps beyond it)
TARGET_VIEWS = (("up", 0.0, 90.0, 180.0), ("down", 0.0, -90.0, 180.0),
                ("back", 180.0, 0.0, None), ("left", 90.0, 0.0, None), ("right", -90.0, 0.0, None))
PROGRESS_DELTA = 250.0   # u: a target must lie this much closer to the finish (geodesic) than
                         # the agent (or than the previous target) to be eligible
REPLAN_SECS = 0.1        # s: a HOLDING window (no target was in reach) redraws this often
SPEED_MARGIN = 300.0     # u/s added to the speed bound of the reach cap (air-strafe gain)


# the compiled target search and window builder (surfgym.rampfast: nearest_pair, candidates,
# ride_face, next_target, window, gf_sample1) - exact ports of the Python path below; None = no
# numba or SURFGYM_NO_NUMBA=1, and the Python / KD path runs
_FAST_SEARCH = _rf.FAST


def find_goal_field(bsp, cell=None):
    """the map's cached GEODESIC goal field (<map>.goal_<cell>.npz beside the .bsp) -> path, or
    None: the run's own --goal-cell when given and baked, else the first baked one (it took the
    first by name whatever the run used - Codex 2026-09-28)"""
    from pathlib import Path as _P
    b = _P(bsp)
    if cell is not None:
        try:
            want = b.parent / f"{b.stem}.goal_{int(float(cell))}.npz"
        except (TypeError, ValueError):
            want = None
        if want is not None and want.exists():
            return str(want)
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
                ride_past=RIDE_PAST, choose_next=None, chosen=None, nearest=None):
    """ONE line through a WINDOW of targets ks = [k0, k1, ...] (tp / tn / tt: per target its
    contact origins, their normals and a cKDTree over them; `fin` = the id that means the finish
    box). Per target: an arrival into its plane (k0 from the node's real coast when given, a
    later one from a ballistic launch off the previous ride) and a RIDE along it at the arrival
    height to its far edge; riding_first: the flight is ON k0 already and the line starts with its
    ride from here. The finish is a target like a ramp: the line arrives in its box and ends.
    A None in ks is chosen ON THE WAY by choose_next(launch point, launch velocity, previous k)
    - the next target from where the previous ride ends, in the same pass - and every resolved
    id is appended to `chosen` (a list) when one is given. nearest(k, points) -> (distance,
    point index, origin index) of the closest pair replaces the KD queries (RampWindows' exact
    numba search; None = tt).
    -> (resampled line float32, raw points)."""
    from .route import resample_polyline
    o = np.asarray(origin, np.float64).reshape(3)
    v = np.asarray(velocity, np.float64).reshape(3)
    g = np.array([0.0, 0.0, -gravity])
    finish = np.asarray(finish, np.float64)
    segs = [o[None]]
    cur_p, cur_v = o, v
    prev_k = None
    at_finish = False
    for j, k in enumerate(ks):
        if k is None:
            k = choose_next(cur_p, cur_v, prev_k)
            if k is None or k == NONE:
                break                              # nothing next: the last ride's lookahead
        k = int(k)
        prev_k = k
        if chosen is not None:
            chosen.append(k)
        spd = max(float(np.linalg.norm(cur_v)), ray_floor)
        if j == 0 and riding_first and k != fin:
            # already riding k0: its level line from here to its far edge
            if nearest is not None:
                nb = tn[k][int(nearest(k, cur_p[None])[2])]
            else:
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
        elif nearest is not None:
            _dm, jj, io = nearest(k, path)
            jj, io = int(jj), int(io)
            pb, nb = tp[k][io], tn[k][io]
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
            at_finish = True
            break                                  # the finish: the line ends in its box
        lv = ride_dir(nb, un)
        ext = float(((tp[k] - end[None]) @ lv).max())
        rs = np.arange(1, max(2, int(max(ext, 0.0) / (sp2 * 0.01))) + 1) * sp2 * 0.01
        segs.append(end[None] + lv[None] * rs[:, None])
        cur_p, cur_v = segs[-1][-1], lv * sp2
    if not at_finish:
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
    closest candidate instead of a uniform draw among the top-k.

    T1 / T2 are SURFACES (what the channel draws and the line rides); the choice, the capture and
    the takeoff are per PIECE (RampVocab.piece: surfaces joined at an edge - a wedge's two sides
    and its end caps)."""

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
        self.replan_ticks = max(1, int(round(REPLAN_SECS * 1000.0 / self.tick_ms)))
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
        # PIECES: target surface -> piece (a vocabulary without them: every surface its own), a
        # lookup table over every surface id for the telemetry (-1: not a target), and per piece
        # its surfaces and its origins together (the takeoff test's "back down onto it")
        pc = getattr(vocab, "piece", None)
        self._pof = {int(k): (int(pc[k]) if pc is not None and int(pc[k]) >= 0 else int(k))
                     for k in self.tp}
        n_ids = max([int(getattr(vocab, "n_surf", 0))] + [k + 1 for k in self.tp])
        self.pmap = np.full(max(n_ids, 1), -1, np.int64)
        for k, q in self._pof.items():
            self.pmap[k] = q
        self.pfaces = {}
        for k in sorted(self.tp):
            self.pfaces.setdefault(self._pof[int(k)], []).append(int(k))
        area = getattr(vocab, "area", None)
        # per piece its main AXIS: the level line of its largest surface (horizontal unit
        # vector; None for a floor) - its ride surface is the one running furthest along it
        nrm_v = getattr(vocab, "normal", None)
        self.p_axis = {}
        for q, fs in self.pfaces.items():
            big = max(fs, key=lambda f: (float(area[f]) if area is not None
                                         else float(len(self.tp[f]))))
            nb = (np.asarray(nrm_v[big], np.float64) if nrm_v is not None
                  else self.tn[big].mean(0))
            ax = np.array([float(nb[1]), -float(nb[0])])          # cross(nb, z) horizontally
            na = float(np.linalg.norm(ax))
            self.p_axis[q] = ax / na if na >= 1e-3 else None
        # per piece its BOX: in its frame (the axis - for a floor piece its origins' principal
        # horizontal direction -, the horizontal normal to it, z) the extent of every validated
        # contact origin of its surfaces, BOX_MARGIN wider on every side:
        # (centre xy, axis, normal, a0, a1, b0, b1, z0, z1)
        pts_all = getattr(vocab, "points", None)
        srf_all = getattr(vocab, "surf", None)
        self.p_box = {}
        for q, fs in self.pfaces.items():
            P = pts_all[np.isin(srf_all, fs)] if pts_all is not None else np.zeros((0, 3))
            if not len(P):
                P = np.concatenate([self.tp[f] for f in fs])
            ax = self.p_axis[q]
            if ax is None:
                ax = np.array([1.0, 0.0])
                if len(P) >= 2:
                    _w, e = np.linalg.eigh(np.cov((P[:, :2] - P[:, :2].mean(0)).T))
                    ax = e[:, -1] / max(float(np.linalg.norm(e[:, -1])), 1e-9)
            nv = np.array([-float(ax[1]), float(ax[0])])
            c = P[:, :2].mean(0)
            a = (P[:, :2] - c) @ ax
            b = (P[:, :2] - c) @ nv
            m = BOX_MARGIN
            self.p_box[q] = (c, np.asarray(ax, np.float64), nv, float(a.min()) - m,
                             float(a.max()) + m, float(b.min()) - m, float(b.max()) + m,
                             float(P[:, 2].min()) - m, float(P[:, 2].max()) + m)
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
        # the numba search's flat view (surfgym.goalramps._FAST_SEARCH): every target's origins
        # in _ids order, where each starts, how many, its piece (compact 0..n_p-1)
        self._jof = {int(k): j for j, k in enumerate(self._ids)}
        self._O = (np.ascontiguousarray(np.concatenate([self.tp[int(k)] for k in self._ids]),
                                        np.float64) if len(self._ids) else np.zeros((0, 3)))
        self._ocount = np.asarray([len(self.tp[int(k)]) for k in self._ids], np.int64)
        self._ostart = np.concatenate([[0], np.cumsum(self._ocount)[:-1]]).astype(np.int64)
        _pc = sorted(self.pfaces)
        self._pcompact = {q: i for i, q in enumerate(_pc)}
        self._pfull = np.asarray(_pc, np.int64)
        self._pj = np.asarray([self._pcompact[self._pof[int(k)]] for k in self._ids], np.int64)
        self._fast = _FAST_SEARCH
        # the boxes as arrays by compact piece, and surface id -> compact piece (-1: none)
        self._cp_of = np.full(len(self.pmap), -1, np.int64)
        for k, q in self._pof.items():
            self._cp_of[k] = self._pcompact[q]
        _B = [self.p_box[int(q)] for q in self._pfull]
        self._bc = np.asarray([b_[0] for b_ in _B], np.float64).reshape(-1, 2)
        self._bu = np.asarray([b_[1] for b_ in _B], np.float64).reshape(-1, 2)
        self._bv = np.asarray([b_[2] for b_ in _B], np.float64).reshape(-1, 2)
        self._blim = np.asarray([b_[3:] for b_ in _B], np.float64).reshape(-1, 6)
        # per PIECE its geodesic distance to the finish - the median over all its contact
        # origins, which decides its ELIGIBILITY (per surface it let a piece in through its end
        # cap alone: utopia's [28, 29, 30] - sides 156.5k / 156.3k, cap 154.0k - was eligible
        # below 155k as the cap, Codex 2026-09-28) - and its lowest part (10th percentile): the
        # next target must lie beyond all of it. d_surf keeps, per surface, its piece's median
        self.d_surf = {}
        self.p_low = {}
        self.p_med = {}
        if self.gf is not None:
            dd = {}
            for k in self._ids:
                d = np.asarray(self.gf.sample(self.tp[int(k)]), np.float64)
                dd[int(k)] = d[d < self.gf.reach_max]
            for q, fs in self.pfaces.items():
                d = np.concatenate([dd[f] for f in fs])
                self.p_low[q] = float(np.percentile(d, 10)) if len(d) else np.inf
                self.p_med[q] = float(np.median(d)) if len(d) else np.inf
            for k in self._ids:
                self.d_surf[int(k)] = self.p_med[self._pof[int(k)]]
        self._dsurf = np.asarray([self.d_surf.get(int(k), np.inf) for k in self._ids], np.float64)
        n = self.N
        self.t1 = np.full(n, NONE, np.int64)
        self.t2 = np.full(n, NONE, np.int64)
        self.prev = np.full(n, NONE, np.int64)
        self.tau = np.full(n, 1 << 30, np.int64)      # ticks since the last takeoff
        # where the last shift's fades START: the surface just left from v0p (1 -> 0), T1 from
        # v0t (-> 1) - the values those surfaces showed at the shift (1 and 0.5 after a full fade)
        self.v0p = np.ones(n, np.float64)
        self.v0t = np.full(n, 0.5, np.float64)
        self.entered = np.zeros(n, bool)               # inside T1's box once, not yet left
        # a window whose last draw found nothing in reach (T1 NONE): redrawn every replan_ticks
        # (an env never spawned has T1 NONE too, and is not holding)
        self.holding = np.zeros(n, bool)
        # --ramp-sequence: a PREDEFINED target list replaces the planner (set_sequence) - seq_k is
        # T1's index in it, seq_done is set once its LAST target is entered
        self.seq = None
        self.seq_k = np.zeros(n, np.int64)
        self.seq_done = np.zeros(n, bool)
        self.source = np.full(n, NONE, np.int64)       # the surface an env spawned on
        self.n_capt = np.zeros(n, np.int64)            # takeoffs (= completed rides) this episode
        self.n_skip = np.zeros(n, np.int64)
        # window shifts on the CURRENT tick per env (on_tick clears it): --ramp-reward pass pays
        # +1 for each
        self.tick_pass = np.zeros(n, np.int64)
        # --ramp-exit-bonus: this tick's PASS exits per env as the height the exit state could
        # climb to (z + |v|^2 / 2g) above the left piece's lowest validated contact, >= 0
        self.tick_exit_h = np.zeros(n, np.float64)
        self.p_zlow = {q: float(min(float(self.tp[f][:, 2].min()) for f in fs))
                       for q, fs in self.pfaces.items()}
        # window shifts since the policy last read them (take_passes, once per decision):
        # --ramp-obs-pass shows the policy the event --ramp-reward pass pays
        self.pass_acc = np.zeros(n, np.int64)
        # per env the pieces this episode has left behind (and the one it spawned on): never a
        # target again - no cycles (Codex: excluding only the last piece let A -> B -> C -> A)
        self.visited = [set() for _ in range(n)]
        self.stats = {"episodes": 0, "rides": 0, "skips": 0, "fin": 0, "holds": 0,
                      "seq_done": 0, "ride_hist": {}}
        # the compiled window (rampfast.window): per target its normals in _ids order; per
        # compact piece its targets (their indices, in pfaces order) and its lowest geodesic
        # part; the goal field's grid for the one-point sampler; a raw-point buffer
        self._fast_win = self._fast is not None and bool(len(self._ids))
        if self._fast_win:
            self._N = np.ascontiguousarray(np.concatenate([self.tn[int(k)] for k in self._ids]),
                                           np.float64)
            pfj, pfs, pfc = [], [], []
            for q in self._pfull:
                fs = self.pfaces[int(q)]
                pfs.append(len(pfj))
                pfc.append(len(fs))
                pfj.extend(self._jof[f] for f in fs)
            self._pf_j = np.asarray(pfj, np.int64)
            self._pf_start = np.asarray(pfs, np.int64)
            self._pf_count = np.asarray(pfc, np.int64)
            self._plow = np.asarray([self.p_low.get(int(q), np.inf) for q in self._pfull],
                                    np.float64)
            self._phas = np.asarray([self.p_axis[int(q)] is not None for q in self._pfull],
                                    np.bool_)
            self._pax = np.asarray([self.p_axis[int(q)][0] if self.p_axis[int(q)] is not None
                                    else 0.0 for q in self._pfull], np.float64)
            self._pay = np.asarray([self.p_axis[int(q)][1] if self.p_axis[int(q)] is not None
                                    else 0.0 for q in self._pfull], np.float64)
            if self.gf is not None and not all(hasattr(self.gf, a) for a in
                                               ("grid", "mins", "cell", "_valid_max",
                                                "sentinel")):
                self._fast_win = False     # a field without a voxel grid: the Python path
            elif self.gf is not None:
                self._gfa = (np.ascontiguousarray(self.gf.grid, np.float32),
                             np.asarray(self.gf.mins, np.float64), float(self.gf.cell),
                             np.float32(self.gf._valid_max), np.float32(self.gf.sentinel),
                             float(self.gf.reach_max))
            else:
                self._gfa = (np.zeros((1, 1, 1), np.float32), np.zeros(3), 1.0,
                             np.float32(0.0), np.float32(0.0), 0.0)
            self._buf = np.empty((100_000, 3), np.float64)

    # ------------------------------------------------------------------ pieces
    def _pc(self, s):
        """surface id(s) -> piece id(s); FIN / NONE (negative) stay as they are"""
        s = np.asarray(s, np.int64)
        return np.where(s >= 0, self.pmap[np.clip(s, 0, len(self.pmap) - 1)], s)

    def _piece(self, s):
        """one surface -> its piece (None: not a target surface, or FIN / NONE)"""
        return self._pof.get(int(s)) if s is not None and int(s) >= 0 else None

    def in_box(self, t, origin):
        """(N,) target surface ids (FIN / NONE: never inside), (N, 3) positions -> (N,) bool: the
        position lies in the target's PIECE box"""
        t = np.asarray(t, np.int64).reshape(-1)
        o = np.asarray(origin, np.float64).reshape(-1, 3)
        p = np.where(t >= 0, self._cp_of[np.clip(t, 0, len(self._cp_of) - 1)], -1)
        ok = p >= 0
        p = np.maximum(p, 0)
        d = o[:, :2] - self._bc[p]
        a = (d * self._bu[p]).sum(1)
        b = (d * self._bv[p]).sum(1)
        L = self._blim[p]
        return (ok & (a >= L[:, 0]) & (a <= L[:, 1]) & (b >= L[:, 2]) & (b <= L[:, 3])
                & (o[:, 2] >= L[:, 4]) & (o[:, 2] <= L[:, 5]))

    def set_sequence(self, seq):
        """--ramp-sequence: a PREDEFINED target list (surface ids, in order) replaces the planner.
        Every episode starts at seq[0]; T1 / T2 are seq[k] / seq[k + 1]; a pass (or a skip into
        T2) advances k; entering the LAST target completes the sequence (seq_done); past the end
        the window holds with no redraw. A surface may appear twice (a ramp left and landed on
        again)"""
        seq = [int(s) for s in seq]
        if not seq:
            raise ValueError("--ramp-sequence: an empty target list")
        bad = [s for s in seq if s not in self.tp]
        if bad:
            raise ValueError(f"--ramp-sequence: {bad} are not target surfaces of this vocabulary")
        self.seq = seq
        self.seq_k[:] = 0
        self.seq_done[:] = False

    def seq_stage(self, i):
        """--ramp-sequence: how far env i got - targets passed, +1 once the last is entered"""
        return int(self.seq_k[i]) + int(self.seq_done[i] and self.seq_k[i] == len(self.seq) - 1)

    def offtarget(self, counts, normals, origin, live=None):
        """(N,) bool: the env touches a RAMP-like plane (contact normal z in RAMP_NZ) outside the
        boxes of the pieces it may ride now - T1, T2, and the piece it spawned on until its first
        pass - i.e. it surfs a ramp the planner did not ask for; back on a piece it already
        passed is off-target too (the user: "surfing on one thing, then going back. This is
        bad"). counts / normals as SurfCore.get_touch returns them."""
        counts = np.asarray(counts)
        nz = np.asarray(normals)[:, :, 2]
        valid = np.arange(nz.shape[1])[None, :] < counts[:, None]
        ramp = (valid & (nz > RAMP_NZ[0]) & (nz < RAMP_NZ[1])).any(1)
        if live is not None:
            ramp &= np.asarray(live, bool)
        rows = np.flatnonzero(ramp)
        if not len(rows):
            return ramp
        o = np.asarray(origin, np.float64).reshape(-1, 3)[rows]
        src = np.where(self.prev[rows] == NONE, self.source[rows], NONE)  # before a first pass
        ok = (self.in_box(self.t1[rows], o) | self.in_box(self.t2[rows], o)
              | self.in_box(src, o))
        ramp[rows[ok]] = False
        return ramp

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

    def _eligible(self, s, d_min, d_max):
        return not self.d_surf or d_min <= self.d_surf.get(int(s), np.inf) < d_max

    def _candidates(self, p, v, exclude, d_max=np.inf, d_min=-np.inf):
        """ELIGIBLE targets (geodesic distance in [d_min, d_max); pieces in `exclude` never) by
        the closest approach of the ballistic arc from (p, v) within the horizon to their PIECE,
        each piece as its RIDE surface (_ride_face): -> [(dist, surface id)] sorted; the finish
        box is a candidate like a target when the band reaches it (d_min <= 0)"""
        path = self._arc(p, v, self.horizon)
        need = self.topk + 1
        best = {}                                  # piece -> closest approach of its surfaces
        dist = {}                                  # surface -> closest approach (the ones seen)
        kth = np.inf                               # the need-th best piece so far
        if len(self._ids) and self._fast is not None:
            ex = np.zeros(len(self._pfull), np.bool_)
            for q in exclude:
                if q in self._pcompact:
                    ex[self._pcompact[q]] = True
            pb_, fd_ = self._fast[1](np.ascontiguousarray(path, np.float64), self._cen, self._rad,
                                     self._O, self._ostart, self._ocount, self._pj,
                                     len(self._pfull), self._dsurf, bool(self.d_surf),
                                     float(d_min), float(d_max), ex, int(need))
            for jc in np.flatnonzero(pb_ < np.inf):
                best[int(self._pfull[jc])] = float(pb_[jc])
            for f in np.flatnonzero(fd_ >= 0.0):
                dist[int(self._ids[f])] = float(fd_[f])
        elif len(self._ids):
            lb = np.maximum(0.0, np.linalg.norm(path[:, None, :] - self._cen[None], axis=2).min(0)
                            - self._rad)
            for j in np.argsort(lb):
                if lb[j] > kth:
                    break                          # no later surface can beat the k-th piece
                s = int(self._ids[j])
                q = self._pof[s]
                if q in exclude or not self._eligible(s, d_min, d_max):
                    continue
                dq, _iq = self.tt[s].query(path, k=1)
                dist[s] = float(dq.min())
                best[q] = min(best.get(q, np.inf), dist[s])
                if len(best) >= need:
                    kth = sorted(best.values())[need - 1]
        vh = np.array([float(v[0]), float(v[1]), 0.0])
        vh = vh / max(float(np.linalg.norm(vh)), 1e-9)
        out = [(dq_, self._ride_face(q, path, vh, d_min, d_max, dist))
               for q, dq_ in sorted(best.items(), key=lambda x: x[1])[:need]]
        # the finish box: the arc's closest approach to the box (0 inside it) - a candidate
        # only within the reach cap (its geodesic distance is 0: eligible when d_min <= 0; it
        # was always a candidate and training drew it as T1 from 10.4% of real states, a median
        # 90k u of geodesic away, Codex 2026-09-28)
        if d_min <= 0.0:
            c = np.clip(path, self.fin_lo[None], self.fin_hi[None])
            out.append((float(np.linalg.norm(path - c, axis=1).min()), FIN))
        return sorted(out)

    def _ride_face(self, q, path, vh, d_min, d_max, dist=None):
        """the surface of piece q the flight rides: the longest run along the piece's main AXIS
        (its largest surface's level line; a floor piece: the travel direction), less the arc's
        distance to it - a wedge's side, never its end cap, whatever the heading (along the
        travel direction a cap won from 38 of 360 headings, Codex 2026-09-28); of two mirror
        sides, the nearer. dist: closest approaches already measured (surface -> u)"""
        ax = self.p_axis.get(q)
        axis = np.asarray(ax if ax is not None else vh[:2], np.float64)
        best, sc = None, -np.inf
        for s in self.pfaces[q]:
            d_s = dist.get(s) if dist is not None else None
            if d_s is None:
                d_s = float(self._nearest(s, path)[0])
            pr = self.tp[s][:, 0] * axis[0] + self.tp[s][:, 1] * axis[1]
            x = float(pr.max() - pr.min()) - d_s
            if x > sc:
                best, sc = s, x
        return int(best)

    def _nearest(self, k, pts):
        """(distance, point index, origin index) of the closest (pts, target k's origins) pair"""
        pts = np.ascontiguousarray(pts, np.float64)
        if self._fast is not None:
            j = self._jof[int(k)]
            return self._fast[0](pts, self._O, self._ostart[j], self._ocount[j])
        dq, iq = self.tt[int(k)].query(pts, k=1)
        jj = int(np.argmin(dq))
        return float(dq[jj]), jj, int(iq[jj])

    def _steer(self, p, v):
        """the velocity a window is laid with: the state's own at a horizontal speed of RAY_FLOOR
        or more; below it RAY_FLOOR along a horizontal direction that BLENDS the state's own into
        the geodesic field's descent direction, weight |v_h| / RAY_FLOOR on its own (all descent
        at a standstill), its own vertical kept - continuous in the state (a hard switch at
        RAY_FLOOR turned two B7 shifts by 45 and 61 deg for a few u/s, Codex 2026-09-28). From
        the state's own alone the ride along a target had no direction and ran back toward the
        start: at a standing spawn (utopia), and at the first capture (Codex 2026-09-28)"""
        v = np.asarray(v, np.float64).reshape(3)
        h = float(np.hypot(v[0], v[1]))
        if self.gf is None or h >= RAY_FLOOR:
            return v
        yaw = np.radians(float(self.gf.descent_yaw(np.asarray(p, np.float64)[None])[0]))
        dd = np.array([np.cos(yaw), np.sin(yaw)])
        w = h / RAY_FLOOR
        m = w * (v[:2] / h if h > 1e-9 else dd) + (1.0 - w) * dd
        nm = float(np.linalg.norm(m))
        m = m / nm if nm > 1e-9 else dd            # its own exactly against descent at w = 1/2
        return np.array([m[0] * RAY_FLOOR, m[1] * RAY_FLOOR, v[2]])

    def _pick(self, cands):
        if not cands:
            return NONE                            # nothing in reach: the window HOLDS
        if self.deterministic:
            return int(cands[0][1])
        k = min(self.topk, len(cands))
        return int(cands[int(self.rng.integers(k))][1])

    def _next_fn(self, exclude):
        """choose_next for window_line: the target closest to the arc launched at the end of the
        previous ride (the previous target's piece and the pieces in `exclude` never)"""
        def f(cp, cv, prev):
            cp = np.asarray(cp, np.float64)
            cv = np.asarray(cv, np.float64)
            q = self._piece(prev)
            d_max, d_min = np.inf, -np.inf
            if self.d_surf:
                # BEYOND the previous piece (below its lowest part: its facing side and its end
                # caps are not a next target) and below the launch point; within the reach cap
                d_ref = min(self.p_low.get(q, np.inf) if q is not None else np.inf,
                            self._d_at(cp))
                if np.isfinite(d_ref):
                    d_max = d_ref - PROGRESS_DELTA
                    d_min = d_ref - self._reach(cv)
            c = self._candidates(cp, cv, exclude | ({q} if q is not None else set()),
                                 d_max=d_max, d_min=d_min)
            return int(c[0][1]) if c else NONE     # nothing in reach: no second target
        return f

    def _draw(self, p, va, q, exclude):
        """T1 for a state (p, its steered velocity va): the pick among the targets in the band
        below the geodesic distance here - and below piece q's lowest part (the piece it spawned
        on or last left) - within the reach cap; the pieces in `exclude` never. NONE when the band
        holds none (the finish is in it only within the cap). Off the field: no band"""
        d0 = self._d_at(p)
        if q is not None and q in self.p_low:
            d0 = min(d0, self.p_low[q])            # beyond the piece it spawned on / left
        if np.isfinite(d0):
            return self._pick(self._candidates(p, va, exclude, d_max=d0 - PROGRESS_DELTA,
                                               d_min=d0 - self._reach(va)))
        return self._pick(self._candidates(p, va, exclude))

    def _hold_line(self, p, v):
        """a HOLDING window's line (no target in reach): RIDE_PAST of lookahead from here along
        the steered velocity - what a window's line runs past its last ride"""
        from .route import resample_polyline
        v = self._steer(p, v)
        d = v / max(float(np.linalg.norm(v)), 1e-9)
        p = np.asarray(p, np.float64).reshape(3)
        pts = np.vstack([p[None], p[None] + d[None] * np.linspace(RIDE_PAST / 8, RIDE_PAST,
                                                                  8)[:, None]])
        line, _total = resample_polyline(pts, RAY_SPACING)
        return np.asarray(line, np.float32)

    def _redraw(self, i, p, v):
        """a HOLDING window (T1 NONE): a fresh draw from here -> its line, or None (still nothing
        in reach). The surface left last keeps fading from the value it shows now"""
        va = self._steer(p, v)
        ex = set(self.visited[i])
        q = self._piece(self.prev[i] if self.prev[i] != NONE else self.source[i])
        t1 = self._draw(p, va, q, ex)
        if t1 == NONE:
            return None
        f = min(1.0, float(self.tau[i]) / self.fade_ticks)
        self.v0p[i] = self.v0p[i] * (1.0 - f) if self.prev[i] != NONE else 0.0
        self.v0t[i] = 0.0                          # T1 showed nothing while holding
        self.t1[i] = t1
        self.holding[i] = False
        ln, t2 = self._window(p, va, t1, ex)
        self.t2[i] = t2
        self.tau[i] = 0
        self.entered[i] = False
        return ln

    def _fast_window(self, p, v, k0, k1, riding, exclude):
        """rampfast.window - the line and the second target in one compiled pass. k1: None =
        choose it off k0's ride (beyond k0's piece and `exclude`), NONE = no second, else fixed.
        -> (line, second target id / FIN / NONE), or None (no compiled path, or its buffer is
        full: the Python path draws this window)"""
        if not self._fast_win:
            return None
        ex = np.zeros(len(self._pfull), np.bool_)
        for q in exclude:
            if q in self._pcompact:
                ex[self._pcompact[q]] = True
        j0 = _rf.FIN if k0 == FIN else self._jof[int(k0)]
        if k1 is None:
            j1 = _rf.CHOOSE
        elif k1 == NONE:
            j1 = _rf.NONE
        elif k1 == FIN:
            j1 = _rf.FIN
        else:
            j1 = self._jof[int(k1)]
        grid, mins, cell, vmax, sent, rmax = self._gfa
        p = np.asarray(p, np.float64).reshape(3)
        v = np.asarray(v, np.float64).reshape(3)
        try:
            ln, t2 = self._fast[4](
                float(p[0]), float(p[1]), float(p[2]), float(v[0]), float(v[1]), float(v[2]),
                int(j0), int(j1), bool(riding), ex, self._cen, self._rad, self._O, self._N,
                self._ostart, self._ocount, self._pj, len(self._pfull), self._dsurf,
                bool(self.d_surf), self._plow, self._pf_start, self._pf_count, self._pf_j,
                self._pax, self._pay, self._phas,
                int(self.topk + 1), grid, mins, cell, vmax, sent, rmax, self.fin_lo,
                self.fin_hi, np.asarray(self.finish, np.float64), float(self.gravity),
                float(self.horizon), float(PROGRESS_DELTA), float(SPEED_MARGIN),
                float(self.dt_path), float(RAMP_COAST), float(RAMP_PRESS), float(RIDE_PAST),
                float(RAY_FLOOR), float(RAY_SPACING), int(self.line_cap), self._buf)
        except ValueError:
            return None
        t2 = int(t2)
        return ln, (t2 if t2 < 0 else int(self._ids[t2]))

    def _window(self, p, v, t1, exclude, riding=False):
        """ONE pass: the line through T1 and a T2 chosen where T1's ride ends -> (line, t2);
        T1 NONE (nothing in reach) -> (the holding line, NONE)"""
        if t1 == NONE:
            return self._hold_line(p, v), NONE
        v = self._steer(p, v)
        r = self._fast_window(p, v, t1, NONE if t1 == FIN else None, riding, exclude)
        if r is not None:
            return r
        if t1 == FIN:
            ln, _raw = window_line(self.tp, self.tn, self.tt, self.finish, p, v, [FIN], fin=FIN,
                                   gravity=self.gravity, dt_path=self.dt_path,
                                   line_cap=self.line_cap, nearest=self._nearest)
            return ln, NONE
        got = []
        ln, _raw = window_line(self.tp, self.tn, self.tt, self.finish, p, v, [t1, None], fin=FIN,
                               riding_first=riding, gravity=self.gravity, dt_path=self.dt_path,
                               line_cap=self.line_cap, choose_next=self._next_fn(exclude),
                               chosen=got, nearest=self._nearest)
        return ln, (got[1] if len(got) > 1 else NONE)

    def _line(self, i, p, v, riding):
        v = self._steer(p, v)
        r = self._fast_window(p, v, int(self.t1[i]), int(self.t2[i]), riding, set())
        if r is not None:
            return r[0]
        ks = [int(self.t1[i])] + ([int(self.t2[i])] if self.t2[i] != NONE else [])
        ln, _raw = window_line(self.tp, self.tn, self.tt, self.finish, p, v, ks, fin=FIN,
                               riding_first=riding, gravity=self.gravity, dt_path=self.dt_path,
                               line_cap=self.line_cap, nearest=self._nearest)
        return ln

    # ------------------------------------------------------------------ spawn
    def spawn(self, idx, origin, velocity, source=None):
        """fresh episodes for envs idx: draw T1 / T2 from their spawn states -> list of lines.
        `source` (len(idx),) = the surface each spawn is on (its piece is never drawn as T1),
        NONE if none"""
        idx = np.asarray(idx, np.int64).reshape(-1)
        origin = np.asarray(origin, np.float64).reshape(-1, 3)
        velocity = np.asarray(velocity, np.float64).reshape(-1, 3)
        src = (np.full(len(idx), NONE) if source is None
               else np.asarray(source, np.int64).reshape(-1))
        lines = []
        for n, i in enumerate(idx):
            p, v = origin[n], velocity[n]
            q = self._piece(src[n])
            ex = {q} if q is not None else set()
            # a slow spawn (standing on the start, say) would draw its arc as a vertical drop:
            # lay it along the geodesic field's descent direction at RAY_FLOOR instead
            va = self._steer(p, v)
            if self.seq is not None:
                # --ramp-sequence: the list's first two targets, never a draw
                t1 = self.seq[0]
                t2 = self.seq[1] if len(self.seq) > 1 else NONE
                self.seq_k[i] = 0
                self.seq_done[i] = False
                self.holding[i] = False
                self.visited[i] = set(ex)
                self.t1[i] = t1
                self.t2[i] = t2
                ln = self._line(int(i), p, va, False)
            else:
                t1 = self._draw(p, va, q, ex)      # NONE: nothing in reach - the window holds
                self.stats["holds"] += int(t1 == NONE)
                self.holding[i] = t1 == NONE
                self.visited[i] = set(ex)
                # the window along the SAME arc the draw used: from a standing spawn's own zero
                # velocity the arrival falls straight down T1's slope and its ride had no
                # direction (utopia's start: the line rode S26 back toward the start)
                ln, t2 = self._window(p, va, t1, ex)
            self.t1[i] = t1
            self.t2[i] = t2
            self.prev[i] = NONE
            self.tau[i] = 1 << 30
            self.v0p[i] = 1.0
            self.v0t[i] = 0.5
            self.entered[i] = False
            self.source[i] = src[n]
            self.n_capt[i] = 0
            self.n_skip[i] = 0
            self.pass_acc[i] = 0                   # a new episode starts with nothing passed
            lines.append(ln)
        return lines

    # ------------------------------------------------------------------ tick
    def on_tick(self, ids, origin, velocity, ended):
        """one physics tick from the post-step positions (ids: unused, the boxes need none);
        ended (N,) bool (those rows are settled and skipped - they respawn through spawn()).
        -> (idx of envs whose line changed, their lines)"""
        origin = np.asarray(origin, np.float64)
        ended = np.asarray(ended, bool)
        self.tau += 1
        self.tick_pass[:] = 0
        self.tick_exit_h[:] = 0.0
        live = ~ended
        changed = {}
        # a HOLDING window (nothing was in reach): a fresh draw every replan_ticks
        for i in np.flatnonzero(self.holding & live & (self.tau % self.replan_ticks == 0)):
            ln = self._redraw(int(i), origin[i], velocity[i])
            if ln is not None:
                changed[int(i)] = ln
        in1 = self.in_box(self.t1, origin) & live
        in2 = self.in_box(self.t2, origin) & live
        # inside T2's box before T1 was ever entered: T1 is SKIPPED, T2 (entered now) is T1
        for i in np.flatnonzero(in2 & ~in1 & ~self.entered):
            self.n_skip[i] += 1
            self.stats["skips"] += 1
            changed[int(i)] = self._shift(i, origin[i], velocity[i], riding=True)
        in1 = self.in_box(self.t1, origin) & live
        # inside T1's box: T1 ENTERED - the line becomes the ride along T1 then T2 from here
        for i in np.flatnonzero(in1 & ~self.entered):
            self.entered[i] = True
            if self.seq is not None and self.seq_k[i] == len(self.seq) - 1:
                self.seq_done[i] = True            # --ramp-sequence: the LAST target entered
            if int(i) not in changed:
                changed[int(i)] = None             # the ride line, built below
        # out of T1's box after entering it: T1 PASSED - the window shifts
        for i in np.flatnonzero(self.entered & ~in1 & live):
            self.n_capt[i] += 1
            self.stats["rides"] += 1
            # --ramp-exit-bonus: the exit's energy, as the height it could climb to above the
            # lowest point of the ramp it leaves
            q = self._piece(self.t1[i])
            if q is not None:
                vv = np.asarray(velocity[i], np.float64)
                h = (float(origin[i][2]) + float(vv @ vv) / (2.0 * self.gravity)
                     - self.p_zlow[q])
                self.tick_exit_h[i] += max(0.0, h)
                self.stats["exit_h"] = self.stats.get("exit_h", 0.0) + max(0.0, h)
            changed[int(i)] = self._shift(i, origin[i], velocity[i], riding=False)
        idx = np.asarray(sorted(changed), np.int64)
        lines = [changed[int(i)] if changed[int(i)] is not None
                 else self._line(int(i), origin[i], velocity[i], True) for i in idx]
        return idx, lines

    def _shift(self, i, p, v, riding):
        """T1 done (left, or skipped): T2 becomes T1, a new T2 chosen where its ride ends - one
        pass that also lays the new line -> the line. The fades start from the values the two
        surviving surfaces show now (continuous under a second shift inside a fade)"""
        f = min(1.0, float(self.tau[i]) / self.fade_ticks)
        v_t1 = self.v0t[i] + (1.0 - self.v0t[i]) * f if self.t1[i] != NONE else 0.0
        v_t2 = 0.5 * f if self.t2[i] != NONE else 0.0
        self.prev[i] = self.t1[i]
        q = self._piece(self.prev[i])
        if q is not None:
            self.visited[i].add(q)
        if self.seq is not None:
            # --ramp-sequence: the next targets in the list, never a draw; past its end the
            # window holds (no redraw). A skip lands INSIDE the new T1: the last one entered so
            # completes the sequence
            k = int(self.seq_k[i]) + 1
            self.seq_k[i] = min(k, len(self.seq))
            if k < len(self.seq):
                self.t1[i] = self.seq[k]
                self.t2[i] = self.seq[k + 1] if k + 1 < len(self.seq) else NONE
                if riding and k == len(self.seq) - 1:
                    self.seq_done[i] = True
                ln = self._line(int(i), p, v, bool(riding))
            else:
                self.t1[i] = NONE
                self.t2[i] = NONE
                ln = self._hold_line(p, v)
            t2 = self.t2[i]
        elif self.t2[i] != NONE:
            self.t1[i] = self.t2[i]
            ln, t2 = self._window(p, v, int(self.t1[i]), set(self.visited[i]),
                                  riding=bool(riding))
        else:
            # nothing was in reach past T1's ride: a fresh draw from here - or the window HOLDS
            # (T1 NONE, the line runs on ahead, a redraw every replan_ticks); never an
            # unreachable finish (Codex 2026-09-28)
            va = self._steer(p, v)
            self.t1[i] = self._draw(p, va, q, set(self.visited[i]))
            self.stats["holds"] += int(self.t1[i] == NONE)
            self.holding[i] = self.t1[i] == NONE
            ln, t2 = self._window(p, va, int(self.t1[i]), set(self.visited[i]))
        self.t2[i] = t2
        self.tau[i] = 0
        self.v0p[i] = v_t1
        self.v0t[i] = v_t2
        self.entered[i] = bool(riding)
        self.tick_pass[i] += 1
        self.pass_acc[i] += 1
        return ln

    def take_passes(self, idx=None):
        """--ramp-obs-pass: (n,) float32, 1 where the window shifted (T1 passed, or skipped for
        T2) since the last call, else 0 - and the count restarts. Read once per decision, it is
        the policy's view of the event --ramp-reward pass pays"""
        sl = slice(None) if idx is None else np.asarray(idx, np.int64)
        out = (self.pass_acc[sl] > 0).astype(np.float32)
        self.pass_acc[sl] = 0
        return out

    def pass_flags(self, idx):
        """take_passes' value for envs idx WITHOUT restarting the count (the truncation
        bootstrap reads the flag its terminal state would have shown)"""
        return (self.pass_acc[np.asarray(idx, np.int64)] > 0).astype(np.float32)

    def settle(self, idx, finished):
        """episodes of envs idx ended (finished: reached the finish box) - the stats"""
        for i, f in zip(np.asarray(idx, np.int64).reshape(-1),
                        np.asarray(finished, bool).reshape(-1)):
            self.stats["episodes"] += 1
            self.stats["fin"] += int(f)
            self.stats["seq_done"] += int(self.seq is not None and self.seq_done[i])
            r = int(self.n_capt[i])
            self.stats["ride_hist"][r] = self.stats["ride_hist"].get(r, 0) + 1

    def pop_stats(self) -> dict:
        s = self.stats
        self.stats = {"episodes": 0, "rides": 0, "skips": 0, "fin": 0, "holds": 0,
                      "seq_done": 0, "ride_hist": {}}
        return s

    # ------------------------------------------------------------------ channel
    def slots(self, idx=None):
        """(ids (n, 3), values (n, 3)): [the target just left, T1, T2] with the continuous takeoff
        cross-fade (1 -> 0, 0.5 -> 1, 0 -> 0.5 over the fade, each from the value its surface
        showed at the shift)"""
        sl = slice(None) if idx is None else np.asarray(idx, np.int64)
        f = np.minimum(1.0, self.tau[sl] / self.fade_ticks)
        ids = np.stack([self.prev[sl], self.t1[sl], self.t2[sl]], axis=1)
        v0p, v0t = self.v0p[sl], self.v0t[sl]
        vals = np.stack([np.where(self.prev[sl] != NONE, v0p * (1.0 - f), 0.0),
                         np.where(self.t1[sl] != NONE, v0t + (1.0 - v0t) * f, 0.0),
                         np.where(self.t2[sl] != NONE, 0.5 * f, 0.0)], axis=1)
        vals[ids == NONE] = 0.0
        return ids, vals.astype(np.float32)

    def describe(self) -> str:
        return (f"ramp windows: {self.N} envs, {len(self.tp)} target surfaces in "
                f"{len(self.pfaces)} pieces, targets "
                + ("ORDERED by the geodesic potential (eligible = closer to the finish by "
                   f"{PROGRESS_DELTA:g} u than the agent / the previous piece), "
                   if self.d_surf else "NOT ordered (no goal field), ")
                + f"T1 drawn {'closest' if self.deterministic else f'among the top {self.topk}'} "
                f"pieces by closest approach of the {self.horizon:g} s ballistic arc, each as "
                f"its surface furthest along the piece's axis; T2 off T1's ride; T1 entered "
                f"inside its piece's box, passed on leaving it (boxes {BOX_MARGIN:g} u past the "
                f"validated contacts); channel fade {self.fade_ticks * self.tick_ms / 1000.0:g} s")


class RampPlanner:
    """--goal-planner ramps: the GoalSystem's planner for the ramp-window task - the vocabulary,
    the training fleet's windows (random among the top-k) and the eval's (one env, the closest
    candidate every time)."""

    ramps = True
    primitive = False

    def __init__(self, vocab, n_envs: int, finish_box, tick_ms: float, *, topk: int, horizon: float,
                 fade: float, gravity: float = 800.0, line_cap: int = 768, seed: int = 0,
                 goal_field=None, sequence=None):
        self.vocab = vocab
        kw = dict(topk=topk, horizon=horizon, fade=fade, gravity=gravity, line_cap=line_cap,
                  goal_field=goal_field)
        self.windows = RampWindows(vocab, n_envs, finish_box, tick_ms,
                                   rng=np.random.default_rng(int(seed)), **kw)
        self.eval_windows = RampWindows(vocab, 1, finish_box, tick_ms, deterministic=True, **kw)
        # --ramp-sequence: the predefined target list, on both
        self.sequence = None if sequence is None else [int(s) for s in sequence]
        if self.sequence is not None:
            self.windows.set_sequence(self.sequence)
            self.eval_windows.set_sequence(self.sequence)
        self.finish_center = self.windows.finish
        self.fin = None

    def describe(self) -> str:
        s = self.vocab.describe() + "\n" + self.windows.describe()
        if self.sequence is not None:
            s += (f"\n--ramp-sequence: a PREDEFINED target list {self.sequence} replaces the "
                  "planner (T1/T2 = the next two in it; entering the last one ends the episode "
                  "as a success)")
        return s


class _ViewCam:
    """one extra direction of the target channel (--target-views 6): the lidar's equiangular grid
    - its H x W and its vertical span - with its own horizontal span. It carries what
    TargetMask.render reads off a lidar: H, W, yoff, poff, device, pinhole, and for the torch path
    the ray buffers and directions (the lidar's own methods, run on this grid)"""

    pinhole = False

    def __init__(self, lidar, hfov_deg=None):
        import torch
        self._lidar_cls = type(lidar)
        self.H, self.W, self.device = int(lidar.H), int(lidar.W), lidar.device
        self.poff = lidar.poff
        if hfov_deg is None:
            self.yoff = lidar.yoff
        else:
            yoff = (float(hfov_deg) * (0.5 - (np.arange(self.W) + 0.5) / self.W)) * np.pi / 180.0
            self.yoff = torch.as_tensor(yoff, dtype=torch.float32, device=self.device)
        self._buf_n = None

    def _ensure_buffers(self, N):
        import torch
        if self._buf_n == N and not (getattr(self, "_dx", None) is not None
                                     and self._dx.is_inference()
                                     and not torch.is_inference_mode_enabled()):
            return
        self._buf_n = N
        sh = (N, self.H, self.W)
        self._dx = torch.empty(sh, device=self.device)
        self._dy = torch.empty(sh, device=self.device)
        self._dz = torch.empty(sh, device=self.device)

    def _dirs_equiangular(self, N, yaw_deg, pitch_deg, d2r):
        import torch
        p = pitch_deg.view(N, 1, 1) * d2r + self.poff.view(1, self.H, 1)
        y = yaw_deg.view(N, 1, 1) * d2r + self.yoff.view(1, 1, self.W)
        cp = torch.cos(p)
        torch.mul(cp, torch.cos(y), out=self._dx)
        torch.mul(cp, torch.sin(y), out=self._dy)
        self._dz.copy_(torch.sin(p).expand_as(self._dz))


class TargetLidar:
    """A lidar with the target channel appended (the GoalBallLidar pattern): channels = the
    wrapped lidar's + 1; render() draws the per-env window slots of `windows` (RampWindows) through
    `tm` (surfgym.targetmask.TargetMask, no occlusion, max-combine). mode "off" renders the same
    channel as zeros - the control arm with the identical architecture. views 6 (--target-views)
    adds the channel from five more directions (TARGET_VIEWS: up, down, back, left, right) after
    the view's own - targets only, the depth image stays the view's own."""

    def __init__(self, lidar, tm, windows=None, mode: str = "live", views: int = 1):
        if mode not in ("live", "off"):
            raise ValueError(f"TargetLidar: mode live|off, got {mode!r}")
        if getattr(lidar, "pinhole", False):
            raise ValueError("TargetLidar mirrors the equiangular camera; --pinhole is separate")
        if int(views) not in (1, 1 + len(TARGET_VIEWS)):
            raise ValueError(f"--target-views: 1 (the view's own) or {1 + len(TARGET_VIEWS)} "
                             f"(+ {', '.join(v[0] for v in TARGET_VIEWS)}), got {views}")
        self.lidar, self.tm, self.windows, self.mode = lidar, tm, windows, mode
        self.base = int(getattr(lidar, "channels", 1))
        self.views = int(views)
        self.channels = self.base + self.views
        # the extra directions' cameras: (camera, yaw offset, pitch) - the level ones reuse the
        # lidar's own grid, the two poles a 180 deg span
        self._cams = ([(_ViewCam(lidar, span), float(dyaw), float(pitch))
                       for _n, dyaw, pitch, span in TARGET_VIEWS] if self.views > 1 else [])
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
                f"after the lidar's {self.base} channel(s)"
                + (f", from {self.views} directions (the view's own, "
                   + ", ".join(v[0] for v in TARGET_VIEWS) + ")" if self.views > 1 else "")
                + f" -> in_ch {self.channels}")

    def render(self, origin, yaw_deg, pitch_deg, ducked, idx=None, **kw):
        import torch
        img = self.lidar.render(origin, yaw_deg, pitch_deg, ducked, **kw)
        if img.dim() == 3:
            img = img.unsqueeze(-1)
        n = origin.shape[0]
        if self.mode == "off" or self.windows is None:
            ch = torch.zeros((n, self.H, self.W, self.views), dtype=img.dtype, device=img.device)
        else:
            ids, vals = self.windows.slots(idx)
            if len(ids) != n:
                raise ValueError(f"TargetLidar: {n} poses but {len(ids)} windows (pass idx for "
                                 "a subset batch)")
            self.tm.set_slots(n, ids, vals, combine="max")
            chs = [self.tm.render(self.lidar, origin, yaw_deg, pitch_deg, ducked)]
            for cam, dyaw, pitch in self._cams:
                chs.append(self.tm.render(cam, origin, yaw_deg + dyaw,
                                          torch.full_like(pitch_deg, pitch, dtype=torch.float32),
                                          ducked))
            ch = torch.stack(chs, dim=-1).to(img.dtype)
        return torch.cat([img, ch], dim=-1)


def make_pass_feed(windows):
    """--ramp-obs-pass in an eval / a recording (the policy wrappers' pass_fn): env 0's flag off
    the 1-env RampWindows that make_ramp_hooks drives - 1 on the decision after its window
    shifted - and 0 for any other env of the core"""
    def feed(core, n=None):
        if n is None:
            n = int(np.asarray(core.states_view["origin"]).shape[0])
        out = np.zeros(int(n), np.float32)
        out[0] = float(windows.take_passes([0])[0])
        return out
    return feed


def make_ramp_hooks(windows, vocab, core, ev: dict, *, line=None):
    """(episode_meta, on_tick) for record_rollout on a core whose env 0 is recorded - the
    ramp-window eval, shared by the trainer and tools/record_ckpt.py. Each episode draws env 0's
    window from its spawn state (windows is a 1-env RampWindows, deterministic for the headline);
    every tick reads env 0's collision telemetry, advances the window and keeps `line` (the eval
    fan) on it. ev tallies episodes, rides, skips and finishes."""
    ev.update({"n": 0, "succ": 0, "rides": [], "skips": 0, "chain": [], "off": 0,
               "stages": [], "exit_h": []})
    # the window as the policy SAW it, for the POV render (tools/render_pov.py --targets): one
    # event per change, [row, prev, T1, T2, tau at that row, fade start of prev, fade start of
    # T1] - the channel's values at row k are the fade of tau + (k - row) (slot_values); rows are
    # the episode's own tick index
    rec = {"t0": None, "events": [], "seq_kill": False}

    def _snap(row):
        rec["events"] = _snap_event(windows, row, rec["events"])

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
        rec["seq_kill"] = False
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
            # --ramp-sequence: the completed sequence is the success (the hook ended it)
            fin = bool(np.asarray(core.goal_hits, bool)[0]) or bool(rec["seq_kill"])
            ev["succ"] += int(fin)
            if windows.seq is not None:
                ev["stages"].append(windows.seq_stage(0))
            ev["rides"].append(int(windows.n_capt[0]))
            ev["skips"] += int(windows.n_skip[0])
            windows.settle([0], [fin])
            return
        cnt, nrm, _pts = core.get_touch()
        sv = core.states_view
        org = sv["origin"][:1].astype(np.float64)
        ev["off"] += int(windows.offtarget(cnt[:1], nrm[:1], org)[0])
        idx, lines = windows.on_tick(None, org, sv["velocity"][:1].astype(np.float64),
                                     np.zeros(1, bool))
        if windows.tick_exit_h[0] > 0.0:
            ev["exit_h"].append(float(windows.tick_exit_h[0]))
        if len(idx):
            if ev["chain"]:
                ev["chain"][-1].append(int(windows.t1[0]))
            if line is not None:
                line.set_lines(idx, lines)
            # the policy sees the new window from the NEXT row on
            _snap(int(t) - int(rec["t0"]) + 1)
        if windows.seq is not None and windows.seq_done[0] and not rec["seq_kill"]:
            # --ramp-sequence: the last target entered - the episode ends as a SUCCESS
            m = np.zeros(core.num_envs, np.uint8)
            m[0] = 1
            core.force_fail(m)
            rec["seq_kill"] = True

    def episode_end(ep):
        out = {"events": list(rec["events"]), "fade_ticks": float(windows.fade_ticks)}
        if windows.seq is not None:
            out.update({"sequence": list(windows.seq), "seq_stage": windows.seq_stage(0),
                        "seq_done": bool(rec["seq_kill"])})
        return {"targets": out}

    episode_meta.episode_end = episode_end
    return episode_meta, on_tick


def replay_events(windows, vocab, rows):
    """A recording made WITHOUT the logged windows (a trainer started before make_ramp_hooks
    wrote them): re-run the deterministic eval windows (a 1-env RampWindows, deterministic=True)
    over its recorded states -> the same event list make_ramp_hooks writes. The windows move on
    positions (the boxes), so this is what the live hooks did - except for a recording made
    under the telemetry rules before them.
    rows: (n, >= 9) the recording's rows (origin 1..3, velocity 4..6, buttons 8)."""
    rows = np.asarray(rows, np.float64)
    n = len(rows)
    if n == 0:
        return []
    org = rows[:, 1:4]
    vel = rows[:, 4:7]
    duck = ((rows[:, 8].astype(np.int64) & 4) != 0).astype(np.int64)
    src = vocab.contact_of(org[:1], duck[:1])
    windows.spawn([0], org[:1], vel[:1], source=src)
    events = []

    def snap(row):
        _snap_event(windows, row, events)
    snap(0)
    for k in range(n - 1):
        # the post-step state of tick k is row k + 1
        idx, _ = windows.on_tick(None, org[k + 1:k + 2], vel[k + 1:k + 2], np.zeros(1, bool))
        if len(idx):
            snap(k + 1)
    return events


def _snap_event(windows, row, events):
    """append env 0's window as an event [row, prev, T1, T2, tau, v0p, v0t] when it changed"""
    e = [int(row), int(windows.prev[0]), int(windows.t1[0]), int(windows.t2[0]),
         int(min(windows.tau[0], 1 << 20)), round(float(windows.v0p[0]), 6),
         round(float(windows.v0t[0]), 6)]
    if not events or events[-1][1:4] != e[1:4]:
        events.append(e)
    return events


def slot_values(events, n_rows, fade_ticks):
    """a recorded episode's window events (make_ramp_hooks' trailer) -> per row the channel's
    three slots: (ids (n, 3) int64, values (n, 3) float32) - exactly RampWindows.slots. An event
    is [row, prev, T1, T2, tau] + [v0p, v0t] (the fade starts; recordings written before those
    were logged start every fade from a full one: 1 and 0.5)"""
    ids = np.full((n_rows, 3), NONE, np.int64)
    vals = np.zeros((n_rows, 3), np.float32)
    ev = sorted(events, key=lambda e: e[0])
    for j, e in enumerate(ev):
        row, prev, t1, t2, tau0 = e[:5]
        v0p, v0t = (float(e[5]), float(e[6])) if len(e) >= 7 else (1.0, 0.5)
        lo = max(0, int(row))
        hi = n_rows if j + 1 == len(ev) else min(n_rows, int(ev[j + 1][0]))
        if hi <= lo:
            continue
        k = np.arange(lo, hi)
        f = np.minimum(1.0, (float(tau0) + (k - lo)) / max(float(fade_ticks), 1.0))
        ids[lo:hi] = (prev, t1, t2)
        vals[lo:hi, 0] = v0p * (1.0 - f) if prev != NONE else 0.0
        vals[lo:hi, 1] = (v0t + (1.0 - v0t) * f) if t1 != NONE else 0.0
        vals[lo:hi, 2] = (0.5 * f) if t2 != NONE else 0.0
    return ids, vals
