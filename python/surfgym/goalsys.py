"""goalsys.py - per-env goal conditioning for train_fast (--goals).

Glue between the trainer loop and :mod:`surfgym.goals`: which goal each env
is chasing, the line it is shown, when it got there, and what that did to
the curriculum. The science (docs/research-plan-goalcond.md):

* the goal is a SPHERE mid-map; entering it ends the episode with the
  success bonus (the type-2/3 button-box logic, pass-through counts);
* goals come from the agent's OWN reached states - the respawn reservoir
  harvests, for every snapshot, the origin the same episode reached k
  ticks later plus the snapshot chain between (``RespawnBuffer(goal_k=)``)
  - so a goal is reachable by construction and k is its difficulty; a
  start that carries no such goal (fresh platform spawns, or by the
  ``air_frac`` coin) gets a random reachable-air goal instead;
* the policy sees the goal as the lookahead FAN on a per-env line: the
  reached-state segment (RDP-smoothed) for reservoir goals, the straight
  chord for air goals - which is the "arrow" encoding, and the fan on a
  line that ends at the goal subsumes the goal vector;
* k widens by a success-band rule (Florensa's 10-90% window) when
  ``curriculum`` is on.

Nothing here runs unless --goals is set; the control path is untouched.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .goals import (AirSampler, GoalStats, KCurriculum, MultiLine,
                    SphereGoals, chord_line, segment_line)

KIND = {0: "achieved", 1: "air", 2: "finish"}


class GoalSystem:
    def __init__(self, core, n_envs: int, line, goal_field, d0,
                 args, device, out_dir, seed: int = 0, ball=None,
                 eval_ball=None, arc=None, reward_fn=None):
        self.core = core
        # --goal-reward arc: the per-env arc coordinate (MultiArcProgress)
        # follows the same lines the fan shows; reward_fn is the
        # RaceReward whose arc diagnostics restart with each new line
        self.arc = arc
        self.reward_fn = reward_fn
        # observation channels: `line` (MultiLine, the fan) and/or `ball`
        # (GoalBallLidar, the depth channel); either may be None
        self.ball = ball
        self.eval_ball = eval_ball
        self.N = int(n_envs)
        self.line = line
        self.radius = float(args.goal_radius)
        self.sphere = SphereGoals(self.N, radius=self.radius)
        mins, maxs = core.map_bounds()
        self.mins, self.maxs = mins, maxs
        excl = None
        self.holdout = None
        if args.goal_holdout:
            lo, hi = (float(v) for v in str(args.goal_holdout).split(","))
            self.holdout = (lo * float(d0), hi * float(d0))

            def excl(p, _f=goal_field, _lo=self.holdout[0], _hi=self.holdout[1]):
                d = np.asarray(_f.sample(np.asarray(p, np.float64)), np.float64)
                return (d >= _lo) & (d <= _hi)
        self.air = AirSampler(mins, maxs, goal_field.reachable, exclude_fn=excl)
        self.air_frac = float(args.goal_air_frac)
        self.k_min = float(args.goal_kmin)
        self.curric = KCurriculum(k_min=self.k_min, k_max=float(args.goal_kmax),
                                  k_cap=float(args.goal_kcap))
        self.use_curric = bool(args.goal_curriculum)
        self.speed_est = 1500.0       # u/s, converts k seconds to an air radius
        # ballistic push: a visited state's recorded velocity, flown on
        # for tau seconds under the sim's own gravity, is a physically
        # plausible place to have been - and it lies BEYOND the visited
        # set, which the anchor-shell rule (goal within 1.5 radii of a
        # visited state) never does: measured on xsG2, the frontier crept
        # ~3,000 u/h on a 198,000 u map with the curriculum starved.
        self.gravity = float(getattr(core.config.phys, "sv_gravity", 800.0))
        self.tau_min = 0.5
        # ROUTE-DEPTH goals (user, 2026-09-02: goals next to the spawn
        # never make the agent surf): a goal placed on the map line
        # delta units of arc AHEAD of the start's projection, delta from
        # the curriculum (speed_est * k). On surfable geometry by
        # construction, the line shown is the route slice, and delta ->
        # end makes the finish box itself a goal. Experience-relative
        # generators can only hop past the visited set; this leaps.
        self.route = None
        self.route_frac = float(getattr(args, "goal_route_frac", 0.0) or 0.0)
        rp = getattr(args, "goal_route", None)
        if rp:
            z = np.load(rp)
            pts = np.asarray(z["route"], np.float64)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            self.route = pts
            self.route_s = np.concatenate(([0.0], np.cumsum(seg)))
            self.route_len = float(self.route_s[-1])
        self.r_min = 300.0
        self.stats = GoalStats()
        self.rng = np.random.default_rng(seed)
        self.k = np.zeros(self.N, np.float64)
        self.kind = np.zeros(self.N, np.int8)
        # start depth as a fraction of d0 (0 = spawn, 1 = finish), so
        # goal success can be reported per 10%% band of the MAP - the
        # number that says whether the agent trains beyond the spawn
        self.d0 = float(d0) if d0 else 1.0
        self.depth = np.zeros(self.N, np.float64)
        self.band_n = np.zeros(10, np.int64)
        self.band_ok = np.zeros(10, np.int64)
        self.pending = np.zeros(self.N, bool)
        self.pool = None
        self.pool_map: dict = {}
        self.n_assigned = np.zeros(3, np.int64)
        # eval side: one env, its own line and sphere
        self.eval_line = MultiLine(1, device=device) if line is not None else None
        self.ev = {"n": 0, "succ": 0, "pending": False, "center": None,
                   "ticks": [], "dists": []}
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._csv_path = out / "goals.csv"
        new = not self._csv_path.exists()
        self._csv = open(self._csv_path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._csv)
        if new:
            self._w.writerow(["step", "success", "succ_achieved", "succ_air",
                              "n", "n_achieved", "n_air", "k_max",
                              "ticks_to_goal", "eval_succ", "eval_n",
                              "succ_route", "n_route"]
                             + [f"route_succ_d{b}" for b in range(10)]
                             + [f"route_n_d{b}" for b in range(10)])
        self._last_eval = (float("nan"), 0)

    # ------------------------------------------------------------ describe
    def describe(self) -> str:
        ho = (f", holdout d in [{self.holdout[0]:,.0f}, {self.holdout[1]:,.0f}]u"
              if self.holdout else "")
        return (f"goals: sphere r={self.radius:g}u, k in [{self.k_min:g}, "
                f"{self.curric.k_max:g}]s (cap {self.curric.k_cap:g}, "
                f"curriculum {'on' if self.use_curric else 'off'}), "
                f"air share {self.air_frac:.0%}{ho}; shown as "
                + ("fan on the per-env line" if self.line is not None else "")
                + (" + " if (self.line is not None and self.ball is not None) else "")
                + ("depth-channel ball" if self.ball is not None else ""))

    # ------------------------------------------------------------ per-iter
    def set_pool(self, pool, goals, segs, seglen) -> None:
        """The spawn pool just uploaded to the core, with its goal columns.
        A fresh spawn is mapped back to its pool row by origin (the core
        copies the row verbatim; only velocity/pitch are perturbed)."""
        self.pool = (np.asarray(goals, np.float32), np.asarray(segs, np.float32),
                     np.asarray(seglen, np.int32))
        org = np.asarray(pool["origin"], np.float64)
        self.pool_map = {(round(float(o[0]), 1), round(float(o[1]), 1),
                          round(float(o[2]), 1)): j for j, o in enumerate(org)}
        # anchors for air goals: the pool's own rows (visited states, so
        # a goal within a short reach of one is physically grounded and
        # never sits at an unvisited ceiling), with their field depth so
        # anchors can be drawn flattened over the track (frontier pull)
        self.pool_org = org
        self.pool_vel = np.asarray(pool["velocity"], np.float64)
        self.pool_d = np.asarray(self._field_sample(org), np.float64)

    def iterate(self, respawn) -> None:
        if self.use_curric:
            self.curric.update()
        if respawn is not None and respawn.goal_k is not None:
            respawn.goal_k = (int(round(self.k_min * 100.0)),
                              int(round(self.curric.k_max * 100.0)))

    def _air_radius(self) -> float:
        r = self.speed_est * self.curric.k_max
        return float(np.clip(r, 1500.0, 60000.0))

    # air goals are ANCHORED on visited states (user, 2026-09-02: goals
    # only around the start, spheres at the very top of the map,
    # unreachable). Candidate anchors = pool rows within the current
    # reach band of THIS start; with probability frontier_p the draw is
    # restricted to anchors DEEPER than the start (lower field d) so the
    # goal distribution pulls toward the frontier as the reservoir
    # deepens; the goal is then a point within anchor_reach of the
    # anchor. No map constants: reach scales with k, anchor_reach with
    # the sphere radius.
    frontier_p = 0.7
    anchor_reach_mult = 1.5

    def _air_goal(self, origin):
        origin = np.asarray(origin, np.float64)
        porg = getattr(self, 'pool_org', None)
        if porg is not None and len(porg):
            dist = np.linalg.norm(porg - origin[None, :], axis=1)
            band = (dist >= self.r_min) & (dist <= self._air_radius())
            if band.any() and self.rng.random() < self.frontier_p:
                d_here = float(self._field_sample(origin[None, :])[0])
                deeper = band & (self.pool_d < d_here - self.radius)
                if deeper.any():
                    band = deeper
            if band.any():
                cand = np.flatnonzero(band)
                for _ in range(4):
                    j = int(cand[self.rng.integers(len(cand))])
                    a, va = porg[j], self.pool_vel[j]
                    # fly the anchor on: tau ~ U[tau_min, k_max], halving
                    # on a rejected point (solid / unreachable / held out)
                    tau = float(self.rng.uniform(self.tau_min,
                                                 max(self.tau_min,
                                                     self.curric.k_max)))
                    for _h in range(4):
                        p = a + va * tau
                        p = p.copy()
                        p[2] -= 0.5 * self.gravity * tau * tau
                        q = p[None, :]
                        okp = bool(self.air.reachable_fn(q)[0]) and bool(
                            np.all(q >= self.mins) and np.all(q <= self.maxs))
                        if okp and self.air.exclude_fn is not None:
                            okp = not bool(self.air.exclude_fn(q)[0])
                        if okp:
                            self._last_tau = tau
                            return p
                        tau *= 0.5
                    try:
                        self._last_tau = 0.0
                        return self.air.sample_near(
                            32, a, 0.0, self.anchor_reach_mult * self.radius,
                            self.rng)[0]
                    except RuntimeError:
                        continue
        try:
            return self.air.sample_near(64, origin, self.r_min,
                                        self._air_radius(), self.rng)[0]
        except RuntimeError:
            # rejection sampling over a mostly-solid AABB needs BATCHES:
            # n=1 x 64 tries crashed xsG2f on a deep route start
            return self.air.sample(512, self.rng)[0]

    def _route_goal(self, origin):
        """-> (goal xyz, line pts, k seconds) or None. The start projects
        to its nearest route vertex; delta ~ U[2.5 R, speed_est * k_max]
        of arc ahead, clamped to the line end (the finish). Held-out
        goals are skipped by pushing delta past the band."""
        d2 = ((self.route - origin[None, :]) ** 2).sum(1)
        i0 = int(d2.argmin())
        s0 = float(self.route_s[i0])
        if s0 >= self.route_len - self.radius:
            return None
        dmax = max(2.5 * self.radius + 1.0, self.speed_est * self.curric.k_max)
        for _ in range(4):
            delta = float(self.rng.uniform(2.5 * self.radius, dmax))
            st = min(s0 + delta, self.route_len)
            ig = int(np.searchsorted(self.route_s, st, side="right") - 1)
            ig = max(i0, min(ig, len(self.route) - 1))
            if ig + 1 < len(self.route) and self.route_s[ig + 1] > self.route_s[ig]:
                f = (st - self.route_s[ig]) / (self.route_s[ig + 1] - self.route_s[ig])
                g = self.route[ig] + f * (self.route[ig + 1] - self.route[ig])
            else:
                g = self.route[ig]
            if self._in_holdout(g):
                dmax = max(dmax, self.speed_est * self.curric.k_max * 2.0)
                continue
            line = np.vstack([origin[None, :], self.route[i0:ig + 1], g[None, :]])
            return g, line, (st - s0) / self.speed_est
        return None

    # ------------------------------------------------------------- assign
    def assign(self, idx) -> None:
        """Give freshly spawned envs ``idx`` their goal + line. Reads the
        core's state view (ended rows are already the new spawn)."""
        idx = np.asarray(idx, np.int64)
        if len(idx) == 0:
            return
        org = self.core.states_view["origin"][idx].astype(np.float64)
        dd = np.asarray(self._field_sample(org), np.float64)
        self.depth[idx] = np.clip(1.0 - dd / self.d0, 0.0, 1.0)
        lines, centers = [], np.zeros((len(idx), 3), np.float32)
        for n, i in enumerate(idx):
            key = (round(float(org[n, 0]), 1), round(float(org[n, 1]), 1),
                   round(float(org[n, 2]), 1))
            j = self.pool_map.get(key) if self.pool is not None else None
            g = None
            rg = None
            if self.route is not None and self.rng.random() < self.route_frac:
                rg = self._route_goal(org[n])
            if rg is not None:
                g, rline, kk = rg
                line = None
                if self.line is not None or self.arc is not None:
                    from .goals import resample_polyline_np
                    line = resample_polyline_np(rline)
                self.kind[i] = 2
                self.k[i] = float(max(1.0, kk))
            elif (j is not None and np.isfinite(self.pool[0][j, 0])
                    and self.rng.random() >= self.air_frac
                    and not self._in_holdout(self.pool[0][j])):
                g = self.pool[0][j].astype(np.float64)
                seg = self.pool[1][j, :int(self.pool[2][j])].astype(np.float64)
                line = None
                if self.line is not None or self.arc is not None:
                    # the fan / arc need a line; the ball does not - skip
                    # the RDP + resample per spawn when neither is used
                    if len(seg) >= 2 and np.linalg.norm(seg[-1] - g) < 1.0:
                        line = segment_line(seg)
                    else:
                        line = chord_line(org[n], g)
                self.kind[i] = 0
                self.k[i] = max(1.0, float(len(seg) - 1))
            else:
                self._last_tau = 0.0
                g = np.asarray(self._air_goal(org[n]), np.float64)
                line = (chord_line(org[n], g)
                        if (self.line is not None or self.arc is not None)
                        else None)
                self.kind[i] = 1
                self.k[i] = float(np.linalg.norm(g - org[n]) / self.speed_est
                                  + self._last_tau)
            _lmax = (self.line.pts.shape[1] if self.line is not None
                     else (self.arc.pts.shape[1] if self.arc is not None
                           else 0))
            if line is not None and _lmax and len(line) > _lmax:
                line = line[-_lmax:]                    # keep the goal end
            lines.append(line)
            centers[n] = g
            self.n_assigned[int(self.kind[i])] += 1
        if self.line is not None:
            self.line.set_lines(idx, lines)
        if self.arc is not None:
            # each line starts at the spawn: arc 0 there, by construction
            self.arc.set_lines(idx, lines)
            rf = self.reward_fn
            if rf is not None and getattr(rf, "_arc_spawn", None) is not None:
                rf._arc_spawn[idx] = 0.0
                rf._arc_max[idx] = 0.0
        if self.ball is not None:
            self.ball.set_goals(idx, centers)
        self.sphere.set(idx, centers)
        self.pending[idx] = False

    # -------------------------------------------------------------- tick
    def on_step(self, done, trunc, ep_len) -> np.ndarray:
        """After fleet.step: settle the episodes that just ended (a pending
        sphere entry is their success), then arm the kill for new entries.
        Returns the goal mask for THIS tick's reward call."""
        ended = (np.asarray(done, bool) | np.asarray(trunc, bool))
        gmask = self.pending & ended
        if ended.any():
            for i in np.flatnonzero(ended):
                self.stats.note(self.k[i], KIND[int(self.kind[i])],
                                bool(gmask[i]), int(ep_len[i]))
                if self.kind[i] == 2:          # route-depth goals only
                    b = min(9, int(self.depth[i] * 10.0))
                    self.band_n[b] += 1
                    self.band_ok[b] += int(gmask[i])
                # every kind feeds the band rule: air goals carry
                # k = distance / speed_est, and they are the bulk
                self.curric.note(self.k[i], bool(gmask[i]))
            self.pending[ended] = False
        hit = self.sphere.hit(self.core.states_view["origin"]) & ~ended \
            & ~self.pending
        if hit.any():
            self.core.force_fail(hit)
            self.pending |= hit
        return gmask

    # --------------------------------------------------------------- logs
    def note(self, step: int) -> str:
        st = self.stats.pop()
        n = int(st.get("n", 0) or 0)
        sr = st.get("success_rate", float("nan"))
        per = st.get("kind", {})
        ach = per.get("achieved", {})
        air = per.get("air", {})
        rte = per.get("finish", {})          # route-depth goals (kind 2)
        ev_s, ev_n = self._last_eval
        self._w.writerow([step, sr, ach.get("success_rate", float("nan")),
                          air.get("success_rate", float("nan")), n,
                          ach.get("n", 0), air.get("n", 0),
                          self.curric.k_max, st.get("ticks_mean", float("nan")),
                          ev_s, ev_n, rte.get("success_rate", float("nan")),
                          rte.get("n", 0)]
                         + [(self.band_ok[b] / self.band_n[b]) if self.band_n[b]
                            else float("nan") for b in range(10)]
                         + [int(v) for v in self.band_n])
        bands = " ".join(f"{b * 10}%:{self.band_ok[b] / self.band_n[b]:.0%}"
                         for b in range(10) if self.band_n[b] >= 20)
        self.band_n[:] = 0
        self.band_ok[:] = 0
        self._csv.flush()
        asg = self.n_assigned.copy()
        self.n_assigned[:] = 0
        if n == 0:
            return (f"  goals -/- kmax {self.curric.k_max:.0f}s "
                    f"asg {asg[0]}/{asg[1]}")
        return (f"  goals {sr:5.1%} (route {rte.get('success_rate', float('nan')):5.1%}"
                f"/{rte.get('n', 0)} ach {ach.get('success_rate', float('nan')):5.1%}"
                f"/{ach.get('n', 0)} air {air.get('success_rate', float('nan')):5.1%}"
                f"/{air.get('n', 0)}) kmax {self.curric.k_max:.0f}s "
                f"asg {asg[2]}/{asg[0]}/{asg[1]}"
                + (f"  depth {bands}" if bands else ""))

    # --------------------------------------------------------------- eval
    def eval_hooks(self, eval_core, seed: int, holdout_only: bool = False):
        """(episode_meta, on_tick) for record_rollout on the 1-env eval core:
        a random reachable-air goal per episode inside the current air
        radius (seeded per recording), the chord as its line, sphere entry
        force-fails the env (the recorder sees a normal episode end)."""
        rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        ev = self.ev
        ev.update({"n": 0, "succ": 0, "pending": False, "center": None,
                   "ticks": [], "dists": [], "t0": 0})
        sampler = self.air
        if holdout_only and self.holdout is not None:
            lo, hi = self.holdout
            gf = self.air.reachable_fn

            def inside(p, _lo=lo, _hi=hi):
                d = np.asarray(self._field_sample(p), np.float64)
                return (d >= _lo) & (d <= _hi)
            sampler = AirSampler(self.mins, self.maxs,
                                 lambda p: gf(p) & inside(p))
        r_max = self._air_radius()

        def episode_meta(ep):
            o = eval_core.states_view["origin"][0].astype(np.float64)
            try:
                g = sampler.sample_near(64, o, self.r_min, r_max, rng)[0]
            except RuntimeError:
                g = sampler.sample(512, rng)[0]
            g = np.asarray(g, np.float64)
            line = chord_line(o, g)
            if self.eval_line is not None:
                self.eval_line.set_lines(np.array([0]), [line])
            if self.eval_ball is not None:
                self.eval_ball.set_goals([0], [g])
            ev["center"] = g
            ev["pending"] = False
            ev["n"] += 1
            ev["dists"].append(float(np.linalg.norm(g - o)))
            thin = line[:: max(1, len(line) // 64)]
            return {"goal": {"center": [float(v) for v in g],
                             "radius": self.radius},
                    "line": [[float(v) for v in p] for p in thin]}

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
            o = eval_core.states_view["origin"][0].astype(np.float64)
            if np.linalg.norm(o - ev["center"]) <= self.radius:
                m = np.zeros(eval_core.num_envs, np.uint8)
                m[0] = 1
                eval_core.force_fail(m)
                ev["pending"] = True

        return episode_meta, on_tick

    def _in_holdout(self, g) -> bool:
        """Reached-state goals inside the held-out band are refused too,
        or G3's held-out test would be trained on through the reservoir."""
        if self.holdout is None:
            return False
        d = float(self._field_sample(np.asarray(g, np.float64)[None, :])[0])
        return self.holdout[0] <= d <= self.holdout[1]

    def _field_sample(self, p):
        # the sampler's reachable_fn is goal_field.reachable (bound method);
        # its __self__ is the field, whose sample() gives the distance
        return self.air.reachable_fn.__self__.sample(np.asarray(p, np.float64))

    def eval_note(self) -> str:
        ev = self.ev
        self._last_eval = ((ev["succ"] / ev["n"]) if ev["n"] else float("nan"),
                           ev["n"])
        md = float(np.mean(ev["dists"])) if ev["dists"] else float("nan")
        mt = (float(np.mean(ev["ticks"])) / 100.0) if ev["ticks"] else float("nan")
        return (f"  goals {ev['succ']}/{ev['n']} (mean dist {md:,.0f}u"
                + (f", {mt:.1f}s" if mt == mt else "") + ")")
