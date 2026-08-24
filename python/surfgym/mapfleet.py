"""mapfleet.py — one shared policy over several maps at once (``--maps``).

The trainer used to hold exactly one of everything that is map-shaped: one
``SurfCore``, one ``GpuLidar``, one goal field, one ``RaceReward``, one
respawn reservoir, one spawn pool. Multi-map training turns each of those
into a **slot** — a map plus the contiguous range of env indices it owns —
and this module is the thing that drives a list of slots as if it were one
vectorised env.

    slot: bsp, core, lidar, goal_field, reward_fn, respawn, spawn pool,
          d0, env range [lo, hi)

Two rules the whole file is built around.

**The single-slot path must stay bit-identical to the pre-``--maps``
trainer.** Every arm in flight resumes a checkpoint trained on it, and surf
is chaotic enough that one differing float forks the trajectory. So every
aggregating method here short-circuits at ``len(slots) == 1`` and returns
the core's own buffers, unconcatenated and uncopied — literally the
expression the trainer used to inline. ``tests/python/test_multimap.py``
pins that against a hand-driven reference core, obs, reward, done/trunc and
the torch RNG cursor included.

**Everything map-shaped is per slot, and nothing map-shaped is shared.** The
reward scale is ``100 / d0`` *of that map*, so finishing any map is worth
100 whatever its length (cannonball d0 = 198,380 u, petrus_lite = 35,637 u;
one shared scale would make the short map invisible). The novelty count
table is keyed by map cells, the respawn reservoir holds raw map
coordinates, the goal field and the spawn pool are that map's — sharing any
of them across maps is nonsense, not an approximation.

The PPO update never learns this module exists: it sees one batch whose
rows happen to come from different maps.

FOLLOW-UP, deliberately not done here: ``ep_ticks`` is still ONE number for
the whole fleet. It is 12,000 (120 s) and tuned for cannonball, whose route
takes ~80 s to run; petrus_lite is about five times shorter, so most of an
episode there is a dead agent waiting out the stall-kill. ``ep_ticks`` maps
onto ``SurfEnvConfig.max_episode_ticks``, which is already per core, so a
per-slot episode cap is a small change - it just is not this one, because it
would move the single-map path's episode length too.
"""
from __future__ import annotations

import numpy as np

__all__ = ["MapSlot", "MapFleet", "map_tag", "map_tags"]


def map_tag(stem: str) -> str:
    """Short per-map label for CSV columns and artifact names.

    ``surf_src_cannonball -> cannonball``, ``surf_petrus_lite ->
    petrus_lite``. Only the shipped ``surf_``/``surf_src_`` prefixes are
    stripped; anything else is returned as-is. :func:`map_tags` falls back
    to full stems if stripping would collide.
    """
    t = str(stem)
    for pre in ("surf_src_", "surf_"):
        if t.startswith(pre):
            return t[len(pre):]
    return t


def map_tags(stems) -> list:
    """Unique short tags for a map list, or the full stems if they collide."""
    stems = [str(s) for s in stems]
    tags = [map_tag(s) for s in stems]
    return tags if len(set(tags)) == len(tags) else stems


class MapSlot:
    """One map's half of the trainer, plus the env range it owns.

    ``lo``/``hi`` are indices into the trainer's (N, ...) arrays; the slot
    owns rows ``[lo, hi)`` of every one of them — actions, obs, rewards,
    the depth-image block, the rollout buffers. Ranges are contiguous and
    in slot order, so every split is a view and never a gather.
    """

    __slots__ = ("name", "bsp", "core", "lo", "hi", "lidar", "goal_field",
                 "reward_field", "goal_box", "d0", "rf_d0", "cell",
                 "goal_cell",
                 "reward_fn", "respawn", "pool", "plat_pool", "eval_core",
                 "map_center", "eval_reward_feed", "eval_latch_feed", "tag",
                 "d_latch", "eval_rank", "finish_kind")

    def __init__(self, name: str, bsp: str, core, lo: int, hi: int):
        self.name = str(name)
        self.tag = map_tag(name)
        self.bsp = str(bsp)
        self.core = core
        self.lo, self.hi = int(lo), int(hi)
        if self.hi - self.lo != core.num_envs:
            raise ValueError(
                f"slot {self.name}: env range [{lo}, {hi}) is "
                f"{self.hi - self.lo} wide but its core has "
                f"{core.num_envs} envs")
        self.lidar = None
        self.goal_field = None
        self.reward_field = None
        self.goal_box = None
        self.d0 = None            # start geodesic on the STANDARD field
        self.rf_d0 = None         # ... on the field the shaping actually uses
        self.cell = None
        # the SHAPING field's voxel size. Defaults to `cell`; --goal-cell
        # decouples them, because perception fidelity and reward resolution
        # are unrelated jobs and the field is cubically cheaper to coarsen.
        self.goal_cell = None
        self.reward_fn = None
        self.respawn = None
        self.pool = None
        self.plat_pool = None
        self.eval_core = None
        self.map_center = None
        self.eval_reward_feed = None
        self.eval_latch_feed = None
        # --race-latch-frac resolves to a different absolute distance per
        # map (frac * this map's own rf_d0); --race-latch is the same number
        # on every slot, which is why it is single-map only
        self.d_latch = 0.0
        # which DDP rank runs this map's greedy eval and writes its
        # trajectory. 0 at world_size 1, so the single-GPU path is unchanged.
        self.eval_rank = 0
        # "trigger" (a real trigger_multiple curtain, median face 808,960 u^2)
        # or "button" (a +use button box padded by 64 u, ~8x smaller in face
        # area, and THE SIMULATOR CANNOT PRESS A BUTTON - arriving inside the
        # box is substituted for the press, CLAUDE.md 4b). A null on a button
        # map is much weaker evidence than a null on a trigger map, so the
        # aggregate is reported split as well as pooled.
        self.finish_kind = "unknown"

    @property
    def n(self) -> int:
        return self.hi - self.lo

    @property
    def sl(self) -> slice:
        """The env slice. A basic slice of a C-contiguous array is a view,
        so ``core.step(actions[slot.sl])`` copies nothing and still passes
        ``validate_actions``' contiguity check."""
        return slice(self.lo, self.hi)

    def __repr__(self) -> str:      # pragma: no cover - diagnostics only
        return (f"MapSlot({self.name!r}, envs [{self.lo}, {self.hi}), "
                f"d0={self.d0})")


class MapFleet:
    """A list of :class:`MapSlot` driven as one vectorised env.

    Everything the trainer's hot loop used to call on ``core`` /
    ``reward_fn`` / ``respawn`` has a method here. With one slot each of
    them returns that slot's own object or buffer untouched.
    """

    # how many calls a sampled reservoir min-depth is reused for. It is a
    # printed diagnostic; sampling 100k states x n_maps on every iteration
    # is real numpy time for a number that moves on the scale of minutes.
    MIND_EVERY = 25

    def __init__(self, slots):
        self._mind: dict = {}
        self._mind_age = self.MIND_EVERY      # first call always samples
        self.slots = list(slots)
        if not self.slots:
            raise ValueError("MapFleet needs at least one slot")
        self.single = len(self.slots) == 1
        self.n_maps = len(self.slots)
        self.n_envs = sum(s.n for s in self.slots)
        exp = 0
        for s in self.slots:
            if s.lo != exp:
                raise ValueError(
                    f"slot {s.name} starts at {s.lo}, expected {exp}: env "
                    "ranges must be contiguous and in slot order")
            exp = s.hi
        self.obs_dim = int(self.slots[0].core.obs_dim)
        for s in self.slots:
            if int(s.core.obs_dim) != self.obs_dim:
                raise ValueError("slots disagree on obs_dim "
                                 f"({s.name}: {s.core.obs_dim})")
        if not self.single:
            n, d = self.n_envs, self.obs_dim
            self._obs = np.zeros((n, d), np.float32)
            self._term = np.zeros((n, d), np.float32)
            self._base = np.zeros(n, np.float32)
            self._done = np.zeros(n, np.uint8)
            self._trunc = np.zeros(n, np.uint8)
            self._goal = np.zeros(n, np.uint8)

    # -- identity helpers ---------------------------------------------------

    @property
    def maps(self) -> list:
        return [s.name for s in self.slots]

    @property
    def tags(self) -> list:
        return map_tags(self.maps)

    def retag(self) -> None:
        """Re-derive slot tags, falling back to full stems on a collision."""
        for s, t in zip(self.slots, self.tags):
            s.tag = t

    def slot_of(self, env: int) -> MapSlot:
        for s in self.slots:
            if s.lo <= env < s.hi:
                return s
        raise IndexError(f"env {env} is outside the fleet's {self.n_envs}")

    # -- hot path -----------------------------------------------------------

    def reset(self, seed: int = 0):
        """Reset every core; returns (N, obs_dim). Slot i is seeded
        ``seed + 1013*i`` so two maps do not draw a correlated spawn
        sequence; slot 0 keeps the caller's seed exactly."""
        if self.single:
            return self.slots[0].core.reset(seed)
        for i, s in enumerate(self.slots):
            self._obs[s.sl] = s.core.reset(int(seed) + 1013 * i)
        return self._obs

    def step(self, actions):
        """``(obs, base_rewards, done, trunc, terminal_obs)``, all (N, ...).

        Single slot: the core's own buffers, exactly as ``core.step``
        returned them before this module existed."""
        if self.single:
            return self.slots[0].core.step(actions)
        for s in self.slots:
            o, r, d, t, to = s.core.step(actions[s.sl])
            self._obs[s.sl] = o
            self._base[s.sl] = r
            self._done[s.sl] = d
            self._trunc[s.sl] = t
            self._term[s.sl] = to
        return self._obs, self._base, self._done, self._trunc, self._term

    def reward(self, prev_obs, obs, term_obs, base_r, done, trunc, goal=None):
        """Each slot's own reward function over its own rows -> (N,) f32.

        The per-map ``scale = 100/d0`` lives inside those functions, so a
        finish is worth 100 on every map regardless of its length."""
        if self.single:
            s = self.slots[0]
            if goal is None:
                return s.reward_fn(prev_obs, obs, term_obs, base_r, done,
                                   trunc, s.core)
            return s.reward_fn(prev_obs, obs, term_obs, base_r, done, trunc,
                               s.core, goal=goal)
        out = np.zeros(self.n_envs, np.float32)
        goal = None if goal is None else np.asarray(goal)
        for s in self.slots:
            kw = {} if goal is None else {"goal": goal[s.sl]}
            out[s.sl] = s.reward_fn(prev_obs[s.sl], obs[s.sl], term_obs[s.sl],
                                    base_r[s.sl], done[s.sl], trunc[s.sl],
                                    s.core, **kw)
        return out

    def goal_hits(self):
        """(N,) uint8 — 1 on the tick an env crossed its map's finish box."""
        if self.single:
            return self.slots[0].core.goal_hits
        for s in self.slots:
            self._goal[s.sl] = s.core.goal_hits
        return self._goal

    def apply_stall_kills(self) -> None:
        """Drain each slot's stagnation mask into its own ``force_fail``."""
        for s in self.slots:
            pop = getattr(s.reward_fn, "pop_stall_mask", None)
            if pop is None:
                continue
            sm = pop()
            if sm is not None:
                s.core.force_fail(sm)

    def stagnant_mask(self):
        """(N,) bool of envs making no progress, or None when no slot
        tracks it."""
        if self.single:
            fn = getattr(self.slots[0].reward_fn, "stagnant_mask", None)
            return None if fn is None else fn()
        out = None
        for s in self.slots:
            fn = getattr(s.reward_fn, "stagnant_mask", None)
            m = None if fn is None else fn()
            if m is None:
                continue
            if out is None:
                out = np.zeros(self.n_envs, bool)
            out[s.sl] = m
        return out

    def latch_flags(self):
        """(N,) bool — the ``--race-latch`` observation column, per map."""
        if self.single:
            return self.slots[0].reward_fn.latch_flags()
        out = np.zeros(self.n_envs, bool)
        for s in self.slots:
            f = s.reward_fn.latch_flags()
            if f is not None:
                out[s.sl] = f
        return out

    def observe_respawn(self, ended, stagnant=None) -> None:
        """Feed each slot's reservoir its own envs. Reservoir states are raw
        map coordinates — a state from one map is meaningless in another, so
        the buffers never mix."""
        for s in self.slots:
            if s.respawn is None:
                continue
            s.respawn.observe(s.core.states_view, ended[s.sl],
                              stagnant=None if stagnant is None
                              else stagnant[s.sl])

    def track_start_bins(self, ended, goal_hits, start_bin) -> None:
        """Attribute ended episodes to the distance bin they STARTED in and
        stash the new episodes' start bins, per slot.

        Bins are indices into that slot's own reservoir statistics, so the
        env indices handed to ``note_spawns`` are slot-LOCAL while
        ``start_bin`` is a fleet-wide array."""
        goal_hits = np.asarray(goal_hits, bool)
        for s in self.slots:
            r = s.respawn
            if r is None:
                continue
            loc = np.flatnonzero(ended[s.sl])
            if not len(loc):
                continue
            ei = loc + s.lo
            gn = goal_hits[ei]
            known = start_bin[ei] >= 0
            if known.any():
                r.note_outcomes(start_bin[ei][known], gn[known])
            nb = r.bin_of(s.core.states_view[loc]["origin"])
            start_bin[ei] = nb
            r.note_spawns(nb, loc)

    def set_step(self, global_step: int) -> None:
        for s in self.slots:
            fn = getattr(s.reward_fn, "set_step", None)
            if fn is not None:
                fn(global_step)

    def on_reset(self) -> None:
        for s in self.slots:
            s.reward_fn.on_reset(s.core)

    # -- vision -------------------------------------------------------------

    def fill_pose(self, vis_np) -> None:
        """Write every slot's live pose into the pinned (N, 6) upload row."""
        for s in self.slots:
            sv = s.core.states_view
            vis_np[s.lo:s.hi, 0:3] = sv["origin"]
            vis_np[s.lo:s.hi, 3] = sv["yaw"]
            vis_np[s.lo:s.hi, 4] = sv["pitch"]
            vis_np[s.lo:s.hi, 5] = sv["ducked"]

    def render(self, vis_gpu, out):
        """Render each slot's depth block with ITS map's SDF.

        ``out`` is the (N, FRAME) staging tensor; single slot returns the
        renderer's own tensor and never touches it."""
        if self.single:
            s = self.slots[0]
            img = s.lidar.render(vis_gpu[:, 0:3], vis_gpu[:, 3],
                                 vis_gpu[:, 4], vis_gpu[:, 5])
            return img.reshape(vis_gpu.shape[0], -1)
        for s in self.slots:
            v = vis_gpu[s.lo:s.hi]
            img = s.lidar.render(v[:, 0:3], v[:, 3], v[:, 4], v[:, 5])
            out[s.lo:s.hi] = img.reshape(s.n, -1)
        return out

    def render_rows(self, idx, origin, yaw_deg, pitch_deg, ducked):
        """Depth for an arbitrary set of env rows, each on ITS map's SDF.

        Used by the truncation bootstrap, which renders a reconstructed
        terminal pose. ``idx`` are env indices; the pose tensors are already
        gathered to those rows and in the same order.
        """
        n = len(idx)
        if self.single:
            return self.slots[0].lidar.render(
                origin, yaw_deg, pitch_deg, ducked).reshape(n, -1)
        import torch
        idx = np.asarray(idx, np.int64)
        out = None
        for s in self.slots:
            m = (idx >= s.lo) & (idx < s.hi)
            if not m.any():
                continue
            j = torch.as_tensor(np.flatnonzero(m), device=origin.device)
            im = s.lidar.render(origin[j], yaw_deg[j], pitch_deg[j],
                                ducked[j]).reshape(int(j.numel()), -1)
            if out is None:
                out = torch.zeros((n, im.shape[1]), device=im.device,
                                  dtype=im.dtype)
            out.index_copy_(0, j, im)
        return out

    # -- truncation bootstrap ----------------------------------------------
    # V(s_T) is rebuilt from the TERMINAL scalar obs, whose slots 12..14 are
    # (pos - map_center) / 2000 — and map_center is per map. A shared centre
    # would put the reconstructed pose thousands of units from where the
    # episode actually ended and render the wrong depth image for it.

    def map_centers(self, idx):
        """(len(idx), 3) f32 of each row's own map centre."""
        idx = np.asarray(idx, np.int64)
        if self.single:
            return self.slots[0].map_center
        out = np.zeros((len(idx), 3), np.float32)
        for s in self.slots:
            m = (idx >= s.lo) & (idx < s.hi)
            if m.any():
                out[m] = s.map_center
        return out

    def terminal_latch(self, idx, pos):
        """The ``--race-latch`` flag at a truncated episode's terminal state:
        the flag one reward call ago, OR'd with the terminal state's own
        distance — evaluated on each row's own field and threshold."""
        idx = np.asarray(idx, np.int64)
        out = np.zeros(len(idx), bool)
        for s in self.slots:
            m = (idx >= s.lo) & (idx < s.hi)
            if not m.any():
                continue
            j = np.flatnonzero(m)
            dT = s.reward_field.sample(pos[j])
            out[j] = (s.reward_fn.latch_boot()[idx[j] - s.lo]
                      | (dT <= s.reward_fn.d_latch))
        return out

    # -- logging ------------------------------------------------------------

    def pop_stats(self) -> dict:
        """Episode outcomes since the last call, pooled over maps.

        Rates are episode-weighted, not slot-averaged: a map whose episodes
        are 5x shorter would otherwise dominate a plain mean."""
        tot = {"success_rate": float("nan"), "finish_s": float("nan"),
               "episodes": 0, "int_per_ep": float("nan")}
        n_ep = 0
        sr = fin = ipe = 0.0
        n_fin = n_int = 0
        for s in self.slots:
            pop = getattr(s.reward_fn, "pop_stats", None)
            if pop is None:
                continue
            st = pop()
            e = int(st.get("episodes") or 0)
            n_ep += e
            if e and st["success_rate"] == st["success_rate"]:
                sr += st["success_rate"] * e
            if st["finish_s"] == st["finish_s"]:
                fin += st["finish_s"]
                n_fin += 1
            if e and st["int_per_ep"] == st["int_per_ep"]:
                ipe += st["int_per_ep"] * e
                n_int += e
        if n_ep:
            tot["success_rate"] = sr / n_ep
            tot["episodes"] = n_ep
        if n_fin:
            tot["finish_s"] = fin / n_fin
        if n_int:
            tot["int_per_ep"] = ipe / n_int
        return tot

    def reservoir_size(self) -> int:
        return sum(s.respawn.size for s in self.slots if s.respawn is not None)

    def reservoir_min_depth(self) -> float:
        """Smallest geodesic distance-to-finish held by ANY slot's reservoir,
        as a FRACTION of that slot's own d0 (maps differ 5x in length, so the
        raw unit is not comparable across them). NaN when nothing is stored.

        This exists because ``race/win_rate`` is the project's third
        deceptive metric: round 19 saw it go 0 -> 18.46% while the frontier
        sat flat, because the reservoir had drifted to 1,485 u from the goal
        and the agent was being respawned next to the finish and walking in.
        A win rate that rises while this falls is measuring the harvest, not
        the policy - so the trainer prints and logs the two together."""
        best = float("nan")
        for s in self.slots:
            r = s.respawn
            if r is None or not r.size or not s.rf_d0:
                continue
            # binned reservoirs already carry the distance per stored state
            # (it is what the bins are computed from), so the min is free.
            d = getattr(r, "_d", None)
            if d is not None:
                lo = float(d[:r.size].min())
            else:
                # uniform mode keeps no distance column, so this has to
                # sample the field - 100k states per slot, every call. It is
                # a step-line diagnostic, not a term in anything, so it is
                # refreshed on a cadence rather than per iteration.
                fld = (s.reward_field if s.reward_field is not None
                       else s.goal_field)
                if fld is None:
                    continue
                cache = self._mind.get(id(r))
                if cache is not None and self._mind_age < self.MIND_EVERY:
                    lo = cache
                else:
                    lo = float(fld.sample(r._store[:r.size]["origin"]).min())
                    self._mind[id(r)] = lo
            if not s.rf_d0:
                continue
            d0 = lo / s.rf_d0
            if best != best or d0 < best:
                best = d0
        self._mind_age = 0 if self._mind_age >= self.MIND_EVERY \
            else self._mind_age + 1
        return best

    # -- DDP: fixed-shape reductions over the (replicated) slot list --------
    # Every rank holds every slot, so each of these is one collective whose
    # shape does not depend on the rank. That property is the reason maps are
    # replicated across ranks rather than sharded over them.

    def stats_vector(self):
        """Fleet-summed RaceReward outcome counters (see
        ``RaceReward.stats_vector``): a reducible f64 vector, so pooling over
        MAPS and pooling over RANKS are the same operation applied twice."""
        tot = None
        for s in self.slots:
            fn = getattr(s.reward_fn, "stats_vector", None)
            if fn is None:
                continue
            v = np.asarray(fn(), np.float64)
            tot = v.copy() if tot is None else tot + v
        return np.zeros(6, np.float64) if tot is None else tot

    def clear_stats(self) -> None:
        for s in self.slots:
            fn = getattr(s.reward_fn, "clear_stats", None)
            if fn is not None:
                fn()

    def counts_delta_sparse(self, dtype):
        """Every slot's novelty-count delta as ONE (slot, cell, inc) array.

        Batched deliberately: one all-gather per iteration regardless of the
        map count. A per-slot gather would be NMAPS collectives, and the
        fleet is meant to reach three digits of maps."""
        out = []
        for i, s in enumerate(self.slots):
            fn = getattr(s.reward_fn, "counts_delta_sparse", None)
            if fn is None or getattr(s.reward_fn, "int_coef", 0.0) <= 0.0:
                continue
            cells, incs = fn()
            if cells is None or not len(cells):
                continue
            a = np.empty(len(cells), dtype)
            a["slot"] = i
            a["cell"] = cells
            a["inc"] = incs
            out.append(a)
        return np.concatenate(out) if out else np.empty(0, dtype)

    def apply_counts_delta_sparse(self, rows) -> None:
        """Scatter a gathered (slot, cell, inc) array back per slot."""
        if not len(rows):
            return
        for i, s in enumerate(self.slots):
            fn = getattr(s.reward_fn, "apply_counts_delta_sparse", None)
            if fn is None:
                continue
            m = rows["slot"] == i
            if m.any():
                fn(rows["cell"][m], rows["inc"][m])
