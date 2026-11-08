"""knob_ge.py - Go-Explore whose explore step runs on ONE knob between the policy and random,
set by one rule from the archive's own yield (unr1b4, slot 4, 2026-11-08).

The loop (autoresearch/PROTOCOL.md, "the research question"): an archive of exact simulator
states, cells keyed by generic quantities (a position cell, the kinetic-energy height
|v|^2 / 2g, optionally the velocity heading), filled only with states the map HOLDS (walkable
ground or a surf contact: a ballistic apex over the void is not a place a line can be continued
from); a cell is chosen with the Nature weight W = 1 / sqrt(1 + times chosen), restored exactly
(``core.set_state``, the run's clock zeroed like every trainer spawn), and explored for
``decisions`` decisions. Each explore RUN commits, for its whole length, to one of two modes
(the Nature paper's policy-based Go-Explore commits 50/50 per explore step):

* RANDOM - a uniform draw over the engine keys (forward, side, jump, duck) and a uniform draw
  of the yaw target in the policy's OWN velocity frame (u ~ U(-1, 1) through the policy's
  off_warp, the October ez-burst draw), held and redrawn with probability 1 - ``repeat`` per
  decision (the paper's 0.95 action repeat; coherent randomness is the one thing every result
  agrees on). The pitch stays the policy's (it moves only the camera).
* POLICY - the checkpoint's own sampled actions (its native temperature).

THE KNOB is the share of runs that are RANDOM, lambda in [floor, 1 - floor]. ONE rule sets it
from the run's own state: probability matching on the NEW-CELL YIELD per decision of each mode
(decayed sums over the runs that ended): lambda = y_rand / (y_rand + y_pol). Equal yields give
the paper's 0.5; a mode that stops finding cells keeps the floor. Nothing here reads a map, a
record, a route or a reward: the constants (cell size, energy bucket, run length, repeat, floor)
are set once and ship unchanged to any map (CLAUDE.md 0b). The knob's trajectory over a run is
logged, which is the user's question ("at which point is the policy-guided search more useful,
and maybe we conclude that it's not useful at all").

Every new cell keeps the exact action table that reached it from its parent cell (the run's
engine rows and view rows), so the deepest cell's chain replays open-loop from its root on a
1-env core (checked state by state at every junction) into the per-tick states of a curriculum
line: the greedy line from the TRUE start up to the root, then the chain.

The roots are the checkpoint's own GREEDY line from the true start, in its last ``roots_secs``
seconds before it stalls (auto_spine's GE rule: the first tick within ``stall_margin`` u of its
best field distance).

Pure helpers (the knob rule, the archive, the chain bookkeeping) are unit-tested on the CPU in
tests/python/test_knob_ge.py; ``knob_ge_search`` runs inside tools/beam_tas.py --ge-knob, which
builds the checkpoint-faithful policy wrappers and cores it needs.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

G_ENERGY = 800.0                 # sv_gravity: energy height h = z + |v|^2 / (2 g)


def knob_share(c_rand: float, n_rand: float, c_pol: float, n_pol: float,
               floor: float = 0.05) -> float:
    """The knob rule: the share of explore runs that act RANDOMLY = probability matching on the
    new-cell yield per decision of each mode (``c`` new cells over ``n`` decisions, the decayed
    sums of the runs that ended). Equal yields -> 0.5; a mode with no decisions yet, or both
    modes without a cell -> 0.5; the result is clipped to [floor, 1 - floor] so neither mode
    is ever starved of the evidence that could bring it back."""
    fl = float(floor)
    if n_rand <= 0.0 or n_pol <= 0.0:
        return 0.5
    yr = max(0.0, float(c_rand)) / float(n_rand)
    yp = max(0.0, float(c_pol)) / float(n_pol)
    if yr + yp <= 0.0:
        return 0.5
    return float(min(1.0 - fl, max(fl, yr / (yr + yp))))


class Knob:
    """The decayed yield sums behind knob_share. ``horizon`` = the number of run ends over
    which an old run's evidence decays by 1/e (set to the population: about one generation of
    runs; the search passes 16 populations: one population (the first smoke, unr1b4_smoke
    2026-11-08) swung lambda 0.55 -> 0.18 -> 0.52 -> 0.95 within 2 minutes while the two modes'
    totals stayed within 10% of each other - new cells come in bursts); ``warm`` = run ends per
    mode before the rule speaks (lambda 0.5 until then);
    ``fixed`` = a constant lambda instead of the rule (the two ends of the knob for an
    ablation: 0 = all policy, 1 = all random)."""

    def __init__(self, horizon: int, floor: float = 0.05, warm: int = 256,
                 fixed: float | None = None) -> None:
        self.decay = 1.0 - 1.0 / max(1.0, float(horizon))
        self.floor = float(floor)
        self.warm = int(warm)
        self.fixed = None if fixed is None else float(fixed)
        self.c = np.zeros(2, np.float64)      # [policy, random] decayed new cells
        self.n = np.zeros(2, np.float64)      # [policy, random] decayed decisions
        self.ends = np.zeros(2, np.int64)     # run ends per mode (undecayed)
        self.cells = np.zeros(2, np.int64)    # new cells per mode (undecayed)
        self.decs = np.zeros(2, np.int64)     # decisions per mode (undecayed)

    def credit(self, mode: np.ndarray, new_cells: np.ndarray, decisions: np.ndarray) -> None:
        """Runs that ended: their mode (0 policy, 1 random), the cells they added, the
        decisions they spent. Each end decays the old sums once (a batch of k ends decays k
        times, the order within the batch does not matter for the sums' ratio much)."""
        mode = np.asarray(mode, np.int64)
        new_cells = np.asarray(new_cells, np.float64)
        decisions = np.asarray(decisions, np.float64)
        k = len(mode)
        if k == 0:
            return
        f = self.decay ** k
        self.c *= f
        self.n *= f
        for m in (0, 1):
            s = mode == m
            self.c[m] += float(new_cells[s].sum())
            self.n[m] += float(decisions[s].sum())
            self.ends[m] += int(s.sum())
            self.cells[m] += int(new_cells[s].sum())
            self.decs[m] += int(decisions[s].sum())

    @property
    def lam(self) -> float:
        if self.fixed is not None:
            return self.fixed
        if self.ends.min() < self.warm:
            return 0.5
        return knob_share(self.c[1], self.n[1], self.c[0], self.n[0], self.floor)

    def yields(self) -> tuple[float, float]:
        """(policy, random) decayed new cells per decision."""
        return (float(self.c[0] / max(self.n[0], 1e-9)),
                float(self.c[1] / max(self.n[1], 1e-9)))


class KnobArchive:
    """Cell -> the FIRST exact state that reached it (never replaced: every chain replays
    exactly), with its observation row (the core's scalar obs at that state, patched into the
    env on a restore so the first decision reads the right velocity), its parent cell (the cell
    its run was RESTORED from; -1 for a root), the run that found it and how many ticks into
    that run (``off``), the root's index on the greedy line (roots only), the field distance d,
    the times it was chosen, ticks from the line's start along its chain (``depth``) and the
    mode that found it (-1 root, 0 policy, 1 random)."""

    def __init__(self, obs_dim: int, state_dtype, cap: int = 1 << 15) -> None:
        self.obs_dim = int(obs_dim)
        self.dtype = state_dtype
        self.n = 0
        self.index: dict[int, int] = {}
        self._alloc(int(cap))

    def _alloc(self, cap: int) -> None:
        old = None if not hasattr(self, "key") else self._cols()
        self.cap = cap
        self.key = np.zeros(cap, np.int64)
        self.state = np.zeros(cap, self.dtype)
        self.obs = np.zeros((cap, self.obs_dim), np.float32)
        self.parent = np.full(cap, -1, np.int64)
        self.run = np.full(cap, -1, np.int64)
        self.off = np.zeros(cap, np.int64)
        self.root = np.full(cap, -1, np.int64)
        self.d = np.full(cap, np.inf, np.float64)
        self.chosen = np.zeros(cap, np.int64)
        self.depth = np.zeros(cap, np.int64)
        self.born = np.full(cap, -1, np.int8)
        self.fin = np.zeros(cap, bool)
        if old is not None:
            for name, arr in old.items():
                getattr(self, name)[:self.n] = arr[:self.n]

    def _cols(self) -> dict:
        return {k: getattr(self, k) for k in ("key", "state", "obs", "parent", "run", "off",
                                              "root", "d", "chosen", "depth", "born", "fin")}

    def add(self, key: int, state, obs_row, parent: int, run: int, off: int, root: int,
            d: float, depth: int, born: int, fin: bool = False) -> int:
        if self.n >= self.cap:
            self._alloc(self.cap * 2)
        i = self.n
        self.key[i] = int(key)
        self.state[i] = state
        if obs_row is not None:
            self.obs[i] = np.asarray(obs_row, np.float32).reshape(-1)[:self.obs_dim]
        self.parent[i] = int(parent)
        self.run[i] = int(run)
        self.off[i] = int(off)
        self.root[i] = int(root)
        self.d[i] = float(d)
        self.depth[i] = int(depth)
        self.born[i] = int(born)
        self.fin[i] = bool(fin)
        self.n += 1
        if not fin:
            self.index[int(key)] = i
        return i

    def select(self, rng, m: int) -> np.ndarray:
        """``m`` cells drawn with W = 1 / sqrt(1 + chosen) (Nature, Methods eq. 4, with the
        times-chosen count); finishing entries are never explored from."""
        n = self.n
        w = 1.0 / np.sqrt(1.0 + self.chosen[:n].astype(np.float64))
        w[self.fin[:n]] = 0.0
        c = np.cumsum(w)
        if c[-1] <= 0.0:
            return np.zeros(int(m), np.int64)
        u = rng.random(int(m)) * c[-1]
        return np.minimum(np.searchsorted(c, u, side="right"), n - 1).astype(np.int64)

    def chain(self, i: int) -> list:
        """Cell indices from the root down to cell ``i`` (inclusive)."""
        out = [int(i)]
        while self.parent[out[-1]] >= 0:
            out.append(int(self.parent[out[-1]]))
            if len(out) > self.n + 1:
                raise RuntimeError("archive parent cycle")
        return out[::-1]

    def best(self) -> int:
        """The deepest entry: a finish first (its depth = ticks from the line start, the
        fastest), else the smallest field distance (ties: the shallowest chain); -1 if none
        is on the field."""
        n = self.n
        if n == 0:
            return -1
        fin = np.flatnonzero(self.fin[:n])
        if len(fin):
            return int(fin[np.argmin(self.depth[fin])])
        d = self.d[:n]
        if not np.isfinite(d).any():
            return -1
        order = np.lexsort((self.depth[:n], d))
        return int(order[0])


def contact_mask(onground, vz_before, vz_after, g_step: float, tol: float = 1.0) -> np.ndarray:
    """The states the map HOLDS: on walkable ground, or a tick whose vz change departs upward
    from free fall (a surf face pushed back; a jump off the ground is ground). ``g_step`` is
    the engine's per-tick vz change in free flight (negative: -8 at 800 / 10 ms)."""
    dvz = np.asarray(vz_after, np.float64) - np.asarray(vz_before, np.float64)
    return (np.asarray(onground) >= 0) | (dvz - float(g_step) > float(tol))


def stall_index(d: np.ndarray, margin: float) -> int:
    """The first index within ``margin`` of the line's best d (auto_spine.stall_tick)."""
    d = np.asarray(d, np.float64)
    if not np.isfinite(d).any():
        return len(d) - 1
    best = float(np.nanmin(np.where(np.isfinite(d), d, np.nan)))
    ok = np.flatnonzero(d <= best + float(margin))
    return int(ok[0]) if len(ok) else int(np.argmin(d))


def root_indices(n: int, stall: int, tps: float, secs: float, stride: int) -> np.ndarray:
    """Line indices in the last ``secs`` seconds up to ``stall``, every ``stride`` ticks
    (always including ``stall`` itself)."""
    stall = int(min(max(stall, 0), n - 1))
    lo = max(0, stall - int(round(float(secs) * float(tps))))
    idx = np.arange(lo, stall + 1, max(1, int(stride)), dtype=np.int64)
    if len(idx) == 0 or idx[-1] != stall:
        idx = np.append(idx, stall)
    return idx


def link_ticks(acts: np.ndarray, view, off: int, k: int):
    """The per-TICK engine rows (and view rows) of the first ``off`` ticks of a run whose
    per-DECISION tables are ``acts`` (D, 6) / ``view`` (D, 2) (None: a discrete line)."""
    a = np.repeat(np.asarray(acts, np.int32).reshape(-1, 6), int(k), axis=0)[:int(off)]
    v = (None if view is None else
         np.repeat(np.asarray(view, np.float32).reshape(-1, 2), int(k), axis=0)[:int(off)])
    if len(a) < int(off):
        raise ValueError(f"link needs {off} ticks, the run holds {len(a)}")
    return a, v


# ---------------------------------------------------------------------------
# the search (inside tools/beam_tas.py --ge-knob)
# ---------------------------------------------------------------------------


def knob_ge_search(args, *, cfg, map_path, ep_cap, tick, arm, K, VIEWC, VIEW_ABS, coreN, spol,
                   obs, ogf, core1, greedy_wrapper, row0, obs_start, gate_seed, out_dir,
                   build_sim, gravity_step, neutral_act, nvec, off_warp, state_dtype) -> int:
    """Run the knob Go-Explore and write ``knob_line.npy`` (the greedy line up to the root of
    the deepest cell's chain + the chain's replayed per-tick states), ``knob_summary.json`` and
    ``knob_log.jsonl`` (one line per report: cells, runs, lambda, yields, best d) into
    ``out_dir``. -> 0 (a line was written) or 2 (no cell beat the line's own best)."""
    from explore_phase1 import CellHash
    if VIEWC and VIEW_ABS not in ("velocity", "world"):
        raise SystemExit("--ge-knob draws its random yaw as a velocity-frame / world TARGET: a "
                         "delta-view checkpoint is not supported")
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed) + 9173)
    N = int(coreN.num_envs)
    tps = 1000.0 / float(tick.ms)
    g_step = float(gravity_step(coreN))
    tol = float(args.ge_knob_contact_tol)

    def field_d(origins):
        o = np.asarray(origins, np.float64).reshape(-1, 3)
        d = np.asarray(ogf.sample(o), np.float64)
        return np.where(d < float(ogf.reach_max) - 0.5 * float(ogf.cell), d, np.inf)

    # ---- 1. the greedy line from the TRUE start (the roots, and the spine's prefix) -------
    core1.reset(int(gate_seed))
    core1.set_state(0, row0)
    o1 = np.array(obs_start, np.float32).reshape(1, -1)
    L_states, L_obs = [], []
    for _t in range(int(args.max_ticks)):
        L_states.append(core1.get_states()[0].copy())
        L_obs.append(o1[0].copy())
        a = greedy_wrapper.act(o1)
        v = getattr(greedy_wrapper, "view", None)
        if v is None:
            o1, _r, done, trunc, _ = core1.step(a)
        else:
            o1, _r, done, trunc, _ = core1.step(a, view=v)
        o1 = np.array(o1, np.float32)
        if bool(np.asarray(done)[0]) or bool(np.asarray(trunc)[0]):
            break
    line = np.array(L_states, dtype=state_dtype)
    line_obs = np.array(L_obs, np.float32)
    ld = field_d(line["origin"])
    stall = stall_index(ld, float(args.ge_knob_stall_margin))
    line_best = float(np.min(ld)) if np.isfinite(ld).any() else float("inf")
    roots = root_indices(len(line), stall, tps, float(args.ge_knob_roots_secs), int(K))
    print(f"knob-GE: greedy line {len(line)} ticks ({len(line) / tps:.1f} s), best d "
          f"{line_best:,.0f} at {int(np.argmin(ld))}, stall {stall} (within "
          f"{float(args.ge_knob_stall_margin):g} u); {len(roots)} roots over "
          f"[{int(roots[0])}, {int(roots[-1])}]", flush=True)

    # ---- 2. the archive -------------------------------------------------------------------
    mins, maxs = coreN.map_bounds()
    hasher = CellHash(mins, maxs, cell=float(args.ge_knob_cell),
                      energy=float(args.ge_knob_energy), heading=int(args.ge_knob_heading))
    arch = KnobArchive(int(np.asarray(obs).shape[1]), state_dtype)
    rkeys = hasher.keys(line[roots])
    for r, kk in zip(roots, rkeys):
        if int(kk) in arch.index:
            continue
        arch.add(int(kk), line[int(r)], line_obs[int(r)], -1, -1, 0, int(r), float(ld[int(r)]),
                 int(r), -1)
    n_roots = arch.n
    print(f"knob-GE: {n_roots} root cells (cell {float(args.ge_knob_cell):g} u, kinetic-energy "
          f"buckets {float(args.ge_knob_energy):g} u, heading sectors "
          f"{int(args.ge_knob_heading)}); runs of {int(args.ge_knob_decisions)} decisions, "
          f"random repeat {float(args.ge_knob_repeat):g}, knob "
          + (f"FIXED at {float(args.ge_knob_lambda):g}" if args.ge_knob_lambda is not None
             else f"by yield (floor {float(args.ge_knob_floor):g})"), flush=True)

    knob = Knob(horizon=int(args.ge_knob_horizon) * N, floor=float(args.ge_knob_floor),
                warm=int(args.ge_knob_warm), fixed=args.ge_knob_lambda)
    L = int(args.ge_knob_decisions)
    rep = float(args.ge_knob_repeat)
    run_cell = np.full(N, -1, np.int64)
    run_id = np.full(N, -1, np.int64)
    run_mode = np.zeros(N, np.int64)
    run_dec = np.zeros(N, np.int64)
    run_ticks = np.zeros(N, np.int64)
    run_new = np.zeros(N, np.int64)
    run_live = np.zeros(N, bool)
    run_acts = np.zeros((N, L, 6), np.int8)
    run_view = np.zeros((N, L, 2), np.float32)
    store: dict[int, tuple] = {}
    next_run = 0
    rnd_a = np.zeros((N, 6), np.int32)
    rnd_yaw = np.zeros(N, np.float32)
    held_a = held_v = None
    obs = np.array(obs, np.float32)
    lam_hist = []
    log_f = open(out_dir / "knob_log.jsonl", "w", encoding="utf-8")
    deadline = t_start + 60.0 * float(args.ge_knob_minutes)
    best_seen = float(np.min(arch.d[:arch.n])) if arch.n else float("inf")
    t = 0
    last_rep = time.time()
    n_new_since = 0
    born_cells = np.zeros(2, np.int64)

    def draw_random(idx):
        m = len(idx)
        if m == 0:
            return
        rnd_a[idx, 0] = neutral_act[0] if VIEWC else rng.integers(0, nvec[0], m)
        rnd_a[idx, 1] = neutral_act[1]
        for j in (2, 3, 4, 5):
            rnd_a[idx, j] = rng.integers(0, nvec[j], m)
        if VIEWC:
            u = np.clip(rng.uniform(-1.0, 1.0, m), -0.999, 0.999)
            rnd_yaw[idx] = (off_warp(u) if VIEW_ABS == "velocity"
                            else rng.uniform(-180.0, 180.0, m)).astype(np.float32)

    while True:
        now = time.time()
        if t % K == 0:
            if now >= deadline:
                break
            # ---- run ends: credit the knob, keep the actions of runs that added a cell ----
            ended = np.flatnonzero((run_id >= 0) & ((~run_live) | (run_dec >= L)))
            if len(ended):
                knob.credit(run_mode[ended], run_new[ended], np.maximum(run_dec[ended], 1))
                for i in ended[run_new[ended] > 0]:
                    store[int(run_id[i])] = (run_acts[i, :run_dec[i]].copy(),
                                             run_view[i, :run_dec[i]].copy() if VIEWC else None)
                run_id[ended] = -1
            # ---- returns: restore W-chosen cells into every free env ----------------------
            free = np.flatnonzero(run_id < 0)
            if len(free):
                pick = arch.select(rng, len(free))
                np.add.at(arch.chosen, pick, 1)
                for i, c in zip(free, pick):
                    st = arch.state[int(c):int(c) + 1].copy()
                    st["tick"] = 0
                    st["stuck_ticks"] = 0
                    coreN.set_state(int(i), st)
                    obs[int(i)] = arch.obs[int(c)]
                run_cell[free] = pick
                run_id[free] = next_run + np.arange(len(free))
                next_run += len(free)
                lam = knob.lam
                run_mode[free] = (rng.random(len(free)) < lam).astype(np.int64)
                run_dec[free] = 0
                run_ticks[free] = 0
                run_new[free] = 0
                run_live[free] = True
                draw_random(free[run_mode[free] == 1])
        a = spol.act(obs)
        v = getattr(spol, "view", None)
        if t % K == 0:
            held_a = np.array(a, np.int32, copy=True)
            held_v = None if v is None else np.array(v, np.float32, copy=True)
            rm = run_live & (run_mode == 1)
            redraw = np.flatnonzero(rm & (run_dec > 0) & (rng.random(N) >= rep))
            draw_random(redraw)
            held_a[rm] = rnd_a[rm]
            if held_v is not None:
                held_v[rm, 0] = rnd_yaw[rm]
            rec = np.flatnonzero(run_live & (run_dec < L))
            run_acts[rec, run_dec[rec]] = held_a[rec]
            if held_v is not None:
                run_view[rec, run_dec[rec]] = held_v[rec]
            run_dec[rec] += 1
        sv = coreN.states_view
        vz0 = np.array(sv["velocity"][:, 2], np.float64)
        pre = coreN.get_states()
        if held_v is None:
            obs, _r, done, trunc, _ = coreN.step(held_a)
        else:
            obs, _r, done, trunc, _ = coreN.step(held_a, view=held_v)
        obs = np.array(obs, np.float32)
        t += 1
        done = np.asarray(done, bool)
        trunc = np.asarray(trunc, bool)
        gh = np.asarray(coreN.goal_hits, bool)
        dead = done | trunc
        live = run_live & ~dead
        run_ticks[run_live] += 1
        # ---- finishes: the pre-finish state ends the chain (d = 0) ------------------------
        for i in np.flatnonzero(run_live & gh):
            c = int(run_cell[i])
            arch.add(-1, pre[i], None, c, int(run_id[i]), int(run_ticks[i]), -1, 0.0,
                     int(arch.depth[c] + run_ticks[i]), int(run_mode[i]), fin=True)
            run_new[i] += 1
            born_cells[int(run_mode[i])] += 1
            n_new_since += 1
        run_live &= ~dead
        # ---- new cells: contact states only --------------------------------------------
        if live.any():
            sv = coreN.states_view
            li = np.flatnonzero(live)
            ct = contact_mask(np.asarray(sv["onground"])[li], vz0[li],
                              np.asarray(sv["velocity"][li, 2], np.float64), g_step, tol)
            li = li[ct]
            if len(li):
                st = coreN.get_states()[li]
                keys = hasher.keys(st)
                newm = np.fromiter((int(k_) not in arch.index for k_ in keys), bool, len(li))
                if newm.any():
                    ni = li[newm]
                    nst = st[newm]
                    nk = keys[newm]
                    nd = field_d(nst["origin"])
                    for j, i in enumerate(ni):
                        if int(nk[j]) in arch.index:      # two envs into one new cell
                            continue
                        c = int(run_cell[i])
                        arch.add(int(nk[j]), nst[j], obs[i], c, int(run_id[i]),
                                 int(run_ticks[i]), -1, float(nd[j]),
                                 int(arch.depth[c] + run_ticks[i]), int(run_mode[i]))
                        run_new[i] += 1
                        born_cells[int(run_mode[i])] += 1
                        n_new_since += 1
        if time.time() - last_rep >= float(args.ge_knob_report_secs):
            last_rep = time.time()
            b = arch.best()
            bd = float(arch.d[b]) if b >= 0 else float("inf")
            yp, yr = knob.yields()
            rec_ = {"t": round(time.time() - t_start, 1), "ticks": int(t), "cells": int(arch.n),
                    "runs": int(next_run), "lam": round(knob.lam, 4), "y_pol": yp, "y_rand": yr,
                    "ends_pol": int(knob.ends[0]), "ends_rand": int(knob.ends[1]),
                    "cells_pol": int(born_cells[0]), "cells_rand": int(born_cells[1]),
                    "best_d": bd, "line_best_d": line_best, "finishes": int(arch.fin[:arch.n].sum()),
                    "new": int(n_new_since)}
            n_new_since = 0
            lam_hist.append(rec_)
            log_f.write(json.dumps(rec_) + "\n")
            log_f.flush()
            print(f"knob-GE {rec_['t']:7.1f}s  cells {arch.n:,}  runs {next_run:,}  lambda "
                  f"{knob.lam:.3f}  yield pol {yp:.4f} rand {yr:.4f} /dec  born pol "
                  f"{born_cells[0]:,} rand {born_cells[1]:,}  best d {bd:,.0f} (line "
                  f"{line_best:,.0f})", flush=True)
    log_f.close()
    # the runs still in flight keep their actions too (their cells may be the best)
    for i in np.flatnonzero((run_id >= 0) & (run_new > 0)):
        store[int(run_id[i])] = (run_acts[i, :run_dec[i]].copy(),
                                 run_view[i, :run_dec[i]].copy() if VIEWC else None)

    # ---- 3. the deepest chain, replayed exactly from its root ------------------------------
    b = arch.best()
    summary = {"cells": int(arch.n), "roots": int(n_roots), "runs": int(next_run),
               "ticks": int(t), "secs": round(time.time() - t_start, 1),
               "lam_final": float(knob.lam), "ends": [int(x) for x in knob.ends],
               "cells_by_mode": [int(x) for x in born_cells],
               "decisions_by_mode": [int(x) for x in knob.decs],
               "line_len": int(len(line)), "line_best_d": line_best, "stall": int(stall),
               "best": int(b), "best_d": float(arch.d[b]) if b >= 0 else None,
               "finishes": int(arch.fin[:arch.n].sum()), "lam_hist": lam_hist[-200:],
               "replay_ok": None, "line": None}
    ok_line = b >= 0 and (arch.fin[b] or float(arch.d[b]) < line_best)
    if not ok_line:
        print(f"knob-GE: no cell beats the greedy line's own best d {line_best:,.0f}", flush=True)
        (out_dir / "knob_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        return 2
    ch = arch.chain(b)
    r0 = int(arch.root[ch[0]])
    core_r = build_sim(cfg, map_path, 1, 10 ** 7, tick=tick)
    arm(core_r)
    core_r.reset(int(gate_seed))
    states = [line[r0].copy()]
    worst = 0.0
    replay_ok = True
    for c in ch[1:]:
        # every junction restores like the search did: the parent's state as REACHED by the
        # replay, its clock and stuck counter zeroed (the search restored every cell so)
        st = np.array(states[-1:], dtype=state_dtype, copy=True)
        st["tick"] = 0
        st["stuck_ticks"] = 0
        core_r.set_state(0, st)
        acts, view = store[int(arch.run[c])]
        at, vt = link_ticks(acts, view, int(arch.off[c]), K)
        for j in range(len(at)):
            aa = np.ascontiguousarray(at[j].reshape(1, 6), dtype=np.int32)
            if vt is None:
                _o, _r, done, trunc, _ = core_r.step(aa)
            else:
                _o, _r, done, trunc, _ = core_r.step(
                    aa, view=np.ascontiguousarray(vt[j].reshape(1, 2), dtype=np.float32))
            last = j == len(at) - 1
            ended = bool(np.asarray(done)[0]) or bool(np.asarray(trunc)[0])
            if last and arch.fin[c]:
                # the chain's finish: the last tick must cross the goal
                if not (ended and bool(np.asarray(core_r.goal_hits)[0])):
                    replay_ok = False
                break
            if ended:
                replay_ok = False
                break
            states.append(core_r.get_states()[0].copy())
        if not replay_ok:
            break
        if not arch.fin[c]:
            err = float(np.linalg.norm(np.asarray(states[-1]["origin"], np.float64)
                                       - np.asarray(arch.state[c]["origin"], np.float64)))
            worst = max(worst, err)
            if err > 1.0:
                replay_ok = False
                break
    chain_states = np.array(states[1:], dtype=state_dtype)
    full = np.concatenate([np.asarray(line[:r0 + 1], state_dtype), chain_states])
    fd = field_d(full["origin"])
    np.save(out_dir / "knob_line.npy", full)
    summary.update(replay_ok=bool(replay_ok), replay_worst_err=worst, root_index=r0,
                   chain_cells=len(ch), chain_ticks=int(len(chain_states)),
                   line=str(out_dir / "knob_line.npy"), line_n=int(len(full)),
                   line_end_d=float(fd[-1]) if len(fd) else None,
                   line_min_d=float(np.min(fd)) if len(fd) else None,
                   chain_modes=[int(arch.born[c]) for c in ch[1:]])
    (out_dir / "knob_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"knob-GE: best cell {b} d {float(arch.d[b]):,.0f} (line {line_best:,.0f}), chain "
          f"{len(ch) - 1} links from root {r0} ({len(chain_states)} ticks, modes "
          f"{summary['chain_modes']}); replay {'OK' if replay_ok else 'FAILED'} (worst "
          f"junction error {worst:.3g} u) -> {out_dir / 'knob_line.npy'} ({len(full)} states, "
          f"end d {summary['line_end_d']:,.0f})", flush=True)
    return 0 if replay_ok else 3
