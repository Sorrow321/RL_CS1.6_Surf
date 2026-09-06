"""Policy-guided population search ("beam TAS") for a record run.

Runs N stochastic copies of a trained race checkpoint's policy in the
batched sim, all starting from ONE identical spawn state, and every R
decisions clones the leaders (ranked by geodesic distance-to-goal) over
the laggards via ``core.set_state`` - carrying each survivor's full
action history with it. Because the whole population runs in lockstep
wall-ticks, every valid env's elapsed episode time equals the global
tick, so the FIRST goal crossing is also the fastest run the search can
ever produce; later crossings only add to the finisher count. The
winning action sequence is then replayed open-loop (no policy) on a
fresh 1-env core from the same spawn state and must reproduce the finish
deterministically, tick for tick - the script asserts the finish tick
and the pre-finish origin/velocity are bit-identical.

This is the deterministic special case of MCTS: the policy is the
proposal distribution, the real simulator is the model, and truncation
selection replaces UCB.

Two search modes share every other phase:
* default: the v1 lockstep population search (cloning every
  --resample-every decisions);
* --commit H: receding-horizon (MPC) mode - windows of H decisions with
  NO intra-window cloning (maximal proposal width), and at each boundary
  the single best lineage's first --commit-frac*H decisions are
  committed and the whole population re-centers on that state
  (see commit_search). --eps adds per-head epsilon-uniform proposal
  mixing in either mode; 0 keeps the eps-free RNG stream byte-identical.

Checkpoint-faithful loading (config handling, Policy construction,
GpuLidar setup, act_every hold, the --obs-reward slot-12 feed) mirrors
tools/record_ckpt.py, and the config audit is IMPORTED from it, so this
tool inherits its refuse-on-unknown-keys safety - every misleading
recording ever shipped came from hand-rolled loading. Config knobs that
carry PER-ENV INFERENCE STATE this tool cannot clone across a set_state
(frame ring, chunk plan, route file) are refused outright; the
--race-latch flag and the --obs-reward feed ARE cloned with the state.

Expert iteration (tools/expert_loop.py) adds three things: --greedy-envs
(a greedy floor: envs [0, G) act greedily on the shared forward, env 0 is
never cloned over, so a wave cannot lose to the policy's own mode),
--score v/dv (rank by the checkpoint's critic instead of the goal field
near the finish, where d selects the dive) and --keep-finishers (the K
fastest distinct finishing lineages' action tables in beam_best.npz, plus
the spawn's obs row, for tools/plan_to_bc.py).

From-scratch expert iteration adds --objective progress|auto: a policy
that finishes nothing needs a planner that does not need finishes. The
honest coordinate is the ORDER-ONLY corridor arc along the champion route
line (tools/eval_honesty.py --order-only 16; surfgym.route.ArcProgress,
the --race-arc reward's own rule, run per env per tick and cloned with
the state at every resample). Elites are the lineages with the furthest
BEST ARC (ties by the --score value, so the critic tie-break stays
available; --arc-quant widens "tie"); a lineage that dies keeps its best
arc and its action table in a bounded hall of the K best; the final
ranking is finishers first by time (auto), then best arc descending with
the tick it was reached at ascending. The best-arc line is replayed
open-loop and its arc must reproduce, like a finisher's finish tick.
beam_best.npz then carries arc_all / arc_tick_all / end_tick_all beside
acts_all for tools/plan_to_bc.py, which trims each line at its last map
contact before distilling it. --objective finish (default) is byte-
identical to the tool before this paragraph.

Two per-env physics fields live OUTSIDE SurfState and are NOT copied by
set_state: consumed push-once trigger flags (``once_used``) and stuck-
nudge bookkeeping (``PmPersist``). surf_src_cannonball has zero
trigger_push entities, and a run that engages the stuck nudge is five
ticks from a fail anyway; the replay assert is the backstop that would
catch either. Same limitation as the trainer's own reservoir respawns.

The traj jsonl rows match surfgym.record byte-for-byte; the trailer
"end" label here is derived from the core's goal_hits flag (honest),
not from record.py's +50-reward heuristic, which mislabels a goal-box
finish as "fail" because the record path sets no waypoints and the core
therefore emits reward 0 on completion.

--tick-ms runs the whole search at another physics tick (surfgym.tick:
7.63 -> the [8, 8, 7] ms pattern, 130.4 Hz); the default is the
checkpoint's own tick (10 for every checkpoint that predates the flag) -
physics parity, like record_ckpt. Every COUNT in here stays in ticks
(--max-ticks, --resample-every x act_every, --greedy-prefix, the commit
windows, the finish tick) and every printed or saved SECOND is the real
time under the pattern (surfgym.tick.ticks_to_secs: [8, 8, 7] summed
from phase 0, exact); beam_best.npz, summary.json and every trajectory
header carry tick_ms + tick_pattern_ms the way record_rollout stamps an
episode, so no reader has to assume 10 ms. At 10 ms the tool is
byte-identical to the tool before the flag. --act-every overrides the
decision interval in ticks (K=4 at 7.67 ms keeps the 30 ms the weights
learned at K=3 / 10 ms); the npz records the K used.

--macro-hold MIN:MAX makes the PROPOSAL a held-key macro instead of a
per-decision draw from the policy: each non-greedy env holds one drawn
(side, forward) pair for a log-uniform duration in [MIN, MAX] seconds
(rounded to whole decisions), with the yaw still the policy's or, under
--macro-yaw track, set analytically to the bin that tracks the velocity
rotation the held key produces. Off by default and byte-identical when
off; the draws come from a private numpy generator seeded from --seed, so
the torch proposal stream is untouched. The winner's cadence (A/D flips
per second, median held-key run, the share of free-flight ticks whose
wishdir is within 0.5 deg of perpendicular to velocity, and the net
free-flight change of 0.5|v_h|^2) is printed and written into
summary.json under "cadence", with or without the flag.

Usage:
    python tools/beam_tas.py                      # F_prime, full search
    python tools/beam_tas.py --greedy-only        # just the sanity gate
    python tools/beam_tas.py CKPT --envs 1024 --resample-every 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch

from eval_honesty import corridor_progress, load_route
from surfgym import SurfCore, default_config
from surfgym.bc import make_eval_feeds, rank_lineages
from surfgym.core import SURF_IN_DUCK, SURF_IN_JUMP, phys_to_dict
from surfgym.rewards import map_spawn_pool
from surfgym.route import ArcProgress
from surfgym.tick import TickClock, header_fields, ticks_to_secs
from surfgym.view import K_MAX, bin_to_view, warp
from train_fast import (A_FWD_NONE, H_FWD, H_SIDE, H_YAW, N_VIEW, NVEC,
                        NEUTRAL_ACT, GreedyTorchPolicy, HeadPacker, Policy,
                        SampledTorchPolicy, sample_padded, sample_view,
                        split_view)
import record_ckpt as _rc   # audit_cfg: inherit refuse-on-unknown-keys

DEF_CKPT = "C:/RL_Surf/runs/frozen/F_prime.pt"
# ABSOLUTE main-checkout maps dir. The worktree's maps/ is a COPY with
# different mtimes, and every cache (goal field, SDF, occ) keys on
# size+mtime_ns of the bsp - resolving the map inside the worktree
# silently triggers a ~30-minute goal-field re-bake (CLAUDE.md).
MAIN_MAPS = Path("C:/RL_Surf/maps")

# Config knobs that add PER-ENV inference state (frame ring, chunk plan)
# or need a side file (route). Cloning an env mid-episode must clone that
# state too; v1 does not implement it, so it refuses rather than run with
# silently wrong semantics. The --race-latch flag IS cloned (its per-env
# state is one bool + one tick per env: surfgym.bc.make_eval_feeds owns
# it, and every set_state below copies donor -> loser or snapshot -> all),
# so the xQR32 / xSTACK finishers can be planned with. --race-arc is
# refused: the slot-12 mirror here is the geodesic term.
UNSUPPORTED = ("route_file", "chunk", "frame_stack", "race_arc")


def resolve_map(name_or_path, cfg_map):
    p = str(name_or_path or cfg_map)
    if p.lower().endswith(".bsp"):
        return p
    for base in (MAIN_MAPS, ROOT / "maps"):
        c = base / (p + ".bsp")
        if c.exists():
            if base != MAIN_MAPS:
                print(f"WARNING: {c} is not the main checkout - a stale "
                      f"mtime here re-bakes every cache")
            return str(c)
    raise SystemExit(f"map not found: {p!r}")


def build_sim(cfg, map_path, num_envs, ep_cap, tick=None):
    """Physics core exactly as record_ckpt.py builds it (eyeless; vision
    is GPU-side). ``tick`` (a surfgym.tick.TickClock) runs it at another
    physics tick the way record_ckpt --tick-ms does: the view rates are
    deg PER TICK in the core, so both are rescaled to keep the same deg
    per SECOND. None / the 10 ms reference builds exactly the core built
    before the flag (no pattern, nothing rescaled)."""
    fix_pitch = cfg.get("fix_pitch")
    pitch_rate = 0.0 if fix_pitch is not None else float(
        cfg.get("pitch_rate", -1.0))
    if tick is None or tick.is_reference:
        pitch_rate_core, _tick_env, tick_ms = pitch_rate, {}, None
    else:
        pitch_rate_core = (0.0 if pitch_rate == 0.0 else
                           tick.per_tick(10.0 if pitch_rate < 0 else pitch_rate))
        # --yaw-adaptive redefines a yaw bin as K_BINS * atan(30/|v|) - the
        # optimal-strafe angle per FRAME, which does NOT depend on the tick.
        # yaw_rate_max_deg is then only (a) a per-tick clamp and (b) the
        # divisor of obs column 10 (env.c: last_yd / yaw_rate_max_deg), so
        # scaling it would cut the clamp to 0.767x - the +-20 / +-8 bins the
        # search proposes sit ON that clamp at surf speeds, i.e. the search's
        # own action space would change - and inflate the action echo the
        # policy reads by 1.304x. Fixed bins scale; adaptive bins keep the
        # reference ceiling (record_ckpt / train_fast, the tick-ms review).
        _tick_env = ({} if cfg.get("yaw_adaptive")
                     else {"yaw_rate_max_deg": tick.per_tick(10.0)})
        tick_ms = tick.requested_ms
    return SurfCore(map_path, default_config(
        num_envs=num_envs, spawn_mode=2, max_episode_ticks=ep_cap,
        water_fail=1,
        sv_maxvelocity=float(cfg.get("maxvel", 2000.0)),
        yaw_adaptive=1 if cfg.get("yaw_adaptive") else 0,
        yaw_blend=float(cfg.get("yaw_blend") or 1.0),
        side_hold_ticks=int(cfg.get("side_hold") or 0),
        lidar_w=0, lidar_h=0,
        pitch_rate_max_deg=pitch_rate_core, **_tick_env), tick_ms=tick_ms)


def tick_header(core):
    """The tick keys of an episode header, as record_rollout writes them:
    the plain ``"tick_ms": 10`` at the reference tick (byte-identical), the
    mean + ``tick_pattern_ms`` + ``tick_phase`` under a --tick-ms pattern."""
    pat = tuple(getattr(core, "tick_pattern", (int(core.config.phys.msec),)))
    return header_fields(float(sum(pat)) / len(pat), pat,
                         int(getattr(core, "tick_phase", 0)))


def phys_header(core):
    """The ``phys`` block of an episode header with ``msec`` = the core's
    NOMINAL tick. ``config.phys.msec`` is a MOVING value under a --tick-ms
    pattern (step() mirrors the tick it just ran into it: 8, 8, 7, ...), so
    a snapshot of it says whatever phase the core happens to be in. The
    nominal tick is ``SurfCore.nominal_msec`` where the core has it
    (branch tick-consumers), else the pattern's first element (what
    reset() restores), else the config tick; at a fixed tick every branch
    is the config value, so the 10 ms header is byte-identical."""
    d = phys_to_dict(core.config.phys)
    nominal = getattr(core, "nominal_msec", None)
    if nominal is None:
        nominal = getattr(core, "_base_msec", None)
    if nominal is None:
        pat = getattr(core, "tick_pattern", None)
        nominal = int(pat[0]) if pat else int(d.get("msec", 10))
    d["msec"] = int(nominal)
    return d


def tick_stamp(tick, cfg_tick=None):
    """The tick keys of summary.json / beam_best.npz: ``tick_ms`` (the
    mean, ms), ``tick_pattern_ms``, ``tick_ms_requested`` (the flag) and,
    under an override, ``tick_ms_ckpt`` (the tick the weights trained at)
    - record_ckpt's header bookkeeping, so a reader never assumes 10 ms."""
    d = {"tick_ms": float(tick.ms),
         "tick_pattern_ms": [int(v) for v in tick.pattern],
         "tick_ms_requested": float(tick.requested_ms)}
    if cfg_tick is not None and abs(float(cfg_tick) - tick.requested_ms) > 1e-9:
        d["tick_ms_ckpt"] = float(cfg_tick)
    return d


def tick_npz(tick, cfg_tick=None):
    """tick_stamp as np.savez fields (int32 pattern, float64 ticks)."""
    return {k: (np.asarray(v, np.int32) if k == "tick_pattern_ms"
                else np.float64(v))
            for k, v in tick_stamp(tick, cfg_tick).items()}


def run_episode(core, pol, obs, fout, max_ticks, header, episode_idx):
    """Roll env 0 ONE episode from the core's current state, writing traj
    rows in surfgym.record's exact format. ``pol`` is an object with
    ``act(obs) -> (N, 6) int32`` and, under --view-continuous, a ``view``
    attribute ((N, 2) float32 or None) read after each act - the policy
    wrappers and Playback both have it. Returns
    (end, ticks, finished, pre_finish_state)."""
    if "tick_pattern_ms" in header:
        # a --tick-ms pattern: where in it this episode's first row lands
        # (0 after a reset) so its duration sums exactly (record_rollout)
        header = {**header, "tick_phase": int(getattr(core, "tick_phase", 0))}
    fout.write(json.dumps({**header, "episode": episode_idx},
                          separators=(",", ":")) + "\n")
    ep_ticks, best_progress = 0, 0.0
    finished, end, pre_state = False, "trunc", None
    for _ in range(max_ticks):
        s0 = core.get_states()[0]           # pre-step snapshot (copy)
        actions = pol.act(obs)
        view = getattr(pol, "view", None)
        if view is None:
            obs, rew, done, trunc, _term = core.step(actions)
        else:
            obs, rew, done, trunc, _term = core.step(actions, view=view)
        r0 = float(rew[0])
        best_progress = max(best_progress, float(s0["best_progress"]),
                            float(s0["progress"]))
        buttons = (SURF_IN_JUMP if actions[0, 4] else 0) | (
            SURF_IN_DUCK if actions[0, 5] else 0)
        ox, oy, oz = (float(v) for v in s0["origin"])
        vx, vy, vz = (float(v) for v in s0["velocity"])
        line = [ep_ticks,
                round(ox, 2), round(oy, 2), round(oz, 2),
                round(vx, 2), round(vy, 2), round(vz, 2),
                round(float(s0["yaw"]), 2),
                int(buttons),
                int(int(s0["onground"]) >= 0),
                round(float(s0["progress"]), 2),
                round(r0, 5),
                round(float(s0["pitch"]), 2),
                int(actions[0, 2]), int(actions[0, 3])]
        fout.write(json.dumps(line, separators=(",", ":")) + "\n")
        ep_ticks += 1
        if done[0] or trunc[0]:
            finished = bool(core.goal_hits[0])
            end = "done" if finished else ("fail" if done[0] else "trunc")
            pre_state = s0
            break
    fout.write(json.dumps({"end": end, "ticks": ep_ticks,
                           "best_progress": round(best_progress, 2)},
                          separators=(",", ":")) + "\n")
    fout.flush()
    return end, ep_ticks, finished, pre_state


class Playback:
    """Open-loop replay: feed a recorded per-tick action sequence, ignore
    the observation entirely. No policy, no GPU. ``view_ticks`` (T, 2)
    float32 is a --view-continuous line's per-tick view command, published
    as ``self.view`` next to each action row (None for a discrete line)."""

    def __init__(self, seq_ticks, view_ticks=None):
        self.seq, self.t = seq_ticks, 0
        self.vseq = (None if view_ticks is None
                     else np.asarray(view_ticks, np.float32).reshape(-1, 2))
        self.view = None

    def act(self, _obs):
        i = min(self.t, len(self.seq) - 1)
        a = self.seq[i]
        if self.vseq is not None:
            self.view = np.ascontiguousarray(self.vseq[i].reshape(1, 2),
                                             dtype=np.float32)
        self.t += 1
        return np.ascontiguousarray(a.reshape(1, 6), dtype=np.int32)


def line_view_ticks(view, k: int):
    """A (D, 2) per-decision view table -> its per-TICK (D*k, 2) float32
    copy (np.repeat, exactly as the action table is expanded); None -> None."""
    if view is None:
        return None
    return np.ascontiguousarray(
        np.repeat(np.asarray(view, np.float32).reshape(-1, 2), int(k), axis=0))


class LineageHall:
    """The K best non-finishing lineages seen so far, by (best arc DESC,
    arc tick ASC). ``offer`` takes a thunk for the action table so the
    (D, 6) copy is only made for a candidate that would actually enter -
    a from-scratch population loses hundreds of lineages per generation
    and copying every one of them would dominate the search. Byte-
    identical tables (clones that died before diverging) enter once."""

    def __init__(self, k: int):
        self.k = max(1, int(k))
        self.cap = 4 * self.k
        self.items = []            # dicts: best_arc, arc_tick, end_tick, acts
        self._seen = set()

    def _worst(self):
        return self.items[-1] if self.items else None

    @staticmethod
    def _key(acts, view):
        b = np.ascontiguousarray(acts).tobytes()
        if view is not None:
            b += np.ascontiguousarray(view).tobytes()
        return b

    def offer(self, best_arc, arc_tick, end_tick, get_acts, raw_arc=None,
              get_view=None):
        best_arc, arc_tick = float(best_arc), int(arc_tick)
        w = self._worst()
        if len(self.items) >= self.cap and w is not None and (
                best_arc < w["best_arc"]
                or (best_arc == w["best_arc"] and arc_tick >= w["arc_tick"])):
            return False
        acts = np.ascontiguousarray(np.asarray(get_acts(), np.int8))
        # --view-continuous: the line is its bins AND its float view
        view = (None if get_view is None
                else np.ascontiguousarray(np.asarray(get_view(), np.float32)))
        key = self._key(acts, view)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.items.append({"finish_tick": 0, "best_arc": best_arc,
                           "arc_tick": arc_tick, "end_tick": int(end_tick),
                           "raw_arc": (best_arc if raw_arc is None
                                       else float(raw_arc)),
                           "acts": acts, "view": view})
        self.items.sort(key=lambda c: (-c["best_arc"], c["arc_tick"],
                                       c["end_tick"]))
        if len(self.items) > self.cap:
            for c in self.items[self.cap:]:
                self._seen.discard(self._key(c["acts"], c.get("view")))
            del self.items[self.cap:]
        return True


def gravity_step(core) -> float:
    """The engine's per-tick change of vz in free flight: -sv_gravity *
    tick (pm.c applies it as two half steps; -8.0 u/tick at 800 / 10 ms).
    A tick whose vz change departs from it is a tick the map pushed back.
    Under a --tick-ms pattern this is the MEAN tick's step (-6.13 at
    [8, 8, 7]); the real per-tick step then swings +-0.4 u around it,
    inside the 1 u --contact-tol, so free flight is never read as contact
    and one constant serves the search, the replay and plan_to_bc's trim."""
    ph = core.config.phys
    tick_ms = float(getattr(core, "tick_ms", ph.msec))
    return -float(ph.sv_gravity) * tick_ms / 1000.0


def replay_arc(core, spawn_state, acts_ticks, arcp, max_ticks, g_step,
               contact_tol: float = 1.0, bank: str = "contact",
               view_ticks=None):
    """Open-loop replay of a per-TICK action table on an armed 1-env core,
    scoring the order-only arc exactly as the population search did (reset
    at the spawn, advance on the post-step position of every live tick,
    the arc BANKED at map-contact ticks under bank='contact').
    -> (banked_arc, banked_tick, end_tick, ended, finished, raw_best)."""
    core.set_state(0, spawn_state)
    arcp.reset(np.asarray(core.states_view["origin"], np.float64))
    best, best_t = float(arcp.arc[0]), 0
    banked, banked_t = best, 0
    t, ended, finished = 0, False, False
    for t in range(min(int(max_ticks), len(acts_ticks))):
        a = np.ascontiguousarray(acts_ticks[t].reshape(1, 6), dtype=np.int32)
        pre_vz = float(core.states_view["velocity"][0, 2])
        if view_ticks is None:
            _o, _r, done, trunc, _term = core.step(a)
        else:
            _o, _r, done, trunc, _term = core.step(
                a, view=np.ascontiguousarray(view_ticks[t].reshape(1, 2),
                                             dtype=np.float32))
        if int(core.goal_hits[0]):
            finished = True
        if done[0] or trunc[0] or finished:
            ended = True
            t += 1
            break
        arcp.advance(core.states_view["origin"])
        if float(arcp.arc[0]) > best:
            best, best_t = float(arcp.arc[0]), t + 1
        dvz = float(core.states_view["velocity"][0, 2]) - pre_vz
        if bank != "contact" or abs(dvz - g_step) > contact_tol:
            if best > banked:
                banked, banked_t = best, best_t
        t += 1
    return banked, banked_t, t, ended, finished, best



def robust_rerank(lines, coreN, spawn_state, K, n_per, jitter_u, rng):
    """Re-rank kept lineages by robustness: each line is replayed open-loop
    n_per times on the search's N-env core, half the copies with the spawn
    origin jittered by +-jitter_u (horizontal), half with one 1-tick delay
    inserted at a random tick. Order: finish rate DESC, median finish tick
    ASC, the search's own finish tick ASC. Attaches robust_rate /
    robust_tick to every candidate dict; lines that did not fit into the
    core get rate 0 and sort last. A fragile line (ledger 2026-09-06: the
    68.54 s line dies on a 1 u offset 59% of the time) cannot be followed
    closed-loop; the policy pays ~0.6 s for the margin it needs."""
    N = int(coreN.config.num_envs)
    n_lines = max(1, min(len(lines), N // max(1, n_per)))
    n_per = max(1, min(int(n_per), N // n_lines))
    states = np.repeat(np.asarray(spawn_state).reshape(1), N, axis=0).copy()
    T = max(int(len(c["acts"])) for c in lines[:n_lines]) * int(K)
    acts_ticks = np.zeros((T, N, 6), np.int32)
    # --view-continuous lines carry a float view per decision: the same
    # table, the same jitter/delay, stepped alongside the bins
    viewc = any(c.get("view") is not None for c in lines[:n_lines])
    view_ticks = np.zeros((T, N, 2), np.float32) if viewc else None
    for j in range(n_lines):
        a = np.repeat(np.asarray(lines[j]["acts"], np.int32), int(K), axis=0)
        v = line_view_ticks(lines[j].get("view"), K) if viewc else None
        if viewc and v is None:
            raise SystemExit("robust_rerank: a discrete line among "
                             "--view-continuous lines")
        for q in range(n_per):
            e = j * n_per + q
            tab = a.copy()
            vtab = None if v is None else v.copy()
            if q % 2 == 0:
                d = rng.uniform(-jitter_u, jitter_u, 2)
                states["origin"][e, 0] += np.float32(d[0])
                states["origin"][e, 1] += np.float32(d[1])
            else:
                t0 = int(rng.integers(0, max(1, len(tab) - 2)))
                tab[t0 + 1:] = tab[t0:-1]
                if vtab is not None:
                    vtab[t0 + 1:] = vtab[t0:-1]
            acts_ticks[:len(tab), e] = tab
            acts_ticks[len(tab):, e] = tab[-1]
            if vtab is not None:
                view_ticks[:len(vtab), e] = vtab
                view_ticks[len(vtab):, e] = vtab[-1]
    for e in range(n_lines * n_per, N):
        acts_ticks[:, e] = acts_ticks[:, 0]
        if view_ticks is not None:
            view_ticks[:, e] = view_ticks[:, 0]
    for e in range(N):
        coreN.set_state(e, states[e])
    fin = np.full(N, -1, np.int64)
    end = np.full(N, T, np.int64)          # survival tick of a copy that dies
    alive = np.ones(N, bool)
    for t in range(T):
        if view_ticks is None:
            _o, _r, done, trunc, _term = coreN.step(np.ascontiguousarray(acts_ticks[t]))
        else:
            _o, _r, done, trunc, _term = coreN.step(
                np.ascontiguousarray(acts_ticks[t]),
                view=np.ascontiguousarray(view_ticks[t]))
        hits = np.asarray(coreN.goal_hits, np.int64) > 0
        fin[alive & hits & (fin < 0)] = t + 1
        died = alive & (np.asarray(done, bool) | np.asarray(trunc, bool)) & ~hits
        end[died] = t + 1
        alive &= ~(np.asarray(done, bool) | np.asarray(trunc, bool) | hits)
        if not alive.any():
            break
    # continuous robustness: finishers count 1, a copy that dies counts the
    # fraction of its line it survived (so lines whose copies get further
    # under the same nudge rank higher even when none finish)
    for j, c in enumerate(lines):
        if j < n_lines:
            sl = slice(j * n_per, (j + 1) * n_per)
            f, e = fin[sl], end[sl]
            L = max(1, int(len(c["acts"])) * int(K))
            surv = np.where(f >= 0, 1.0, np.minimum(e, L) / float(L))
            ok = f[f >= 0]
            c["robust_rate"] = float(len(ok)) / float(n_per)
            c["robust_surv"] = float(surv.mean())
            c["robust_tick"] = int(np.median(ok)) if len(ok) else 0
        else:
            c["robust_rate"], c["robust_surv"], c["robust_tick"] = 0.0, 0.0, 0
    order = sorted(range(len(lines)),
                   key=lambda i: (-lines[i]["robust_rate"], -lines[i]["robust_surv"],
                                  lines[i]["robust_tick"] or 10 ** 9,
                                  int(lines[i].get("finish_tick") or 0) or 10 ** 9))
    print(f"robust re-rank ({n_per} copies/line, +-{jitter_u:.1f} u or a 1-tick delay): "
          + ", ".join(f"#{i} fin {int(lines[i].get('finish_tick') or 0)} rate {lines[i]['robust_rate']:.2f} "
                      f"surv {lines[i]['robust_surv']:.3f} med {lines[i]['robust_tick']}" for i in order[:8]))
    return [lines[i] for i in order]


DEDUP_DECISIONS = 12       # prefix length hashed for --dedup (user spec)
DEDUP_ATTEMPTS = 3         # rerolls before keeping a colliding sample


class EpsSampledTorchPolicy(SampledTorchPolicy):
    """--eps: per-head epsilon-uniform proposal mixing. At each decision
    each of the 6 heads INDEPENDENTLY replaces its policy sample with a
    uniform draw over that head's bins with probability eps, so action
    combinations pi would (almost) never emit become reachable - the
    proposal-filter fix for a search that can only select among pi's own
    samples. eps=0 must NOT be routed here (unless --dedup is on, which
    changes the stream anyway): the extra RNG draws would shift the
    global torch stream and break byte-reproduction of the eps-free
    searches.

    --dedup (population reroll): within each window the first
    DEDUP_DECISIONS decisions of every candidate are prefix-hashed as
    they form; a candidate whose running prefix collides with a
    lower-indexed one gets its CURRENT decision re-sampled from the same
    per-env distribution, up to DEDUP_ATTEMPTS times, then kept as-is.
    (The full window cannot be pre-sampled - actions are closed-loop on
    sim state - so the reroll happens per decision as the prefix forms.)
    dd_stats collects per-window (before, after) duplicate counts at the
    12-decision mark; "before" is measured on a shadow hash fed the
    attempt-0 samples, i.e. the raw proposal stream's narrowness."""

    def __init__(self, *a, eps: float, dedup: bool = False, **kw):
        super().__init__(*a, **kw)
        self._eps = float(eps)
        self._nvec_t = None
        self._dedup = bool(dedup)
        self._dd_h = self._dd_hs = None
        self._dd_n = DEDUP_DECISIONS       # inert until dedup_reset()
        self.dd_stats = []                 # (before_dups, after_dups)

    def dedup_reset(self):
        """Window start: begin a fresh prefix (commit mode calls this)."""
        self._dd_h = self._dd_hs = None
        self._dd_n = 0 if self._dedup else DEDUP_DECISIONS

    def _sample(self, padded):
        act, _ = sample_padded(padded)
        if self._eps > 0.0:
            if self._nvec_t is None:
                self._nvec_t = torch.tensor(NVEC, device=act.device,
                                            dtype=act.dtype)
            mask = torch.rand(act.shape, device=act.device) < self._eps
            u = (torch.rand(act.shape, device=act.device)
                 * self._nvec_t).to(act.dtype).clamp_max(self._nvec_t - 1)
            act = torch.where(mask, u, act)
        return act

    def _sample_view(self, padded, mu):
        """--view-continuous: the four categorical heads by _sample's rule
        on the padded slice (eps-uniform per head), the view by the
        policy's own draw with, per head and with probability eps, a
        UNIFORM u in (-1, 1) in place of it (z = atanh(u): the continuous
        twin of a uniform bin). -> (act (N, 6) long, z (N, 2))."""
        act, z, _ = sample_view(padded, mu, self.policy.log_std())
        if self._eps > 0.0:
            if self._nvec_t is None:
                self._nvec_t = torch.tensor(NVEC, device=act.device,
                                            dtype=act.dtype)
            mask = torch.rand(act.shape, device=act.device) < self._eps
            u = (torch.rand(act.shape, device=act.device)
                 * self._nvec_t).to(act.dtype).clamp_max(self._nvec_t - 1)
            mask[:, :N_VIEW] = False              # the view columns are NEUTRAL
            act = torch.where(mask, u, act)
            vm = torch.rand(z.shape, device=z.device) < self._eps
            uu = (torch.rand(z.shape, device=z.device) * 2.0 - 1.0) * 0.999
            z = torch.where(vm, torch.atanh(uu), z)
        return act, z

    @staticmethod
    def _hmix(h, acts, z=None):
        """Rolling uint64 prefix hash: 6 bins < 15 pack into 24 bits; a
        --view-continuous z pair is QUANTISED to 0.05 and packed into 12
        bits each above them (clipped at +-102, far outside any tanh
        argument that matters)."""
        packed = np.zeros(len(acts), np.uint64)
        for k in range(6):
            packed |= acts[:, k].astype(np.uint64) << np.uint64(4 * k)
        if z is not None:
            q = np.clip(np.rint(np.asarray(z, np.float64) / 0.05), -2047, 2047)
            q = (q + 2048).astype(np.uint64)
            packed |= q[:, 0] << np.uint64(24)
            packed |= q[:, 1] << np.uint64(36)
        with np.errstate(over="ignore"):
            return h * np.uint64(1099511628211) ^ packed

    @torch.inference_mode()
    def _decide(self, obs):
        logits, _ = self.policy(self._obs(obs))
        if self.view_continuous:
            cat, mu = split_view(logits.float())
            padded = self.packer.pad(cat)
            act, z = self._sample_view(padded, mu)
            a = act.to("cpu").numpy().astype(np.int32)
            zn = z.to("cpu").numpy().astype(np.float32)
        else:
            padded = self.packer.pad(logits)
            act = self._sample(padded)
            a = act.to("cpu").numpy().astype(np.int32)
            zn = None
        if self._dd_n < DEDUP_DECISIONS:
            n = len(a)
            if self._dd_h is None:
                self._dd_h = np.zeros(n, np.uint64)
                self._dd_hs = np.zeros(n, np.uint64)
            self._dd_hs = self._hmix(self._dd_hs, a, zn)   # shadow: no reroll
            cur = self._hmix(self._dd_h, a, zn)
            for _ in range(DEDUP_ATTEMPTS):
                _, first = np.unique(cur, return_index=True)
                dup = np.ones(n, bool)
                dup[first] = False
                if not dup.any():
                    break
                if zn is None:
                    a2 = self._sample(padded).to("cpu").numpy().astype(np.int32)
                else:
                    act2, z2 = self._sample_view(padded, mu)
                    a2 = act2.to("cpu").numpy().astype(np.int32)
                    zn[dup] = z2.to("cpu").numpy().astype(np.float32)[dup]
                a[dup] = a2[dup]
                cur = self._hmix(self._dd_h, a, zn)
            self._dd_h = cur
            self._dd_n += 1
            if self._dd_n == DEDUP_DECISIONS:
                self.dd_stats.append(
                    (int(n - len(np.unique(self._dd_hs))),
                     int(n - len(np.unique(self._dd_h)))))
        if zn is not None:
            # the (possibly rerolled) z decides the view command
            return self._finish_view(torch.as_tensor(a[:, N_VIEW:]),
                                     torch.as_tensor(zn, device=mu.device))
        return a


class MixedTorchPolicy(SampledTorchPolicy):
    """--greedy-envs G: envs [0, G) take the ARGMAX of the very logits the
    other envs sample from - one obs assembly, one forward, one RNG draw
    for the whole batch (the sampled rows' stream is byte-identical to
    SampledTorchPolicy's for the same seed). A greedy env continues the
    policy's deterministic line from WHATEVER state it holds, so after a
    resample the greedy losers are greedy continuations of the top elites:
    the search can never do worse than "greedy from the best state found
    so far", which matters for a tight finisher (xQR32) whose sampled
    proposals are slower than its mode and whose population otherwise
    arrives at the finish window late, low, and dead."""

    def __init__(self, *a, n_greedy: int, **kw):
        super().__init__(*a, **kw)
        self._ng = int(n_greedy)

    @torch.inference_mode()
    def _decide(self, obs):
        logits, _ = self.policy(self._obs(obs))
        if self.view_continuous:
            # the greedy envs take z = mu (the view's argmax twin) and the
            # argmax of the four categorical heads, off the same forward
            cat, mu = split_view(logits.float())
            padded = self.packer.pad(cat)
            act, z, _ = sample_view(padded, mu, self.policy.log_std())
            if self._ng > 0:
                act[:self._ng, N_VIEW:] = padded[:self._ng, N_VIEW:].argmax(-1)
                z[:self._ng] = mu[:self._ng]
            return self._finish_view(act[:, N_VIEW:], z)
        padded = self.packer.pad(logits)
        act, _ = sample_padded(padded)
        if self._ng > 0:
            act[:self._ng] = padded[:self._ng].argmax(-1)
        return act.to("cpu").numpy().astype(np.int32)


def branch_draw(rng, n: int, jitter: int):
    """--branch-at: one held (yaw, side) pair per forked env, drawn from a
    PRIVATE numpy generator so the torch stream - and therefore every
    unbranched env's proposal - is untouched.

    jitter 0: the yaw column is an ABSOLUTE bin (a random heading).
    jitter J: it is an OFFSET in [-J, J] applied to the policy's own bin
    (a sustained deviation around the mode). Measured on the cannonball
    finish room, absolute headings held 0.77 s kill 96% of the forked
    lineages before they reach the ramp; offsets do not.
    -> (n, 2) int32: [yaw, side]."""
    yaw = (rng.integers(-int(jitter), int(jitter) + 1, size=n) if jitter > 0
           else rng.integers(0, NVEC[H_YAW], size=n))
    return np.stack((yaw, rng.integers(0, NVEC[H_SIDE], size=n)),
                    axis=1).astype(np.int32)


def branch_apply(act, idx, draw, jitter: int):
    """Overwrite ONLY the two heads that steer a surf flight on the forked
    envs; the rest stay the policy's own choice. Returns a new array - the
    caller's `act` is the policy wrapper's held buffer and must not be
    written through."""
    a = np.array(act)
    a[idx, H_YAW] = (np.clip(a[idx, H_YAW] + draw[:, 0], 0, NVEC[H_YAW] - 1)
                     if jitter > 0 else draw[:, 0])
    a[idx, H_SIDE] = draw[:, 1]
    return a


def branch_draw_view(rng, n: int, jitter: float):
    """--branch-at under --view-continuous: the yaw draw is a float yaw
    COMMAND. jitter 0: an absolute one, K = warp(u) with u uniform in
    (-1, 1) - uniform in the policy's own squashed coordinate, so the
    density matches its resolution (dense near zero); jitter J > 0: an
    OFFSET in K units, uniform in [-J, J], on the policy's own command.
    The side key is drawn as branch_draw draws it. Draws the same NUMBER of
    variates as branch_draw so a private generator's stream stays aligned.
    -> (n,) float64 yaw, (n,) int32 side."""
    if jitter > 0:
        yaw = rng.uniform(-float(jitter), float(jitter), size=n)
    else:
        yaw = warp(rng.uniform(-1.0, 1.0, size=n))
    side = rng.integers(0, NVEC[H_SIDE], size=n).astype(np.int32)
    return yaw, side


def branch_apply_view(act, view, idx, yaw, side, jitter: float):
    """branch_apply's --view-continuous twin: the yaw lands in the VIEW
    table (the int row's yaw column stays NEUTRAL), the side key in the
    action row. Returns NEW arrays (act', view')."""
    a = np.array(act)
    v = np.array(view, np.float32)
    v[idx, 0] = (np.clip(v[idx, 0] + yaw, -K_MAX, K_MAX) if jitter > 0
                 else np.clip(yaw, -K_MAX, K_MAX))
    a[idx, H_SIDE] = side
    return a, v


def branch_grid_parse(spec: str, K: int, view: bool = False):
    """--branch-grid SPEC -> (plans, meta). ``view`` (--view-continuous):
    the ``yaw`` field is a list of FLOAT yaw-command offsets in K units
    (default -3,-2,-1,0,1,2,3) applied to the policy's own command in the
    view table, instead of bin offsets.

    SPEC is ``key=vals`` fields joined by ':' - ``yaw`` (bin OFFSETS applied
    to the policy's own bin), ``side`` (absolute side bins, or ``p`` to keep
    the policy's own), ``hold`` (ticks the segment is held) and ``seg``
    (1 or 2). Unset fields take the defaults below, which are the
    168-plan grid the finish-room junction was written for:
    7 offsets x 3 keys x 4 durations x 2 segments.

    A plan is ``(yaw_off, side, hold, mirror)``. With ``mirror`` the macro
    is followed by a SECOND segment of the same length holding the mirrored
    manoeuvre (-yaw_off, side 0<->2) - a turn and its counter-turn, which
    is what a surf correction looks like - and without it the policy takes
    over as soon as the first segment ends.

    Holds are rounded UP to a whole number of decisions for the same reason
    --branch-burst is: the wrapper holds one action for K ticks and `hist`
    records it once per decision, so a switch inside a decision would make
    the winner's open-loop replay disagree with the search.

    Deterministic and RNG-free: unlike --branch-at this draws nothing, so
    the torch stream and every unbranched env's proposal are untouched by
    construction rather than by using a private generator.
    """
    fields = {"yaw": ("-3,-2,-1,0,1,2,3" if view else "-9,-6,-3,0,3,6,9"),
              "side": "0,1,2", "hold": "21,42,84,168", "seg": "2"}
    for part in str(spec).split(":"):
        if not part:
            continue
        k, _, v = part.partition("=")
        if k not in fields or not v:
            raise SystemExit(f"--branch-grid: bad field {part!r}; keys are "
                             + ", ".join(sorted(fields)))
        fields[k] = v
    yaws = [(float(v) if view else int(v)) for v in fields["yaw"].split(",")]
    sides = [-1 if v.strip() == "p" else int(v)
             for v in fields["side"].split(",")]
    if any(s < -1 or s >= NVEC[H_SIDE] for s in sides):
        raise SystemExit(f"--branch-grid side must be p or 0..{NVEC[H_SIDE]-1}")
    holds = []
    for v in fields["hold"].split(","):
        h = int(v)
        if h <= 0:
            raise SystemExit("--branch-grid hold must be positive")
        if h % K:
            h += K - h % K
        if h not in holds:
            holds.append(h)
    nseg = int(fields["seg"])
    if nseg not in (1, 2):
        raise SystemExit("--branch-grid seg must be 1 or 2")
    mirrors = (False, True) if nseg == 2 else (False,)
    plans = [(y, s, h, m) for y in yaws for s in sides for h in holds
             for m in mirrors]
    meta = {"yaw": yaws, "side": sides, "hold": holds, "seg": nseg,
            "plans": len(plans), "view": bool(view),
            "max_ticks": max(h * (2 if nseg == 2 else 1) for h in holds)}
    return plans, meta


def branch_grid_apply(act, idx, plans, assign, k):
    """Overwrite the yaw and side heads of the forked envs with their plan's
    macro at offset ``k`` ticks after the fork. Envs whose plan has already
    run out are left with the policy's own action, so the population goes
    back to being the policy's proposal one plan at a time. Returns a new
    array - the caller's `act` is the policy wrapper's held buffer."""
    a = np.array(act)
    for j, p in zip(idx, assign):
        y_off, side, hold, mirror = plans[p]
        if k < hold:
            pass
        elif mirror and k < 2 * hold:
            y_off, side = -y_off, (2 - side if side >= 0 else -1)
        else:
            continue
        a[j, H_YAW] = min(max(int(a[j, H_YAW]) + y_off, 0), NVEC[H_YAW] - 1)
        if side >= 0:
            a[j, H_SIDE] = side
    return a


def branch_grid_apply_view(act, view, idx, plans, assign, k):
    """branch_grid_apply's --view-continuous twin: the plan's yaw offset
    (K units, float) lands in the VIEW table, clipped to +-K_MAX; the side
    key in the action row. Returns NEW arrays (act', view')."""
    a = np.array(act)
    v = np.array(view, np.float32)
    for j, p in zip(idx, assign):
        y_off, side, hold, mirror = plans[p]
        if k < hold:
            pass
        elif mirror and k < 2 * hold:
            y_off, side = -y_off, (2 - side if side >= 0 else -1)
        else:
            continue
        v[j, 0] = float(np.clip(v[j, 0] + float(y_off), -K_MAX, K_MAX))
        if side >= 0:
            a[j, H_SIDE] = side
    return a, v
# ---------------------------------------------------------------------------
# --macro-hold: HELD-KEY macro-actions as the search's proposal
# ---------------------------------------------------------------------------
# Why (round 30 day 2): the planner's line and the human record's line are
# the SAME line to within 1,309 u, and 3.70 of the remaining 5.12 s accrue
# with NO line separation - it is strafe EXECUTION. Measured on the two
# lines: A/D flips per second 2.14 (planner) vs 0.42 (record), median side-
# key hold 0.046 s vs 0.418 s, wishdir within 0.5 deg of perpendicular to
# velocity on 50.5 % vs 82.6 % of free-flight steps, net air-strafe energy
# +0.07 M vs +1.15 M. The planner proposes each decision from the policy's
# own action distribution, so its candidates dither exactly like the policy,
# and selection cannot pick what is never proposed.
#
# In this engine (src/pm.c PM_AirAccelerate) a strafe gains speed only while
# the wish direction stays within atan(30/|v|) of perpendicular to velocity
# - about 0.6 deg at 2,800 u/s - and a key flip costs ~3 dead frames while
# the view re-aligns. So the proposal that can express what the record does
# is a HELD KEY: one (side, forward) pair drawn once and held for a
# log-uniform duration, with the yaw either still the policy's or driven
# analytically to track the velocity's rotation.
MACRO_MIN_S, MACRO_MAX_S = 0.01, 10.0     # sanity bounds on --macro-hold


def macro_draw(rng, n: int, lo_s: float, hi_s: float, dec_s: float,
               fwd_mode: str = "draw"):
    """One macro per row: (side bin, forward bin, duration in DECISIONS).

    The duration is LOG-uniform in [lo_s, hi_s] seconds - a scale-free draw
    over a range that spans an order of magnitude, so 0.2 s and 0.8 s get
    the same share of the draws - and is rounded to a whole number of
    decisions (at least 1), because the action table records one row per
    decision and the winner's open-loop replay repeats that row for
    act_every ticks: a hold that ended mid-decision could not be replayed.

    fwd_mode 'draw' draws the forward/back key uniformly and holds it with
    the side key (the literal macro: hold THE KEYS). 'none' pins it to
    neutral for the hold (the record holds W/S on 0 % of its airborne
    frames; at flight speed W swings wishdir to 45 deg off velocity and
    buys nothing). 'policy' leaves the forward head to the policy - the
    macro is then the side key alone, which isolates the held-key question
    from the forward-key question.
    -> (side (n,), fwd (n,) or None, dur_decisions (n,)), all int32."""
    side = rng.integers(0, NVEC[H_SIDE], size=n).astype(np.int32)
    if fwd_mode == "draw":
        fwd = rng.integers(0, NVEC[H_FWD], size=n).astype(np.int32)
    elif fwd_mode == "none":
        fwd = np.full(n, A_FWD_NONE, np.int32)
    elif fwd_mode == "policy":
        fwd = None
        rng.integers(0, NVEC[H_FWD], size=n)   # keep the stream shape-stable
    else:
        raise ValueError(f"macro fwd mode {fwd_mode!r}")
    u = rng.random(n)
    dur_s = np.exp(np.log(lo_s) + u * (np.log(hi_s) - np.log(lo_s)))
    dur = np.maximum(1, np.rint(dur_s / float(dec_s))).astype(np.int32)
    return side, fwd, dur


def macro_yaw_bins(vel, side, yaw_adaptive: bool, yaw_rate_max_deg: float):
    """--macro-yaw track: the yaw bin whose PER-TICK view turn best matches
    the rotation an optimal air strafe with the HELD key would give the
    velocity, per env, from the live velocity.

    The physics (src/pm.c angle_vectors at roll 0, src/env.c
    surf_pm_step_usercmd): forward = (cos yaw, sin yaw), right =
    (sin yaw, -cos yaw), and wishdir = normalize(forward*fmove +
    right*smove). So +side (bin 2, D, smove +400) accelerates along
    `right`, which is 90 deg CLOCKWISE of the view, and holding wishdir
    perpendicular to a velocity that is being rotated clockwise needs a
    NEGATIVE yaw delta; -side (bin 0, A) needs a positive one. That sign
    convention is train_fast's act/yaw_side_agree, pinned in
    tests/python/test_yaw_cond.py.

    The magnitude is the engine's own optimal-strafe rate: PM_AirAccelerate
    adds at most 30 u/s per FRAME perpendicular to v, so the velocity turns
    by atan(30/|v_h|) per tick and the view must turn by the same amount to
    stay on the optimum. Under --yaw-adaptive the bins ARE multiples of
    that angle (src/env.c K_BINS), so this picks k = -1 for D and k = +1
    for A at every speed; with fixed bins it picks the closest degree bin,
    which depends on the speed.

    Rows whose side key is neutral get -1: there is no held key to track,
    so the policy keeps the yaw head.
    -> (n,) int32 bin, -1 where there is no override."""
    v = np.asarray(vel, np.float64)
    side = np.asarray(side, np.int64)
    vh = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1.0)      # env.c clamps at 1
    w = np.degrees(np.arctan(30.0 / vh))                  # optimal per tick
    # +1 for A (bin 0, needs a POSITIVE yaw delta), -1 for D (bin 2)
    sign = np.where(side == 0, 1.0, np.where(side == 2, -1.0, 0.0))
    target = sign * w
    if yaw_adaptive:
        from surfgym.obsaux import _K_BINS
        cand = np.clip(np.asarray(_K_BINS, np.float64)[None, :] * w[:, None],
                       -float(yaw_rate_max_deg), float(yaw_rate_max_deg))
    else:
        from surfgym.core import YAW_BINS
        cand = np.repeat((np.asarray(YAW_BINS, np.float64)
                          * (float(yaw_rate_max_deg) / 10.0))[None, :],
                         len(side), axis=0)
    bins = np.abs(cand - target[:, None]).argmin(axis=1).astype(np.int32)
    return np.where(sign != 0.0, bins, -1).astype(np.int32)


def macro_yaw_k(vel, side, yaw_adaptive: bool, yaw_rate_max_deg: float):
    """macro_yaw_bins' --view-continuous twin: the yaw COMMAND that tracks
    the held key's velocity rotation EXACTLY - no nearest bin. Under
    --yaw-adaptive it is k = +1 for A and -1 for D at every speed (the
    core multiplies by atan(30/|v_h|) itself); with fixed bins the command
    is the analytic per-tick angle atan(30/|v_h|) expressed in the fixed
    path's units (k/2 deg at the reference rate), clipped to +-K_MAX.
    -> (n,) float32 command, NaN where the side key is neutral."""
    v = np.asarray(vel, np.float64)
    side = np.asarray(side, np.int64)
    sign = np.where(side == 0, 1.0, np.where(side == 2, -1.0, 0.0))
    if yaw_adaptive:
        k = sign
    else:
        vh = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1.0)
        w = np.degrees(np.arctan(30.0 / vh))
        k = np.clip(sign * w / (0.5 * float(yaw_rate_max_deg) / 10.0),
                    -K_MAX, K_MAX)
    return np.where(sign != 0.0, k, np.nan).astype(np.float32)


class MacroHold:
    """Per-env held-key macro state for --macro-hold.

    One macro per env: a side key, a forward key and a countdown of
    DECISIONS left to hold them. ``decide`` is called once per decision (at
    a t % act_every == 0 boundary, where the action table records its row);
    it redraws the expired envs, writes the held keys - and, under
    ``yaw='track'``, the analytic yaw bin - into a COPY of the policy's
    action array and counts down.

    Cloning: the macro state is CLONED donor -> loser at every resample,
    beside the action table, the arc anchor and the obs-reward feed. A
    macro is part of the lineage's ongoing manoeuvre, and the whole point
    is holds of 0.2-0.8 s while a generation is 0.575 s: redrawing at
    boundaries would truncate exactly the holds under test, and would make
    the elite's manoeuvre unreachable by construction. (The winner's
    open-loop replay is unaffected either way - it replays the recorded
    action table - so this is a search-behaviour choice, not a
    correctness one.)

    Draws come from a PRIVATE numpy generator, so the torch proposal
    stream - and therefore every run without the flag - is untouched, and
    a full batch is drawn every decision so the stream does not depend on
    how many envs happen to expire."""

    def __init__(self, n_envs: int, idx, lo_s: float, hi_s: float,
                 dec_s: float, rng, yaw: str = "policy",
                 fwd: str = "draw"):
        self.n = int(n_envs)
        self.idx = np.asarray(idx, np.int64)
        self.lo_s, self.hi_s = float(lo_s), float(hi_s)
        self.dec_s = float(dec_s)
        self.rng, self.yaw_mode, self.fwd_mode = rng, str(yaw), str(fwd)
        self.side = np.full(self.n, A_FWD_NONE, np.int32)
        self.fwd = np.full(self.n, A_FWD_NONE, np.int32)
        self.left = np.zeros(self.n, np.int32)     # 0 -> draw on decision 0
        self.draws = 0

    def decide(self, act, vel, yaw_adaptive: bool, yaw_rate_max_deg: float):
        """Redraw expired macros, then overwrite the held heads on
        ``self.idx``. Returns a NEW array: ``act`` is the policy wrapper's
        held buffer and writing through it would corrupt the rest of the
        act_every hold (tests/python/test_beam_branch.py's rule)."""
        i = self.idx
        m = len(i)
        side_d, fwd_d, dur_d = macro_draw(self.rng, m, self.lo_s, self.hi_s,
                                          self.dec_s, self.fwd_mode)
        exp = self.left[i] <= 0
        self.draws += int(exp.sum())
        self.side[i] = np.where(exp, side_d, self.side[i])
        if fwd_d is not None:
            self.fwd[i] = np.where(exp, fwd_d, self.fwd[i])
        self.left[i] = np.where(exp, dur_d, self.left[i])
        a = np.array(act)
        a[i, H_SIDE] = self.side[i]
        if self.fwd_mode != "policy":
            a[i, H_FWD] = self.fwd[i]
        if self.yaw_mode == "track":
            yb = macro_yaw_bins(np.asarray(vel)[i], self.side[i],
                                yaw_adaptive, yaw_rate_max_deg)
            a[i, H_YAW] = np.where(yb >= 0, yb, a[i, H_YAW])
        self.left[i] -= 1
        return a

    def decide_view(self, act, view, vel, yaw_adaptive: bool,
                    yaw_rate_max_deg: float):
        """decide() for --view-continuous: the held keys land in the action
        row exactly as before and, under ``yaw='track'``, the analytic yaw
        COMMAND (macro_yaw_k) in the view table. Returns NEW arrays
        (act', view')."""
        i = self.idx
        a = self.decide(act, vel, yaw_adaptive, yaw_rate_max_deg)
        a[i, H_YAW] = NEUTRAL_ACT[H_YAW]       # the view table owns the yaw
        v = np.array(view, np.float32)
        if self.yaw_mode == "track":
            kk = macro_yaw_k(np.asarray(vel)[i], self.side[i], yaw_adaptive,
                             yaw_rate_max_deg)
            ok = np.isfinite(kk)
            v[i[ok], 0] = kk[ok]
        return a, v

    def clone(self, losers, donors):
        """A loser now carries the donor's action history, so it carries
        the donor's macro."""
        for arr in (self.side, self.fwd, self.left):
            arr[losers] = arr[donors]


# ---------------------------------------------------------------------------
# strafe-cadence diagnostics (the mechanism check, in the log)
# ---------------------------------------------------------------------------
def strafe_cadence(vel, yaw, onground, fwd, side, dt, g_step,
                   contact_tol: float = 1.0, perp_tol_deg: float = 0.5):
    """The four numbers round 30 day 2 measured on the planner's line
    against the human record's, computed per TICK from a replayed episode.

    Rows are the recorder's (surfgym.record / run_episode): ``vel``,
    ``yaw`` and ``onground`` are PRE-step, ``fwd``/``side`` are the action
    taken on that tick, ``dt`` is that tick's real duration in seconds and
    ``g_step`` that tick's free-flight change of vz.

    * flips_per_s     - A/D DIRECTION flips per second: transitions between
                        opposite non-neutral side keys, with neutral
                        stretches skipped (releasing D and pressing it
                        again is not a flip). side_changes_per_s counts
                        every change of the side bin instead.
    * hold_med_s      - median duration of a run of one NON-NEUTRAL side
                        key (a held A or D); hold_med_all_s includes the
                        neutral runs.
    * perp_share      - of FREE-FLIGHT ticks (airborne and the map did not
                        push back: |dvz - g_step| <= contact_tol), the
                        share whose wish direction is within perp_tol_deg
                        of perpendicular to the horizontal velocity - the
                        band inside which PM_AirAccelerate actually pays
                        (atan(30/|v|), ~0.6 deg at 2,800 u/s).
    * strafe_energy_M - net free-flight change of 0.5*|v_h|^2, in millions:
                        what the air strafe actually banked.

    The wish direction uses the yaw the MOVE ran at, which is the NEXT
    row's pre-step yaw (src/env.c adds the yaw delta before pm_tick), and
    the velocity at the start of that move, which is this row's. In the WR
    demo the frame's OWN viewangle is the one its usercmd drove the step
    with (tools/demo/compare_wr.load_wr), so the reference numbers below
    are that line measured under its own convention - the same physical
    quantity, one index apart in two file formats.

    Calibration (round 30 day 3, these definitions, whole run):

        line                flips/s  med hold  perp 0.5deg  energy
        WR demo               0.84     0.422s     79.3 %    +1.17 M
        m763fix planner       3.27     0.046s     43.1 %    -0.38 M

    Median hold reproduces round 30 day 2's 0.418 / 0.046 exactly and the
    WR's energy its +1.15 M; the flip and perpendicular counts sit on the
    same ordering at a different absolute level, so compare arms with each
    other under THESE numbers, not with day 2's."""
    v = np.asarray(vel, np.float64)
    yaw = np.asarray(yaw, np.float64)
    og = np.asarray(onground, np.int64)
    fwd = np.asarray(fwd, np.int64)
    side = np.asarray(side, np.int64)
    dt = np.asarray(dt, np.float64)
    n = len(v)
    g_step = np.broadcast_to(np.asarray(g_step, np.float64), (n,))
    out = {"ticks": int(n), "episode_s": float(dt[:n].sum())}
    if n < 2:
        return out
    # ---- side-key cadence (whole episode, per tick = per decision) -----
    chg = np.nonzero(side[1:] != side[:-1])[0] + 1
    out["side_changes"] = int(len(chg))
    out["side_changes_per_s"] = float(len(chg) / max(out["episode_s"], 1e-9))
    nz = side[side != A_FWD_NONE]
    flips = int((nz[1:] != nz[:-1]).sum()) if len(nz) > 1 else 0
    out["flips"] = flips
    out["flips_per_s"] = float(flips / max(out["episode_s"], 1e-9))
    edges = np.concatenate(([0], chg, [n]))
    runs = [(int(side[a]), float(dt[a:b].sum()))
            for a, b in zip(edges[:-1], edges[1:])]
    held = [d for k, d in runs if k != A_FWD_NONE]
    out["hold_med_s"] = float(np.median(held)) if held else 0.0
    out["hold_med_all_s"] = float(np.median([d for _k, d in runs]))
    out["hold_max_s"] = float(max(held)) if held else 0.0
    out["side_runs"] = int(len(runs))
    # ---- free flight: airborne and nothing pushed back ------------------
    dvz = v[1:, 2] - v[:-1, 2]
    free = (og[:-1] == 0) & (np.abs(dvz - g_step[:-1]) <= contact_tol)
    nf = int(free.sum())
    out["free_ticks"] = nf
    out["free_share"] = float(nf / max(n - 1, 1))
    if nf == 0:
        out["perp_share"] = 0.0
        out["strafe_energy_M"] = 0.0
        out["fwd_air"] = 0.0
        return out
    # wishdir = normalize(forward*fmove + right*smove) at the MOVE's yaw
    ym = np.radians(yaw[1:])
    fm = np.where(fwd[:-1] <= 0, -400.0, np.where(fwd[:-1] >= 2, 400.0, 0.0))
    sm = np.where(side[:-1] <= 0, -400.0, np.where(side[:-1] >= 2, 400.0, 0.0))
    wx = np.cos(ym) * fm + np.sin(ym) * sm
    wy = np.sin(ym) * fm - np.cos(ym) * sm
    wn = np.hypot(wx, wy)
    vh = np.hypot(v[:-1, 0], v[:-1, 1])
    ok = free & (wn > 1e-9) & (vh > 1e-9)
    cosang = np.zeros(n - 1)
    np.divide(wx * v[:-1, 0] + wy * v[:-1, 1], np.maximum(wn * vh, 1e-9),
              out=cosang, where=ok)
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    out["perp_share"] = float((ok & (np.abs(ang - 90.0) <= perp_tol_deg)
                               ).sum() / nf)
    e = 0.5 * (v[:, 0] ** 2 + v[:, 1] ** 2)
    out["strafe_energy_M"] = float((e[1:] - e[:-1])[free].sum() / 1e6)
    out["fwd_air"] = float((fwd[:-1][free] != A_FWD_NONE).sum() / nf)
    return out


def cadence_from_traj(path, tick, sv_gravity: float,
                      contact_tol: float = 1.0, start: int = 0):
    """strafe_cadence for a trajectory file run_episode just wrote (one
    episode, from tick-pattern phase 0). ``start`` drops the first N ticks,
    which is how a --prefix-line run reads the part the search actually
    proposed: 88 % of such a winner is a replay of the reference line, and
    the whole-line cadence is that line's, not the arm's. Returns None if
    it cannot be read - a diagnostic must never take a run down."""
    try:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                if isinstance(r, list):
                    rows.append(r)
        if len(rows) < 2:
            return None
        a = np.asarray(rows, np.float64)
        pat = [float(x) for x in tick.pattern]
        dt = np.asarray([pat[i % len(pat)] / 1000.0 for i in range(len(a))])
        g = -float(sv_gravity) * dt
        s = max(0, int(start))
        if s >= len(a) - 1:
            return None
        return strafe_cadence(a[s:, 4:7], a[s:, 7], a[s:, 9], a[s:, 13],
                              a[s:, 14], dt[s:], g[s:],
                              contact_tol=contact_tol)
    except Exception as exc:                       # pragma: no cover
        print(f"  (cadence diagnostics unavailable: {exc})")
        return None


def print_cadence(c, label="winner"):
    """The mechanism check, in the log. The reference figures are the WR
    demo and the m763fix planner line measured under strafe_cadence's own
    definitions (see its docstring's calibration table)."""
    if not c:
        return
    print(f"cadence [{label}]: A/D flips {c['flips']} "
          f"({c['flips_per_s']:.2f}/s; record 0.84, planner 3.27), "
          f"side changes {c['side_changes_per_s']:.2f}/s, median held-key "
          f"run {c['hold_med_s']:.3f}s (record 0.422, planner 0.046), "
          f"max {c['hold_max_s']:.2f}s")
    print(f"  free flight {c['free_ticks']}/{c['ticks']} ticks "
          f"({100 * c['free_share']:.1f}%): wishdir within 0.5 deg of "
          f"perpendicular on {100 * c['perp_share']:.1f}% (record 79.3, "
          f"planner 43.1), net strafe energy "
          f"{c['strafe_energy_M']:+.2f} M (record +1.17, planner -0.38), "
          f"fwd key held on {100 * c['fwd_air']:.1f}%")


def make_scorer(gf, route_file, corridor, mode, value_fn=None,
                v_switch: float = 0.0):
    """-> score(states, obs) = (higher is better, geodesic d).

    mode 'd' is the plain -geodesic used by the spawn-to-finish search.
    mode 'route' ranks by how far along the ROUTE a candidate is, which is
    what a search AT THE WALL needs: this map's geodesic field has its
    along-route minimum AT the wall (vertex 1601, d=6,568) and goal-
    adjacent airspace below the ramp scores lower still, so -d ranking
    near the wall actively selects the dive. Route vertex index is
    monotone by construction; d only breaks ties within a vertex.
    mode 'v' ranks by the checkpoint's own critic V(s) (round 27: "the
    critic knows what the field does not" - it cleared the kill funnel
    d-ranking died in); 'dv' is d until the population's frontier d drops
    below v_switch, then V - the endgame is where d lies most.
    """
    if mode == "d":
        def score(states, obs=None):
            d = gf.sample(states["origin"]).astype(np.float64)
            return -d, d
        return score
    if mode in ("v", "dv"):
        if value_fn is None:
            raise SystemExit(f"--score {mode} needs the critic (value_fn)")

        def score(states, obs=None):
            d = gf.sample(states["origin"]).astype(np.float64)
            if mode == "dv" and float(d.min()) > v_switch:
                return -d, d
            return value_fn(obs).astype(np.float64), d
        return score

    pts, _spacing = load_route(Path(route_file))
    P = np.asarray(pts, np.float64)

    def score(states, obs=None):
        o = np.asarray(states["origin"], np.float64)
        d2 = ((o[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        near = np.sqrt(d2.min(axis=1))
        idx = d2.argmin(axis=1).astype(np.float64)
        d = gf.sample(states["origin"]).astype(np.float64)
        # out-of-corridor candidates rank below every in-corridor one
        return np.where(near <= corridor, idx, -1.0) * 1e6 - d, d
    return score


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 1.0


def commit_search(coreN, spol, gf, obs, N, K, H, C, max_ticks, feed_state,
                  value_fn=None, latch_fn=None, tick=None, viewc=False):
    """Receding-horizon (MPC) search: windows of H decisions with NO
    intra-window cloning - 2048 maximally diverse continuations, judged
    only at the window boundary. Boundary order: (1) finished inside the
    window, earlier tick first (time flows forward, so the window's
    first goal hit is provably its best - the window ends right there);
    (2) alive at the boundary, smaller geodesic d first; (3) died inside
    the window, always last (a dive ends in a kill_z death, so dives are
    self-defeating). The single best lineage's first C decisions are
    committed and ALL envs re-center on its commit-point state, captured
    mid-window via get_states along with its obs row and obs-reward feed
    value - a boundary-only snapshot would be C*K ticks too late.

    Returns (best_or_None, info, dnf_reason_or_None); best has the same
    shape the v1 population search produces, so the replay/summary tail
    is shared."""
    import os
    debug = bool(os.environ.get("BEAM_DEBUG"))
    tick = tick or TickClock(10.0)

    def secs(n):
        return ticks_to_secs(n, tick.ms, tick.pattern)

    committed = []                # list of (C, 6) int8 blocks
    committed_v = []              # --view-continuous: the (C, 2) view blocks
    committed_ticks = 0
    windows = 0
    sim_ticks = 0
    rank_sp, rank_ov = [], []     # --boundary-v: d-rank vs V-rank per window

    def _info():
        d = {"windows": windows, "sim_ticks": sim_ticks,
             "committed_ticks": committed_ticks}
        if rank_sp:
            d["rank_agreement"] = {
                "spearman_mean": round(float(np.mean(rank_sp)), 3),
                "top64_overlap_mean": round(float(np.mean(rank_ov)), 3),
                "windows": len(rank_sp)}
        if getattr(spol, "dd_stats", None):
            b = [x for x, _ in spol.dd_stats]
            a = [y for _, y in spol.dd_stats]
            d["collision_before"] = round(float(np.mean(b)) / N, 4)
            d["collision_after"] = round(float(np.mean(a)) / N, 4)
        return d

    while committed_ticks < max_ticks:
        if hasattr(spol, "dedup_reset"):
            spol.dedup_reset()    # --dedup prefixes are per-window
        hist = np.zeros((H, N, 6), np.int8)
        hist_v = np.zeros((H, N, 2), np.float32) if viewc else None
        fin_tick = np.zeros(N, np.int64)     # 1-based in-window tick
        finish_pre = {}
        died = np.zeros(N, bool)
        snap = None
        if debug:
            dth_t = np.zeros(N, np.int64)    # in-window tick of first death
            dth_z = np.zeros(N, np.float32)  # pre-step z at that death
        for t in range(H * K):
            d = t // K
            a = spol.act(obs)
            v = spol.view if viewc else None
            if t % K == 0:
                hist[d] = a
                if viewc:
                    hist_v[d] = v
            sv = coreN.states_view
            pre_o = sv["origin"].copy()
            pre_v = sv["velocity"].copy()
            if v is None:
                obs, _rew, done, trunc, _term = coreN.step(a)
            else:
                obs, _rew, done, trunc, _term = coreN.step(a, view=v)
            sim_ticks += 1
            gh = coreN.goal_hits
            hit = False
            if gh.any():
                for i in np.nonzero(gh)[0]:
                    if not died[i]:
                        fin_tick[i] = t + 1
                        finish_pre[int(i)] = (pre_o[i].copy(),
                                              pre_v[i].copy())
                        hit = True
            dd = np.asarray(done, bool) | np.asarray(trunc, bool)
            if debug and dd.any():
                new = dd & ~died
                dth_t[new] = t + 1
                dth_z[new] = pre_o[new, 2]
            died |= dd
            if hit:
                break     # the first hit is the window's earliest finish
            if t + 1 == C * K:
                snap = (coreN.get_states(), np.array(obs),
                        None if (feed_state is None
                                 or feed_state.get("d") is None)
                        else feed_state["d"].copy(),
                        None if (latch_fn is None
                                 or latch_fn.state.get("f") is None)
                        else latch_fn.state["f"].copy())
        if debug and died.any():
            dt_ = dth_t[dth_t > 0]
            dz_ = dth_z[dth_t > 0]
            q = np.percentile(dt_, [10, 50, 90]).astype(int)
            zq = np.percentile(dz_, [10, 50, 90]).astype(int)
            print(f"  dbg win {windows + 1}: deaths {len(dt_)} "
                  f"tick q10/50/90 {q[0]}/{q[1]}/{q[2]} "
                  f"z q10/50/90 {zq[0]}/{zq[1]}/{zq[2]} "
                  f"void(z<-500) {int((dz_ < -500).sum())}")
        windows += 1
        fins = np.nonzero(fin_tick > 0)[0]
        if len(fins):
            i = int(fins[np.argmin(fin_tick[fins])])
            f = int(fin_tick[i])
            total = committed_ticks + f
            dfin = (f - 1) // K
            blocks = committed + [hist[:dfin + 1, i].copy()]
            po, pv = finish_pre[i]
            print(f"win {windows}: FINISH env {i} at in-window tick {f} "
                  f"-> total {total} ({secs(total):.2f}s)")
            best = {"tick": total, "acts": np.concatenate(blocks, axis=0),
                    "pre_origin": po, "pre_vel": pv}
            if viewc:
                best["view"] = np.concatenate(
                    committed_v + [hist_v[:dfin + 1, i].copy()], axis=0)
            return best, _info(), None
        alive = ~died
        if not alive.any():
            reason = (f"window {windows}: all {N} candidates died "
                      f"(committed {committed_ticks} ticks so far)")
            print("DNF: " + reason)
            return None, _info(), reason
        dgeo = gf.sample(coreN.get_states()["origin"]).astype(np.float64)
        if value_fn is not None:
            # --boundary-v: rank the alive candidates by the checkpoint's
            # own critic instead of the speed-blind (and kill-volume-
            # blind) geodesic d. Log how much the orderings disagree.
            vals = value_fn(obs).astype(np.float64)
            lead = int(np.argmax(np.where(alive, vals, -np.inf)))
            ai = np.nonzero(alive)[0]
            if len(ai) >= 2:
                sp = _spearman(dgeo[ai], -vals[ai])
                k = min(64, len(ai))
                top_d = set(ai[np.argsort(dgeo[ai])[:k]].tolist())
                top_v = set(ai[np.argsort(-vals[ai])[:k]].tolist())
                ov = len(top_d & top_v) / k
                rank_sp.append(sp)
                rank_ov.append(ov)
                print(f"  rank d-vs-V: spearman {sp:+.3f} "
                      f"top{k} overlap {ov:.2f} "
                      f"(d-pick {int(np.argmin(np.where(alive, dgeo, np.inf)))}"
                      f" V-pick {lead})")
        else:
            lead = int(np.argmin(np.where(alive, dgeo, np.inf)))
        committed.append(hist[:C, lead].copy())
        if viewc:
            committed_v.append(hist_v[:C, lead].copy())
        committed_ticks += C * K
        st_row = snap[0][lead:lead + 1].copy()
        # the committed state's episode clock must equal the assembled
        # tick count, or the finish-time accounting is broken
        assert int(st_row["tick"][0]) == committed_ticks, \
            (int(st_row["tick"][0]), committed_ticks)
        for j in range(N):
            coreN.set_state(j, st_row)
        # a --tick-ms pattern: the committed timeline is committed_ticks
        # ticks old while the core has stepped a whole window - re-phase
        # it so the open-loop replay from tick 0 meets the same ms sequence
        if hasattr(coreN, "set_tick_phase"):
            coreN.set_tick_phase(committed_ticks)
        obs = np.array(obs)
        obs[:] = snap[1][lead][None, :]
        if feed_state is not None:
            feed_state["d"] = (None if snap[2] is None
                               else np.full(N, snap[2][lead]))
        if latch_fn is not None and snap[3] is not None:
            # the --race-latch flag is episode history: every env now IS
            # the lead's history, so it carries the lead's flag
            latch_fn.state["f"] = np.full(N, bool(snap[3][lead]))
            latch_fn.state["tick"] = np.full(N, int(st_row["tick"][0]),
                                             np.int64)
        print(f"win {windows:3d} t={committed_ticks:5d} "
              f"alive={int(alive.sum()):4d} died={int(died.sum()):4d} "
              f"lead_d={dgeo[lead]:8.0f} "
              f"commit_d={float(gf.sample(st_row['origin'])[0]):8.0f} "
              f"z={float(st_row['origin'][0][2]):7.0f} "
              f"vz={float(st_row['velocity'][0][2]):6.0f}")
    return None, _info(), \
        f"cap: {committed_ticks} committed ticks without a finish"


def main():
    ap = argparse.ArgumentParser(
        description="policy-guided population search for a record run")
    ap.add_argument("ckpt", nargs="?", default=DEF_CKPT)
    ap.add_argument("--map", default=None, help="defaults to the ckpt's map, "
                    "resolved in the MAIN checkout's maps/ (cache mtimes)")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--resample-every", type=int, default=25,
                    help="decisions per generation (25 dec x act_every 3 "
                    "= 75 ticks = 0.75 s between clonings)")
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=0.0,
                    help="per-head epsilon-uniform proposal mixing: each "
                    "head independently goes uniform over its bins with "
                    "this probability at every decision (0 = pure pi, "
                    "byte-identical to the eps-free tool)")
    ap.add_argument("--commit", type=int, default=0,
                    help="receding-horizon mode: window length in "
                    "DECISIONS (act_every 3: 167 =~ 5s, 333 =~ 10s). No "
                    "intra-window cloning; --resample-every/--elite-frac/"
                    "--gens are ignored. 0 = v1 population search")
    ap.add_argument("--commit-frac", type=float, default=0.5,
                    help="fraction of each window committed from the best "
                    "lineage at the boundary (MPC overlap: committing the "
                    "whole window would lock in window-tail traps)")
    ap.add_argument("--boundary-v", action="store_true",
                    help="commit mode: rank alive boundary candidates by "
                    "the checkpoint's own critic V(s) instead of geodesic "
                    "d (finished-first / died-last unchanged), and log "
                    "how the two orderings disagree per window. d is "
                    "speed-blind and kill-volume-blind; a critic that has "
                    "seen finishes is neither")
    ap.add_argument("--dedup", action="store_true",
                    help="commit mode: population prefix-dedup + reroll - "
                    "re-sample a candidate's decision when its running "
                    "first-12-decision prefix collides with a lower-"
                    "indexed candidate's (up to 3 attempts, then keep); "
                    "reports before/after collision rates per window")
    ap.add_argument("--greedy-prefix", type=int, default=0,
                    help="search AT THE WALL: run the policy GREEDILY for "
                    "this many ticks first, then switch to sampling. Greedy "
                    "from a fixed spawn is deterministic, so this restores "
                    "one exact pre-wall state into all N envs without "
                    "feeding a reconstructed observation - and the recorded "
                    "history already holds the prefix, so the winner's "
                    "replay is a spliced start-to-finish run, asserted "
                    "bit-exact like any other")
    ap.add_argument("--prefix-line", default=None, metavar="NPZ[:TICKS]",
                    help="ROUTE SEARCH: replay a saved line's own action "
                    "table (beam_best.npz) open-loop into every env for "
                    "TICKS ticks, then start the population search from "
                    "that state. Costs physics only - no lidar, no policy "
                    "forward - so a search that only asks about the LAST "
                    "few seconds of a 75 s line costs the last few seconds. "
                    "The spawn state and the gate seed come from the npz, "
                    "the prefix decisions are written into the action "
                    "history, and the winner's replay is still a spliced "
                    "start-to-finish run asserted bit-exact. Default TICKS "
                    "= the whole table.")
    ap.add_argument("--branch-at", default=None, metavar="WHERE:N",
                    help="ROUTE SEARCH: fork N envs at a junction with a "
                    "temporally correlated action burst, so the population "
                    "can contain a line the policy's own proposals never "
                    "emit. WHERE is tTICK (a tick), dVALUE (the leader's "
                    "geodesic d first at or below VALUE) or aVALUE (the "
                    "leader's arc at or above VALUE, --objective "
                    "progress/auto only). The fork fires ONCE, at the first "
                    "resample boundary at which the trigger holds, and the "
                    "branched envs are the LAST N (so --greedy-envs keeps "
                    "envs [0, G))")
    ap.add_argument("--branch-burst", type=int, default=200,
                    help="ticks of correlated override after the fork")
    ap.add_argument("--branch-hold", type=int, default=100,
                    help="ticks one random (yaw, side) pair is held before "
                    "it is redrawn - the temporal correlation length. White "
                    "noise per decision discovers falling, not ramps "
                    "(round 27's eps sweep); a held key is a manoeuvre")
    ap.add_argument("--branch-protect", type=int, default=0,
                    help="generations after the fork during which branched "
                    "lineages are exempt from being cloned over. This is "
                    "the selection-horizon half: a manoeuvre that costs "
                    "speed now and pays a second later is culled at the "
                    "next boundary without it")
    ap.add_argument("--branch-jitter", type=int, default=0,
                    help="0 = the burst draws the yaw bin UNIFORMLY (a "
                    "random heading); J > 0 = it OFFSETS the policy's own "
                    "yaw bin by U{-J..J} instead, still holding one side "
                    "key for the window. Measured on the cannonball finish "
                    "room: a uniform hold of 0.77 s kills 96%% of the "
                    "forked lineages before they reach the ramp, so the "
                    "useful perturbation is a sustained deviation AROUND "
                    "the mode, not a random heading")
    ap.add_argument("--branch-grid", default=None, metavar="WHERE:SPEC",
                    help="ROUTE SEARCH, deterministic: same fork as "
                    "--branch-at (WHERE is tTICK / dVALUE / aVALUE) but the "
                    "forked envs get an ENUMERATED grid of held-key macro "
                    "plans instead of random bursts, replicated round-robin "
                    "over the population. SPEC is 'yaw=..:side=..:hold=..:"
                    "seg=1|2' (see branch_grid_parse); the default is "
                    "7 yaw-bin offsets x 3 side keys x 4 hold durations x "
                    "{macro, macro+mirror} = 168 plans. The policy continues "
                    "after a plan's macro ends, --branch-protect keeps the "
                    "grid lineages alive across the boundaries that would "
                    "cull a manoeuvre paying off a second later, and the "
                    "winner's summary names the plan it descends from. "
                    "Draws no randomness at all")
    ap.add_argument("--branch-seed", type=int, default=None,
                    help="RNG for the burst (default: --torch-seed). Drawn "
                    "from a private numpy generator, so the torch stream - "
                    "and therefore every unbranched env's proposal - is "
                    "untouched")
    ap.add_argument("--macro-hold", default=None, metavar="MIN:MAX",
                    help="HELD-KEY PROPOSAL: replace the per-decision side "
                    "(and forward) key of every non-greedy env with a MACRO "
                    "- one drawn pair of keys HELD for a log-uniform "
                    "duration in [MIN, MAX] SECONDS, rounded to whole "
                    "decisions. Off by default and byte-identical when off. "
                    "The planner's line dithers A/D 2.14 times a second "
                    "against the record's 0.42 because it proposes from the "
                    "policy, and a strafe only pays while wishdir stays "
                    "within atan(30/|v|) of perpendicular; a flip costs ~3 "
                    "dead frames. The macro state is CLONED with the action "
                    "table at every resample, so a hold survives a "
                    "generation boundary")
    ap.add_argument("--macro-yaw", choices=["policy", "track"],
                    default="policy",
                    help="--macro-hold: where the yaw bin comes from. "
                    "'policy' (default) keeps the policy's own yaw head - "
                    "the macro is the KEYS only. 'track' sets it "
                    "analytically each decision to the bin whose per-tick "
                    "view turn best matches the velocity rotation the held "
                    "key produces (atan(30/|v|) per frame, NEGATIVE for D "
                    "and positive for A); under --yaw-adaptive that is "
                    "K_BINS k = -+1, the optimal per-frame strafe. Ignored "
                    "without --macro-hold")
    ap.add_argument("--macro-fwd", choices=["draw", "none", "policy"],
                    default="draw",
                    help="--macro-hold: the forward/back key. 'draw' "
                    "(default) draws it with the side key and holds it; "
                    "'none' pins it neutral for the hold (the record holds "
                    "W/S on 0%% of its airborne frames); 'policy' leaves "
                    "the head to the policy, isolating the side-key "
                    "question. Ignored without --macro-hold")
    ap.add_argument("--macro-frac", type=float, default=1.0,
                    help="--macro-hold: the fraction of the non-greedy envs "
                    "that carry a macro; the rest keep proposing from the "
                    "policy per decision. 1.0 (default) REPLACES the "
                    "proposal, which is only a fair test if held keys are "
                    "better everywhere; below 1.0 the two distributions "
                    "COMPETE inside one population and selection decides "
                    "per generation. Ignored without --macro-hold")
    ap.add_argument("--macro-seed", type=int, default=None,
                    help="RNG for the macro draws (default: --seed). A "
                    "private numpy generator, so the torch stream - and "
                    "every run without --macro-hold - is untouched")
    ap.add_argument("--skip-gate", action="store_true",
                    help="capture the spawn state but do NOT roll the gate "
                    "episode. row0/obs_start come from the reset itself, so "
                    "for a --greedy-prefix search the gate rollout is pure "
                    "overhead (a full single-env episode per wave)")
    ap.add_argument("--allow-nonfinisher", action="store_true",
                    help="do not require the greedy gate to finish (implied "
                    "by --greedy-prefix: searching from the wall is only "
                    "interesting for a policy that does NOT finish)")
    ap.add_argument("--score", choices=["d", "route", "v", "dv"],
                    default="d",
                    help="boundary ranking: 'd' = geodesic (spawn-to-finish "
                    "search), 'route' = route vertex index then d (wall "
                    "search - see make_scorer), 'v' = the checkpoint's own "
                    "critic, 'dv' = d until the frontier is within "
                    "--v-switch of the goal, then the critic")
    ap.add_argument("--v-switch", type=float, default=20000.0,
                    help="--score dv: switch from d to V(s) once the "
                    "population's smallest geodesic d is below this")
    ap.add_argument("--greedy-envs", type=int, default=0,
                    help="population mode: this many envs act GREEDILY on "
                    "the shared forward (MixedTorchPolicy) - greedy "
                    "continuations of the elites after every resample, "
                    "and env 0 is never cloned over while alive, so the "
                    "search is bounded below by the greedy line itself. "
                    "0 = every env samples, as before")
    ap.add_argument("--route-file",
                    default="C:/RL_Surf/maps/surf_src_cannonball.route.npz")
    ap.add_argument("--corridor", type=float, default=1500.0)
    ap.add_argument("--max-ticks", type=int, default=12000,
                    help="search cap in physics TICKS, not seconds (12000 "
                    "= 120 s at 10 ms, 92 s at 7.63 ms; the log prints "
                    "both - raise it if a run at a shorter tick needs more)")
    ap.add_argument("--tick-ms", type=float, default=None,
                    help="OVERRIDE the physics tick (ms) for the whole "
                    "search. Default: PHYSICS PARITY, the checkpoint's own "
                    "tick_ms (10 for every checkpoint that predates the "
                    "flag). 7.63 = the WR demo's 131 fps, run as the "
                    "[8, 8, 7] ms pattern. Every count stays in ticks and "
                    "every printed/saved second is real time under the "
                    "pattern; beam_best.npz / summary.json / the trajectory "
                    "headers carry tick_ms + tick_pattern_ms. Logged "
                    "loudly, like record_ckpt --tick-ms")
    ap.add_argument("--act-every", type=int, default=None,
                    help="OVERRIDE the decision interval in physics ticks "
                    "(default: the checkpoint's own act_every). With "
                    "--tick-ms this keeps the decision interval in SECONDS "
                    "where the weights learned it (K=3 at 10 ms = 30 ms; "
                    "K=4 at 7.67 ms = 30.7 ms). Logged loudly; the npz "
                    "records the K used, and plan_to_bc refuses a plan "
                    "whose K is not the checkpoint's")
    ap.add_argument("--gens", type=int, default=0,
                    help="stop this many generations after the first finish "
                    "(a later finish can never be faster; 0 = run to "
                    "--max-ticks and count finishers)")
    ap.add_argument("--seed", type=int, default=0, help="spawn draw seed")
    ap.add_argument("--torch-seed", type=int, default=0,
                    help="proposal sampling seed")
    ap.add_argument("--greedy-eps", type=int, default=3,
                    help="greedy sanity episodes to try (first finisher "
                    "supplies the matched spawn state and baseline time)")
    ap.add_argument("--greedy-only", action="store_true",
                    help="run the greedy sanity gate and exit")
    ap.add_argument("--robust", type=int, default=0,
                    help="re-rank the kept lineages by robustness: each replayed this many "
                         "times open-loop with the spawn jittered (+-robust-jitter u) or one "
                         "1-tick delay; order = finish rate, then median finish tick. 0 = off, "
                         "byte-identical. The 68.54 s line dies on a 1 u offset 59% of the time")
    ap.add_argument("--robust-jitter", type=float, default=0.5)
    ap.add_argument("--keep-finishers", "--keep-lines", type=int, default=1,
                    dest="keep_finishers",
                    help="population mode: also save the action histories "
                    "of the fastest K distinct finishing lineages "
                    "(acts_all in beam_best.npz) for the expert-iteration "
                    "BC builder; 1 = the winner only, as before. Under "
                    "--objective progress/auto the same K counts kept "
                    "lineages of either kind (finishers first)")
    ap.add_argument("--objective", choices=["finish", "progress", "auto"],
                    default="finish",
                    help="population mode: what the search optimises. "
                    "'finish' (default) = finish time, as before - a "
                    "crossing is required. 'progress' = the furthest "
                    "order-only corridor arc along --route-file "
                    "(eval_honesty's --order-only rule): elites by best "
                    "arc (ties by --score), dead lineages keep their best "
                    "arc, no crossing needed. 'auto' = progress, with any "
                    "finisher ranked first by time")
    ap.add_argument("--arc-window", type=int, default=16,
                    help="--objective progress/auto: ArcProgress local "
                    "window in route vertices (eval_honesty --order-only 16)")
    ap.add_argument("--arc-quant", type=float, default=0.0,
                    help="--objective progress/auto: rank elites by best arc "
                    "in bins of this many map units so the --score value "
                    "(geodesic d, or the critic under dv/v) decides inside "
                    "a bin; 0 = exact arc, --score only on exact ties")
    ap.add_argument("--arc-bank", choices=["contact", "raw"],
                    default="contact",
                    help="--objective progress/auto: 'contact' (default) "
                    "credits a lineage's best arc only at ticks the map "
                    "pushed back (its vz change departs from the gravity "
                    "step by more than --contact-tol) - the arc it BANKED "
                    "before its last contact, which is exactly what "
                    "plan_to_bc's last-contact trim keeps, so a fall earns "
                    "nothing. 'raw' credits every live tick (a fall along "
                    "the route counts until the lineage dies)")
    ap.add_argument("--contact-tol", type=float, default=1.0,
                    help="--arc-bank contact: |dvz - gravity step| above "
                    "this is a contact tick (plan_to_bc --contact-tol)")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "beam_tas"))
    ap.add_argument("--log-every", type=int, default=5,
                    help="print every Nth generation")
    ap.add_argument("--no-config-audit", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    step = int(ck.get("global_step", 0))
    if cfg.get("reward") != "race":
        raise SystemExit("beam_tas needs a race checkpoint (goal-box "
                         f"finish); this one has reward={cfg.get('reward')!r}")
    bad = [k for k in UNSUPPORTED if cfg.get(k)]
    if bad:
        raise SystemExit("beam_tas v1 cannot clone the per-env inference "
                         "state of: " + ", ".join(bad))
    _rc.audit_cfg(cfg, strict=not args.no_config_audit)

    # --tick-ms: the physics tick the weights trained under (checkpoints
    # that predate the flag have no key: 10 ms), overridable the way
    # record_ckpt's is. TICK owns every ticks<->seconds conversion below.
    cfg_tick = float(cfg.get("tick_ms") or 10.0)
    tick_ms = cfg_tick if args.tick_ms is None else float(args.tick_ms)
    TICK = TickClock(tick_ms)
    tick_override = abs(tick_ms - cfg_tick) > 1e-9
    tick_json = tick_stamp(TICK, cfg_tick)
    tick_json["view_continuous"] = bool(cfg.get("view_continuous"))

    def secs(n):
        """ticks from an episode start -> seconds at the REAL tick (the
        [8, 8, 7] pattern summed from phase 0; ticks / 100.0 at 10 ms)."""
        return ticks_to_secs(n, TICK.ms, TICK.pattern)

    def gain_s(a, b):
        """seconds a - b: the legacy (a - b) / 100.0 at 10 ms, bit for
        bit; the difference of two exactly-summed times otherwise."""
        return secs(a - b) if TICK.is_reference else secs(a) - secs(b)

    if args.act_every is not None \
            and int(args.act_every) != int(cfg.get("act_every", 1)):
        print(f"!! --act-every OVERRIDE: deciding every {int(args.act_every)} "
              f"ticks ({int(args.act_every) * TICK.ms:.1f} ms) instead of the "
              f"checkpoint's {int(cfg.get('act_every', 1))} "
              f"({int(cfg.get('act_every', 1)) * cfg_tick:.1f} ms).")
        cfg["act_every"] = int(args.act_every)
    if tick_override:
        print(f"!! --tick-ms OVERRIDE: searching at {TICK.describe()} instead "
              f"of the checkpoint's {cfg_tick:g} ms. The tick is part of the "
              f"dynamics these weights were trained under (the air-accelerate "
              f"impulse is per FRAME), so the proposals are NOT physics "
              f"parity; every time below is seconds at the real tick.")
    else:
        print(f"tick: {TICK.describe()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.torch_seed)
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    act_every = int(cfg.get("act_every", 1))
    K = act_every
    cfg_ep_ticks = int(cfg.get("ep_ticks", 700))
    if tick_override:
        # the checkpoint's episode cap counts ITS ticks: carry it over in
        # seconds (train_fast's tick transfer does the same)
        cfg_ep_ticks = TICK.secs_to_ticks(cfg_ep_ticks * cfg_tick / 1000.0,
                                          "round")
    ep_cap = max(cfg_ep_ticks, int(args.max_ticks))
    map_path = resolve_map(args.map, cfg.get("map", "surf_ski_2"))
    print(f"ckpt step {step:,}  act_every {K}  map {map_path}")
    _mt = int(args.max_ticks)
    print(f"caps: episode {ep_cap} ticks ({secs(ep_cap):.1f}s), search "
          f"--max-ticks {_mt} ticks ({secs(_mt):.1f}s)"
          + ("" if TICK.is_reference else
             f" - the same count is {_mt / 100.0:.1f}s at 10 ms; a run that "
             f"needs more than {secs(_mt):.1f}s at this tick is cut short "
             f"(raise --max-ticks)"))

    # ---- 1-env core: greedy sanity gate + spawn-state capture ----------
    core1 = build_sim(cfg, map_path, 1, ep_cap, tick=TICK)
    from surfgym.mapfleet import map_tag
    from surfgym.vision import GpuLidar, pick_cell
    cell = float((cfg.get("map_cells") or {}).get(
        map_tag(Path(map_path).stem),
        cfg.get("lidar_cell") or pick_cell(core1)))
    # goal-field cell: --goal-cell decoupled it from the lidar cell;
    # asking for an unbaked cell would rebuild a field already on disk
    gcell = cell
    gcells = cfg.get("goal_cells")
    gc = cfg.get("goal_cell")
    if isinstance(gcells, dict) and gcells:
        gcell = float(gcells.get(map_tag(Path(map_path).stem), gcell))
    elif isinstance(gc, str) and "," in gc:
        parts = [x.strip() for x in gc.split(",")]
        names = cfg.get("maps") or []
        tag = map_tag(Path(map_path).stem)
        i = next((j for j, m in enumerate(names) if map_tag(m) == tag), None)
        if i is not None and i < len(parts) and parts[i]:
            gcell = float(parts[i])
    elif gc:
        gcell = float(gc)

    from surfgym.goalfield import EuclidField, build_goal_field
    from surfgym.zones import load_zones
    zones = load_zones(core1.bsp_path)
    t0 = time.time()
    gf = (EuclidField(zones["end"]) if cfg.get("race_dist") == "euclid"
          else build_goal_field(core1, zones["end"], cell=gcell))
    dt = time.time() - t0
    print(f"goal field @ cell {gcell:g} in {dt:.1f}s"
          + ("  ** WARNING: that smells like a RE-BAKE - wrong map "
             "path/mtime? **" if dt > 30 else ""))

    raw = map_spawn_pool(core1)
    pool = map_spawn_pool(core1, yaw=gf.descent_yaw(raw["origin"]))
    pool["pitch"] = -10.0
    if cfg.get("fix_pitch") is not None:
        pool["pitch"] = float(cfg["fix_pitch"])
    d0 = float(np.mean(gf.sample(raw["origin"])))
    print(f"race: start geodesic {d0:.0f}u, spawn pool {len(pool)} points")

    def arm(core):
        core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
        if cfg.get("teleport_fail") or cfg.get("reward") == "race":
            core.set_teleport_fail(True)
        core.set_spawn_pool(pool)

    arm(core1)
    lidar = GpuLidar(core1, lw, lh,
                     range_units=float(cfg.get("lidar_range", 2000.0)),
                     near_range=cfg.get("lidar_near"),
                     cell=cell, device=device,
                     surf_mask=bool(cfg.get("surf_mask", 0)),
                     pinhole=bool(cfg.get("pinhole", 0)))
    stack = max(1, int(cfg.get("frame_stack") or 1))   # 1: refused above
    extra = (12,) if cfg.get("obs_reward") else ()
    # --race-latch: one observation column, concatenated LAST on the scalar
    # side (train_fast N_LATCH; record_ckpt latch_dim) - the checkpoint's
    # first Linear is one column wider and would not load without it
    n_latch = 1 if (float(cfg.get("race_latch") or 0.0) > 0.0
                    or float(cfg.get("race_latch_frac") or 0.0) > 0.0) else 0
    # --view-continuous is MIRRORED (record_ckpt's reason: it adds tensors
    # AND changes what an action is). Every proposal below then carries a
    # float view next to its int row: the wrappers publish `.view`, the
    # history keeps a (D, N, 2) twin of the action table, the npz a `view`.
    VIEWC = bool(cfg.get("view_continuous"))
    if cfg.get("view_absolute"):
        raise SystemExit(
            f"this checkpoint was trained with --view-absolute "
            f"{cfg['view_absolute']}: the planner's proposals, macros, "
            "branch commands and dedup all read the view row as a DELTA "
            "command (K, pitch rate); absolute targets are not "
            "implemented here (docs/contyaw.md, Absolute targets)")
    policy = Policy(core1.obs_dim + n_latch + lw * lh * lidar.channels * stack,
                    lw, lh,
                    emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)),
                    trunk=str(cfg.get("trunk") or "plain"),
                    tower_depth=int(cfg.get("tower_depth") or 2),
                    conv_mult=int(cfg.get("conv_mult") or 1),
                    extra_feat=extra,
                    in_ch=lidar.channels * stack,
                    n_codes=0, chunk=0, route_dim=n_latch,
                    route_critic_only=bool(cfg.get("route_critic_only")),
                    view_continuous=VIEWC
                    ).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    packer = HeadPacker(device)
    PITCH_MAX = float(core1.config.pitch_rate_max_deg)
    if VIEWC:
        _ls = policy.log_std().exp().tolist()
        print(f"--view-continuous: proposals draw z ~ N(mu, sigma) per view "
              f"head (sigma {_ls[0]:.3f}/{_ls[1]:.3f}), greedy envs take mu; "
              f"K = warp(tanh z), pitch = tanh z * {PITCH_MAX:g} deg/tick")

    def mk_feed():
        """--obs-reward slot-12 feed and the --race-latch flag, per
        train_fast's eval mirrors (surfgym.bc.make_eval_feeds - one copy
        shared with record_ckpt-style evals and the BC dataset builder, so
        the planner's proposals and the distillation rows see the very
        same side channels). A missing feed hands the policy absolute
        position where it expects tanh(reward) and kills the agent in
        seconds. d0 is the trainer's own formula (train_fast.py: mean field
        over the RAW map spawns) - record_ckpt samples pre-reset states
        there, which is a latent bug this tool does not copy.

        Returns (slot, fn, state, latch_fn); state['d'] and latch_fn.state
        ('f', 'tick') are per-env and must be cloned donor->loser at a
        resample (population mode) / snapshot->all at a commit."""
        slot, rf, lf = make_eval_feeds(cfg, gf, d0, K,
                                       tick_ms=TICK.requested_ms)
        return slot, rf, (None if rf is None else rf.state), lf

    header1 = {"map": Path(core1.bsp_path).stem,
               "tick_ms": int(core1.config.phys.msec),
               "phys": phys_header(core1),
               **tick_header(core1)}

    # ---- --prefix-line: the saved line whose state the search resumes ---
    # Parsed here because it REPLACES the greedy gate: the spawn state, the
    # gate seed and the action prefix all come out of the npz, and rolling a
    # gate episode would only re-derive a spawn this file already carries.
    pre_acts, pre_ticks = None, 0
    if args.prefix_line:
        spec = str(args.prefix_line)
        pspec, _, tspec = spec.rpartition(":")
        if pspec and tspec.isdigit():
            pl_path, pl_ticks = Path(pspec), int(tspec)
        else:
            pl_path, pl_ticks = Path(spec), 0
        pz = np.load(pl_path, allow_pickle=False)
        if int(pz["act_every"]) != K:
            raise SystemExit(f"--prefix-line act_every {int(pz['act_every'])}"
                             f" != this checkpoint's {K}")
        pre_acts = np.asarray(pz["acts"], np.int32)          # (D, 6)
        # a --view-continuous checkpoint replays a discrete line through
        # the bins' own view values (bit-identical, surfgym.view); a
        # discrete checkpoint cannot replay a continuous line
        pre_view = (np.asarray(pz["view"], np.float32).reshape(-1, 2)
                    if "view" in pz.files else None)
        if pre_view is not None and not VIEWC:
            raise SystemExit("--prefix-line carries a continuous `view` but "
                             "this checkpoint is discrete")
        if VIEWC and pre_view is None:
            pre_view = bin_to_view(pre_acts, bool(cfg.get("yaw_adaptive")),
                                   PITCH_MAX)
            pre_acts = pre_acts.copy()
            pre_acts[:, 0], pre_acts[:, 1] = NEUTRAL_ACT[0], NEUTRAL_ACT[1]
            print("--prefix-line: discrete line, view derived from its bins")
        full = len(pre_acts) * K
        pre_ticks = full if pl_ticks <= 0 else min(pl_ticks, full)
        pre_ticks -= pre_ticks % K            # switch on a decision boundary
        if pre_ticks <= 0 or pre_ticks >= max(1, int(args.max_ticks)):
            raise SystemExit(f"--prefix-line TICKS {pre_ticks} must be in "
                             f"(0, --max-ticks)")
        pre_row0 = pz["spawn_state"]
        pre_obs0 = np.asarray(pz["obs_start"], np.float32)
        pre_seed = int(pz["gate_seed"])
        print(f"--prefix-line {pl_path.name}: {len(pre_acts)} decisions "
              f"({full} ticks), replaying {pre_ticks} ticks "
              f"({secs(pre_ticks):.2f}s) into every env; gate skipped")

    # ---- phase 1: greedy sanity gate -----------------------------------
    # Fresh wrapper + fresh reset per episode so every episode's act_every
    # hold is aligned to its own tick 0, exactly like the search and the
    # replay (record_rollout's single global cadence would misalign
    # episodes 2+). The first FINISHING episode supplies the matched
    # spawn state row0 and the baseline time.
    gpath = out_dir / "greedy_baseline.jsonl"
    greedy_ticks, row0, obs_start = None, None, None
    nonfin, gate_seed = None, args.seed
    if pre_acts is not None:
        nonfin, gate_seed = (0, pre_row0, pre_obs0), pre_seed
        core1.reset(pre_seed)
    with open(gpath, "w", encoding="utf-8", newline="\n") as f:
        for e in range(0 if pre_acts is not None
                       else max(1, args.greedy_eps)):
            obs = core1.reset(args.seed + e)
            row = core1.get_states()[0:1].copy()      # STATE_DTYPE copy
            o0 = obs[0].copy()
            if args.skip_gate:
                nonfin = (0, row, o0)
                print(f"gate skipped: spawn state captured (seed "
                      f"{args.seed + e})")
                break
            es1, ef1, _, lf1 = mk_feed()
            gpol = GreedyTorchPolicy(policy, packer, device, lidar, core1,
                                     K, stack, extra_slot=es1, extra_fn=ef1,
                                     latch_fn=lf1)
            end, ticks, fin, _ = run_episode(core1, gpol, obs, f,
                                             ep_cap, header1, e)
            print(f"greedy ep{e} (spawn seed {args.seed + e}): {end} in "
                  f"{ticks} ticks ({secs(ticks):.2f}s)")
            if nonfin is None:
                # the FIRST episode's spawn is the one a wall search rides:
                # greedy is deterministic, so replaying it reproduces this
                # episode exactly
                nonfin = (ticks, row, o0)
                gate_seed = args.seed + e
            if fin:
                greedy_ticks, row0, obs_start = ticks, row, o0
                gate_seed = args.seed + e
                break
    allow_nonfin = (args.allow_nonfinisher or args.greedy_prefix > 0
                    or pre_acts is not None)
    if greedy_ticks is None:
        if not allow_nonfin:
            raise SystemExit(
                f"GATE FAILED: {Path(args.ckpt).name} did not finish in "
                f"{args.greedy_eps} greedy episode(s) - wrong checkpoint "
                "for a beam search; stopping per plan. Trajectories: "
                + str(gpath))
        _t, row0, obs_start = nonfin
        if pre_acts is None:
            print(f"gate: no finish in {args.greedy_eps} greedy episode(s) "
                  f"(first ran {_t} ticks) - continuing, this search does "
                  "not need one")
    else:
        print(f"greedy baseline: {greedy_ticks} ticks = "
              f"{secs(greedy_ticks):.2f}s -> {gpath}")
    if args.greedy_only:
        return

    # ---- phase 2: the search -------------------------------------------
    N = int(args.envs)
    R = int(args.resample_every)
    gen_ticks = R * K
    max_ticks = int(args.max_ticks)
    n_elite = max(1, int(round(N * args.elite_frac)))
    coreN = build_sim(cfg, map_path, N, ep_cap, tick=TICK)
    arm(coreN)
    obs = np.array(coreN.reset(args.seed))        # copy, then overwrite
    for i in range(N):
        coreN.set_state(i, row0)
    obs[:] = obs_start[None, :]
    esN, efN, feed_state, latchN = mk_feed()
    if (args.boundary_v or args.dedup) and args.commit <= 0:
        raise SystemExit("--boundary-v/--dedup are commit-mode features "
                         "(pass --commit H)")
    progress_mode = args.objective != "finish"
    if progress_mode and args.commit > 0:
        raise SystemExit("--objective progress/auto is a population-mode "
                         "feature and excludes --commit")
    if args.greedy_envs > 0:
        if args.eps > 0.0 or args.dedup or args.commit > 0:
            raise SystemExit("--greedy-envs is a population-mode feature "
                             "and excludes --eps/--dedup/--commit")
        if args.greedy_envs > N:
            raise SystemExit("--greedy-envs must be <= --envs")
        spol = MixedTorchPolicy(policy, packer, device, lidar, coreN,
                                K, stack, n_greedy=args.greedy_envs,
                                extra_slot=esN, extra_fn=efN,
                                latch_fn=latchN)
    elif args.eps > 0.0 or args.dedup:
        spol = EpsSampledTorchPolicy(policy, packer, device, lidar, coreN,
                                     K, stack, eps=args.eps,
                                     dedup=args.dedup,
                                     extra_slot=esN, extra_fn=efN,
                                     latch_fn=latchN)
    else:   # never route eps=0 through the mixer: RNG-stream parity
        spol = SampledTorchPolicy(policy, packer, device, lidar, coreN,
                                  K, stack, extra_slot=esN, extra_fn=efN,
                                  latch_fn=latchN)
    # --greedy-prefix: the SAME feed instance backs both wrappers, so the
    # obs-reward d-history is continuous across the switch (two feeds would
    # hand the sampler a zeroed slot on its first decision).
    prefix = int(args.greedy_prefix)
    if prefix % K:
        prefix -= prefix % K        # switch only on a decision boundary
        print(f"--greedy-prefix rounded down to {prefix} (act_every {K})")
    gpolN = None
    if prefix > 0:
        gpolN = GreedyTorchPolicy(policy, packer, device, lidar, coreN,
                                  K, stack, extra_slot=esN, extra_fn=efN,
                                  latch_fn=latchN)
    value_fn = None
    if args.boundary_v or args.score in ("v", "dv"):
        def value_fn(o):
            # one extra batched critic forward on the boundary obs. The
            # obs assembly is the wrapper's own (_obs), so it matches
            # what a decision would see; the obs-reward feed's per-env
            # state is snapshotted and restored around it because _obs
            # advances it as a side effect.
            saved = None if feed_state is None else feed_state.get("d")
            lsaved = (None if latchN is None
                      else (latchN.state.get("f"), latchN.state.get("tick")))
            with torch.inference_mode():
                _, v = policy(spol._obs(o))
            if feed_state is not None:
                feed_state["d"] = saved
            if lsaved is not None:
                latchN.state["f"], latchN.state["tick"] = lsaved
            return v.detach().float().reshape(-1).cpu().numpy()
    scorer = make_scorer(gf, args.route_file, args.corridor, args.score,
                         value_fn=value_fn, v_switch=args.v_switch)

    # ---- --branch-at / --branch-grid: the junction fork -----------------
    br_kind, br_val, br_n = None, 0.0, 0
    br_plans, br_meta, br_assign = None, None, None
    if args.branch_at and args.branch_grid:
        raise SystemExit("--branch-at and --branch-grid are the same fork "
                         "with two different fills; pick one")
    if args.branch_at or args.branch_grid:
        _flag = "--branch-at" if args.branch_at else "--branch-grid"
        if args.commit > 0:
            raise SystemExit(f"{_flag} is a population-mode feature "
                             "and excludes --commit")
        where, _, nstr = str(args.branch_at or args.branch_grid
                             ).partition(":")
        if not where or not nstr:
            raise SystemExit(f"{_flag} wants WHERE:"
                             + ("N, e.g. d7900:64" if args.branch_at
                                else "SPEC, e.g. t8600:hold=42,84"))
        br_kind, br_val = where[0], float(where[1:])
        if br_kind not in ("t", "d", "a"):
            raise SystemExit(f"{_flag} WHERE must start with t, d or a")
        if br_kind == "a" and args.objective == "finish":
            raise SystemExit(f"{_flag} aVALUE needs --objective "
                             "progress/auto (the arc tracker)")
        if args.branch_grid:
            # the whole non-greedy population carries the grid; plans are
            # replicated round-robin so every plan gets floor(n/P) envs and
            # the first n%P get one more - deterministic, no draw
            br_n = N - max(0, args.greedy_envs)
            br_plans, br_meta = branch_grid_parse(nstr, K, view=VIEWC)
            if br_n < br_meta["plans"]:
                raise SystemExit(f"--branch-grid: {br_meta['plans']} plans "
                                 f"but only {br_n} forkable envs")
            br_assign = np.arange(br_n) % br_meta["plans"]
            print(f"--branch-grid: {br_meta['plans']} plans "
                  f"({len(br_meta['yaw'])} yaw offsets x "
                  f"{len(br_meta['side'])} side keys x "
                  f"{len(br_meta['hold'])} holds x {br_meta['seg']} seg) "
                  f"over {br_n} envs, {br_n // br_meta['plans']}-"
                  f"{-(-br_n // br_meta['plans'])} envs each; macro at most "
                  f"{br_meta['max_ticks']} ticks "
                  f"({secs(br_meta['max_ticks']):.2f}s)")
        else:
            if not nstr.isdigit():
                raise SystemExit("--branch-at wants WHERE:N, e.g. d7900:64")
            br_n = int(nstr)
            if not 0 < br_n <= N - max(0, args.greedy_envs):
                raise SystemExit(f"--branch-at N must be in (0, "
                                 f"{N - max(0, args.greedy_envs)}]")
            # both windows must be whole DECISIONS: the burst overrides the
            # action the wrapper holds for K ticks, and hist records it once
            # per decision, so a redraw inside a decision would make the
            # winner's open-loop replay disagree with the search
            for _nm in ("branch_burst", "branch_hold"):
                _v = int(getattr(args, _nm))
                if _v % K:
                    _v += K - _v % K
                    print(f"--{_nm.replace('_', '-')} rounded up to {_v} "
                          f"(act_every {K})")
                    setattr(args, _nm, _v)
    if args.prefix_line and (args.commit > 0 or args.greedy_prefix > 0):
        raise SystemExit("--prefix-line excludes --commit/--greedy-prefix")

    # ---- --macro-hold: the held-key proposal ---------------------------
    macro, macro_lo, macro_hi = None, 0.0, 0.0
    if args.macro_hold:
        if args.commit > 0:
            raise SystemExit("--macro-hold is a population-mode feature "
                             "and excludes --commit")
        spec = str(args.macro_hold)
        try:
            lo_s, hi_s = (float(x) for x in spec.split(":"))
        except ValueError:
            raise SystemExit("--macro-hold wants MIN:MAX seconds, "
                             "e.g. 0.2:0.8")
        if not (MACRO_MIN_S <= lo_s <= hi_s <= MACRO_MAX_S):
            raise SystemExit(f"--macro-hold MIN:MAX must satisfy "
                             f"{MACRO_MIN_S} <= MIN <= MAX <= {MACRO_MAX_S}")
        g0 = max(0, int(args.greedy_envs))
        if g0 >= N:
            raise SystemExit("--macro-hold needs at least one non-greedy "
                             "env to propose from")
        if not 0.0 < float(args.macro_frac) <= 1.0:
            raise SystemExit("--macro-frac must be in (0, 1]")
        # the macro envs are the LAST ones, so --greedy-envs keeps [0, G)
        # and the policy-sampled envs sit between the two blocks
        n_macro = max(1, int(round(float(args.macro_frac) * (N - g0))))
        m_lo = N - n_macro
        macro_lo, macro_hi = lo_s, hi_s
        dec_s = secs(K)                 # one decision, in real seconds
        macro = MacroHold(N, np.arange(m_lo, N), lo_s, hi_s, dec_s,
                          np.random.default_rng(args.seed
                                                if args.macro_seed is None
                                                else args.macro_seed),
                          yaw=args.macro_yaw, fwd=args.macro_fwd)
        print(f"--macro-hold {lo_s:g}:{hi_s:g}s: envs [{m_lo}, {N}) propose "
              f"HELD keys - side"
              + ("" if args.macro_fwd == "policy" else f" + fwd "
                 f"({args.macro_fwd})")
              + f", drawn log-uniform over "
              f"{max(1, int(round(lo_s / dec_s)))}-"
              f"{max(1, int(round(hi_s / dec_s)))} decisions "
              f"({dec_s * 1000:.1f} ms each), yaw from {args.macro_yaw}, "
              f"seed {args.seed if args.macro_seed is None else args.macro_seed}"
              + (f"; envs [0, {g0}) stay the greedy floor" if g0 else "")
              + (f"; envs [{g0}, {m_lo}) keep proposing from the policy"
                 if m_lo > g0 else ""))

    if args.commit > 0:
        # -- receding-horizon (MPC) mode --
        H = int(args.commit)
        C = max(1, min(H, int(round(args.commit_frac * H))))
        print(f"receding-horizon search: {N} envs, window H={H} decisions "
              f"({H * K} ticks = {secs(H * K):.2f}s), commit {C} ({C * K} "
              f"ticks = {secs(C * K):.2f}s), eps {args.eps:g}, cap "
              f"{max_ticks} ticks ({secs(max_ticks):.1f}s)")
        t_loop = time.time()
        best, cinfo, dnf = commit_search(coreN, spol, gf, obs, N, K, H, C,
                                         max_ticks, feed_state,
                                         value_fn=value_fn, latch_fn=latchN,
                                         tick=TICK, viewc=VIEWC)
        dt_loop = time.time() - t_loop
        fps = cinfo["sim_ticks"] * N / max(dt_loop, 1e-9)
        print(f"search done: {cinfo['sim_ticks']} sim ticks x {N} envs in "
              f"{dt_loop:.0f}s ({fps:,.0f} env-steps/s), "
              f"{cinfo['windows']} windows")
        sinfo = {"mode": "commit", "eps": args.eps, "commit": H,
                 "commit_frac": args.commit_frac,
                 "boundary_v": bool(args.boundary_v),
                 "dedup": bool(args.dedup),
                 "windows": cinfo["windows"],
                 "committed_ticks": cinfo["committed_ticks"]}
        for k in ("rank_agreement", "collision_before", "collision_after"):
            if k in cinfo:
                sinfo[k] = cinfo[k]
        if args.dedup and getattr(spol, "dd_stats", None):
            print(f"dedup: prefix collisions before/after = "
                  f"{sinfo.get('collision_before', 0):.1%} / "
                  f"{sinfo.get('collision_after', 0):.1%} of {N} "
                  f"(mean over {len(spol.dd_stats)} windows)")
        if best is None:
            # DNF is dose information, not failure: write it down and
            # exit cleanly so a campaign driver can read it
            summary = {"ckpt": str(args.ckpt), "map": map_path, "envs": N,
                       **tick_json,
                       **sinfo, "dnf": True, "dnf_reason": dnf,
                       "greedy_ticks": greedy_ticks,
                       "greedy_s": (secs(greedy_ticks) if greedy_ticks
                                    else None),
                       "search_wall_s": round(dt_loop, 1),
                       "env_steps_per_s": round(fps)}
            (out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
            print(f"beam TAS: DNF ({dnf}); greedy was "
                  + (f"{secs(greedy_ticks):.2f}s" if greedy_ticks
                     else "no finish"))
            return
        del dnf
        v1_search = False
        fin_hist = [(best["tick"], best["acts"].copy(), 0.0, 0,
                     best.get("view"))]
    else:
        v1_search = True
    if v1_search:
        D_total = (max_ticks + K - 1) // K
        hist = np.zeros((D_total, N, 6), np.int8)  # per-decision actions
        # --view-continuous: the per-decision view twin of `hist`, cloned,
        # captured and saved beside it everywhere hist is
        hist_v = np.zeros((D_total, N, 2), np.float32) if VIEWC else None
        valid = np.ones(N, bool)   # history describes this env from tick 0
        idx = np.arange(N)
        # ---- --prefix-line: open-loop replay into every env -------------
        # Physics + the two scalar feeds only. The feeds are pure functions
        # of the core state per DECISION, so advancing them by hand here is
        # exactly what _obs would have done - and skipping the lidar and the
        # forward is what makes a 65 s prefix cost 6 s instead of 30 min.
        t_start, gen_origin = 0, 0
        if pre_acts is not None:
            tP = time.time()
            for t in range(pre_ticks):
                if t % K == 0:
                    if efN is not None:
                        efN(coreN)
                    if latchN is not None:
                        latchN(coreN)
                    hist[t // K] = pre_acts[t // K][None, :]
                    if VIEWC:
                        hist_v[t // K] = pre_view[t // K][None, :]
                a = np.ascontiguousarray(
                    np.repeat(pre_acts[t // K].reshape(1, 6), N, axis=0),
                    dtype=np.int32)
                if VIEWC:
                    obs, _r, _dn, _tr, _tm = coreN.step(
                        a, view=np.ascontiguousarray(np.repeat(
                            pre_view[t // K].reshape(1, 2), N, axis=0),
                            dtype=np.float32))
                else:
                    obs, _r, _dn, _tr, _tm = coreN.step(a)
                if np.asarray(_dn).any() or np.asarray(_tr).any():
                    raise SystemExit(f"--prefix-line died at tick {t}: the "
                                     "npz's line does not replay on this "
                                     "checkpoint/map/tick")
            t_start = gen_origin = pre_ticks
            _po = coreN.states_view["origin"]
            assert np.abs(_po - _po[0][None, :]).max() < 1e-6, \
                "--prefix-line: envs diverged during an open-loop replay"
            print(f"prefix-line: {pre_ticks} ticks x {N} envs in "
                  f"{time.time() - tP:.0f}s -> origin "
                  f"({_po[0][0]:.0f}, {_po[0][1]:.0f}, {_po[0][2]:.0f}), "
                  f"d={gf.sample(_po)[0]:,.0f}u; search resumes at "
                  f"{secs(pre_ticks):.2f}s")
        # ---- --branch-at state ------------------------------------------
        br_rng = (np.random.default_rng(args.torch_seed
                                        if args.branch_seed is None
                                        else args.branch_seed)
                  if br_kind else None)
        br_idx = (idx[N - br_n:] if br_kind else None)   # the forked envs
        br_tag = np.zeros(N, bool)      # descends from the fork; cloned
        # --branch-grid: which PLAN a lineage descends from, cloned with the
        # history exactly as br_tag is, so the winner can be named
        br_who = np.full(N, -1, np.int32)
        br_fired, br_gen0, br_until, br_act = -1, -1, -1, None
        br_live, br_el, br_d = 0, 0, float("nan")     # per-gen diagnostics
        finishes = []              # (tick, env, gen)
        # --keep-finishers: the fastest K finishing lineages' whole action
        # histories, captured AT the finish (the env autoresets on that
        # tick and the next resample clones over it). Lockstep means the
        # first K hits are the K fastest, so a bounded append is enough.
        fin_hist = []              # (tick, (D, 6) int8 acts, arc, arc_tick)
        best = None
        gen, best_gen = 0, None
        dth_t, dth_o = [], []
        frontier = {"vert": -1, "d": float("inf"), "tick": -1}
        # --objective progress/auto: the order-only arc coordinate per env
        # (eval_honesty --order-only: ArcProgress's local-window rule), its
        # per-env anchor cloned with the state at every resample; the best
        # arc a lineage reached, when, and a hall of the K best lineages
        # that DIED (their action tables captured at death, as fin_hist
        # captures a finisher's at the finish)
        arcp, arc_total = None, 0.0
        if progress_mode:
            pts_arc, sp_arc = load_route(Path(args.route_file))
            arcp = ArcProgress(np.asarray(pts_arc, np.float64), sp_arc,
                               corridor=args.corridor, window=args.arc_window,
                               source=str(args.route_file))
            arc_total = float(arcp.length)
            arcp.reset(np.asarray(coreN.states_view["origin"], np.float64))
            # raw_arc: the furthest arc a lineage's live positions reached;
            # best_arc: the arc it BANKED - raw_arc as of its last map-
            # contact tick (--arc-bank contact), so the part of a fall that
            # happens to run along the route earns nothing (a fall's arc
            # would otherwise be trimmed away by plan_to_bc and the line
            # ranked on progress it does not carry). Under --arc-bank raw
            # the two are the same array.
            raw_arc = arcp.arc.copy()
            raw_tick = np.zeros(N, np.int64)
            best_arc = arcp.arc.copy()
            arc_tick = np.zeros(N, np.int64)
            g_step = gravity_step(coreN)
            bank_contact = args.arc_bank == "contact"
            hall = LineageHall(int(args.keep_finishers))
            print(f"objective {args.objective}: {arcp.describe()}, elites by "
                  f"best arc" + (f" in {args.arc_quant:g}u bins"
                                 if args.arc_quant > 0 else "")
                  + f", ties by score {args.score}, arc banked at "
                  + (f"map contact (|dvz - {g_step:g}| > {args.contact_tol:g})"
                     if bank_contact else "every live tick (raw)"))
        print(f"search: {N} envs, resample every {R} decisions "
              f"({gen_ticks} ticks = {secs(gen_ticks):.3f}s), elite {n_elite}, "
              f"cap {max_ticks} ticks ({secs(max_ticks):.1f}s)"
              + (f", greedy prefix {prefix} ticks ({secs(prefix):.2f}s)"
                 if prefix else "") + f", score {args.score}")
        t_loop = time.time()
        for t in range(t_start, max_ticks):
            d = t // K
            # greedy (deterministic, all envs identical) up to the prefix,
            # then the policy's own sampling supplies the diversity
            if t < prefix:
                a = gpolN.act(obs)
                v = gpolN.view if VIEWC else None
            else:
                a = spol.act(obs)
                v = spol.view if VIEWC else None
            if macro is not None and t >= prefix:
                # --macro-hold: the held-key proposal. Decided ONCE per
                # decision, exactly where hist records its row, and reused
                # for the rest of the act_every hold - recomputing the
                # analytic yaw every tick would make the winner's
                # open-loop replay (one row per decision) disagree with
                # the search. Applied BEFORE --branch-at, so a fork still
                # overrides its own envs.
                if t % K == 0:
                    if VIEWC:
                        a, v = macro.decide_view(
                            a, v, coreN.states_view["velocity"],
                            bool(coreN.config.yaw_adaptive),
                            float(coreN.config.yaw_rate_max_deg))
                    else:
                        a = macro.decide(a, coreN.states_view["velocity"],
                                         bool(coreN.config.yaw_adaptive),
                                         float(coreN.config.yaw_rate_max_deg))
                    macro_held = (a, v)
                else:
                    a, v = macro_held
            if br_until > t and br_plans is not None:
                # --branch-grid: each forked env runs ITS OWN enumerated
                # macro plan, offset k ticks after the fork. Nothing is
                # drawn, so the torch stream and every unbranched env's
                # proposal are untouched by construction.
                if VIEWC:
                    a, v = branch_grid_apply_view(a, v, br_idx, br_plans,
                                                  br_assign, t - br_fired)
                else:
                    a = branch_grid_apply(a, br_idx, br_plans, br_assign,
                                          t - br_fired)
            elif br_until > t:
                # --branch-at: hold one random (yaw, side) pair per forked
                # env for --branch-hold ticks. The draws come from a private
                # numpy generator, so the torch stream (and every other
                # env's proposal) is untouched; only the two heads that
                # steer a surf flight are overridden, the rest stay the
                # policy's own choice.
                if (t - br_fired) % max(1, args.branch_hold) == 0:
                    br_act = (branch_draw_view(br_rng, br_n,
                                               float(args.branch_jitter))
                              if VIEWC else
                              branch_draw(br_rng, br_n,
                                          int(args.branch_jitter)))
                if VIEWC:
                    a, v = branch_apply_view(a, v, br_idx, br_act[0],
                                             br_act[1],
                                             float(args.branch_jitter))
                else:
                    a = branch_apply(a, br_idx, br_act,
                                     int(args.branch_jitter))
            if t % K == 0:
                hist[d] = a                # bins < 15: int8 is lossless
                if VIEWC:
                    hist_v[d] = v
            sv = coreN.states_view
            pre_o = sv["origin"].copy()
            pre_v = sv["velocity"].copy()
            if VIEWC:
                obs, _rew, done, trunc, _term = coreN.step(
                    a, view=np.ascontiguousarray(v, dtype=np.float32))
            else:
                obs, _rew, done, trunc, _term = coreN.step(a)
            gh = coreN.goal_hits
            dead = np.asarray(done, bool) | np.asarray(trunc, bool)
            if arcp is not None:
                # the post-step position of every LIVE lineage; an env that
                # died this tick already holds its respawn, so its arc is
                # frozen at its last live reading
                arcp.advance(coreN.states_view["origin"])
                live = valid & ~dead
                imp = live & (arcp.arc > raw_arc)
                if imp.any():
                    raw_arc[imp] = arcp.arc[imp]
                    raw_tick[imp] = t + 1
                if bank_contact:
                    dvz = coreN.states_view["velocity"][:, 2] - pre_v[:, 2]
                    contact = live & (np.abs(dvz - g_step) > args.contact_tol)
                    bimp = contact & (raw_arc > best_arc)
                else:
                    bimp = imp
                if bimp.any():
                    best_arc[bimp] = raw_arc[bimp]
                    arc_tick[bimp] = raw_tick[bimp]
            if gh.any():
                for i in np.nonzero(gh)[0]:
                    if not valid[i]:
                        continue           # respawned body: not a real run
                    finishes.append((t + 1, int(i), gen))
                    if len(fin_hist) < int(args.keep_finishers):
                        fin_hist.append((t + 1, hist[:d + 1, i].copy(),
                                         (float(best_arc[i]) if arcp is not None
                                          else 0.0),
                                         (int(arc_tick[i]) if arcp is not None
                                          else 0),
                                         (hist_v[:d + 1, i].copy() if VIEWC
                                          else None)))
                    if best is None:       # lockstep: first hit is fastest
                        best = {"tick": t + 1, "env": int(i),
                                "acts": hist[:d + 1, i].copy(),
                                "view": (hist_v[:d + 1, i].copy() if VIEWC
                                         else None),
                                "pre_origin": pre_o[i].copy(),
                                "pre_vel": pre_v[i].copy()}
                        best_gen = gen
                        print(f"FINISH: env {i} at tick {t + 1} "
                              f"({secs(t + 1):.2f}s), gen {gen}"
                              + ((f" [BRANCH lineage, plan "
                                  f"{int(br_who[i])} = "
                                  f"{br_plans[int(br_who[i])]}]"
                                  if br_plans is not None and br_who[i] >= 0
                                  else " [BRANCH lineage]") if br_tag[i]
                                 else (" [not a branch lineage]"
                                       if br_fired >= 0 else "")))
            if dead.any():
                # forensics for a search that never crosses: WHERE the real
                # lineages ended, at their last live position
                newly = np.nonzero(dead & valid)[0]
                if len(newly):
                    dth_t.append(np.full(len(newly), t + 1, np.int64))
                    dth_o.append(pre_o[newly].copy())
                if arcp is not None and len(newly):
                    # a lineage that dies keeps its best arc: offer its
                    # table to the hall (a finish is fin_hist's business)
                    ghb = np.asarray(gh, bool)
                    for i in newly:
                        if ghb[i]:
                            continue
                        hall.offer(best_arc[i], arc_tick[i], t + 1,
                                   lambda i=i, d=d: hist[:d + 1, i].copy(),
                                   raw_arc=raw_arc[i],
                                   get_view=((lambda i=i, d=d:
                                              hist_v[:d + 1, i].copy())
                                             if VIEWC else None))
                valid &= ~dead
            if best is not None and args.gens > 0 \
                    and gen - best_gen >= args.gens:
                break
            if ((t + 1 - gen_origin) % gen_ticks == 0 and (t + 1) < max_ticks
                    and (t + 1) > prefix):
                # no cloning during the greedy prefix: every env is the
                # same state, so there is nothing to select between
                gen += 1
                states = coreN.get_states()
                sc, dgeo = scorer(states, obs)
                if arcp is not None:
                    # best arc first (quantized if asked), the --score
                    # value inside a tie; invalid rows sort last. lexsort
                    # is stable and keys are applied last-to-first.
                    akey = best_arc if args.arc_quant <= 0 else \
                        np.floor(best_arc / args.arc_quant)
                    order = np.lexsort((np.where(valid, -sc, np.inf),
                                        np.where(valid, -akey, np.inf)))
                else:
                    order = np.argsort(np.where(valid, -sc, np.inf),
                                       kind="stable")
                elig = order[valid[order]]
                if len(elig) == 0:
                    # every lineage finished or died inside this window; a
                    # reseed would restart episode clocks against the
                    # global tick and corrupt the finish-time accounting,
                    # so stop
                    print(f"gen {gen}: population extinct at t={t + 1} "
                          f"(all lineages finished/died); stopping search")
                    break
                # lockstep invariant: every valid env's episode clock is
                # the global tick (clones inherit the donor's; a violation
                # means the finish-time accounting is wrong)
                assert int(states["tick"][elig[0]]) == t + 1, \
                    (int(states["tick"][elig[0]]), t + 1)
                keep = elig[:n_elite]
                keep_set = np.zeros(N, bool)
                keep_set[keep] = True
                if args.greedy_envs > 0 and valid[0]:
                    # env 0 = the untouched greedy line: never cloned
                    # over while alive, so its finish is the floor
                    keep_set[0] = True
                if br_kind and 0 <= br_gen0 and gen - br_gen0 < max(
                        0, args.branch_protect):
                    # --branch-protect: a manoeuvre that trades speed now
                    # for position later loses the very next boundary, so
                    # the branch's own descendants are exempt from being
                    # cloned over for this many generations. They are NOT
                    # given donor slots: protection buys a horizon, not a
                    # share of the population.
                    keep_set |= br_tag & valid
                losers = idx[~keep_set]
                donors = keep[np.arange(len(losers)) % len(keep)]
                for j, don in zip(losers, donors):
                    coreN.set_state(int(j), states[don])
                hist[:d + 1, losers] = hist[:d + 1, donors]
                if VIEWC:
                    hist_v[:d + 1, losers] = hist_v[:d + 1, donors]
                obs = np.array(obs)        # patch clones' scalar obs too
                obs[losers] = obs[donors]
                valid[:] = True
                if feed_state is not None \
                        and feed_state.get("d") is not None:
                    feed_state["d"][losers] = feed_state["d"][donors]
                if latchN is not None and latchN.state.get("f") is not None:
                    # the latch is episode history and the loser now
                    # carries the donor's history
                    latchN.state["f"][losers] = latchN.state["f"][donors]
                    latchN.state["tick"][losers] = \
                        latchN.state["tick"][donors]
                if macro is not None:
                    # --macro-hold: a macro in progress travels with the
                    # history. A generation is shorter than the holds
                    # under test, so redrawing here would truncate every
                    # one of them (MacroHold's docstring).
                    macro.clone(losers, donors)
                if arcp is not None:
                    # the arc anchor and the lineage's record travel with
                    # the history: the loser IS the donor's lineage now
                    arcp.arc[losers] = arcp.arc[donors]
                    arcp.idx[losers] = arcp.idx[donors]
                    best_arc[losers] = best_arc[donors]
                    arc_tick[losers] = arc_tick[donors]
                    raw_arc[losers] = raw_arc[donors]
                    raw_tick[losers] = raw_tick[donors]
                if br_kind is not None:
                    # Read the diagnostics BEFORE propagating the tag: sc /
                    # dgeo and `elig` describe the PRE-clone population, so
                    # pairing them with post-clone ancestry would credit a
                    # loser's old distance to its new lineage. elig is
                    # exactly the set that was alive at this boundary.
                    _bm = br_tag[elig]
                    br_live, br_el = int(_bm.sum()), int(br_tag[keep].sum())
                    br_d = (float(dgeo[elig][_bm].min()) if br_live
                            else float("nan"))
                    # the tag IS lineage identity: a loser now carries the
                    # donor's history, so it carries the donor's ancestry
                    br_tag[losers] = br_tag[donors]
                    br_who[losers] = br_who[donors]
                    if br_fired < 0:
                        lead = (float(dgeo[keep[0]]) if br_kind == "d"
                                else (float(best_arc[keep[0]])
                                      if br_kind == "a" else float(t + 1)))
                        hit = (lead <= br_val if br_kind == "d"
                               else lead >= br_val)
                        if hit:
                            br_fired, br_gen0 = t + 1, gen
                            _burst = (br_meta["max_ticks"] if br_plans
                                      else max(0, args.branch_burst))
                            br_until = t + 1 + _burst
                            br_tag[:] = False
                            br_tag[br_idx] = True
                            br_who[:] = -1
                            if br_plans is not None:
                                br_who[br_idx] = br_assign
                            print(f"BRANCH: fork {br_n} envs at tick "
                                  f"{t + 1} ({secs(t + 1):.2f}s), gen {gen}, "
                                  f"d={dgeo[keep[0]]:,.0f}u - "
                                  + (f"grid {br_meta['plans']} plans, macro "
                                     f"<= {_burst} ticks "
                                     f"({secs(_burst):.2f}s)" if br_plans
                                     else f"burst {args.branch_burst} ticks "
                                     f"({secs(args.branch_burst):.2f}s), "
                                     f"hold {args.branch_hold}")
                                  + f", protect {args.branch_protect} gens")
                lead_vert = int(sc[keep[0]] // 1e6) if args.score == "route" \
                    else -1
                if (lead_vert > frontier["vert"]
                        or (lead_vert == frontier["vert"]
                            and dgeo[keep[0]] < frontier["d"])):
                    frontier = {"vert": lead_vert,
                                "d": float(dgeo[keep[0]]), "tick": t + 1}
                if gen % max(1, args.log_every) == 0:
                    print(f"gen {gen:3d} t={t + 1:5d} valid={len(elig):4d} "
                          + (f"vert={lead_vert:5d} " if lead_vert >= 0 else "")
                          + (f"arc={best_arc[keep[0]]:8.0f} "
                             f"({100 * best_arc[keep[0]] / max(arc_total, 1):4.1f}%) "
                             f"med_arc={np.median(best_arc[keep]):8.0f} "
                             if arcp is not None else "")
                          + f"min_d={dgeo[keep[0]]:8.0f} "
                          f"med_d={np.median(dgeo[keep]):8.0f} "
                          + (f"br={br_live:4d}/{br_el:4d}el "
                             f"brd={br_d:8.0f} "
                             if br_fired >= 0 else "")
                          + f"finishes={len(finishes)}")
        dt_loop = time.time() - t_loop
        n_ticks = t + 1 - t_start
        fps = n_ticks * N / max(dt_loop, 1e-9)
        print(f"search done: {n_ticks} ticks x {N} envs in {dt_loop:.0f}s "
              f"({fps:,.0f} env-steps/s), {len(finishes)} finishes, "
              f"{gen} generations"
              + (f"; branch lineages alive at the end: "
                 f"{int((br_tag & valid).sum())} of {br_n}"
                 if br_fired >= 0 else
                 ("; BRANCH NEVER FIRED" if br_kind else "")))
        sinfo = {"mode": "population", "eps": args.eps,
                 "resample_every_decisions": R,
                 "gen_ticks": gen_ticks, "gen_s": secs(gen_ticks),
                 "elite_frac": args.elite_frac,
                 "greedy_prefix": prefix, "score": args.score,
                 "prefix_line": (str(args.prefix_line) if pre_acts is not None
                                 else None),
                 "prefix_ticks": int(t_start),
                 "prefix_s": (secs(t_start) if t_start else None),
                 "branch_at": args.branch_at,
                 "branch_grid": args.branch_grid,
                 "branch_grid_spec": br_meta,
                 "branch_grid_plans": ([list(p) for p in br_plans]
                                       if br_plans is not None else None),
                 "branch_grid_alive_end": (
                     sorted({int(w) for w in br_who[valid] if w >= 0})
                     if br_plans is not None and br_fired >= 0 else None),
                 "branch_grid_finishers": (
                     [[int(_ft), int(br_who[_i])] for _ft, _i, _g in finishes
                      if br_who[_i] >= 0]
                     if br_plans is not None and br_fired >= 0 else None),
                 # the burst/hold/jitter window is --branch-at's; a grid
                 # fork sizes its own window per plan, so recording them
                 # here would describe a knob that did nothing
                 "branch_burst": (int(args.branch_burst)
                                  if br_kind and br_plans is None else None),
                 "branch_hold": (int(args.branch_hold)
                                 if br_kind and br_plans is None else None),
                 "branch_jitter": (int(args.branch_jitter)
                                   if br_kind and br_plans is None else None),
                 "branch_protect": (int(args.branch_protect) if br_kind
                                    else None),
                 "branch_n": (int(br_n) if br_kind else None),
                 "branch_fired_tick": (int(br_fired) if br_fired >= 0
                                       else None),
                 "branch_fired_s": (secs(br_fired) if br_fired >= 0
                                    else None),
                 "branch_alive_end": (int((br_tag & valid).sum())
                                      if br_fired >= 0 else None),
                 "branch_finishes": (int(sum(1 for _ft, _i, _g in finishes
                                             if br_tag[_i]))
                                     if br_fired >= 0 else None),
                 "macro_hold": (args.macro_hold if macro is not None
                                else None),
                 "macro_hold_min_s": (macro_lo if macro is not None else None),
                 "macro_hold_max_s": (macro_hi if macro is not None else None),
                 "macro_yaw": (args.macro_yaw if macro is not None else None),
                 "macro_fwd": (args.macro_fwd if macro is not None else None),
                 "macro_frac": (float(args.macro_frac) if macro is not None
                                else None),
                 "macro_envs": (int(len(macro.idx)) if macro is not None
                                else None),
                 "macro_draws": (int(macro.draws) if macro is not None
                                 else None),
                 "greedy_envs": int(args.greedy_envs),
                 "v_switch": (args.v_switch if args.score == "dv" else None),
                 "finishes": len(finishes),
                 "finish_ticks": sorted(ft for ft, _, _ in finishes),
                 "generations": gen,
                 "objective": args.objective}
        if arcp is not None:
            sinfo.update(arc_window=int(args.arc_window),
                         arc_quant=float(args.arc_quant),
                         arc_corridor=float(args.corridor),
                         arc_total=arc_total, route_file=str(args.route_file),
                         arc_bank=args.arc_bank, gravity_step=g_step,
                         contact_tol=float(args.contact_tol),
                         raw_arc_max=float(raw_arc[valid].max())
                         if valid.any() else None)

        # every kept lineage: the finishers (fastest first) and, under
        # --objective progress/auto, the best-arc lineages that died or
        # were still alive when the search stopped (rank_lineages orders
        # and de-duplicates them: finishers by time, then arc desc /
        # arc tick asc)
        cands = [{"finish_tick": int(ft), "acts": np.asarray(ah, np.int8),
                  "best_arc": float(fa), "arc_tick": int(fat),
                  "end_tick": int(ft), "raw_arc": float(fa),
                  "view": (None if av is None
                           else np.asarray(av, np.float32))}
                 for ft, ah, fa, fat, av in fin_hist]
        if arcp is not None:
            for i in np.nonzero(valid)[0]:
                hall.offer(best_arc[i], arc_tick[i], t + 1,
                           lambda i=i, d=d: hist[:d + 1, i].copy(),
                           raw_arc=raw_arc[i],
                           get_view=((lambda i=i, d=d:
                                      hist_v[:d + 1, i].copy())
                                     if VIEWC else None))
            cands += list(hall.items)
        lines = rank_lineages(cands, int(args.keep_finishers))
        if int(getattr(args, 'robust', 0) or 0) > 0 and len(lines) > 1:
            lines = robust_rerank(lines, coreN, row0, K, int(args.robust),
                                  float(args.robust_jitter),
                                  np.random.default_rng(int(args.seed) + 7))

        def verify_lines(lines):
            """Replay every NON-finishing kept lineage open-loop from the
            spawn and require the arc it reaches to be the arc the search
            credited (the progress twin of the finisher's bit-exact finish
            assert). A lineage whose replay diverges is dropped and
            counted; the finishers are asserted by phase 3 / plan_to_bc."""
            if arcp is None:
                return list(lines), 0
            core1v = build_sim(cfg, map_path, 1, ep_cap, tick=TICK)
            arm(core1v)
            arcp1 = ArcProgress(np.asarray(pts_arc, np.float64), sp_arc,
                                corridor=args.corridor,
                                window=args.arc_window)
            ok, bad = [], 0
            for c in lines:
                if c["finish_tick"] > 0:
                    ok.append(c)
                    continue
                core1v.reset(gate_seed)
                at = np.repeat(c["acts"].astype(np.int32), K, axis=0)
                ra, rat, ret, _ended, fin, rraw = replay_arc(
                    core1v, row0, at, arcp1, ep_cap, g_step,
                    contact_tol=args.contact_tol, bank=args.arc_bank,
                    view_ticks=line_view_ticks(c.get("view"), K))
                c["replay_arc"], c["replay_arc_tick"] = float(ra), int(rat)
                c["replay_end_tick"], c["replay_raw_arc"] = int(ret), float(rraw)
                c["replay_ok"] = bool(abs(ra - c["best_arc"]) <= 0.5
                                      and not fin)
                if c["replay_ok"]:
                    ok.append(c)
                else:
                    bad += 1
                    print(f"  lineage DIVERGED on replay: search arc "
                          f"{c['best_arc']:,.1f}u at tick {c['arc_tick']}, "
                          f"replay {ra:,.1f}u at tick {rat}"
                          + (" (finished!)" if fin else ""))
            return ok, bad

        def pack_lines(lines):
            d_max = max(len(c["acts"]) for c in lines)
            n = len(lines)
            out = {"acts_all": np.zeros((n, d_max, 6), np.int8),
                   "acts_len": np.zeros(n, np.int32),
                   "finish_ticks_all": np.zeros(n, np.int32),
                   "arc_all": np.zeros(n, np.float64),
                   "arc_tick_all": np.zeros(n, np.int32),
                   "end_tick_all": np.zeros(n, np.int32),
                   "replay_arc_all": np.zeros(n, np.float64),
                   "raw_arc_all": np.zeros(n, np.float64)}
            if VIEWC:
                out["view_all"] = np.zeros((n, d_max, 2), np.float32)
            for j, c in enumerate(lines):
                ah = c["acts"]
                out["acts_all"][j, :len(ah)] = ah
                out["acts_len"][j] = len(ah)
                if VIEWC:
                    out["view_all"][j, :len(ah)] = np.asarray(c["view"],
                                                             np.float32)
                out["finish_ticks_all"][j] = int(c["finish_tick"])
                out["arc_all"][j] = float(c.get("best_arc") or 0.0)
                out["arc_tick_all"][j] = int(c.get("arc_tick") or 0)
                out["end_tick_all"][j] = int(c.get("end_tick") or 0)
                out["replay_arc_all"][j] = float(c.get("replay_arc")
                                                 or c.get("best_arc") or 0.0)
                out["raw_arc_all"][j] = float(c.get("raw_arc")
                                              or c.get("best_arc") or 0.0)
            return out

        arc_meta = {"objective": np.str_(args.objective),
                    "view_continuous": np.int32(1 if VIEWC else 0),
                    "route_file": np.str_(str(args.route_file)),
                    "arc_corridor": np.float64(args.corridor),
                    "arc_window": np.int32(args.arc_window),
                    "arc_total": np.float64(arc_total),
                    "arc_bank": np.str_(args.arc_bank),
                    "contact_tol": np.float64(args.contact_tol),
                    "gravity_step": np.float64(gravity_step(coreN))}

        if best is None and arcp is not None:
            # ---- progress objective, no crossing: the best-arc lineage
            # is the result. Replay-verify the kept lines, write the best
            # one's trajectory, save every verified line for distillation.
            lines_ok, n_bad = verify_lines(lines)
            diag = {"frontier_vertex": frontier["vert"],
                    "frontier_d": frontier["d"],
                    "frontier_tick": frontier["tick"]}
            if dth_t:
                tt = np.concatenate(dth_t)
                oo = np.concatenate(dth_o).astype(np.float64)
                qt = np.percentile(tt, [10, 50, 90]).astype(int)
                qz = np.percentile(oo[:, 2], [10, 50, 90]).astype(int)
                diag.update(deaths=int(len(tt)),
                            death_tick_q=[int(x) for x in qt],
                            death_z_q=[int(x) for x in qz])
            if not lines_ok:
                summary = {"ckpt": str(args.ckpt), "map": map_path,
                           "envs": N, **tick_json,
                           **sinfo, "crossed": False, **diag,
                           "kept_lines": 0, "diverged_lines": int(n_bad),
                           "best_arc": None, "best_arc_tick": None,
                           "greedy_ticks": greedy_ticks,
                           "search_wall_s": round(dt_loop, 1),
                           "env_steps_per_s": round(fps)}
                (out_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8")
                print("beam TAS: PROGRESS objective kept NO reproducible "
                      f"lineage ({n_bad} diverged)")
                return
            top = lines_ok[0]
            core1b = build_sim(cfg, map_path, 1, ep_cap, tick=TICK)
            arm(core1b)
            core1b.reset(gate_seed)
            core1b.set_state(0, row0)
            acts_ticks = np.repeat(top["acts"].astype(np.int32), K, axis=0)
            rpath = out_dir / "beam_best.jsonl"
            hdr = {"map": Path(core1b.bsp_path).stem,
                   "tick_ms": int(core1b.config.phys.msec),
                   "phys": phys_header(core1b),
                   **tick_header(core1b)}
            with open(rpath, "w", encoding="utf-8", newline="\n") as f:
                end, ticks, _fin, _pre = run_episode(
                    core1b, Playback(acts_ticks,
                                     line_view_ticks(top.get("view"), K)),
                    np.zeros((1, core1b.obs_dim), np.float32), f,
                    len(acts_ticks), hdr, 0)
            cad = cadence_from_traj(rpath, TICK,
                                    core1b.config.phys.sv_gravity)
            print_cadence(cad, "best-arc line")
            pk = pack_lines(lines_ok)
            np.savez(out_dir / "beam_best.npz",
                     acts=top["acts"], act_every=np.int32(K),
                     **({"view": np.asarray(top["view"], np.float32)}
                        if VIEWC else {}),
                     finish_ticks=np.int32(0),
                     best_arc=np.float64(top["best_arc"]),
                     best_arc_tick=np.int32(top["arc_tick"]),
                     spawn_state=row0,
                     obs_start=np.asarray(obs_start, np.float32),
                     gate_seed=np.int32(gate_seed), **pk,
                     greedy_ticks=np.int32(greedy_ticks or 0),
                     greedy_prefix=np.int32(prefix),
                     seed=np.int32(args.seed),
                     torch_seed=np.int32(args.torch_seed),
                     eps=np.float32(args.eps), commit=np.int32(args.commit),
                     ckpt=np.str_(str(args.ckpt)), map=np.str_(map_path),
                     **tick_npz(TICK, cfg_tick), **arc_meta)
            summary = {"ckpt": str(args.ckpt), "map": map_path, "envs": N,
                       **tick_json,
                       **sinfo, "crossed": False, **diag,
                       "best_arc": float(top["best_arc"]),
                       "best_arc_tick": int(top["arc_tick"]),
                       "best_arc_s": secs(int(top["arc_tick"])),
                       "best_end_tick": int(top["end_tick"]),
                       "arc_pct": (100.0 * float(top["best_arc"])
                                   / max(arc_total, 1.0)),
                       "kept_lines": len(lines_ok),
                       "diverged_lines": int(n_bad),
                       "kept_arcs": [float(c["best_arc"]) for c in lines_ok],
                       "kept_arc_ticks": [int(c["arc_tick"])
                                          for c in lines_ok],
                       "kept_raw_arcs": [float(c.get("raw_arc") or 0.0)
                                         for c in lines_ok],
                       "best_raw_arc": float(top.get("raw_arc") or 0.0),
                       "replay_arc": float(top["replay_arc"]),
                       "replay_bit_exact": bool(top["replay_ok"]),
                       "replay_end": end, "replay_ticks": int(ticks),
                       "greedy_ticks": greedy_ticks,
                       "cadence": cad,
                       "search_wall_s": round(dt_loop, 1),
                       "env_steps_per_s": round(fps)}
            (out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
            print(f"beam TAS: PROGRESS best arc {top['best_arc']:,.0f}u "
                  f"({summary['arc_pct']:.1f}% of {arc_total:,.0f}u) at "
                  f"tick {top['arc_tick']} ({secs(top['arc_tick']):.2f}s), "
                  f"raw (fall included) {summary['best_raw_arc']:,.0f}u, "
                  f"{len(lines_ok)} lines kept ({n_bad} diverged), replay "
                  f"{'exact' if top['replay_ok'] else 'OFF'}, total wall "
                  f"{time.time() - t_all:.0f}s")
            return

        if best is None:
            # No crossing. That is a result, so report WHERE it stopped
            # rather than just failing: frontier reached, and whether the
            # deaths cluster at one place.
            diag = {"frontier_vertex": frontier["vert"],
                    "frontier_d": frontier["d"],
                    "frontier_tick": frontier["tick"]}
            if dth_t:
                tt = np.concatenate(dth_t)
                oo = np.concatenate(dth_o).astype(np.float64)
                P = np.asarray(load_route(Path(args.route_file))[0],
                               np.float64)
                d2 = ((oo[:, None, :] - P[None, :, :]) ** 2).sum(-1)
                vert = d2.argmin(axis=1)
                offl = np.sqrt(d2.min(axis=1))
                qt = np.percentile(tt, [10, 50, 90]).astype(int)
                qv = np.percentile(vert, [10, 50, 90]).astype(int)
                qz = np.percentile(oo[:, 2], [10, 50, 90]).astype(int)
                diag.update(deaths=int(len(tt)),
                            death_tick_q=[int(x) for x in qt],
                            death_vertex_q=[int(x) for x in qv],
                            death_z_q=[int(x) for x in qz],
                            death_offline_med=float(np.median(offl)))
                print(f"no crossing. deaths {len(tt)}: tick q10/50/90 "
                      f"{qt[0]}/{qt[1]}/{qt[2]}, route vertex "
                      f"{qv[0]}/{qv[1]}/{qv[2]}, z {qz[0]}/{qz[1]}/{qz[2]}, "
                      f"median off-line {np.median(offl):.0f}u")
            # frontier tick -1: no resample ever ranked a live lineage
            diag["frontier_s"] = (secs(frontier["tick"])
                                  if frontier["tick"] >= 0 else None)
            print(f"frontier: vertex {frontier['vert']} d {frontier['d']:,.0f}"
                  f"u at tick {frontier['tick']}"
                  + (f" ({diag['frontier_s']:.2f}s)"
                     if diag["frontier_s"] is not None else ""))
            summary = {"ckpt": str(args.ckpt), "map": map_path, "envs": N,
                       **tick_json,
                       **sinfo, "crossed": False, **diag,
                       "greedy_ticks": greedy_ticks,
                       "search_wall_s": round(dt_loop, 1),
                       "env_steps_per_s": round(fps)}
            (out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
            if allow_nonfin:
                return
            raise SystemExit(
                "search produced no finisher - nothing to replay "
                f"(greedy baseline was {greedy_ticks} ticks; try "
                "more ticks/envs or a different --torch-seed)")

    # ---- phase 3: deterministic open-loop replay of the winner ---------
    core1b = build_sim(cfg, map_path, 1, ep_cap, tick=TICK)
    arm(core1b)
    core1b.reset(args.seed)                # arbitrary; state overwritten
    core1b.set_state(0, row0)
    acts_ticks = np.repeat(best["acts"].astype(np.int32), K, axis=0)
    rpath = out_dir / "beam_best.jsonl"
    hdr = {"map": Path(core1b.bsp_path).stem,
           "tick_ms": int(core1b.config.phys.msec),
           "phys": phys_header(core1b),
           **tick_header(core1b)}
    with open(rpath, "w", encoding="utf-8", newline="\n") as f:
        end, ticks, fin, pre_state = run_episode(
            core1b, Playback(acts_ticks, line_view_ticks(best.get("view"), K)),
            np.zeros((1, core1b.obs_dim), np.float32), f, ep_cap, hdr, 0)
    if not fin:
        raise SystemExit(f"REPLAY DIVERGED: open-loop replay ended "
                         f"'{end}' at tick {ticks}, search finished at "
                         f"{best['tick']} - determinism broken")
    assert ticks == best["tick"], \
        f"replay finished at {ticks}, search said {best['tick']}"
    same_o = np.array_equal(pre_state["origin"], best["pre_origin"])
    same_v = np.array_equal(pre_state["velocity"], best["pre_vel"])
    assert same_o and same_v, (
        "replay finish tick matched but the pre-finish state is not "
        f"bit-identical: origin {pre_state['origin']} vs "
        f"{best['pre_origin']}, velocity {pre_state['velocity']} vs "
        f"{best['pre_vel']}")
    print(f"replay: bit-exact finish reproduced at tick {ticks} "
          f"({secs(ticks):.2f}s) -> {rpath}")
    # the mechanism check, on the winner's own replayed line - and, when a
    # --prefix-line supplied most of it, on the SEARCHED suffix alone
    _sv_g = core1b.config.phys.sv_gravity
    cad = cadence_from_traj(rpath, TICK, _sv_g)
    print_cadence(cad, "winner")
    cad_suffix = None
    _pre = int(locals().get("pre_ticks") or 0)
    if _pre > 0:
        cad_suffix = cadence_from_traj(rpath, TICK, _sv_g, start=_pre)
        print_cadence(cad_suffix, f"searched suffix, from tick {_pre}")

    # every kept lineage, deduplicated (clones that crossed on the same
    # tick share a history), fastest first, padded to one table so the
    # BC builder (tools/plan_to_bc.py) can replay each one. Commit mode
    # keeps only the winner; population mode ranked `lines` above (and
    # under --objective progress/auto appends the verified best-arc
    # non-finishers after the finishers).
    if v1_search:
        lines_ok, n_bad = verify_lines(lines)
        pk = pack_lines(lines_ok)
        extra = dict(arc_meta)
    else:
        pk = {"acts_all": np.asarray(best["acts"], np.int8)[None],
              "acts_len": np.array([len(best["acts"])], np.int32),
              "finish_ticks_all": np.array([best["tick"]], np.int32)}
        if VIEWC:
            pk["view_all"] = np.asarray(best["view"], np.float32)[None]
        lines_ok, n_bad = [None], 0
        extra = {"view_continuous": np.int32(1 if VIEWC else 0)}
    npz = out_dir / "beam_best.npz"
    np.savez(npz,
             acts=best["acts"], act_every=np.int32(K),
             **({"view": np.asarray(best["view"], np.float32)}
                if VIEWC else {}),
             finish_ticks=np.int32(best["tick"]),
             spawn_state=row0,
             # the spawn's 15 core scalars (the reset obs): a replay from
             # set_state has no other way to hand the policy decision 0's
             # row, since set_state does not refresh the obs buffer
             obs_start=np.asarray(obs_start, np.float32),
             gate_seed=np.int32(gate_seed), **pk,
             greedy_ticks=np.int32(greedy_ticks or 0),
             greedy_prefix=np.int32(prefix),
             seed=np.int32(args.seed), torch_seed=np.int32(args.torch_seed),
             eps=np.float32(args.eps), commit=np.int32(args.commit),
             ckpt=np.str_(str(args.ckpt)), map=np.str_(map_path),
             **tick_npz(TICK, cfg_tick), **extra)
    summary = {
        "ckpt": str(args.ckpt), "map": map_path, "envs": N, **tick_json,
        **sinfo, "crossed": True,
        "greedy_ticks": greedy_ticks,
        "greedy_s": secs(greedy_ticks) if greedy_ticks else None,
        "best_ticks": best["tick"], "best_s": secs(best["tick"]),
        # WHICH env crossed first: below --greedy-envs it is a greedy
        # continuation of an elite, and at or above --macro-hold's block
        # start it is a held-key lineage. Which of the two the search
        # actually picks is the whole question a proposal arm asks.
        "best_env": best.get("env"),
        "gain_s": (gain_s(greedy_ticks, best["tick"])
                   if greedy_ticks else None),
        "search_wall_s": round(dt_loop, 1),
        "env_steps_per_s": round(fps),
        "replay_bit_exact": bool(same_o and same_v),
        "kept_lines": len(lines_ok),
        "cadence": cad,
        "cadence_searched": cad_suffix,
    }
    if v1_search and lines_ok and lines_ok[0] is not None:
        summary["kept_finishers"] = int(sum(1 for c in lines_ok
                                            if c["finish_tick"] > 0))
        summary["diverged_lines"] = int(n_bad)
        if progress_mode:
            summary["best_arc"] = float(max(c["best_arc"] for c in lines_ok))
            summary["arc_pct"] = 100.0 * summary["best_arc"] / max(arc_total,
                                                                   1.0)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    if greedy_ticks is None:
        print(f"beam TAS: {'SPLICED ' if prefix else ''}CROSSING in "
              f"{secs(best['tick']):.2f}s "
              + (f"(greedy prefix {prefix} ticks + searched suffix)" if prefix
                 else "(no greedy baseline: the gate did not finish)")
              + f", replay bit-exact, total wall {time.time() - t_all:.0f}s")
        return
    print(f"beam TAS: greedy {secs(greedy_ticks):.2f}s -> best "
          f"{secs(best['tick']):.2f}s "
          f"({gain_s(greedy_ticks, best['tick']):+.2f}s), total wall "
          f"{time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
