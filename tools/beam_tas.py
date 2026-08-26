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
(frame ring, chunk plan, latch flag, route file) are refused outright.

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
from surfgym.core import SURF_IN_DUCK, SURF_IN_JUMP, phys_to_dict
from surfgym.rewards import map_spawn_pool
from train_fast import (NVEC, GreedyTorchPolicy, HeadPacker, Policy,
                        SampledTorchPolicy, sample_padded)
import record_ckpt as _rc   # audit_cfg: inherit refuse-on-unknown-keys

DEF_CKPT = "C:/RL_Surf/runs/frozen/F_prime.pt"
# ABSOLUTE main-checkout maps dir. The worktree's maps/ is a COPY with
# different mtimes, and every cache (goal field, SDF, occ) keys on
# size+mtime_ns of the bsp - resolving the map inside the worktree
# silently triggers a ~30-minute goal-field re-bake (CLAUDE.md).
MAIN_MAPS = Path("C:/RL_Surf/maps")

# Config knobs that add PER-ENV inference state (frame ring, chunk plan,
# latch flag) or need a side file (route). Cloning an env mid-episode
# must clone that state too; v1 does not implement it, so it refuses
# rather than run with silently wrong semantics.
UNSUPPORTED = ("route_file", "race_latch", "race_latch_frac", "chunk",
               "frame_stack")


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


def build_sim(cfg, map_path, num_envs, ep_cap):
    """Physics core exactly as record_ckpt.py builds it (eyeless; vision
    is GPU-side)."""
    fix_pitch = cfg.get("fix_pitch")
    pitch_rate = 0.0 if fix_pitch is not None else float(
        cfg.get("pitch_rate", -1.0))
    return SurfCore(map_path, default_config(
        num_envs=num_envs, spawn_mode=2, max_episode_ticks=ep_cap,
        water_fail=1,
        sv_maxvelocity=float(cfg.get("maxvel", 2000.0)),
        yaw_adaptive=1 if cfg.get("yaw_adaptive") else 0,
        lidar_w=0, lidar_h=0,
        pitch_rate_max_deg=pitch_rate))


def run_episode(core, act_fn, obs, fout, max_ticks, header, episode_idx):
    """Roll env 0 ONE episode from the core's current state, writing traj
    rows in surfgym.record's exact format. Returns
    (end, ticks, finished, pre_finish_state)."""
    fout.write(json.dumps({**header, "episode": episode_idx},
                          separators=(",", ":")) + "\n")
    ep_ticks, best_progress = 0, 0.0
    finished, end, pre_state = False, "trunc", None
    for _ in range(max_ticks):
        s0 = core.get_states()[0]           # pre-step snapshot (copy)
        actions = act_fn(obs)
        obs, rew, done, trunc, _term = core.step(actions)
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
    the observation entirely. No policy, no GPU."""

    def __init__(self, seq_ticks):
        self.seq, self.t = seq_ticks, 0

    def act(self, _obs):
        a = self.seq[min(self.t, len(self.seq) - 1)]
        self.t += 1
        return np.ascontiguousarray(a.reshape(1, 6), dtype=np.int32)


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

    @staticmethod
    def _hmix(h, acts):
        """Rolling uint64 prefix hash: 6 bins < 15 pack into 24 bits."""
        packed = np.zeros(len(acts), np.uint64)
        for k in range(6):
            packed |= acts[:, k].astype(np.uint64) << np.uint64(4 * k)
        with np.errstate(over="ignore"):
            return h * np.uint64(1099511628211) ^ packed

    @torch.inference_mode()
    def _decide(self, obs):
        logits, _ = self.policy(self._obs(obs))
        padded = self.packer.pad(logits)
        act = self._sample(padded)
        a = act.to("cpu").numpy().astype(np.int32)
        if self._dd_n < DEDUP_DECISIONS:
            n = len(a)
            if self._dd_h is None:
                self._dd_h = np.zeros(n, np.uint64)
                self._dd_hs = np.zeros(n, np.uint64)
            self._dd_hs = self._hmix(self._dd_hs, a)   # shadow: no reroll
            cur = self._hmix(self._dd_h, a)
            for _ in range(DEDUP_ATTEMPTS):
                _, first = np.unique(cur, return_index=True)
                dup = np.ones(n, bool)
                dup[first] = False
                if not dup.any():
                    break
                a2 = self._sample(padded).to("cpu").numpy().astype(np.int32)
                a[dup] = a2[dup]
                cur = self._hmix(self._dd_h, a)
            self._dd_h = cur
            self._dd_n += 1
            if self._dd_n == DEDUP_DECISIONS:
                self.dd_stats.append(
                    (int(n - len(np.unique(self._dd_hs))),
                     int(n - len(np.unique(self._dd_h)))))
        return a


def make_scorer(gf, route_file, corridor, mode):
    """-> score(states) = (higher is better, geodesic d).

    mode 'd' is the plain -geodesic used by the spawn-to-finish search.
    mode 'route' ranks by how far along the ROUTE a candidate is, which is
    what a search AT THE WALL needs: this map's geodesic field has its
    along-route minimum AT the wall (vertex 1601, d=6,568) and goal-
    adjacent airspace below the ramp scores lower still, so -d ranking
    near the wall actively selects the dive. Route vertex index is
    monotone by construction; d only breaks ties within a vertex.
    """
    if mode == "d":
        def score(states):
            d = gf.sample(states["origin"]).astype(np.float64)
            return -d, d
        return score

    pts, _spacing = load_route(Path(route_file))
    P = np.asarray(pts, np.float64)

    def score(states):
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
                  value_fn=None):
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
    committed = []                # list of (C, 6) int8 blocks
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
            if t % K == 0:
                hist[d] = a
            sv = coreN.states_view
            pre_o = sv["origin"].copy()
            pre_v = sv["velocity"].copy()
            obs, _rew, done, trunc, _term = coreN.step(a)
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
                        else feed_state["d"].copy())
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
                  f"-> total {total} ({total / 100:.2f}s)")
            return ({"tick": total, "acts": np.concatenate(blocks, axis=0),
                     "pre_origin": po, "pre_vel": pv},
                    _info(), None)
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
        committed_ticks += C * K
        st_row = snap[0][lead:lead + 1].copy()
        # the committed state's episode clock must equal the assembled
        # tick count, or the finish-time accounting is broken
        assert int(st_row["tick"][0]) == committed_ticks, \
            (int(st_row["tick"][0]), committed_ticks)
        for j in range(N):
            coreN.set_state(j, st_row)
        obs = np.array(obs)
        obs[:] = snap[1][lead][None, :]
        if feed_state is not None:
            feed_state["d"] = (None if snap[2] is None
                               else np.full(N, snap[2][lead]))
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
    ap.add_argument("--skip-gate", action="store_true",
                    help="capture the spawn state but do NOT roll the gate "
                    "episode. row0/obs_start come from the reset itself, so "
                    "for a --greedy-prefix search the gate rollout is pure "
                    "overhead (a full single-env episode per wave)")
    ap.add_argument("--allow-nonfinisher", action="store_true",
                    help="do not require the greedy gate to finish (implied "
                    "by --greedy-prefix: searching from the wall is only "
                    "interesting for a policy that does NOT finish)")
    ap.add_argument("--score", choices=["d", "route"], default="d",
                    help="boundary ranking: 'd' = geodesic (spawn-to-finish "
                    "search), 'route' = route vertex index then d (wall "
                    "search - see make_scorer)")
    ap.add_argument("--route-file",
                    default="C:/RL_Surf/maps/surf_src_cannonball.route.npz")
    ap.add_argument("--corridor", type=float, default=1500.0)
    ap.add_argument("--max-ticks", type=int, default=12000)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.torch_seed)
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    act_every = int(cfg.get("act_every", 1))
    K = act_every
    cfg_ep_ticks = int(cfg.get("ep_ticks", 700))
    ep_cap = max(cfg_ep_ticks, int(args.max_ticks))
    map_path = resolve_map(args.map, cfg.get("map", "surf_ski_2"))
    print(f"ckpt step {step:,}  act_every {K}  map {map_path}")

    # ---- 1-env core: greedy sanity gate + spawn-state capture ----------
    core1 = build_sim(cfg, map_path, 1, ep_cap)
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
    policy = Policy(core1.obs_dim + lw * lh * lidar.channels * stack, lw, lh,
                    emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)),
                    extra_feat=extra,
                    in_ch=lidar.channels * stack,
                    n_codes=0, chunk=0, route_dim=0,
                    route_critic_only=bool(cfg.get("route_critic_only"))
                    ).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    packer = HeadPacker(device)

    def mk_feed():
        """--obs-reward slot-12 feed, per record_ckpt.py (a missing feed
        hands the policy absolute position where it expects tanh(reward)
        and kills the agent in seconds). d0 is the trainer's own formula
        (train_fast.py: mean field over the RAW map spawns) - record_ckpt
        samples pre-reset states there, which is a latent bug this tool
        does not copy. Returns (slot, fn, state); state['d'] is per-env
        prev-d and must be cloned donor->loser at a resample."""
        if not cfg.get("obs_reward"):
            return -1, None, None
        scale = 100.0 / max(d0, 1.0) * float(cfg.get("race_shaping") or 1.0)
        tp = float(cfg.get("time_pen") or 0.005)
        d_floor = float(cfg.get("race_dfloor") or 0.0)
        st = {"d": None}

        def _feed(c, _f=gf, _s=scale, _tp=tp, _k=K, _fl=d_floor):
            dd = _f.sample(c.states_view["origin"]).astype(np.float64)
            if _fl > 0.0:
                dd = np.maximum(dd, _fl)
            prev, st["d"] = st["d"], dd
            if prev is None or len(prev) != len(dd):
                return np.zeros(len(dd), np.float32)
            delta = np.clip(prev - dd, -100.0 * _k, 100.0 * _k)
            return np.tanh((delta * _s - _tp * _k) / 0.1).astype(np.float32)

        return 12, _feed, st

    header1 = {"map": Path(core1.bsp_path).stem,
               "tick_ms": int(core1.config.phys.msec),
               "phys": phys_to_dict(core1.config.phys)}

    # ---- phase 1: greedy sanity gate -----------------------------------
    # Fresh wrapper + fresh reset per episode so every episode's act_every
    # hold is aligned to its own tick 0, exactly like the search and the
    # replay (record_rollout's single global cadence would misalign
    # episodes 2+). The first FINISHING episode supplies the matched
    # spawn state row0 and the baseline time.
    gpath = out_dir / "greedy_baseline.jsonl"
    greedy_ticks, row0, obs_start = None, None, None
    nonfin = None
    with open(gpath, "w", encoding="utf-8", newline="\n") as f:
        for e in range(max(1, args.greedy_eps)):
            obs = core1.reset(args.seed + e)
            row = core1.get_states()[0:1].copy()      # STATE_DTYPE copy
            o0 = obs[0].copy()
            if args.skip_gate:
                nonfin = (0, row, o0)
                print(f"gate skipped: spawn state captured (seed "
                      f"{args.seed + e})")
                break
            es1, ef1, _ = mk_feed()
            gpol = GreedyTorchPolicy(policy, packer, device, lidar, core1,
                                     K, stack, extra_slot=es1, extra_fn=ef1)
            end, ticks, fin, _ = run_episode(core1, gpol.act, obs, f,
                                             ep_cap, header1, e)
            print(f"greedy ep{e} (spawn seed {args.seed + e}): {end} in "
                  f"{ticks} ticks ({ticks / 100:.2f}s)")
            if nonfin is None:
                # the FIRST episode's spawn is the one a wall search rides:
                # greedy is deterministic, so replaying it reproduces this
                # episode exactly
                nonfin = (ticks, row, o0)
            if fin:
                greedy_ticks, row0, obs_start = ticks, row, o0
                break
    allow_nonfin = args.allow_nonfinisher or args.greedy_prefix > 0
    if greedy_ticks is None:
        if not allow_nonfin:
            raise SystemExit(
                f"GATE FAILED: {Path(args.ckpt).name} did not finish in "
                f"{args.greedy_eps} greedy episode(s) - wrong checkpoint "
                "for a beam search; stopping per plan. Trajectories: "
                + str(gpath))
        _t, row0, obs_start = nonfin
        print(f"gate: no finish in {args.greedy_eps} greedy episode(s) "
              f"(first ran {_t} ticks) - continuing, this search does not "
              "need one")
    else:
        print(f"greedy baseline: {greedy_ticks} ticks = "
              f"{greedy_ticks / 100:.2f}s -> {gpath}")
    if args.greedy_only:
        return

    # ---- phase 2: the search -------------------------------------------
    N = int(args.envs)
    R = int(args.resample_every)
    gen_ticks = R * K
    max_ticks = int(args.max_ticks)
    n_elite = max(1, int(round(N * args.elite_frac)))
    coreN = build_sim(cfg, map_path, N, ep_cap)
    arm(coreN)
    obs = np.array(coreN.reset(args.seed))        # copy, then overwrite
    for i in range(N):
        coreN.set_state(i, row0)
    obs[:] = obs_start[None, :]
    esN, efN, feed_state = mk_feed()
    if (args.boundary_v or args.dedup) and args.commit <= 0:
        raise SystemExit("--boundary-v/--dedup are commit-mode features "
                         "(pass --commit H)")
    if args.eps > 0.0 or args.dedup:
        spol = EpsSampledTorchPolicy(policy, packer, device, lidar, coreN,
                                     K, stack, eps=args.eps,
                                     dedup=args.dedup,
                                     extra_slot=esN, extra_fn=efN)
    else:   # never route eps=0 through the mixer: RNG-stream parity
        spol = SampledTorchPolicy(policy, packer, device, lidar, coreN,
                                  K, stack, extra_slot=esN, extra_fn=efN)
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
                                  K, stack, extra_slot=esN, extra_fn=efN)
    scorer = make_scorer(gf, args.route_file, args.corridor, args.score)
    value_fn = None
    if args.boundary_v:
        def value_fn(o):
            # one extra batched critic forward on the boundary obs. The
            # obs assembly is the wrapper's own (_obs), so it matches
            # what a decision would see; the obs-reward feed's per-env
            # state is snapshotted and restored around it because _obs
            # advances it as a side effect.
            saved = None if feed_state is None else feed_state.get("d")
            with torch.inference_mode():
                _, v = policy(spol._obs(o))
            if feed_state is not None:
                feed_state["d"] = saved
            return v.detach().float().reshape(-1).cpu().numpy()

    if args.commit > 0:
        # -- receding-horizon (MPC) mode --
        H = int(args.commit)
        C = max(1, min(H, int(round(args.commit_frac * H))))
        print(f"receding-horizon search: {N} envs, window H={H} decisions "
              f"({H * K} ticks), commit {C} ({C * K} ticks), "
              f"eps {args.eps:g}, cap {max_ticks} ticks")
        t_loop = time.time()
        best, cinfo, dnf = commit_search(coreN, spol, gf, obs, N, K, H, C,
                                         max_ticks, feed_state,
                                         value_fn=value_fn)
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
                       **sinfo, "dnf": True, "dnf_reason": dnf,
                       "greedy_ticks": greedy_ticks,
                       "greedy_s": greedy_ticks / 100.0,
                       "search_wall_s": round(dt_loop, 1),
                       "env_steps_per_s": round(fps)}
            (out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
            print(f"beam TAS: DNF ({dnf}); greedy was "
                  f"{greedy_ticks / 100:.2f}s")
            return
        del dnf
        v1_search = False
    else:
        v1_search = True
    if v1_search:
        D_total = (max_ticks + K - 1) // K
        hist = np.zeros((D_total, N, 6), np.int8)  # per-decision actions
        valid = np.ones(N, bool)   # history describes this env from tick 0
        idx = np.arange(N)
        finishes = []              # (tick, env, gen)
        best = None
        gen, best_gen = 0, None
        dth_t, dth_o = [], []
        frontier = {"vert": -1, "d": float("inf"), "tick": -1}
        print(f"search: {N} envs, resample every {R} decisions "
              f"({gen_ticks} ticks), elite {n_elite}, cap {max_ticks} ticks"
              + (f", greedy prefix {prefix} ticks ({prefix / 100:.2f}s)"
                 if prefix else "") + f", score {args.score}")
        t_loop = time.time()
        for t in range(max_ticks):
            d = t // K
            # greedy (deterministic, all envs identical) up to the prefix,
            # then the policy's own sampling supplies the diversity
            a = (gpolN.act(obs) if t < prefix else spol.act(obs))
            if t % K == 0:
                hist[d] = a                # bins < 15: int8 is lossless
            sv = coreN.states_view
            pre_o = sv["origin"].copy()
            pre_v = sv["velocity"].copy()
            obs, _rew, done, trunc, _term = coreN.step(a)
            gh = coreN.goal_hits
            if gh.any():
                for i in np.nonzero(gh)[0]:
                    if not valid[i]:
                        continue           # respawned body: not a real run
                    finishes.append((t + 1, int(i), gen))
                    if best is None:       # lockstep: first hit is fastest
                        best = {"tick": t + 1,
                                "acts": hist[:d + 1, i].copy(),
                                "pre_origin": pre_o[i].copy(),
                                "pre_vel": pre_v[i].copy()}
                        best_gen = gen
                        print(f"FINISH: env {i} at tick {t + 1} "
                              f"({(t + 1) / 100:.2f}s), gen {gen}")
            dead = np.asarray(done, bool) | np.asarray(trunc, bool)
            if dead.any():
                # forensics for a search that never crosses: WHERE the real
                # lineages ended, at their last live position
                newly = np.nonzero(dead & valid)[0]
                if len(newly):
                    dth_t.append(np.full(len(newly), t + 1, np.int64))
                    dth_o.append(pre_o[newly].copy())
                valid &= ~dead
            if best is not None and args.gens > 0 \
                    and gen - best_gen >= args.gens:
                break
            if ((t + 1) % gen_ticks == 0 and (t + 1) < max_ticks
                    and (t + 1) > prefix):
                # no cloning during the greedy prefix: every env is the
                # same state, so there is nothing to select between
                gen += 1
                states = coreN.get_states()
                sc, dgeo = scorer(states)
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
                losers = idx[~keep_set]
                donors = keep[np.arange(len(losers)) % len(keep)]
                for j, don in zip(losers, donors):
                    coreN.set_state(int(j), states[don])
                hist[:d + 1, losers] = hist[:d + 1, donors]
                obs = np.array(obs)        # patch clones' scalar obs too
                obs[losers] = obs[donors]
                valid[:] = True
                if feed_state is not None \
                        and feed_state.get("d") is not None:
                    feed_state["d"][losers] = feed_state["d"][donors]
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
                          + f"min_d={dgeo[keep[0]]:8.0f} "
                          f"med_d={np.median(dgeo[keep]):8.0f} "
                          f"finishes={len(finishes)}")
        dt_loop = time.time() - t_loop
        fps = (t + 1) * N / max(dt_loop, 1e-9)
        print(f"search done: {t + 1} ticks x {N} envs in {dt_loop:.0f}s "
              f"({fps:,.0f} env-steps/s), {len(finishes)} finishes, "
              f"{gen} generations")
        sinfo = {"mode": "population", "eps": args.eps,
                 "resample_every_decisions": R,
                 "elite_frac": args.elite_frac,
                 "greedy_prefix": prefix, "score": args.score,
                 "finishes": len(finishes),
                 "finish_ticks": sorted(ft for ft, _, _ in finishes),
                 "generations": gen}

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
            print(f"frontier: vertex {frontier['vert']} d {frontier['d']:,.0f}"
                  f"u at tick {frontier['tick']}")
            summary = {"ckpt": str(args.ckpt), "map": map_path, "envs": N,
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
    core1b = build_sim(cfg, map_path, 1, ep_cap)
    arm(core1b)
    core1b.reset(args.seed)                # arbitrary; state overwritten
    core1b.set_state(0, row0)
    acts_ticks = np.repeat(best["acts"].astype(np.int32), K, axis=0)
    rpath = out_dir / "beam_best.jsonl"
    hdr = {"map": Path(core1b.bsp_path).stem,
           "tick_ms": int(core1b.config.phys.msec),
           "phys": phys_to_dict(core1b.config.phys)}
    with open(rpath, "w", encoding="utf-8", newline="\n") as f:
        end, ticks, fin, pre_state = run_episode(
            core1b, Playback(acts_ticks).act,
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
          f"({ticks / 100:.2f}s) -> {rpath}")

    npz = out_dir / "beam_best.npz"
    np.savez(npz,
             acts=best["acts"], act_every=np.int32(K),
             finish_ticks=np.int32(best["tick"]),
             spawn_state=row0,
             greedy_ticks=np.int32(greedy_ticks or 0),
             greedy_prefix=np.int32(prefix),
             seed=np.int32(args.seed), torch_seed=np.int32(args.torch_seed),
             eps=np.float32(args.eps), commit=np.int32(args.commit),
             ckpt=np.str_(str(args.ckpt)), map=np.str_(map_path))
    summary = {
        "ckpt": str(args.ckpt), "map": map_path, "envs": N,
        **sinfo, "crossed": True,
        "greedy_ticks": greedy_ticks,
        "greedy_s": (greedy_ticks / 100.0) if greedy_ticks else None,
        "best_ticks": best["tick"], "best_s": best["tick"] / 100.0,
        "gain_s": ((greedy_ticks - best["tick"]) / 100.0
                   if greedy_ticks else None),
        "search_wall_s": round(dt_loop, 1),
        "env_steps_per_s": round(fps),
        "replay_bit_exact": bool(same_o and same_v),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    if greedy_ticks is None:
        print(f"beam TAS: SPLICED CROSSING in {best['tick'] / 100:.2f}s "
              f"(greedy prefix {prefix} ticks + searched suffix), "
              f"replay bit-exact")
        return
    print(f"beam TAS: greedy {greedy_ticks / 100:.2f}s -> best "
          f"{best['tick'] / 100:.2f}s "
          f"({(greedy_ticks - best['tick']) / 100:+.2f}s), total wall "
          f"{time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
