#!/usr/bin/env python3
"""credit_diag.py - does a NEWLY DISCOVERED success actually train the actions
that caused it, or does the credit die on the way back?

The discount horizon is settled (gamma 0.9995 per PHYSICS TICK, raised to
act_every by the trainer, = 20.0 s; CLAUDE.md section 5 - do not shorten it).
That does NOT settle how efficiently a success reaches the decisions taken
5-10 s earlier, because PPO does not propagate the reward with gamma alone.
It propagates it with GAE, and the direct weight of a TD residual k decisions
later is ``(gamma^act_every * lambda)^k``. On the scratch config
(act_every 4 at 10 ms, lambda 0.95, n_steps 128 decisions = 5.12 s) that is
0.0013 at 5 s against the discount's own 0.78 - a factor of 600. Whatever the
remaining 0.78 is worth, it is worth it only THROUGH THE CRITIC: the term
``gamma^k * V(s_{t+k})`` is how a distant reward reaches decision t, and it is
only as good as V is at states the policy has barely visited.

So this is the bounded diagnostic. From pre-wall start states, roll COMPLETE
terminal continuations with the real simulator and the run's own reward, and
ask three questions with numbers:

  1. **Is the critic right there?**  V(s_t) against the empirical discounted
     return G_t, at start states 0 / 1 / 2 / 5 / 10 s before the terminal
     event of the checkpoint's own greedy episode.
  2. **Do successful continuations exist at all under sampling?**  The share
     of continuations that pass the wall (corridor arc > 205,440 u on
     surf_src_cannonball) or finish.
  3. **If they exist, does their FIRST ACTION get credit?**  The advantage
     that first decision receives under

       (a) the trainer's own GAE - the run's lambda and n_steps, cut at
           rollout-buffer boundaries exactly as train_fast.py cuts them
           (``lastgae`` restarts at 0 at every buffer edge and the edge
           bootstraps with V), averaged over the buffer phase the episode
           could have started at;
       (b) lambda = 1 over the WHOLE episode - the Monte-Carlo advantage
           ``G_t - V_t``, which is what "longer traces" would buy;
       (c) the run's lambda over the whole episode, no truncation - which
           separates "lambda is too short" from "the rollout is too short".

If successful continuations exist and (b) credits their first action while
(a) does not, the fix is longer traces (lambda -> 1 with rollouts covering the
manoeuvre), not a longer horizon. If (a) ~ (b) the credit path is fine and the
wall is elsewhere.

    python tools/credit_diag.py C:/RL_Surf_cyn/runs/cyPOTLC/ckpt_latest.pt \\
        --map C:/RL_Surf/maps/surf_src_cannonball.bsp \\
        --route C:/RL_Surf/maps/surf_src_cannonball.route.npz \\
        --k 48 --out runs/research/creditdiag/cyPOTLC

Start states come from the checkpoint's OWN greedy episode (a wall-stopping
one for a stuck policy), so nothing outside the checkpoint is needed; pass
``--from-spine spine.npy --at-tick T`` to probe from a recorded episode
instead. Everything is measured on the run's own constants, read out of the
checkpoint config: gamma, lambda, n_steps, act_every, the reward flags.

Worktree trap (CLAUDE.md): pass the MAIN checkout's map and route. A "bake"
line means a cache miss and a 30-minute rebuild on your clock. Needs the GPU
for the lidar; a few hundred envs at 64x32 is a couple of minutes.

The pure-arithmetic half (returns, TD residuals, the three GAE variants, the
discount table) is in the ``# --- credit arithmetic ---`` block below and is
covered by tests/python/test_credit_diag.py with no GPU, no map and no
checkpoint - including a line-by-line cross-check of variant (a) against a
literal transcription of train_fast.py's GAE loop.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

WALL_U = 205440.0           # surf_src_cannonball: the 88.8 % wall, route units
TICK_MS = 10.0
HORIZONS = (0.0, 1.0, 2.0, 5.0, 10.0)   # seconds before the terminal event


# ===========================================================================
# --- credit arithmetic ---  (pure numpy; no torch, no core, no GPU)
# ===========================================================================
def nonterm_mask(length: int, ended_last: bool = True) -> np.ndarray:
    """The trainer's ``1 - b_done[t]`` for ONE episode of ``length``
    decisions: 1 everywhere, 0 on the decision the episode ended on."""
    nt = np.ones(int(length), np.float64)
    if ended_last and length:
        nt[-1] = 0.0
    return nt


def discounted_returns(r: np.ndarray, g: float,
                       nonterm: np.ndarray) -> np.ndarray:
    """G_t = r_t + g * nonterm_t * G_{t+1}, the empirical discounted return
    of the rest of THIS episode (no bootstrap - the episode terminated)."""
    r = np.asarray(r, np.float64)
    G = np.zeros_like(r)
    run = 0.0
    for t in range(len(r) - 1, -1, -1):
        run = r[t] + g * nonterm[t] * run
        G[t] = run
    return G


def td_residuals(r: np.ndarray, V: np.ndarray, g: float,
                 nonterm: np.ndarray) -> np.ndarray:
    """delta_t = r_t + g * V_{t+1} * nonterm_t - V_t (train_fast.py's
    ``delta`` verbatim; V_{t+1} past the end is never used because the
    terminal decision's nonterm is 0)."""
    r = np.asarray(r, np.float64)
    V = np.asarray(V, np.float64)
    nxt = np.concatenate([V[1:], [0.0]])
    return r + g * nxt * nonterm - V


def gae(delta: np.ndarray, g: float, lam: float, nonterm: np.ndarray,
        n_steps: int | None = None, phase: int = 0) -> np.ndarray:
    """The trainer's backward pass, restricted to one episode.

    ``lastgae = delta + g * lam * nonterm * lastgae``, walked backwards.
    ``n_steps`` (with ``phase`` = the global decision index the episode
    STARTED at, mod n_steps) reproduces the rollout-buffer cut: the buffer's
    backward pass begins with ``lastgae = 0``, so the last decision of every
    buffer sees its own residual and nothing after it. ``n_steps=None`` is
    the no-truncation variant (the whole episode in one pass)."""
    delta = np.asarray(delta, np.float64)
    L = len(delta)
    adv = np.zeros(L, np.float64)
    last = 0.0
    for t in range(L - 1, -1, -1):
        if n_steps and ((int(phase) + t) % int(n_steps)) == int(n_steps) - 1:
            last = 0.0                     # t is the LAST row of its buffer
        adv[t] = delta[t] + g * lam * nonterm[t] * last
        last = adv[t]
    return adv


def gae_phase_mean(delta, g, lam, nonterm, n_steps):
    """Variant (a) over every buffer phase the episode could have started
    at -> (mean, min, max) advantage per decision, each (L,).

    An episode's phase in the trainer is whatever the rollout iteration
    happened to be at, i.e. uniform over 0..n_steps-1; reporting the mean
    plus the envelope is the honest way to say "the credit this decision
    gets depends on where the buffer edge fell"."""
    delta = np.asarray(delta, np.float64)
    L = len(delta)
    T = int(n_steps)
    # the recursion is elementwise in the phase, so ONE backward pass over
    # the L decisions serves all T phases (the scalar form above is L*T)
    ph = np.arange(T)
    A = np.empty((T, L), np.float64)
    last = np.zeros(T, np.float64)
    for t in range(L - 1, -1, -1):
        last = np.where(((ph + t) % T) == T - 1, 0.0, last)
        last = delta[t] + g * lam * nonterm[t] * last
        A[:, t] = last
    return A.mean(0), A.min(0), A.max(0)


def discount_table(gamma_tick: float, act_every: int, lam: float,
                   ks=(25, 50, 125, 250), tick_ms: float = TICK_MS):
    """The reviewer's contrast, from the run's own constants: the DIRECT GAE
    weight of a TD residual k decisions later against the discount alone.

    -> list of dicts: k, seconds, ticks, (g*lam)^k, g^k, ratio."""
    g = float(gamma_tick) ** int(act_every)
    out = []
    for k in ks:
        secs = k * act_every * tick_ms / 1000.0
        w_gae = (g * float(lam)) ** k
        w_dis = g ** k
        out.append({"k_decisions": int(k), "seconds": secs,
                    "ticks": int(k * act_every),
                    "gae_weight": w_gae, "discount_weight": w_dis,
                    "ratio": (w_dis / w_gae) if w_gae > 0 else float("inf")})
    return out


def pct(a, q):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


# ===========================================================================
# checkpoint -> core / lidar / policy   (the diversity_bench subset, plus
# --obs-reward, which that tool refuses and the stuck checkpoint needs)
# ===========================================================================
UNSUPPORTED = ("route_file", "act_hist", "obs_compass", "priv_critic", "chunk",
               "frame_stack", "mask_forward_air", "jump_cooldown",
               "duck_air_mask", "yaw_cond", "fix_pitch", "pitch_fixed",
               "goals", "race_latch", "race_latch_frac", "race_arc",
               "curiosity_cond", "maps", "heldout_maps", "ret_norm")


def check_supported(cfg: dict) -> None:
    bad = [k for k in UNSUPPORTED if cfg.get(k)]
    if cfg.get("rnn") not in (None, "none"):
        bad.append("rnn")
    if cfg.get("tick_ms") not in (None, 10, 10.0):
        bad.append("tick_ms")
    if cfg.get("reward") != "race":
        bad.append("reward")
    if bad:
        raise SystemExit(
            "credit_diag does not mirror "
            + ", ".join(f"{k}={cfg.get(k)!r}" for k in bad)
            + " - the diagnostic must run the TRAINER's reward and the "
              "TRAINER's observation or its V and its G are not the run's; "
              "extend the builders here rather than approximating")


def build_policy(ck: dict, core, lidar, device):
    """diversity_bench.build_policy plus ``extra_feat``: under --obs-reward
    the trainer re-enables scalar slot 12 (record_ckpt.py:725), and a policy
    built without it would not even load these weights."""
    from train_fast import Policy
    cfg = ck.get("config") or {}
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    policy = Policy(core.obs_dim + lw * lh * lidar.channels, lw, lh,
                    emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)),
                    extra_feat=((12,) if cfg.get("obs_reward") else ()),
                    trunk=str(cfg.get("trunk") or "plain"),
                    tower_depth=int(cfg.get("tower_depth") or 2),
                    conv_mult=int(cfg.get("conv_mult") or 1),
                    obs_fourier=int(cfg.get("obs_fourier") or 0),
                    in_ch=lidar.channels,
                    view_continuous=bool(cfg.get("view_continuous")),
                    view_absolute=(cfg.get("view_absolute") or None)
                    ).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy


def build_reward(core, cfg: dict, field, ck: dict, int_coef: float | None):
    """The run's own RaceReward, constant for constant (train_fast.py:7328).

    ``scale = 100 / rf_d0 * race_shaping`` with rf_d0 the map's start
    geodesic computed the trainer's way (mean field over the raw map spawns).
    The novelty table is restored from the checkpoint when it carries one -
    an empty table would pay this policy a premium on the beaten path that
    training long ago wore away."""
    from surfgym.rewards import RaceReward, map_spawn_pool
    from surfgym.tick import TickClock
    TICK = TickClock(float(cfg.get("tick_ms") or TICK_MS))
    d0 = float(np.mean(field.sample(map_spawn_pool(core)["origin"])))
    ic = float(cfg.get("int_coef") or 0.0) if int_coef is None \
        else float(int_coef)
    rf = RaceReward(
        field,
        scale=100.0 / max(d0, 1.0) * float(cfg.get("race_shaping") or 1.0),
        time_pen=TICK.per_tick(float(cfg.get("time_pen") or 0.005)),
        success_bonus=float(cfg.get("success_bonus") or 50.0),
        stall_ticks=TICK.secs_to_ticks(float(cfg.get("stall_secs") or 15.0)),
        stall_eps=TICK.per_tick(float(cfg.get("stall_eps") or 32.0)),
        max_step=float(cfg.get("max_step") or 100.0),
        int_coef=ic,
        int_view=int(cfg.get("int_view") or 0),
        int_speed=int(cfg.get("int_speed") or 0),
        speed_equiv=float(cfg.get("speed_equiv") or 0.0),
        fail_pen=float(cfg.get("fail_pen") or 0.0),
        finish_k=float(cfg.get("finish_k") or 0.0),
        finish_tref=float(cfg.get("finish_tref") or 120.0),
        every=1,
        d_floor=float(cfg.get("race_dfloor") or 0.0),
        ng=int(cfg.get("race_ng") or 0),
        ng_gamma=TICK.gamma(float(cfg.get("gamma") or 0.9995)),
        ng_d0=d0,
        death_charge=float(cfg.get("death_charge") or 0.0),
        tick_ms=TICK.ms)
    rf.speed_coef = TICK.per_tick(float(cfg.get("speed_coef") or 0.0))
    if ic > 0.0:
        icnt = ck.get("int_counts")
        if isinstance(icnt, dict):
            icnt = next(iter(icnt.values()), None)
        if icnt is not None:
            rf.restore_counts(np.asarray(icnt))
            print(f"novelty table restored "
                  f"({int(np.asarray(icnt).sum(dtype=np.int64)):,} visits); "
                  f"int_coef {ic:g}")
        else:
            print(f"int_coef {ic:g} but the checkpoint carries no novelty "
                  f"table - the diagnostic would pay an inflated bonus; "
                  f"pass --int-coef 0 if that matters")
    return rf, d0


# ===========================================================================
# policies that also publish V(s_t)
# ===========================================================================
def make_value_policies():
    """Subclasses that tap the value head on the way past. The trainer's own
    wrappers call ``self._net(x)`` inside ``_decide``; overriding it is the
    one place V(s_t) can be read for exactly the row the policy acted on."""
    from train_fast import GreedyTorchPolicy, SampledTorchPolicy

    class _ValueTap:
        last_value = None

        def _net(self, x):
            out = super()._net(x)
            self.last_value = out[1].detach().float().to("cpu").numpy().copy()
            return out

    class GreedyV(_ValueTap, GreedyTorchPolicy):
        pass

    class SampledV(_ValueTap, SampledTorchPolicy):
        pass

    from train_fast import TemperedTorchPolicy

    class TemperedV(_ValueTap, TemperedTorchPolicy):
        pass

    return GreedyV, SampledV, TemperedV


# ===========================================================================
# rollouts at DECISION granularity
# ===========================================================================
def roll(core, pol, reward_fn, seed: int, max_ticks: int, sample_seed: int,
         obsr_hold=None, stall_kill: bool = True, obsr_prime: float = 0.0):
    """One batch of terminal continuations from the core's one-entry spawn
    pool, recorded the way the trainer's buffer records them.

    Per DECISION d (act_every physics ticks): V[d] = the critic on the row
    the policy acted on, R[d] = the reward summed over the decision's ticks
    (the trainer's ``r_acc``), DONE[d] = the episode ended on that decision.
    Rows are dropped at their first end - unlike the trainer, which lets the
    couple of post-autoreset sub-ticks inherit the held action; that
    contamination has no place in a return.

    -> dict of arrays; pos is (ticks, N, 3) with NaN after an end."""
    import torch
    torch.manual_seed(int(sample_seed))
    if obsr_hold is not None:
        # --obs-reward: slot 12 is what the PROBE had in it at this tick, not
        # zero. It is read by the very first decision - the one every number
        # in the report is about - so zeroing it would score that action on
        # an observation the policy was never in.
        obsr_hold[:] = float(obsr_prime)
    obs = core.reset(int(seed))
    # the trainer arms the reward AT the reset (train_fast.py:8871
    # fleet.on_reset -> mapfleet.py:503), so the very first tick's shaping is
    # measured against the spawn rather than being silently zero
    reward_fn.on_reset(core)
    prev_obs = obs.copy()
    n = core.num_envs
    k = int(pol._k)
    ndec = max_ticks // k
    V = np.full((ndec, n), np.nan, np.float32)
    R = np.zeros((ndec, n), np.float32)
    DONE = np.zeros((ndec, n), bool)       # episode ENDED on this decision
    pos = np.full((ndec * k, n, 3), np.nan, np.float32)
    alive = np.ones(n, bool)
    end_dec = np.full(n, -1, np.int64)
    fin = np.zeros(n, bool)
    trunc_end = np.zeros(n, bool)
    sv = core.states_view
    d = 0
    for d in range(ndec):
        if stall_kill:
            sm = reward_fn.pop_stall_mask()
            if sm is not None:
                core.force_fail(sm)
        r_acc = np.zeros(n, np.float32)
        ended_acc = np.zeros(n, bool)
        for j in range(k):
            pos[d * k + j, alive] = np.asarray(sv["origin"])[alive]
            act = pol.act(obs)
            if j == 0:
                V[d] = np.asarray(pol.last_value, np.float32)
            view = getattr(pol, "view", None)
            if view is None:
                obs, base_r, done, trunc, term = core.step(act)
            else:
                obs, base_r, done, trunc, term = core.step(act, view=view)
            r = np.asarray(reward_fn(prev_obs, obs, term, base_r, done, trunc,
                                     core), np.float32)
            prev_obs = obs.copy()
            live = alive & ~ended_acc
            r_acc[live] += r[live]
            dn = np.asarray(done).astype(bool)
            tr = np.asarray(trunc).astype(bool)
            goal = np.asarray(core.goal_hits).astype(bool)
            fin |= goal & live
            trunc_end |= tr & ~dn & live
            ended_acc |= (dn | tr) & live
        R[d] = r_acc
        newly = ended_acc & alive
        DONE[d] = newly
        end_dec[newly] = d
        alive &= ~newly
        if obsr_hold is not None:
            # --obs-reward: the reward just earned becomes part of the NEXT
            # decision's observation, tanh(r/0.1) - train_fast.py:10442
            obsr_hold[:] = np.tanh(r_acc / 0.1)
        if not alive.any():
            break
    used = d + 1
    return {"V": V[:used], "R": R[:used], "DONE": DONE[:used],
            "pos": pos[:used * k], "end_dec": end_dec, "fin": fin,
            "trunc": trunc_end, "alive": alive, "ndec": used}


# ===========================================================================
# scoring one batch
# ===========================================================================
def score_batch(batch, g, lam, n_steps, pts, spacing, wall_u, lams=()):
    """Per continuation: the return, the critic's error, the three advantage
    variants at decision 0 (the action taken FROM the start state) and the
    corridor arc it reached. -> (per-episode list of dicts, summary dict)."""
    from diversity_bench import forward_fill, order_only_progress
    V, R, end_dec = batch["V"], batch["R"], batch["end_dec"]
    posf = forward_fill(batch["pos"])
    arc = order_only_progress(posf, pts, spacing)
    arc0 = order_only_progress(posf[:1], pts, spacing)
    n = V.shape[1]
    eps = []
    for i in range(n):
        L = int(end_dec[i] + 1) if end_dec[i] >= 0 else batch["ndec"]
        if L <= 0:
            continue
        # a TRUNCATED continuation's return is missing the trainer's
        # bootstrap g*V(s_T), so it is recorded and then excluded from every
        # statistic rather than quietly biasing the returns downward. An env
        # still alive when the batch stopped is the same case.
        cut = bool(batch["trunc"][i]) or bool(end_dec[i] < 0)
        r = R[:L, i].astype(np.float64)
        v = V[:L, i].astype(np.float64)
        nt = nonterm_mask(L, ended_last=True)
        if cut:
            # nothing here is scored (no bootstrap), and a 120 s cut episode
            # is 3,000 decisions x n_steps phases of pure-Python recursion
            nan = np.full(L, np.nan)
            G = delta = b = c = a_mean = a_min = a_max = nan
        else:
            G = discounted_returns(r, g, nt)
            delta = td_residuals(r, v, g, nt)
            a_mean, a_min, a_max = gae_phase_mean(delta, g, lam, nt, n_steps)
            b = gae(delta, g, 1.0, nt, n_steps=None)      # lambda = 1, full
            c = gae(delta, g, lam, nt, n_steps=None)      # run lambda, full
        eps.append({
            "env": i, "decisions": L,
            "seconds": L * 0.0,             # filled by the caller (act_every)
            "terminated": not cut,
            "truncated": cut,
            "finished": bool(batch["fin"][i]),
            "arc0": float(arc0[i]), "arc": float(arc[i]),
            "past_wall": bool(arc[i] > wall_u),
            "ret": float(G[0]), "V0": float(v[0]),
            "err0": float(v[0] - G[0]),
            "adv_a": float(a_mean[0]), "adv_a_min": float(a_min[0]),
            "adv_a_max": float(a_max[0]),
            "adv_b": float(b[0]), "adv_c": float(c[0]),
            "ep_reward": float(r.sum()),
        })
        # --lam-sweep: variant (a) at other lambdas, this run's n_steps and
        # the same phase average. The question the report has to answer is
        # not "is 0.95 lossy" but "which lambda would an arm use", and that
        # is pure arithmetic on the trajectories already rolled.
        for j, L2 in enumerate(lams):
            eps[-1][f"adv_l{j}"] = (
                float("nan") if cut else
                float(gae_phase_mean(delta, g, float(L2), nt, n_steps)[0][0]))
    return eps


def summarize(eps, tag, horizon, mode, act_every, tick_ms=TICK_MS,
              ref_arc=None, beat_u=128.0):
    """One row per (start state, mode).

    Two success sets, because on a stuck checkpoint the absolute one is
    routinely EMPTY and an empty set answers nothing:

      * ``success`` - corridor arc past the wall, or a finish. The
        frontier definition CLAUDE.md insists on.
      * ``beat``    - arc more than ``beat_u`` (one route vertex) past what
        the GREEDY continuation from the same state reached. A graded
        version of the same question: did the sampled draw find anything
        the deterministic policy did not, and did that draw's first action
        get credit for it?
    """
    n_trunc = int(sum(e["truncated"] for e in eps))
    eps = [e for e in eps if not e["truncated"]]
    if not eps:
        return {"tag": tag, "horizon_s": horizon, "mode": mode, "n": 0,
                "n_trunc": n_trunc}
    sec = act_every * tick_ms / 1000.0
    A = {k: np.array([e[k] for e in eps], np.float64)
         for k in ("ret", "V0", "err0", "adv_a", "adv_b", "adv_c", "arc",
                   "arc0", "decisions")}
    succ = np.array([e["past_wall"] or e["finished"] for e in eps], bool)
    row = {
        "tag": tag, "horizon_s": horizon, "mode": mode, "n": len(eps),
        "len_s_mean": float(A["decisions"].mean() * sec),
        "arc_start": float(A["arc0"].mean()),
        "arc_mean": float(A["arc"].mean()), "arc_max": float(A["arc"].max()),
        "V0_mean": float(A["V0"].mean()), "G0_mean": float(A["ret"].mean()),
        "bias_mean": float(A["err0"].mean()),
        "bias_p10": pct(A["err0"], 10), "bias_p50": pct(A["err0"], 50),
        "bias_p90": pct(A["err0"], 90),
        "bias_sd": float(A["err0"].std()),
        "n_success": int(succ.sum()),
        "frac_success": float(succ.mean()),
        "n_finish": int(sum(e["finished"] for e in eps)),
        "n_trunc": n_trunc,
        "adv_a_all": float(A["adv_a"].mean()),
        "adv_b_all": float(A["adv_b"].mean()),
        "adv_c_all": float(A["adv_c"].mean()),
    }
    beat = (A["arc"] > float(ref_arc) + beat_u) if ref_arc is not None \
        else np.zeros(len(eps), bool)
    row["ref_arc"] = float(ref_arc) if ref_arc is not None else float("nan")
    row["n_beat"] = int(beat.sum())
    row["frac_beat"] = float(beat.mean())
    for name, m in (("succ", succ), ("fail", ~succ),
                    ("beat", beat), ("nobeat", ~beat)):
        if m.any():
            row[f"adv_a_{name}"] = float(A["adv_a"][m].mean())
            row[f"adv_b_{name}"] = float(A["adv_b"][m].mean())
            row[f"adv_c_{name}"] = float(A["adv_c"][m].mean())
            row[f"ret_{name}"] = float(A["ret"][m].mean())
            row[f"bias_{name}"] = float(A["err0"][m].mean())
        else:
            for p in ("adv_a", "adv_b", "adv_c", "ret", "bias"):
                row[f"{p}_{name}"] = float("nan")
    # The separation the question is about: does the winner's first action
    # stand out from the loser's, under each variant?
    #
    # And in the SCALE THAT SURVIVES THE UPDATE. PPO standardises the
    # advantages inside every minibatch (train_fast.py:9153,
    # `a = (a - a.mean()) / (a.std() + 1e-8)`), so a variant that is
    # uniformly 8x smaller than another is not 8x weaker - the scale is
    # divided out. What the policy gradient actually sees is the separation
    # in units of the batch's own spread, so ``_d`` / ``_bd`` (the gap over
    # the batch sd) is the decision-relevant number and the raw gaps are
    # diagnostics for it.
    for p in ("adv_a", "adv_b", "adv_c"):
        row[f"{p}_gap"] = row[f"{p}_succ"] - row[f"{p}_fail"]
        row[f"{p}_bgap"] = row[f"{p}_beat"] - row[f"{p}_nobeat"]
        sd = float(A[p].std())
        row[f"{p}_sd"] = sd
        row[f"{p}_d"] = (row[f"{p}_gap"] / sd) if sd > 0 else float("nan")
        row[f"{p}_bd"] = (row[f"{p}_bgap"] / sd) if sd > 0 else float("nan")
    for k in eps[0]:
        if not k.startswith("adv_l"):
            continue
        a = np.array([e[k] for e in eps], np.float64)
        sd = float(a.std())
        row[f"{k}_bd"] = (((a[beat].mean() - a[~beat].mean()) / sd)
                          if (sd > 0 and beat.any() and (~beat).any())
                          else float("nan"))
    return row


# ===========================================================================
# reporting
# ===========================================================================
def md_lam_sweep(rows, lams, n_steps, act_every, tick_ms=TICK_MS):
    """Beat-set d' (the separation that survives PPO's per-minibatch
    advantage normalisation) at each swept lambda, for the rows that HAVE a
    beat split. This is the table an arm is chosen from."""
    if not lams:
        return ""
    use = [r for r in rows if r.get("n") and 0 < r.get("n_beat", 0) < r["n"]]
    dec_s = act_every * tick_ms / 1000.0
    if not use:
        return ("\nNo (start state, mode) produced BOTH continuations that "
                "beat the greedy one and continuations that did not, so "
                "there is no separation to sweep lambda against.\n")
    head = ["horizon", "mode", "beat", "ep len (dec)"] + [
        f"lam {L:g}" for L in lams]
    out = ["", "Beat-set d' of the FIRST action, variant (a) at this run's "
           f"n_steps = {n_steps} decisions = {n_steps * dec_s:.2f} s:", "",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in use:
        out.append("| " + " | ".join(
            [f"{r['horizon_s']:g} s", r["mode"], f"{r['n_beat']}/{r['n']}",
             f"{r['len_s_mean'] / dec_s:.0f}"]
            + [fmt(r.get(f"adv_l{j}_bd", float("nan")))
               for j in range(len(lams))]) + " |")
    return "\n".join(out) + "\n"


COLS = ["tag", "horizon_s", "mode", "n", "len_s_mean", "arc_start",
        "arc_mean", "arc_max", "ref_arc", "V0_mean", "G0_mean", "bias_mean",
        "bias_sd", "bias_p10", "bias_p50", "bias_p90", "n_success",
        "frac_success", "n_beat", "frac_beat", "n_finish", "n_trunc",
        "adv_a_all", "adv_b_all", "adv_c_all",
        "adv_a_succ", "adv_b_succ", "adv_c_succ", "adv_a_fail", "adv_b_fail",
        "adv_c_fail", "adv_a_beat", "adv_b_beat", "adv_c_beat",
        "adv_a_nobeat", "adv_b_nobeat", "adv_c_nobeat",
        "ret_succ", "ret_fail", "ret_beat", "ret_nobeat",
        "bias_succ", "bias_fail", "bias_beat", "bias_nobeat",
        "adv_a_gap", "adv_b_gap", "adv_c_gap",
        "adv_a_bgap", "adv_b_bgap", "adv_c_bgap",
        "adv_a_sd", "adv_b_sd", "adv_c_sd",
        "adv_a_d", "adv_b_d", "adv_c_d",
        "adv_a_bd", "adv_b_bd", "adv_c_bd"]


def fmt(v):
    if isinstance(v, float):
        if v != v:
            return "n/a"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.4g}"
    return str(v)


def md_table(rows):
    head = ["horizon", "mode", "n", "len s", "arc start", "arc max",
            "V(s0)", "G(s0)", "V-G mean", "V-G p10/p50/p90", "past wall",
            "fin", "beat greedy", "adv0 a/b/c", "adv0 beat a/b/c",
            "adv0 no-beat a/b/c", "beat d' a/b/c"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        if not r.get("n"):
            out.append(f"| {r['horizon_s']:g} s | {r['mode']} | 0 |"
                       + " |" * (len(head) - 3))
            continue
        out.append("| " + " | ".join([
            f"{r['horizon_s']:g} s", r["mode"], str(r["n"]),
            fmt(r["len_s_mean"]), fmt(r["arc_start"]), fmt(r["arc_max"]),
            fmt(r["V0_mean"]), fmt(r["G0_mean"]), fmt(r["bias_mean"]),
            "/".join(fmt(r[k]) for k in ("bias_p10", "bias_p50", "bias_p90")),
            f"{r['n_success']} ({100 * r['frac_success']:.0f}%)",
            str(r["n_finish"]),
            f"{r['n_beat']} ({100 * r['frac_beat']:.0f}%)",
            "/".join(fmt(r[f"{p}_all"]) for p in ("adv_a", "adv_b", "adv_c")),
            "/".join(fmt(r[f"{p}_beat"]) for p in ("adv_a", "adv_b", "adv_c")),
            "/".join(fmt(r[f"{p}_nobeat"])
                     for p in ("adv_a", "adv_b", "adv_c")),
            "/".join(fmt(r[f"{p}_bd"]) for p in ("adv_a", "adv_b", "adv_c")),
        ]) + " |")
    return "\n".join(out)


def md_discount(tbl, gamma_tick, act_every, lam, n_steps):
    lines = [f"gamma per physics tick {gamma_tick:g}; act_every {act_every} "
             f"= {act_every * TICK_MS:g} ms per decision; "
             f"gamma^act_every = {gamma_tick ** act_every:.6f}; "
             f"lambda {lam:g}; n_steps {n_steps} decisions = "
             f"{n_steps * act_every * TICK_MS / 1000.0:.2f} s of rollout.",
             "",
             "| k decisions | seconds | (gamma^act_every * lambda)^k | "
             "gamma^act_every^k | ratio |",
             "|---|---|---|---|---|"]
    for r in tbl:
        lines.append(f"| {r['k_decisions']} | {r['seconds']:.2f} | "
                     f"{r['gae_weight']:.3g} | {r['discount_weight']:.3g} | "
                     f"{r['ratio']:,.0f}x |")
    return "\n".join(lines)


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--map", required=True, help="the MAIN checkout's .bsp")
    ap.add_argument("--route", required=True, help="route .npz next to it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=48,
                    help="continuations per start state (envs)")
    ap.add_argument("--horizons", default=",".join(f"{h:g}" for h in HORIZONS),
                    help="seconds before the probe episode's terminal event")
    ap.add_argument("--temps", default="0",
                    help="sampling temperatures for the stochastic batches "
                         "(diversity_bench's sigma knob: view sigma x (1+T), "
                         "logits / (1+T)). 0 is the trainer's own behaviour "
                         "policy; a wider T says whether ANY continuation "
                         "from this state succeeds, which a null at T=0 "
                         "cannot")
    ap.add_argument("--seed", type=int, default=0,
                    help="core.reset seed of the probe spawn")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--ep-ticks", type=int, default=None)
    ap.add_argument("--lam", type=float, default=None,
                    help="override the checkpoint's GAE lambda")
    ap.add_argument("--lam-sweep", default="0.95,0.97,0.99,0.995,1",
                    help="also score variant (a) at these lambdas (this "
                         "run's n_steps, same phase average), so the report "
                         "can name a value rather than only say that 0.95 "
                         "is lossy. Empty string turns it off.")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="override the checkpoint's rollout length")
    ap.add_argument("--int-coef", type=float, default=None,
                    help="override the run's novelty coefficient (0 = off)")
    ap.add_argument("--no-stall-kill", action="store_true",
                    help="do NOT mirror training's stagnation kill (evals do "
                         "not stall-kill; the trainer does - CLAUDE.md)")
    ap.add_argument("--from-spine", default=None,
                    help="STATE_DTYPE .npy: probe from this recorded episode "
                         "instead of the checkpoint's own greedy one")
    ap.add_argument("--at-tick", type=int, default=0,
                    help="--from-spine: the tick the probe episode ENDS at "
                         "(default: the spine's last row)")
    args = ap.parse_args()

    import torch
    from surfgym.core import STATE_DTYPE
    from diversity_bench import (build_core, build_lidar, forward_fill,
                                 goal_field_and_box, order_only_progress,
                                 race_start_pool)
    from eval_honesty import load_route
    from train_fast import HeadPacker

    t0 = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    check_supported(cfg)
    step = int(ck.get("global_step", 0))
    act_every = int(cfg.get("act_every", 1))
    gamma_tick = float(cfg.get("gamma") or 0.9995)
    lam = float(args.lam if args.lam is not None else (cfg.get("gae") or 0.95))
    n_steps = int(args.n_steps if args.n_steps is not None
                  else (cfg.get("n_steps") or 128))
    g = gamma_tick ** act_every
    ep_ticks = int(args.ep_ticks or cfg.get("ep_ticks", 12000))
    horizons = [float(x) for x in args.horizons.split(",") if x.strip()]
    temps = [float(x) for x in args.temps.split(",") if x.strip()]
    lams = [float(x) for x in args.lam_sweep.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stem = Path(args.map).stem
    if cfg.get("map") and cfg["map"] != stem:
        print(f"WARNING: checkpoint map {cfg['map']!r} != --map {stem!r}")
    print(f"ckpt {Path(args.ckpt).name}  step {step:,}  map {stem}  "
          f"act_every {act_every}  gamma {gamma_tick:g}  lambda {lam:g}  "
          f"n_steps {n_steps}  obs_reward {bool(cfg.get('obs_reward'))}  "
          f"device {device}")

    # -- the probe: the checkpoint's own greedy episode from the eval spawn --
    ref = build_core(str(args.map), cfg, 1, ep_ticks)
    gf, zones, lcell = goal_field_and_box(ref, cfg)
    ref.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    ref.set_spawn_pool(race_start_pool(ref, gf))
    ref.reset(int(args.seed))
    spawn0 = ref.get_states()[0].copy()
    ref.close()

    core = build_core(str(args.map), cfg, int(args.k), ep_ticks,
                      yaw_jitter=0.0)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    lidar = build_lidar(core, cfg, lcell, device, field=gf)
    policy = build_policy(ck, core, lidar, device)
    packer = HeadPacker(device)
    reward_fn, d0 = build_reward(core, cfg, gf, ck, args.int_coef)
    GreedyV, SampledV, TemperedV = make_value_policies()
    pts, spacing = load_route(args.route)
    print(f"route {len(pts)} pts x {spacing:g}u = "
          f"{(len(pts) - 1) * spacing:,.0f}u; start geodesic {d0:,.0f}u; "
          f"reward scale {reward_fn.scale:.6g}/u; setup "
          f"{time.perf_counter() - t0:.0f}s")

    obsr_hold = (np.zeros(int(args.k), np.float32)
                 if cfg.get("obs_reward") else None)
    extra_slot = 12 if obsr_hold is not None else -1
    extra_fn = (lambda _c: obsr_hold) if obsr_hold is not None else None

    def mk(cls):
        return cls(policy, packer, device, lidar, core, act_every, 1,
                   extra_slot=extra_slot, extra_fn=extra_fn)

    def set_start(state):
        core.set_spawn_pool(np.array([state], dtype=STATE_DTYPE))

    # ---- probe episode: greedy from the spawn, every tick's state kept ----
    if args.from_spine:
        spine = np.load(args.from_spine)
        if spine.dtype != STATE_DTYPE:
            raise SystemExit(f"{args.from_spine} is not a STATE_DTYPE spine")
        end_tick = (int(args.at_tick) if args.at_tick
                    else int(spine["tick"].max()))
        ticks = spine["tick"].astype(np.int64)
        probe_states = spine
        probe_ticks = ticks
        probe_end = end_tick
        probe_obsr = None
        if obsr_hold is not None:
            print("  NOTE: --from-spine carries no slot-12 trace, so every "
                  "continuation's FIRST decision reads --obs-reward 0")
        print(f"probe: spine {args.from_spine} ({len(spine)} states, ticks "
              f"{ticks.min()}..{ticks.max()}); terminal event at tick "
              f"{probe_end}")
    else:
        set_start(spawn0)
        pol = mk(GreedyV)
        st = np.empty(ep_ticks, dtype=STATE_DTYPE)
        b = _probe_roll(core, pol, reward_fn, args.seed, ep_ticks, st,
                        obsr_hold, not args.no_stall_kill)
        probe_end = int(b["end_tick"])
        probe_states = st[:probe_end + 1]
        probe_ticks = np.arange(probe_end + 1, dtype=np.int64)
        probe_obsr = b["obsr"][:probe_end + 1]
        pf = forward_fill(b["pos"][:probe_end + 1])
        parc = float(order_only_progress(pf, pts, spacing)[0])
        print(f"probe (greedy from the eval spawn, seed {args.seed}): "
              f"{probe_end + 1} ticks = {(probe_end + 1) / 100.0:.2f} s, "
              f"corridor arc {parc:,.0f}u "
              f"({100.0 * parc / ((len(pts) - 1) * spacing):.2f}%), "
              f"finished {bool(b['fin'][0])}, end z "
              f"{float(probe_states['origin'][-1][2]):,.0f}")

    # ---- one batch per (horizon, mode) ------------------------------------
    rows, per_ep = [], []
    for h in horizons:
        back = int(round(h * 1000.0 / TICK_MS))
        tgt = probe_end - back
        if tgt < 0:
            print(f"  horizon {h:g}s: the probe episode is only "
                  f"{probe_end / 100.0:.2f}s long - skipped")
            continue
        # snap to a DECISION boundary of the probe so the continuation's
        # phase matches the one the policy was acting on
        tgt -= tgt % act_every
        j = int(np.argmin(np.abs(probe_ticks - tgt)))
        s0 = probe_states[j].copy()
        prime = (float(probe_obsr[j]) if (probe_obsr is not None
                                          and obsr_hold is not None) else 0.0)
        set_start(s0)
        modes = [("greedy", GreedyV, 0.0)] + \
                [(("sampled" if T == 0 else f"sampled_T{T:g}"), None, T)
                 for T in temps]
        ref_arc = None
        for mode, cls, T in modes:
            t1 = time.perf_counter()
            if cls is GreedyV:
                pol = mk(GreedyV)
            elif T == 0.0:
                pol = mk(SampledV)
            else:
                pol = TemperedV(policy, packer, device, lidar, core,
                                act_every, 1, extra_slot=extra_slot,
                                extra_fn=extra_fn, temp=1.0 + T)
            batch = roll(core, pol, reward_fn, args.seed, ep_ticks,
                         args.sample_seed, obsr_hold,
                         not args.no_stall_kill, obsr_prime=prime)
            eps = score_batch(batch, g, lam, n_steps, pts, spacing, WALL_U,
                              lams=lams)
            for e in eps:
                e["seconds"] = e["decisions"] * act_every * TICK_MS / 1000.0
                e["horizon_s"] = h
                e["mode"] = mode
            per_ep.extend(eps)
            if mode == "greedy" and eps:
                ref_arc = float(np.median([e["arc"] for e in eps]))
                # SELF-CHECK: a greedy restart from a state h seconds before
                # the probe's terminal event must reproduce the probe's tail,
                # i.e. last about h seconds. A shorter one means the state
                # capture lost something the physics needed.
                gl = float(np.mean([e["seconds"] for e in eps]))
                if h > 0 and abs(gl - h) > max(0.2, 0.15 * h):
                    print(f"    WARNING: the greedy restart at h={h:g}s "
                          f"lasted {gl:.2f}s, not ~{h:g}s - the restart does "
                          f"NOT reproduce the probe's tail")
            row = summarize(eps, Path(args.ckpt).parent.name, h, mode,
                            act_every, ref_arc=ref_arc)
            rows.append(row)
            if not row["n"]:
                print(f"  h={h:4g}s {mode:<12} every continuation was cut "
                      f"at the episode cap ({row['n_trunc']}) - no return "
                      f"to score")
                continue
            print(f"  h={h:4g}s {mode:<12} n={row['n']:3d}  "
                  f"len {row['len_s_mean']:5.2f}s  arc "
                  f"{row['arc_start']:,.0f} -> {row['arc_max']:,.0f}  "
                  f"V0 {row['V0_mean']:8.3f}  G0 {row['G0_mean']:8.3f}  "
                  f"V-G {row['bias_mean']:+8.3f}  wall "
                  f"{row['n_success']:3d}/{row['n']}  fin {row['n_finish']}  "
                  f"beat {row['n_beat']:3d}  adv0 a/b/c "
                  f"{row['adv_a_all']:+.4f}/"
                  f"{row['adv_b_all']:+.4f}/{row['adv_c_all']:+.4f}  "
                  f"beat-d' {row['adv_a_bd']:+5.2f}/{row['adv_b_bd']:+5.2f}/"
                  f"{row['adv_c_bd']:+5.2f}  "
                  f"[{time.perf_counter() - t1:.0f}s]")

    tbl = discount_table(gamma_tick, act_every, lam)
    with open(out / "credit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})
    (out / "episodes.json").write_text(json.dumps(per_ep, indent=0),
                                       encoding="utf-8")
    title = (f"{Path(args.ckpt).parent.name}/{Path(args.ckpt).name} @ "
             f"{step:,}  K={args.k}")
    md = (f"# credit_diag: {title}\n\n"
          + md_discount(tbl, gamma_tick, act_every, lam, n_steps)
          + "\n\nStart states are the probe episode's own states at "
            "0/1/2/5/10 s before its terminal event; each row is K terminal "
            "continuations from that one state, greedy (identical by "
            "construction) or sampled. `adv0` is the advantage the FIRST "
            "decision receives: (a) the trainer's GAE at this run's lambda "
            "and n_steps, averaged over the buffer phase; (b) lambda = 1 "
            "over the whole episode; (c) this lambda, no truncation. "
            "`success` = corridor arc past "
          + f"{WALL_U:,.0f}u or a finish.\n\n"
          + md_table(rows)
          + md_lam_sweep(rows, lams, n_steps, act_every) + "\n")
    (out / "credit.md").write_text(md, encoding="utf-8")
    print("\n" + md_discount(tbl, gamma_tick, act_every, lam, n_steps))
    print("\n" + md_table(rows))
    print(md_lam_sweep(rows, lams, n_steps, act_every))
    print(f"\nwrote {out / 'credit.csv'}, credit.md, episodes.json  "
          f"[{time.perf_counter() - t0:.0f}s total]")
    return 0


def _probe_roll(core, pol, reward_fn, seed, max_ticks, st_out, obsr_hold,
                stall_kill):
    """The probe pass: like :func:`roll` but keeps the FULL STATE of every
    tick (``st_out``), which is what a continuation has to start from - a
    trajectory row is lossy (ducked / induck / duck_time / fuser2 /
    oldbuttons / basevelocity are not in it, and a wrong hull can start an
    episode inside geometry; tools/traj_to_spine.py's opening argument)."""
    import torch
    torch.manual_seed(0)
    if obsr_hold is not None:
        obsr_hold[:] = 0.0
    obs = core.reset(int(seed))
    reward_fn.on_reset(core)
    prev_obs = obs.copy()
    n = core.num_envs
    k = int(pol._k)
    pos = np.full((max_ticks, n, 3), np.nan, np.float32)
    fin = np.zeros(n, bool)
    end_tick = max_ticks - 1
    sv = core.states_view
    r_acc = np.zeros(n, np.float32)
    obsr_trace = np.zeros(max_ticks, np.float32)
    for t in range(max_ticks):
        if t % k == 0:
            if stall_kill:
                sm = reward_fn.pop_stall_mask()
                if sm is not None:
                    core.force_fail(sm)
            r_acc[:] = 0.0
        if obsr_hold is not None:
            obsr_trace[t] = float(obsr_hold[0])
        st_out[t] = core.get_states()[0]
        pos[t] = np.asarray(sv["origin"])
        act = pol.act(obs)
        view = getattr(pol, "view", None)
        if view is None:
            obs, base_r, done, trunc, term = core.step(act)
        else:
            obs, base_r, done, trunc, term = core.step(act, view=view)
        r_acc += np.asarray(reward_fn(prev_obs, obs, term, base_r, done,
                                      trunc, core), np.float32)
        prev_obs = obs.copy()
        if obsr_hold is not None and (t + 1) % k == 0:
            obsr_hold[:] = np.tanh(r_acc / 0.1)
        fin |= np.asarray(core.goal_hits).astype(bool)
        if bool(np.asarray(done)[0]) or bool(np.asarray(trunc)[0]):
            end_tick = t
            break
    return {"pos": pos, "end_tick": end_tick, "fin": fin,
            "obsr": obsr_trace}


if __name__ == "__main__":
    raise SystemExit(main())
