#!/usr/bin/env python3
"""ramp_mpc.py - the witness search for ramp-to-ramp transitions: "from this exact engine state,
touch target ramp B alive, with the most energy", flown by receding-horizon CEM (MPC) in the real
engine, and chains of such edges.

The planner is tools/mpc_bench.py's: plan --plan s ahead, execute --exec ticks of the best plan,
replan from where it landed, warm-started. Controls per knot: side key (A / none / D), forward key,
jump, and the view yaw offset from the live velocity heading. No training, no human line in any
score. A candidate plan scores:

  touched B alive (the FIRST physical contact with B's piece - the trainer's --ramp-touch rule:
  the tick's collision telemetry classified by the map's ramp vocabulary)
      -> 1e6 + the energy height (z + |v|^2/2g) at that touch;
  else alive at the plan's end
      -> -(closest approach to B's standing contact origins over the plan and a PASSIVE
         continuation of --passive s: no keys, view along the velocity)
         + --energy-w x (energy at the plan's end - energy now)
         - --passive-death if the passive continuation dies before touching B;
  died inside the plan before touching B -> discarded.
Reaching the map's finish zone outranks everything. Every constant is generic: none is read off a
map or a record.

Modes:
  --bench-demo <frames.npz> --sequence ...   ANALYSIS BENCH (CLAUDE.md section 0): for each
      consecutive pair (k, k+1) of the sequence, start from the record's state at its first touch
      of target k (RampVocab.contact_of per frame) and fly to target k+1. The record's inputs are
      never read; its own touch times and energies are printed as a ruler only.
  --chain --sequence ...   from the TRUE spawn, fly edge after edge from the search's OWN touch
      states, then replay the whole chain from the spawn to check it is exact. With
      --sequence-source demo the sequence is a TEST HARNESS: it measures whether the transitions
      are flyable, and nothing this run writes may become a training input.

    python tools/ramp_mpc.py --map maps_pool/surf_unitfarmer2.bsp \\
        --vocab runs/research/ramps_mesh_surf_unitfarmer2.npz \\
        --sequence 19,17,18,19,20,22,25,30,33,35,36,37,39,42,43,47,14 --sequence-source demo \\
        --chain --out runs/research/rampmpc/chain_uf2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym import SurfCore, default_config          # noqa: E402
from surfgym.core import STATE_DTYPE                  # noqa: E402
from surfgym.rampvocab import RampVocab               # noqa: E402
from surfgym.zones import load_zones                  # noqa: E402
from kino_search import states_rw                     # noqa: E402

G = 800.0
NORMAL_PREFILTER = 0.99      # a telemetry normal this close to one of B's faces is classified


def energy(o, v):
    o = np.asarray(o, np.float64)
    v = np.asarray(v, np.float64)
    return o[..., 2] + (v * v).sum(-1) / (2 * G)


def strafe_angle(vel, cap_deg):
    """deg: the view offset beyond which an air strafe adds no speed, asin(30 / |v_h|) (30 u/s is
    the engine's per-frame air cap, pm.c), capped at cap_deg (slow and standing states)"""
    v = np.asarray(vel, np.float64)
    vh = np.hypot(v[..., 0], v[..., 1])
    return np.degrees(np.arcsin(np.minimum(1.0, 30.0 / np.maximum(vh, 1e-6)))).clip(0.0, cap_deg)


class Target:
    """one target surface id -> its PIECE (the faces the trainer's touch rule accepts), their
    normals, and a KD-tree over their standing contact origins (the guidance distance)"""

    def __init__(self, voc: RampVocab, sid: int):
        if not (0 <= sid < voc.n_surf) or not voc.is_target[sid]:
            raise ValueError(f"S{sid} is not a target surface of {voc.path}")
        self.sid = int(sid)
        self.piece = int(voc.piece[sid])
        self.faces = np.asarray(voc.piece_faces[self.piece], np.int64)
        self.normals = voc.normal[self.faces].astype(np.float32)
        pts = voc.points[np.isin(voc.surf, self.faces)]
        self.tree = cKDTree(pts)
        self.z_lo, self.z_hi = float(pts[:, 2].min()), float(pts[:, 2].max())


class Engine:
    """the batched core plus the per-tick touch test against one target"""

    def __init__(self, a):
        self.a = a
        bsp = Path(a.map)
        self.zones = load_zones(str(bsp))
        self.core = SurfCore(str(bsp), default_config(
            num_envs=a.pop, lidar_w=0, lidar_h=0, view_mode=1, lookahead_k=0,
            sv_maxvelocity=4000.0, max_episode_ticks=10 ** 8, spawn_mode=1, yaw_jitter_deg=0.0),
            tick_ms=float(a.tick_ms))
        self.core.set_teleport_fail(True)
        self.core.set_goal_box(self.zones["end"]["mins"], self.zones["end"]["maxs"])
        self.core.reset(0)
        self.rw = states_rw(self.core)
        self.sv = self.core.states_view
        self.voc = RampVocab(a.vocab, bsp)
        self.spawn = self.core.get_states()[0].copy()
        self.spawn["velocity"] = (0.0, 0.0, 0.0)
        self.spawn["tick"], self.spawn["stuck_ticks"] = 0, 0
        self.act = np.zeros((a.pop, 6), np.int32)
        self.act[:, 0], self.act[:, 1], self.act[:, 2] = 7, 3, 1
        self.view = np.zeros((a.pop, 2), np.float32)
        self.view[:, 1] = -10.0
        self.TS = a.tick_ms / 1000.0

    def touched(self, tgt: Target, mask):
        """(pop,) bool: this tick's collision telemetry holds a contact with tgt's piece (rows in
        mask only)"""
        counts, normals, points = self.core.get_touch()
        out = np.zeros(len(counts), bool)
        rows = np.flatnonzero((counts > 0) & mask)
        if not len(rows):
            return out
        valid = np.arange(normals.shape[1])[None, :] < counts[rows][:, None]
        dots = np.einsum("rtj,fj->rtf", normals[rows], tgt.normals).max(-1)
        rows = rows[(valid & (dots >= NORMAL_PREFILTER)).any(1)]
        if not len(rows):
            return out
        sid = self.voc.classify(counts[rows], normals[rows], points[rows],
                                np.zeros(len(rows), np.int64))
        pc = np.where(sid >= 0, self.voc.piece[np.maximum(sid, 0)], -2)
        out[rows[(pc == tgt.piece).any(1)]] = True
        return out


def fly_edge(eng: Engine, tgt: Target, start, rng, a):
    """MPC from the exact state `start` until it touches tgt alive (success), dies, or runs
    out of --edge-timeout s. -> dict(result, state at the end, per-tick controls and states)"""
    core, rw, sv, act, view, TS = eng.core, eng.rw, eng.sv, eng.act, eng.view, eng.TS
    pop = a.pop
    Hp = int(round(a.plan / TS))
    Hr = int(round(a.passive / TS))
    K = Hp // a.knot + 1
    n_el = max(4, int(pop * a.elite))
    sig0 = 1.0 if a.off_units == "strafe" else 3.0   # strafe units: 1 = the no-gain edge
    off_lim = a.max_offset if a.off_units == "deg" else a.max_offset_u
    mu = np.zeros(K)
    sig = np.full(K, sig0)
    p_side = np.tile([0.4, 0.2, 0.4], (K, 1))
    p_fwd = np.full(K, a.fwd_p0)
    p_jump = np.full(K, a.jump_p0)
    best = None
    cur = start.copy()
    c_side, c_fwd, c_off, c_jump, c_state = [], [], [], [], []
    n_ticks = int(round(a.edge_timeout / TS))
    t0 = time.perf_counter()
    result, touch_E = "timeout", None
    everyone = np.ones(pop, bool)

    def sample():
        u = rng.random((pop, K, 1))
        side = (u > np.cumsum(p_side, 1)[None, :, :]).sum(-1).clip(0, 2).astype(np.int32)
        if a.hold_p > 0:                           # held keys: a knot repeats the one before
            keep = rng.random((pop, K)) < a.hold_p
            for j in range(1, K):
                side[:, j] = np.where(keep[:, j], side[:, j - 1], side[:, j])
        fwd = np.where(rng.random((pop, K)) < p_fwd, 2, 1).astype(np.int32)
        z = rng.standard_normal((pop, K))
        if a.off_rho > 0:                          # correlated view noise (AR(1) over knots)
            c = np.sqrt(1.0 - a.off_rho ** 2)
            for j in range(1, K):
                z[:, j] = a.off_rho * z[:, j - 1] + c * z[:, j]
        off = np.clip(mu + sig * z, -off_lim, off_lim)
        jmp = (rng.random((pop, K)) < p_jump).astype(np.int32)
        return side, fwd, off, jmp

    Hv = int(round(a.viable / TS))
    pts_b = np.asarray(tgt.tree.data, np.float64)

    def score_plans(side, fwd, off, jmp, e0):
        """every candidate: the plan (Hp ticks), then the HOMING rollout (Hr ticks: offset 0, the
        side key that turns the velocity toward B's nearest contact origin - the strafe that
        both turns and gains speed), then Hv ticks of viability after a touch"""
        rw[:] = cur
        alive = np.ones(pop, bool)
        t_touch = np.full(pop, -1, np.int64)
        e_touch = np.full(pop, -np.inf)
        viable = np.zeros(pop, bool)
        dmin = np.full(pop, np.inf)
        died_plan = np.zeros(pop, bool)
        t_dead = np.zeros(pop, np.float64)
        died_pass = np.zeros(pop, bool)
        goal = np.zeros(pop, bool)
        e_end = np.full(pop, -np.inf)
        home = np.ones(pop, np.int32)              # the homing rollout's controls per env
        h_fwd = np.ones(pop, np.int32)
        h_off = np.zeros(pop, np.float32)
        e_post = np.full(pop, -np.inf)              # energy --viable s after the touch
        for k in range(Hp + Hr + Hv):
            if k < Hp:
                j, f = divmod(k, a.knot)
                j1 = min(j + 1, K - 1)
                u = off[:, j] + (off[:, j1] - off[:, j]) * (f / a.knot)
                view[:, 0] = (np.clip(u * strafe_angle(sv["velocity"], a.max_offset),
                                      -a.max_offset, a.max_offset)
                              if a.off_units == "strafe" else u)
                act[:, 2], act[:, 3], act[:, 4] = fwd[:, j], side[:, j], jmp[:, j]
            else:
                if k == Hp:
                    act[:, 4] = 0
                act[:, 2], act[:, 3] = h_fwd, home
                view[:, 0] = h_off
            _o, _r, done, trunc, _t = core.step(act, view=view)
            ended = (done | trunc).astype(bool) & alive
            if ended.any():
                gh = core.goal_hits.astype(bool) & ended
                goal |= gh
                pre = ended & ~gh & (t_touch < 0)
                if k < Hp:
                    died_plan |= pre
                    t_dead[pre] = k
                else:
                    died_pass |= pre
                alive &= ~ended                    # a touch that dies here is never viable
            newly = alive & (t_touch >= 0) & ~viable & (k - t_touch >= Hv)
            if newly.any():
                viable |= newly
                e_post[newly] = energy(sv["origin"][newly], sv["velocity"][newly])
            open_ = alive & (t_touch < 0)
            if open_.any() and k < Hp + Hr:
                hit = eng.touched(tgt, open_)
                if hit.any():
                    t_touch[hit] = k
                    e_touch[hit] = energy(sv["origin"][hit], sv["velocity"][hit])
                    open_ &= ~hit
                if open_.any() and (k % a.dist_every == 0 or k == Hp - 1):
                    o = np.asarray(sv["origin"], np.float64)[open_]
                    d, idx = tgt.tree.query(o, k=1)
                    dmin[open_] = np.minimum(dmin[open_], d)
                    if k >= Hp - 1:                # the homing controls: toward B
                        q = pts_b[idx]
                        vv = np.asarray(sv["velocity"], np.float64)[open_]
                        dx, dy = q[:, 0] - o[:, 0], q[:, 1] - o[:, 1]
                        cross = vv[:, 0] * dy - vv[:, 1] * dx
                        ground = np.asarray(sv["onground"])[open_] >= 0
                        # in the air: strafe toward B at offset 0 (turns and gains speed);
                        # on the ground: walk toward B (W, the view turned to B's bearing; the
                        # core's base is the velocity heading at >= 100 u/s, else the yaw)
                        vh = np.hypot(vv[:, 0], vv[:, 1])
                        base = np.where(vh >= 100.0, np.degrees(np.arctan2(vv[:, 1], vv[:, 0])),
                                        np.asarray(sv["yaw"], np.float64)[open_])
                        rel = (np.degrees(np.arctan2(dy, dx)) - base + 180.0) % 360.0 - 180.0
                        if not a.ground_homing:
                            ground[:] = False
                        home[open_] = np.where(ground, 1, np.where(cross > 0, 0, 2))
                        h_fwd[open_] = np.where(ground, 2, 1)
                        h_off[open_] = np.where(ground, rel, 0.0).astype(np.float32)
            if k == Hp - 1:
                e_end = np.where(alive, energy(sv["origin"], sv["velocity"]), -np.inf)
            pending = alive & (((t_touch < 0) & (k < Hp + Hr - 1)) | ((t_touch >= 0) & ~viable))
            if not pending.any():
                break
        late = alive & (t_touch >= 0) & ~viable    # still alive when the simulation ran out
        viable |= late
        e_post[late] = energy(sv["origin"][late], sv["velocity"][late])
        score = np.full(pop, -np.inf)
        # died inside the plan before touching B: ranked by how long it lived, below every
        # survivor, so the distribution can still move when no candidate survives the plan
        score[died_plan] = -1e9 + t_dead[died_plan]
        ok = t_touch >= 0
        if a.score == "soft":
            # ONE currency, energy height: a touch is worth its energy + --touch-bonus; a plan that
            # has not touched is worth its energy at the plan's end less --dist-w per u still to go
            # (the closest approach of plan + homing rollout); a touch that dies within --viable s
            # forfeits --touch-bonus twice. A low-energy touch no longer outranks a carve that keeps
            # its energy on the way.
            et = e_post if a.touch_score == "post" else e_touch
            score[ok & viable] = et[ok & viable] + a.touch_bonus
            score[ok & ~viable] = e_touch[ok & ~viable] - a.touch_bonus
            nt = ~ok & ~died_plan & np.isfinite(e_end)
            score[nt] = e_end[nt] - a.dist_w * dmin[nt] - a.passive_death * died_pass[nt]
            score[goal] = 1e7
            return score, t_touch
        if a.touch_score == "post":                # the entry's quality: energy kept riding B
            score[ok & viable] = 1e6 + e_post[ok & viable]
        else:
            score[ok & viable] = 1e6 + e_touch[ok & viable]
        score[ok & ~viable] = 5e5 + e_touch[ok & ~viable]     # touched, then died: not viable
        nt = ~ok & ~died_plan & np.isfinite(e_end)
        score[nt] = (-dmin[nt] + a.energy_w * (e_end[nt] - e0)
                     - a.passive_death * died_pass[nt])
        score[goal] = 2e6
        return score, t_touch

    replans = 0
    d_best, t_best = np.inf, 0
    while len(c_side) < n_ticks:
        e0 = float(energy(cur["origin"], cur["velocity"]))
        for _it in range(a.iters):
            side, fwd, off, jmp = sample()
            if best is not None:
                side[0], fwd[0], off[0], jmp[0] = best
            score, _tt = score_plans(side, fwd, off, jmp, e0)
            el = np.argsort(-score)[:n_el]
            el = el[np.isfinite(score[el])]
            if not len(el):
                continue
            best = (side[el[0]].copy(), fwd[el[0]].copy(), off[el[0]].copy(), jmp[el[0]].copy())
            if len(el) >= 2:
                mu = 0.7 * off[el].mean(0) + 0.3 * mu
                sig = np.maximum(0.7 * off[el].std(0) + 0.3 * sig, 0.05 * sig0)
                ps = np.stack([(side[el] == c).mean(0) for c in range(3)], 1)
                p_side = np.clip(0.7 * ps + 0.3 * p_side, 0.02, 0.98)
                p_side /= p_side.sum(1, keepdims=True)
                p_fwd = np.clip(0.7 * (fwd[el] == 2).mean(0) + 0.3 * p_fwd, 0.02, 0.98)
                p_jump = np.clip(0.7 * (jmp[el] == 1).mean(0) + 0.3 * p_jump, 0.02, 0.98)
        replans += 1
        if best is None:
            result = "no surviving plan"
            break
        b_side, b_fwd, b_off, b_jmp = best
        rw[:] = cur
        stop = False
        for k in range(a.exec):
            j, f = divmod(k, a.knot)
            j1 = min(j + 1, K - 1)
            u = b_off[j] + (b_off[j1] - b_off[j]) * (f / a.knot)
            view[:, 0] = (np.clip(u * strafe_angle(sv["velocity"], a.max_offset),
                                  -a.max_offset, a.max_offset)
                          if a.off_units == "strafe" else u)
            act[:, 2], act[:, 3], act[:, 4] = b_fwd[j], b_side[j], b_jmp[j]
            _o, _r, done, trunc, _t = core.step(act, view=view)
            c_side.append(int(b_side[j])); c_fwd.append(int(b_fwd[j]))
            c_off.append(float(view[0, 0])); c_jump.append(int(b_jmp[j]))
            c_state.append(sv[0].copy())
            if done[0] or trunc[0]:
                result = "FINISH" if core.goal_hits[0] else "DIED"
                stop = True
                break
            if eng.touched(tgt, everyone)[0]:
                result = "TOUCH"
                touch_E = float(energy(sv["origin"][0], sv["velocity"][0]))
                stop = True
                break
        cur = sv[0].copy()
        if stop:
            break
        if getattr(a, "stall_abort", 0) > 0:      # no progress toward B for --stall-abort s
            d0_, _ = tgt.tree.query(np.asarray(cur["origin"], np.float64)[None], k=1)
            if float(d0_[0]) < d_best - 50.0:
                d_best, t_best = float(d0_[0]), len(c_side)
            elif (len(c_side) - t_best) * TS > a.stall_abort:
                result = "stalled"
                break
        s = a.exec // a.knot
        mu = np.r_[mu[s:], np.repeat(mu[-1], s)]
        sig = np.r_[np.maximum(sig[s:], sig0 / 3.0), np.full(s, sig0)]
        p_side = np.r_[p_side[s:], np.tile([0.4, 0.2, 0.4], (s, 1))]
        p_fwd = np.r_[p_fwd[s:], np.full(s, a.fwd_p0)]
        p_jump = np.r_[p_jump[s:], np.full(s, a.jump_p0)]
        best = tuple(np.r_[x[s:], np.repeat(x[-1:], s)] for x in best)
    return dict(result=result, state=cur, touch_E=touch_E,
                game_s=len(c_side) * TS, compute_s=time.perf_counter() - t0, replans=replans,
                side=np.asarray(c_side, np.int8), fwd=np.asarray(c_fwd, np.int8),
                off=np.asarray(c_off, np.float32), jump=np.asarray(c_jump, np.int8),
                states=np.asarray(c_state, STATE_DTYPE))


def fly_with_retries(eng, tgt, start, a, seed_base):
    tries = []
    for r in range(a.retries + 1):
        res = fly_edge(eng, tgt, start, np.random.default_rng(seed_base + 7919 * r), a)
        tries.append(res)
        if res["result"] in ("TOUCH", "FINISH"):
            break
    return tries[-1], tries


def replay(eng, start, side, fwd, off, jump):
    """the chain's controls from `start` on env 0 alone -> the per-tick states"""
    core, rw, sv, act, view = eng.core, eng.rw, eng.sv, eng.act, eng.view
    rw[:] = start
    out = []
    for k in range(len(side)):
        view[:, 0] = off[k]
        act[:, 2], act[:, 3], act[:, 4] = fwd[k], side[k], jump[k]
        core.step(act, view=view)
        out.append(sv[0].copy())
    return np.asarray(out, STATE_DTYPE)


def record_touches(voc, w, seq):
    """the record's first touch of each sequence target in order (frame indices, -1 missing),
    from its positions alone (RampVocab.contact_of, standing hull)"""
    org = np.asarray(w["simorg"], np.float64)
    sid = voc.contact_of(org, np.zeros(len(org), np.int64))
    pc = np.where(sid >= 0, voc.piece[np.maximum(sid, 0)], -2)
    out, i0 = [], 0
    for s in seq:
        hit = np.flatnonzero(pc[i0:] == voc.piece[s])
        if not len(hit):
            out.append(-1)
            continue
        i = i0 + int(hit[0])
        out.append(i)
        i0 = i + 1
    return out


def state_from_frame(eng, w, i):
    st = eng.spawn.copy()
    st["origin"], st["velocity"] = w["simorg"][i], w["simvel"][i]
    st["yaw"], st["onground"] = float(w["viewangles"][i, 1]), -1
    st["tick"], st["stuck_ticks"] = 0, 0
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--vocab", required=True, help="tools/ramps_mesh.py v4 npz of this map")
    ap.add_argument("--tick-ms", type=int, default=10)
    ap.add_argument("--sequence", required=True, help="comma-separated target surface ids")
    ap.add_argument("--sequence-source", required=True, choices=("demo", "self", "search"),
                    help="where the sequence came from; 'demo' = a human record: TEST HARNESS only")
    ap.add_argument("--bench-demo", default=None, help="per-edge ANALYSIS BENCH from a record")
    ap.add_argument("--edges", default=None, help="bench: edge indices, e.g. 0-3 or 2,5")
    ap.add_argument("--chain", action="store_true", help="from the true spawn, edge after edge")
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--elite", type=float, default=0.05)
    ap.add_argument("--plan", type=float, default=1.2, help="planning horizon, s")
    ap.add_argument("--passive", type=float, default=2.0,
                    help="s of the HOMING rollout after the plan (offset 0, the side key that turns "
                         "toward B)")
    ap.add_argument("--score", default="lex", choices=("lex", "soft"),
                    help="lex: any viable touch outranks every non-touch (1e6 + energy); soft: one "
                         "energy currency (touch: energy + --touch-bonus; no touch: energy - "
                         "--dist-w x distance to go)")
    ap.add_argument("--touch-bonus", type=float, default=300.0)
    ap.add_argument("--dist-w", type=float, default=0.2)
    ap.add_argument("--touch-score", default="touch", choices=("touch", "post"),
                    help="a viable touch scores its energy AT the touch, or --viable s AFTER it "
                         "(post: an entry that cannot ride B loses that energy)")
    ap.add_argument("--ground-homing", type=int, default=0, choices=(0, 1),
                    help="1 = on the ground the homing rollout walks toward B (W, view to its "
                         "bearing); 0 = it strafes as in the air (the first-edge test preferred 0)")
    ap.add_argument("--viable", type=float, default=1.0,
                    help="s a touch must survive (plan / homing controls) to count as viable")
    ap.add_argument("--exec", type=int, default=8, help="ticks executed per replan")
    ap.add_argument("--knot", type=int, default=4)
    ap.add_argument("--max-offset", type=float, default=8.0, help="deg: the view offset's clip")
    ap.add_argument("--off-units", default="deg", choices=("deg", "strafe"),
                    help="sampled view offsets in degrees, or in units of the strafe angle "
                         "asin(30/|v_h|) - the offset past which a strafe stops adding speed "
                         "(0.57 deg at 3,000 u/s); the result is clipped to --max-offset deg")
    ap.add_argument("--max-offset-u", type=float, default=4.0,
                    help="--off-units strafe: the sampled offset's clip, in strafe units")
    ap.add_argument("--fwd-p0", type=float, default=0.3)
    ap.add_argument("--jump-p0", type=float, default=0.1)
    ap.add_argument("--energy-w", type=float, default=1.0)
    ap.add_argument("--passive-death", type=float, default=1000.0)
    ap.add_argument("--dist-every", type=int, default=4)
    ap.add_argument("--edge-timeout", type=float, default=10.0, help="game s per edge")
    ap.add_argument("--stall-abort", type=float, default=0.0,
                    help="give an edge up when its distance to B has not improved by 50 u for "
                         "this many game s (0 = off)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--hold-p", type=float, default=0.8,
                    help="sampled side key repeats the previous knot's with this probability")
    ap.add_argument("--off-rho", type=float, default=0.8,
                    help="AR(1) correlation of the sampled view offsets across knots")
    ap.add_argument("--bench-back", type=int, default=5,
                    help="bench: start this many record frames before its touch of target k")
    ap.add_argument("--start-npz", default=None, help="chain: resume this saved chain")
    ap.add_argument("--start-index", type=int, default=-1, help="its state after this tick index")
    ap.add_argument("--seq-from", type=int, default=0, help="the sequence index to fly to first")
    ap.add_argument("--seq-to", type=int, default=-1, help="stop after this sequence index")
    ap.add_argument("--beam", type=int, default=1,
                    help="chain: entries tried per edge before backtracking (1 = no backtracking, "
                         "--retries per edge)")
    ap.add_argument("--budget", type=int, default=1000, help="chain: total edge flights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.bench_demo and not a.chain:
        ap.error("pick --bench-demo and/or --chain")

    eng = Engine(a)
    seq = [int(s) for s in a.sequence.split(",") if s.strip()]
    tg = {s: Target(eng.voc, s) for s in set(seq)}
    print(eng.voc.describe(), flush=True)
    print(f"sequence ({a.sequence_source}): " + " -> ".join(f"S{s}" for s in seq), flush=True)
    if a.sequence_source == "demo":
        print("!! the sequence was read off a human record: TEST HARNESS only - nothing this run "
              "writes may become a training input (CLAUDE.md section 0)", flush=True)
    summary = dict(args=vars(a), bench=[], chain=[])

    if a.bench_demo:
        w = np.load(a.bench_demo)
        wt = w["t"] - w["t"][0]
        idx = record_touches(eng.voc, w, seq)
        print(f"!! ANALYSIS BENCH from {Path(a.bench_demo).name}: starts at the record's touch "
              f"states; its inputs are never read", flush=True)
        print("  record touches: " + ", ".join(
            f"S{s}@{wt[i]:.2f}s" if i >= 0 else f"S{s}@-" for s, i in zip(seq, idx)), flush=True)
        if a.edges:
            sel = []
            for part in a.edges.split(","):
                lo, _, hi = part.partition("-")
                sel += list(range(int(lo), int(hi or lo) + 1))
        else:
            sel = list(range(len(seq) - 1))
        for e in sel:
            i0, i1 = idx[e], idx[e + 1]
            if i0 < 0:
                print(f"  edge {e}: S{seq[e]} -> S{seq[e + 1]}: the record never touches S{seq[e]}")
                continue
            st = state_from_frame(eng, w, max(0, i0 - a.bench_back))
            e_st = float(energy(st["origin"], st["velocity"]))
            res, tries = fly_with_retries(eng, tg[seq[e + 1]], st, a, a.seed + 1000 * e)
            rec = (f"record {wt[i1] - wt[i0]:5.2f} s, E {energy(w['simorg'][i1], w['simvel'][i1]):5.0f}"
                   if i1 >= 0 else "record -")
            got = (f"{res['game_s']:5.2f} s, E {res['touch_E']:5.0f}" if res["touch_E"] is not None
                   else f"{res['game_s']:5.2f} s")
            print(f"  edge {e:2d}: S{seq[e]:<3d} -> S{seq[e + 1]:<3d} E0 {e_st:5.0f}: "
                  f"{res['result']:<17s} {got} | {rec} | tries {len(tries)}, "
                  f"{sum(t['compute_s'] for t in tries):5.1f}s compute", flush=True)
            summary["bench"].append(dict(edge=e, src=seq[e], dst=seq[e + 1], result=res["result"],
                                         game_s=res["game_s"], touch_E=res["touch_E"], E0=e_st,
                                         tries=len(tries),
                                         compute_s=sum(t["compute_s"] for t in tries)))

    if a.chain:
        cur = eng.spawn.copy()
        parts = []
        k0 = 0
        if a.start_npz:                     # resume a saved chain: its state after tick i
            z = np.load(a.start_npz)
            i = a.start_index if a.start_index >= 0 else len(z["states"]) + a.start_index
            parts.append(dict(side=z["side"][:i + 1], fwd=z["fwd"][:i + 1], off=z["off"][:i + 1],
                              jump=z["jump"][:i + 1], states=z["states"][:i + 1]))
            cur = z["states"][i].copy()
            k0 = a.seq_from
            print(f"CHAIN resumed from {Path(a.start_npz).name} after tick {i + 1} "
                  f"(z {cur['origin'][2]:.0f}, E {energy(cur['origin'], cur['velocity']):.0f}), "
                  f"from sequence index {k0}", flush=True)
        else:
            print(f"CHAIN from the true spawn (z {cur['origin'][2]:.0f}), tick {a.tick_ms} ms",
                  flush=True)
        game0 = len(parts[0]["side"]) * eng.TS if parts else 0.0
        # depth-first over edges: at each edge up to --beam flights from the SAME state (new seeds),
        # each successful one recursed into; a failure downstream comes back and tries the next
        # entry - the touch state an edge leaves decides whether the next edge is flyable
        n_fl = [0]
        best = dict(k=-1, parts=list(parts))
        last_fail = {}

        def dfs(k, cur, chain, game, depth):
            while k < len(seq) and (k < k0 or (k > 0 and seq[k - 1] == seq[k])):
                k += 1
            if k >= len(seq) or (a.seq_to >= 0 and k > a.seq_to):
                return chain
            s_k = seq[k]
            for m in range(a.beam):
                if n_fl[0] >= a.budget:
                    return None
                if a.beam == 1:
                    res, tries = fly_with_retries(eng, tg[s_k], cur, a, a.seed + 100 * k)
                else:
                    res = fly_edge(eng, tg[s_k], cur,
                                   np.random.default_rng(a.seed + 100 * k + 7919 * m), a)
                    tries = [res]
                n_fl[0] += len(tries)
                g = game + res["game_s"]
                e_txt = f"E {res['touch_E']:5.0f}" if res["touch_E"] is not None else ""
                print(f"  {'  ' * depth}-> S{s_k:<3d} [{m}] {res['result']:<17s} +{res['game_s']:5.2f} s "
                      f"(t {g:6.2f}) {e_txt} | {sum(t['compute_s'] for t in tries):5.1f}s compute, "
                      f"flight {n_fl[0]}", flush=True)
                summary["chain"].append(dict(k=k, m=m, dst=s_k, result=res["result"],
                                             game_s=res["game_s"], t=g, touch_E=res["touch_E"],
                                             tries=len(tries),
                                             compute_s=sum(t["compute_s"] for t in tries)))
                if res["result"] not in ("TOUCH", "FINISH"):
                    last_fail[id(chain)] = res
                    continue
                nxt = chain + [res]
                if k > best["k"]:
                    best.update(k=k, parts=nxt)
                if res["result"] == "FINISH":
                    return nxt
                r = dfs(k + 1, res["state"], nxt, g, depth + 1)
                if r is not None:
                    return r
            return None

        found = dfs(k0, cur, list(parts), game0, 0)
        parts = found if found is not None else best["parts"]
        if found is None and id(best["parts"]) in last_fail:
            parts = parts + [last_fail[id(best["parts"])]]   # its failing flight, for diagnosis
            summary["last_segment_failed"] = True
        print(f"  {'COMPLETE' if found is not None else 'deepest'}: sequence index {best['k']} "
              f"after {n_fl[0]} flights", flush=True)
        if parts:
            side = np.concatenate([p["side"] for p in parts])
            fwd = np.concatenate([p["fwd"] for p in parts])
            off = np.concatenate([p["off"] for p in parts])
            jump = np.concatenate([p["jump"] for p in parts])
            states = np.concatenate([p["states"] for p in parts])
            rep = replay(eng, eng.spawn, side, fwd, off, jump)
            err = float(np.abs(np.asarray(rep["origin"], np.float64)
                               - np.asarray(states["origin"], np.float64)).max()) if len(rep) else 0.0
            print(f"  replay from the spawn: {len(side)} ticks, max position error {err:.2f} u",
                  flush=True)
            summary["replay_err"] = err
            if a.out:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f"{a.out}.npz", side=side, fwd=fwd, off=off, jump=jump,
                                    states=states, spawn=np.asarray([eng.spawn], STATE_DTYPE),
                                    sequence=np.asarray(seq), tick_ms=a.tick_ms,
                                    sequence_source=a.sequence_source)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(f"{a.out}.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
