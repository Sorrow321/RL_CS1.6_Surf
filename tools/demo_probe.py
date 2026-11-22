#!/usr/bin/env python3
"""demo_probe.py - teacher-forced evaluation of a checkpoint's policy ALONG a recorded line.

The user (2026-11-22), with their own pass of surf_left_turn: "substitute my actions into the
network and compute the probabilities, look at the policy, look at the value function, and see
which of the actions are super not preferred by the network ... maybe some layers explode ... a
kind of error analysis". ANALYSIS ONLY (CLAUDE.md section 0: a demo is never training data; this
computes no gradient and writes nothing a trainer reads).

At every decision of the line (every act_every ticks) the simulator is PUT at the recorded state
(core.set_state, tick counter advanced by exactly act_every so the wrapper's per-decision side
state - held keys, action history, ratchet record, surf flag, field rate - updates as in a real
episode), the 15 core scalars are computed from that state with src/env.c write_obs's formulas
(the policy reads 0..6 and 9..11; heading and position 7, 8, 12..14 are masked out of its input),
and the wrapper builds the observation the policy would read there: the recorder's renderer at the
recorded pose, the run's side features. One forward gives the key heads' distribution (masked as
the agent samples them), the view target's mean, and the value. The line's OWN keys and view are
then forced as the decision (AzPolicy's forced rows), so the held keys and the action history
the next decision reads are the line's, as if the agent had flown it.

Per decision: the probability of the line's key state on each key group (fwd / side / jump /
duck; a held head's probability is the explicit bin plus "keep" when the held key already equals
it), the policy's own choice and its probability, the view target the policy prefers next to the
line's (degrees off the velocity heading, and in units of the policy's own sigma), the value, the
field distance. Forward hooks (tools/act_probe.Probe) collect the per-layer activations along the
line for comparison with the agent's own flights.

Input (--az-probe-line, an .npz written without pickles): states STATE_DTYPE (n,) one per
simulator tick at the checkpoint's tick, keys int (n, 4) ENGINE bins (fwd 0 back / 1 none /
2 forward, side 0 left / 1 none / 2 right, jump 0/1, duck 0/1), view float (n, 2) (yaw offset deg
off the velocity heading, pitch deg in the simulator's sign), t float (n,) seconds.

    python tools/beam_tas.py CKPT --map maps/M.bsp ... --az --az-k 1 --envs 1 --az-sims 1 \\
        --az-probe-line runs/line.npz --out-dir runs/probe
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

GROUPS = ("fwd", "side", "jump", "duck")


def core_scalars(st, prev, yaw_rate, pitch_rate, obs_dim):
    """src/env.c write_obs for one state (o[10] / o[11]: the last tick's view change)."""
    o = np.zeros((1, int(obs_dim)), np.float32)
    yaw = np.radians(float(st["yaw"]))
    cy, sy = np.cos(yaw), np.sin(yaw)
    vx, vy, vz = (float(x) for x in st["velocity"])
    o[0, 0] = (vx * cy + vy * sy) / 1000.0
    o[0, 1] = (-vx * sy + vy * cy) / 1000.0
    o[0, 2] = vz / 1000.0
    o[0, 3] = np.hypot(vx, vy) / 1000.0
    o[0, 4] = 1.0 if int(st["onground"]) != -1 else 0.0
    o[0, 5] = 1.0 if int(st["ducked"]) else 0.0
    o[0, 6] = 1.0 if (int(st["oldbuttons"]) & 2) else 0.0
    o[0, 7], o[0, 8] = sy, cy
    o[0, 9] = float(st["pitch"]) / 90.0
    yd = (float(st["yaw"]) - float(prev["yaw"]) + 180.0) % 360.0 - 180.0
    pd = float(st["pitch"]) - float(prev["pitch"])
    o[0, 10] = yd / yaw_rate if yaw_rate > 0 else 0.0
    o[0, 11] = pd / pitch_rate if pitch_rate > 0 else 0.0
    return o


def run_probe_line(ctx: dict, a) -> dict:
    import torch
    from act_probe import Probe
    from az_mcts import make_az_policy
    from surfgym.view import view_from_z_abs, z_from_view_abs
    from train_fast import N_VIEW, sample_view, split_view

    AzPolicy = make_az_policy()

    class ProbePolicy(AzPolicy):
        """AzPolicy whose decision also keeps the forward it was made from."""
        rec = None

        @torch.inference_mode()
        def _decide(self, obs):
            logits, v = self._net(self._obs(obs))
            cat, mu = split_view(logits.float())
            padded = self._mask_padded(self.packer.pad(cat), obs)
            self.rec = {"padded": padded.float().cpu().numpy(), "mu": mu.float().cpu().numpy(),
                        "value": v.float().reshape(-1).cpu().numpy()}
            act, z, _ = sample_view(padded, mu, self.policy.log_std())
            if self.force_idx is not None and len(self.force_idx):
                fi = torch.as_tensor(self.force_idx, device=act.device)
                act[fi, N_VIEW:] = torch.as_tensor(self.force_act, device=act.device, dtype=act.dtype)
                z[fi] = torch.as_tensor(self.force_z, device=z.device, dtype=z.dtype)
            return self._finish_view(act[:, N_VIEW:], z)

    if not ctx["VIEWC"] or ctx["VIEW_ABS"] != "velocity":
        raise SystemExit("--az-probe-line reads the view as a velocity-frame target: a "
                         "--view-continuous --view-absolute velocity checkpoint")
    K = int(ctx["K"])
    cfg = ctx["cfg"]
    L = np.load(a.az_probe_line, allow_pickle=False)
    S, KEYS, VIEW = L["states"], np.asarray(L["keys"], np.int64), np.asarray(L["view"], np.float64)
    T = np.asarray(L["t"], np.float64) if "t" in L.files else np.arange(len(S)) * float(ctx["TICK"].ms) / 1e3
    core = ctx["build_sim"](cfg, ctx["map_path"], 1, ctx["ep_cap"], tick=ctx["TICK"])
    ctx["arm"](core)
    obs = np.array(core.reset(int(ctx["gate_seed"])))
    core.set_state(0, ctx["row0"])
    obs[:] = np.asarray(ctx["obs_start"], np.float32)[None, :]
    es, ef, fstate, lf = ctx["mk_feed"]()
    xkw, xfe = ctx["XF"].make(core)
    pol = ProbePolicy(ctx["policy"], ctx["packer"], ctx["device"], ctx["lidar"], core, K,
                      ctx["stack"], extra_slot=es, extra_fn=ef, latch_fn=lf, cc_fn=ctx["cc_fn1"],
                      keys_hold=ctx["KEYS"], **xkw)
    pol.spawn_duck = ctx["SPAWN_DUCK"]
    pol.greedy = np.zeros(1, bool)
    with torch.inference_mode():
        pol._obs(obs)
    yaw_rate = float(core.config.yaw_rate_max_deg)
    pitch_rate = float(core.config.pitch_rate_max_deg)
    obs_dim = int(np.asarray(obs).shape[1])
    keys_hold = bool(ctx["KEYS"])
    log_std = ctx["policy"].log_std().float().detach().cpu().numpy().ravel()
    probe = Probe(ctx["policy"], 0)
    gf = ctx["gf"]
    from mcts_search import field_d
    rows = []
    n_dec = (len(S) - 1) // K
    for k in range(n_dec):
        j = k * K
        st = np.array(S[j], copy=True).reshape(1)
        st["tick"] = j
        st["stuck_ticks"] = 0
        core.set_state(0, st[0])
        prev = S[j - 1] if j > 0 else S[j]
        o = core_scalars(st[0], prev, yaw_rate, pitch_rate, obs_dim)
        held = (np.asarray(pol.keys.state[0]).copy() if (keys_hold and pol.keys is not None)
                else None)
        hk = KEYS[j]
        # the line's keys as the policy's row: a held head's explicit bin is engine bin + 1
        if keys_hold:
            row = np.array([hk[0] + 1, hk[1] + 1, hk[2], hk[3] + 1], np.int64)
        else:
            row = np.array(hk, np.int64)
        zl = z_from_view_abs(VIEW[j][None], "velocity")[0]
        pol.force_idx = np.array([0])
        pol.force_act = row[None]
        pol.force_z = zl[None]
        pol._tick, pol._held = 0, None
        pol.act(o)
        r = pol.rec
        P = r["padded"][0, N_VIEW:]                  # (4 key heads, NPAD) masked logits
        rec = {"k": k, "tick": j, "t": float(T[j]),
               "origin": [round(float(x), 1) for x in st[0]["origin"]],
               "hspeed": round(float(np.hypot(*st[0]["velocity"][:2])), 1),
               "vz": round(float(st[0]["velocity"][2]), 1), "onground": int(st[0]["onground"] != -1),
               "value": float(r["value"][0]),
               "d": float(field_d(gf, np.asarray(st[0]["origin"], np.float64).reshape(1, 3))[0])}
        for h, g in enumerate(GROUPS):
            lg = P[h].astype(np.float64)
            p = np.exp(lg - lg.max())
            p = p / p.sum()
            if keys_hold and g != "jump":
                # a held head: bin 0 keeps the held key, bins 1.. are engine keys 0..; the
                # probability of ENDING UP with engine key e is the explicit bin plus "keep"
                # when the held key is already e
                ne = int(np.sum(lg > -1e29)) - 1
                hh = int(held[{"fwd": 0, "side": 1, "duck": 2}[g]]) if held is not None else -1
                pe = np.array([p[e + 1] + (p[0] if hh == e else 0.0) for e in range(ne)])
            else:
                pe = p[: int(np.sum(lg > -1e29))]
            key = int(np.argmax(pe))
            rec[g] = {"line": int(hk[h]), "p_line": round(float(pe[int(hk[h])]), 4),
                      "policy_key": key, "p_policy": round(float(pe[key]), 4),
                      "p_keep": round(float(p[0]), 4) if (keys_hold and g != "jump") else None}
        mu = r["mu"][0].astype(np.float64)
        vw = view_from_z_abs(mu[None], "velocity")[0]
        rec["view"] = {"line_yaw_off": round(float(VIEW[j][0]), 2), "policy_yaw_off": round(float(vw[0]), 2),
                       "line_pitch": round(float(VIEW[j][1]), 2), "policy_pitch": round(float(vw[1]), 2),
                       "yaw_sigmas": round(float((zl[0] - mu[0]) / np.exp(log_std[0])), 2),
                       "pitch_sigmas": round(float((zl[1] - mu[1]) / np.exp(log_std[1])), 2)}
        rows.append(rec)
    probe.on = False
    acts = probe.summary()
    probe.remove()
    out = {"line": str(a.az_probe_line), "decisions": len(rows), "K": K,
           "tick_ms": float(ctx["TICK"].ms), "rows": rows, "activations": acts}
    return out


def summarize(res: dict) -> dict:
    """The headline numbers of a probe result (printed and kept in the summary)."""
    rows = res["rows"]
    s = {"decisions": len(rows)}
    for g in GROUPS:
        p = np.array([r[g]["p_line"] for r in rows])
        s[g] = {"mean_p_line": float(p.mean()), "share_below_0.1": float(np.mean(p < 0.1)),
                "share_below_0.01": float(np.mean(p < 0.01)),
                "agree_with_policy": float(np.mean([r[g]["line"] == r[g]["policy_key"] for r in rows]))}
    ys = np.array([abs(r["view"]["yaw_sigmas"]) for r in rows])
    yo = np.array([abs(r["view"]["line_yaw_off"] - r["view"]["policy_yaw_off"]) for r in rows])
    s["view"] = {"median_abs_yaw_gap_deg": float(np.median(yo)), "median_abs_yaw_sigmas": float(np.median(ys)),
                 "share_beyond_3_sigma": float(np.mean(ys > 3))}
    v = np.array([r["value"] for r in rows])
    s["value"] = {"first": float(v[0]), "min": float(v.min()), "max": float(v.max()), "last": float(v[-1])}
    return s
