#!/usr/bin/env python3
"""gru_probe.py - does a trained ``--rnn gru`` policy actually USE its memory?

The owner's question after xW1GRU / xW2GRU came out flat was two questions in
one: "did the GRU not help at all" and "are we sure of the implementation".
This answers the second half directly, on CPU, with no trainer and no rental.

Three conditions are recorded from the same checkpoint, the same spawn seed
and the same goal draw:

  a) ``carry``  the trained policy: h is carried across decisions and zeroed
                at every episode start (``_TorchPolicyBase._net``'s rule).
  b) ``zero``   h is forced to ZERO entering every decision - the tower still
                sees ``gru(f_t, 0)``, so the GRU module is present and its
                weights are used, but there is NO history. This is the
                ablation that isolates the RECURRENCE.
  c) ``lag``    the chain runs normally, but the tower is handed the PREVIOUS
                decision's GRU output while the reactive features are
                current - a one-decision shuffle that breaks the alignment
                between the memory and the observation without changing the
                magnitude or the distribution of what the tower reads.

and, inside ``carry``, the (b) action is ALSO computed at every state (a)
visits, so the two policies are compared on ONE state sequence rather than on
two that have already diverged. Per-head disagreement over that paired
sequence is the decisive number: a policy whose action never changes when its
memory is deleted is not using its memory, whatever the loss curve says.

    python tools/gru_probe.py runs/research/xW1GRU/ckpt_2000683008.pt \
        --episodes 3 --out-dir runs/research/gru_probe

Everything else - the depth render, the goal field, the spawn pool, the stall
rule, the config audit - is ``tools/record_ckpt.py``'s, driven in-process, so
there is no second recorder to drift out of sync (the argument that file
makes about itself at length).
"""
from __future__ import annotations

import os

# CPU ONLY. The local GPU belongs to the user; this must never take it. Set
# before torch is imported anywhere - a CUDA context is created on first use,
# and this is the only way to be sure there is nothing to create it on.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

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

# CUDA_VISIBLE_DEVICES="" does NOT make torch 2.11 report the GPU gone:
# torch.cuda.is_available() still returns True while device_count() is 0, so
# record_ckpt picks device="cuda" and nn.GRU's flatten_parameters dies in
# cudnn._init() with "min() arg is an empty sequence". Say it plainly instead.
torch.cuda.is_available = lambda: False
torch.backends.cudnn.enabled = False

import record_ckpt
import train_fast
from surfgym import goals as goals_mod
from surfgym.goals import MultiLine as _RealMultiLine
from surfgym.goals import resample_polyline_np

NVEC = train_fast.NVEC
HEADS = ("yaw", "pitch", "fwd", "side", "jump", "duck")

# set by main() before record_ckpt.main() runs; read by ProbeGreedy.__init__
MODE = "carry"
LIVE = {}                      # the constructed policy wrapper, for its stats


# --------------------------------------------------------------- the probe
class ProbeGreedy(train_fast.GreedyTorchPolicy):
    """GreedyTorchPolicy with the hidden state under our control.

    ``forward_split`` is inlined here (features -> gru_step -> heads) rather
    than called, because the three conditions differ only in WHICH state the
    towers are handed and that is not a parameter of ``Policy.forward``. The
    arithmetic is the same ops in the same order, so ``carry`` reproduces the
    stock recorder exactly.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.mode = MODE
        self._gprev = None            # previous decision's GRU output
        self.n_dec = 0
        self.dis = np.zeros(len(NVEC), np.int64)     # per-head disagreements
        self.dis_any = 0
        self.hnorm = []               # |h| per decision
        self.dlogit = []              # max |logit(a) - logit(b)| per decision
        self.identity = None          # inlined forward == Policy.forward?
        LIVE["pol"] = self

    def _net(self, x):
        pol = self.policy
        if pol.gru is None:
            return pol(x)
        n = x.shape[0]
        if self._h is None or self._h.shape[0] != n:
            self._h = torch.zeros(n, pol.rnn_size, device=x.device)
            self._gprev = torch.zeros_like(self._h)
            self._h_tick = None
        # episode starts, off the core's tick counter - the base class's rule,
        # unchanged (a respawn is a new episode and h must collapse)
        if self.core is not None:
            tick = np.asarray(self.core.states_view["tick"], np.int64)
            if self._h_tick is not None:
                started = tick < self._h_tick + self._period
                if started.any():
                    keep = torch.as_tensor(
                        np.ascontiguousarray(~started).astype(np.float32),
                        device=x.device).unsqueeze(1)
                    self._h = self._h * keep
                    self._gprev = self._gprev * keep
            self._h_tick = tick.copy()

        scal, img = x[:, :pol.scal_dim], x[:, pol.scal_dim:]
        f = pol.features(scal, img)
        zero = torch.zeros_like(self._h)
        h_in = zero if self.mode == "zero" else self._h
        h1 = pol.gru_step(f, h_in)
        if self.mode == "lag":
            g = self._gprev
        else:
            g = h1
        logits, value = pol.heads(f, scal, g)

        if self.identity is None and self.mode == "carry":
            # the inlined path IS Policy.forward - checked once, on a real
            # observation, so "carry" is provably the stock recorder and only
            # the ablations differ from it
            rl, rv, rh = pol(x, h_in)
            self.identity = {
                "logit_max_abs_diff": float((rl - logits).abs().max()),
                "value_max_abs_diff": float((rv - value).abs().max()),
                "h_max_abs_diff": float((rh - h1).abs().max())}

        if self.mode == "carry":
            # the (b) action AT THIS STATE: same f, same scalars, no history
            g0 = pol.gru_step(f, zero)
            lz, _ = pol.heads(f, scal, g0)
            a_a = self.packer.pad(logits).argmax(-1)[0]
            a_b = self.packer.pad(lz).argmax(-1)[0]
            d = (a_a != a_b).to(torch.int64).cpu().numpy()
            self.dis += d
            self.dis_any += int(d.any())
            self.dlogit.append(float((logits - lz).abs().max()))
        self.n_dec += 1
        self.hnorm.append(float(h1.norm(dim=1).mean()))
        self._gprev = h1
        self._h = h1
        return logits, value


# ------------------------------------------------------- route-goal patches
# The checkpoints trained with --goal-fixed, whose EVAL draws a goal uniformly
# from the fixed route set ahead of the start (goalsys.eval_hooks). record_ckpt
# does not mirror --goal-fixed: it samples a random reachable AIR point, which
# would send the agent off the route and make corridor progress meaningless.
# record_ckpt imports AirSampler / chord_line / MultiLine INSIDE main(), so
# rebinding them on surfgym.goals is enough to substitute the trainer's own
# eval goal without a second copy of the recorder.
class _FixedRouteSampler:
    """AirSampler's interface, goalsys._fixed_route_goal's answer."""

    def __init__(self, route_pts, spacing, radius, finish_center, finish_arc_r):
        self.route = np.asarray(route_pts, np.float64)
        seg = np.linalg.norm(np.diff(self.route, axis=0), axis=1)
        self.route_s = np.concatenate(([0.0], np.cumsum(seg)))
        self.route_len = float(self.route_s[-1])
        self.radius = float(radius)
        arcs = np.arange(spacing, self.route_len - self.radius, spacing)
        self.fixed_s = np.concatenate([arcs, [self.route_len]])
        self.finish_center = finish_center
        self.finish_arc_r = float(finish_arc_r)
        self.last_line = None
        self.picks = []

    def sample_near(self, n, origin, lo, hi, rng):
        o = np.asarray(origin, np.float64).reshape(3)
        d2 = ((self.route - o[None, :]) ** 2).sum(1)
        i0 = int(d2.argmin())
        s0 = float(self.route_s[i0])
        ahead = self.fixed_s[self.fixed_s >= s0 + 2.5 * self.radius]
        st = (float(ahead[int(rng.integers(len(ahead)))]) if len(ahead)
              else float(self.route_len))
        ig = int(np.searchsorted(self.route_s, st, side="right") - 1)
        ig = max(i0, min(ig, len(self.route) - 1))
        if ig + 1 < len(self.route) and self.route_s[ig + 1] > self.route_s[ig]:
            fr = ((st - self.route_s[ig])
                  / (self.route_s[ig + 1] - self.route_s[ig]))
            g = self.route[ig] + fr * (self.route[ig + 1] - self.route[ig])
        else:
            g = self.route[ig]
        if self.finish_center is not None and st >= self.route_len - self.radius:
            g = self.finish_center.copy()
        line = np.vstack([o[None, :], self.route[i0:ig + 1], g[None, :]])
        self.last_line = resample_polyline_np(line)
        self.picks.append({"arc": st, "arc_frac": st / self.route_len,
                           "goal": [float(v) for v in g],
                           "dist": float(np.linalg.norm(g - o))})
        return np.asarray([g], np.float64)

    def sample(self, n, rng):                       # the fallback path
        return self.sample_near(n, self.route[0], 0.0, 0.0, rng)


def install_route_goals(cfg, zones, radius):
    rp = cfg.get("goal_route")
    if not rp:
        return None
    p = Path(rp)
    if not p.is_absolute():
        p = ROOT / rp
    z = np.load(p)
    pts = np.asarray(z["route"], np.float64)
    mins = np.asarray(zones["end"]["mins"], np.float64)
    maxs = np.asarray(zones["end"]["maxs"], np.float64)
    smp = _FixedRouteSampler(
        pts, float(cfg.get("goal_fixed_spacing") or 2000.0), radius,
        0.5 * (mins + maxs), max(radius, 0.5 * float(np.max(maxs - mins))))

    def _air_sampler(*a, **kw):
        return smp

    def _chord(o, g, spacing=None):
        return smp.last_line

    def _multiline(n_envs, **kw):
        kw.setdefault("l_max", 4096)       # a full-route slice exceeds 768
        return _RealMultiLine(n_envs, **kw)

    goals_mod.AirSampler = _air_sampler
    goals_mod.chord_line = _chord
    goals_mod.MultiLine = _multiline
    return smp


# ------------------------------------------------------------------- driver
def gru_param_report(ck, cfg):
    """Sanity 3: the GRU tensors are there and the count is the formula."""
    sd = ck["policy"]
    keys = sorted(k for k in sd if k.startswith("gru."))
    n = int(sum(sd[k].numel() for k in keys))
    H = int(cfg.get("rnn_size") or 0)
    inw = int(sd["gru.weight_ih_l0"].shape[1]) if "gru.weight_ih_l0" in sd else 0
    return {"keys": keys, "shapes": {k: list(sd[k].shape) for k in keys},
            "n_params": n, "rnn_size": H, "gru_input_width": inw,
            "formula_3H_in_plus_H_plus_2": 3 * (H * (inw + H + 2)),
            "matches": n == 3 * (H * (inw + H + 2)),
            "l2_norm": {k: float(sd[k].float().norm()) for k in keys}}


def run_one(ckpt, mode, out, episodes, goal_seed, ep_ticks):
    global MODE
    MODE = mode
    LIVE.pop("pol", None)
    record_ckpt.GreedyTorchPolicy = ProbeGreedy
    argv = ["record_ckpt.py", str(ckpt), "--episodes", str(episodes),
            "--out", str(out), "--goal-seed", str(goal_seed)]
    if ep_ticks:
        argv += ["--ep-ticks", str(ep_ticks)]
    old = sys.argv
    sys.argv = argv
    t0 = time.time()
    try:
        record_ckpt.main()
    finally:
        sys.argv = old
    pol = LIVE.get("pol")
    st = {"mode": mode, "out": str(out), "wall_s": round(time.time() - t0, 1),
          "decisions": int(pol.n_dec) if pol else 0}
    if pol is not None and pol.n_dec:
        st["forward_identity"] = pol.identity
        st["h_norm_mean"] = round(float(np.mean(pol.hnorm)), 4)
        st["h_norm_max"] = round(float(np.max(pol.hnorm)), 4)
        if mode == "carry":
            st["disagree_any"] = int(pol.dis_any)
            st["disagree_frac"] = round(pol.dis_any / pol.n_dec, 6)
            st["per_head"] = {h: int(v) for h, v in zip(HEADS, pol.dis)}
            st["per_head_frac"] = {h: round(float(v) / pol.n_dec, 6)
                                   for h, v in zip(HEADS, pol.dis)}
            st["logit_gap_mean"] = round(float(np.mean(pol.dlogit)), 6)
            st["logit_gap_max"] = round(float(np.max(pol.dlogit)), 6)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--goal-seed", type=int, default=7)
    ap.add_argument("--ep-ticks", type=int, default=None)
    ap.add_argument("--modes", default="carry,zero,lag")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "research"
                                             / "gru_probe"))
    args = ap.parse_args()
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))

    from surfgym.zones import load_zones
    report = {"device": "cpu", "episodes": args.episodes,
              "goal_seed": args.goal_seed, "runs": []}
    for c in args.ckpt:
        c = Path(c)
        tag = c.parent.name
        ck = torch.load(c, map_location="cpu", weights_only=False)
        cfg = ck.get("config") or {}
        ent = {"ckpt": str(c), "tag": tag,
               "global_step": int(ck.get("global_step", 0)),
               "cfg_rnn": cfg.get("rnn"), "cfg_rnn_size": cfg.get("rnn_size"),
               "tower_depth": cfg.get("tower_depth"),
               "conv_mult": cfg.get("conv_mult"),
               "act_every": cfg.get("act_every"),
               "gru": gru_param_report(ck, cfg), "conditions": []}
        del ck
        zones = load_zones(str(ROOT / "maps"
                               / f"{cfg.get('map')}.bsp"))
        for mode in args.modes.split(","):
            mode = mode.strip()
            if not mode:
                continue
            smp = install_route_goals(cfg, zones,
                                      float(cfg.get("goal_radius") or 192.0))
            out = outd / f"{tag}_{mode}.jsonl"
            print(f"\n===== {tag} :: {mode} -> {out} =====", flush=True)
            st = run_one(c, mode, out, args.episodes, args.goal_seed,
                         args.ep_ticks)
            # the goal draw must be IDENTICAL across conditions or the three
            # recordings are three different tasks
            st["goals"] = list(smp.picks) if smp is not None else []
            ent["conditions"].append(st)
        report["runs"].append(ent)
        (outd / "probe.json").write_text(json.dumps(report, indent=2),
                                         encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
