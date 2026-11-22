#!/usr/bin/env python3
"""act_probe.py - the activations of a checkpoint's policy flying greedy episodes on a map.

The user (2026-11-22): "run the model on some of the maps ... and analyze ... the activations.
Maybe the activations explode." A weight audit of the seven-map agent found every layer 9-60x
its orthogonal init (no weight decay), one direction holding 62% of the image embedding's
energy, and 13 of the 448 final policy channels carrying 99% of what reaches the action head.
This records what those weights DO on real observations.

The policy, the core, the renderer and the greedy wrapper come from record_ckpt.build() - the
dashboard's own recording - and surfgym.record.record_rollout flies them, so the observations
are exactly the recorder's. Forward hooks on the policy's layers collect, per layer:
  rms / absmax / non-finite count of the output; for ReLU layers the units that never fire
  on any observation (dead) and the share of zeros; for the image embedding and the two final
  LayerNorms a reservoir of rows -> the share of variance on the top principal component and
  the participation ratio (effective dimension); for the residual blocks ||branch|| / ||input||;
  for the action head each USED key group's argmax probability (how sure the agent is) and the
  share of decisions above 0.99; for the view head the share of pre-tanh means past |2| and |3|
  (saturation); the value range.
The same captured observations then go through (a) the checkpoint's architecture at its
initialisation (the trainer's own construction, record_ckpt --cold-policy 0): the trained /
fresh activation ratio is the "explosion" factor on identical inputs; and (b) the trained policy
under bf16 autocast (the trainer's precision; the recorder flies fp32): how often the greedy
key choice and the view target change.

    python tools/act_probe.py runs/ckpt.pt --map maps/M.bsp --episodes 3 --out runs/act/M.json \\
        [--maxvel 4000] [--ema 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

KEEP_ROWS = 4096          # reservoir rows per PCA layer
KEEP_OBS = 2048           # observations kept for the fresh-weights and bf16 passes
GROUPS = (("fwd", 22, 26), ("side", 26, 30), ("jump", 30, 32), ("duck", 32, 35))


class Reservoir:
    def __init__(self, cap, rng):
        self.cap, self.rng, self.rows, self.seen = int(cap), rng, [], 0

    def add(self, a):
        for r in a:
            self.seen += 1
            if len(self.rows) < self.cap:
                self.rows.append(r.copy())
            else:
                j = int(self.rng.integers(0, self.seen))
                if j < self.cap:
                    self.rows[j] = r.copy()

    def array(self):
        return np.stack(self.rows) if self.rows else np.zeros((0, 0))


def pca_summary(x):
    if len(x) < 8:
        return {}
    x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x.astype(np.float64), compute_uv=False)
    v = s ** 2
    tot = float(v.sum()) or 1.0
    return {"top_pc_share": float(v[0] / tot), "top5_share": float(v[:5].sum() / tot),
            "participation_ratio": float(tot ** 2 / float((v ** 2).sum() or 1.0)),
            "dims_for_90pct": int(np.searchsorted(np.cumsum(v) / tot, 0.9) + 1)}


class Probe:
    """Forward hooks on a Policy; one instance per (policy, observation set)."""

    def __init__(self, policy, seed=0):
        self.rng = np.random.default_rng(seed)
        self.on = True
        self.s = {}
        self.res = {}
        self.hooks = []
        self.obs = Reservoir(KEEP_OBS, self.rng)
        mods = dict(policy.named_modules())
        for name, m in mods.items():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.ReLU)):
                self.hooks.append(m.register_forward_hook(self._make(name, m)))
            elif type(m).__name__ == "_SimbaBlock":
                self.hooks.append(m.register_forward_hook(self._make_block(name)))
        self.hooks.append(policy.register_forward_pre_hook(self._capture))

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _capture(self, mod, args):
        if self.on and args:
            self.obs.add(args[0].detach().float().cpu().numpy())

    def _st(self, name):
        return self.s.setdefault(name, {"n": 0, "sumsq": 0.0, "absmax": 0.0, "nonfinite": 0,
                                        "zeros": 0, "elems": 0, "chan_max": None})

    def _make(self, name, m):
        relu_like = isinstance(m, torch.nn.ReLU) or name.endswith("fc1")
        pca = name in ("conv.8", "pi.2", "vf.2")

        def hook(mod, inp, out):
            if not self.on or not torch.is_tensor(out):
                return
            a = out.detach().float()
            st = self._st(name)
            fin = torch.isfinite(a)
            st["nonfinite"] += int((~fin).sum())
            a = torch.where(fin, a, torch.zeros_like(a))
            st["n"] += int(a.shape[0])
            st["elems"] += int(a.numel())
            st["sumsq"] += float((a * a).sum())
            st["absmax"] = max(st["absmax"], float(a.abs().max()))
            if relu_like:
                r = torch.relu(a)            # fc1's output is pre-ReLU: its ReLU is inside forward
                st["zeros"] += int((r <= 0).sum())
                cm = (r.amax(dim=(0, 2, 3)) if r.dim() == 4 else r.amax(dim=0)).cpu().numpy()
                st["chan_max"] = cm if st["chan_max"] is None else np.maximum(st["chan_max"], cm)
            if pca:
                self.res.setdefault(name, Reservoir(KEEP_ROWS, self.rng)).add(a.cpu().numpy())
            if name == "action_head":
                g = st.setdefault("groups", {})
                for gn, lo, hi in GROUPS:
                    p = torch.softmax(a[:, lo:hi], dim=-1)
                    mx = p.max(-1).values
                    gg = g.setdefault(gn, {"n": 0, "pmax_sum": 0.0, "sure99": 0, "ent_sum": 0.0,
                                           "logit_absmax": 0.0})
                    gg["n"] += int(a.shape[0])
                    gg["pmax_sum"] += float(mx.sum())
                    gg["sure99"] += int((mx > 0.99).sum())
                    gg["ent_sum"] += float(-(p * torch.log(p.clamp_min(1e-12))).sum())
                    gg["logit_absmax"] = max(gg["logit_absmax"], float(a[:, lo:hi].abs().max()))
            if name == "view_head":
                st["sat2"] = st.get("sat2", 0) + int((a.abs() > 2).sum())
                st["sat3"] = st.get("sat3", 0) + int((a.abs() > 3).sum())
            if name == "value_head":
                st["vmin"] = min(st.get("vmin", np.inf), float(a.min()))
                st["vmax"] = max(st.get("vmax", -np.inf), float(a.max()))
        return hook

    def _make_block(self, name):
        def hook(mod, inp, out):
            if not self.on:
                return
            x = inp[0].detach().float()
            br = out.detach().float() - x
            st = self.s.setdefault(name + ".residual", {"n": 0, "ratio_sum": 0.0, "ratio_max": 0.0})
            r = br.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-12)
            st["n"] += int(r.numel())
            st["ratio_sum"] += float(r.sum())
            st["ratio_max"] = max(st["ratio_max"], float(r.max()))
        return hook

    def summary(self):
        out = {}
        for name, st in self.s.items():
            if name.endswith(".residual"):
                out[name] = {"mean_branch_over_input": st["ratio_sum"] / max(st["n"], 1),
                             "max_branch_over_input": st["ratio_max"]}
                continue
            r = {"rms": float(np.sqrt(st["sumsq"] / max(st["elems"], 1))), "absmax": st["absmax"],
                 "nonfinite": st["nonfinite"], "rows": st["n"]}
            if st["chan_max"] is not None:
                r["units"] = int(st["chan_max"].size)
                r["dead_units"] = int((st["chan_max"] <= 0).sum())
                r["zero_share"] = st["zeros"] / max(st["elems"], 1)
            for k in ("sat2", "sat3"):
                if k in st:
                    r[k + "_share"] = st[k] / max(st["elems"], 1)
            for k in ("vmin", "vmax"):
                if k in st:
                    r[k] = st[k]
            if "groups" in st:
                r["groups"] = {g: {"mean_pmax": v["pmax_sum"] / max(v["n"], 1),
                                   "share_sure99": v["sure99"] / max(v["n"], 1),
                                   "mean_entropy": v["ent_sum"] / max(v["n"], 1),
                                   "logit_absmax": v["logit_absmax"]} for g, v in st["groups"].items()}
            if name in self.res:
                r.update(pca_summary(self.res[name].array()))
            out[name] = r
        return out


def run_batches(policy, x, bs=512, autocast=False):
    outs = []
    dev = next(policy.parameters()).device
    with torch.inference_mode():
        for i in range(0, len(x), bs):
            xb = torch.as_tensor(x[i:i + bs], device=dev)
            if autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, v = policy(xb)
            else:
                lg, v = policy(xb)
            outs.append((lg.float().cpu().numpy(), v.float().cpu().numpy()))
    return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--map", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--maxvel", default=None)
    ap.add_argument("--ema", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import record_ckpt
    from surfgym.record import record_rollout

    extra = (["--maxvel", str(a.maxvel)] if a.maxvel else []) + (["--ema", str(a.ema)] if a.ema else [])
    b = record_ckpt.build([a.ckpt, "--map", a.map, "--episodes", str(a.episodes)] + extra)
    pol = b.policy
    pr = Probe(pol, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    traj = out.with_suffix(".traj.jsonl")
    ends = record_rollout(b.core, b.pol, traj, episodes=int(a.episodes),
                          max_ticks=int(a.episodes) * int(b.ep_ticks), seed=int(a.seed),
                          on_tick=b.on_tick, episode_meta=b.episode_meta, header_extra=b.header_extra)
    pr.on = False
    trained = pr.summary()
    x = pr.obs.array().astype(np.float32)
    pr.remove()

    # (a) the same observations through the architecture at initialisation
    cb = record_ckpt.build([a.ckpt, "--map", a.map, "--episodes", "1", "--cold-policy", "0"] + extra)
    cold = cb.policy
    pc = Probe(cold, a.seed)
    run_batches(cold, x)
    fresh = pc.summary()
    pc.remove()

    # (b) the trained policy under bf16 autocast on the same observations
    lg32, v32 = run_batches(pol, x)
    lg16, v16 = run_batches(pol, x, autocast=True)
    bf = {"rows": int(len(x))}
    for gn, lo, hi in GROUPS:
        bf[gn + "_argmax_agree"] = float(np.mean(lg32[:, lo:hi].argmax(1) == lg16[:, lo:hi].argmax(1)))
    nv = lg32.shape[1] - 35
    if nv > 0:
        bf["view_mu_absdiff_max"] = float(np.abs(lg32[:, 35:] - lg16[:, 35:]).max())
        bf["view_mu_absdiff_mean"] = float(np.abs(lg32[:, 35:] - lg16[:, 35:]).mean())
    bf["value_absdiff_mean"] = float(np.abs(v32 - v16).mean())
    bf["value_rms"] = float(np.sqrt(np.mean(v32 ** 2)))
    bf["logit_absdiff_max"] = float(np.abs(lg32[:, 22:35] - lg16[:, 22:35]).max())

    eps = []
    for e in ends or []:
        eps.append({k: e.get(k) for k in ("end", "ticks", "best_progress") if k in e})
    ratio = {}
    for k, r in trained.items():
        f = fresh.get(k, {})
        if "rms" in r and f.get("rms"):
            ratio[k] = r["rms"] / f["rms"]
    res = {"map": a.map, "ckpt": a.ckpt, "ema": a.ema, "episodes": eps, "obs_kept": int(len(x)),
           "trained": trained, "fresh": fresh, "rms_trained_over_fresh": ratio, "bf16": bf}
    out.write_text(json.dumps(res, indent=1, default=float))
    print("ACT_PROBE " + json.dumps({"map": Path(a.map).stem, "episodes": eps, "bf16": bf,
                                     "worst_ratio": max(ratio.items(), key=lambda kv: kv[1]) if ratio else None}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
