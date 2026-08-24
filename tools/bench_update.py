#!/usr/bin/env python3
"""bench_update.py — decompose (and A/B) one PPO minibatch step.

The phase timer says the update is 71.5% of the iteration and 99.8% of that
is GPU-side (docs/perf-results.md), at ~17 effective TFLOPS on a card whose
bf16 peak is ~209. That gap is the whole software budget, so it needs a
breakdown before anything is rewritten.

This mirrors train_fast.py's minibatch step exactly — same shapes, same
autocast, same fused Adam, same grad clip — on random data, so a variant can
be tried in seconds instead of a 40-iteration training benchmark.

    python3 tools/bench_update.py                 # stages + variants
    python3 tools/bench_update.py --profile       # + torch.profiler op table
    python3 tools/bench_update.py --variants eager,nchw,bf16obs,compile

Variants:
  eager     what train_fast.py does today (channels_last trunk, S9)
  nchw      undo S9 — reproduces the pre-channels_last number
  bf16obs   S3: the 8192-pixel image slice stored bf16, scalars fp32
  avgpool   AvgPool2d(2) instead of the equivalent AdaptiveAvgPool2d((4,8))
  compile   S6: torch.compile(mode="reduce-overhead") on forward+loss
  compile_ml  same with mode="max-autotune-no-cudagraphs"
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (NACT, N_SCALAR, HeadPacker, Policy,   # noqa: E402
                        logprob_entropy_padded)

LIDAR_W, LIDAR_H = 128, 64
PIX = LIDAR_W * LIDAR_H
OBS = N_SCALAR + PIX


def _ev():
    return torch.cuda.Event(enable_timing=True)


def timed(fn, iters: int = 10, warmup: int = 5) -> float:
    """Median ms of `fn` on the GPU (each call individually event-bracketed)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = _ev(), _ev()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


class Step:
    """One PPO minibatch, split into callables so stages can be timed apart."""

    def __init__(self, mb: int, buf_rows: int, device, bf16_obs=False,
                 nchw=False, avgpool=False, ent_coef=0.005, vf_coef=0.5,
                 clip=0.2):
        self.mb, self.device = mb, device
        self.ent_coef, self.vf_coef, self.clip = ent_coef, vf_coef, clip
        self.policy = Policy(OBS, LIDAR_W, LIDAR_H, emb=512, hidden=448).to(device)
        if avgpool:
            # AdaptiveAvgPool2d((4,8)) on an (8,16) input IS AvgPool2d(2) —
            # same arithmetic, but the adaptive backward is an atomics kernel
            self.policy.conv[6] = nn.AvgPool2d(2)
        if nchw:
            # undo S9, to keep the pre-channels_last number reproducible
            self.policy.conv = self.policy.conv.to(
                memory_format=torch.contiguous_format)
        self.nchw = nchw
        self.packer = HeadPacker(device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=3e-4, eps=1e-5,
                                    fused=True)
        self.bf16_obs = bf16_obs
        if bf16_obs:
            self.f_scal = torch.randn(buf_rows, N_SCALAR, device=device)
            self.f_img = torch.rand(buf_rows, PIX, device=device,
                                    dtype=torch.bfloat16)
        else:
            self.f_obs = torch.rand(buf_rows, OBS, device=device)
        self.f_act = torch.stack([torch.randint(0, n, (buf_rows,), device=device)
                                  for n in (15, 7, 3, 3, 2, 2)], dim=1)
        self.f_logp = torch.randn(buf_rows, device=device)
        self.f_adv = torch.randn(buf_rows, device=device)
        self.f_ret = torch.randn(buf_rows, device=device)
        self.idx = torch.randperm(buf_rows, device=device)[:mb]
        self.amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self._loss = None

    # -- stages -------------------------------------------------------------
    def gather(self):
        if self.bf16_obs:
            return (self.f_scal[self.idx], self.f_img[self.idx])
        return self.f_obs[self.idx]

    def _policy_fwd(self, obs):
        """Production `Policy.forward` unless a variant needs otherwise.

        The unsplit path IS production, so it calls the module. The S3 split
        path mirrors it with two tensors: the image slice is then already
        contiguous, so the NHWC restride is a pure view+permute and conv has
        no layout copy to make at all — which is part of what S3 buys.
        """
        p = self.policy
        if not self.bf16_obs and not self.nchw:
            return p(obs)                     # the production path, verbatim
        if self.bf16_obs:
            scal, flat = obs
            sc = scal[:, p.feat_idx]
        else:
            flat, sc = obs[:, N_SCALAR:], obs[:, p.feat_idx]
        # `nchw` has to undo BOTH halves of S9 — the weight layout (done in
        # __init__) and this restride. Reverting only the weights leaves the
        # input declaring NHWC, and cudnn picks channels_last when EITHER side
        # suggests it, so the variant silently measured S9 again.
        im = (flat.reshape(-1, 1, LIDAR_H, LIDAR_W) if self.nchw
              else flat.reshape(-1, LIDAR_H, LIDAR_W, 1).permute(0, 3, 1, 2))
        f = torch.cat([sc, p.conv(im)], dim=1)
        return p.action_head(p.pi(f)), p.value_head(p.vf(f)).squeeze(-1)

    def fwd_loss(self, obs):
        with self.amp:
            logits, value = self._policy_fwd(obs)
            logp, ent = logprob_entropy_padded(
                self.packer.pad(logits.float()), self.f_act[self.idx])
            value = value.float()
        ratio = torch.exp(logp - self.f_logp[self.idx])
        a = self.f_adv[self.idx]
        a = (a - a.mean()) / (a.std() + 1e-8)
        pg = torch.max(-a * ratio,
                       -a * torch.clamp(ratio, 1 - self.clip, 1 + self.clip)).mean()
        vl = 0.5 * (value - self.f_ret[self.idx]).pow(2).mean()
        el = -ent.mean()
        return pg + self.vf_coef * vl + self.ent_coef * el

    def backward(self):
        self.opt.zero_grad(set_to_none=True)
        self._loss.backward()

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

    def opt_step(self):
        self.opt.step()

    def full(self):
        obs = self.gather()
        loss = self.fwd_loss(obs)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.opt.step()


def stage_table(st: Step) -> None:
    print("\n-- stages (ms, median of 10) --")
    obs = st.gather()
    t_gather = timed(st.gather)

    def fwd():
        st._loss = st.fwd_loss(obs)
    t_fwd = timed(fwd)
    fwd()
    t_bwd = timed(lambda: (st.backward(), fwd())[0]) - t_fwd
    t_clip = timed(st.clip_grads)
    t_opt = timed(st.opt_step)
    t_full = timed(st.full)
    named = [("gather", t_gather), ("forward+loss", t_fwd), ("backward", t_bwd),
             ("clip_grad_norm", t_clip), ("opt.step", t_opt)]
    for k, v in named:
        print(f"  {k:<16}{v:8.2f}  {100.0 * v / t_full:5.1f}%")
    print(f"  {'FULL STEP':<16}{t_full:8.2f}")
    print(f"  {'(sum of parts)':<16}{sum(v for _, v in named):8.2f}")


def conv_table(st: Step) -> None:
    """Layer-by-layer forward cost of the conv trunk."""
    print("\n-- conv trunk forward (ms, median of 10) --")
    B = st.mb
    x = torch.rand(B, 1, LIDAR_H, LIDAR_W, device=st.device)
    if not st.nchw:
        x = x.to(memory_format=torch.channels_last)
    with torch.no_grad(), st.amp:
        cur = x
        for i, layer in enumerate(st.policy.conv):
            frozen = cur
            t = timed(lambda l=layer, f=frozen: l(f))
            cur = layer(cur)
            print(f"  [{i}] {type(layer).__name__:<18}{t:7.2f}  "
                  f"-> {tuple(cur.shape)}")


def verify_compile(mb: int, buf_rows: int, dev, mode: str) -> None:
    """S6's gate: the compiled step must agree with eager on the SAME inputs.

    Not "the training curves look similar" — that hides a real drift behind
    PPO's own noise. Same weights, same minibatch, compare the loss terms and
    then every gradient after backward. Tolerance is bf16 epsilon-scale, not
    exact: inductor is allowed to reassociate, and the eager path is itself
    running bf16 convs.
    """
    torch.manual_seed(0)
    st = Step(mb, buf_rows, dev)
    obs = st.gather()
    eager = st.fwd_loss
    comp = torch.compile(eager, mode=mode)

    def grads(fn):
        st.opt.zero_grad(set_to_none=True)
        loss = fn(obs)
        loss.backward()
        g = {n: p.grad.detach().float().clone()
             for n, p in st.policy.named_parameters() if p.grad is not None}
        return float(loss.detach()), g

    l_e, g_e = grads(eager)
    l_c, g_c = grads(comp)
    print(f"\n-- compiled vs eager ({mode}) --")
    print(f"  loss  eager {l_e:.8f}  compiled {l_c:.8f}  "
          f"|d| {abs(l_e - l_c):.3e}")
    worst = ("", 0.0)
    for n in g_e:
        d = (g_e[n] - g_c[n]).abs().max().item()
        scale = max(1e-12, g_e[n].abs().max().item())
        if d / scale > worst[1]:
            worst = (n, d / scale)
    print(f"  worst relative grad drift: {worst[0]} {worst[1]:.3e}")
    ok = abs(l_e - l_c) <= 1e-4 * max(1.0, abs(l_e)) and worst[1] <= 5e-2
    print(f"  VERDICT: {'agrees within bf16 noise' if ok else 'DRIFT — do not ship'}")


def verify_adv(mb: int, buf_rows: int, dev) -> None:
    """DDP gate (docs/ddp-plan.md §5 tier B): the hoisted global-moment
    advantage estimator must reproduce the in-region per-minibatch one on
    identical inputs. The moments travel as f64 sum/sumsq with ddof=1 —
    exactly what train_fast.py all-reduces — so this catches the ddof slip
    (a 1.00003 scale factor invisible to learning, fatal to comparisons).
    """
    torch.manual_seed(0)
    st = Step(mb, buf_rows, dev)
    obs = st.gather()

    # -- the normalized advantage itself, both ways ------------------------
    a = st.f_adv[st.idx]
    a_inline = (a - a.mean()) / (a.std() + 1e-8)
    ad = a.double()
    s1, s2 = ad.sum(), ad.pow(2).sum()
    n_g = float(mb)
    m64 = s1 / n_g
    v64 = (s2 - s1 * m64) / (n_g - 1.0)
    a_mean, a_std = m64.float(), v64.clamp_min(0).sqrt().float()
    a_hoist = (a - a_mean) / (a_std + 1e-8)
    drift = ((a_inline - a_hoist).abs().max()
             / a_inline.abs().max().clamp_min(1e-12)).item()
    print(f"\n-- hoisted vs inline advantage normalisation --")
    print(f"  moments: mean |d| {abs(float(a_mean) - float(a.mean())):.3e}  "
          f"std |d| {abs(float(a_std) - float(a.std())):.3e}")
    print(f"  normalized-a worst relative drift: {drift:.3e}")

    # -- and through the full loss + every gradient ------------------------
    def loss_with(norm_a):
        with st.amp:
            logits, value = st._policy_fwd(obs)
            logp, ent = logprob_entropy_padded(
                st.packer.pad(logits.float()), st.f_act[st.idx])
            value = value.float()
        ratio = torch.exp(logp - st.f_logp[st.idx])
        pg = torch.max(-norm_a * ratio,
                       -norm_a * torch.clamp(ratio, 1 - st.clip,
                                             1 + st.clip)).mean()
        vl = 0.5 * (value - st.f_ret[st.idx]).pow(2).mean()
        el = -ent.mean()
        return pg + st.vf_coef * vl + st.ent_coef * el

    def grads(norm_a):
        st.opt.zero_grad(set_to_none=True)
        loss = loss_with(norm_a)
        loss.backward()
        g = {n: p.grad.detach().float().clone()
             for n, p in st.policy.named_parameters() if p.grad is not None}
        return float(loss.detach()), g

    l_i, g_i = grads(a_inline.detach())
    l_h, g_h = grads(a_hoist.detach())
    worst = 0.0
    for n in g_i:
        d = (g_i[n] - g_h[n]).abs().max().item()
        scale = max(1e-12, g_i[n].abs().max().item())
        worst = max(worst, d / scale)
    l_rel = abs(l_i - l_h) / max(1.0, abs(l_i))
    print(f"  loss  inline {l_i:.8f}  hoisted {l_h:.8f}  rel {l_rel:.3e}")
    print(f"  worst relative grad drift: {worst:.3e}")
    ok = drift <= 1e-5 and l_rel <= 1e-5 and worst <= 1e-3
    print(f"  VERDICT: {'agrees (fp32 rounding only)' if ok else 'DRIFT — do not ship'}")
    if not ok:
        raise SystemExit(1)


def profile(st: Step, rows: int = 22) -> None:
    from torch.profiler import ProfilerActivity, profile as tprofile
    for _ in range(5):
        st.full()
    torch.cuda.synchronize()
    with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            st.full()
        torch.cuda.synchronize()
    print("\n-- torch.profiler: top CUDA ops over 3 full steps --")
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=16384)     # T*N//minibatches
    ap.add_argument("--buf-rows", type=int, default=128 * 2048)
    ap.add_argument("--variants", default="eager,nchw,bf16obs,avgpool,compile,bf16obs+compile")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--verify", default=None,
                    metavar="MODE",
                    help="compare a torch.compile MODE against eager on "
                         "identical inputs (loss + every gradient) and exit")
    ap.add_argument("--verify-adv", action="store_true",
                    help="compare the DDP hoisted-moment advantage "
                         "normalisation against the in-region estimator on "
                         "identical inputs (docs/ddp-plan.md tier B) and exit")
    ap.add_argument("--stages", action="store_true", default=True)
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}  "
          f"mb={args.mb} buf_rows={args.buf_rows}")
    if args.verify:
        verify_compile(args.mb, args.buf_rows, dev, args.verify)
        return
    if args.verify_adv:
        verify_adv(args.mb, args.buf_rows, dev)
        return

    base = Step(args.mb, args.buf_rows, dev)
    if args.stages:
        stage_table(base)
        conv_table(base)
    if args.profile:
        profile(base)
    ref = timed(base.full)
    del base
    torch.cuda.empty_cache()

    print("\n-- variants (full step ms) --")
    print(f"  {'eager':<22}{ref:8.2f}  1.000x")
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if name == "eager":
            continue
        feats = set(name.split("+"))
        unknown = feats - {"bf16obs", "nchw", "avgpool", "compile",
                           "compile_ml", "eager"}
        if unknown:
            print(f"  {name:<22}   SKIPPED  unknown feature(s) {sorted(unknown)}")
            continue
        try:
            st = Step(args.mb, args.buf_rows, dev,
                      bf16_obs="bf16obs" in feats, nchw="nchw" in feats,
                      avgpool="avgpool" in feats)
            if feats & {"compile", "compile_ml"}:
                mode = ("max-autotune-no-cudagraphs" if "compile_ml" in feats
                        else "reduce-overhead")
                st.fwd_loss = torch.compile(st.fwd_loss, mode=mode)
            t = timed(st.full, iters=10, warmup=8)
            print(f"  {name:<22}{t:8.2f}  {ref / t:.3f}x")
            del st
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  {name:<22}   FAILED  {exc!r}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n({time.perf_counter() - t0:.0f}s)")
