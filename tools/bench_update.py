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
        if not self.bf16_obs:
            return p(obs)
        scal, img = obs
        im = (img.reshape(-1, LIDAR_H, LIDAR_W, 1).permute(0, 3, 1, 2)
              if not self.nchw else img.reshape(-1, 1, LIDAR_H, LIDAR_W))
        f = torch.cat([scal[:, p.feat_idx], p.conv(im)], dim=1)
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
    ap.add_argument("--stages", action="store_true", default=True)
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}  "
          f"mb={args.mb} buf_rows={args.buf_rows}")

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
