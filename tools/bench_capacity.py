#!/usr/bin/env python3
"""bench_capacity.py — what does MORE capacity actually cost?

The perf pass established where the time goes (docs/perf-results.md): the PPO
update is ~71% of the iteration and is DRAM-bandwidth-bound on the conv
trunk's activations, while the whole MLP head is a rounding error. That has a
useful corollary — some ways of making the model bigger are nearly free and
some are brutal, and the difference is not intuitive.

This measures one PPO minibatch step (the same shapes and autocast the
trainer uses) across capacity axes, so "can I afford X?" is a number rather
than an argument:

  * network width   — emb / hidden / conv channels
  * network depth   — extra tower layer
  * depth image     — lidar resolution up AND down
  * frame stacking  — N depth frames as N conv input channels

    python3 tools/bench_capacity.py
    python3 tools/bench_capacity.py --variants base,hidden896,stack4

Note on frame stacking: it needs NO extra rollout storage. b_img already
holds all T timesteps, so a stack is a different gather, not a bigger buffer.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (N_SCALAR, HeadPacker, Policy,        # noqa: E402
                        logprob_entropy_padded)

# name -> (lidar_w, lidar_h, emb, hidden, conv_mult, frames, extra_tower)
VARIANTS = {
    "base":          (128, 64, 512, 448, 1, 1, 0),
    "hidden896":     (128, 64, 512, 896, 1, 1, 0),
    "hidden1792":    (128, 64, 512, 1792, 1, 1, 0),
    "emb1024":       (128, 64, 1024, 448, 1, 1, 0),
    "emb1024+h896":  (128, 64, 1024, 896, 1, 1, 0),
    "tower+1":       (128, 64, 512, 448, 1, 1, 1),
    "conv2x":        (128, 64, 512, 448, 2, 1, 0),
    "conv4x":        (128, 64, 512, 448, 4, 1, 0),
    "stack2":        (128, 64, 512, 448, 1, 2, 0),
    "stack4":        (128, 64, 512, 448, 1, 4, 0),
    "stack8":        (128, 64, 512, 448, 1, 8, 0),
    "lidar64x32":    (64, 32, 512, 448, 1, 1, 0),
    "lidar192x96":   (192, 96, 512, 448, 1, 1, 0),
    "lidar256x128":  (256, 128, 512, 448, 1, 1, 0),
}


def timed(fn, iters=8, warmup=4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def build(spec, mb, dev):
    lw, lh, emb, hidden, cmult, frames, extra = spec
    p = Policy(N_SCALAR + lw * lh, lw, lh, emb=emb, hidden=hidden).to(dev)
    if cmult != 1:
        c1, c2, c3 = 16 * cmult, 32 * cmult, 64 * cmult
        p.conv[0] = nn.Conv2d(frames, c1, 5, stride=2, padding=2)
        p.conv[2] = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        p.conv[4] = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        p.conv[8] = nn.Linear(c3 * 4 * 8, emb)
    elif frames != 1:
        # stacking widens ONLY conv1's input — every downstream activation,
        # which is what the update is bandwidth-bound on, is unchanged
        p.conv[0] = nn.Conv2d(frames, 16, 5, stride=2, padding=2)
    if extra:
        for tower in ("pi", "vf"):
            t = getattr(p, tower)
            setattr(p, tower, nn.Sequential(*list(t) + [
                nn.Linear(hidden, hidden), nn.Tanh()]))
    p = p.to(dev)
    p.conv = p.conv.to(memory_format=torch.channels_last)
    return p, lw, lh, frames


def make_step(spec, mb, buf_rows, dev):
    p, lw, lh, frames = build(spec, mb, dev)
    packer = HeadPacker(dev)
    opt = torch.optim.Adam(p.parameters(), lr=3e-4, eps=1e-5, fused=True)
    f_scal = torch.randn(buf_rows, N_SCALAR, device=dev)
    f_img = torch.rand(buf_rows, lw * lh, device=dev, dtype=torch.bfloat16)
    f_act = torch.stack([torch.randint(0, n, (buf_rows,), device=dev)
                         for n in (15, 7, 3, 3, 2, 2)], dim=1)
    f_logp = torch.randn(buf_rows, device=dev)
    f_adv = torch.randn(buf_rows, device=dev)
    f_ret = torch.randn(buf_rows, device=dev)
    idx = torch.randperm(buf_rows, device=dev)[:mb]
    # a stack gathers `frames` consecutive timesteps: same buffer, wider gather
    sidx = torch.stack([(idx - k * 2048).clamp_min(0) for k in range(frames)], 0)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def step():
        with amp:
            img = f_img[sidx].reshape(frames, mb, lh, lw) \
                             .permute(1, 0, 2, 3) if frames > 1 else \
                f_img[idx].reshape(-1, lh, lw, 1).permute(0, 3, 1, 2)
            f = torch.cat([f_scal[idx][:, p.feat_idx], p.conv(img)], dim=1)
            logits = p.action_head(p.pi(f))
            value = p.value_head(p.vf(f)).squeeze(-1)
            logp, ent = logprob_entropy_padded(packer.pad(logits.float()),
                                               f_act[idx])
            value = value.float()
        ratio = torch.exp(logp - f_logp[idx])
        a = f_adv[idx]
        a = (a - a.mean()) / (a.std() + 1e-8)
        pg = torch.max(-a * ratio, -a * torch.clamp(ratio, 0.8, 1.2)).mean()
        loss = pg + 0.5 * (0.5 * (value - f_ret[idx]).pow(2).mean()) \
            + 0.005 * (-ent.mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(p.parameters(), 0.5)
        opt.step()

    n_par = sum(x.numel() for x in p.parameters())
    return step, n_par


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=16384)
    ap.add_argument("--buf-rows", type=int, default=128 * 2048)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--iter-ms", type=float, default=2793.9,
                    help="whole-iteration ms, to convert a step delta into an "
                         "end-to-end one (64 minibatches per iteration)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"{torch.cuda.get_device_name(0)}  mb={args.mb}  "
          f"iteration reference {args.iter_ms:.0f} ms\n")
    print(f"{'variant':<15}{'params':>10}{'ms/mb':>9}{'x base':>8}"
          f"{'iter ms':>10}{'end-to-end':>12}")
    print("-" * 64)
    base_ms = None
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if name not in VARIANTS:
            print(f"{name:<15}   unknown")
            continue
        try:
            step, n_par = make_step(VARIANTS[name], args.mb, args.buf_rows, dev)
            ms = timed(step)
            if base_ms is None:
                base_ms = ms
            # the update is 64 minibatches; everything else in the iteration
            # is unchanged by a capacity knob on the update side
            d_iter = args.iter_ms + 64.0 * (ms - base_ms)
            print(f"{name:<15}{n_par / 1e6:>9.2f}M{ms:>9.2f}{ms / base_ms:>8.2f}"
                  f"{d_iter:>10.0f}{args.iter_ms / d_iter:>11.3f}x")
            del step
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"{name:<15}   FAILED  {type(exc).__name__}: {exc}")
    print("\nend-to-end column: iteration ms if ONLY the update changes "
          "(lidar/env/reward held fixed).\nA lidar-resolution variant also "
          "changes the render, so its true cost is worse than shown.")


if __name__ == "__main__":
    main()
