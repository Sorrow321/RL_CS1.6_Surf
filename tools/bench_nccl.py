#!/usr/bin/env python3
"""bench_nccl.py — step 0 of docs/ddp-plan.md: measure the collective FIRST.

The DDP projection has exactly one input that cannot be derived from the
repo: NCCL busbw between this box's GPUs (PCIe topology, ACS state, NUMA
hops all fold into it). This times ONE ITERATION's real gradient traffic —
64 all-reduces of the policy-sized flat tensor — plus the per-epoch moment
collective, and prints the decision-gate number.

    torchrun --standalone --nproc-per-node=4 tools/bench_nccl.py

Decision gate (plan step 0): < 40 ms/iter -> ship the exposed all-reduce;
40-80 ms -> ship, schedule the overlap hook; > 120 ms -> overlap first or
reconsider the rank count. Record the number in docs/perf-results.md.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def main() -> None:
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")

    # the real parameter count, so the number is the trainer's number
    from train_fast import N_SCALAR, Policy
    n_par = sum(p.numel() for p in
                Policy(N_SCALAR + 64 * 32, 64, 32).parameters())

    flat = torch.zeros(n_par, device=f"cuda:{local}")
    mom = torch.zeros(2, 16, dtype=torch.float64, device=f"cuda:{local}")

    for _ in range(10):                       # warm
        dist.all_reduce(flat)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(64):                       # one iteration's grad traffic
        dist.all_reduce(flat)
    torch.cuda.synchronize()
    grad_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for _ in range(4):                        # one iteration's moment traffic
        dist.all_reduce(mom)
    torch.cuda.synchronize()
    mom_ms = (time.perf_counter() - t0) * 1e3

    # algorithm-bandwidth -> bus-bandwidth for ring all-reduce
    bytes_moved = n_par * 4
    algbw = bytes_moved * 64 / (grad_ms / 1e3) / 1e9
    busbw = algbw * 2 * (ws - 1) / ws

    if rank == 0:
        gate = ("SHIP as written" if grad_ms < 40 else
                "ship + schedule overlap hook" if grad_ms < 80 else
                "overlap first / reconsider rank count" if grad_ms > 120
                else "ship; watch the allreduce field")
        print(f"NCCL {ws} ranks | model {n_par:,} params "
              f"({bytes_moved / 1e6:.1f} MB fp32)")
        print(f"64x grad all-reduce: {grad_ms:.1f} ms/iter "
              f"({grad_ms / 64:.2f} ms each)  "
              f"algbw {algbw:.1f} GB/s  busbw {busbw:.1f} GB/s")
        print(f"4x (2,16) f64 moment all-reduce: {mom_ms:.2f} ms/iter")
        print(f"GATE: {gate}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
