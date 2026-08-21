"""distributed.py — facade over torch.distributed for the trainer's DDP path.

Design contract (docs/ddp-plan.md, step 1): at ``world_size == 1`` every
helper is a LITERAL no-op that returns its input unchanged and never touches
``torch.distributed``. That is what keeps the single-GPU path provably
identical and the Windows dev loop working (NCCL is Linux-only). The
multi-GPU path exists only under torchrun on the rented Linux boxes.

There is deliberately NO ``torch.nn.parallel.DistributedDataParallel``
anywhere in this repo — the trainer calls ``policy.forward_split`` which a
DDP wrapper does not intercept, so a wrapper would compute correct numbers
and silently never all-reduce (plan §1). The gradient path is a manual flat
all-reduce owned by train_fast.py.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta

import numpy as np
import torch

__all__ = ["Dist", "init"]


class Dist:
    """Process-group facade. ``enabled`` is False at world_size==1."""

    def __init__(self) -> None:
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_main = True
        self.enabled = False
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    # -- collectives (all identity at world_size==1) -------------------------
    def all_reduce_sum_(self, t: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            import torch.distributed as dist
            dist.all_reduce(t)
        return t

    def all_reduce_mean_(self, t: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            import torch.distributed as dist
            dist.all_reduce(t)
            t.div_(self.world_size)
        return t

    def all_reduce_min_scalar(self, v: int) -> int:
        if not self.enabled:
            return int(v)
        import torch.distributed as dist
        t = torch.tensor([int(v)], dtype=torch.int64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return int(t)

    def all_reduce_max_scalar(self, v: float) -> float:
        if not self.enabled:
            return float(v)
        import torch.distributed as dist
        t = torch.tensor([float(v)], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t)

    def broadcast_(self, t: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.enabled:
            import torch.distributed as dist
            dist.broadcast(t, src=src)
        return t

    def all_gather_var_bytes(self, data: bytes) -> list[bytes]:
        """Variable-length byte gather, rank order. NEVER a fixed pad: a
        correlated mass-truncation burst can overflow any cap and silently
        discard frontier states (plan §6 trap 11) — size-gather first, then
        pad to the observed max."""
        if not self.enabled:
            return [data]
        import torch.distributed as dist
        n = torch.tensor([len(data)], dtype=torch.int64, device=self.device)
        sizes = [torch.zeros(1, dtype=torch.int64, device=self.device)
                 for _ in range(self.world_size)]
        dist.all_gather(sizes, n)
        sizes = [int(s) for s in sizes]
        m = max(sizes)
        if m == 0:                      # rank-symmetric: sizes are global
            return [b"" for _ in range(self.world_size)]
        buf = torch.zeros(m, dtype=torch.uint8, device=self.device)
        if len(data):
            buf[:len(data)] = torch.from_numpy(
                np.frombuffer(data, np.uint8).copy()).to(self.device)
        outs = [torch.zeros(m, dtype=torch.uint8, device=self.device)
                for _ in range(self.world_size)]
        dist.all_gather(outs, buf)
        return [outs[r][:sizes[r]].cpu().numpy().tobytes()
                for r in range(self.world_size)]

    def barrier(self) -> None:
        if self.enabled:
            import torch.distributed as dist
            dist.barrier()

    @contextmanager
    def rank0_first(self):
        """Rank 0 runs the body while the others wait, then they follow —
        the in-process fallback for cache builds when --warm-caches was not
        run out of band."""
        if self.enabled and not self.is_main:
            self.barrier()
        yield
        if self.enabled and self.is_main:
            self.barrier()

    # -- invariant checks ----------------------------------------------------
    def assert_equal(self, tag: str, t: torch.Tensor) -> None:
        """Every rank must hold the exact same vector (f64 or i64)."""
        if not self.enabled:
            return
        import torch.distributed as dist
        lo, hi = t.clone(), t.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if not torch.equal(lo, hi):
            raise RuntimeError(
                f"[ddp] rank-divergent {tag}: min={lo.tolist()} "
                f"max={hi.tolist()} (rank {self.rank} holds {t.tolist()})")

    def assert_distinct(self, tag: str, value: int) -> None:
        """Every rank must hold a DIFFERENT value — catches the silent
        R-copies-of-the-same-fleet failure (plan §6 trap 1), where no logged
        number changes but the global batch is R duplicates."""
        if not self.enabled:
            return
        import torch.distributed as dist
        t = torch.tensor([int(value)], dtype=torch.int64, device=self.device)
        outs = [torch.zeros_like(t) for _ in range(self.world_size)]
        dist.all_gather(outs, t)
        vals = [int(o) for o in outs]
        if len(set(vals)) != len(vals):
            raise RuntimeError(
                f"[ddp] {tag} not rank-distinct: {vals} — rank streams "
                "collapsed (a reverted core.reset seed, a stray global "
                "manual_seed, or a 'reproducibility fix')")

    def finalize(self) -> None:
        if self.enabled:
            import torch.distributed as dist
            dist.barrier()
            dist.destroy_process_group()


def init() -> Dist:
    """Read the torchrun env; single process (no env) => disabled facade.

    ``torch.cuda.set_device(local_rank)`` runs FIRST so that every later
    bare ``cuda`` default (goalfield bake, graph capture, lidar) resolves to
    this rank's card instead of four processes piling onto cuda:0.
    """
    d = Dist()
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return d
    if os.name == "nt":
        raise SystemExit("DDP path is Linux-only (NCCL); run single-process "
                         "on Windows")
    import torch.distributed as dist
    d.rank = int(os.environ["RANK"])
    d.world_size = ws
    d.local_rank = int(os.environ.get("LOCAL_RANK", d.rank))
    torch.cuda.set_device(d.local_rank)
    d.device = torch.device("cuda", d.local_rank)
    # 45 min, not NCCL's 10: a cold goal-field bake is 10-30 min (DEPLOY.md)
    # and rank 0's eval/record stall can reach minutes on a surviving policy
    dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    d.is_main = d.rank == 0
    d.enabled = True
    # four ranks autotuning into one FileLock'd inductor cache either
    # serializes the compile 4x or races it (plan step 14)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                          f"/tmp/torchinductor_rank{d.rank}")
    return d
