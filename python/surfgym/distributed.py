"""distributed.py — facade over torch.distributed for the trainer's DDP path.

Design contract (docs/ddp-plan.md, step 1): at ``world_size == 1`` every
helper is a LITERAL no-op that returns its input unchanged and never touches
``torch.distributed``. That is what keeps the single-GPU path provably
identical and the Windows dev loop working (NCCL is Linux-only). The
multi-GPU path exists only under torchrun on the rented Linux boxes; without
CUDA (``CUDA_VISIBLE_DEVICES=-1``, the droplet) the same facade runs on gloo
over CPU tensors, which is how the 2-rank tests and smokes run there.

There is deliberately NO ``torch.nn.parallel.DistributedDataParallel``
anywhere in this repo — the trainer calls ``policy.forward_split`` which a
DDP wrapper does not intercept, so a wrapper would compute correct numbers
and silently never all-reduce (plan §1). The gradient path is a manual flat
all-reduce owned by train_fast.py.
"""
from __future__ import annotations

import io
import os
import pickle
from contextlib import contextmanager, nullcontext
from datetime import timedelta

import numpy as np
import torch

__all__ = ["Dist", "init", "RankPrefix", "shard_maps", "FlatGradSync"]


class Dist:
    """Process-group facade. ``enabled`` is False at world_size==1."""

    def __init__(self) -> None:
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_main = True
        self.enabled = False
        self.backend = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    # -- collectives (all identity at world_size==1) -------------------------
    def all_reduce_sum_(self, t: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            import torch.distributed as dist
            dist.all_reduce(t)
        return t

    def all_reduce_async(self, t: torch.Tensor):
        """Launch a SUM all-reduce without blocking the issuing stream's
        host thread; returns the work handle (None at world_size==1)."""
        if not self.enabled:
            return None
        import torch.distributed as dist
        return dist.all_reduce(t, async_op=True)

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

    def all_gather_object(self, obj) -> list:
        """Any picklable object from every rank, in rank order ([obj] at world_size==1).
        Built on all_gather_var_bytes, so the payload sizes may differ by rank (a rank
        with nothing to send passes None)."""
        if not self.enabled:
            return [obj]
        parts = self.all_gather_var_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        return [pickle.load(io.BytesIO(b)) for b in parts]

    def barrier(self) -> None:
        if self.enabled:
            import torch.distributed as dist
            dist.barrier()

    def rank0_first_unless(self, sharded: bool):
        """rank0_first() for a cache every rank touches; a no-op context when ``sharded``
        (--ddp-maps shard: each map lives on ONE rank, so its caches have one writer, and
        the ranks build different numbers of maps - a barrier per map would pair rank 0's
        k-th map with another rank's next collective)."""
        return nullcontext() if sharded else self.rank0_first()

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
    if torch.cuda.is_available():
        torch.cuda.set_device(d.local_rank)
        d.device = torch.device("cuda", d.local_rank)
        d.backend = "nccl"
    else:
        # no CUDA (the droplet, CUDA_VISIBLE_DEVICES=-1): gloo over CPU tensors - the same
        # collectives, so a 2-rank run of the real trainer is testable without a GPU
        d.device = torch.device("cpu")
        d.backend = "gloo"
    # 45 min, not NCCL's 10: a cold goal-field bake is 10-30 min (DEPLOY.md)
    # and rank 0's eval/record stall can reach minutes on a surviving policy
    dist.init_process_group(d.backend, timeout=timedelta(minutes=45))
    d.is_main = d.rank == 0
    d.enabled = True
    # four ranks autotuning into one FileLock'd inductor cache either
    # serializes the compile 4x or races it (plan step 14) — always
    # rank-suffix, a user-set base dir included
    base = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = (
        os.path.join(base, f"rank{d.rank}") if base
        else f"/tmp/torchinductor_rank{d.rank}")
    return d


def shard_maps(n_maps: int, world_size: int, rank: int) -> list:
    """--ddp-maps shard: the GLOBAL indices of the maps rank ``rank`` trains - round-robin over
    the --maps order (rank r gets maps r, r + W, r + 2W, ...). A run setting, the same rule on
    every run: it reads nothing about the maps."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} of world size {world_size}")
    return list(range(rank, int(n_maps), int(world_size)))


class RankPrefix(io.TextIOBase):
    """stdout for a non-main rank under --ddp-maps shard: every LINE goes to ``stream`` (the
    shared launch log) prefixed ``prefix``, written whole in one call, so two ranks' lines
    interleave but never tear. A partial line waits for its newline (or close)."""

    def __init__(self, stream, prefix: str):
        super().__init__()
        self._s = stream
        self._p = str(prefix)
        self._buf = ""

    def writable(self) -> bool:
        return True

    def write(self, text) -> int:
        self._buf += str(text)
        if "\n" in self._buf:
            *done, self._buf = self._buf.split("\n")
            self._s.write("".join(self._p + ln + "\n" for ln in done))
            self._s.flush()
        return len(text)

    def flush(self) -> None:
        self._s.flush()

    def close(self) -> None:
        if self._buf:
            self._s.write(self._p + self._buf + "\n")
            self._buf = ""
        self._s.flush()


class FlatGradSync:
    """The trainer's gradient all-reduce (docs/ddp-plan.md steps 9 + 15): bucketed flat SUM
    all-reduces whose views borrow each parameter's own strides (pure memcpys), optionally
    launched from post-accumulate-grad hooks the moment a bucket's last gradient lands
    (``register_hooks``: the wire time overlaps the rest of backward), then ``sync()`` pays
    only the exposed remainder and copies the reduced gradient back into every p.grad.

    ``weight`` None = the replicated design's mean over ranks (SUM, then x 1/world_size).
    ``weight`` w = SAMPLE-weighted (--ddp-maps shard): each rank scales its own gradient by
    w = its rows / the union's rows BEFORE the SUM, so ranks holding different row counts
    give exactly one process's gradient of the mean loss over the union of their rows."""

    def __init__(self, buckets, dist: Dist, device, weight=None):
        self.buckets = [list(b) for b in buckets]
        self.D = dist
        self.weight = None if weight is None else float(weight)
        self.inv_ws = 1.0 / dist.world_size
        self.flat, self.views = [], []
        for b in self.buckets:
            fl = torch.zeros(sum(p.numel() for p in b), device=device)
            vs, off = [], 0
            for p in b:
                vs.append(torch.as_strided(fl, p.shape, p.stride(), off))
                off += p.numel()
            self.flat.append(fl)
            self.views.append(vs)
        self.arrived = [0] * len(self.buckets)
        self.handles: list = [None] * len(self.buckets)

    def launch(self, bi: int) -> None:
        # RE-READ p.grad at launch: zero_grad(set_to_none=True) rebinds
        # every grad each minibatch - a cached list would sync stale
        # storage while opt.step() uses fresh tensors: divergent nets,
        # no error, caught only by the checksum
        torch._foreach_copy_(self.views[bi], [p.grad for p in self.buckets[bi]])
        if self.weight is not None:
            self.flat[bi].mul_(self.weight)
        self.handles[bi] = self.D.all_reduce_async(self.flat[bi])

    def register_hooks(self) -> None:
        """Registered AFTER any compile warm-up on purpose: the warm-up backward must not
        issue collectives (a rank whose inductor failed never reaches its backward, and the
        fleet would deadlock)."""
        def _mk_hook(bi):
            def _hook(_p):
                self.arrived[bi] += 1
                if self.arrived[bi] == len(self.buckets[bi]):
                    self.launch(bi)
            return _hook
        for bi, b in enumerate(self.buckets):
            for p in b:
                p.register_post_accumulate_grad_hook(_mk_hook(bi))

    def sync(self) -> None:
        for bi in range(len(self.buckets)):
            if self.handles[bi] is None:      # no hooks, or a starved one
                self.launch(bi)
            self.handles[bi].wait()           # a stream dependency, not a
            # host block: the scale and copy-back queue behind the wire
            if self.weight is None:
                self.flat[bi].mul_(self.inv_ws)
            torch._foreach_copy_([p.grad for p in self.buckets[bi]], self.views[bi])
            self.handles[bi] = None
            self.arrived[bi] = 0
