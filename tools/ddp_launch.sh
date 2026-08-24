#!/bin/bash
# ddp_launch.sh — the only sanctioned way to start a multi-GPU run.
#
#   bash tools/ddp_launch.sh <nproc> <run-name> [train_fast.py args...]
#   e.g. bash tools/ddp_launch.sh 4 ddp_smoke --ckpt runs_ckpt.pt --steps 9e9
#
# Two things it exists to enforce (docs/ddp-plan.md step 4/14):
#   1. --warm-caches runs ONCE, single-process, before torchrun: four ranks
#      baking the 9-11 GB geodesic goal field concurrently is a guaranteed
#      OOM, and a torn cache npz read by another rank is silently wrong
#      vision for the entire run.
#   2. an explicit --run (the trainer refuses the per-process timestamp
#      default under torchrun anyway).
# Per-rank TORCHINDUCTOR_CACHE_DIR is set inside surfgym.distributed.init().
set -euo pipefail

NPROC="${1:?usage: ddp_launch.sh <nproc> <run-name> [args...]}"
RUN="${2:?usage: ddp_launch.sh <nproc> <run-name> [args...]}"
shift 2

# NOTE on NCCL_SHM_USE_CUDA_MEMCPY=1: on P2P-less consumer boards it made
# the isolated collective 4.9x faster (tools/bench_nccl.py: 7.08 -> 1.44 ms
# per 7.8 MB all-reduce) — and then DEADLOCKED the full trainer's very
# first broadcast on the same box (4x3090, EPYC 7502, torch 2.7.1/NCCL
# bundled). Do NOT default it on; export it yourself only after the C1
# invariants pass with it on your box.

# torchrun force-exports OMP_NUM_THREADS=1 to workers unless it is already
# set - which would silently cripple the C env step (the whole reason
# _default_omp_threads exists). Compute the per-rank team here instead.
#
# nproc is NOT the budget on a fractional rental. A vast box at gpu_frac
# 0.25 reports all 255 host cores through nproc/affinity while the CFS quota
# hands out 7.68 - and this line would then ask for 63 threads per rank on
# 7.68 CPUs. Measured cost of exactly that mistake: 21.7% (280,673 vs
# 341,697 steps/s). Read the quota, take the smaller. Same rule and same
# reason as train_fast._default_omp_threads; it is duplicated here because
# torchrun clamps the env before the trainer ever gets to look.
CORES=$(nproc)
QUOTA=""
if [ -r /sys/fs/cgroup/cpu.max ]; then                       # cgroup v2
  read -r Q P < /sys/fs/cgroup/cpu.max || true
  [ "${Q:-max}" != "max" ] && QUOTA=$(( Q / P ))
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then        # cgroup v1
  Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
  P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
  [ "$Q" -gt 0 ] && QUOTA=$(( Q / P ))
fi
if [ -n "$QUOTA" ] && [ "$QUOTA" -gt 0 ] && [ "$QUOTA" -lt "$CORES" ]; then
  echo "== cgroup CPU quota $QUOTA < nproc $CORES - sizing off the quota"
  CORES="$QUOTA"
fi
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  OMP_NUM_THREADS=$(( CORES / (2 * NPROC) ))
  [ "$OMP_NUM_THREADS" -lt 4 ] && OMP_NUM_THREADS=4
  [ "$OMP_NUM_THREADS" -gt 32 ] && OMP_NUM_THREADS=32
  export OMP_NUM_THREADS
fi
echo "== OMP_NUM_THREADS=$OMP_NUM_THREADS per rank ($CORES usable cores, "\
     "$NPROC ranks; the knee is ~8 threads/rank and the iteration gets "\
     "SLOWER past it)"

# numba sizes its parallel pool off HOST cpu_count - NOT the cgroup quota,
# NOT OMP_NUM_THREADS (the tbb layer ignores both). On a fractional rental
# that is a 255-thread pool PER RANK, and goalfield._FAST_SAMPLE
# (@njit(parallel=True)) is called once per SLOT per decision from
# mapfleet.reward - 107 pool synchronizations per decision on the full
# pool, on ~256-point batches (one point per thread, pure sync overhead).
# Measured 2026-08-24 on 4x3090 m16571 (nproc 255, quota 108.8): each rank
# held 294 threads (255 numba + 13 OMP + torch/NCCL), iteration 1 of the
# 107-map pool ran 10+ minutes of full-quota CPU spin at 0% GPU, and
# faulthandler stacks put the ranks INSIDE the rollout (core.step /
# goalfield.sample) - alive and crawling. This is what the "4-rank DDP
# deadlock" (128 AUTOTUNE lines, ~620% CPU, 0% GPU) actually was; it
# reproduced at 2 ranks with 13 OMP threads/rank on a CPU-adequate box,
# so it was never rank count and never CPU starvation.
if [ -z "${NUMBA_NUM_THREADS:-}" ]; then
  export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
fi
echo "== NUMBA_NUM_THREADS=$NUMBA_NUM_THREADS per rank"

cd "$(dirname "$0")/.."

echo "== warm caches (single process)"
python3 -u python/train_fast.py --warm-caches --run "${RUN}_warm" "$@"

echo "== torchrun x$NPROC : $RUN"
# --run-name, not --run: torchrun prefix-matches --run against its own
# --run-path even inside script args and dies on the ambiguity
exec torchrun --standalone --nproc-per-node="$NPROC" \
    python/train_fast.py --run-name "$RUN" "$@"
