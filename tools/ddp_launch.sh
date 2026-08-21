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

# Consumer boards (GeForce driver) report P2P "Chipset Not Supported", so
# NCCL stages through host SHM; the CE-driven copy path is 4.9x faster
# there (measured on a 4x3090 EPYC 7502 box with tools/bench_nccl.py:
# 7.08 -> 1.44 ms per 7.8 MB all-reduce, busbw 1.7 -> 8.1 GB/s). Harmless
# where P2P/NVLink exists — the SHM transport is not used there.
export NCCL_SHM_USE_CUDA_MEMCPY="${NCCL_SHM_USE_CUDA_MEMCPY:-1}"

# torchrun force-exports OMP_NUM_THREADS=1 to workers unless it is already
# set — which would silently cripple the C env step (the whole reason
# _default_omp_threads exists). Compute the per-rank team here instead.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  OMP_NUM_THREADS=$(( $(nproc) / (2 * NPROC) ))
  [ "$OMP_NUM_THREADS" -lt 4 ] && OMP_NUM_THREADS=4
  [ "$OMP_NUM_THREADS" -gt 32 ] && OMP_NUM_THREADS=32
  export OMP_NUM_THREADS
fi
echo "== OMP_NUM_THREADS=$OMP_NUM_THREADS per rank"

cd "$(dirname "$0")/.."

echo "== warm caches (single process)"
python3 -u python/train_fast.py --warm-caches --run "${RUN}_warm" "$@"

echo "== torchrun x$NPROC : $RUN"
# --run-name, not --run: torchrun prefix-matches --run against its own
# --run-path even inside script args and dies on the ambiguity
exec torchrun --standalone --nproc-per-node="$NPROC" \
    python/train_fast.py --run-name "$RUN" "$@"
