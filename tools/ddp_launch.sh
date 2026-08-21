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

cd "$(dirname "$0")/.."

echo "== warm caches (single process)"
python3 -u python/train_fast.py --warm-caches --run "${RUN}_warm" "$@"

echo "== torchrun x$NPROC : $RUN"
exec torchrun --standalone --nproc-per-node="$NPROC" \
    python/train_fast.py --run "$RUN" "$@"
