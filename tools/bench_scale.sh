#!/bin/bash
# bench_scale.sh - ONE rung of the scaling ladder. Runs ON THE BOX.
#
#   bash tools/bench_scale.sh <ranks> <envs_per_rank> [n_steps] [iters] [extra...]
#
#   bash tools/bench_scale.sh 4 16384          # 4-rank DDP, 65,536 GLOBAL envs
#   bash tools/bench_scale.sh 1 32768 32 24    # single GPU, 24 iterations
#
# Why this exists (CLAUDE.md rule 4): the throughput of a box is only
# comparable across boxes if every rung carries the IDENTICAL argument set.
# The list below is run_arm.sh's SCRATCH branch verbatim - the user-pinned
# from-scratch baseline (cannonball, 64x32 depth, no --obs-reward) - with
# exactly three deliberate changes, all of them measurement-only:
#     --n-steps      swept (the learning frontier is T=32, not the 128 default)
#     --envs         swept; under DDP this is the GLOBAL fleet, split across
#                    ranks by train_fast.py:1407 - it is NOT per rank
#     --record-every disabled, so no eval lands inside a timed window
# Nothing else may differ, and nothing is hand-typed.
#
# OMP_NUM_THREADS is NEVER set here. ddp_launch.sh's nproc/(2*ranks) rule is
# load-bearing: sizing each rank to the single-rank OMP knee is a 1.4-1.6x
# LOSS under DDP (round 21, `env` blew up 9.9-13.2x).
set -euo pipefail

RANKS="${1:?usage: bench_scale.sh <ranks> <envs_per_rank> [n_steps] [iters]}"
NPR="${2:?usage: bench_scale.sh <ranks> <envs_per_rank> [n_steps] [iters]}"
T="${3:-32}"
ITERS="${4:-32}"
shift 4 2>/dev/null || shift $#
MB="${MB:-16}"
MAP="${MAP:-maps/surf_src_cannonball.bsp}"
ACT_EVERY=3

cd "$(dirname "$0")/.."
mkdir -p runs

ENVS=$(( RANKS * NPR ))
STEPS=$(( ITERS * ENVS * T * ACT_EVERY ))
TAG="r${RANKS}_n${NPR}_t${T}"
LOG="runs/bench_${TAG}.log"

ARGS=(--map "$MAP" --reward race --envs "$ENVS" --spawn platform
      --lidar-w 64 --lidar-h 32 --lidar-cell 32
      --lidar-range 11500 --lidar-near 2000
      --emb 512 --hidden 448
      --act-every 3 --pitch-rate 1.33 --teleport-fail
      --lr 3e-4 --gamma 0.9995 --gae 0.95 --clip 0.2 --vf 0.5 --ent 0.005
      --n-steps "$T" --epochs 4 --minibatches "$MB"
      --ep-ticks 12000 --time-pen 0.005
      --success-bonus 50 --finish-k 0 --stall-secs 15
      --race-dist geodesic --maxvel 4000 --train-stride 1 --yaw-adaptive
      --respawn-frac 0.9 --respawn-margin 10 --respawn-reservoir 100000
      --int-coef 0.25 --int-view 8 --int-speed 3
      --steps "$STEPS" --ckpt-every 1e12
      --record-every 1e12 --eval-eps 1 --eval-greedy-only
      --timing "$@")

echo "== rung $TAG : $RANKS rank(s) x $NPR envs = $ENVS GLOBAL, T=$T, mb=$MB"
echo "   $ITERS iterations = $STEPS steps   log $LOG"

# peak VRAM + peak host RSS, sampled while the rung runs. nvidia-smi is the
# honest number (torch's allocator reserves more than it reports as
# allocated, and the SDF/caches live outside the caching allocator).
SAMP="runs/bench_${TAG}.vram"
: > "$SAMP"
( while true; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk '{printf "V %s %s\n",$1,$2}' >> "$SAMP"
    ps -eo rss=,comm=,args= 2>/dev/null | grep train_fast | grep -v grep \
      | awk '{printf "R %s\n",$1}' >> "$SAMP"
    sleep 5
  done ) & SAMP_PID=$!
trap 'kill $SAMP_PID 2>/dev/null || true' EXIT

set +e
if [ "$RANKS" = "1" ]; then
  # no torchrun, so no forced OMP_NUM_THREADS=1; train_fast.py's own
  # _default_omp_threads (cores/2, capped 32) applies, which is the same
  # team ddp_launch.sh would compute for a single rank.
  python3 -u python/train_fast.py --run "bench_$TAG" "${ARGS[@]}" > "$LOG" 2>&1
else
  bash tools/ddp_launch.sh "$RANKS" "bench_$TAG" "${ARGS[@]}" > "$LOG" 2>&1
fi
RC=$?
set -e
kill $SAMP_PID 2>/dev/null || true

echo "== exit $RC"
if [ "$RC" != "0" ]; then
  echo "-- failure tail --"
  grep -iE "out of memory|OutOfMemory|CUDA error|Traceback|SystemExit|Error" "$LOG" | tail -8
  tail -5 "$LOG"
fi

python3 - "$LOG" "$SAMP" "$ENVS" "$T" "$ACT_EVERY" "$RANKS" "$NPR" <<'PY'
import sys, statistics as st
log, samp, envs, T, ae, ranks, npr = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:])
rows = []
for line in open(log, errors="replace"):
    if not line.startswith("TIMING "):
        continue
    d = {}
    for tok in line.split()[1:]:
        k, _, v = tok.partition("=")
        try:
            d[k] = float(v)
        except ValueError:
            pass
    rows.append(d)
print(f"RUNG ranks={ranks} envs_per_rank={npr} global_envs={envs} T={T} iters={len(rows)}")
if not rows:
    print("RESULT no TIMING lines - the rung did not run")
    raise SystemExit(0)
# drop compile warm-up: the last half of the iterations, minimum 4
keep = rows[len(rows) // 2:] if len(rows) >= 8 else rows[-4:]
def med(k):
    v = [r[k] for r in keep if k in r]
    return st.median(v) if v else 0.0
tot = med("total")
sps = envs * T * ae / (tot / 1000.0) if tot else 0.0
phases = ("total", "rollout_wall", "env", "lidar", "reward_py", "rollout_fwd",
          "sync_copy", "respawn", "book", "boot", "vis_cpu", "gae", "update",
          "update_gpu", "mb_gpu", "allreduce", "skew", "share", "misc")
print("PHASES " + " ".join(f"{k}={med(k):.1f}" for k in phases if med(k)))
print(f"RESULT steps_per_s={sps:,.0f} iter_ms={tot:.1f} "
      f"per_rank={sps / ranks:,.0f} "
      f"update_us_per_sample={med('update') * 1000 / (T * npr):.2f}")
vram, rss = {}, []
for line in open(samp, errors="replace"):
    p = line.split()
    if p[:1] == ["V"] and len(p) == 3:
        vram[p[1]] = max(vram.get(p[1], 0), int(p[2]))
    elif p[:1] == ["R"] and len(p) == 2:
        rss.append(int(p[1]))
if vram:
    print("VRAM_MiB_peak " + " ".join(f"gpu{k}={v}" for k, v in sorted(vram.items())))
if rss:
    print(f"RSS_GB_peak_per_proc={max(rss) / 1048576:.2f}")
PY
