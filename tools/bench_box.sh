#!/bin/bash
# bench_box.sh <port> <host> [label] - what does a box cost us per step?
#
# The listing tells you the GPU model and the price; neither predicts our
# throughput. Capacity is irrelevant (the whole run fits in <5 GB), the
# workload is ~59% GPU-bound with a bf16-GEMM-heavy PPO update and a
# bandwidth-heavy triton sphere-trace, and the CPU side is throttled by
# whatever cgroup quota the host hands out. So measure the real thing:
# deploy, run a short --timing training probe, report steady-state
# steps/s and the per-phase split. Divide by $/h to compare rentals.
#
# Assumes tools/deploy_box.sh has already run against the box.
set -euo pipefail
PORT="${1:?usage: bench_box.sh <port> <host> [label]}"
HOST="${2:?usage: bench_box.sh <port> <host> [label]}"
LABEL="${3:-$HOST:$PORT}"
ITERS="${ITERS:-60}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=25"

echo "== $LABEL: card health"
$SSH -p "$PORT" "root@$HOST" "cd /root/RL_Surf && python3 tools/gpu_health.py --all 2>&1 | grep -E 'HBM copy|bf16 GEMM|VERDICT'"

echo "== $LABEL: env stepping (CPU side, thread sweep)"
$SSH -p "$PORT" "root@$HOST" "cd /root/RL_Surf && python3 tools/bench_env.py --envs 2048 --iters 200 --threads 8,16,32,64 2>&1 | tail -6"

echo "== $LABEL: full training probe ($ITERS iters from scratch)"
$SSH -p "$PORT" "root@$HOST" "cd /root/RL_Surf && rm -rf runs/bench runs/bench.log && timeout 900 python3 -u python/train_fast.py \
    --map maps/surf_src_cannonball.bsp --run bench --reward race --spawn platform \
    --respawn-frac 0.9 --respawn-speed 1.0 1.5 --maxvel 4000 \
    --lidar-w 64 --lidar-h 32 --lidar-range 11500 --lidar-near 2000 \
    --act-every 3 --pitch-rate 1.33 --gamma 0.9995 --int-coef 0.25 \
    --int-speed 3 --int-view 8 --record-every 1e12 --timing \
    --steps \$(( $ITERS * 786432 )) > runs/bench.log 2>&1; \
  python3 tools/perf_report.py runs/bench.log 2>&1 | tail -12; \
  grep -E '^done:' runs/bench.log"

cat <<'MSG'

Compare boxes on the "done: ... avg N steps/s" line divided by $/h.
Note the avg includes torch.compile warmup (~1-2 min), so it understates
steady state on short probes - use the same ITERS across boxes.
MSG
