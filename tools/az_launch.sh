#!/bin/bash
# az_launch.sh - launch a TREE-ONLY run (azr1b4, slot 4 of the one-search series) on a box.
#
#   bash tools/az_launch.sh <run> [extra beam_tas flags ...]
#
# 1. THE STEP-0 NETWORK: tools/train_fast.py with run_arm.sh's SCRATCH preset (the view, keys and
#    potential defaults) + run 5's exact flag list (autoresearch/config.json incumbent_recipe) and
#    --steps 0 -> runs/<run>_init/ckpt_final.pt: run 5's architecture and observation pipeline
#    with untrained weights (the trainer's own init; no rollout, no gradient step). Skipped when
#    that file exists (a crash-resume relaunches with the same command).
# 2. THE LOOP: tools/beam_tas.py <init> --mcts ... --az-loop --out-dir runs/<run> under nohup; the
#    search's flags are run 5's (auto_spine's beam_tas call + --auto-spine-mcts-args), the az
#    flags the loop's own (tools/az_loop.py). The full command is the first line of
#    runs/<run>_launch.txt; the pid goes to runs/<run>.pid. When runs/<run>/ckpt_latest.pt exists
#    the loop RESUMES from it (its round, its tick count, its labels): the same run.
set -euo pipefail
RUN="${1:?usage: az_launch.sh <run> [extra beam_tas flags]}"
shift || true
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
MAP="${MAP:-maps/surf_src_sidistic.bsp}"
mkdir -p runs
INIT="runs/${RUN}_init/ckpt_final.pt"
if [ ! -f "$INIT" ]; then
  mapfile -t RECIPE < <($PY -c "import json; [print(x) for x in json.load(open('autoresearch/config.json'))['incumbent_recipe']]")
  IARGS=(--map "$MAP" --run "${RUN}_init" --reward race --envs 2048 --spawn platform
         --lidar-w 64 --lidar-h 32 --lidar-cell 32
         --lidar-range 11500 --lidar-near 2000
         --emb 512 --hidden 448
         --act-every 4 --pitch-rate 1.33 --teleport-fail
         --lr 3e-4 --gamma 0.9995 --gae 0.95 --clip 0.2 --vf 0.5 --ent 0.005
         --n-steps 128 --epochs 4 --minibatches 16
         --ep-ticks 12000 --time-pen 0.005
         --success-bonus 50 --finish-k 0 --stall-secs 15
         --race-dist geodesic --maxvel 4000 --train-stride 1 --yaw-adaptive
         --respawn-frac 0.9 --respawn-margin 10 --respawn-reservoir 100000
         --int-coef 0.25 --int-view 8 --int-speed 3
         --steps 0 --ckpt-every 1e9 --record-every 75e6 --eval-eps 9 --eval-greedy-only
         --view-continuous --view-absolute velocity --keys-hold
         --obs-potential norm --obs-potential-curtain "${RECIPE[@]}")
  echo "== init: $PY -u python/train_fast.py ${IARGS[*]}" | tee "runs/${RUN}_init_launch.txt"
  $PY -u python/train_fast.py "${IARGS[@]}" >> "runs/${RUN}_init_launch.txt" 2>&1
  test -f "$INIT" || { echo "!! no $INIT"; tail -30 "runs/${RUN}_init_launch.txt"; exit 1; }
  grep -E "^done:" "runs/${RUN}_init_launch.txt" || true
fi
ARGS=(-u tools/beam_tas.py "$INIT" --map "$MAP" --envs 2048 --resample-every 25 --score d
      --greedy-envs 32 --allow-nonfinisher --skip-gate --route-file none --max-ticks 30000
      --torch-seed 0 --log-every 5 --mcts-log-every 50 --out-dir "runs/$RUN"
      --mcts --mcts-minutes 6 --mcts-random-frac 1.0 --mcts-perturb-frac 0
      --az-loop --az-run "$RUN" "$@")
LOG="runs/${RUN}_launch.txt"
{ echo "== az launch $(date -u +%Y-%m-%dT%H:%M:%SZ) commit $(git rev-parse HEAD)";
  echo "== cmd: $PY ${ARGS[*]}"; } >> "$LOG"
nohup $PY "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
echo "$PID" > "runs/${RUN}.pid"
echo "== pid $PID, log $LOG"
sleep 60
if ! kill -0 "$PID" 2>/dev/null; then
  echo "!! the loop exited during startup:"; tail -30 "$LOG"; exit 1
fi
echo "== ALIVE"; tail -5 "$LOG"
