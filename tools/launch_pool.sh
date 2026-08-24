#!/bin/bash
# launch_pool.sh - THE launcher for the full-pool multi-map DDP run.
# Runs ON the box. See docs/multimap-ddp-runbook.md for the whole
# rent-to-training sequence; this file is step 5 of it.
#
#   NMAPS=107 ENVS=131072 RUN=mmPOOL bash launch_pool.sh
#
# Every value below is the mmSMOKE config verbatim (runs/mmSMOKE/run.json),
# with exactly three deliberate changes, so nothing is hand-typed and no
# flag can go silently missing the way Round 17 lost two runs:
#   * --act-every 4 instead of 3   (user-set; measured 1.2403x throughput)
#   * --maps/--goal-cell from tools/pool_args.py, not the 5 smoke maps
#   * --envs and --steps scaled for the fleet and a multi-day budget
#
# --goal-cell comes from pool_args.py, which reads the cell off the
# .goal_<cell>.npz actually present next to each .bsp. Passing the wrong
# cell makes every map miss its cache and rebake at startup.
set -euo pipefail
cd /root/RL_Surf
export PYTHONPATH=python

RUN="${RUN:-mmPOOL}"
NMAPS="${NMAPS:-107}"
ENVS="${ENVS:-131072}"
STEPS="${STEPS:-500e9}"
# Eval cost is LINEAR in map count and sharded only world_size ways.
# Measured on the 107-map pool: 1 map/minute, i.e. 102 minutes per eval at
# --eval-eps 3. At --record-every 1.5e9 that is ~74% of wall-clock spent
# evaluating. --eval-eps 1 is not a compromise for a DETERMINISTIC greedy
# policy - CLAUDE.md: "MEAN tracks MAX is NOT corroboration ... for a
# deterministic greedy policy inside one mode it is automatic, and more
# eval episodes cannot fix it".
EVAL_EPS="${EVAL_EPS:-1}"
RECORD_EVERY="${RECORD_EVERY:-10e9}"
RANKS="${RANKS:-4}"

POOL=$(python3 tools/pool_args.py --limit "$NMAPS")
[ -z "$POOL" ] && { echo "!! pool_args produced nothing"; exit 1; }
echo "== $NMAPS maps, $ENVS envs global, $RANKS ranks -> $((ENVS/RANKS)) envs/rank"
echo "== $((ENVS/RANKS/NMAPS)) envs per (rank,map)"

bash tools/ddp_launch.sh "$RANKS" "$RUN" $POOL \
  --reward race --envs "$ENVS" --spawn platform \
  --lidar-w 64 --lidar-h 32 --lidar-cell 32 \
  --lidar-range 11500 --lidar-near 2000 \
  --emb 512 --hidden 448 \
  --act-every 4 --pitch-rate 1.33 --teleport-fail \
  --lr 3e-4 --gamma 0.9995 --gae 0.95 --clip 0.2 --vf 0.5 --ent 0.005 \
  --n-steps 32 --epochs 4 --minibatches 16 \
  --ep-ticks 6000 --time-pen 0.005 \
  --success-bonus 50 --finish-k 0 --stall-secs 15 \
  --race-dist geodesic --maxvel 4000 --train-stride 1 --yaw-adaptive \
  --respawn-frac 0.9 --respawn-margin 10 --respawn-reservoir 100000 \
  --int-coef 0.25 --int-view 8 --int-speed 3 \
  --steps "$STEPS" --ckpt-every 2e9 \
  --record-every "$RECORD_EVERY" --eval-eps "$EVAL_EPS" --eval-greedy-only \n  --timing --no-eval-at-start
