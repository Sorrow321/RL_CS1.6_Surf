#!/bin/bash
# run_arm.sh - THE launcher for every rented research arm. Runs ON THE BOX.
#
#   bash tools/run_arm.sh <run-name> [extra trainer flags ...]
#
#   bash tools/run_arm.sh xCTL                       # the baseline, verbatim
#   bash tools/run_arm.sh xROUTE --route maps/surf_src_cannonball.route.npz
#
# CLAUDE.md rule 4: all runs start from this script, an arm is a SMALL EDIT
# (here: extra flags appended). Nothing is hand-typed, because a hand-typed
# line that "worked" all session is missing half its flags the first time
# there is no checkpoint behind it - which is how Round 17 lost two runs.
#
# What it guarantees before anything trains:
#   * the base checkpoint is THE stuck checkpoint, by md5, not by filename;
#   * the config inside that checkpoint still matches the pinned baseline,
#     so an arm can never silently be measured against a different control;
#   * the run is alive with output afterwards, or this exits 1.
#
# Then it starts the trainer detached and prints the pid, so ssh dropping
# does not take the run with it.
set -euo pipefail

RUN="${1:?usage: run_arm.sh <run-name> [extra trainer flags ...]}"
shift || true

CKPT="${CKPT:-runs_ckpt.pt}"
# md5 of runs/sOBSR2/ckpt_latest.pt @ step 3,782,737,920 - the stuck agent:
# gets most of the way down the map, then fails for want of exploration.
EXPECT_MD5="${EXPECT_MD5:-1ba1fd2936af3ae1ad3608e3cd6b1e9e}"
# ~1 hour on a single 3090 (measured: 0.75-0.9e9 steps/h on this config,
# minus eval overhead). CLAUDE.md rule 2: one hour per ablation.
BUDGET="${BUDGET:-800000000}"
# 96 iterations at 2048x128x3 = 75,497,472 steps, ~5.7 min on a 3090. The
# baseline grid is 150,208,512; every second eval here lands within one
# iteration (0.5%) of a baseline point, and the finer cadence is what makes
# the "stationary for 10 minutes = failed" rule observable at all.
RECORD_EVERY="${RECORD_EVERY:-75e6}"
EVAL_EPS="${EVAL_EPS:-9}"

cd "$(dirname "$0")/.."

# MULTIMAP=<n_gpus>: the multi-map DDP arm. FROM SCRATCH, because none of
# these maps has ever been trained and the stuck checkpoint is a cannonball
# artifact - so, exactly like the SCRATCH branch below, the COMPLETE argument
# set has to live here. A resumed run restores --respawn-frac and --int-coef
# from its checkpoint; a scratch run restores nothing, which is how Round 17
# lost two runs to a hand-typed line that looked fine all session.
#
#   MAPS=maps/a.bsp,maps/b.bsp GOAL_CELLS=48,48 MULTIMAP=2 \
#       bash tools/run_arm.sh mmSMOKE
#
# MAPS and GOAL_CELLS come verbatim out of tools/stage_maps.py's manifest -
# it prints both. GOAL_CELLS is per map on purpose: 21 of the 110 usable maps
# TUNNEL at cell 48 and must keep 32, and a single global value either wastes
# 3.3x of bake and RAM on the 89 that are fine or ships a nonsense field for
# the 21 that are not.
#
# --envs is the GLOBAL fleet and is split TWICE - world_size, then maps - so
# it must be a multiple of n_gpus * n_maps. The trainer refuses otherwise
# rather than truncating a slot behind its own logs.
#
# MULTIMAP=1 is a SINGLE-GPU multi-map run: no torchrun, no warm-caches
# pre-pass (one process bakes any missing cache inline, exactly like the
# single-map trainer). Same argument set as the DDP branch.
#
# HELDOUT_MAPS=maps/x.bsp,maps/y.bsp HELDOUT_CELLS=48,48 adds EVAL-ONLY maps
# (--heldout-maps): evaluated at every eval, never trained on, their own
# race/heldout_*.<tag> columns and traj_<step>_<tag>.jsonl. That is the
# generalisation probe - a policy that learned surfing makes progress on a
# map it has never seen; a map-memoriser does not. HELDOUT_CELLS is the
# goal cell each held-out field was BAKED at (pool_args.py rule), one per map.
if [ -n "${MULTIMAP:-}" ]; then
  NGPU="$MULTIMAP"
  MAPS="${MAPS:?MULTIMAP needs MAPS=maps/a.bsp,maps/b.bsp (pool_args.py prints it)}"
  GOAL_CELLS="${GOAL_CELLS:?MULTIMAP needs GOAL_CELLS=48,48,... one per map}"
  NMAPS=$(awk -F, '{print NF}' <<<"$MAPS")
  ENVS="${ENVS:-16000}"
  if [ $(( ENVS % (NGPU * NMAPS) )) -ne 0 ]; then
    echo "!! --envs $ENVS is not a multiple of ${NGPU} ranks x ${NMAPS} maps"
    exit 1
  fi
  echo "== MULTIMAP: $NMAPS maps x $NGPU ranks, $ENVS global envs"
  echo "   $(( ENVS / NGPU )) envs/rank, $(( ENVS / NGPU / NMAPS )) envs/slot"
  HELD=()
  if [ -n "${HELDOUT_MAPS:-}" ]; then
    HELD=(--heldout-maps "$HELDOUT_MAPS")
    if [ -n "${HELDOUT_CELLS:-}" ]; then
      HELD+=(--heldout-goal-cell "$HELDOUT_CELLS")
    fi
    echo "   held-out (EVAL-ONLY, never trained): $HELDOUT_MAPS"
  fi
  mkdir -p runs
  LOG="runs/${RUN}_launch.txt"
  # Deviations from the pinned scratch baseline, all forced and all stated:
  #   --maps/--goal-cell         the point of the arm
  #   --act-every 4              the from-scratch baseline since 2026-08-24
  #                              (1.22x throughput for 0.92x sample
  #                              efficiency = ~1.16x per wall-clock hour).
  #                              gamma is NOT adjusted: it is per physics
  #                              tick and the trainer raises it to act_every
  #                              itself, so the 20 s horizon is unmoved.
  #   --n-steps 32               round 21's optimum, and large --envs forces
  #                              small T for VRAM anyway - the two agree.
  #                              CAVEAT: --n-steps counts DECISIONS, so at
  #                              act_every 4 that optimum is 128 physics
  #                              ticks of GAE window rather than the 96 it
  #                              was measured at. It does not transfer
  #                              unchanged and has not been re-measured here
  #   --ep-ticks 6000            these routes are 12.9k-39.6k u, a third of
  #                              cannonball's 198k; 12000 ticks would leave
  #                              most of an episode as a dead agent waiting
  #                              out the stall-kill. ep_ticks is still ONE
  #                              number for the whole fleet (mapfleet.py).
  #   no --obs-reward            same as the scratch baseline, and it drops
  #                              the known truncation-bootstrap bug with it
  ARGS=(--maps "$MAPS" --goal-cell "$GOAL_CELLS"
        --reward race --envs "$ENVS" --spawn platform
        --lidar-w 64 --lidar-h 32 --lidar-cell 32
        --lidar-range 11500 --lidar-near 2000
        --emb 512 --hidden 448
        --act-every 4 --pitch-rate 1.33 --teleport-fail
        --lr 3e-4 --gamma 0.9995 --gae 0.95 --clip 0.2 --vf 0.5 --ent 0.005
        --n-steps 32 --epochs 4 --minibatches 16
        --ep-ticks 6000 --time-pen 0.005
        --success-bonus 50 --finish-k 0 --stall-secs 15
        --race-dist geodesic --maxvel 4000 --train-stride 1 --yaw-adaptive
        --respawn-frac 0.9 --respawn-margin 10 --respawn-reservoir 100000
        --int-coef 0.25 --int-view 8 --int-speed 3
        --steps "${BUDGET_MM:-3e9}" --ckpt-every 1e9
        --record-every "${RECORD_EVERY_MM:-150e6}"
        --eval-eps "${EVAL_EPS_MM:-3}" --eval-greedy-only
        ${HELD[@]+"${HELD[@]}"} "$@")
  if [ "$NGPU" = "1" ]; then
    echo "== launch (single process: one GPU, no torchrun)"
    echo "   python3 -u python/train_fast.py --run $RUN ${ARGS[*]}"
    nohup python3 -u python/train_fast.py --run "$RUN" "${ARGS[@]}" \
        > "$LOG" 2>&1 < /dev/null &
  else
    echo "== launch (ddp_launch.sh warms the caches once, then torchrun x$NGPU)"
    echo "   bash tools/ddp_launch.sh $NGPU $RUN ${ARGS[*]}"
    nohup bash tools/ddp_launch.sh "$NGPU" "$RUN" "${ARGS[@]}" \
        > "$LOG" 2>&1 < /dev/null &
  fi
  PID=$!
  disown "$PID" 2>/dev/null || true
  echo "$PID" > "runs/${RUN}.pid"
  echo "   pid $PID   log $LOG"
  if [ "$NGPU" = "1" ]; then
    echo "== liveness (300s: the iteration-1 eval of every map runs before the first step line)"
  else
    echo "== liveness (300s: the warm-caches pass runs before torchrun)"
  fi
  for _ in $(seq 60); do
    sleep 5
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "!! launcher exited during startup. Log tail:"; tail -30 "$LOG"; exit 1
    fi
    if grep -qE "^step " "$LOG" 2>/dev/null; then break; fi
  done
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "!! not running. Log tail:"; tail -30 "$LOG"; exit 1
  fi
  echo "== ALIVE pid $PID"
  grep -E "^step |^race\[|^  slot |^heldout\[|AGGREGATE|HELD-OUT|warm-caches" "$LOG" | head -24 || true
  exit 0
fi

# SCRATCH=1: train FROM SCRATCH instead of resuming the stuck checkpoint
# (user-set baseline, 2026-08-23: cannonball, 64x32 depth, NO --obs-reward,
# from scratch, one hour per ablation).
#
# A scratch run restores NOTHING from a checkpoint, so the complete argument
# set has to be here. That is not defensive style, it is the exact failure
# that lost two runs in Round 17: a hand-typed line silently missing
# --respawn-frac and --int-coef looks fine on every resumed run and only
# breaks the first time there is no checkpoint behind it.
#
# The values are the pinned baseline above, MINUS --obs-reward. Deliberate
# deviations from that dict, both user-set: obs_reward is off, and the run
# starts from nothing. respawn_margin stays at the pinned 10.0 rather than
# the 2.0 that Round 18 found better, because these are hyperparameter
# ablations and the only thing that matters is that it is IDENTICAL across
# arms - changing it here would confound every comparison with a second
# treatment.
if [ "${SCRATCH:-0}" = "1" ]; then
  echo "== SCRATCH: training from nothing (no checkpoint, no md5 gate)"
  MAP="${MAP:-maps/surf_src_cannonball.bsp}"
  mkdir -p runs
  LOG="runs/${RUN}_launch.txt"
  ARGS=(--map "$MAP" --run "$RUN" --reward race --envs 2048 --spawn platform
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
        --steps "$BUDGET" --ckpt-every 1e9
        --record-every "$RECORD_EVERY" --eval-eps "$EVAL_EPS"
        --eval-greedy-only "$@")
  echo "== launch"
  echo "   python3 -u python/train_fast.py ${ARGS[*]}"
  echo "   budget $BUDGET steps from zero   log $LOG"
  nohup python3 -u python/train_fast.py "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
  PID=$!
  disown "$PID" 2>/dev/null || true
  echo "$PID" > "runs/${RUN}.pid"
  echo "   pid $PID"
  echo "== liveness (90s)"
  for _ in $(seq 18); do
    sleep 5
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "!! trainer exited during startup. Log tail:"; tail -25 "$LOG"; exit 1
    fi
    if grep -qE "^step " "$LOG" 2>/dev/null; then break; fi
  done
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "!! trainer is not running. Log tail:"; tail -25 "$LOG"; exit 1
  fi
  echo "== ALIVE pid $PID"
  grep -E "^step |reservoir d:|^race:" "$LOG" | head -6 || true
  exit 0
fi

# ARM_RESUME=1: continuing an arm's OWN checkpoint rather than starting from
# the stuck one. Both gates below exist to stop an arm being silently measured
# against the wrong control; neither applies to a continuation, whose ckpt has
# the arm's own md5 and the arm's own (deliberately changed) config. Loud, and
# it must be stated in the ledger entry.
if [ "${ARM_RESUME:-0}" = "1" ]; then
  echo "== ARM_RESUME: skipping the md5 and pinned-baseline gates"
  echo "   this is a CONTINUATION of $CKPT, not a fresh arm off the stuck ckpt"
  EXPECT_MD5=""
  SKIP_CFG_GUARD=1
fi

echo "== base checkpoint"
test -f "$CKPT" || { echo "!! no checkpoint at $CKPT"; exit 1; }
GOT=$(md5sum "$CKPT" | cut -d' ' -f1)
if [ -n "$EXPECT_MD5" ] && [ "$GOT" != "$EXPECT_MD5" ]; then
  echo "!! $CKPT md5 $GOT != the stuck checkpoint $EXPECT_MD5"
  echo "!! every arm is compared against a baseline measured from THAT one."
  echo "!! (set EXPECT_MD5= to override, and say so in the ledger)"
  exit 1
fi
echo "   $CKPT $GOT  (stuck checkpoint, verified)"

# the step handoff file, as a path BOTH this shell and the python it calls
# resolve: under Git Bash /tmp is NOT C:\tmp, so a Windows python writing
# '/tmp/ck_step' lands on a different file than `cat /tmp/ck_step` reads
# (found by tools/expert_loop.py, the first local user of this branch)
CKF="/tmp/ck_step"
if command -v cygpath >/dev/null 2>&1; then CKF_PY=$(cygpath -w "$CKF"); else CKF_PY="$CKF"; fi
if [ "${SKIP_CFG_GUARD:-0}" = "1" ]; then
  echo "== baseline config guard SKIPPED (ARM_RESUME)"
  python3 -c "import sys,torch;ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False);open(sys.argv[2],'w').write(str(int(ck['global_step'])));print('   ckpt step',int(ck['global_step']))" "$CKPT" "$CKF_PY"
else
echo "== baseline config guard"
python3 - "$CKPT" "$CKF_PY" <<'PY'
import sys, torch
# Pinned from runs/research/*/run.json - the config every arm inherits. A
# mismatch means the control curve in CLAUDE.md does not apply to this run.
WANT = {"reward": "race", "map": "surf_src_cannonball", "envs": 2048,
        "spawn": "platform", "lr": 3e-4, "lidar_w": 64, "lidar_h": 32,
        "emb": 512, "hidden": 448, "gps": False, "act_every": 3,
        "lidar_cell": 32.0, "time_pen": 0.005, "success_bonus": 50.0,
        "finish_k": 0.0, "train_stride": 1, "obs_reward": True,
        "stall_secs": 15.0, "race_dist": "geodesic", "int_coef": 0.25,
        "int_view": 8, "int_speed": 3, "maxvel": 4000.0,
        "yaw_adaptive": True, "respawn_frac": 0.9, "respawn_margin": 10.0,
        "ep_ticks": 12000, "gamma": 0.9995, "gae": 0.95, "clip": 0.2,
        "ent": 0.005, "vf": 0.5, "epochs": 4}
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = ck.get("config") or {}
bad = []
for k, v in WANT.items():
    got = cfg.get(k)
    if isinstance(v, float):
        ok = got is not None and abs(float(got) - v) < 1e-9
    else:
        ok = got == v
    if not ok:
        bad.append(f"{k}: ckpt={got!r} baseline={v!r}")
if bad:
    print("!! checkpoint config has drifted from the pinned baseline:")
    for b in bad:
        print("   " + b)
    raise SystemExit(1)
print(f"   config matches the pinned baseline; ckpt step {int(ck['global_step']):,}")
open(sys.argv[2], "w").write(str(int(ck["global_step"])))
PY

fi

CKSTEP=$(cat "$CKF")
# --steps is the ABSOLUTE resumed counter, not a budget
STOP=$((CKSTEP + BUDGET))

mkdir -p runs
LOG="runs/${RUN}_launch.txt"
ARGS=(--ckpt "$CKPT" --run "$RUN" --steps "$STOP"
      --record-every "$RECORD_EVERY" --eval-eps "$EVAL_EPS"
      --eval-greedy-only --ckpt-every 1e9 "$@")

echo "== launch"
echo "   python3 -u python/train_fast.py ${ARGS[*]}"
echo "   budget $BUDGET steps -> stop at $STOP   log $LOG"
# nohup + background, NOT setsid: with setsid $! is the setsid wrapper, which
# may or may not still exist a second later, and the liveness check below
# would be testing the wrong pid. nohup alone already survives the ssh
# session ending, which is the property that matters.
nohup python3 -u python/train_fast.py "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
echo "$PID" > "runs/${RUN}.pid"
echo "   pid $PID"

echo "== liveness (60s)"
for _ in $(seq 12); do
  sleep 5
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "!! trainer exited during startup. Log tail:"
    tail -25 "$LOG"
    exit 1
  fi
  if grep -q "resumed .* at step" "$LOG" 2>/dev/null; then
    break
  fi
done
if ! kill -0 "$PID" 2>/dev/null; then
  echo "!! trainer is not running. Log tail:"; tail -25 "$LOG"; exit 1
fi
echo "== ALIVE pid $PID"
# --act-hist/--obs-compass print the same kind of warm-start notice --route
# and --priv-critic do (train_fast.widen_for_obs): the arm's first eval is
# the checkpoint's own line only if that notice actually appeared, so it
# belongs in the launcher's first twelve lines like the rest.
grep -E "restored from checkpoint|^route |^obs aux:|--route:|--act-hist |--obs-compass |--priv-critic:|^race:|resumed |reservoir d:" "$LOG" | head -14 || true
echo "== dashboard: python3 tools/dashboard.py --port 8600"
