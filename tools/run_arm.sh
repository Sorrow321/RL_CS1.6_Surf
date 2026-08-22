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

if [ "${SKIP_CFG_GUARD:-0}" = "1" ]; then
  echo "== baseline config guard SKIPPED (ARM_RESUME)"
  python3 -c "import sys,torch;ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False);open('/tmp/ck_step','w').write(str(int(ck['global_step'])));print('   ckpt step',int(ck['global_step']))" "$CKPT"
else
echo "== baseline config guard"
python3 - "$CKPT" <<'PY'
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
open("/tmp/ck_step", "w").write(str(int(ck["global_step"])))
PY

fi

CKSTEP=$(cat /tmp/ck_step)
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
grep -E "restored from checkpoint|^route |--route:|^race:|resumed |reservoir d:" "$LOG" | head -12 || true
echo "== dashboard: python3 tools/dashboard.py --port 8600"
