#!/bin/bash
# run_arm.sh - THE launcher for every rented research arm. Runs ON THE BOX.
#
#   bash tools/run_arm.sh <run-name> [extra trainer flags ...]
#
#   bash tools/run_arm.sh xCTL                       # the baseline, verbatim
#   bash tools/run_arm.sh xROUTE --route maps/surf_src_cannonball.route.npz
#
#   SCRATCH=1 BUDGET=40e9 bash tools/run_arm.sh xG1 \
#       --gamma 1.0 --respawn-margin 2 ...          # from RANDOM INIT
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
# SCRATCH=1 trains from a random init instead. There is then no checkpoint
# to verify and, more to the point, NOTHING IS RESTORED: every field a
# resumed run inherits silently has to be on the command line or the run is
# a different experiment that looks fine for an hour. So scratch mode does
# not accept a hand-written arg list either - it carries the COMPLETE
# baseline set below (SCRATCH_BASE), refuses to launch if a pinned field is
# missing from it AND from the caller's flags (REQ_FROM_CALLER), and after
# launch diffs the run's own run.json against the pinned baseline config so
# any remaining drift is on screen before the box has burned an hour.
#
# Then it starts the trainer detached and prints the pid, so ssh dropping
# does not take the run with it.
set -euo pipefail

RUN="${1:?usage: run_arm.sh <run-name> [extra trainer flags ...]}"
shift || true

SCRATCH="${SCRATCH:-0}"
CKPT="${CKPT:-runs_ckpt.pt}"
# md5 of runs/sOBSR2/ckpt_latest.pt @ step 3,782,737,920 - the stuck agent:
# gets most of the way down the map, then fails for want of exploration.
EXPECT_MD5="${EXPECT_MD5:-1ba1fd2936af3ae1ad3608e3cd6b1e9e}"
# ~1 hour on a single 3090 (measured: 0.75-0.9e9 steps/h on this config,
# minus eval overhead). CLAUDE.md rule 2: one hour per ablation.
# Resume: added to the checkpoint's step counter. Scratch: used as-is,
# because a scratch counter starts at 0.
BUDGET="${BUDGET:-800000000}"
# tagged archives; ckpt_latest.pt is rewritten every 60 s regardless, so
# this only sets how much history the box keeps (153 MB each)
CKPT_EVERY="${CKPT_EVERY:-1e9}"
# 96 iterations at 2048x128x3 = 75,497,472 steps, ~5.7 min on a 3090. The
# baseline grid is 150,208,512; every second eval here lands within one
# iteration (0.5%) of a baseline point, and the finer cadence is what makes
# the "stationary for 10 minutes = failed" rule observable at all.
RECORD_EVERY="${RECORD_EVERY:-75e6}"
EVAL_EPS="${EVAL_EPS:-9}"

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# The config every arm inherits, as flags. Taken field for field from the
# stuck checkpoint's own saved config (runs/sOBSR2/run.json == the "config"
# dict inside ckpt_latest.pt); a resumed run restores all of it and a
# scratch run restores NONE of it.
#
# Anything absent here falls back to train_fast.py's argparse defaults, and
# those are NOT the baseline: --int-coef defaults to 0 (curiosity off),
# --respawn-frac to 0 (no reservoir starts), --maxvel to 2000, --lidar-w/h
# to 128x64, --lidar-range to 2000, --gamma to 0.995. Round 17's two lost
# scratch runs flatlined at -time_pen for exactly the first two of those.
# --gps stays OFF, which for a store_true flag means it must NOT appear.
SCRATCH_BASE=(
  --map maps/surf_src_cannonball.bsp --reward race --envs 2048
  --spawn platform
  --lidar-w 64 --lidar-h 32 --lidar-cell 32
  --lidar-range 11500 --lidar-near 2000
  --emb 512 --hidden 448 --act-every 3 --pitch-rate 1.33 --teleport-fail
  --lr 3e-4 --gae 0.95 --clip 0.2 --vf 0.5 --ent 0.005 --epochs 4
  --ep-ticks 12000 --time-pen 0.005 --success-bonus 50 --finish-k 0
  --stall-secs 15 --race-dist geodesic --maxvel 4000 --train-stride 1
  --obs-reward --yaw-adaptive
  --respawn-frac 0.9 --respawn-reservoir 100000 --respawn-speed 1.0 1.5
  --int-coef 0.25 --int-view 8 --int-speed 3
)
# The two baseline-pinned fields deliberately NOT in SCRATCH_BASE, because
# in this round both are the thing under test: --gamma (0.9995 = a 20 s
# horizon) and --respawn-margin (CLAUDE.md: every start-state arm now runs
# on top of 2, not the 10 the checkpoint was trained at). Leaving them out
# of the base and demanding them from the caller means a scratch arm has to
# STATE its horizon and its harvest margin rather than inherit one silently.
REQ_FROM_CALLER=(--gamma --respawn-margin)

if [ "$SCRATCH" = "1" ]; then
  echo "== SCRATCH: random init, no checkpoint, nothing restored"
  for need in "${REQ_FROM_CALLER[@]}"; do
    seen=0
    for a in ${@+"$@"}; do
      case "$a" in "$need"|"$need"=*) seen=1 ;; esac
    done
    if [ "$seen" != 1 ]; then
      echo "!! scratch mode needs $need on the command line."
      echo "!! it is pinned by the baseline but NOT in SCRATCH_BASE, and a"
      echo "!! scratch run restores nothing - omitting it silently trains"
      echo "!! train_fast.py's argparse default instead (CLAUDE.md rule 4)."
      exit 1
    fi
  done
  echo "   complete baseline set + caller flags, verified present"
  # a scratch counter starts at 0, so the budget IS the absolute stop
  STOP="$BUDGET"
  BASE_ARGS=("${SCRATCH_BASE[@]}")
  # no "resumed ... at step" line will ever appear; run.json is written once
  # the map, goal field, env pool, vision and policy are all built, which is
  # the whole of startup risk
  LIVE_MARK="^pool("
else

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

CKSTEP=$(cat /tmp/ck_step)
# --steps is the ABSOLUTE resumed counter, not a budget
STOP=$((CKSTEP + BUDGET))
BASE_ARGS=(--ckpt "$CKPT")
LIVE_MARK="resumed .* at step"

fi

mkdir -p runs
LOG="runs/${RUN}_launch.txt"
ARGS=("${BASE_ARGS[@]}" --run "$RUN" --steps "$STOP"
      --record-every "$RECORD_EVERY" --eval-eps "$EVAL_EPS"
      --eval-greedy-only --ckpt-every "$CKPT_EVERY" ${@+"$@"})

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
  if grep -q "$LIVE_MARK" "$LOG" 2>/dev/null; then
    break
  fi
done
if ! kill -0 "$PID" 2>/dev/null; then
  echo "!! trainer is not running. Log tail:"; tail -25 "$LOG"; exit 1
fi
echo "== ALIVE pid $PID"
grep -E "restored from checkpoint|^route |--route:|^race:|resumed |reservoir d:" "$LOG" | head -12 || true

if [ "$SCRATCH" = "1" ]; then
  echo "== config drift (run.json vs the stuck checkpoint's own config)"
  # run.json is written once the policy is up. Waiting for it here is the
  # scratch equivalent of the resume path's md5 + config guard, except it
  # checks what the trainer ACTUALLY parsed rather than what we meant.
  for _ in $(seq 24); do
    [ -f "runs/${RUN}/run.json" ] && break
    sleep 5
  done
  python3 - "runs/${RUN}/run.json" <<'PY' || true
import json, sys
# The stuck checkpoint's saved config, verbatim, minus the three fields that
# are per-run by definition (trainer/steps/envs are asserted by the flags).
BASE = json.loads('''
{"act_every": 3, "bf16": true, "blend": null, "clip": 0.2, "compile":
true, "drop_max": 800.0, "drop_min": 400.0, "emb": 512, "ent": 0.005,
"ent_final": null, "ep_ticks": 12000, "epochs": 4, "eval_eps": 9,
"eval_greedy_only": true, "ez_eps": 0.0, "ez_max": 60, "ez_mu": 2.0,
"fail_pen": 0.0, "finish_k": 0.0, "finish_tref": 120.0, "fix_pitch":
null, "frame_stack": 0, "gae": 0.95, "gamma": 0.9995, "gps": false,
"graphs": true, "hidden": 448, "int_coef": 0.25, "int_speed": 3,
"int_view": 8, "lidar_cell": 32.0, "lidar_h": 32, "lidar_near": 2000.0,
"lidar_range": 11500.0, "lidar_w": 64, "lr": 0.0003, "map":
"surf_src_cannonball", "maxvel": 4000.0, "obs_reward": true, "pinhole":
0, "pitch_rate": 1.33, "punch_max": 400.0, "punch_min": 100.0,
"race_dist": "geodesic", "race_kill_aware": 0, "respawn_binned": 0,
"respawn_frac": 0.9, "respawn_margin": 10.0, "respawn_reservoir":
100000, "respawn_speed": [1.0, 1.5], "revisit_pen": null, "reward":
"race", "reward_per_decision": false, "rnd_coef": 0.0, "spawn":
"platform", "speed_coef": 0.0, "speed_equiv": 0.0, "stall_secs": 15.0,
"success_bonus": 50.0, "surf_mask": 0, "teleport_fail": true,
"time_pen": 0.005, "train_stride": 1, "vf": 0.5, "yaw_adaptive": true}
''')
cfg = json.load(open(sys.argv[1]))["config"]
drift, added = [], []
for k in sorted(set(BASE) | set(cfg)):
    if k not in BASE:
        added.append(f"{k}={cfg[k]!r}")       # field the ckpt predates
        continue
    a, b = BASE[k], cfg.get(k, "<absent>")
    same = (abs(a - b) < 1e-12 if isinstance(a, float)
            and isinstance(b, float) else a == b)
    if not same:
        drift.append(f"   {k}: baseline {a!r} -> this run {b!r}")
print("\n".join(drift) if drift else "   no field differs")
if added:
    print("   (fields the checkpoint predates, at their off/neutral "
          "defaults: " + ", ".join(added) + ")")
PY
fi
echo "== dashboard: python3 tools/dashboard.py --port 8600"
