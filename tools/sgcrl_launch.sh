#!/bin/bash
# sgcrl_launch.sh - start python/train_sgcrl.py (Single-Goal Contrastive RL)
# detached, ON THE BOX. Not a tools/run_arm.sh arm: SGCRL is a separate,
# reward-free trainer with its own networks - no PPO checkpoint lineage, no
# record gate, no train_fast flags - so it has its own few lines here.
#
#   bash tools/sgcrl_launch.sh <run> <map> [extra train_sgcrl.py flags ...]
#
#   bash tools/sgcrl_launch.sh sgLAB100 labyrinth_left100
#   bash tools/sgcrl_launch.sh sgEASY100 labyrinth_left100 --minutes 12 \
#       --eval-every 5e6 --goal-point 512,-1060,40      # TEST-ONLY easy goal
#
# <map> is a stem (looked up in maps/, then maps_pool/) or a path to a .bsp;
# its <stem>.zones.json must sit next to it (the finish box comes from it).
# Budget: --minutes 88 unless the flags carry --minutes or --steps; the
# trainer then runs a final eval, writes ckpt_latest.pt, stamps run.json
# "finished" and exits by itself. Give the watchdog deadline >= 95 min.
#
# Writes runs/<run>.pid (fleet_watchdog --pid-file) and the log
# runs/<run>_launch.txt. Harvest runs/<run>/ (progress.csv, run.json,
# ckpt_latest.pt, traj_*.jsonl, evals.jsonl, coverage.npz) and the log.
# Exits 1 unless the trainer is alive and learning within 90 s.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?usage: sgcrl_launch.sh <run> <map> [train_sgcrl.py flags ...]}"; shift
MAP="${1:?usage: sgcrl_launch.sh <run> <map> [train_sgcrl.py flags ...]}"; shift
PY="${PY:-python3}"

if [ -f "$MAP" ]; then BSP="$MAP"
elif [ -f "maps/$MAP.bsp" ]; then BSP="maps/$MAP.bsp"
elif [ -f "maps_pool/$MAP.bsp" ]; then BSP="maps_pool/$MAP.bsp"
else echo "!! map $MAP not found (tried maps/ and maps_pool/)"; exit 1; fi
ZJ="${BSP%.bsp}.zones.json"
if [ ! -f "$ZJ" ]; then echo "!! $ZJ missing - the finish box is read from it"; exit 1; fi
mkdir -p runs
if [ -e "runs/$RUN/progress.csv" ]; then
  echo "!! runs/$RUN already has a progress.csv - pick a new run name (or pass --resume)"
  case " $* " in *" --resume "*) ;; *) exit 1;; esac
fi

# the core's OpenMP pool (256 envs x 4 ticks is ~0.4 ms of simulation per
# decision; more threads buy nothing) and numba's pool (goalfield sampling
# sizes itself off the HOST nproc otherwise, 255 threads on a big rental)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

ARGS=(--map "$BSP" --run "$RUN")
case " $* " in *" --minutes "*|*" --steps "*) ;; *) ARGS+=(--minutes 88);; esac
LOG="runs/${RUN}_launch.txt"
echo "== $PY -u python/train_sgcrl.py ${ARGS[*]} $*"
# nohup + background (not setsid): $! is then the trainer itself
nohup "$PY" -u python/train_sgcrl.py "${ARGS[@]}" "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
echo "$PID" > "runs/${RUN}.pid"
echo "   pid $PID   log $LOG"

for _ in $(seq 18); do
  sleep 5
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "!! trainer exited during startup. Log tail:"
    tail -30 "$LOG"
    exit 1
  fi
  if grep -q "running eager" "$LOG"; then
    echo "!! CUDA graph capture failed - training eager (~6x fewer updates/s). Log tail:"
    tail -5 "$LOG"
  fi
  if [ "$(grep -c ' fps |' "$LOG")" -ge 2 ]; then
    echo "== alive and learning:"
    grep -E "^SGCRL|^g\* =|captured|^\[eval" "$LOG" | head -5
    tail -1 "$LOG"
    exit 0
  fi
done
echo "!! no learning heartbeat in 90 s. Log tail:"
tail -30 "$LOG"
exit 1
