#!/usr/bin/env bash
# run_and_wait.sh - launch ONE arm and BLOCK until it is over.
#
# CLAUDE.md section 0: a run launched detached from the harness is invisible.
# Invoke this with the Bash tool's run_in_background, so the harness fires its
# completion notification when this script exits - for 100% of runs.
#
#   bash tools/run_and_wait.sh <worktree> <run> <map.bsp> [flags...]
set -u
W="${1:?worktree, e.g. C:/RL_Surf_n2}"; RUN="${2:?run name}"; MAP="${3:?map .bsp}"
shift 3
WB=$(echo "$W" | sed 's|^/\([a-zA-Z]\)/|\1:/|')          # /c/x -> c:/x
IDLE=${IDLE:-180}                                        # csv silence = over
LOG="$W/runs/launch_${RUN}.out"

echo "== $(date +%H:%M:%S) launching $RUN"
MSYS_NO_PATHCONV=1 powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
  "\$env:MAP='$MAP'; & '$WB/tools/launch_local.ps1' -Preset scratch_ablate -Arg1 $RUN -Arg2 '' $*" \
  > "$LOG" 2>&1
CSV="$W/runs/$RUN/progress.csv"

# the launcher returns once the trainer is up; wait for the csv to appear
for _ in $(seq 1 30); do [ -f "$CSV" ] && break; sleep 5; done
if [ ! -f "$CSV" ]; then
  echo "!! $RUN NEVER STARTED - no progress.csv. Log tail:"; tail -c 800 "$LOG" | tr -d '\000' | tail -10
  exit 1
fi

echo "== $RUN training"
last=0; quiet=0
while true; do
  sleep 20
  now=$(stat -c %Y "$CSV" 2>/dev/null || echo 0)
  if [ "$now" -eq "$last" ]; then quiet=$((quiet + 20)); else quiet=0; last=$now; fi
  [ "$quiet" -ge "$IDLE" ] && break
done

steps=$(tail -1 "$CSV" | cut -d, -f1); rew=$(tail -1 "$CSV" | cut -d, -f2)
echo "== $(date +%H:%M:%S) $RUN ENDED at $steps steps, ep_rew_mean $rew"
t=$(ls -1 "$W/runs/$RUN"/traj_*.jsonl 2>/dev/null | sort | tail -1)
if [ -n "$t" ]; then
  for ruler in wrroute fieldroute; do
    r="C:/RL_Surf/maps/surf_petrus_lite.${ruler}.npz"
    [ -f "$r" ] || continue
    echo "-- $ruler:"
    python "$W/tools/eval_honesty.py" --route "$r" --order-only 16 "$t" 2>/dev/null \
      | grep -iE "ORDER-ONLY|finishes" | tail -2
  done
fi
