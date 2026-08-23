#!/bin/bash
# harvest_box.sh — pull an arm's results off a rented box BEFORE destroying it.
#
#   bash tools/harvest_box.sh <port> <host> <arm> [<arm2> ...]
#
# Rentals delete their disks with them: no harvest = no results row, no
# rerun resume, no champion base, and the Go-Explore reservoir dies too
# (it lives inside the checkpoint). Per docs/research-plan.md, writing the
# docs/research-results.md row is a precondition for destroying a box.
#
# Pulls per arm, into runs/research/<arm>/ on this workstation:
#   progress.csv, run.json, the newest PERIODIC ckpt (never ckpt_latest —
#   torn-read hazard), and the last two trajectory recordings.
set -euo pipefail

PORT="${1:?usage: harvest_box.sh <port> <host> <arm> [arm...]}"
HOST="${2:?usage: harvest_box.sh <port> <host> <arm> [arm...]}"
shift 2
[ "$#" -ge 1 ] || { echo "name at least one arm (runs/<arm> on the box)"; exit 1; }
LOCAL_REPO="${LOCAL_REPO:-/c/RL_Surf}"
# TRAJ_N: how many trajectory recordings to pull, newest first. Two is
# enough to eyeball an arm; a corridor-progress comparison across arms
# needs EVERY eval, because the frontier is a max over all of them and
# the last two evals alone hide a mid-run peak.
TRAJ_N="${TRAJ_N:-2}"
SSH="ssh -o BatchMode=yes"

for ARM in "$@"; do
  DEST="$LOCAL_REPO/runs/research/$ARM"
  mkdir -p "$DEST"
  echo "== harvesting $ARM from $HOST:$PORT -> $DEST"
  # tar on the box side resolves the newest periodic ckpt + last two trajs;
  # missing trajs are tolerated, a missing progress.csv is not
  $SSH -p "$PORT" "root@$HOST" "cd /root/RL_Surf/runs/$ARM && \
    tar cf - progress.csv run.json \
      \$(ls -t ckpt_[0-9]*.pt 2>/dev/null | head -1) \
      \$(ls -t traj_*.jsonl 2>/dev/null | head -$TRAJ_N)" \
    | tar xf - -C "$DEST"
  ls -la "$DEST"
done
echo "done. Now write the docs/research-results.md row(s); only then destroy the box."
