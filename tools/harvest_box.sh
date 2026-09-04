#!/bin/bash
# harvest_box.sh - pull an arm's results off a rented box BEFORE destroying it.
#
#   bash tools/harvest_box.sh <port> <host> <arm> [<arm2> ...]
#
# Rentals delete their disks with them: no harvest = no results row, no
# rerun resume, no champion base, and the Go-Explore reservoir dies too
# (it lives inside the checkpoint). Per docs/research-plan.md, writing the
# docs/research-results.md row is a precondition for destroying a box.
#
# Pulls per arm, into runs/research/<arm>/ on this workstation:
#   progress.csv, run.json, the newest PERIODIC ckpt (never ckpt_latest -
#   torn-read hazard), and the last two trajectory recordings.
#
# tools/fleet_watchdog.py runs exactly this, unattended, 20 minutes before a
# box's deadline - so what this script pulls is what survives a wave. Three
# optional env vars extend it; with none of them set the behaviour is
# unchanged, byte for byte:
#
#   HARVEST_EXTRA="runs/x/expert_summary.jsonl,runs/x/notes.txt"
#       literal paths relative to /root/RL_Surf. Each is optional.
#   HARVEST_NEWEST="runs/x/round_*/train/ckpt_final.pt"
#       globs; each pulls its NEWEST single match. An expert loop writes one
#       checkpoint per round and only the last one is wanted (pulling all of
#       them is gigabytes).
#   HARVEST_ONLY_EXTRA=1
#       skip the standard per-arm pull; the arm names are then only
#       destination folders. An expert-loop box has no runs/<arm>/progress.csv
#       at all - its runs live in runs/<arm>/round_<n>/train/.
#
# Extras land in runs/research/<first arm>/extra/, keeping the path they had
# on the box. Extras are attempted even when an arm's own pull failed, and the
# exit code is non-zero if anything failed - the daemon retries on it.
set -uo pipefail          # NOT -e: a failed arm must not skip the extras

PORT="${1:?usage: harvest_box.sh <port> <host> <arm> [arm...]}"
HOST="${2:?usage: harvest_box.sh <port> <host> <arm> [arm...]}"
shift 2
[ "$#" -ge 1 ] || { echo "name at least one arm (runs/<arm> on the box)"; exit 1; }
FIRST="$1"
LOCAL_REPO="${LOCAL_REPO:-/c/RL_Surf}"
SSH="ssh -o BatchMode=yes"
FAIL=0

for ARM in "$@"; do
  DEST="$LOCAL_REPO/runs/research/$ARM"
  mkdir -p "$DEST"
  if [ "${HARVEST_ONLY_EXTRA:-0}" = "1" ]; then
    echo "== $ARM: HARVEST_ONLY_EXTRA=1, no standard pull"
    continue
  fi
  echo "== harvesting $ARM from $HOST:$PORT -> $DEST"
  # tar on the box side resolves the newest periodic ckpt + last two trajs;
  # missing trajs are tolerated, a missing progress.csv is not
  if $SSH -p "$PORT" "root@$HOST" "cd /root/RL_Surf/runs/$ARM && \
    tar cf - progress.csv run.json \
      \$(ls -t ckpt_[0-9]*.pt 2>/dev/null | head -1) \
      \$(ls -t traj_*.jsonl 2>/dev/null | head -2)" \
    | tar xf - -C "$DEST"; then
    ls -la "$DEST"
  else
    echo "!! harvest FAILED for $ARM ($HOST:$PORT)"
    FAIL=1
  fi
done

if [ -n "${HARVEST_EXTRA:-}${HARVEST_NEWEST:-}" ]; then
  EDEST="$LOCAL_REPO/runs/research/$FIRST/extra"
  mkdir -p "$EDEST"
  echo "== extras -> $EDEST"
  # the box resolves the list (literals must exist, globs take their newest
  # match) and tars it in one round trip; -v goes to stderr, the archive to
  # stdout, so the file names show up in the log.
  # `set -f` is load bearing: without it the SPLIT of the pattern list
  # expands the globs itself, and round_*/train/ckpt_final.pt arrives as one
  # word per round - i.e. every round's checkpoint, gigabytes, instead of the
  # newest one. It is re-enabled around each `ls`, which is what must glob.
  if $SSH -p "$PORT" "root@$HOST" "cd /root/RL_Surf && { set -f; \
      for f in \$(echo '${HARVEST_EXTRA:-}' | tr ',' ' '); do [ -e \"\$f\" ] && echo \"\$f\"; done; \
      for g in \$(echo '${HARVEST_NEWEST:-}' | tr ',' ' '); do set +f; ls -t \$g 2>/dev/null | head -1; set -f; done; \
    } | sort -u | tar cvf - -T -" | tar xf - -C "$EDEST"; then
    N=$(find "$EDEST" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "== extras: $N file(s) under $EDEST"
    [ "$N" = "0" ] && echo "!! extras matched NOTHING on the box - check the spec"
    ls -laR "$EDEST" | tail -20
  else
    echo "!! extras FAILED for $FIRST ($HOST:$PORT)"
    FAIL=1
  fi
fi

if [ "$FAIL" != "0" ]; then
  echo "!! harvest incomplete - do NOT destroy this box on this run's evidence"
  exit 1
fi
echo "done. Now write the docs/research-results.md row(s); only then destroy the box."
