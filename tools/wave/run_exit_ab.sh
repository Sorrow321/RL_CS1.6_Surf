#!/bin/bash
# run_exit_ab.sh - put ONE arm of the expert-iteration A/B on a box that is
# already deployed (baseline checked out, deps installed, caches shipped and
# the bsp mtime pinned - tools/wave/box_finish.sh or rent_expert_box6.sh).
#
#   bash run_exit_ab.sh <port> <host> <instance> <name> <hours> <ckpt> [trainer flags...]
#
#   bash tools/wave/run_exit_ab.sh 41234 ssh5.vast.ai 48512345 exitCAT 6 \
#       /c/RL_Surf_base/runs/research/xENT131/ckpt_10774118400.pt \
#       --bc-target argmax --bc-value-coef 0
#   bash tools/wave/run_exit_ab.sh 41235 ssh6.vast.ai 48512346 exitTPT 6 \
#       /c/RL_Surf_base/runs/research/xENT131/ckpt_10774118400.pt \
#       --bc-target dist --bc-value-coef 0.25
#
# Everything after <ckpt> is passed to the TRAINER verbatim, through
# expert_loop's --train-extra (argparse.REMAINDER, so it has to be last on
# the line - this script puts it last). The two arms above are the A/B:
# same seed, same planner, same BC file (tools/plan_to_bc.py writes both
# search-derived targets by default), only the trainer's target differs.
#
# EXTRA_LOOP_FLAGS carries flags for expert_loop ITSELF, which --train-extra
# cannot (REMAINDER eats the rest of the line). --dagger-k is one:
#
#   EXTRA_LOOP_FLAGS='--dagger-k 600' bash tools/wave/run_exit_ab.sh \
#       41236 ssh7.vast.ai 48512347 exitDAG 6 \
#       /c/RL_Surf_base/runs/research/exitTPT/extra/runs/exitTPT/round_1/train/ckpt_final.pt \
#       --bc-target dist --bc-value-coef 0.25
#
# What it does, in order:
#   1. ship the checkpoint to $BOXROOT/RL_Surf/runs_seed.pt and verify its
#      MD5 ON THE BOX - CLAUDE.md: scp can truncate a 150 MB file and still
#      exit 0, and only the md5 catches it. A mismatch retries once, then
#      gives up without launching anything;
#   2. ship box_watchdog.sh + the vast API key (idempotent; a box deployed
#      by box_finish.sh already has both);
#   3. register the fleet deadline BEFORE the launch, so the box is never
#      rented-but-unregistered even if the launch fails;
#   4. launch expert_loop.py detached, with the DRIVER's pid in
#      runs/<name>.pid - that is the pid both watchdogs poll;
#   5. start the dashboard and the on-box self-destruct watchdog with a
#      40-minute pid grace (the daemon needs ~10 min to see the pid gone
#      twice and up to 15 to pull);
#   6. re-register with the HARVEST SPEC, at a deadline 5 minutes UNDER the
#      on-box one. An expert loop has no runs/<name>/progress.csv - its runs
#      are runs/<name>/round_<n>/train/ - hence --harvest-only-extra plus
#      the summary and the NEWEST round's artifacts.
#
# TESTING THE QUOTING (the previous reuse scripts broke on it twice):
#   PRINT_ONLY=1 bash run_exit_ab.sh ... > /tmp/remote.sh   # the exact remote script
#   SSH_CMD=... SCP_CMD=... BOXROOT=<tmpdir> ...            # run it against a fake box
# tests/python/test_run_exit_ab.py does the second, with stub binaries, and
# asserts the flags arrive verbatim.
set -euo pipefail
PORT="${1:?usage: run_exit_ab.sh <port> <host> <instance> <name> <hours> <ckpt> [flags...]}"
HOST="${2:?host}"; ID="${3:?instance}"; NAME="${4:?run name}"
HOURS="${5:?hours}"; CKPT="${6:?checkpoint path}"
shift 6

MAIN="${MAIN:-/c/RL_Surf_base}"          # where fleet_watchdog's registry lives
BOXROOT="${BOXROOT:-/root}"              # overridden by the fake-box test
SP="${SP:-/c/Users/bulti/AppData/Local/Temp/claude/C--RL-Surf/e56a2b21-7ab5-4fab-a437-f0bf1163e752/scratchpad}"
WAVE="$(cd "$(dirname "$0")" && pwd)"
SEED="$BOXROOT/RL_Surf/runs_seed.pt"
THREADS="${THREADS:-16}"
ROUNDS="${ROUNDS:-2}"
TRAIN_STEPS="${TRAIN_STEPS:-3e8}"
PLAN_BUDGET="${PLAN_BUDGET:-600}"
EPISODES="${EPISODES:-9}"
MAP="${MAP:-$BOXROOT/RL_Surf/maps/surf_src_cannonball.bsp}"
ROUTE="${ROUTE:-$BOXROOT/RL_Surf/maps/surf_src_cannonball.route.npz}"
GRACE="${GRACE:-40}"
VAST_KEY="${VAST_KEY:-/c/Users/bulti/.config/vastai/vast_api_key}"
SSH_CMD="${SSH_CMD:-ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT root@$HOST}"
# overridable so the fake-box test can run the whole thing locally, and
# so a box behind a jump host needs no edit here
SCP_CMD="${SCP_CMD:-scp -q -P $PORT}"
PY3="${PY3:-python3}"                    # the BOX's interpreter
export PYTHONIOENCODING=utf-8

# every flag after <ckpt>, quoted for the REMOTE bash. printf %q, not "$*":
# one flag with a space in it would otherwise split into two on the box and
# expert_loop would hand the trainer a flag it does not know.
XTRA=""
for a in "$@"; do XTRA="$XTRA $(printf '%q' "$a")"; done
[ -n "$XTRA" ] || { echo "!! no trainer flags given - the two A/B arms differ ONLY in them" >&2; exit 2; }

# EXTRA_LOOP_FLAGS: flags for expert_loop ITSELF, not for the trainer. The
# positional tail above goes after --train-extra (argparse.REMAINDER), which
# swallows the whole rest of the line, so a LOOP flag - --dagger-k 600,
# --objective finish, --bc-coef - can never be passed that way. These land
# BEFORE --train-extra.
#
#   EXTRA_LOOP_FLAGS='--dagger-k 600 --dagger-copies 256' \
#     bash run_exit_ab.sh 41234 ssh5.vast.ai 48512345 exitDAG 6 <ckpt> \
#     --bc-target dist --bc-value-coef 0.25
#
# `eval set --` so a value with a space in it (--dagger-extra '--episodes 4
# --temp 0.7') stays ONE token here and ONE token on the box; each token is
# then printf %q'd for the remote bash exactly like the trainer flags. $@ is
# already consumed into XTRA above, so clobbering it here is safe.
LOOPX=""
if [ -n "${EXTRA_LOOP_FLAGS:-}" ]; then
  eval "set -- $EXTRA_LOOP_FLAGS"
  for a in "$@"; do LOOPX="$LOOPX $(printf '%q' "$a")"; done
fi

DL=$(( $(date +%s) + $(python -c "print(int($HOURS*3600))") ))
# The remote script. Single string, so every $ that must be evaluated ON THE
# BOX is escaped (\$! for the background pid, \$(cat ...) for reading it
# back) and every $ that must be substituted HERE is not.
REMOTE="set -o pipefail; cd $BOXROOT/RL_Surf || exit 1; \
$PY3 tools/restamp_maps.py 2>&1 | tail -1; \
git log --oneline -1; \
(NUMBA_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS nohup $PY3 -u tools/expert_loop.py $SEED \
 --name $NAME --rounds $ROUNDS --train-steps $TRAIN_STEPS --plan-budget $PLAN_BUDGET \
 --episodes $EPISODES --map $MAP --route $ROUTE$LOOPX \
 --train-extra$XTRA > runs/${NAME}_driver.txt 2>&1 < /dev/null & echo \$! > runs/$NAME.pid); \
sleep 20; \
kill -0 \$(cat runs/$NAME.pid) 2>/dev/null && echo \"driver alive pid \$(cat runs/$NAME.pid)\" || { echo '!! driver died'; tail -25 runs/${NAME}_driver.txt; exit 1; }; \
tail -3 runs/${NAME}_driver.txt | cut -c1-200; \
(nohup $PY3 tools/dashboard.py --port 8000 > $BOXROOT/dashboard.log 2>&1 < /dev/null &); \
(nohup bash $BOXROOT/box_watchdog.sh $ID $NAME $DL $GRACE ${PARK:-0} > /dev/null 2>&1 < /dev/null &); \
sleep 2; tail -1 $BOXROOT/box_watchdog.log 2>/dev/null || echo '!! the on-box watchdog wrote no log - CHECK IT, this box has no self-destruct'"

if [ -n "${PRINT_ONLY:-}" ]; then printf '%s\n' "$REMOTE"; exit 0; fi

echo "== $NAME -> $HOST:$PORT (instance $ID), $HOURS h, deadline $(date -d "@$DL" +%H:%M:%S)"
echo "== seed $CKPT -> $SEED (md5-verified on the box)"
test -f "$CKPT" || { echo "!! no such checkpoint: $CKPT" >&2; exit 2; }
MD5=$(md5sum "$CKPT" | awk '{print $1}')
echo "   local md5 $MD5  ($(du -m "$CKPT" | awk '{print $1}') MB)"
ok=0
for try in 1 2; do
  $SCP_CMD "$CKPT" "root@$HOST:$SEED" || true
  # CLAUDE.md: never trust the scp exit code on a checkpoint. The box says.
  if $SSH_CMD "echo '$MD5  $SEED' | md5sum -c -" 2>&1 | grep -q ': OK'; then ok=1; break; fi
  echo "   md5 MISMATCH on attempt $try - re-sending"
done
[ "$ok" = 1 ] || { echo "!! the checkpoint did not arrive intact after 2 tries" >&2; exit 1; }
echo "   md5 OK on the box"

echo "== watchdog + key (idempotent)"
$SCP_CMD "$WAVE/box_watchdog.sh" "root@$HOST:$BOXROOT/box_watchdog.sh"
if [ -f "$VAST_KEY" ]; then
  $SCP_CMD "$VAST_KEY" "root@$HOST:$BOXROOT/.vast_api_key"
  $SSH_CMD "mkdir -p $BOXROOT/.config/vastai && cp $BOXROOT/.vast_api_key $BOXROOT/.config/vastai/vast_api_key && chmod 600 $BOXROOT/.vast_api_key $BOXROOT/.config/vastai/vast_api_key && echo 'key in place'"
fi

# register FIRST with the deadline alone: a box that is rented must be in the
# registry even if the launch below fails (CLAUDE.md: register on create).
echo "== fleet deadline"
( cd "$MAIN" && python tools/fleet_watchdog.py register "$ID" \
    --minutes "$(python -c "print(int($HOURS*60+60))")" --label "exitab-$NAME" 2>&1 | tail -1 )

echo "== launch $NAME (expert_loop${LOOPX:+, loop flags$LOOPX}, --train-extra$XTRA)"
# a previous run's on-box watchdog would poll the OLD pid file and self-destruct the box
# mid-run (2026-09-05). Kill it in its OWN ssh call: a kill sharing a command line with
# the launch below matches that shell and kills it first (which is what happened at 12:05).
$SSH_CMD "pkill -f 'box_watchdo[g].sh' 2>/dev/null; true" 2>&1 | grep -v "Welcome\|Have fun" || true
$SSH_CMD "$REMOTE" 2>&1 | grep -v "Welcome\|Have fun"  || true

# Re-register with the HARVEST SPEC now that the ssh endpoint and the pid
# file are proved. The registry deadline goes 5 min UNDER the on-box one:
# they run off different clocks, and a box that destroys itself before the
# daemon's harvest window opens is how six 5090s died unharvested.
MINS=$(( (DL - $(date +%s)) / 60 - 5 ))
echo "== harvest spec ($MINS min)"
( cd "$MAIN" && python tools/fleet_watchdog.py register "$ID" --minutes "$MINS" \
    --label "exitab-$NAME" --harvest "$PORT $HOST $NAME" --harvest-only-extra \
    --pid-file "runs/$NAME.pid" \
    --harvest-extra "runs/$NAME/expert_summary.jsonl,runs/${NAME}_driver.txt" \
    --harvest-newest "runs/$NAME/round_*/train/ckpt_final.pt,runs/$NAME/round_*/train/progress.csv,runs/$NAME/round_*/train/run.json,runs/$NAME/round_*/train/bc_log.csv,runs/$NAME/round_*/train/traj_*.jsonl,runs/$NAME/round_*/bc_summary.json,runs/$NAME/round_*/round.json" \
    2>&1 | tail -3 )
echo "== done: $NAME on $HOST:$PORT (instance $ID), auto-harvest armed for $MINS min"
