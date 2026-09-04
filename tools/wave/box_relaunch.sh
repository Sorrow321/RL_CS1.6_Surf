#!/bin/bash
# box_relaunch.sh - swap the arm running on a box (workstation side):
# stop the on-box watchdog FIRST (it destroys the box 10 min after the
# trainer dies), stop the trainer, update the code to the pushed branch,
# launch the new arm through tools/run_arm.sh, restart the watchdog with
# the new run name and deadline. The dashboard keeps running.
#
#   bash box_relaunch.sh <port> <host> <instance_id> <old_run> <new_run> <deadline_epoch> <trainer flags...>
set -euo pipefail
PORT="$1"; HOST="$2"; ID="$3"; OLD="$4"; RUN="$5"; DEADLINE="$6"; shift 6
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT root@$HOST"

echo "== stop watchdog + trainer ($OLD)"
# pkill -f with a bracketed regex cannot match this very command line
$SSH "pkill -f '[b]ox_watchdog.sh' && echo 'watchdog stopped' || echo 'no watchdog running'; \
      cd /root/RL_Surf && P=\$(cat runs/$OLD.pid 2>/dev/null || echo 0); \
      if [ \"\$P\" != 0 ] && kill -0 \$P 2>/dev/null; then kill \$P; sleep 5; kill -0 \$P 2>/dev/null && kill -9 \$P; echo \"trainer \$P stopped\"; else echo 'trainer not running'; fi; \
      nvidia-smi --query-gpu=memory.used --format=csv,noheader"

echo "== update code"
$SSH "cd /root/RL_Surf && git config http.version HTTP/1.1 && timeout 240 git fetch --depth 1 origin goallines 2>&1 | tail -1; \
      git checkout -q -B goallines FETCH_HEAD && git log --oneline -1 && bash build.sh 2>&1 | tail -1"

echo "== launch $RUN via tools/run_arm.sh (SCRATCH)"
$SSH "cd /root/RL_Surf && NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 SCRATCH=1 BUDGET=40000000000 RECORD_EVERY=75e6 EVAL_EPS=9 bash tools/run_arm.sh $RUN $* 2>&1 | tail -6"

echo "== on-box watchdog for $RUN"
$SSH "(nohup bash /root/box_watchdog.sh $ID $RUN $DEADLINE > /dev/null 2>&1 < /dev/null &); sleep 2; tail -1 /root/box_watchdog.log; \
      ps -eo pid,args | grep '[d]ashboard.py --port 8000' | head -1 | awk '{print \"dashboard pid\", \$1}'"
echo "== done: $RUN on $HOST:$PORT (instance $ID)"
