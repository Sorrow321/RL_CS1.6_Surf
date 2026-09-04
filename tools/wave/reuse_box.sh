#!/bin/bash
# reuse_box.sh - relaunch a NEW arm on a box that already finished (and was
# harvested from) a previous arm: no image pull, no deps, no 60 s lottery.
# Stops the old trainer and the old on-box watchdog (which would otherwise
# destroy the box 10 min after the old pid died), moves the checkout to the
# current baseline head, re-registers the fleet deadline, then runs
# box_finish.sh exactly as wave_launch does (caches, pool maps, launch,
# dashboard, new watchdog).
#   ARM_ENV="..." POOL_MAPS="a,b" bash reuse_box.sh <port> <host> <instance> <run> <hours> <trainer flags...>
set -euo pipefail
PORT="$1"; HOST="$2"; ID="$3"; RUN="$4"; HOURS="$5"; shift 5
SP="/c/Users/bulti/AppData/Local/Temp/claude/C--RL-Surf/e56a2b21-7ab5-4fab-a437-f0bf1163e752/scratchpad"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT root@$HOST"
export PYTHONIOENCODING=utf-8
echo "== $RUN -> $HOST:$PORT (instance $ID): stop old trainer + old watchdog"
# bracket trick: the pattern never matches the shell that carries it (CLAUDE.md: pkill -f self-match)
$SSH "cd /root/RL_Surf; for p in \$(ps -eo pid,args | awk '/[b]ox_watchdog.sh/ {print \$1}'); do kill \$p && echo \"killed watchdog \$p\"; done; \
      for p in \$(ps -eo pid,args | awk '/[t]rain_fast.py/ {print \$1}'); do kill \$p && echo \"killed trainer \$p\"; done; \
      for i in \$(seq 30); do ps -eo args | grep -q '[t]rain_fast.py' || break; sleep 2; done; ps -eo args | grep -c '[t]rain_fast.py' || echo 'no trainer left'; \
      nvidia-smi --query-gpu=memory.used --format=csv,noheader" 2>&1 | grep -v "Welcome\|Have fun"
echo "== checkout current baseline head"
$SSH "cd /root/RL_Surf && git -c http.version=HTTP/1.1 fetch -q --depth 1 origin baseline && git checkout -q -B baseline FETCH_HEAD && git log --oneline -1 && git status --short | head -5 && bash build.sh 2>&1 | tail -1 && nm -D build/libsurfcore.so | grep -c surf_set_msec" 2>&1 | grep -v "Welcome\|Have fun"
echo "== fleet deadline: $HOURS h + 1 h"
( cd /c/RL_Surf_base && python tools/fleet_watchdog.py register "$ID" --minutes $(python -c "print(int($HOURS*60+60))") --label "reuse-$RUN" 2>&1 | tail -1 )
DL=$(( $(date +%s) + $(python -c "print(int($HOURS*3600))") ))
echo "== box_finish (deadline $(date -d @$DL +%H:%M:%S))"
bash "$SP/box_finish.sh" "$PORT" "$HOST" "$ID" "$RUN" "$DL" "$@"
