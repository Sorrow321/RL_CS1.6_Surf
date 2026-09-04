#!/bin/bash
# box_finish.sh - runs FROM THE WORKSTATION after deploy_box.sh: ship the
# baked cannonball caches (gitignored, ~57 MB) and pin the bsp mtime so the
# cache signatures match (else a 30-minute goal-field rebake on rented time),
# install the vastai CLI + API key for the on-box watchdog, launch the arm
# through tools/run_arm.sh (CLAUDE.md rule 4), start the dashboard, start the
# self-destruct watchdog.
#
#   bash box_finish.sh <port> <host> <instance_id> <run> <deadline_epoch> <trainer flags...>
set -euo pipefail
PORT="$1"; HOST="$2"; ID="$3"; RUN="$4"; DEADLINE="$5"; shift 5
SP="/c/Users/bulti/AppData/Local/Temp/claude/C--RL-Surf/e56a2b21-7ab5-4fab-a437-f0bf1163e752/scratchpad"
LOCAL_MAPS=/c/RL_Surf/maps
MAP=surf_src_cannonball
BSP_MTIME=1776021647154187400
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT root@$HOST"

echo "== caches -> box (md5-verified on the box)"
CACHES=""
for f in goal_32 sdf_32 occ_32 slabocc_32; do CACHES="$CACHES $LOCAL_MAPS/$MAP.$f.npz"; done
# POOL_MAPS=a,b: multi-map arms - ship each pool map's bsp, zones, goal_48/sdf_32/occ_32/slabocc_32 caches (restamped on the box)
POOLFILES=""
for m in $(echo "${POOL_MAPS:-}" | tr "," " "); do for f in /c/RL_Surf/maps_pool/$m.bsp /c/RL_Surf/maps_pool/$m.zones.json /c/RL_Surf/maps_pool/$m.goal_48.npz /c/RL_Surf/maps_pool/$m.sdf_32.npz /c/RL_Surf/maps_pool/$m.occ_32.npz /c/RL_Surf/maps_pool/$m.slabocc_32.npz; do test -f "$f" && POOLFILES="$POOLFILES $f"; done; done
for f in $(echo "${EXTRA_FILES:-}" | tr "," " "); do test -f "$f" && POOLFILES="$POOLFILES $f"; done
md5sum $CACHES | sed "s#$LOCAL_MAPS/##" > "$SP/cache_md5_$ID.txt"
scp -q -P "$PORT" $CACHES $POOLFILES "$SP/cache_md5_$ID.txt" "root@$HOST:/root/RL_Surf/maps/"
if [ -n "$POOLFILES" ]; then md5sum $POOLFILES | sed -E 's#( \*?)[^ ]*/#\1#' > "$SP/pool_md5_$ID.txt"; scp -q -P "$PORT" "$SP/pool_md5_$ID.txt" "root@$HOST:/root/RL_Surf/maps/"; $SSH "cd /root/RL_Surf/maps && md5sum -c pool_md5_$ID.txt | grep -c OK && rm pool_md5_$ID.txt && cd .. && python3 tools/restamp_maps.py 2>&1 | tail -2"; fi
$SSH "cd /root/RL_Surf/maps && md5sum -c cache_md5_$ID.txt && rm cache_md5_$ID.txt && \
      python3 -c \"import os;M=$BSP_MTIME;os.utime('$MAP.bsp',ns=(M,M));print('bsp mtime pinned',os.stat('$MAP.bsp').st_mtime_ns)\" && \
      ls -la $MAP.selfgoal.npz $MAP.route.npz $MAP.zones.json | awk '{print \$5, \$9}'"

echo "== vastai CLI + key for the on-box watchdog"
scp -q -P "$PORT" "$SP/box_watchdog.sh" "root@$HOST:/root/box_watchdog.sh"
scp -q -P "$PORT" /c/Users/bulti/.config/vastai/vast_api_key "root@$HOST:/root/.vast_api_key"
$SSH "mkdir -p /root/.config/vastai && cp /root/.vast_api_key /root/.config/vastai/vast_api_key && chmod 600 /root/.vast_api_key /root/.config/vastai/vast_api_key && \
      (pip install --break-system-packages -q vastai > /root/pip_vastai.log 2>&1 || pip install -q vastai >> /root/pip_vastai.log 2>&1); \
      vastai show instances --raw | python3 -c 'import json,sys; d=json.load(sys.stdin); print(\"vastai CLI ok, sees\", [i[\"id\"] for i in d])'"

echo "== launch $RUN via tools/run_arm.sh (SCRATCH)"
# 256-thread host: numba's parallel pool and OpenMP size off nproc unless
# capped (memory: numba-pools-and-false-deadlocks); 16 per box, two boxes
LAUNCH_ENV="${ARM_ENV:-SCRATCH=1 BUDGET=40000000000 RECORD_EVERY=75e6 EVAL_EPS=9}"
$SSH "cd /root/RL_Surf && NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 $LAUNCH_ENV bash tools/run_arm.sh $RUN $* 2>&1 | tail -8"

echo "== dashboard :8000 + on-box watchdog"
$SSH "cd /root/RL_Surf && (nohup python3 tools/dashboard.py --port 8000 > /root/dashboard.log 2>&1 < /dev/null &) ; \
      (nohup bash /root/box_watchdog.sh $ID $RUN $DEADLINE > /dev/null 2>&1 < /dev/null &) ; sleep 3; \
      ps -eo pid,args | grep '[d]ashboard.py --port 8000' | head -1 | awk '{print \"dashboard pid\", \$1}'; tail -1 /root/box_watchdog.log"
echo "== done: $RUN on $HOST:$PORT (instance $ID)"
