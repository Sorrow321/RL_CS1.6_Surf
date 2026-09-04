#!/bin/bash
# rent_expert_box.sh - rent one 5090 (race 3, keep the first ready), deploy
# baseline, ship caches + key + watchdog, launch the from-scratch expert loop
# with the driver pid as the watchdog's liveness file. Workstation side.
#   bash rent_expert_box.sh <out_dir> <hours>
set -uo pipefail
OUT="$1"; HOURS="${2:-8}"
SP="/c/Users/bulti/AppData/Local/Temp/claude/C--RL-Surf/e56a2b21-7ab5-4fab-a437-f0bf1163e752/scratchpad"
export PYTHONIOENCODING=utf-8
cd /c/RL_Surf_base
mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/log.txt"; }
OFFERS=$(python tools/vast_pick.py --gpu RTX_5090 -n 4 2>&1 | awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {print $1}' | head -4)
log "offers: $OFFERS"
IDS=""
for o in $OFFERS; do
  id=$(vastai create instance $o --image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel --disk 60 --ssh --direct --raw 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin).get('new_contract',''))" 2>/dev/null)
  [ -n "$id" ] && IDS="$IDS $id" && python tools/fleet_watchdog.py register $id --minutes $(( HOURS * 60 + 60 )) --label exitdag >/dev/null 2>&1
done
log "created:$IDS"
sleep 72
vastai show instances --raw > "$OUT/inst.json" 2>/dev/null
KEEP=""; HOST=""; PORT=""
probe() { timeout 30 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p "$2" "root@$1" "hostname; nvidia-smi --query-gpu=name --format=csv,noheader" 2>&1 | grep -q "RTX"; }
for id in $IDS; do
  hp=$(python -c "
import json; d=json.load(open(r'$OUT/inst.json'.replace('/c/','C:/')))
i=[x for x in d if x['id']==$id]
print((i[0].get('ssh_host') or '-')+' '+str(i[0].get('ssh_port') or 0)+' '+str(i[0].get('actual_status')) ) if i else print('- 0 gone')")
  set -- $hp; h=$1; p=$2; st=$3
  ok=0
  if [ "$h" != "-" ] && probe "$h" "$p"; then ok=1; fi
  if [ $ok = 0 ] && [ "$st" = "running" ]; then sleep 20; probe "$h" "$p" && ok=1; fi
  log "probe $id $h:$p status=$st ok=$ok"
  if [ -z "$KEEP" ] && [ $ok = 1 ]; then
    KEEP=$id; HOST=$h; PORT=$p; log "ready $id $h:$p"
  elif [ $ok = 1 ]; then
    python tools/fleet_watchdog.py release $id 2>&1 | tail -1 | tee -a "$OUT/log.txt"
  else
    python tools/vast_pick.py --block $id --reason network --detail "readiness: no ssh 72-95 s after create, status $st (exitdag race)" >/dev/null 2>&1
    python tools/fleet_watchdog.py release $id 2>&1 | tail -1 | tee -a "$OUT/log.txt"
  fi
done
[ -z "$KEEP" ] && { log "no box came up"; exit 1; }
S="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT root@$HOST"
log "deploy"
$S "(setsid nohup bash -c 'DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg; pip install --break-system-packages scipy numpy opencv-python-headless numba pytest vastai psutil' > /root/pip.log 2>&1 < /dev/null &); sleep 1; git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/Sorrow321/RL_CS1.6_Surf /root/RL_Surf 2>&1 | tail -1; cd /root/RL_Surf && git config http.version HTTP/1.1 && git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*' && timeout 240 git fetch --depth 1 origin baseline 2>&1 | tail -1; git checkout -q -B baseline FETCH_HEAD && git log --oneline -1 && mkdir -p runs && bash build.sh 2>&1 | tail -1" 2>&1 | grep -v "Welcome\|Have fun" | tee -a "$OUT/log.txt"
for i in $(seq 40); do $S "python3 -c 'import torch,triton,scipy,numba,cv2,vastai,psutil' 2>/dev/null && echo DEPS_OK" 2>/dev/null | grep -q DEPS_OK && break; sleep 15; done
$S "cd /root/RL_Surf && NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 python3 tools/gpu_health.py --all 2>&1 | tail -2" 2>&1 | grep -v "Welcome\|Have fun" | tee -a "$OUT/log.txt" | grep -q "VERDICT: healthy" || { log "gpu unhealthy - releasing"; python tools/vast_pick.py --block $KEEP --reason gpu_capped --detail "gpu_health below spec (exit_scratch)" >/dev/null 2>&1; python tools/fleet_watchdog.py release $KEEP; exit 1; }
log "caches + key + watchdog"
CACHES=""; for f in goal_32 sdf_32 occ_32 slabocc_32; do CACHES="$CACHES /c/RL_Surf/maps/surf_src_cannonball.$f.npz"; done
md5sum $CACHES | sed 's#/c/RL_Surf/maps/##' > "$OUT/cache_md5.txt"
CK=/c/RL_Surf_exit/runs/exit10/round_23/train/ckpt_final.pt; md5sum "$CK" | cut -d" " -f1 > "$OUT/ck_md5.txt"; scp -q -P "$PORT" $CACHES "$OUT/cache_md5.txt" "$SP/box_watchdog.sh" /c/Users/bulti/.config/vastai/vast_api_key "$CK" "$OUT/ck_md5.txt" "root@$HOST:/root/"
$S "cd /root && mv surf_src_cannonball.*.npz RL_Surf/maps/ && mv cache_md5.txt RL_Surf/maps/ && cd RL_Surf/maps && md5sum -c cache_md5.txt && python3 -c \"import os;M=1776021647154187400;os.utime('surf_src_cannonball.bsp',ns=(M,M));print('pinned')\" && mkdir -p /root/.config/vastai && cp /root/vast_api_key /root/.vast_api_key && cp /root/vast_api_key /root/.config/vastai/vast_api_key && chmod 600 /root/.vast_api_key /root/.config/vastai/vast_api_key && mkdir -p /root/RL_Surf/runs/seed && mv /root/ckpt_final.pt /root/RL_Surf/runs/seed/ckpt_final.pt && test \"$(md5sum /root/RL_Surf/runs/seed/ckpt_final.pt | cut -d' ' -f1)\" = \"$(cat /root/ck_md5.txt)\" && echo 'seed ckpt md5 OK'" 2>&1 | grep -v "Welcome\|Have fun" | tee -a "$OUT/log.txt"
log "launch exitdag"
DL=$(( $(date +%s) + HOURS * 3600 ))
$S "cd /root/RL_Surf && python3 tools/restamp_maps.py 2>&1 | tail -1; (NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 nohup python3 -u tools/expert_loop.py /root/RL_Surf/runs/seed/ckpt_final.pt --name exitdag --rounds 8 --train-steps 3e8 --plan-budget 300 --objective finish --map /root/RL_Surf/maps/surf_src_cannonball.bsp --dagger-k 600 --dagger-window 3 --dagger-budget 600 --dagger-envs 2048 --dagger-copies 256 --dagger-share 0.25 > runs/exitdag_driver.txt 2>&1 < /dev/null & echo \$! > runs/exitdag.pid); sleep 20; kill -0 \$(cat runs/exitdag.pid) && echo 'driver alive' ; tail -3 runs/exitdag_driver.txt | cut -c1-160; (nohup python3 tools/dashboard.py --port 8000 > /root/dashboard.log 2>&1 < /dev/null &); (nohup bash /root/box_watchdog.sh $KEEP exitdag $DL > /dev/null 2>&1 < /dev/null &); sleep 2; tail -1 /root/box_watchdog.log" 2>&1 | grep -v "Welcome\|Have fun" | tee -a "$OUT/log.txt"
echo "{\"name\": \"exitdag\", \"host\": \"$HOST\", \"port\": $PORT, \"instance\": $KEEP, \"run\": \"exitdag\"}" > "$OUT/box.json"
# Automatic harvest - see rent_expert_box6.sh. The driver's pid dying is the
# NORMAL end of an expert run, and the on-box watchdog destroys the box 10 min
# later; the daemon pulls the summary and the newest round's checkpoint as
# soon as it has seen the pid gone twice, and again 20 min before the
# deadline, which is why the registry deadline goes 5 min under the on-box one.
MINS=$(( (DL - $(date +%s)) / 60 - 5 ))
python tools/fleet_watchdog.py register "$KEEP" --minutes "$MINS" --label exitdag \
  --harvest "$PORT $HOST exitdag" --harvest-only-extra \
  --pid-file runs/exitdag.pid \
  --harvest-extra "runs/exitdag/expert_summary.jsonl,runs/exitdag_driver.txt" \
  --harvest-newest "runs/exitdag/round_*/train/ckpt_final.pt,runs/exitdag/round_*/train/progress.csv,runs/exitdag/round_*/train/run.json,runs/exitdag/round_*/train/traj_*.jsonl" \
  2>&1 | tail -3 | tee -a "$OUT/log.txt"
log "done: exitdag on $HOST:$PORT instance $KEEP (auto-harvest armed, $MINS min)"
