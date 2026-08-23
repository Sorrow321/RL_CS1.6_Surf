#!/bin/bash
# bake_box.sh - stand up a rented box and start one shard of the goal-field
# bake. Lean sibling of deploy_box.sh: no checkpoint, no test suite, no
# training - clone, build, ship the shard's BSPs, gate on gpu_health, launch.
#
#   bash tools/bake_box.sh <port> <host> <shard_n>
#
# Run FROM the workstation. maps_full_dataset/ is the user's data and stays
# READ-ONLY: the shard's BSPs are tarred out of it and everything derived
# lands in /root/pool/out on the box.
#
# The one non-obvious step is `--pin-mtimes`. Cache signatures embed the
# bsp's size AND st_mtime_ns (vision._map_sig), so a field baked against a
# COPIED bsp is rejected the moment it comes home and silently re-bakes.
# The pool json carries the workstation's mtime_ns per map and the bake
# restores it before touching anything.
set -euo pipefail

PORT="${1:?usage: bake_box.sh <port> <host> <shard_n>}"
HOST="${2:?usage: bake_box.sh <port> <host> <shard_n>}"
SHARD="${3:?usage: bake_box.sh <port> <host> <shard_n>}"

LOCAL_REPO="${LOCAL_REPO:-/c/RL_Surf}"
DATASET="${DATASET:-$LOCAL_REPO/maps_full_dataset}"
WORK="${WORK:?set WORK to the directory holding pool.json and shard<N>.txt}"
REPO="${REPO:-https://github.com/Sorrow321/RL_CS1.6_Surf}"
BRANCH="${BRANCH:-goalbake}"
TIMEOUT="${TIMEOUT:-2400}"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

SH="$WORK/shard${SHARD}.txt"
test -f "$SH" || { echo "no shard file $SH"; exit 1; }
NMAP=$(grep -c . "$SH")

echo "== 1/5 recon $HOST:$PORT (shard $SHARD, $NMAP maps)"
$SSH -p "$PORT" "root@$HOST" "nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; \
  nproc; free -g | awk 'NR==2{print \"RAM \" \$2 \"G\"}'; df -h / | tail -1"

echo "== 2/5 clone $BRANCH + build (torch comes from the image)"
$SSH -p "$PORT" "root@$HOST" "git clone --depth 1 --single-branch --branch $BRANCH $REPO /root/RL_Surf 2>&1 | tail -1; \
  cd /root/RL_Surf && git log --oneline -1 && bash build.sh 2>&1 | tail -1; \
  mkdir -p /root/pool/maps /root/pool/out; \
  python3 -c 'import torch,numpy;print(\"torch\",torch.__version__,torch.cuda.device_count(),\"GPU\")'"

echo "== 3/5 gpu_health (the real gate - a capped card bakes slowly, a"
echo "        foreign tenant OOMs a 600 Mvoxel bake)"
$SSH -p "$PORT" "root@$HOST" "cd /root/RL_Surf && python3 tools/gpu_health.py --all"

echo "== 4/5 ship shard $SHARD ($NMAP maps) + pool meta, pin mtimes"
tar -czf - -C "$DATASET" $(sed 's/$/.bsp/' "$SH" | tr '\n' ' ') \
  | $SSH -p "$PORT" "root@$HOST" "tar -xzf - -C /root/pool/maps"
scp -q -P "$PORT" "$WORK/pool.json" "$SH" "root@$HOST:/root/pool/"
$SSH -p "$PORT" "root@$HOST" "cd /root/RL_Surf && \
  ls /root/pool/maps/*.bsp | wc -l && \
  python3 tools/bake_pool.py --pool /root/pool/pool.json \
    --maps-dir /root/pool/maps --out /root/pool/out --pin-mtimes"

echo "== 5/5 launch bake (detached; log /root/pool/bake.log)"
$SSH -p "$PORT" "root@$HOST" "cd /root/RL_Surf && \
  (setsid nohup python3 -u tools/bake_pool.py \
     --pool /root/pool/pool.json --shard /root/pool/shard${SHARD}.txt \
     --maps-dir /root/pool/maps --out /root/pool/out --timeout $TIMEOUT \
     > /root/pool/bake.log 2>&1 < /dev/null &); sleep 6; head -3 /root/pool/bake.log"
echo "shard $SHARD running on $HOST:$PORT"
