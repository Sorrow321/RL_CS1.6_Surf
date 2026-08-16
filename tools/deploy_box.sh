#!/bin/bash
# deploy_box.sh — stand up a fresh rented GPU box, per DEPLOY.md, in one call.
#
#   bash tools/deploy_box.sh <new_port> <new_host>          # seed from here
#   SEED_PORT=39455 SEED_HOST=1.2.3.4 bash tools/deploy_box.sh <port> <host>
#
# Run this FROM your workstation. It bootstraps the new box (clone, build,
# torch) and then installs the checkpoint + baked caches, WITHOUT which the
# first launch spends 10-30 GPU-minutes re-baking the geodesic goal field.
#
# Default is to push them from this workstation (~84 MB, measured at 1.7 MB/s
# = under a minute). Set SEED_HOST/SEED_PORT to pull box-to-box instead, which
# is faster but only works while the old box still exists — in practice the
# previous box is usually deleted before the new one is rented, which is why
# local is the default.
#
# Two things that have already gone wrong once each and are handled below:
#   * appending to authorized_keys with `echo >>` when the file has no
#     trailing newline FUSES your key onto the previous one and locks you out;
#   * cache .npz signatures embed the bsp's size AND st_mtime_ns, so a copied
#     cache silently rebakes (a 10-30 min GPU bake) unless the mtime is pinned.
set -euo pipefail

PORT="${1:?usage: deploy_box.sh <port> <host>}"
HOST="${2:?usage: deploy_box.sh <port> <host>}"
SEED_PORT="${SEED_PORT:-}"
SEED_HOST="${SEED_HOST:-}"
# where the ckpt + caches live on THIS workstation (the local-seed default)
LOCAL_REPO="${LOCAL_REPO:-/c/RL_Surf}"
LOCAL_CKPT="${LOCAL_CKPT:-$LOCAL_REPO/runs/race_respawn/ckpt_6348079104.pt}"
REPO="${REPO:-https://github.com/Sorrow321/RL_CS1.6_Surf}"
MAP="${MAP:-surf_src_cannonball}"
BSP_MTIME="${BSP_MTIME:-1776021647154187400}"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

echo "== 1/5 recon $HOST:$PORT"
$SSH -p "$PORT" "root@$HOST" "nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv; \
  nvidia-smi topo -m 2>/dev/null | head -8; \
  lscpu | grep -iE 'model name|^cpu\(s\)|max mhz|numa node\(s\)'; free -g | head -2; df -h / | tail -1"

echo "== 2/5 torch (backgrounded; it is the long pole) + clone + build"
$SSH -p "$PORT" "root@$HOST" "(setsid nohup pip install --break-system-packages torch scipy numpy \
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
    > /root/pip.log 2>&1 < /dev/null &); sleep 2; \
  git clone --depth 1 $REPO /root/RL_Surf 2>&1 | tail -1; \
  cd /root/RL_Surf && git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*' \
    && git fetch origin --quiet && git checkout -q -B main origin/main && git log --oneline -1 \
    && mkdir -p runs && bash build.sh 2>&1 | tail -1"

if [ -z "$SEED_HOST" ]; then
  echo "== 3/5 skipped (seeding from this workstation)"
else
echo "== 3/5 authorise the seed box (newline-safe)"
SEED_KEY=$($SSH -p "$SEED_PORT" "root@$SEED_HOST" \
    "test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 -q; cat ~/.ssh/id_ed25519.pub")
$SSH -p "$PORT" "root@$HOST" "mkdir -p ~/.ssh && python3 - <<PY
p='/root/.ssh/authorized_keys'
k='''$SEED_KEY'''.strip()
import os
s=open(p).read() if os.path.exists(p) else ''
s=s.replace(k,'')
open(p,'w').write(s.rstrip('\n')+'\n'+k+'\n')
import stat; os.chmod(p, 0o600)
print('authorized_keys rows:', sum(1 for _ in open(p)))
PY"

fi

if [ -z "$SEED_HOST" ]; then
  echo "== 4/5 push ckpt + caches from this workstation, pin the bsp mtime"
  test -f "$LOCAL_CKPT" || { echo "no local ckpt at $LOCAL_CKPT"; exit 1; }
  scp -q -P "$PORT" "$LOCAL_REPO/maps/$MAP".*_*.npz "$LOCAL_CKPT" "root@$HOST:/root/"
  $SSH -p "$PORT" "root@$HOST" "cd /root/RL_Surf && mkdir -p runs && mv /root/*.npz maps/ &&     mv /root/$(basename "$LOCAL_CKPT") runs_ckpt.pt &&     python3 -c \"import os;M=$BSP_MTIME;os.utime('maps/$MAP.bsp',ns=(M,M));print('bsp mtime pinned',os.stat('maps/$MAP.bsp').st_mtime_ns)\" &&     md5sum runs_ckpt.pt && ls maps/*.npz | wc -l"
else
echo "== 4/5 pull ckpt + caches from $SEED_HOST, pin the bsp mtime"
$SSH -p "$SEED_PORT" "root@$SEED_HOST" "cd /root/RL_Surf && \
  scp -q -o StrictHostKeyChecking=accept-new -o BatchMode=yes -P $PORT \
    runs_ckpt.pt maps/$MAP.occ_*.npz maps/$MAP.slabocc_*.npz maps/$MAP.sdf_*.npz maps/$MAP.goal_*.npz \
    root@$HOST:/root/ && \
  ssh -o BatchMode=yes -p $PORT root@$HOST \"mv /root/*.npz /root/RL_Surf/maps/ && mv /root/runs_ckpt.pt /root/RL_Surf/ && \
    python3 -c \\\"import os;M=$BSP_MTIME;os.utime('/root/RL_Surf/maps/$MAP.bsp',ns=(M,M));print('bsp mtime pinned',os.stat('/root/RL_Surf/maps/$MAP.bsp').st_mtime_ns)\\\" && \
    md5sum /root/RL_Surf/runs_ckpt.pt && ls /root/RL_Surf/maps/*.npz | wc -l\""

fi

echo "== 5/6 wait for torch, then run the test suite"
until $SSH -p "$PORT" "root@$HOST" "python3 -c 'import torch,triton' 2>/dev/null"; do sleep 30; done
$SSH -p "$PORT" "root@$HOST" "python3 -c 'import torch,triton;print(\"torch\",torch.__version__,\"triton\",triton.__version__,torch.cuda.device_count(),\"GPUs\")'; \
  cd /root/RL_Surf && python3 -m pytest tests/python -q 2>&1 | tail -2"

echo "== 6/6 GPU health — a rented card can be the right model and still be capped"
if ! $SSH -p "$PORT" "root@$HOST" "cd /root/RL_Surf && python3 tools/gpu_health.py --all"; then
  cat <<'BAD'

!! This box's GPUs are below spec. A measured instance ran sustained bf16
!! GEMM at 166 TFLOPS against 234 on a healthy 5090 -- SM pinned to 1987 MHz
!! instead of ~2890, at 392 W of a 575 W limit and 62 C, "SW Power Cap"
!! active, and -lgc denied inside the container. It had the fastest CPU of
!! any box measured and was still 21% slower per GPU. Switch instances.
BAD
  exit 1
fi

cat <<MSG

ready. benchmark it with (per-box baseline, always paired):
  ssh -p $PORT root@$HOST "cd /root/RL_Surf && \\
    python3 -u python/train_fast.py --ckpt runs_ckpt.pt --run pb --record-every 1e12 \\
      --steps \$((6348079104 + 40*786432)) --timing > runs/pb.log 2>&1"
  ssh -p $PORT root@$HOST "cd /root/RL_Surf && python3 tools/perf_report.py runs/pb.log"
MSG
