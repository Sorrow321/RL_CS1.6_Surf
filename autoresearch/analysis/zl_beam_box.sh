#!/bin/bash
# zlr2b2: one policy-beam search on the box from the TRUE start (runs in the background).
#   bash autoresearch/analysis/zl_beam_box.sh <tag> <ckpt> [extra beam_tas flags...]
# Writes runs/beam/<tag>/ (beam_frontier.npz with --save-frontier) and runs/beam/<tag>.log.
# The search: tools/beam_tas.py on the policy's own proposals (+ whatever the flags add);
# nothing in it names a place on the map.
set -u
TAG=$1; CK=$2; shift 2
cd /root/RL_Surf
mkdir -p runs/beam/$TAG
nohup python3 -u tools/beam_tas.py "$CK" --map maps/surf_za_loop.bsp \
    --allow-nonfinisher --skip-gate --save-frontier --out-dir runs/beam/$TAG "$@" \
    > runs/beam/$TAG.log 2>&1 &
echo $! > runs/beam/$TAG.pid
echo "beam $TAG pid $(cat runs/beam/$TAG.pid)"
