#!/bin/bash
# uf2r2b1: speed searches on the incumbent's weights (runs_ckpt.pt = incumbent 13.046B) from the TRUE start.
# The policy's own proposals (beam_tas); nothing from the record. Run on the box from /root/RL_Surf.
cd /root/RL_Surf
CK=${CK:-runs_ckpt.pt}
OUT=${OUT:-runs/beam_r2}
mkdir -p $OUT
COMMON="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8 --gens 3"
# S1: the plain beam (round 1's fintest rule) on these weights
python -u tools/beam_tas.py $CK $COMMON --torch-seed 0 --out-dir $OUT/s1 > $OUT/s1.log 2>&1
# S2: a view-only fork inside the first pit lap (t ~5.0 s = tick 650 at 7.667 ms): half the population holds
#     view offsets for 150 ticks around the policy's yaw, protected 3 generations
python -u tools/beam_tas.py $CK $COMMON --torch-seed 1 --branch-at t650:2048 --branch-burst 150 --branch-hold 50 --branch-jitter 45 --branch-protect 3 --branch-view-only --out-dir $OUT/s2 > $OUT/s2.log 2>&1
# S3: hotter proposals from the true start (keys logits / 1.5, yaw sigma x 2)
python -u tools/beam_tas.py $CK $COMMON --torch-seed 2 --keys-temp 1.5 --view-temp 2 --out-dir $OUT/s3 > $OUT/s3.log 2>&1
echo DONE > $OUT/DONE
