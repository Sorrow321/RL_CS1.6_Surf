#!/bin/bash
# uf2r2b1: one expert-iteration search round on a checkpoint CK from the TRUE start: six beam_tas searches
# (plain d-ranked x3 seeds, view-only forks in the pit x3), each keeping its 8 fastest finishing lines.
# Usage: CK=<ckpt> OUT=runs/beam_itN bash autoresearch/r2b1/beams_it.sh
cd /root/RL_Surf
mkdir -p $OUT
COMMON="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8 --gens 3"
python -u tools/beam_tas.py $CK $COMMON --torch-seed 100 --out-dir $OUT/p0 > $OUT/p0.log 2>&1
python -u tools/beam_tas.py $CK $COMMON --torch-seed 101 --branch-at t650:2048 --branch-burst 150 --branch-hold 50 --branch-jitter 45 --branch-protect 3 --branch-view-only --out-dir $OUT/f1 > $OUT/f1.log 2>&1
python -u tools/beam_tas.py $CK $COMMON --torch-seed 102 --out-dir $OUT/p2 > $OUT/p2.log 2>&1
python -u tools/beam_tas.py $CK $COMMON --torch-seed 103 --branch-at t300:3072 --branch-burst 130 --branch-hold 65 --branch-jitter 90 --branch-protect 8 --branch-view-only --out-dir $OUT/f3 > $OUT/f3.log 2>&1
python -u tools/beam_tas.py $CK $COMMON --torch-seed 104 --out-dir $OUT/p4 > $OUT/p4.log 2>&1
python -u tools/beam_tas.py $CK $COMMON --torch-seed 105 --branch-at t3900:2048 --branch-burst 130 --branch-hold 65 --branch-jitter 30 --branch-protect 4 --branch-view-only --out-dir $OUT/f5 > $OUT/f5.log 2>&1
echo DONE > $OUT/DONE
