#!/bin/bash
# uf2r2b1: more MONTE CARLO lines with a FIRST-LAP pit exit (beam_tas --elite-frac 1.0) from the TRUE start:
# view-only held-offset forks at t 3.07 s (tick 400) around the 37.34 s winner's settings, and plain MC.
cd /root/RL_Surf
CK=${CK:-runs/beam_it2/ck.pt}
OUT=${OUT:-runs/beam_mc2}
mkdir -p $OUT
COMMON="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --elite-frac 1.0 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8"
i=300
for SP in "60 120 120" "60 120 120" "60 120 120" "60 120 60" "60 120 60" "80 120 120" "80 120 120" "40 160 80" "50 100 100" "70 140 140"; do
  set -- $SP; i=$((i+1))
  python -u tools/beam_tas.py $CK $COMMON --torch-seed $i --branch-at t400:3072 --branch-jitter $1 --branch-burst $2 --branch-hold $3 --branch-protect 0 --branch-view-only --out-dir $OUT/m$i > $OUT/m$i.log 2>&1
done
for s in 401 402; do
  python -u tools/beam_tas.py $CK $COMMON --torch-seed $s --out-dir $OUT/p$s > $OUT/p$s.log 2>&1
done
echo DONE > $OUT/DONE
