#!/bin/bash
# uf2r2b1: MONTE CARLO (beam_tas --elite-frac 1.0: no cloning, every lineage runs to its end) from the TRUE
# start with view-only forks of held heading offsets at the first lap's north turn (t 3.1 / 3.8 s): does any
# lineage leave the start pit on its FIRST lap and finish? (a d-ranked beam culls a detour north)
cd /root/RL_Surf
CK=${CK:-runs_ckpt.pt}
OUT=${OUT:-runs/beam_mc1}
mkdir -p $OUT
COMMON="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --elite-frac 1.0 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8"
i=200
for SP in "400 20 60 30" "400 40 120 60" "400 60 120 120" "500 20 60 30" "500 40 120 60" "500 60 120 120" "400 0 60 60" "500 0 60 60"; do
  set -- $SP; i=$((i+1))
  python -u tools/beam_tas.py $CK $COMMON --torch-seed $i --branch-at t$1:3072 --branch-jitter $2 --branch-burst $3 --branch-hold $4 --branch-protect 0 --branch-view-only --out-dir $OUT/m$1_$2_$3 > $OUT/m$1_$2_$3.log 2>&1
done
echo DONE > $OUT/DONE
