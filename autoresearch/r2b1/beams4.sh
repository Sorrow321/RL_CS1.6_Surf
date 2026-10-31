#!/bin/bash
# uf2r2b1: forks at the pit ENTRY (t 2.3-2.7 s: where the record curves west into the pit and the policy
# curves east), long protection, on the incumbent's weights from the TRUE start.
cd /root/RL_Surf
while [ ! -e runs/beam_r2c/DONE ]; do sleep 5; done
CK=${CK:-runs_ckpt.pt}
OUT=${OUT:-runs/beam_r2d}
mkdir -p $OUT
COMMON="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8 --gens 3"
i=40
for SP in "300 0 65 30 8" "300 60 100 50 8" "300 90 130 65 8" "400 0 65 30 7" "400 60 100 50 7" "400 90 130 65 7"; do
  set -- $SP; i=$((i+1))
  python -u tools/beam_tas.py $CK $COMMON --torch-seed $i --branch-at t$1:3072 --branch-jitter $2 --branch-burst $3 --branch-hold $4 --branch-protect $5 --branch-view-only --out-dir $OUT/e$1_$2 > $OUT/e$1_$2.log 2>&1
done
echo DONE > $OUT/DONE
