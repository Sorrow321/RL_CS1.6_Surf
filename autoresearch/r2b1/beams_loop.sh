#!/bin/bash
# uf2r2b1: Monte Carlo at the 60-70% LOOP: replay the best real-finish line's first PREFIX ticks (beam_tas
# --prefix-line), then 4,096 rollouts with NO cloning (--elite-frac 1.0) and a view-only held-offset fork at
# tick T (t ~25.4-26.1 s, where the record turns east into the bowl and the policy carries on north and loops).
cd /root/RL_Surf
CK=${CK:-runs/beam_real1/ck.pt}
LINE=${LINE:-runs/beam_real1/pf723/beam_best.npz}
OUT=${OUT:-runs/beam_loop1}
mkdir -p $OUT
MC="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --elite-frac 1.0 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8"
i=800
for SP in "3200 3300 60 120" "3200 3300 90 120" "3200 3300 30 120" "3200 3400 60 120" "3200 3400 90 120" "3100 3200 60 120" "3200 3300 60 65" "3300 3400 60 120"; do
  set -- $SP; i=$((i+1))
  python -u tools/beam_tas.py $CK $MC --torch-seed $i --prefix-line $LINE:$1 --branch-at t$2:3072 --branch-jitter $3 --branch-burst 120 --branch-hold $4 --branch-protect 0 --branch-view-only --out-dir $OUT/l$i > $OUT/l$i.log 2>&1
done
echo DONE > $OUT/DONE
