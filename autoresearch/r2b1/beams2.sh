#!/bin/bash
# uf2r2b1: fork sweep for a FIRST-LAP exit of the start pit, on the incumbent's weights from the TRUE start.
# View-only forks (the policy's own keys; held view-target offsets) at several ticks of the first pit lap.
cd /root/RL_Surf
CK=${CK:-runs_ckpt.pt}
OUT=${OUT:-runs/beam_r2b}
mkdir -p $OUT
COMMON="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8 --gens 3"
i=0
for T in 522 587 652 717; do        # t 4.0 / 4.5 / 5.0 / 5.5 s at 7.667 ms
  i=$((i+1))
  python -u tools/beam_tas.py $CK $COMMON --torch-seed $((10+i)) --branch-at t$T:3072 --branch-burst 130 --branch-hold 52 --branch-jitter 0 --branch-protect 4 --branch-view-only --out-dir $OUT/u$T > $OUT/u$T.log 2>&1
  python -u tools/beam_tas.py $CK $COMMON --torch-seed $((20+i)) --branch-at t$T:3072 --branch-burst 196 --branch-hold 100 --branch-jitter 30 --branch-protect 5 --branch-view-only --out-dir $OUT/j$T > $OUT/j$T.log 2>&1
done
echo DONE > $OUT/DONE
