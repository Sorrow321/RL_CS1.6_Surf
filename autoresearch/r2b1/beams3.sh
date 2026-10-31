#!/bin/bash
# uf2r2b1: held view offsets on every proposal env (--view-hold SIGMA PERIOD) from the TRUE start, on the
# incumbent's weights: a held heading change is what the collapsed yaw sigma never proposes.
cd /root/RL_Surf
while [ ! -e runs/beam_r2b/DONE ]; do sleep 5; done
CK=${CK:-runs_ckpt.pt}
OUT=${OUT:-runs/beam_r2c}
mkdir -p $OUT
COMMON="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8 --gens 3"
i=30
for SP in "0.5 10" "1.0 10" "2.0 10" "0.5 25" "1.0 25" "2.0 25"; do
  set -- $SP; i=$((i+1))
  python -u tools/beam_tas.py $CK $COMMON --torch-seed $i --view-hold $1 $2 --out-dir $OUT/vh$1_$2 > $OUT/vh$1_$2.log 2>&1
done
echo DONE > $OUT/DONE
