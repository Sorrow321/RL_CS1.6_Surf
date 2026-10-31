#!/bin/bash
# uf2r2b1: Monte Carlo on the CURRENT weights (one-lap pit exits from the policy as it is now), then prefix beams
# (the best Monte Carlo line's first 1,174 ticks replayed, a d-ranked search for the rest) on the same weights.
cd /root/RL_Surf
CK=${CK:-runs/beam_mc3/ck.pt}
OUT=${OUT:-runs/beam_mc3}
mkdir -p $OUT
MC="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --elite-frac 1.0 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8"
for s in 601 602 603 604 605 606; do
  python -u tools/beam_tas.py $CK $MC --torch-seed $s --branch-at t400:3072 --branch-jitter 60 --branch-burst 120 --branch-hold 120 --branch-protect 0 --branch-view-only --out-dir $OUT/m$s > $OUT/m$s.log 2>&1
done
for s in 611 612; do
  python -u tools/beam_tas.py $CK $MC --torch-seed $s --out-dir $OUT/p$s > $OUT/p$s.log 2>&1
done
# the fastest Monte Carlo finisher -> prefix beams
BEST=$(for f in $OUT/m6*.log $OUT/p6*.log; do t=$(grep -ho 'CROSSING in [0-9.]*s' $f | grep -o '[0-9.]*' | head -1); [ -n "$t" ] && echo "$t $(dirname $f)/$(basename $f .log)"; done | sort -n | head -1 | cut -d' ' -f2)
echo "best Monte Carlo line: $BEST" > $OUT/best.txt
BS="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8 --gens 3"
for s in 621 622 623; do
  python -u tools/beam_tas.py $CK $BS --torch-seed $s --prefix-line $BEST/beam_best.npz:1174 --out-dir $OUT/pf$s > $OUT/pf$s.log 2>&1
done
echo DONE > $OUT/DONE
