#!/bin/bash
# uf2r2b1: search round on a policy that already takes the one-lap pit exit: d-ranked beams from the TRUE start,
# Monte Carlo (elite-frac 1.0), then prefix beams on the fastest finisher (its first 1,174 ticks replayed).
cd /root/RL_Surf
CK=${CK:-runs/beam_it4/ck.pt}
OUT=${OUT:-runs/beam_it4}
mkdir -p $OUT
BS="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8 --gens 3"
MC="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --elite-frac 1.0 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8"
for s in 701 702; do python -u tools/beam_tas.py $CK $BS --torch-seed $s --out-dir $OUT/b$s > $OUT/b$s.log 2>&1; done
for s in 711 712; do python -u tools/beam_tas.py $CK $MC --torch-seed $s --out-dir $OUT/m$s > $OUT/m$s.log 2>&1; done
python -u tools/beam_tas.py $CK $MC --torch-seed 713 --branch-at t400:3072 --branch-jitter 60 --branch-burst 120 --branch-hold 120 --branch-protect 0 --branch-view-only --out-dir $OUT/m713 > $OUT/m713.log 2>&1
BEST=$(for f in $OUT/m7*.log $OUT/b7*.log; do t=$(grep -ho 'CROSSING in [0-9.]*s' $f | grep -o '[0-9.]*' | head -1); [ -n "$t" ] && echo "$t $(dirname $f)/$(basename $f .log)"; done | sort -n | head -1 | cut -d' ' -f2)
echo "best line: $BEST" > $OUT/best.txt
for s in 721 722 723; do python -u tools/beam_tas.py $CK $BS --torch-seed $s --prefix-line $BEST/beam_best.npz:1174 --out-dir $OUT/pf$s > $OUT/pf$s.log 2>&1; done
echo DONE > $OUT/DONE
