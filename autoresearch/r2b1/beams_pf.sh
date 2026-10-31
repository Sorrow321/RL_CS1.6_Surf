#!/bin/bash
# uf2r2b1: ROUTE SEARCH after a first-lap pit exit: replay a Monte Carlo line's own action table open-loop for
# its first PREFIX ticks (beam_tas --prefix-line: the one-lap exit), then a d-ranked population search for the
# rest of the course on the newest weights. The winner is a spliced start-to-finish line, replayed bit-exact.
cd /root/RL_Surf
while [ ! -e runs/beam_mc2/DONE ]; do sleep 5; done
CK=${CK:-runs/beam_pf/ck.pt}
OUT=${OUT:-runs/beam_pf}
mkdir -p $OUT
COMMON="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 7000 --log-every 5 --keep-finishers 8 --gens 3"
i=500
for P in "runs/beam_mc1/m400_60_120/beam_best.npz:1174" "runs/beam_mc1/m400_60_120/beam_best.npz:1043" "runs/beam_mc2/m302/beam_best.npz:1174" "runs/beam_mc2/m301/beam_best.npz:1174" "runs/beam_mc1/m400_60_120/beam_best.npz:1174"; do
  i=$((i+1))
  python -u tools/beam_tas.py $CK $COMMON --torch-seed $i --prefix-line $P --out-dir $OUT/pf$i > $OUT/pf$i.log 2>&1
done
echo DONE > $OUT/DONE
