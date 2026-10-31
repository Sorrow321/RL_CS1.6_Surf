#!/bin/bash
# uf2r2b1: d-ranked beams STARTED at Go-Explore archive states (the record-like early turn at the 60-70% loop,
# found from the policy's own pre-loop states): does the policy fly on from there and finish sooner?
cd /root/RL_Surf
CK=${CK:-runs/beam_real2/ck.pt}
OUT=${OUT:-runs/beam_ge1}
mkdir -p $OUT
python3 - <<'PY'
import numpy as np, sys
sys.path.insert(0, 'tools')
from auto_spine import clean
z = np.load('runs/ge_loop1/archive.npz', allow_pickle=False)
for c in (915679, 269831, 577288, 184627, 619591):
    np.save(f'runs/beam_ge1/s{c}.npy', clean(z['state'][[c]]))
PY
BS="--map maps/surf_unitfarmer2.bsp --envs 4096 --resample-every 25 --score d --greedy-envs 32 --allow-nonfinisher --save-frontier --skip-gate --route-file none --max-ticks 3000 --log-every 5 --keep-finishers 8 --gens 3"
i=900
for c in 915679 269831 577288 184627 619591; do
  i=$((i+1))
  python -u tools/beam_tas.py $CK $BS --torch-seed $i --start-state $OUT/s$c.npy:0 --out-dir $OUT/g$c > $OUT/g$c.log 2>&1
done
echo DONE > $OUT/DONE
