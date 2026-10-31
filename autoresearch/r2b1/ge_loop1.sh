#!/bin/bash
# uf2r2b1: Go-Explore (uniform held keys + view-follow offsets, exact chains) rooted at the policy's OWN states
# just before the 60-70% loop (rows 3,195-3,380 of the 37.50 s real-finish line r2_real723 = t 24.5-25.9 s):
# does any chain reach 62-64% progress EARLIER than the line (which loops and gets there ~1.5 s after the turn)?
cd /root/RL_Surf
while [ ! -e runs/beam_real2/DONE ]; do sleep 5; done
python - <<'PY'
import numpy as np
sp = np.load("runs/beam/r2_real723.npy")
np.save("runs/beam/r2_loop_roots.npy", sp[3195:3381:5])
print(len(sp[3195:3381:5]), "roots")
PY
python -u tools/explore_phase1.py --map maps/surf_unitfarmer2.bsp --out runs/ge_loop1 --envs 4096 \
  --cell 64 --cell-energy 32 --cell-heading 1 --view-follow=-90,-60,-30,-15,0,15,30,60,90 \
  --select-energy 3 --act-every 4 --decisions 100 --exact-chains --roots-spine runs/beam/r2_loop_roots.npy \
  --roots-keep-all --maxvel 3500 --tick-ms 8 --max-minutes 6 --goals 1 --threads 12 --seed 9 > runs/ge_loop1.log 2>&1
echo DONE > runs/ge_loop1/DONE
