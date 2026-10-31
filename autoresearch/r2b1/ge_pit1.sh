#!/bin/bash
# uf2r2b1: Go-Explore (uniform held keys, view-follow offsets, exact chains) rooted at the policy's OWN
# first-lap pit states (rows 450-850 of the r2s2 beam line = t 3.45-6.5 s): does any chain climb out of
# the start pit on the FIRST lap? (the line itself leaves it on the second, ~11.3 s)
cd /root/RL_Surf
while [ ! -e runs/beam_r2d/DONE ]; do sleep 5; done
python - <<'PY'
import numpy as np
sp = np.load("runs/beam/r2s2.npy")
np.save("runs/beam/r2s2_pitroots.npy", sp[450:851:10])
print(len(sp[450:851:10]), "roots")
PY
python -u tools/explore_phase1.py --map maps/surf_unitfarmer2.bsp --out runs/ge_pit1 --envs 4096 \
  --cell 64 --cell-energy 32 --cell-heading 1 --view-follow=-90,-60,-30,-15,0,15,30,60,90 \
  --select-energy 3 --act-every 4 --decisions 100 --exact-chains --roots-spine runs/beam/r2s2_pitroots.npy \
  --roots-keep-all --maxvel 3500 --tick-ms 8 --max-minutes 8 --goals 1 --threads 12 --seed 5 > runs/ge_pit1.log 2>&1
echo DONE > runs/ge_pit1/DONE
