#!/bin/bash
# zlr2b4: deploy + launch the CEM-line spine run + snapcopy, once box.py rent has succeeded.
# Granted before 14:00 UTC: the full 97.8% line (backward curriculum from its end, milestones at
# the GE line, the launch apex and the first air line). Later: the launch-only 79% spine, whose
# curriculum reaches the launch at once (a short window cannot wait for the far stages).
set -e
cd /home/steam/ar_loop/wt/b4
export AR_HOME=/home/steam/ar_loop AR_AGENT=zlr2b4 RLSURF_FLEET_DIR=/home/steam/ar/fleet PYTHONIOENCODING=utf-8
B=autoresearch/box.py
python $B status --agent zlr2b4
HOUR=$(date -u +%H)
if [ "$HOUR" -lt 14 ]; then
  SPINE=autoresearch/spines/zlr2b4_cem978_spine.npy; MILES=5859,1334,422; RUN=zlr2b4_s1
  WHAT="to its own state at 169.3 s on the 3rd ramp, then tools/cem_launch.py physics-only CEM controls (a 0.77 s launch from that exact state, then two stages of air control onto a higher ramp and up it) replayed in the core to 97.81%"
else
  SPINE=autoresearch/spines/zlr2b4_cem164_spine.npy; MILES=5859; RUN=zlr2b4_l1
  WHAT="to its own state at 164.4 s on the 3rd ramp, then a tools/cem_launch.py physics-only CEM control (2.5 s from that exact state) replayed in the core to the launch apex (77.68%)"
fi
echo "spine $SPINE milestones $MILES run $RUN"
python $B deploy --agent zlr2b4 --branch ar/zlr2b4 --ckpt /home/steam/ar_loop/incumbent/ckpt.pt
python $B launch --agent zlr2b4 --run $RUN --steps 3e9 --resume \
  --self-states "zlr2b4 CEM line spine: the incumbent's (zlr1b4 E13 ckpt_4032299008) OWN greedy line from the true start (beam_tas --frontier-last, CPU) $WHAT; view offsets in atan(30/|v|) units + side/forward keys; no demo, no record data" \
  -- --demo-file $SPINE --demo-cooldown 5 --demo-grow-max 40 \
     --demo-milestones $MILES --fail-pen 3 --ckpt-every 2e8
python $B ssh --agent zlr2b4 -- "cd /root/RL_Surf && P=\$(cat runs/$RUN.pid) && nohup python3 tools/snapcopy.py runs/$RUN --pid \$P > runs/${RUN}_snapcopy.txt 2>&1 < /dev/null & echo snapcopy started"
echo BOX_GO_DONE $RUN
