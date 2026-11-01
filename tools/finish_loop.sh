#!/bin/bash
# finish_loop.sh RUN POOL [CYCLE_MIN] [CEM_THREADS] [SEED_LINES_GLOB]
#
# A DAgger-style finish loop beside a training run (zlr3b3, 2026-11-01): every CYCLE_MIN minutes
#   1. the trainer's newest checkpoint records 6 GREEDY episodes from the true start (record_ckpt
#      --dump-states, capped at 210 s: a recording has no eval patience) - the policy's own states;
#   2. tools/frontier_states.py cuts its frontier (the states around its last contact-record gain);
#   3. tools/finish_roots.py picks the contact states nearest to the roots finishes were planned
#      from before (the seed lines on the first cycle);
#   4. tools/cem_finish.py plans goal entries from them (physics only, CEM_THREADS threads);
#   5. tools/finish_pool.py rebuilds POOL = that frontier + every finishing line so far at 40%,
#      written atomically, so the trainer's --spawn-states (re-read when the file changes) follows.
# It exits when runs/RUN.pid is gone. Logs: runs/fl_RUN/loop.log. Nothing here reads a place.
set -u
RUN=$1
POOL=$2
CYC=${3:-20}
THR=${4:-8}
SEED=${5:-autoresearch/spines/zlr3b3_cem/*_fin_*.npz}
D=runs/fl_$RUN
mkdir -p "$D/lines" runs/reach
for f in $SEED; do [ -e "$f" ] && cp -n "$f" "$D/lines/seed_$(basename "$f")"; done
CKPT=runs/$RUN/ckpt_latest.pt
FIELD=maps/surf_za_loop.goalc_32.npz
i=0
echo "$(date -u +%FT%TZ) finish_loop: run $RUN pool $POOL cycle ${CYC} min, $THR CEM threads, $(ls "$D/lines" | wc -l) seed lines" >> "$D/loop.log"
while kill -0 "$(cat runs/$RUN.pid 2>/dev/null)" 2>/dev/null; do
  i=$((i + 1))
  t0=$(date +%s)
  if [ ! -s "$CKPT" ]; then sleep 60; continue; fi
  cp "$CKPT" "$D/ck_$i.pt"
  python3 tools/record_ckpt.py "$D/ck_$i.pt" --episodes 6 --ep-ticks 30000 \
      --out "$D/rec_$i.jsonl" --dump-states "$D/states_$i.npz" > "$D/rec_$i.log" 2>&1
  python3 tools/frontier_states.py "$D/states_$i.npz" --field "$FIELD" --tick-ms 7 \
      --out "$D/front_$i.npy" > "$D/front_$i.log" 2>&1
  if [ ! -s "$D/front_$i.npy" ]; then
    echo "$(date -u +%FT%TZ) cycle $i: no frontier pool (see $D/front_$i.log)" >> "$D/loop.log"
    rm -f "$D/ck_$i.pt"; sleep 120; continue
  fi
  ROOTS=$(python3 tools/finish_roots.py "$D/front_$i.npy" --lines "$D/lines/*.npz" --k 12 2>> "$D/loop.log")
  OMP_NUM_THREADS=$THR python3 tools/cem_finish.py "$D/ck_$i.pt" "$D/front_$i.npy" --roots "$ROOTS" \
      --out "$D/cem_$i" --pop 4096 --horizon 2.5 --iters 40 --elite 64 --root-fin 2 --max-fin 40 \
      --seed "$i" --reach-cache runs/reach/za_loop.npy > "$D/cem_$i.log" 2>&1
  for f in "$D/cem_$i"/fin_*.npz; do [ -e "$f" ] && cp "$f" "$D/lines/c${i}_$(basename "$f")"; done
  python3 tools/finish_pool.py --base "$D/front_$i.npy" --lines "$D/lines/*.npz" --secs 3 --frac 0.4 \
      --out "$POOL" > "$D/pool_$i.log" 2>&1
  rm -f "$D/ck_$i.pt"
  echo "$(date -u +%FT%TZ) cycle $i: roots $ROOTS | $(grep -c FINISH "$D/cem_$i.log") new finishes, $(ls "$D/lines" | wc -l) lines | $(tail -1 "$D/pool_$i.log")" >> "$D/loop.log"
  left=$(( CYC * 60 - ($(date +%s) - t0) ))
  [ "$left" -gt 0 ] && sleep "$left"
done
echo "$(date -u +%FT%TZ) finish_loop: trainer gone, exiting" >> "$D/loop.log"
