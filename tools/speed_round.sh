#!/bin/bash
# speed_round.sh CKPT OUTDIR [speed_forks args...] - one search+distil round ON THE BOX
# (uf2r2b2, 2026-10-31): tools/speed_forks.py from the TRUE start on CKPT (plain beam + branched
# beams inside the line's longest stalls), then tools/plan_to_bc.py over every crossing beam's
# kept finishers -> OUTDIR/bc.npz (the BC rows, slower lines down-weighted) + OUTDIR/spine.npy
# (+ .keys.npy: the fastest line's per-tick states and held keys). Every line is the policy's
# own proposals (plus held view offsets at the forks) rooted at the map spawn: SELF_STATES.
# Writes OUTDIR/DONE (or OUTDIR/FAILED) at the end. Launch detached:
#   setsid nohup bash tools/speed_round.sh runs_ckpt.pt runs/sf/r1 > runs/sf/r1.log 2>&1 < /dev/null &
set -u
CK="$1"; OUT="$2"; shift 2
cd /root/RL_Surf || exit 1
PY=python3
mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT/FAILED"
MAP=maps/surf_unitfarmer2.bsp
FIELD=maps/surf_unitfarmer2.goalc_32.npz
echo "== $(date -u +%H:%M:%S) speed_forks on $CK"
$PY -u tools/speed_forks.py --ckpt "$CK" --map "$MAP" --field "$FIELD" --out "$OUT" "$@" \
    > "$OUT/speed_forks.log" 2>&1
rc=$?
tail -5 "$OUT/speed_forks.log"
NPZS=$($PY -c "import json,sys;print(' '.join(json.load(open(sys.argv[1])).get('crossing_npz',[])))" "$OUT/summary.json" 2>/dev/null)
if [ -z "$NPZS" ]; then
  echo "== no crossing beam (speed_forks rc $rc)"; touch "$OUT/FAILED"; exit 2
fi
echo "== $(date -u +%H:%M:%S) plan_to_bc over: $NPZS"
$PY -u tools/plan_to_bc.py --plan $NPZS --ckpt "$CK" --out "$OUT/bc.npz" --spine "$OUT/spine.npy" \
    --map "$MAP" --no-value-target --line-weight-decay "${LWD:-20}" --summary-out "$OUT/distil.json" \
    > "$OUT/distil.log" 2>&1
rc=$?
tail -8 "$OUT/distil.log"
if [ $rc -ne 0 ] || [ ! -f "$OUT/bc.npz" ]; then
  echo "== plan_to_bc failed (rc $rc)"; touch "$OUT/FAILED"; exit 3
fi
md5sum "$OUT/bc.npz" "$OUT/spine.npy" "$OUT"/spine.keys.npy 2>/dev/null
$PY -c "import numpy as np,sys;print('spine rows', len(np.load(sys.argv[1])))" "$OUT/spine.npy"
echo "== $(date -u +%H:%M:%S) DONE"
touch "$OUT/DONE"
