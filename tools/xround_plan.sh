#!/usr/bin/env bash
# xround_plan.sh CKPT OUTDIR WAVES SEED0 - one hand-run expert-iteration PLAN + DISTIL step (pbfr1b2, 2026-10-29):
# WAVES tools/beam_tas.py waves from the map's true start on CKPT (the policy's own proposals, d-ranked cloning,
# a 128-env greedy floor, 16 finishers kept; torch seed SEED0+w, spawn seed SEED0+w+1), then tools/plan_to_bc.py
# per wave (--no-value-target), merged into OUTDIR/bc.npz, and the waves' best lines concatenated into
# OUTDIR/spine.npy (+ .keys.npy). The pbx expert loop's plan/distil arguments (tools/exit_local.py), run by hand
# because exit_local launches train_fast itself (PROTOCOL: training goes through box.py launch).
set -euo pipefail
CKPT=$1; OUT=$2; WAVES=${3:-3}; SEED0=${4:-0}
MAP=maps/surf_princessburglar_b26.bsp
mkdir -p "$OUT"
cp "$CKPT" "$OUT/base.pt"
for w in $(seq 0 $((WAVES - 1))); do
  python3 -u tools/beam_tas.py "$OUT/base.pt" --map $MAP --envs 4096 --greedy-envs 128 --score d \
    --resample-every 25 --greedy-eps 1 --max-ticks 9800 --keep-finishers 16 --route-file none \
    --torch-seed $((SEED0 + w)) --seed $((SEED0 + w + 1)) --out-dir "$OUT/w$w" --allow-nonfinisher \
    > "$OUT/w$w.log" 2>&1 || true
  tail -1 "$OUT/w$w.log"
done
ok=()
for w in $(seq 0 $((WAVES - 1))); do
  [ -f "$OUT/w$w/beam_best.npz" ] || continue
  if python3 -u tools/plan_to_bc.py --plan "$OUT/w$w/beam_best.npz" --ckpt "$OUT/base.pt" --out "$OUT/bc_$w.npz" \
       --spine "$OUT/spine_$w.npy" --map $MAP --no-value-target > "$OUT/distil_$w.log" 2>&1; then ok+=($w); fi
done
python3 - "$OUT" "${ok[@]}" <<'PY'
import sys, numpy as np
sys.path.insert(0, 'python')
from surfgym.bc import merge_bc_files
out, ws = sys.argv[1], sys.argv[2:]
merge_bc_files([f'{out}/bc_{w}.npz' for w in ws], f'{out}/bc.npz')
S = [np.load(f'{out}/spine_{w}.npy') for w in ws]
K = [np.load(f'{out}/spine_{w}.keys.npy') for w in ws]
np.save(f'{out}/spine.npy', np.concatenate(S)); np.save(f'{out}/spine.keys.npy', np.concatenate(K))
print(f'distilled waves {ws}: spine {sum(len(s) for s in S)} states')
PY
md5sum "$OUT/bc.npz" "$OUT/spine.npy" "$OUT/spine.keys.npy"
