#!/bin/bash
# zlr2b2: one iteration of the perturbation-spine loop ON THE BOX (CPU physics; the recording uses the GPU):
#   bash autoresearch/analysis/zl_perturb_iter.sh <ckpt> <outdir> [threads]
# 1. record the policy's greedy line from the TRUE start with full states;
# 2. zl_perturb_search.py over its last half (roots every 35 ticks, held view offsets, field progress);
# 3. rebuild the best line exactly as <outdir>/spine.npy (clocks zeroed) + spine.json (provenance).
# Generic: the roots are evenly spaced along the policy's own line; nothing names a place.
set -eu
CK=$1; OUT=$2; TH=${3:-12}
cd /root/RL_Surf 2>/dev/null || cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
python3 tools/record_ckpt.py "$CK" --episodes 1 --seed 0 --out "$OUT/inc_greedy.jsonl" \
    --dump-states "$OUT/inc_greedy_states.npz" > "$OUT/record.log" 2>&1
python3 - "$OUT" <<'PY'
import sys, numpy as np
o = sys.argv[1]
s = np.load(f"{o}/inc_greedy_states.npz")["ep_0000"]
np.save(f"{o}/inc_greedy_states.npy", s)
print("recorded", len(s), "ticks")
PY
N=$(python3 -c "import numpy as np; print(len(np.load('$OUT/inc_greedy_states.npy')))")
T0=$((N / 2))
OMP_NUM_THREADS=$TH ZL_CKPT="$CK" ZL_REC_DIR="$OUT" CUDA_VISIBLE_DEVICES=-1 \
    python3 autoresearch/analysis/zl_perturb_search.py "$T0" 35 > "$OUT/search.log" 2>&1
tail -8 "$OUT/search.log"
read R C D <<< "$(python3 -c "import json; b=json.load(open('$OUT/psearch_best.json'))[0]; print(b[1], b[2], b[3])")"
OMP_NUM_THREADS=$TH ZL_CKPT="$CK" ZL_REC_DIR="$OUT" CUDA_VISIBLE_DEVICES=-1 \
    python3 autoresearch/analysis/zl_perturb_spine.py "$R" "$C" "$D" "$OUT/spine_raw.npy" | tee "$OUT/spine.log"
python3 - "$OUT" "$CK" "$R" "$C" "$D" <<'PY'
import sys, json, hashlib, numpy as np
o, ck, r, c, d = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
s = np.load(f"{o}/spine_raw.npy").copy()
for k in ("tick", "stuck_ticks"): s[k] = 0
for k in ("progress", "best_progress"): s[k] = 0.0
np.save(f"{o}/spine.npy", s)
md5 = hashlib.md5(open(f"{o}/spine.npy", "rb").read()).hexdigest()
json.dump({"ckpt": ck, "root_tick": r, "offset_deg": c, "hold_ticks": d, "states": int(len(s)),
           "md5": md5}, open(f"{o}/spine.json", "w"), indent=1)
print("spine", len(s), "states md5", md5)
PY
