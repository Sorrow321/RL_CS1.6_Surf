"""Bench a checkpoint from EXACT rows of a spine (pbr7b4, 2026-10-27; a diagnostic, never training):
run ON THE BOX from /root/RL_Surf, record_ckpt greedy episodes starting from spine rows [lo, hi),
then report per episode the best progress and the energy height at two levels, and how many cross
the second level with at least a given energy. Before / after a contact it tells whether the policy
lacks the contact or what comes after it (pbr7b4: 0/12 vs 12/12 around e02's 71.6% bounce, and all
12 energetic ones still stopped at 74.2% - the passage technique was the missing skill).

    python autoresearch/analysis/pb_bench_spine.py CKPT SPINE LO HI OUT_DIR [N] [L1,L2] [HMIN]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ck, sp, lo, hi, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
n = int(sys.argv[6]) if len(sys.argv) > 6 else 12
l1, l2 = (float(x) for x in (sys.argv[7] if len(sys.argv) > 7 else "71.8,73.44").split(","))
hmin = float(sys.argv[8]) if len(sys.argv) > 8 else -1e9
out.mkdir(parents=True, exist_ok=True)
S = np.load(sp)
win = out / f"rows_{lo}_{hi}.npy"
np.save(win, S[lo:hi])
traj = out / f"bench_{lo}_{hi}.jsonl"
subprocess.run([sys.executable, "tools/record_ckpt.py", ck, "--map", "maps/surf_princessburglar_b26.bsp",
                "--spawn-states", str(win), "--episodes", str(n), "--out", str(traj), "--eval-stall", "0",
                "--maxvel", "3500"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import score  # noqa: E402
F = score.field()
ok, rows = 0, []
for h, r, tr in score.episodes(traj):
    p, v = r[:, 1:4], r[:, 4:7]
    d = F.sample(p).astype(np.float64)
    prog = np.maximum.accumulate(np.where(d < F.reach_max - 16, 100 * (57149 - d) / 57149, -1))
    hh = p[:, 2] + (v * v).sum(1) / 1600

    def at(L):
        i = np.flatnonzero(prog >= L)
        return float(hh[i[0]]) if len(i) else float("nan")
    a1, a2 = at(l1), at(l2)
    good = a2 == a2 and a2 >= hmin
    ok += good
    rows.append(f"{prog.max():.2f}%/{a1:.0f}/{a2:.0f}{'*' if good else ''}")
print(json.dumps({"rows": [lo, hi], "episodes": len(rows), "cross_with_energy": ok, "levels": [l1, l2],
                  "h_min": hmin}))
print(" ".join(rows))
