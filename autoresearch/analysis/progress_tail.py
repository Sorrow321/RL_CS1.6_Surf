"""progress_tail.py - the last N rows of a run's progress.csv, the columns worth watching.

    python3 autoresearch/analysis/progress_tail.py runs/<run>/progress.csv [N]
"""
import csv
import sys

COLS = [("time/total_timesteps", "steps_M", 1e-6, "{:8.1f}"), ("time/fps", "fps_k", 1e-3, "{:6.0f}"),
        ("rollout/ep_rew_mean", "rew", 1, "{:7.2f}"), ("rollout/ep_len_mean", "len", 1, "{:6.0f}"),
        ("race/eval_progress", "evalprog", 1, "{:8.0f}"), ("race/success_rate", "win", 1, "{:5.2f}"),
        ("race/stall_frac", "stall", 1, "{:5.2f}"), ("unstuck/T", "T", 1, "{:5.2f}"),
        ("dip/max_survived_depth", "dipmax", 1, "{:6.0f}"), ("dip/p90_survived_depth", "dip90", 1, "{:6.0f}"),
        ("dip/survived_per_ep", "dip/ep", 1, "{:6.3f}"), ("novel/visited", "visited", 1, "{:7.0f}"),
        ("fpert/best_prog", "fpbest", 1, "{:7.0f}")]
rows = list(csv.DictReader(open(sys.argv[1])))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
print(" ".join(f"{c[1]:>8s}" for c in COLS))
for r in rows[-n:]:
    out = []
    for k, nm, sc, fmt in COLS:
        v = r.get(k, "")
        try:
            out.append(f"{fmt.format(float(v) * sc):>8s}")
        except ValueError:
            out.append(f"{'-':>8s}")
    print(" ".join(out))
