"""run_watch.py <run_dir> [n] - the columns that say whether an exploration arm moves the frontier:
per progress.csv row the step, fps, eval progress, the unstuck T and its eval-frontier best, the
novel ceiling (deepest REAL episode; spawn-state episodes count as real), and the share of novel
rows; then the last greedy evals' best/median progress on the goal field (score.py's measure).
Runs on the box (cd /root/RL_Surf) or locally on a pulled run."""
import csv
import sys
from pathlib import Path

D0 = 97768.0
run = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rows = list(csv.DictReader(open(run / "progress.csv", encoding="utf-8")))
cols = [("time/total_timesteps", "step", "{:,.0f}"), ("time/fps", "fps", "{:,.0f}"),
        ("race/eval_progress", "evalp", "{:,.0f}"), ("unstuck/T", "T", "{:.2f}"),
        ("unstuck/best", "best_u", "{:,.0f}"), ("novel/ceiling", "ceil_u", "{:,.0f}"),
        ("novel/share", "nov_sh", "{:.2f}"), ("rollout/ep_len_mean", "eplen", "{:.0f}")]
print("  ".join(c[1].rjust(13) for c in cols) + "   ceil%")
for r in rows[-n:]:
    out = []
    for k, _, f in cols:
        v = r.get(k, "")
        try:
            out.append(f.format(float(v)).rjust(13))
        except ValueError:
            out.append("-".rjust(13))
    try:
        out.append("  %.2f%%" % (100.0 * float(r["novel/ceiling"]) / D0))
    except (KeyError, ValueError):
        pass
    print("  ".join(out))
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import score
    evs = sorted(run.glob("traj_*.jsonl"))
    import re
    evs = [p for p in evs if re.fullmatch(r"traj_\d+\.jsonl", p.name)][-4:]
    for p in evs:
        s = score.score_file(p)
        print(p.name, "score", s.get("score"), "best", s.get("best_progress_pct"),
              "median", s.get("median_progress_pct"), "CP", s.get("max_cp"))
except Exception as e:      # the box may lack the field; the csv part still printed
    print("eval scoring skipped:", e)
