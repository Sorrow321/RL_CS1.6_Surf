"""finish_rate.py - how steadily a run finishes once it has finished (azr1b2, slot 2): from its
first finishing eval on, the share of evals with at least one finish, the mean finishes per eval,
the best and the median best finish time, and the evals that fell back below the finish. Scores
each greedy eval with autoresearch/score.py (the round's own scorer); N caps the evals counted so
two runs compare over the same number of evals.

    python -I autoresearch/analysis/finish_rate.py [--n N] <name>=<run dir> [<name>=<run dir> ...]
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score  # noqa: E402  (autoresearch/score.py)


def evals(d: Path):
    fs = [f for f in d.glob("traj_*.jsonl") if re.fullmatch(r"traj_\d+\.jsonl", f.name)]
    return sorted(fs, key=lambda f: int(f.stem.split("_")[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="count at most N evals from the first finish")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    for spec in a.runs:
        name, d = spec.split("=", 1)
        rows, started = [], False
        for f in evals(Path(d)):
            s = score.score_file(f)
            if s.get("score") is None:
                continue
            if not started and s["finishes"] == 0:
                continue
            started = True
            rows.append((int(f.stem.split("_")[1]), s))
            if a.n and len(rows) >= a.n:
                break
        if not rows:
            print(f"{name}: no finishing eval")
            continue
        n = len(rows)
        fin = [s["finishes"] for _, s in rows]
        eps = [s["episodes"] for _, s in rows]
        times = [s["best_finish_s"] for _, s in rows if s["best_finish_s"]]
        times.sort()
        med = times[len(times) // 2] if times else None
        drops = [(st, s["best_progress_pct"], s["max_cp"]) for st, s in rows if s["finishes"] == 0]
        best = max(rows, key=lambda r: r[1]["score"])
        print(f"{name}: {n} evals from the first finish ({rows[0][0]:,} .. {rows[-1][0]:,} steps): "
              f"finish in {sum(1 for x in fin if x)}/{n} = {100.0 * sum(1 for x in fin if x) / n:.0f}%,"
              f" mean {sum(fin) / n:.2f}/{sum(eps) / n:.0f} finishes, max {max(fin)}, "
              f"best time {times[0]:.2f} s, median best {med:.2f} s; best score "
              f"{best[1]['score']} at {best[0]:,}")
        if drops:
            print("   no finish at: " + ", ".join(f"{st / 1e9:.3f}B ({p:.1f}%, CP{c})"
                                               for st, p, c in drops))


if __name__ == "__main__":
    main()
