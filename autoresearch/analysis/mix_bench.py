"""mix_bench.py - side-by-side of paired MCTS bench arms (azr1b2, slot 2): run 5's fixed knob
('log' bookkeeping) vs the rule ('sib'). Reads each arm's beam_tas summary.json; prints depth
(banked and raw), coverage, waves, the rule's mean / final shares and who won the expansions.

    python -I autoresearch/analysis/mix_bench.py <bench_dir> <tag> [<tag> ...]
"""
import json
import sys
from pathlib import Path


def row(p: Path) -> dict:
    s = json.loads((p / "summary.json").read_text())
    return s


def main():
    base = Path(sys.argv[1])
    tags = sys.argv[2:]
    keys = ("best_pct", "raw_best_pct", "greedy_pct", "cells", "waves", "nodes", "best_depth",
            "max_depth", "finishes", "search_wall_s", "env_steps_per_s", "roots", "greedy_end")
    for t in tags:
        for arm in ("r5", "sib"):
            p = base / f"{t}_{arm}"
            if not (p / "summary.json").exists():
                print(f"{t}_{arm}: no summary yet")
                continue
            s = row(p)
            print(f"== {t}_{arm}: " + ", ".join(f"{k} {s.get(k)}" for k in keys))
            if "mix_rule" in s:
                print(f"   mix {s['mix_rule']}: mean {s['mix_share_mean']}, final "
                      f"{s['mix_share_final']}")
                print(f"   wins {s['mix_wins']} (on value {s['mix_wins_value']}, on cells "
                      f"{s['mix_wins_cells']})")
                print(f"   wins/child {s['mix_wins_per_child']}, new cells/child "
                      f"{s['mix_new_cells_per_child']}, alive at macro end {s['mix_alive_end']}")
                print(f"   best line's macros: {s.get('best_path_kinds')}")
                traj = s.get("mix_share_by_wave") or []
                if traj:
                    pick = traj[:: max(1, len(traj) // 12)]
                    print("   shares by wave (wave: P/E/R): " + "  ".join(
                        f"{w[0]}: {w[1]:.2f}/{w[2]:.2f}/{w[3]:.2f}" for w in pick))
            print(f"   best by wave: {s.get('best_by_wave')}")


if __name__ == "__main__":
    main()
