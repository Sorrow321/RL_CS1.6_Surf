"""mix_searches.py - one row per search of a run (azr1b2, slot 2): when it started (from the launch
log), its depth (banked / raw, the greedy root line's), coverage, the child-mix rule's mean and final
shares, wins per child of each kind, the kinds along its best line, and what the trainer did with
the line (adopted / stale / none). Reads a pulled copy of runs/<run>/auto/k*/ (summary.json,
auto_spine.log) and the launch log.

    python -I autoresearch/analysis/mix_searches.py <pulled auto dir> [<launch log>]
"""
import json
import re
import sys
from pathlib import Path


def outcomes(launch: Path) -> dict:
    """search k -> (start step, outcome text) from the trainer's launch log."""
    out = {}
    if not launch or not launch.exists():
        return out
    for ln in launch.read_text(errors="replace").splitlines():
        m = re.match(r"\[\s*([\d,]+)\] auto-spine search (\d+): (.*)", ln)
        if not m:
            continue
        step, k, rest = int(m.group(1).replace(",", "")), int(m.group(2)), m.group(3)
        if rest.startswith("/") or rest.startswith("running"):
            out.setdefault(k, [step, ""])
        elif "NEW SPINE" in rest:
            e = re.search(r"end d ([\d,]*\d)", rest)
            out.setdefault(k, [step, ""])[1] = f"ADOPTED at {step:,} (end d {e.group(1) if e else '?'})"
        elif "STALE" in rest or "stale" in rest:
            out.setdefault(k, [step, ""])[1] = "stale"
        elif "no spine" in rest.lower() or "no line" in rest.lower() or "nothing" in rest.lower():
            out.setdefault(k, [step, ""])[1] = "no line"
    return out


def main():
    base = Path(sys.argv[1])
    launch = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    oc = outcomes(launch)
    print(f"{'k':>3} {'step':>14} {'waves':>5} {'cells':>6} {'greedy':>7} {'best':>7} {'raw':>7} "
          f"{'mean P/E/R':>15} {'final P/E/R':>15} {'wins/child G P E R':>27}  line kinds -> outcome")
    for d in sorted(base.glob("k*")):
        f = d / "beam" / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        k = int(d.name[1:])
        m, fi = s.get("mix_share_mean"), s.get("mix_share_final")
        w = s.get("mix_wins_per_child") or {}
        pk = s.get("best_path_kinds", "")
        cnt = "".join(f"{c}{pk.count(c)}" for c in "GPER" if pk.count(c))
        st, o = oc.get(k, [0, ""])
        print(f"{k:>3} {st:>14,} {s.get('waves', 0):>5} {s.get('cells', 0):>6} "
              f"{s.get('greedy_pct', 0):>6.2f}% {s.get('best_pct', 0):>6.2f}% "
              f"{(s.get('raw_best_pct') or 0):>6.2f}% "
              + (f"{m['P']:.2f}/{m['E']:.2f}/{m['R']:.2f}".rjust(15)
                 + f"{fi['P']:.2f}/{fi['E']:.2f}/{fi['R']:.2f}".rjust(16) if m else " " * 31)
              + " " + " ".join(f"{w.get(c, 0):.4f}" for c in "GPER").rjust(27)
              + f"  {cnt} -> {o}")


if __name__ == "__main__":
    main()
