"""search_outcomes.py - per search, from a trainer launch log (azr1b2, slot 2): start step, the greedy d
the trainer passed, and the outcome (ADOPT / stale / none) with the line's end d (or the beam's best d).

    python -I autoresearch/analysis/search_outcomes.py <launch log> [...]
"""
import re, sys, json
for path in sys.argv[1:]:
    rows = {}
    for ln in open(path, errors="replace"):
        m = re.match(r"\[\s*([\d,]+)\] auto-spine search (\d+): (.*)", ln)
        if not m:
            continue
        step, k, rest = int(m.group(1).replace(",", "")), int(m.group(2)), m.group(3)
        r = rows.setdefault(k, {"k": k})
        if rest.startswith("/"):
            g = re.search(r"--greedy-d ([\d.]+)", rest)
            r["start"] = step
            r["greedy_d"] = float(g.group(1)) if g else None
        elif "NEW SPINE" in rest:
            e = re.search(r"end d ([\d,]+)", rest)
            r["out"] = "ADOPT"; r["end_d"] = int(e.group(1).replace(",", ""))
        elif "STALE" in rest:
            e = re.search(r"end d ([\d,]+)", rest)
            r["out"] = "stale"; r["end_d"] = int(e.group(1).replace(",", "")) if e else None
        elif "NO new spine" in rest:
            j = re.search(r"\((\{.*\})\)", rest)
            try:
                d = json.loads(j.group(1)); r["out"] = "none"; r["end_d"] = round(d.get("beam_best_d") or 0)
            except Exception:
                r["out"] = "none"
    print(path)
    n = len(rows); ad = sum(1 for r in rows.values() if r.get("out") == "ADOPT")
    for r in sorted(rows.values(), key=lambda r: r["k"]):
        print(f"  {r['k']:>3} {r.get('start', 0):>15,} greedy_d {r.get('greedy_d') or 0:>9,.0f} -> "
              f"{r.get('out', '?'):>6} end d {r.get('end_d') or 0:>7,}")
    print(f"  searches {n}, adopted {ad}")
