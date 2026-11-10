"""gain_by_root.py - per run, how far each MCTS search got from its own greedy root (azr1b2, slot 2):
gain = best banked % - greedy root % (field progress, summary.json), binned by where the root died.

    python -I autoresearch/analysis/gain_by_root.py <name>=<dir holding k*/beam/summary.json> [...]
"""
import json, sys
from pathlib import Path
import numpy as np
bins = [(0, 22, "root < 22% (first ramp..pit)"), (22, 35, "root 22-35% (pit..CP3)"),
        (35, 70, "root 35-70%"), (70, 101, "root >= 70%")]
for spec in sys.argv[1:]:
    name, d = spec.split("=")
    rows = []
    for f in sorted(Path(d).glob("k*/beam/summary.json")):
        s = json.loads(f.read_text())
        g, b, raw = s.get("greedy_pct", 0), s.get("best_pct", 0), s.get("raw_best_pct") or 0
        rows.append((int(f.parts[-3][1:]), g, b, raw, s.get("cells", 0), s.get("waves", 0)))
    print(f"== {name}: {len(rows)} searches")
    for lo, hi, lab in bins:
        sel = [r for r in rows if lo <= r[1] < hi]
        if not sel:
            continue
        gain = np.array([r[2] - r[1] for r in sel])
        print(f"   {lab:<30} n {len(sel):>2}  gain mean {gain.mean():6.2f}  median {np.median(gain):6.2f}  "
              f">=10 pts {int((gain >= 10).sum()):>2}  max {gain.max():6.2f}  cells med {int(np.median([r[4] for r in sel])):>6}")
