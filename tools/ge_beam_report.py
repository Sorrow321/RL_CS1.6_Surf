#!/usr/bin/env python3
"""ge_beam_report.py - a tools/ge_beam.py archive (and its lines) scored like the grader (unr1b2).

Progress on autoresearch/config.json's field (sidistic: goalc_32, start_d from the config), the
furthest sgcp_N curtain any archived state / line state is inside (autoresearch/score.py's
curtains), and which kind of run (policy / random, the knob) found the deepest cells.

    python tools/ge_beam_report.py runs/geA_k0 [runs/geA_k1 ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "autoresearch"))


def main(argv=None) -> int:
    import score as sc
    dirs = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    f, cps = sc.field(), sc.curtains()
    d0 = float(sc.START_D)

    def prog(o):
        d = np.asarray(f.sample(np.asarray(o, np.float64).reshape(-1, 3)), np.float64)
        d = np.where(d < f.reach_max - 0.5 * f.cell, d, np.inf)
        return 100.0 * (d0 - d) / d0

    def max_cp(o):
        o = np.asarray(o, np.float64).reshape(-1, 3)
        m = 0
        for n, lo, hi in cps:
            if np.any(np.all((o >= lo) & (o <= hi), axis=1)):
                m = max(m, n)
        return m

    for dd in dirs:
        out = {"dir": str(dd)}
        a = dd / "archive.npz"
        if a.exists():
            z = np.load(a, allow_pickle=False)
            o = z["state"]["origin"]
            p = prog(o)
            p = np.where(np.isfinite(p), p, -np.inf)
            cur = z["cur"].astype(bool)
            surv = z["surv"]
            mode = z["mode"]
            viable = cur & (surv >= 100) & (mode != 2)
            out.update(nodes=int(len(o)), cells=int(cur.sum()),
                       deepest=round(float(p.max()), 3),
                       deepest_viable=round(float(p[viable].max()), 3) if viable.any() else None,
                       max_cp=max_cp(o),
                       new_policy=int((mode == 0).sum()), new_random=int((mode == 1).sum()),
                       new_burst=int((mode == 3).sum()),
                       deepest_burst=round(float(p[mode == 3].max()), 3)
                       if (mode == 3).any() else None,
                       deepest_policy=round(float(p[mode == 0].max()), 3)
                       if (mode == 0).any() else None,
                       deepest_random=round(float(p[mode == 1].max()), 3)
                       if (mode == 1).any() else None)
            top = np.argsort(-p)[:200]
            out["top200_policy_share"] = round(float(np.mean(mode[top] == 0)), 3)
        lines = sorted(dd.glob("line_*.npy"))
        if lines:
            ln = np.load(lines[-1])
            lp = prog(ln["origin"])
            out.update(last_line=lines[-1].name, line_secs=round(len(ln) / 100.0, 2),
                       line_progress=round(float(np.nanmax(np.where(np.isfinite(lp), lp,
                                                                     -np.inf))), 3),
                       line_cp=max_cp(ln["origin"]))
        st = dd / "status.json"
        if st.exists():
            s = json.loads(st.read_text(encoding="utf-8"))
            out.update(steps=s.get("step"), runs=s.get("runs"),
                       minutes=s.get("elapsed_min"))
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
