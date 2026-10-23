"""score.py - the deterministic grader J of fork-and-flush (arXiv 2610.07447), on config.map.

One number per GREEDY eval from the TRUE map start - the trajectories the trainer writes every
--record-every steps (runs/<run>/traj_<step>.jsonl, 9 episodes). Higher is better:

    score = best_progress_pct + 10 * finish_frac + (10 * (1 - best_finish_s / 300) if any finish)

* best_progress_pct: max over episodes of 100 * (d0 - min d) / d0 on config.field, the goal
  graph WITHOUT corner cutting (maps/<map>.goalc_32.npz, the field the race reward shapes on
  since round 5; before round 7 the grader read goal_32.npz, which leaks through the finish
  funnel's slanted wall and scored a dead end outside it above the way over the rim), d0 = the
  field at the episode's first row. A finished episode counts 100. This is the honest frontier
  (CLAUDE.md section 3: corridor MAX and finishes), not race/eval_progress's mean.
* only the first config.score_episodes episodes count (header episode index < N; 9 = what
  the trainer records): the score is a max over episodes, so a larger eval would only buy
  more lottery tickets.
* finish_frac: episodes that ended "done" (inside the finish) / episodes.
* the time term only exists once something finishes: faster is better, 300 s = the cap.

Validation - an eval that fails it scores None and is never selected:
* the header names the map;
* every episode starts at the map start: its first-row d within 2% of config.start_d (the
  geodesic at the spawn), so a reservoir or novel spawn can never be passed off as an eval;
* at least 5 episodes.

Also reported per eval: the furthest checkpoint curtain (sgcp_N) any episode entered, the
median progress, and the per-episode ends.

    python autoresearch/score.py runs/x/traj_0755499008.jsonl [more ...]
    python autoresearch/score.py --best runs/x/                # the best eval in a directory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))
CFG = json.loads((REPO / "autoresearch" / "config.json").read_text(encoding="utf-8"))
START_D = float(CFG.get("start_d", 97768.0))
SCORE_EPISODES = int(CFG.get("score_episodes") or 0)
EP_CAP_S = 300.0
_FIELD = None
_CPS = None


def field():
    global _FIELD
    if _FIELD is None:
        from surfgym.goalfield import GoalField
        z = np.load(REPO / CFG["field"], allow_pickle=False)
        _FIELD = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                           float(z["cell"]), float(z["reach_max"]))
    return _FIELD


def curtains():
    """[(n, lo, hi)] of the map's sgcp_N checkpoint curtains, padded by the player hull."""
    global _CPS
    if _CPS is None:
        from surfgym.zones import parse_bsp
        ents, bb = parse_bsp(str(REPO / CFG["map"]))
        _CPS = []
        for e in ents:
            tn = e.get("targetname", "")
            if tn.startswith("sgcp") and e.get("model", "").startswith("*"):
                m = int(e["model"][1:])
                _CPS.append((int(tn.split("_")[-1]),
                             np.asarray(bb[m][0]) - [16, 16, 36],
                             np.asarray(bb[m][1]) + [16, 16, 36]))
        _CPS.sort()
    return _CPS


def episodes(path: Path):
    """[(header, rows (n, k) float64, trailer)] - the record.py format: a header line, the
    rows, a trailer {"end": fail|done|trunc, ...}."""
    eps, hdr, cur = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None:
                eps.append((hdr, np.asarray(cur, np.float64), d))
                hdr, cur = None, []
            else:
                hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return eps


def first_episodes(eps, n: int):
    """The episodes the score counts: those whose header episode index is below n, so the
    choice does not depend on the order episodes are written in, or the first n in file order
    when a header has no index. n <= 0 keeps them all."""
    if n <= 0:
        return eps
    if all(isinstance(h.get("episode"), int) for h, _, _ in eps):
        return [e for e in eps if e[0]["episode"] < n]
    return eps[:n]


def score_file(path: Path) -> dict:
    path = Path(path)
    out = {"file": str(path), "score": None}
    try:
        eps = first_episodes(episodes(path), SCORE_EPISODES)
    except (OSError, ValueError) as e:
        out["error"] = f"unreadable: {e}"
        return out
    want_map = Path(CFG["map"]).stem
    if len(eps) < 5:
        out["error"] = f"{len(eps)} episodes (< 5)"
        return out
    f, cps = field(), curtains()
    prog, cp_max, fins, ends = [], 0, [], []
    for hdr, r, tr in eps:
        if hdr.get("map") != want_map:
            out["error"] = f"map {hdr.get('map')!r} is not {want_map!r}"
            return out
        if not len(r):
            out["error"] = "an empty episode"
            return out
        p = r[:, 1:4]
        d = f.sample(p).astype(np.float64)
        d0 = float(d[0])
        # the unreachable sentinel sits just above reach_max, which on sidistic is within 1% of
        # the start distance: an off-field first row must not pass for the map start
        if d0 >= f.reach_max - 0.5 * f.cell or abs(d0 - START_D) > 0.02 * START_D:
            out["error"] = (f"an episode starts at d {d0:,.0f}, not the map start "
                            f"({START_D:,.0f} +- 2%): not an eval from the true start")
            return out
        ok = d < f.reach_max - 0.5 * f.cell
        dmin = float(d[ok].min()) if ok.any() else d0
        done = tr.get("end") == "done"
        prog.append(100.0 if done else max(0.0, 100.0 * (d0 - dmin) / d0))
        for n, lo, hi in cps:
            if np.any(np.all((p >= lo) & (p <= hi), axis=1)):
                cp_max = max(cp_max, n)
        if done:
            fins.append(len(r) * float(hdr.get("tick_ms", 10.0)) / 1000.0)
        ends.append(tr.get("end"))
    best = max(prog)
    frac = len(fins) / len(eps)
    s = best + 10.0 * frac
    if fins:
        s += 10.0 * max(0.0, 1.0 - min(fins) / EP_CAP_S)
    out.update({"score": round(s, 3), "best_progress_pct": round(best, 2),
                "median_progress_pct": round(float(np.median(prog)), 2),
                "finishes": len(fins), "episodes": len(eps),
                "best_finish_s": round(min(fins), 2) if fins else None,
                "max_cp": cp_max, "ends": ends})
    return out


def best_in(paths) -> dict:
    files = []
    for p in paths:
        p = Path(p)
        files += (sorted(p.glob("traj_*.jsonl")) if p.is_dir() else [p])
    # greedy evals only: traj_<step>.jsonl (the record gate's traj_<step>_rec_* are not evals)
    import re
    files = [x for x in files if re.fullmatch(r"traj_\d+\.jsonl", x.name)]
    scored = [score_file(x) for x in files]
    good = [s for s in scored if s["score"] is not None]
    best = max(good, key=lambda s: s["score"]) if good else None
    return {"best": best, "n_evals": len(scored), "n_valid": len(good),
            "rejected": [s for s in scored if s["score"] is None][:5]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--best", action="store_true", help="only the best eval, as JSON")
    a = ap.parse_args(argv)
    if a.best:
        print(json.dumps(best_in(a.paths), indent=1))
        return
    for p in a.paths:
        pp = Path(p)
        for x in (sorted(pp.glob("traj_*.jsonl")) if pp.is_dir() else [pp]):
            s = score_file(x)
            if s["score"] is None:
                print(f"{x.name}: REJECTED - {s.get('error')}")
            else:
                print(f"{x.name}: score {s['score']:7.3f}  best {s['best_progress_pct']:5.1f}%  "
                      f"median {s['median_progress_pct']:5.1f}%  CP{s['max_cp']}  "
                      f"finishes {s['finishes']}/{s['episodes']}"
                      + (f" best {s['best_finish_s']:.2f} s" if s['finishes'] else ""))


if __name__ == "__main__":
    main()
