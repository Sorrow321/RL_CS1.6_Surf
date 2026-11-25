"""wrclock.py - a finish on the RECORD clock, and the comparison with the human world record: one
rule for the grader (autoresearch/score.py, autoresearch/wr_metric.py) and the trainer's eval
(train_fast.py --wr-file).

A human record (speedrun.eu) is timed from the START trigger to the finish trigger: the timer
restarts on every touch of the start zone, so it runs from the LAST touch before the finish. The
grader and the trainer time from spawn, which is longer by however long the run takes to leave the
start zone (0.96 s on cannonball, docs/research-results.md). Here a finished episode's record time
is (n - i_s) ticks, i_s the last recorded row whose origin is inside the start trigger and n the
episode's rows (the finish fires on the move after row n - 1, so spawn -> finish is n ticks, the
grader's own count). Counting from the last touch rather than from the first row outside it is
one tick longer, so the clock never flatters the policy.

"Inside the start trigger" is the engine's rule for a brush trigger (SV_TouchLinks, as
surfgym.goalhull does for the finish): the origin inside the trigger model's clip hull, hull 1
standing or hull 3 ducked. A plain box model's clip hull IS its bounds grown by the standing hull
(16, 16, 36), so that is tested directly; any other model through GoalHull.contains, behind the
grown-box precheck. A zone with several boxes ("boxes") is touched when any of them is.

Comparable with a record only when both the start and the finish are triggers: a button finish or
start (types 2 and 3, CLAUDE.md 4b) is a padded box the simulator arrives in instead of pressing,
and the gateway's boxes are not the map's own. A map no record exists for is not comparable either.

    sz = StartZone.for_map(bsp)                     # None: the map has no start zone
    eps = read_episodes("runs/x/traj_0001000000_cannonball.jsonl")
    best = grade(eps, sz, goal)                     # {"n", "done", "verified", "best_spawn_s",
                                                    #  "best_record_s", "touched"}
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

GROW = np.array([16.0, 16.0, 36.0])     # the standing player's half extents (hull 1)


def _is_button(zone: dict, zones: dict | None = None) -> bool:
    """The rule train_fast.py applies to the finish (slot.finish_kind): a padded button box or the
    gateway service's, not a brush trigger of the map."""
    return bool(zone.get("true_aabb") is not None or zone.get("from") == "func_button"
                or (zones or {}).get("source") == "gateway")


class StartZone:
    """The start trigger of one map: touch(pos) -> which origins are inside it."""

    def __init__(self, bsp_path, zone: dict, zones: dict | None = None):
        from surfgym.goalhull import GoalHull
        self.trigger = not _is_button(zone, zones)
        self.boxes, self.hulls = [], []
        for b in (zone.get("boxes") or [zone]):
            lo = np.asarray(b["mins"], np.float64) - GROW
            hi = np.asarray(b["maxs"], np.float64) + GROW
            g = GoalHull.match(bsp_path, b) if self.trigger else None
            if g is not None and not g.plain_box():
                self.hulls.append((lo, hi, g))
            else:
                self.boxes.append((lo, hi))

    @classmethod
    def for_map(cls, bsp_path) -> "StartZone | None":
        """From the map's zones.json (never re-extracted here: the trainer and the grader read the
        same file); None when it holds no start zone."""
        zp = Path(bsp_path).with_suffix(".zones.json")
        try:
            zones = json.loads(zp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        st = zones.get("start")
        if not st or not st.get("mins") or not st.get("maxs"):
            return None
        return cls(bsp_path, st, zones)

    def touch(self, pos) -> np.ndarray:
        p = np.asarray(pos, np.float64).reshape(-1, 3)
        m = np.zeros(len(p), bool)
        for lo, hi in self.boxes:
            m |= np.all((p >= lo) & (p <= hi), axis=1)
        for lo, hi, g in self.hulls:
            pre = np.flatnonzero(~m & np.all((p >= lo) & (p <= hi), axis=1))
            for i in pre:
                m[i] = g.contains(p[i])
        return m

    def record_ticks(self, rows) -> int | None:
        """n - i_s for an episode's rows [tick, x, y, z, ...]; None when it never touched the
        start zone (the human timer would never have started)."""
        r = np.asarray(rows, np.float64)
        if not len(r):
            return None
        hit = np.flatnonzero(self.touch(r[:, 1:4]))
        return int(len(r) - hit[-1]) if len(hit) else None


def tick_seconds(hdr: dict) -> float:
    """The episode's realised tick in seconds: the mean of a --tick-ms pattern, else tick_ms (the
    header carries the mean already; the pattern wins when both are there)."""
    pat = hdr.get("tick_pattern_ms")
    return (float(np.mean(pat)) if pat else float(hdr.get("tick_ms", 10.0))) / 1000.0


def read_episodes(path):
    """[(header, rows (n, k) float64, trailer)] of a recorded trajectory (record.py's format: a
    header line, the rows, a trailer {"end": ...}); .jsonl.gz too."""
    eps, hdr, cur = [], None, []
    _open = gzip.open if str(path).endswith(".gz") else open
    with _open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
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


def grade(eps, start: "StartZone | None", goal=None) -> dict:
    """The finishes of the episodes and their best times on both clocks. A "done" episode counts
    only when `goal` (surfgym.goalhull's test for a finish whose clip hull is not a plain box,
    None otherwise) agrees, as score.py counts it. best_record_s is None when no finish touched
    the start zone or the map has none."""
    out = {"n": len(eps), "done": 0, "verified": 0, "touched": 0,
           "best_spawn_s": None, "best_record_s": None}
    for hdr, r, tr in eps:
        if tr.get("end") != "done" or not len(r):
            continue
        out["done"] += 1
        tick_s = tick_seconds(hdr)
        if goal is not None and not goal.finished(r, tick_s):
            continue
        out["verified"] += 1
        # the spawn clock exactly as score.py writes it (rows x the header's mean tick)
        t = len(r) * float(hdr.get("tick_ms", 10.0)) / 1000.0
        if out["best_spawn_s"] is None or t < out["best_spawn_s"]:
            out["best_spawn_s"] = t
        rt = start.record_ticks(r) if start is not None else None
        if rt is not None:
            out["touched"] += 1
            t = rt * float(hdr.get("tick_ms", 10.0)) / 1000.0
            if out["best_record_s"] is None or t < out["best_record_s"]:
                out["best_record_s"] = t
    return out


def load_records(path) -> dict:
    """{map stem: world record in seconds} from a records snapshot (tools/leaderboard.py
    snapshot): {"maps": {stem: {"wr_s": ...}}}; a stem whose wr_s is null has no record."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: float(v["wr_s"]) for k, v in (d.get("maps") or {}).items()
            if v.get("wr_s") is not None}


def summarise(rec_s: dict, wr: dict, comparable: dict) -> dict:
    """The run-level numbers from each map's best record-clock time at one eval (None: no verified
    finish there): of the maps with a comparable record, how many beat it, and the median of time
    / record over those that finished."""
    keys = [k for k in rec_s if comparable.get(k) and k in wr]
    beat = [k for k in keys if rec_s[k] is not None and rec_s[k] < wr[k]]
    ratios = [rec_s[k] / wr[k] for k in keys if rec_s[k] is not None]
    return {"n_comparable": len(keys), "n_beat": len(beat), "beat": beat,
            "n_finished": len(ratios),
            "ratio_median": float(np.median(ratios)) if ratios else float("nan")}
