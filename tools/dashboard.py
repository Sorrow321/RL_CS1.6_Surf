"""dashboard.py — local W&B-style runs dashboard for RL_Surf.

    python tools\\dashboard.py [--port 8000]

Serves the repo root (viewer + runs) plus two JSON APIs:

    /api/runs               run list w/ metadata, trajectory artifacts, ckpts
    /api/metrics?run=NAME   scalar curves: progress.csv if the run wrote one,
                            else parsed from its TensorBoard event files

Open http://localhost:8000/  (redirects to the dashboard page).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import math
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MAX_POINTS = 600  # per-series downsample cap

# in-flight POV renders: resolved traj path -> Popen
_RENDERS: dict = {}
# in-flight rollout recordings: "run/mode" -> Popen
_RECORDS: dict = {}


def _downsample(steps, values, extras=None):
    """Stable bucketed downsample: bucket edges are fixed in row space, so a
    live run appending rows only ever changes the final bucket (index-based
    sampling shifted every sample point each poll, visibly rewriting the
    whole curve every refresh). Bucket size doubles only when the run
    outgrows MAX_POINTS*size — a rare, one-time reflow.

    ``extras`` are further X-LIKE axes (iteration index, wall-clock hours)
    parallel to ``steps``. They are bucketed exactly as steps is - last of
    the bucket, never averaged - so every axis stays aligned point for
    point with the values the chart draws. Returns (steps, values, extras).
    """
    n = len(steps)
    extras = extras or {}
    b = 1
    while (n + b - 1) // b > MAX_POINTS:
        b *= 2
    if b == 1:
        return steps, values, {k: list(v) for k, v in extras.items()}
    s_out, v_out = [], []
    e_out = {k: [] for k in extras}
    for i in range(0, n, b):
        chunk = values[i:i + b]
        j = min(i + b - 1, n - 1)
        s_out.append(steps[j])
        v_out.append(sum(chunk) / len(chunk))
        for k, arr in extras.items():
            e_out[k].append(arr[j])
    return s_out, v_out, e_out


def _wall_hours(xs, fpss):
    """Hours since the process started, per row - DERIVED, because
    progress.csv carries no timestamp column at all.

    What it does carry is ``time/fps``, and that is the CUMULATIVE mean
    rate since the process started: train_fast.py computes
    ``(global_step - step_start) / (perf_counter() - t_start)`` against a
    t_start fixed once before the loop. So elapsed_i = (x_i - x0) / fps_i
    EXACTLY, for the one unknown x0 = the step the process started from
    (0 from scratch, the checkpoint's step on a resume). x0 is not in the
    file either, so it is recovered from the first two rows: an iteration
    is a fixed number of steps, hence x0 = x_1 - (x_2 - x_1).

    Rows whose fps is missing or zero carry the previous value forward,
    and the series is forced non-decreasing - a wall-clock axis that goes
    backwards would make uPlot draw garbage. Returns None when the column
    is unusable, and the axis is then simply not offered.
    """
    pts = [(x, f) for x, f in zip(xs, fpss)
           if x is not None and f is not None and f > 0]
    if len(pts) < 2:
        return None
    x0 = pts[0][0] - (pts[1][0] - pts[0][0])
    out, best = [], 0.0
    for x, f in zip(xs, fpss):
        if x is not None and f is not None and f > 0:
            h = max(0.0, (x - x0) / f) / 3600.0
            if h >= best:
                best = h
        out.append(round(best, 5))
    return out


def _f(s):
    """A csv cell as a finite float, or None. A 'nan' cell (an eval metric
    with no measurement yet) parses as float NaN and must never reach the
    JSON: python emits a bare NaN token that every browser's JSON.parse
    rejects, which is how a run with 13 healthy series once rendered as
    "No metrics logged"."""
    if s in ("", None):
        return None
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _metrics_from_csv(path: Path):
    cols: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    xkey = "time/total_timesteps"
    # A directory that was relaunched under the same name (or resumed by
    # an older trainer) holds rows whose step folds back to 0, and the
    # plot draws one line per life. Only the LAST monotone segment is the
    # live run; everything before it is a previous life. The trainer now
    # refuses such launches; this keeps the old files readable.
    start, last = 0, None
    for i, r in enumerate(rows):
        try:
            x = float(r.get(xkey, ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x):
            continue
        if last is not None and x < last:
            start = i
        last = x
    rows = rows[start:]
    # The alternate X axes the page can plot against, computed over the
    # rows of THIS life only - never across the fold, or an overlay would
    # silently mix two lives on one x. "iter" is the row's index in the
    # live segment (1-based); "wall" is derived hours (see _wall_hours).
    rx = [_f(r.get(xkey)) for r in rows]
    rf = [_f(r.get("time/fps")) for r in rows]
    rwall = _wall_hours(rx, rf)
    out = {}
    for key in rows[0].keys():
        if key == xkey:
            continue
        steps, values, iters, walls = [], [], [], []
        for i, r in enumerate(rows):
            v, x = r.get(key, ""), r.get(xkey, "")
            if v not in ("", None) and x not in ("", None):
                try:
                    fv, fx = float(v), float(x)
                except ValueError:
                    continue
                # a 'nan' cell (a metric with no measurement yet, e.g. the
                # eval/* columns before the first eval) parses as float NaN,
                # and json.dumps then emits a bare NaN token - which python's
                # json accepts but every BROWSER's JSON.parse rejects, so the
                # dashboard rendered "No metrics logged" for a run whose API
                # response held 13 healthy series. An unmeasured value is not
                # a point; drop it rather than ship it.
                if math.isfinite(fv) and math.isfinite(fx):
                    values.append(fv); steps.append(fx)
                    iters.append(i + 1)
                    if rwall is not None:
                        walls.append(rwall[i])
        if len(values) >= 2:
            ex = {"iter": iters}
            if rwall is not None:
                ex["wall"] = walls
            s, v, ex = _downsample(steps, values, ex)
            out[key] = dict({"steps": s, "values": v}, **ex)
    return out



# ---- expert loops (tools/expert_loop.py): ONE row per loop -----------------
# runs/<loop>/ holds driver.log, expert_summary.jsonl and round_<n>/ dirs;
# each round has the PPO run at round_<n>/train (progress.csv, ckpts) and
# the loop's own greedy start-line evals one level up (eval_in.jsonl before
# the round's training, eval_out.jsonl after). The dashboard shows the loop
# as a single run: training curves concatenated over rounds (the step
# counter carries across rounds), the per-round scoreboard on the same x,
# and every eval recording as a trajectory artifact.

def _is_loop(d: Path) -> bool:
    return (d / "driver.log").exists() and (
        (d / "expert_summary.jsonl").exists() or any(d.glob("round_*")))


def _loop_rounds(d: Path):
    rs = []
    for r in d.glob("round_*"):
        if r.is_dir():
            try:
                rs.append((int(r.name[6:]), r))
            except ValueError:
                continue
    return [r for _, r in sorted(rs)]


def _loop_summary(d: Path):
    out = {}
    es = d / "expert_summary.jsonl"
    if es.exists():
        with open(es, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    out[int(row.get("round"))] = row
                except (ValueError, TypeError):
                    continue
    return out


def _last_step(csv_path: Path):
    """time/total_timesteps of the last row of a progress.csv, else None."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            last = None
            for r in reader:
                last = r
        return float(last["time/total_timesteps"]) if last else None
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _ckpt_for(d: Path):
    """the checkpoint the record buttons act on: a plain run's
    ckpt_latest.pt; a loop's newest round checkpoint."""
    if _is_loop(d):
        for r in reversed(_loop_rounds(d)):
            for nm in ("ckpt_latest.pt", "ckpt_final.pt"):
                c = r / "train" / nm
                if c.exists():
                    return c
    return d / "ckpt_latest.pt"


def _run_json_for(traj: Path):
    """the run.json describing the policy that made a recording: next to
    it for a plain run, round_<n>/train/run.json for a loop's eval."""
    for c in (traj.parent / "run.json", traj.parent / "train" / "run.json"):
        if c.exists():
            return c
    return traj.parent / "run.json"


def _fin(v):
    """'7/9' -> 7, 7 -> 7, None -> None"""
    if isinstance(v, str) and "/" in v:
        v = v.split("/")[0]
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _loop_info(d: Path):
    name = d.name
    summ = _loop_summary(d)
    rounds = _loop_rounds(d)
    trajs, cfg, last_step = [], {}, 0
    for r in rounds:
        n = int(r.name[6:])
        tr = r / "train"
        st = None
        rj = tr / "run.json"
        if rj.exists():
            try:
                m = json.loads(rj.read_text(encoding="utf-8"))
                cfg = m.get("config") or cfg
                st = m.get("total_steps")
            except Exception:
                pass
        if st is None and (tr / "progress.csv").exists():
            st = _last_step(tr / "progress.csv")
        row = summ.get(n, {})
        for nm in ("eval_in", "eval_out"):
            q = r / f"{nm}.jsonl"
            if not q.exists() or (nm == "eval_in" and n != 0):
                continue          # eval_in of round n = eval_out of n-1
            if nm == "eval_in":
                best, mean, fin = row.get("greedy_in_best_s"), row.get("greedy_in_mean_s"), _fin(row.get("greedy_in_finishes"))
                stp, what = last_step, "seed"
            else:
                best, mean, fin = row.get("greedy_out_best_s"), row.get("greedy_out_mean_s"), _fin(row.get("greedy_out_finishes"))
                stp, what = (st or last_step), f"after round {n}"
            tail = ""
            if best is not None:
                tail = f" - best {float(best):.2f} s"
                if mean is not None:
                    tail += f", mean {float(mean):.2f} s"
                if fin is not None:
                    tail += f", {fin}/9 finished"
            elif nm == "eval_out":
                tail = " - scoring"
            pov = r / f"{nm}.pov.mp4"
            trajs.append({"file": f"/runs/{name}/{r.name}/{q.name}", "steps": int(stp or 0),
                          "kb": q.stat().st_size // 1024,
                          "mode": f"{what} greedy x9{tail}", "map": None,
                          "pov": f"/runs/{name}/{r.name}/{pov.name}" if pov.exists() else None})
        if st:
            last_step = st
    # on-demand recordings made from this row's record buttons land in the
    # loop dir itself, like a plain run's
    for p in sorted(d.glob("traj_*.jsonl")):
        try:
            steps = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            steps = last_step
        pov = d / f"{p.stem}.pov.mp4"
        mode = "stoch" if p.stem.endswith("_stoch") else "greedy"
        trajs.append({"file": f"/runs/{name}/{p.name}", "steps": steps,
                      "kb": p.stat().st_size // 1024, "mode": f"recorded {mode}",
                      "map": None,
                      "pov": f"/runs/{name}/{pov.name}" if pov.exists() else None})
    dl = d / "driver.log"
    fin_line, finished, txt = False, None, ""
    try:
        txt = dl.read_text(encoding="utf-8", errors="replace")
        fin_line = "finished" in txt[-400:]
        if fin_line:
            finished = datetime.fromtimestamp(dl.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        pass
    mtime = dl.stat().st_mtime if dl.exists() else d.stat().st_mtime
    live = not fin_line and (time.time() - mtime) < 1800
    started = datetime.fromtimestamp(dl.stat().st_ctime if dl.exists() else d.stat().st_ctime)
    dur = (dl.stat().st_mtime - dl.stat().st_ctime) if dl.exists() else None
    ndone = len(summ)
    phase = ""
    if live and txt:
        last_line = txt.rstrip().rsplit("\n", 1)[-1][:160]
        for key, lab in (("plan wave", "planning"), ("train:", "training"),
                         ("distil", "distilling"), ("eval_in", "evaluating"),
                         ("eval_out", "evaluating"), ("SUMMARY", "round closed"),
                         ("=== round", "starting round")):
            if key in last_line:
                phase = ", " + lab
                break
    plural = "s" if ndone != 1 else ""
    return {
        "_mtime": mtime,
        "name": name,
        "label": f"{name} (expert loop, {ndone} round{plural} done{phase})",
        "started": started.isoformat(timespec="seconds"),
        "finished": finished,
        "duration_s": dur,
        "status": "live" if live else ("finished" if fin_line else "interrupted"),
        "config": cfg,
        "steps": last_step or None,
        "trajs": trajs,
        "checkpoints": [],
        "has_metrics": bool(rounds),
    }


def _metrics_from_loop_dir(d: Path):
    """training curves concatenated over rounds (the trainer's step counter
    continues from the round's checkpoint, so x stays monotone) plus the
    per-round scoreboard on the same x. Time semantics: round n's planner
    line is found BEFORE its training (x = the round's first step); the
    greedy eval after the round sits at its last step; the seed's own eval
    (greedy_in of round 0) opens the series at the loop's first step. The
    trainer's race/finish_s is from-SPAWN time over respawn-curriculum
    episodes; loop/* is the start-line clock that matters."""
    series, x0, x1 = {}, {}, {}
    for r in _loop_rounds(d):
        n = int(r.name[6:])
        csvp = r / "train" / "progress.csv"
        if not csvp.exists():
            continue
        part = _metrics_from_csv(csvp)
        for k, v in part.items():
            s = series.setdefault(k, {"steps": [], "values": []})
            s["steps"].extend(v["steps"])
            s["values"].extend(v["values"])
        firsts = [v["steps"][0] for v in part.values() if v["steps"]]
        lasts = [v["steps"][-1] for v in part.values() if v["steps"]]
        if firsts:
            x0[n], x1[n] = min(firsts), max(lasts)
    summ = _loop_summary(d)
    if not summ and not x0:
        return series
    rounds = sorted(set(summ) | set(x0))
    # a round without a csv yet (planning / evaluating) starts where the
    # previous one ended
    xs_start, xs_end, prev_end = {}, {}, 0.0
    for n in rounds:
        xs_start[n] = x0.get(n, prev_end)
        xs_end[n] = x1.get(n, xs_start[n])
        prev_end = xs_end[n]

    def put(tag, x, v):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return
        if math.isfinite(fv):
            s = series.setdefault(tag, {"steps": [], "values": []})
            s["steps"].append(x); s["values"].append(fv)

    first = summ.get(rounds[0]) if rounds else None
    if first is not None and rounds[0] == 0:
        put("loop/greedy_best_s", xs_start[0], first.get("greedy_in_best_s"))
        put("loop/greedy_mean_s", xs_start[0], first.get("greedy_in_mean_s"))
        put("loop/finishes_of_9", xs_start[0], _fin(first.get("greedy_in_finishes")))
    for n in rounds:
        row = summ.get(n)
        if row is None:
            continue
        put("loop/planner_s", xs_start[n], row.get("planner_best_s"))
        put("loop/greedy_best_s", xs_end[n], row.get("greedy_out_best_s"))
        put("loop/greedy_mean_s", xs_end[n], row.get("greedy_out_mean_s"))
        put("loop/finishes_of_9", xs_end[n], _fin(row.get("greedy_out_finishes")))
    return series


def _metrics_from_loop(path: Path):
    """An expert loop's per-round scoreboard, from runs/<loop>/expert_summary
    .jsonl: greedy start-line finish times (best / mean of the E eval
    episodes), the planner's line, finishes. training's race/finish_s inside
    a round is from-SPAWN time over respawn-curriculum episodes and is NOT
    the clock that matters; this is. x = round index."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    cols = {"loop/greedy_best_s": "greedy_out_best_s",
            "loop/greedy_mean_s": "greedy_out_mean_s",
            "loop/planner_s": "planner_best_s",
            "loop/greedy_in_best_s": "greedy_in_best_s",
            "loop/finishes": "greedy_out_finishes"}
    out = {}
    for tag, key in cols.items():
        xs, ys = [], []
        for r in rows:
            v = r.get(key)
            if isinstance(v, str) and "/" in v:      # "7/9"
                v = v.split("/")[0]
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            xs.append(float(r.get("round", len(xs))))
            ys.append(v)
        if ys:
            out[tag] = {"steps": xs, "values": ys}
    return out


def _metrics_from_tb(run: str):
    """Fallback for runs that only logged TensorBoard (runs/tb/<run>_N)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {}
    cands = sorted((RUNS / "tb").glob(f"{run}_*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        return {}
    ea = EventAccumulator(str(cands[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        ev = ea.Scalars(tag)
        w0 = ev[0].wall_time if ev else 0.0
        ex = {"iter": list(range(1, len(ev) + 1)),
              "wall": [round(max(0.0, e.wall_time - w0) / 3600.0, 5)
                       for e in ev]}
        s, v, ex = _downsample([e.step for e in ev],
                               [e.value for e in ev], ex)
        if len(v) >= 2:
            out[tag] = dict({"steps": s, "values": v}, **ex)
    return out


def _run_info(d: Path):
    meta = {}
    mj = d / "run.json"
    if mj.exists():
        try:
            meta = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    # an expert loop's PPO runs live at runs/<loop>/round_<n>/train: the name
    # is the RUNS-relative path, which is what every /runs/<name>/... URL and
    # RUNS / name below need. Top-level runs keep their plain name.
    if d.parent == RUNS:
        name = d.name          # a junction like runs/xENT -> runs/research/xENT keeps its own name
    else:
        try:
            name = d.resolve().relative_to(RUNS.resolve()).as_posix()
        except ValueError:
            name = d.name
    # a top-level symlink to a round (the old round_links.sh workaround)
    # resolves to the same nested dir, so judge the layout on the target
    d = d.resolve() if d.is_symlink() else d
    if "/" in name and d.parent.name.startswith("round_"):
        meta["label"] = f"{d.parent.parent.name} r{d.parent.name[6:]} {d.name}"
    # the loop directory itself: label it, count rounds, and call it live
    # while the driver is writing (its trainer is only up ~half the time)
    loop_live = False
    es, dl = d / "expert_summary.jsonl", d / "driver.log"
    if es.exists() or dl.exists():          # round 0 has a driver.log only
        nrounds = 0
        if es.exists():
            try:
                nrounds = sum(1 for _ in open(es, encoding="utf-8"))
            except Exception:
                nrounds = 0
        fin = False
        if dl.exists():
            try:
                fin = "finished" in dl.read_text(encoding="utf-8", errors="replace")[-400:]
            except Exception:
                fin = False
            if fin:
                meta.setdefault("finished", datetime.fromtimestamp(
                    dl.stat().st_mtime).isoformat(timespec="seconds"))
            loop_live = not fin and (time.time() - dl.stat().st_mtime) < 1800
        meta.setdefault("label", f"{d.name} (expert loop, {nrounds} rounds done)")
    trajs = []
    # sort by the STEP in the name, not by string: traj_9661579264 must come
    # before traj_10718543872 (an 11-digit step sorts first as a string)
    def _traj_step(pth):
        m = re.search(r"traj_(\d+)", pth.name)
        return (int(m.group(1)) if m else -1, pth.name)
    for p in sorted(d.glob("traj_*.jsonl"), key=_traj_step):
        try:
            steps = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            steps = -1
        pov = d / f"{p.stem}.pov.mp4"
        mode = "stoch" if p.stem.endswith("_stoch") else "greedy"
        for tag in ("reservoir", "mixed", "ramp", "platform"):
            if f"_{tag}" in p.stem:                  # spawn-override records
                mode = f"{tag}-spawns · {mode}"
                break
        # --maps: the trainer writes one recording per map per eval, named
        # traj_<step>_<maptag>.jsonl. Without this every map's line reads
        # "greedy" and the list is unusable on a multi-map run.
        # a --maps run writes traj_<step>_<maptag>.jsonl. Carry the map as
        # its OWN field: at 40 maps the viewer has to BUCKET by it, and
        # parsing it back out of a display label would be fragile.
        tag = None
        for m in (meta.get("config", {}).get("maps") or []):
            mt = m.replace("surf_src_", "").replace("surf_", "")
            if f"_{mt}" in p.stem:
                tag, mode = mt, f"{mt} · {mode}"
                break
        trajs.append({"file": f"/runs/{name}/{p.name}", "steps": steps,
                      "kb": p.stat().st_size // 1024, "mode": mode, "map": tag,
                      "pov": f"/runs/{name}/{pov.name}" if pov.exists() else None})
    # expert-iteration rounds: the loop's greedy evals are record_ckpt
    # trajectories one level up (round_<n>/eval_in.jsonl = the policy this
    # round started from, eval_out.jsonl = after its training), not
    # traj_<step>.jsonl inside train/, so the dashboard never listed them
    if d.parent.name.startswith("round_"):      # round.json only lands at the round's end
        rel_parent = name.rsplit("/", 1)[0] if "/" in name else d.parent.name
        st = int(meta.get("total_steps") or 0)
        for nm, stp in (("eval_in", 0), ("eval_out", st)):
            q = d.parent / f"{nm}.jsonl"
            if q.exists() and (nm == "eval_out" or d.parent.name == "round_0"):
                trajs.append({"file": f"/runs/{rel_parent}/{q.name}", "steps": stp,
                              "kb": q.stat().st_size // 1024,
                              "mode": f"{nm} · greedy", "map": None, "pov": None})
    ckpts = [p.name for p in sorted(d.glob("*.zip"))]
    mtime = max([p.stat().st_mtime for p in d.iterdir()] or [d.stat().st_mtime])
    # trainers touch progress.csv/ckpt every few seconds; 30s of silence
    # without a finished stamp = the run was killed
    live = loop_live or (meta.get("finished") is None and (time.time() - mtime) < 30)
    return {
        "_mtime": mtime,
        "name": name,
        "label": meta.get("label", name),
        "started": meta.get("started") or datetime.fromtimestamp(
            d.stat().st_ctime).isoformat(timespec="seconds"),
        "finished": meta.get("finished"),
        "duration_s": meta.get("duration_s"),
        "status": "live" if live else ("finished" if meta.get("finished") else "interrupted"),
        "config": meta.get("config", {}),
        "steps": meta.get("total_steps") or (trajs[-1]["steps"] if trajs else None),
        "trajs": trajs,
        "checkpoints": ckpts,
        "has_metrics": (d / "progress.csv").exists() or (d / "expert_summary.jsonl").exists() or
                       bool(list((RUNS / "tb").glob(f"{d.name}_*"))),
    }


class Handler(SimpleHTTPRequestHandler):
    # mimetypes reads the WINDOWS REGISTRY first, where a stray entry
    # serves .js as text/plain and the browser then refuses to execute
    # runs.js or the vendored uPlot. extensions_map wins over the registry,
    # so pin the three types the dashboard actually depends on.
    extensions_map = dict(SimpleHTTPRequestHandler.extensions_map)
    extensions_map.update({".js": "text/javascript", ".css": "text/css",
                           ".json": "application/json",
                           ".html": "text/html"})

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, *a):  # quiet
        pass

    def end_headers(self):
        # viewer html/js must always revalidate — a stale cached runs.js kept
        # showing the old dashboard after server updates
        if self.path.endswith((".html", ".js", ".css")) or self.path == "/":
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_ranged(self, urlpath):
        """Range-aware media serving — browsers can't seek a <video> without
        206 partial-content support, which SimpleHTTPRequestHandler lacks."""
        f = Path(self.translate_path(urlpath))
        if not f.is_file():
            self.send_error(404)
            return
        size = f.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            if a:
                start = int(a)
            if b:
                end = min(int(b), size - 1)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with open(f, "rb") as fh:
                fh.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = fh.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (ConnectionAbortedError, BrokenPipeError):
            pass

    def do_POST(self):
        # /api/save_video?traj=/runs/<...>/x.jsonl&ep=N  body = webm bytes
        # The viewer's recorder posts its recording here so a headless run
        # (or a user who does not want a browser download) lands the file
        # next to the trajectory as <stem>_ep<N>.zone.webm.
        url = urllib.parse.urlparse(self.path)
        if url.path != "/api/save_video":
            return self._json({"error": "unknown endpoint"}, 404)
        q = urllib.parse.parse_qs(url.query)
        rel = (q.get("traj") or [""])[0].lstrip("/")
        ep = re.sub(r"[^0-9]", "", (q.get("ep") or ["1"])[0]) or "1"
        p = (ROOT / rel).resolve()
        if not str(p).startswith(str(RUNS.resolve())) or not p.name.endswith(".jsonl") or not p.exists():
            return self._json({"error": "bad traj path"}, 400)
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 2_000_000_000:
            return self._json({"error": "bad length"}, 400)
        out = p.parent / f"{p.stem}_ep{ep}.zone.webm"
        with open(out, "wb") as f:
            left = n
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        return self._json({"saved": f"/runs/{out.relative_to(RUNS).as_posix()}", "bytes": n - left})

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path.endswith(".mp4"):
            return self._serve_ranged(url.path)
        if url.path == "/":
            self.send_response(302)
            self.send_header("Location", "/viewer/runs.html")
            self.end_headers()
            return
        if url.path == "/api/runs":
            runs = []
            if RUNS.exists():
                for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime,
                                reverse=True):
                    if d.is_dir() and d.name != "tb":
                        runs.append(_loop_info(d) if _is_loop(d) else _run_info(d))
                # live rows first, then newest activity first
                runs.sort(key=lambda r: (r["status"] != "live",
                                         -(r.get("_mtime") or 0)))
                for r in runs:
                    r.pop("_mtime", None)
            return self._json({"runs": runs})
        if url.path == "/api/render_pov":
            # kick off (or poll) a POV-video render for a trajectory:
            # /api/render_pov?traj=/runs/<run>/traj_X.jsonl
            # -> {"status": "started"|"rendering"|"done"|"failed"}
            q = urllib.parse.parse_qs(url.query)
            rel = (q.get("traj") or [""])[0].lstrip("/")
            p = (ROOT / rel).resolve()
            if (not str(p).startswith(str(RUNS.resolve())) or
                    not p.name.endswith(".jsonl") or not p.exists()):
                return self._json({"error": "bad traj path"}, 400)
            # the run's OWN vision config: a POV that does not match what
            # the policy actually saw is a misleading picture, and a
            # --surf-mask run needs its second channel or the panel silently
            # shows depth only. Mask renders get their own filename so a
            # stale depth-only mp4 is never served in their place.
            vis, rj = [], _run_json_for(p)
            if rj.exists():
                try:
                    rcfg = json.loads(rj.read_text(encoding="utf-8")).get("config", {})
                except Exception:
                    rcfg = {}
                if rcfg.get("surf_mask"):
                    vis.append("--surf-mask")
                if rcfg.get("normals"):
                    # --normals: the ego-frame normal channels as an RGB
                    # panel under the depth (render_pov.py --normals)
                    vis.append("--normals")
                if rcfg.get("goal_obs") in ("ball", "both"):
                    # the goal-ball view channels the policy receives,
                    # stacked under the depth panel
                    vis += ["--goal-ball", str(int(rcfg.get("goal_views") or 4)),
                            "--goal-radius",
                            str(float(rcfg.get("goal_radius") or 192.0))]
                if rcfg.get("lidar_w"):
                    vis += ["--w", str(int(rcfg["lidar_w"]))]
                if rcfg.get("lidar_h"):
                    vis += ["--h", str(int(rcfg["lidar_h"]))]
                # --lidar-hfov/--lidar-vfov: the aspect correction and the
                # ball wrapper both follow the run's own camera
                if rcfg.get("lidar_hfov"):
                    vis += ["--hfov", str(float(rcfg["lidar_hfov"]))]
                if rcfg.get("lidar_vfov"):
                    vis += ["--vfov", str(float(rcfg["lidar_vfov"]))]
            # every extra panel gets its own filename, so a stale render of
            # another channel set is never served in its place
            tags = (["nrm"] if "--normals" in vis else []) \
                + (["ball"] if "--goal-ball" in vis
                   else ["mask"] if "--surf-mask" in vis else [])
            sfx = "." + ".".join(tags + ["pov", "mp4"])
            stem = p.stem.replace(".traj", "") if p.stem.endswith(".traj") else p.stem
            pov = p.parent / (stem + sfx)
            # check the PROCESS before the file: ffmpeg creates the mp4 at
            # render start and finalizes it only on exit — exists() alone
            # reported "done" on a half-written file (empty first playback)
            proc = _RENDERS.get(str(p))
            if proc is not None:
                if proc.poll() is None:
                    return self._json({"status": "rendering"})
                _RENDERS.pop(str(p), None)
                if proc.returncode != 0:
                    ef = p.parent / f"{p.stem}.pov.err"
                    msg = ""
                    if ef.exists():
                        msg = ef.read_text(errors="replace").strip().splitlines()
                        msg = msg[-1] if msg else ""
                    return self._json({"status": "failed",
                                       "rc": proc.returncode, "error": msg})
            if pov.exists():
                return self._json({"status": "done",
                                   "pov": "/" + pov.relative_to(ROOT).as_posix()})
            # keep stderr: a job that dies (missing cv2, bad ckpt) used to be
            # indistinguishable from one still running, and the UI just said
            # "retry" forever
            errf = open(p.parent / f"{p.stem}.pov.err", "wb")
            _RENDERS[str(p)] = subprocess.Popen(
                [sys.executable, str(ROOT / "tools" / "render_pov.py"),
                 str(p), "--out", str(pov)] + vis,
                stdout=subprocess.DEVNULL, stderr=errf)
            return self._json({"status": "started"})
        if url.path == "/api/record":
            # record fresh rollouts from a run's ckpt_latest.pt:
            # /api/record?run=NAME&mode=stoch|greedy[&spawn=mixed|ramp|platform]
            # (spawn override: race runs record from the start line by
            # default — pass spawn=mixed to see the training drop spawns)
            q = urllib.parse.parse_qs(url.query)
            run = (q.get("run") or [""])[0]
            mode = (q.get("mode") or ["stoch"])[0]
            spawn = (q.get("spawn") or [None])[0]
            # multi-map: record ONE map, not whichever the ckpt names first.
            # record_ckpt.py already takes --map and warns when it is not in
            # the checkpoint's list; validate here too so a typo is a 400
            # rather than a silently wrong recording.
            wanted = (q.get("map") or [None])[0]
            d = RUNS / run
            ck = _ckpt_for(d)
            if (not run or not d.is_dir() or mode not in ("stoch", "greedy")
                    or spawn not in (None, "platform", "ramp", "mixed",
                                     "reservoir")):
                return self._json({"error": "bad request"}, 400)
            if not ck.exists():
                return self._json({"error": "no ckpt_latest.pt"}, 400)
            key = f"{run}/{mode}/{spawn or 'default'}/{wanted or 'all'}"
            proc = _RECORDS.get(key)
            if proc is not None:
                if proc.poll() is None:
                    pf = d / f"record_{mode}_{spawn or 'default'}.progress"
                    info = {}
                    try:
                        info = json.loads(pf.read_text(encoding="utf-8"))
                    except Exception:
                        pass          # not written yet = still starting up
                    return self._json({"status": "recording",
                                       "pct": info.get("pct"),
                                       "phase": info.get("phase"),
                                       "episode": info.get("episode"),
                                       "episodes": info.get("episodes")})
                _RECORDS.pop(key, None)
                ef = d / f"record_{mode}_{spawn or 'default'}.err"
                msg = ""
                if proc.returncode != 0 and ef.exists():
                    lines = ef.read_text(errors="replace").strip().splitlines()
                    msg = lines[-1] if lines else ""
                return self._json(
                    {"status": "done" if proc.returncode == 0 else "failed",
                     "rc": proc.returncode, "error": msg})
            # Hand recordings run ONE env, so they get none of training's
            # 2048-way parallelism: ~10^5 fps is an AGGREGATE, about 116
            # ticks/s per env. Cost is therefore just the tick budget, and
            # record_ckpt used to silently restore a race ckpt's 12000-tick
            # episode cap over whatever was passed here - so "2 x 2000"
            # actually ran 24,000 ticks (~100 s), and got SLOWER the better
            # the agent got. The cap is honoured now; 2 x 3000 = 6000 ticks
            # is 60 s of game time per episode and lands in well under a minute.
            prog = d / f"record_{mode}_{spawn or 'default'}.progress"
            try:
                prog.unlink()          # stale % from a previous run misleads
            except FileNotFoundError:
                pass
            if wanted:
                try:
                    _m = json.loads((d / "run.json").read_text(encoding="utf-8"))
                except Exception:
                    _m = {}
                cfg_maps = (_m.get("config") or {}).get("maps") or []
                tags = {m.replace("surf_src_", "").replace("surf_", ""): m
                        for m in cfg_maps}
                full = tags.get(wanted, wanted)
                bsp = ROOT / "maps" / f"{full}.bsp"
                if cfg_maps and wanted not in tags and full not in cfg_maps:
                    return self._json({"error": f"map {wanted!r} not in this run"}, 400)
                if not bsp.exists():
                    return self._json({"error": f"no bsp for {full!r}"}, 400)
            cmd = [sys.executable, str(ROOT / "tools" / "record_ckpt.py"), str(ck),
                   "--episodes", "2", "--ep-ticks", "3000",
                   "--progress-file", str(prog)]
            if spawn:
                cmd += ["--spawn", spawn]
            if wanted:
                cmd += ["--map", str(bsp)]
            if mode == "stoch":
                cmd.append("--stochastic")
            errf = open(d / f"record_{mode}_{spawn or 'default'}.err", "wb")
            _RECORDS[key] = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=errf)
            return self._json({"status": "started"})
        if url.path == "/api/metrics":
            q = urllib.parse.parse_qs(url.query)
            run = (q.get("run") or [""])[0]
            d = RUNS / run
            if not run or not d.is_dir():
                return self._json({"error": "unknown run"}, 404)
            csv_path = d / "progress.csv"
            if _is_loop(d):
                series = _metrics_from_loop_dir(d)
            elif csv_path.exists():
                series = _metrics_from_csv(csv_path)
            else:
                series = _metrics_from_tb(run)
            # Which X axes the page may offer FOR THIS RUN. "wall" is
            # derived from time/fps, so a run without that column (or with
            # a single row) simply does not get the option rather than
            # getting a fabricated one.
            axes = ["steps"]
            for k in ("iter", "wall"):
                if any(k in s for s in series.values()):
                    axes.append(k)
            return self._json({"run": run, "series": series, "axes": axes})
        return super().do_GET()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"RL_Surf dashboard: http://localhost:{args.port}/")
    srv.serve_forever()


if __name__ == "__main__":
    main()
