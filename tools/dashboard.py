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


def _downsample(steps, values):
    """Stable bucketed downsample: bucket edges are fixed in row space, so a
    live run appending rows only ever changes the final bucket (index-based
    sampling shifted every sample point each poll, visibly rewriting the
    whole curve every refresh). Bucket size doubles only when the run
    outgrows MAX_POINTS*size — a rare, one-time reflow."""
    n = len(steps)
    if n <= MAX_POINTS:
        return steps, values
    b = 1
    while (n + b - 1) // b > MAX_POINTS:
        b *= 2
    s_out, v_out = [], []
    for i in range(0, n, b):
        chunk = values[i:i + b]
        s_out.append(steps[min(i + b - 1, n - 1)])
        v_out.append(sum(chunk) / len(chunk))
    return s_out, v_out


def _metrics_from_csv(path: Path):
    cols: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    xkey = "time/total_timesteps"
    out = {}
    for key in rows[0].keys():
        if key == xkey:
            continue
        steps, values = [], []
        for r in rows:
            v, x = r.get(key, ""), r.get(xkey, "")
            if v not in ("", None) and x not in ("", None):
                try:
                    values.append(float(v)); steps.append(float(x))
                except ValueError:
                    pass
        if len(values) >= 2:
            s, v = _downsample(steps, values)
            out[key] = {"steps": s, "values": v}
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
        s, v = _downsample([e.step for e in ev], [e.value for e in ev])
        if len(v) >= 2:
            out[tag] = {"steps": s, "values": v}
    return out


def _run_info(d: Path):
    meta = {}
    mj = d / "run.json"
    if mj.exists():
        try:
            meta = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    trajs = []
    for p in sorted(d.glob("traj_*.jsonl")):
        try:
            steps = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            steps = -1
        pov = d / f"{p.stem}.pov.mp4"
        trajs.append({"file": f"/runs/{d.name}/{p.name}", "steps": steps,
                      "kb": p.stat().st_size // 1024,
                      "mode": "stoch" if p.stem.endswith("_stoch") else "greedy",
                      "pov": f"/runs/{d.name}/{pov.name}" if pov.exists() else None})
    ckpts = [p.name for p in sorted(d.glob("*.zip"))]
    mtime = max([p.stat().st_mtime for p in d.iterdir()] or [d.stat().st_mtime])
    # trainers touch progress.csv/ckpt every few seconds; 30s of silence
    # without a finished stamp = the run was killed
    live = meta.get("finished") is None and (time.time() - mtime) < 30
    return {
        "name": d.name,
        "label": meta.get("label", d.name),
        "started": meta.get("started") or datetime.fromtimestamp(
            d.stat().st_ctime).isoformat(timespec="seconds"),
        "finished": meta.get("finished"),
        "duration_s": meta.get("duration_s"),
        "status": "live" if live else ("finished" if meta.get("finished") else "interrupted"),
        "config": meta.get("config", {}),
        "steps": meta.get("total_steps") or (trajs[-1]["steps"] if trajs else None),
        "trajs": trajs,
        "checkpoints": ckpts,
        "has_metrics": (d / "progress.csv").exists() or
                       bool(list((RUNS / "tb").glob(f"{d.name}_*"))),
    }


class Handler(SimpleHTTPRequestHandler):
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
                        runs.append(_run_info(d))
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
            pov = p.parent / f"{p.stem.replace('.traj', '')}.pov.mp4" \
                if p.stem.endswith(".traj") else p.parent / f"{p.stem}.pov.mp4"
            if pov.exists():
                _RENDERS.pop(str(p), None)
                return self._json({"status": "done", "pov": rel})
            proc = _RENDERS.get(str(p))
            if proc is not None:
                if proc.poll() is None:
                    return self._json({"status": "rendering"})
                _RENDERS.pop(str(p), None)
                return self._json({"status": "failed", "rc": proc.returncode})
            _RENDERS[str(p)] = subprocess.Popen(
                [sys.executable, str(ROOT / "tools" / "render_pov.py"), str(p)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return self._json({"status": "started"})
        if url.path == "/api/metrics":
            q = urllib.parse.parse_qs(url.query)
            run = (q.get("run") or [""])[0]
            d = RUNS / run
            if not run or not d.is_dir():
                return self._json({"error": "unknown run"}, 404)
            csv_path = d / "progress.csv"
            series = _metrics_from_csv(csv_path) if csv_path.exists() \
                else _metrics_from_tb(run)
            return self._json({"run": run, "series": series})
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
