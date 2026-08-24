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
        trajs.append({"file": f"/runs/{d.name}/{p.name}", "steps": steps,
                      "kb": p.stat().st_size // 1024, "mode": mode, "map": tag,
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
            # the run's OWN vision config: a POV that does not match what
            # the policy actually saw is a misleading picture, and a
            # --surf-mask run needs its second channel or the panel silently
            # shows depth only. Mask renders get their own filename so a
            # stale depth-only mp4 is never served in their place.
            vis, rj = [], p.parent / "run.json"
            if rj.exists():
                try:
                    rcfg = json.loads(rj.read_text(encoding="utf-8")).get("config", {})
                except Exception:
                    rcfg = {}
                if rcfg.get("surf_mask"):
                    vis.append("--surf-mask")
                if rcfg.get("lidar_w"):
                    vis += ["--w", str(int(rcfg["lidar_w"]))]
                if rcfg.get("lidar_h"):
                    vis += ["--h", str(int(rcfg["lidar_h"]))]
            sfx = ".mask.pov.mp4" if "--surf-mask" in vis else ".pov.mp4"
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
            ck = d / "ckpt_latest.pt"
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
