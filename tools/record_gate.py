#!/usr/bin/env python3
"""record_gate.py - the launcher's RECORD GATE: a run is not launched until
its checkpoint records.

User rule (2026-09-11, restated 2026-09-12 after the third broken button):
every dashboard button must work on every run, and every run's launcher must
first try to record one trajectory - if that fails, the run does not run.
Three times a freshly launched arm's record button died on a flag that
record_ckpt.py did not mirror or declare (dip_diag, the respawn_frontier
block, gate_boxes, the unstuck block), and once on a map that lives in
maps_pool/; every time the failure surfaced when the user clicked.

What it does, with the SAME backend the buttons call (tools/record_ckpt.py
of this checkout):

  * ``record_gate.py <run> --pid <trainer pid>``: waits for
    ``runs/<run>/ckpt_latest.pt`` (the trainer writes it about a minute after
    its first iteration), copies it aside (the trainer keeps rewriting it),
    and records ONE short episode per map of the run in every mode the UI
    offers on a fresh run: greedy, stochastic, and the drop-spawn pool
    (``--spawn mixed``). The reservoir mode is skipped here - iteration 1
    has no reservoir yet, and that button says so itself. Any failure prints
    the recorder's error, kills the trainer and exits 1.
  * ``record_gate.py <run> --ckpt <path>``: the same check on a given
    checkpoint, before a resume is launched (fast fail before compile).

    python tools/record_gate.py gbCELunstuck --pid 1234 --wait-secs 900
    python tools/record_gate.py jt3ANCHU --ckpt runs/jt3ANCHU/ckpt_final.pt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def find_bsp(stem: str) -> Path | None:
    for d in (ROOT / "maps", ROOT / "maps_pool"):
        p = d / f"{stem}.bsp"
        if p.is_file():
            return p
    return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True,
                                 text=True).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def kill_pid(pid: int) -> None:
    if pid <= 0:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(pid, 15)
    except Exception as e:
        print(f"   (could not kill {pid}: {e})")


def run_config(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get("config") or {}
    except Exception:
        return {}


def maps_of(cfg: dict, ck_path: Path) -> list[str]:
    maps = cfg.get("maps") or []
    if maps:
        return [str(m) for m in maps]
    if cfg.get("map"):
        return [str(cfg["map"])]
    try:
        import torch
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        c = ck.get("config") or {}
        return [str(m) for m in (c.get("maps") or [])] or ([str(c["map"])] if c.get("map") else [])
    except Exception:
        return []


def record_once(ck: Path, out: Path, bsp: Path | None, extra: list[str], timeout: float) -> tuple[bool, str]:
    cmd = [sys.executable, str(ROOT / "tools" / "record_ckpt.py"), str(ck),
           "--episodes", "1", "--ep-ticks", "200", "--out", str(out)] + extra
    if bsp is not None:
        cmd += ["--map", str(bsp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:.0f}s: {' '.join(cmd)}"
    if r.returncode != 0 or not out.exists():
        tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-12:])
        return False, f"rc {r.returncode}: {' '.join(cmd)}\n{tail}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--pid", type=int, default=0, help="the trainer; killed on failure")
    ap.add_argument("--ckpt", default=None, help="check this checkpoint instead of waiting for ckpt_latest")
    ap.add_argument("--wait-secs", type=float, default=900.0)
    ap.add_argument("--timeout", type=float, default=420.0, help="per recording")
    ap.add_argument("--modes", default="greedy,stoch,mixed",
                    help="comma list of greedy | stoch | mixed | reservoir")
    a = ap.parse_args()

    run_dir = RUNS / a.run
    if a.ckpt:
        ck_src = Path(a.ckpt)
        if not ck_src.exists():
            print(f"!! record gate: no checkpoint at {ck_src}")
            return 1
    else:
        ck_src = run_dir / "ckpt_latest.pt"
        t0 = time.time()
        print(f"== record gate: waiting up to {a.wait_secs:.0f}s for {ck_src} (trainer pid {a.pid or '-'})",
              flush=True)
        while not ck_src.exists():
            if a.pid and not pid_alive(a.pid):
                print("!! record gate: the trainer died before writing ckpt_latest.pt")
                return 1
            if time.time() - t0 > a.wait_secs:
                print(f"!! record gate: no ckpt_latest.pt after {a.wait_secs:.0f}s; killing pid {a.pid}")
                kill_pid(a.pid)
                return 1
            time.sleep(10.0)
        time.sleep(3.0)                      # let the save finish
    # the trainer rewrites ckpt_latest every minute: check a copy
    tmp_dir = run_dir if run_dir.is_dir() else ROOT / "runs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ck = tmp_dir / f".record_gate_{a.run}.pt"
    for attempt in range(5):
        try:
            shutil.copyfile(ck_src, ck)
            break
        except Exception as e:
            if attempt == 4:
                print(f"!! record gate: cannot copy {ck_src}: {e}")
                kill_pid(a.pid)
                return 1
            time.sleep(5.0)
    cfg = run_config(run_dir)
    stems = maps_of(cfg, ck)
    targets = [(s, find_bsp(s)) for s in stems] or [(None, None)]
    if any(s is not None and b is None for s, b in targets):
        missing = [s for s, b in targets if b is None]
        print(f"!! record gate: no .bsp under maps/ or maps_pool/ for {missing}")
        kill_pid(a.pid)
        ck.unlink(missing_ok=True)
        return 1
    modes = {"greedy": [], "stoch": ["--stochastic"], "mixed": ["--spawn", "mixed"],
             "reservoir": ["--spawn", "reservoir"]}
    wanted = [m.strip() for m in a.modes.split(",") if m.strip()]
    n_ok = 0
    t0 = time.time()
    for stem, bsp in targets:
        # the run-level buttons pass no --map on a single-map run; the
        # per-map buttons pass it on a --maps run. Mirror both shapes.
        bsp_arg = bsp if (len(targets) > 1) else None
        for m in wanted:
            out = tmp_dir / f".record_gate_{a.run}_{m}{('_' + stem) if stem else ''}.jsonl"
            ok, err = record_once(ck, out, bsp_arg, modes[m], a.timeout)
            label = f"{m}{' ' + stem if stem else ''}"
            if not ok:
                print(f"!! record gate FAILED on {label}:\n{err}")
                print(f"!! the dashboard's record button would fail the same way; killing trainer pid {a.pid or '-'}")
                kill_pid(a.pid)
                ck.unlink(missing_ok=True)
                return 1
            n_ok += 1
            out.unlink(missing_ok=True)
            print(f"   record gate: {label} ok", flush=True)
    ck.unlink(missing_ok=True)
    print(f"== record gate PASSED: {n_ok} recording(s) in {time.time() - t0:.0f}s "
          f"({', '.join(wanted)}{'; per map: ' + ', '.join(s for s, _ in targets) if len(targets) > 1 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
