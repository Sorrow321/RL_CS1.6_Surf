#!/usr/bin/env python3
"""speed_forks.py - a FASTER line from the policy's own search, forked where its line stalls
(uf2r2b2, 2026-10-31).

Once a policy finishes, the time it loses sits where its line makes no progress: the running
best goal-field distance stands still while the policy builds energy, waits, or laps (on
surf_unitfarmer2 the start pit: ~9 s of laps before the exit, ~4 s more than a direct line).
The policy's own proposals (a collapsed yaw sigma) never try another manoeuvre there, so a
plain beam reproduces the same stall. This driver:

1. BEAM: tools/beam_tas.py from the TRUE start on the checkpoint (population search ranked by
   goal-field d, a greedy floor, the K fastest finishers kept for plan_to_bc).
2. STALLS: on the winner's per-tick states, the gaps between successive improvements of its
   running-best d by at least --stall-margin u; the --stalls longest are the stall windows.
3. FORKS: one branched beam per fork tick, --forks ticks spread evenly inside each window
   (resample boundaries), half the population forked with HELD view offsets (beam_tas
   --branch-at ... --branch-view-only: U(-J, J) deg around the policy's own aim, redrawn every
   --branch-hold-secs for --branch-burst-secs, protected --branch-protect generations).
4. The fastest finishing line over all beams is reported (summary.json) and its beam_best.npz
   copied to <out>/best/ for tools/plan_to_bc.py; every crossing wave's npz is listed so the
   BC set can pool the K fastest of each.

No constant comes from a map: the stall rule is a field-distance margin, the forks are a count
spread over the policy's own stall, and the burst is the auto-spine branch's (auto_spine.py
--branch-*). No demo or record data enters (CLAUDE.md sections 0 and 0b): every line is the
policy's own proposals plus a held view offset, rooted at the true spawn.

    python tools/speed_forks.py --ckpt runs/X/ckpt.pt --map maps/M.bsp \\
        --field maps/M.goalc_32.npz --out runs/sf/a --max-ticks 6500

    python tools/speed_forks.py --selftest
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def stall_windows(d, margin: float, n: int = 1):
    """[(start, end)] tick windows of the n longest stalls of a line with per-tick goal-field
    distance d: an EVENT is a tick at which the running best improves on the best at the
    previous event by at least ``margin``; a stall is the gap between two consecutive events
    (or from the last event to the end of the line). Longest first."""
    d = np.asarray(d, np.float64)
    if len(d) == 0:
        return []
    ev = [0]
    best = d[0] if np.isfinite(d[0]) else np.inf
    for t in range(1, len(d)):
        if np.isfinite(d[t]) and d[t] <= best - margin:
            ev.append(t)
            best = d[t]
    ev.append(len(d) - 1)
    gaps = [(ev[i], ev[i + 1]) for i in range(len(ev) - 1) if ev[i + 1] > ev[i]]
    gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
    return gaps[:max(0, int(n))]


def fork_ticks(windows, k: int, gen_ticks: int, lead: int = 0):
    """k fork ticks spread evenly inside each window, snapped to resample boundaries (multiples
    of gen_ticks), at least one generation after the window opens and ``lead`` ticks before
    it closes; duplicates dropped, sorted."""
    out = set()
    for a, b in windows:
        lo = (a // gen_ticks + 1) * gen_ticks
        hi = b - max(lead, gen_ticks)
        if hi < lo:
            continue
        for x in np.linspace(lo, hi, max(1, int(k))):
            out.add(int(round(x / gen_ticks)) * gen_ticks)
    return sorted(t for t in out if t > 0)


def run(cmd, log: Path, timeout):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        try:
            p = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=str(ROOT), timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    print(f"  rc {rc} in {time.time() - t0:.0f} s -> {log}", flush=True)
    return rc


def beam(a, out: Path, tag: str, fork: int = 0, extra=()):
    bd = out / tag
    bd.mkdir(parents=True, exist_ok=True)
    tps = 1000.0 / a.tick
    cmd = [sys.executable, "-u", ROOT / "tools" / "beam_tas.py", a.ckpt, "--map", a.map,
           "--envs", a.envs, "--resample-every", a.resample, "--score", a.score,
           "--score-energy", a.score_energy,
           "--greedy-envs", a.greedy_envs, "--allow-nonfinisher", "--save-frontier",
           "--skip-gate", "--route-file", "none", "--max-ticks", a.max_ticks,
           "--keep-finishers", a.keep, "--gens", a.gens,
           "--torch-seed", a.seed, "--log-every", 10, "--out-dir", bd, *extra]
    if fork > 0:
        cmd += ["--branch-at", f"t{int(fork)}:{int(a.envs) // 2}",
                "--branch-burst", int(round(a.branch_burst_secs * tps)),
                "--branch-hold", int(round(a.branch_hold_secs * tps)),
                "--branch-jitter", int(round(a.branch_jitter)),
                "--branch-protect", int(a.branch_protect), "--branch-view-only"]
    rc = run(cmd, out / f"{tag}.log", a.timeout)
    res = {"tag": tag, "fork": int(fork), "rc": rc, "dir": str(bd)}
    sj = bd / "summary.json"
    if sj.exists():
        s = json.loads(sj.read_text(encoding="utf-8"))
        res.update(crossed=bool(s.get("crossed")), best_s=s.get("best_s"),
                   finishes=s.get("finishes"), branch_fired_tick=s.get("branch_fired_tick"),
                   branch_finishes=s.get("branch_finishes"), wall_s=s.get("search_wall_s"))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--map")
    ap.add_argument("--field")
    ap.add_argument("--out")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--greedy-envs", type=int, default=32)
    ap.add_argument("--resample", type=int, default=25, help="decisions per generation")
    ap.add_argument("--max-ticks", type=int, default=6500)
    ap.add_argument("--keep", type=int, default=16, help="beam_tas --keep-finishers")
    ap.add_argument("--gens", type=int, default=2,
                    help="beam_tas --gens: stop this many generations after the first finish")
    ap.add_argument("--seed", type=int, default=0, help="beam_tas --torch-seed")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--stall-margin", type=float, default=500.0,
                    help="u: a running-best improvement smaller than this does not end a stall")
    ap.add_argument("--stalls", type=int, default=1, help="how many stall windows to fork in")
    ap.add_argument("--forks", type=int, default=8, help="fork ticks per stall window")
    ap.add_argument("--branch-burst-secs", type=float, default=2.0)
    ap.add_argument("--branch-hold-secs", type=float, default=0.5)
    ap.add_argument("--branch-jitter", type=float, default=45.0)
    ap.add_argument("--branch-protect", type=int, default=3)
    ap.add_argument("--score", choices=["d", "dh"], default="d",
                    help="beam_tas --score for every beam: d = goal-field distance; dh = -d + "
                         "--score-energy x the energy height z + |v|^2/2g. A held offset that "
                         "drives the player into a wall trades ~1,000 u of energy for ~150 u of "
                         "height, and d alone ranks that crash first (uf2r2b2's CPU probe)")
    ap.add_argument("--score-energy", type=float, default=1.0,
                    help="--score dh: u of field distance one u of energy height is worth")
    ap.add_argument("--plain-extra", default="",
                    help="extra beam_tas flags for the plain beam only (one string)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    for k in ("ckpt", "map", "field", "out"):
        if getattr(a, k) is None:
            ap.error(f"--{k} is required")
    from auto_spine import ckpt_tick, load_field, field_d
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    a.tick = ckpt_tick(a.ckpt)
    gen_ticks = int(a.resample) * 4
    try:
        import torch
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        gen_ticks = int(a.resample) * int((ck.get("config") or {}).get("act_every") or 4)
    except Exception:                                    # noqa: BLE001
        pass
    field = load_field(a.field)
    res = {"ckpt": a.ckpt, "tick_ms": a.tick, "gen_ticks": gen_ticks, "beams": []}
    plain = beam(a, out, "plain", 0, a.plain_extra.split() if a.plain_extra else ())
    res["beams"].append(plain)
    fz = Path(plain["dir"]) / "beam_frontier.npz"
    if not fz.exists():
        print("plain beam saved no line - nothing to fork on", flush=True)
        (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 2
    z = np.load(fz, allow_pickle=False)
    d = field_d(field, z["states"])
    wins = stall_windows(d, a.stall_margin, a.stalls)
    forks = fork_ticks(wins, a.forks, gen_ticks)
    tps = 1000.0 / a.tick
    res.update(plain_len=int(len(d)), stalls=[[int(s), int(e)] for s, e in wins], forks=forks)
    print(f"plain line: {len(d)} ticks, best_s {plain.get('best_s')}; stalls "
          + ", ".join(f"{s}-{e} ({s / tps:.2f}-{e / tps:.2f} s)" for s, e in wins)
          + f"; forks at {forks}", flush=True)
    for t in forks:
        r = beam(a, out, f"fork_{t}", t)
        res["beams"].append(r)
        print(f"fork {t} ({t / tps:.2f} s): crossed {r.get('crossed')} best_s {r.get('best_s')} "
              f"branch finishes {r.get('branch_finishes')}", flush=True)
        (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    fin = [b for b in res["beams"] if b.get("crossed") and b.get("best_s")]
    if fin:
        best = min(fin, key=lambda b: b["best_s"])
        res["best"] = best
        bdir = out / "best"
        bdir.mkdir(exist_ok=True)
        for f in ("beam_best.npz", "beam_best.jsonl", "beam_frontier.npz", "summary.json"):
            src = Path(best["dir"]) / f
            if src.exists():
                shutil.copyfile(src, bdir / f)
        res["crossing_npz"] = [str(Path(b["dir"]) / "beam_best.npz") for b in
                               sorted(fin, key=lambda b: b["best_s"])]
        print(f"FASTEST: {best['tag']} {best['best_s']} s (plain {plain.get('best_s')} s)",
              flush=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("SPEED_FORKS " + json.dumps({k: res.get(k) for k in ("forks", "stalls")}
                                      | {"best": res.get("best", {}).get("best_s"),
                                         "plain": plain.get("best_s")}), flush=True)
    return 0 if fin else 2


def selftest() -> int:
    # a line that progresses, stalls (rises and falls back) for 1000 ticks, then progresses
    d = np.concatenate([np.linspace(30000, 29000, 200), 29000 + 2000 * np.sin(
        np.linspace(0, np.pi, 1000)), np.linspace(29000, 0, 3000)])
    w = stall_windows(d, 500.0, 1)
    assert len(w) == 1, w
    s, e = w[0]
    assert 90 <= s <= 200 and 1200 <= e <= 1300, w
    f = fork_ticks(w, 5, 100)
    assert f and f[0] >= s + 1 and f[-1] <= e - 100 and all(x % 100 == 0 for x in f), f
    assert len(set(f)) == len(f)
    # two stalls, ranked by length
    d2 = np.concatenate([np.full(300, 1000.0), np.linspace(1000, 0, 10), np.full(500, 0.0)])
    w2 = stall_windows(d2, 100.0, 2)
    assert w2[0][1] - w2[0][0] >= w2[1][1] - w2[1][0], w2
    print("speed_forks selftest OK", w, f, w2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
