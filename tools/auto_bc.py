#!/usr/bin/env python3
"""auto_bc.py - one expert-iteration data round from the policy's OWN search (uf2r2b2, 2026-10-31).

The search half of the expert loop (tools/exit_local.py's PLAN + DISTIL), written as one step a
trainer can run beside its training (train_fast --auto-bc), the way tools/auto_spine.py is the
search half of the search -> curriculum loop:

1. PLAN: --waves beam_tas populations from the TRUE start on the checkpoint (the policy's own
   proposals, a greedy floor, the leaders by --score cloned every 25 decisions, the --keep
   fastest finishers kept), torch seeds base+0 .. base+W-1.
2. DISTIL: tools/plan_to_bc.py over every wave that finished (one spawn, so one call): the BC
   rows (slower lines down-weighted by --line-weight-decay) and the fastest line's per-tick spine
   + held keys (for train_fast --demo-file / --demo-keys).
3. The last stdout line is ``AUTO_BC {json}``: best_s (the fastest line), the files, their md5.

No constant here comes from a map: a population size, a ranking, a seed, a decay. No demo or record
data enters (CLAUDE.md section 0): every line is the checkpoint's own proposals from the map spawn.
On surf_unitfarmer2 the incumbent's greedy line is 42.50 s and 96-env CPU waves of its own
proposals finish in 41.29-41.59 s (`--score dh` 41.29 / 41.32, `--score d` 41.59).

    python tools/auto_bc.py --ckpt runs/X/ckpt_latest.pt --map maps/M.bsp --out runs/X/auto_bc/k01
    python tools/auto_bc.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, log: Path, timeout: float) -> int:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        try:
            rc = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    print(f"  rc {rc} in {time.time() - t0:.0f} s -> {log}", flush=True)
    return rc


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wave_result(wdir: Path) -> dict:
    """-> {crossed, best_s, npz} of one beam_tas wave directory (summary.json + beam_best.npz)."""
    sj = wdir / "summary.json"
    out = {"dir": str(wdir), "crossed": False, "best_s": None, "npz": None}
    if not sj.exists():
        return out
    s = json.loads(sj.read_text(encoding="utf-8"))
    npz = wdir / "beam_best.npz"
    if s.get("crossed") and s.get("best_s") and npz.exists():
        out.update(crossed=True, best_s=float(s["best_s"]), npz=str(npz))
    return out


def pick(waves) -> list:
    """The crossing waves, fastest first (plan_to_bc takes the best line's spine from its first
    file's lines... it ranks every kept line by finish time itself; the order is for the log)."""
    return sorted([w for w in waves if w["crossed"]], key=lambda w: w["best_s"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--map")
    ap.add_argument("--out")
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--greedy-envs", type=int, default=64)
    ap.add_argument("--score", choices=["d", "dh"], default="dh")
    ap.add_argument("--score-energy", type=float, default=1.0)
    ap.add_argument("--keep", type=int, default=16, help="beam_tas --keep-finishers per wave")
    ap.add_argument("--max-ticks", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0, help="the first wave's torch seed")
    ap.add_argument("--line-weight-decay", type=float, default=20.0)
    ap.add_argument("--timeout", type=float, default=1800.0, help="per wave / distil (s)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    for k in ("ckpt", "map", "out"):
        if getattr(a, k) is None:
            ap.error(f"--{k} is required")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"ckpt": str(a.ckpt), "waves": [], "bc": None}
    for w in range(max(1, int(a.waves))):
        wdir = out / f"wave_{w}"
        wdir.mkdir(exist_ok=True)
        rc = run([sys.executable, "-u", ROOT / "tools" / "beam_tas.py", a.ckpt, "--map", a.map,
                  "--envs", a.envs, "--greedy-envs", a.greedy_envs, "--resample-every", 25,
                  "--score", a.score, "--score-energy", a.score_energy, "--allow-nonfinisher",
                  "--skip-gate", "--route-file", "none", "--max-ticks", a.max_ticks,
                  "--keep-finishers", a.keep, "--gens", 2, "--torch-seed", int(a.seed) + w,
                  "--log-every", 10, "--out-dir", wdir], out / f"wave_{w}.log", a.timeout)
        r = wave_result(wdir)
        r["rc"] = rc
        res["waves"].append(r)
        print(f"wave {w}: rc {rc}, crossed {r['crossed']}, best {r['best_s']} s", flush=True)
    fin = pick(res["waves"])
    if not fin:
        print("AUTO_BC " + json.dumps(res), flush=True)
        return 2
    res["best_s"] = fin[0]["best_s"]
    bc, spine = out / "bc.npz", out / "spine.npy"
    rc = run([sys.executable, "-u", ROOT / "tools" / "plan_to_bc.py", "--plan",
              *[w["npz"] for w in fin], "--ckpt", a.ckpt, "--out", bc, "--spine", spine,
              "--map", a.map, "--no-value-target", "--line-weight-decay", a.line_weight_decay,
              "--summary-out", out / "distil.json"], out / "distil.log", a.timeout)
    keys = spine.with_suffix(".keys.npy")
    if rc == 0 and bc.exists() and spine.exists():
        res.update(bc=str(bc), spine=str(spine), bc_md5=md5(bc), spine_md5=md5(spine),
                   keys=str(keys) if keys.exists() else None)
    res["distil_rc"] = rc
    print("AUTO_BC " + json.dumps(res), flush=True)
    return 0 if res["bc"] else 3


def parse_result(text: str):
    """The last ``AUTO_BC {json}`` line of a log, or None (the trainer's reader)."""
    for ln in str(text).splitlines()[::-1]:
        if ln.startswith("AUTO_BC "):
            try:
                return json.loads(ln[len("AUTO_BC "):])
            except ValueError:
                return None
    return None


def selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        for i, (crossed, best) in enumerate([(True, 41.5), (False, None), (True, 41.2)]):
            w = t / f"wave_{i}"
            w.mkdir()
            (w / "summary.json").write_text(json.dumps({"crossed": crossed, "best_s": best}))
            if crossed:
                (w / "beam_best.npz").write_bytes(b"x")
        ws = [wave_result(t / f"wave_{i}") for i in range(3)]
        assert [w["crossed"] for w in ws] == [True, False, True], ws
        p = pick(ws)
        assert [w["best_s"] for w in p] == [41.2, 41.5], p
        assert wave_result(t / "nope")["crossed"] is False
    assert parse_result("x\nAUTO_BC {\"best_s\": 41.2, \"bc\": \"a\"}\n")["best_s"] == 41.2
    assert parse_result("AUTO_BC {broken") is None and parse_result("") is None
    print("auto_bc selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
