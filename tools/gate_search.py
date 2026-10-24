#!/usr/bin/env python3
"""gate_search.py - one gate of the search -> curriculum loop, as one command (pbr3b1, 2026-10-24).

Round 2 crossed every gate the same way, by hand (autoresearch/NOTES.md, "The method so far"):
a policy-guided beam from the TRUE start on the newest weights (tools/beam_tas.py
--save-frontier); when the beam stalls at the same gate as the greedy line, Go-Explore with exact
chains rooted at the beam line's own states in the seconds before the stall
(tools/explore_phase1.py --exact-chains) and the deepest chain replayed (tools/ge_replay.py) and
spliced onto the line at its root; the result cut at its deepest point is the next
`train_fast --demo-file` spine. This runs those steps in that order and writes
<out>/spine.npy + <out>/gate_search.json. It launches nothing: the curriculum run goes through
`box.py launch --resume --self-states ...` as before.

    python tools/gate_search.py --ckpt runs_ckpt.pt --out runs/pbr3b1_gs1 --need 20.6

* `--need P`: the progress (% of the grader field) a beam line must reach for the GE stage to be
  skipped - set it a little past the greedy line's best, so a beam that only re-finds the greedy
  line falls through to GE.
* GE roots: the beam line's states from `--ge-from` s to `--ge-to` s before the beam frontier's
  tick (the last resample before the population died), every `--ge-stride` ticks.
* Every constant is generic (cells, burst length, envs, minutes): nothing reads the map's
  geometry, a demo or a record. The spines are the policy's own (beam) and its own reward-free
  search (GE), the provenance `box.py launch --self-states` records.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import spine_tool  # noqa: E402


def run(cmd, log: Path) -> int:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        fh.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    return r.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=None, help="default: autoresearch/config.json's map")
    ap.add_argument("--field", default=None, help="default: autoresearch/config.json's field")
    ap.add_argument("--maxvel", type=float, default=None,
                    help="GE core sv_maxvelocity (default: the checkpoint config's maxvel)")
    ap.add_argument("--need", type=float, required=True,
                    help="progress %% a beam line must reach to skip the GE stage")
    ap.add_argument("--beam-envs", type=int, default=4096)
    ap.add_argument("--beam-greedy", type=int, default=32)
    ap.add_argument("--beam-ticks", type=int, default=3000)
    ap.add_argument("--beam-seed", type=int, default=0)
    ap.add_argument("--beam-keys-temp", type=float, default=1.0,
                    help="beam_tas --prop-keys-temp (the sampled envs' key temperature)")
    ap.add_argument("--beam-yaw-scale", type=float, default=1.0,
                    help="beam_tas --prop-yaw-scale (the sampled envs' yaw sigma multiplier)")
    ap.add_argument("--skip-beam", default=None, metavar="NPZ",
                    help="reuse a beam_frontier.npz instead of running the beam")
    ap.add_argument("--ge-minutes", type=float, default=12.0, help="0 = never run GE")
    ap.add_argument("--ge-envs", type=int, default=4096)
    ap.add_argument("--ge-threads", type=int, default=0)
    ap.add_argument("--ge-from", type=float, default=3.0)
    ap.add_argument("--ge-to", type=float, default=0.0)
    ap.add_argument("--ge-stride", type=int, default=10)
    ap.add_argument("--ge-cell", type=float, default=32.0)
    ap.add_argument("--ge-energy", type=float, default=16.0)
    ap.add_argument("--cut", default="contact",
                    help="spine_tool cut --end: 'contact' (default: the last map contact at or "
                         "before the deepest state, never mid-fall) or 'auto' (the deepest)")
    a = ap.parse_args(argv)

    cfg = json.loads((ROOT / "autoresearch" / "config.json").read_text(encoding="utf-8"))
    mp = a.map or str(ROOT / cfg["map"])
    field = a.field or str(ROOT / cfg["field"])
    d0 = float(cfg.get("start_d") or 0.0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "gate_search.log"
    f = spine_tool.load_field(field)
    res = {"ckpt": str(a.ckpt), "need": a.need, "t0": time.time()}
    prog = lambda d: 100.0 * (d0 - d) / d0

    # ---- stage 1: the policy-guided beam from the true start --------------------------------
    if a.skip_beam:
        fz = Path(a.skip_beam)
    else:
        bdir = out / "beam"
        rc = run([sys.executable, ROOT / "tools" / "beam_tas.py", a.ckpt,
                  "--envs", a.beam_envs, "--resample-every", 25, "--score", "d",
                  "--greedy-envs", a.beam_greedy, "--allow-nonfinisher", "--save-frontier",
                  "--skip-gate", "--route-file", "none", "--max-ticks", a.beam_ticks,
                  "--torch-seed", a.beam_seed, "--prop-keys-temp", a.beam_keys_temp,
                  "--prop-yaw-scale", a.beam_yaw_scale, "--out-dir", bdir], log)
        fz = bdir / "beam_frontier.npz"
        if rc != 0 or not fz.exists():
            raise SystemExit(f"beam_tas failed (exit {rc}); see {log}")
    z = np.load(fz, allow_pickle=False)
    line = spine_tool.zero_clocks(z["states"])
    np.save(out / "beam_line.npy", line)
    d = spine_tool.depths(line, f)
    ib = int(np.argmin(d))
    res["beam"] = {"file": str(fz), "n": int(len(line)), "best_idx": ib,
                   "prog_best": round(prog(d[ib]), 3), "frontier_tick": int(z["frontier_tick"])}
    print(f"beam line: {len(line)} states, best {prog(d[ib]):.3f}% at state {ib}", flush=True)
    spine_src = line
    if prog(d[ib]) < a.need and a.ge_minutes > 0:
        # ---- stage 2: Go-Explore rooted on the beam line before its stall -------------------
        ft = min(int(z["frontier_tick"]), len(line) - 1)
        lo = max(0, ft - int(round(a.ge_from * 100)))
        hi = max(lo + 1, ft - int(round(a.ge_to * 100)))
        roots = line[lo:hi]
        np.save(out / "ge_roots.npy", roots)
        gdir = out / "ge"
        maxvel = a.maxvel
        if maxvel is None:
            import torch
            ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
            maxvel = float((ck.get("config") or {}).get("maxvel") or 0.0) or None
        cmd = [sys.executable, ROOT / "tools" / "explore_phase1.py", "--map", mp, "--out", gdir,
               "--envs", a.ge_envs, "--cell", a.ge_cell, "--cell-energy", a.ge_energy,
               "--exact-chains", "--act-every", 4, "--decisions", 100,
               "--roots-spine", out / "ge_roots.npy", "--roots-stride", a.ge_stride,
               "--max-minutes", a.ge_minutes, "--ep-ticks", 30000, "--print-every", 20,
               "--goals", 0]
        if maxvel:
            cmd += ["--maxvel", maxvel]
        if a.ge_threads:
            cmd += ["--threads", a.ge_threads]
        res["ge"] = {"roots": [lo, hi], "stride": a.ge_stride, "minutes": a.ge_minutes}
        rc = run(cmd, log)
        if not (gdir / "archive.npz").exists():
            raise SystemExit(f"explore_phase1 wrote no archive (exit {rc}); see {log}")
        rc = run([sys.executable, ROOT / "tools" / "ge_replay.py", "--map", mp,
                  "--archive", gdir / "archive.npz", "--field", field,
                  "--out", out / "ge_chain.npy"], log)
        if (out / "ge_chain.npy").exists():
            chain = np.load(out / "ge_chain.npy", allow_pickle=False)
            spine_tool.main(["splice", "--line", str(out / "beam_line.npy"),
                             "--chain", str(out / "ge_chain.npy"), "--field", field,
                             "--out", str(out / "ge_line.npy")])
            gl = np.load(out / "ge_line.npy", allow_pickle=False)
            dg = spine_tool.depths(gl, f)
            res["ge"].update({"replay_exit": rc, "chain_n": int(len(chain)),
                              "prog_best": round(prog(float(dg.min())), 3)})
            print(f"GE line: {len(gl)} states, best {prog(float(dg.min())):.3f}%", flush=True)
            if dg.min() < d[ib]:
                spine_src = gl
    np.save(out / "source_line.npy", spine_src)
    spine_tool.main(["cut", "--src", str(out / "source_line.npy"), "--field", field,
                     "--end", a.cut, "--out", str(out / "spine.npy")])
    sp = np.load(out / "spine.npy", allow_pickle=False)
    ds = spine_tool.depths(sp, f)
    res["spine"] = {"file": str(out / "spine.npy"), "n": int(len(sp)),
                    "prog_end": round(prog(float(ds[-1])), 3),
                    "from": "ge" if spine_src is not line else "beam",
                    "past_need": bool(prog(float(ds[-1])) >= a.need)}
    res["wall_s"] = round(time.time() - res["t0"], 1)
    (out / "gate_search.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["spine"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
