#!/usr/bin/env python3
"""ge_stack_spine.py - stack exact Go-Explore chains of several stages onto a spine (pbr5b4, 2026-10-25).

A second explorer stage rooted at the first stage's own cells (tools/explore_phase1.py --exact-chains
--roots-spine <cells of stage 1>) finds what the first could not; its chain starts at a stage-1 cell,
whose chain starts on the spine. This replays each stage's chain to its end cell (tools/ge_replay.py
--cell), finds where it starts on the line built so far (an exact copy of a state there), cuts the line
at that state and appends the chain, with its views put along the velocity and its held keys
translated so the wish direction survives (the auto_spine rule: tools/spine_canon_view.py,
tools/spine_canon_keys.py). Writes the spine and its .keys.npy for train_fast --demo-file / --demo-keys.

    python tools/ge_stack_spine.py --spine S.npy --keys S.keys.npy --map maps/M.bsp \\
        --field maps/M.goalc_32.npz --stage runs/ge1/archive.npz:next --stage runs/ge2/archive.npz:1035240 \\
        --out runs/spine_new.npy

A stage's end cell is an archive index, or ``next``: the stage-1 cell that the next stage's chain
starts from (found by its state). Nothing here comes from a demo or a record: the spine is the
policy's own beam line and the chains are the explorer's own."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def find_cell_by_state(archive, state, tol: float = 0.5) -> int:
    """The archive cell whose state is ``state`` (origin within ``tol`` u, then the nearest velocity)."""
    z = np.load(archive, allow_pickle=False)
    o = z["state"]["origin"].astype(np.float64)
    dist = np.linalg.norm(o - np.asarray(state["origin"], np.float64), axis=1)
    near = np.flatnonzero(dist <= tol)
    if not len(near):
        raise SystemExit(f"no cell of {archive} at {np.asarray(state['origin']).round(1)} "
                         f"(nearest {dist.min():.2f} u)")
    v = z["state"]["velocity"][near].astype(np.float64)
    dv = np.linalg.norm(v - np.asarray(state["velocity"], np.float64), axis=1)
    return int(near[np.argmin(dv)])


def chain_root_state(archive, cell: int):
    from ge_replay import chain_to
    z = np.load(archive, allow_pickle=False)
    return z["state"][chain_to(z["parent"], int(cell))[0]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spine", required=True)
    ap.add_argument("--keys", default=None, help="the spine's held keys (.keys.npy); none = neutral")
    ap.add_argument("--map", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--stage", action="append", required=True, help="ARCHIVE:CELL or ARCHIVE:next")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1.0, help="u: a chain's root must sit this close "
                                                            "to a state of the line")
    a = ap.parse_args(argv)
    from auto_spine import NEUTRAL_KEYS, chain_keys, load_field
    from spine_canon_keys import translate
    from spine_canon_view import canon_view
    from surfgym.core import STATE_DTYPE
    field = load_field(a.field)
    line = np.load(a.spine)
    keys = (np.asarray(np.load(a.keys), np.int32) if a.keys
            else np.tile(np.asarray(NEUTRAL_KEYS, np.int32), (len(line), 1)))
    assert keys.shape == (len(line), 3), (keys.shape, len(line))
    stages = [s.rsplit(":", 1) for s in a.stage]
    # resolve 'next': the cell of stage k whose state is stage k+1's chain root
    cells = [None] * len(stages)
    for k in range(len(stages) - 1, -1, -1):
        arch, c = stages[k]
        if c == "next":
            if k + 1 >= len(stages) or cells[k + 1] is None:
                raise SystemExit("'next' needs a later stage with a cell")
            cells[k] = find_cell_by_state(arch, chain_root_state(stages[k + 1][0], cells[k + 1]))
        else:
            cells[k] = int(c)
    out = Path(a.out)
    info = {"spine": a.spine, "stages": []}
    for k, ((arch, _), cell) in enumerate(zip(stages, cells)):
        lp = out.with_name(out.stem + f"_stage{k + 1}.npy")
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "ge_replay.py"), "--map", a.map,
                            "--archive", arch, "--field", a.field, "--out", str(lp),
                            "--cell", str(cell)], capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode != 0:
            raise SystemExit(f"stage {k + 1}: ge_replay failed ({r.returncode}): {r.stderr[-400:]}")
        gl = np.load(lp)
        dist = np.linalg.norm(line["origin"].astype(np.float64)
                              - np.asarray(gl[0]["origin"], np.float64), axis=1)
        j = int(np.argmin(dist))
        if dist[j] > a.tol:
            raise SystemExit(f"stage {k + 1}: its root is {dist[j]:.2f} u from the line")
        ck = chain_keys(arch, field, len(gl), cell=cell)
        if ck is None:
            raise SystemExit(f"stage {k + 1}: the chain's link actions do not match its line")
        v = np.asarray(gl["velocity"][1:], np.float64)
        hd = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
        fast = np.hypot(v[:, 0], v[:, 1]) >= 100.0
        ck = ck.copy()
        ck[fast] = translate(ck[fast], np.asarray(gl["yaw"][1:], np.float64)[fast], hd[fast])
        chain, _m = canon_view(gl[1:])
        line = np.concatenate([np.asarray(line[:j + 1], STATE_DTYPE), np.asarray(chain, STATE_DTYPE)])
        keys = np.concatenate([keys[:j + 1], ck.astype(np.int32)])
        info["stages"].append({"archive": arch, "cell": int(cell), "root_index": j,
                               "root_err": float(dist[j]), "chain_len": int(len(gl))})
        print(f"stage {k + 1}: cell {cell}, root at line index {j} ({dist[j]:.2f} u), chain "
              f"{len(gl)} states -> line {len(line)}")
    line = np.array(line, copy=True)
    line["tick"] = 0
    line["stuck_ticks"] = 0
    np.save(out, line)
    np.save(out.with_suffix(".keys.npy"), keys.astype(np.int32))
    o = line["origin"].astype(np.float64)
    vv = line["velocity"].astype(np.float64)
    d = np.asarray(field.sample(o), np.float64)
    h = o[:, 2] + (vv * vv).sum(1) / 1600.0
    info.update(n=int(len(line)), end_d=float(d[-1]), end_h=float(h[-1]))
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(f"spine {out}: {len(line)} states, end d {d[-1]:,.0f}, end energy height {h[-1]:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
