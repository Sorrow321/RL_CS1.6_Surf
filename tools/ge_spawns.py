#!/usr/bin/env python3
"""ge_spawns.py - a Go-Explore archive (tools/explore_phase1.py) -> a train_fast --spawn-states file.

The archive is a REWARD-FREE search rooted only at the map's true spawn: no demo, no record, no
reward, no map constant. Its states are this run's own search states (PROTOCOL.md: "a search's
states rooted at the true spawn"), so the file goes to the trainer through `box.py launch
--self-states "<provenance>" -- --spawn-states <file> --spawn-states-frac F`.

What is kept: per --place-cell u place (position only), the state with the highest ENERGY HEIGHT
h = z + |v|^2 / (2 g) - the ranking explore_phase1's --select-energy already uses. An energy
archive holds many cells per place (energy buckets x heading); a uniform draw over all of them
spends most spawns on the low-energy remains of failed runs, while the place's best state is the
one a run can still go somewhere from. --all keeps every cell instead. Clocks (tick,
stuck_ticks) are zeroed, as the trainer does.

--watch S re-converts whenever the archive file changes (checked every S seconds) and replaces
the output atomically (write + rename): train_fast re-reads a --spawn-states file whose mtime
changed every 20 iterations, so a trainer launched on the first conversion follows a search that
keeps running beside it on the same box.

    python tools/ge_spawns.py --archive runs/ge/archive.npz --out runs/ge_spawns.npy
    python tools/ge_spawns.py --archive runs/ge/archive.npz --out runs/ge_spawns.npy --watch 120
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE  # noqa: E402

G = 800.0


def energy_height(st: np.ndarray) -> np.ndarray:
    """h = z + |v|^2 / (2 g) of STATE_DTYPE rows."""
    v = st["velocity"].astype(np.float64)
    return st["origin"][:, 2].astype(np.float64) + (v * v).sum(1) / (2.0 * G)


def best_per_place(st: np.ndarray, place_cell: float) -> np.ndarray:
    """Indices of the most energetic state of every place_cell-u place (ties: the earlier row)."""
    p = np.floor(st["origin"].astype(np.float64) / float(place_cell)).astype(np.int64)
    p -= p.min(0)
    span = p.max(0) + 1
    key = p[:, 0] + span[0] * (p[:, 1] + span[1] * p[:, 2])
    h = energy_height(st)
    # sort by place, then by energy DEscending, then by row: the first row of each place wins
    order = np.lexsort((np.arange(len(st)), -h, key))
    first = np.ones(len(order), bool)
    first[1:] = key[order][1:] != key[order][:-1]
    return np.sort(order[first])


def convert(archive: Path, place_cell: float, keep_all: bool, max_rows: int,
            seed: int, speed_scale: tuple[float, float] | None = None,
            view_on_velocity: bool = False) -> tuple[np.ndarray, str]:
    with np.load(archive, allow_pickle=False) as z:
        st = z["state"]
    idx = np.arange(len(st)) if keep_all else best_per_place(st, place_cell)
    if max_rows > 0 and len(idx) > max_rows:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_rows, replace=False))
    rows = np.zeros(len(idx), STATE_DTYPE)
    for f in STATE_DTYPE.names:
        rows[f] = st[f][idx]
    rows["tick"] = 0
    rows["stuck_ticks"] = 0
    if speed_scale is not None:
        # --speed-scale: each row's velocity times U(lo, hi), the recipe's own-state perturbation
        # (--respawn-front-perturb-speed) applied to the search's states
        k = np.random.default_rng(seed + 1).uniform(speed_scale[0], speed_scale[1], len(rows))
        rows["velocity"] = (rows["velocity"] * k[:, None]).astype(np.float32)
    if view_on_velocity:
        # --view-on-velocity: the view re-aimed along the horizontal velocity (|v_h| >= 100 u/s),
        # where a --view-absolute velocity policy looks; the explorer's bursts held random offsets
        v = rows["velocity"].astype(np.float64)
        fast = np.hypot(v[:, 0], v[:, 1]) >= 100.0
        rows["yaw"][fast] = np.degrees(np.arctan2(v[fast, 1], v[fast, 0])).astype(np.float32)
    h = energy_height(rows)
    note = (f"{len(st):,} archive cells -> {len(rows):,} spawn states "
            f"({'all cells' if keep_all else f'the most energetic per {place_cell:g} u place'}"
            f"{f', subsampled to {max_rows:,}' if max_rows > 0 and len(idx) == max_rows else ''}"
            f"{f', velocity x U({speed_scale[0]:g}, {speed_scale[1]:g})' if speed_scale else ''}"
            f"{', view on the velocity' if view_on_velocity else ''}); "
            f"energy height p50 {np.median(h):,.0f} p90 {np.percentile(h, 90):,.0f} "
            f"max {h.max():,.0f}")
    return rows, note


def write_atomic(rows: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npy")
    np.save(tmp, rows)
    os.replace(tmp, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", required=True, help="explore_phase1's archive.npz")
    ap.add_argument("--out", required=True, help="the .npy of STATE_DTYPE rows to write")
    ap.add_argument("--place-cell", type=float, default=64.0,
                    help="place size in u for the per-place energy maximum (default 64, the "
                         "explorer's --cell)")
    ap.add_argument("--all", action="store_true", help="keep every archive cell instead")
    ap.add_argument("--max", type=int, default=0,
                    help="subsample to at most this many rows (0 = no cap)")
    ap.add_argument("--speed-scale", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="multiply each row's velocity by U(LO, HI) (default: unscaled)")
    ap.add_argument("--view-on-velocity", action="store_true",
                    help="re-aim each row's view along its horizontal velocity (|v_h| >= 100 u/s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--watch", type=float, default=0.0,
                    help="re-convert whenever the archive changes, checking every this many "
                         "seconds (0 = convert once)")
    a = ap.parse_args(argv)
    archive, out = Path(a.archive), Path(a.out)
    last = None
    while True:
        try:
            mt = archive.stat().st_mtime
        except OSError:
            mt = None
        if mt is not None and mt != last:
            try:
                # a snapshot being written is not a valid zip yet: skip it, the next check
                # sees the finished file (its mtime moves again when the write ends)
                rows, note = convert(archive, a.place_cell, a.all, a.max, a.seed,
                                     tuple(a.speed_scale) if a.speed_scale else None,
                                     a.view_on_velocity)
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
                print(f"{time.strftime('%H:%M:%S')} {archive}: not readable yet ({e})",
                      flush=True)
            else:
                write_atomic(rows, out)
                last = mt
                print(f"{time.strftime('%H:%M:%S')} {note} -> {out}", flush=True)
        if a.watch <= 0:
            return 0 if last is not None else 1
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
