#!/usr/bin/env python3
"""frontier_archive.py - Go-Explore phase 1 rooted at the policy's OWN frontier, as a spawn-state
source for the trainer's ``--spawn-states`` (which re-reads the file when its mtime changes).

    python tools/frontier_archive.py --roots runs/<run>/traj_<step>.jsonl --out <states.npy>
    python tools/frontier_archive.py --watch runs/<run> --roots <traj.jsonl> --out <states.npy> \\
        --minutes 360           # loop: re-root on every new greedy eval of the run

WHY (2026-10-20, sidistic past the CP2 slot). The incumbent's greedy line dies on a wall the goal
field routes straight into, and every perturbation of its own line - nudges, ez bursts, novel
spawns at T = 2 - stays within a few hundred units of that line. Novel spawns teleport the
policy's states into unvisited cells, but they keep the source's velocity (often meaningless after
a 256-2,048 u move), they skip every cell within one cell of the policy's line, and nothing a novel
episode discovers can be returned to (it is never harvested). A simulator search from the same
frontier states reaches regions none of that reaches (on sidistic, in 150 batches: the room past
the slot, the cavity under the pit ramp, the lower level - 13,782 cells, analysis in
autoresearch/NOTES.md). This tool keeps those states, physically flown from the policy's own,
and hands them to the trainer, so the policy trains where it has never been with velocities that
really occur there, and anything the archive or the policy finds beyond can be returned to.

WHAT (no map constant anywhere; the same constants on every map):
  * ROOTS - the last --secs seconds (default 6) before the end of EVERY episode of a greedy eval
    file (the trainer's own traj_<step>.jsonl, from the true start), one state every --every ticks.
    That is where the policy fails, by definition. No demo, no reference line.
  * CELLS - position (--cell 128 u) x horizontal speed (400 u/s buckets, capped at 9). A cell
    keeps the FIRST state that reached it (Go-Explore's archive), and only states whose position
    is on the goal field's reachable set (finite d) are admitted.
  * EXPLORE - each batch picks --envs cells with weight 1 / sqrt(1 + times chosen) (count-based,
    reward-free), restores them into a batched core and flies random action segments (4-24 ticks;
    the yaw target is the velocity heading + N(0, 45 deg) or uniform; fwd/side/duck random) for up
    to --horizon ticks, admitting every new cell an alive env enters (checked every 4 ticks). The
    core runs the trainer's physics and fail rules (teleport_fail; kill triggers, water, kill_z,
    5 blocked ticks).
  * OUTPUT - up to --cap rows, half drawn from the deepest quarter of the archive by goal-field d
    (the shell nearest the goal, as --respawn-novel's deep shell), half uniformly; written
    atomically (tmp + rename) so the trainer never reads a half-written file. Root cells are
    excluded: the reservoir already holds the policy's own line.

Analysis-only flags: --report prints where the archive's cells lie by depth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

SPD_BUCKET = 400.0
SPD_MAX_BUCKET = 9
CHECK_EVERY = 4                 # ticks between cell admissions
SEG_MIN, SEG_MAX = 4, 25        # random action segment length, ticks
YAW_SIGMA = 45.0                # deg around the velocity heading, half the segments
P_FWD = (0.25, 0.4, 0.35)       # back / none / forward
P_SIDE = (0.3, 0.4, 0.3)        # left / none / right
P_DUCK = 0.2


def episodes(path):
    """[(header, rows (n, k) float64)] of a record.py trajectory file (header, rows, trailer)."""
    eps, hdr, cur = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None:
                eps.append((hdr, np.asarray(cur, np.float64)))
                hdr, cur = None, []
            else:
                hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return eps


def roots_from_traj(path, template, secs=6.0, every=10):
    """STATE_DTYPE rows: the last ``secs`` seconds of every episode, one row every ``every`` ticks.
    A row is (origin, velocity, yaw, pitch, duck from the buttons, contact) - the full state of a
    player in the air; everything else is the template's (a fresh core state)."""
    out = []
    for hdr, r in episodes(path):
        if not len(r):
            continue
        tick_ms = float(hdr.get("tick_ms", 10.0))
        n = max(1, int(round(secs * 1000.0 / tick_ms)))
        for t in range(max(0, len(r) - n), len(r), max(1, int(every))):
            row = r[t]
            s = template.copy()
            s["origin"] = row[1:4]
            s["velocity"] = row[4:7]
            s["basevelocity"] = 0
            s["yaw"] = row[7]
            s["pitch"] = row[12] if len(row) > 12 else 0.0
            s["onground"] = -1 if row[9] == 0 else 0
            s["ducked"] = 1 if (int(row[8]) & 4) else 0
            s["induck"] = 0
            s["duck_time"] = 0
            s["stuck_ticks"] = 0
            s["tick"] = 0
            s["progress"] = 0
            s["best_progress"] = 0
            s["seg_hint"] = 0
            out.append(s)
    if not out:
        return np.zeros(0, template.dtype)
    return np.array(out, dtype=template.dtype)


def cell_keys(origin, velocity, cell):
    """int64 key per row: (x, y, z) cell of ``cell`` u and the horizontal-speed bucket."""
    o = np.asarray(origin, np.float64)
    v = np.asarray(velocity, np.float64)
    c = np.floor(o / cell).astype(np.int64) + (1 << 15)
    sp = np.minimum(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2) // SPD_BUCKET,
                    SPD_MAX_BUCKET).astype(np.int64)
    return (c[:, 0] & 0xFFFF) | ((c[:, 1] & 0xFFFF) << 16) | ((c[:, 2] & 0xFFFF) << 32) | (sp << 48)


class FrontierArchive:
    def __init__(self, core, field, cell=128.0, horizon=200, seed=0):
        self.core = core
        self.field = field
        self.cell = float(cell)
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(seed)
        self.n = core.num_envs
        core.reset(0)
        self.template = core.get_states()[0].copy()
        self.states = np.zeros(0, self.template.dtype)
        self.d = np.zeros(0, np.float64)
        self.chosen = np.zeros(0, np.int64)
        self.is_root = np.zeros(0, bool)
        self.known = set()
        self.batches = 0
        self.sim_ticks = 0

    def _admit(self, st, root=False):
        if not len(st):
            return 0
        d = self.field.sample(st["origin"].astype(np.float64))
        ok = d < self.field.reach_max - 0.5 * self.field.cell
        st, d = st[ok], d[ok]
        keys = cell_keys(st["origin"], st["velocity"], self.cell)
        keys, first = np.unique(keys, return_index=True)
        new = np.array([k not in self.known for k in keys.tolist()], bool)
        if not new.any():
            return 0
        idx = first[new]
        self.known.update(keys[new].tolist())
        rows = st[idx].copy()
        rows["tick"] = 0
        rows["stuck_ticks"] = 0
        self.states = np.concatenate([self.states, rows])
        self.d = np.concatenate([self.d, d[idx]])
        self.chosen = np.concatenate([self.chosen, np.zeros(len(idx), np.int64)])
        self.is_root = np.concatenate([self.is_root, np.full(len(idx), bool(root))])
        return int(len(idx))

    def add_roots(self, roots):
        return self._admit(roots, root=True)

    def explore(self):
        """One batch: pick cells by 1/sqrt(1+chosen), restore, fly random segments, admit."""
        if not len(self.states):
            return 0
        w = 1.0 / np.sqrt(1.0 + self.chosen)
        pick = self.rng.choice(len(self.states), self.n, p=w / w.sum())
        np.add.at(self.chosen, pick, 1)
        core, rng, n = self.core, self.rng, self.n
        for i, j in enumerate(pick):
            core.set_state(i, self.states[j])
        alive = np.ones(n, bool)
        seg = np.zeros(n, np.int64)
        yaw = np.zeros(n, np.float32)
        fw = np.ones(n, np.int32)
        sd = np.ones(n, np.int32)
        dk = np.zeros(n, np.int32)
        acts = np.zeros((n, 6), np.int32)
        acts[:, 1] = 3
        view = np.zeros((n, 2), np.float32)
        hmax = rng.integers(max(8, self.horizon // 8), self.horizon + 1, n)
        st = core.get_states()
        added = 0
        for t in range(self.horizon):
            new = seg <= 0
            if new.any():
                m = int(new.sum())
                seg[new] = rng.integers(SEG_MIN, SEG_MAX, m)
                hd = np.degrees(np.arctan2(st["velocity"][new, 1], st["velocity"][new, 0]))
                yaw[new] = np.where(rng.random(m) < 0.5, hd + rng.normal(0.0, YAW_SIGMA, m),
                                    rng.uniform(0.0, 360.0, m)) % 360.0
                fw[new] = rng.choice(3, m, p=P_FWD)
                sd[new] = rng.choice(3, m, p=P_SIDE)
                dk[new] = (rng.random(m) < P_DUCK).astype(np.int32)
            seg -= 1
            acts[:, 2] = fw
            acts[:, 3] = sd
            acts[:, 5] = dk
            view[:, 0] = yaw
            _, _, done, trunc, _ = core.step(acts, view=view)
            alive &= ~(done.astype(bool) | trunc.astype(bool))
            alive &= t < hmax
            self.sim_ticks += int(alive.sum())
            if t % CHECK_EVERY == CHECK_EVERY - 1:
                # the states are read every CHECK_EVERY ticks: admissions, and the heading a new
                # segment's yaw is drawn around (at most CHECK_EVERY - 1 ticks old)
                st = core.get_states()
                if alive.any():
                    added += self._admit(st[alive])
            if not alive.any():
                break
        self.batches += 1
        return added

    def output(self, cap=20000, shell=0.5, shell_width=0.25):
        """Up to ``cap`` non-root rows: ``shell`` of them from the deepest ``shell_width`` of the
        archive by d, the rest uniform (with replacement only if the archive is smaller)."""
        idx = np.flatnonzero(~self.is_root)
        if not len(idx):
            return np.zeros(0, self.template.dtype)
        if len(idx) <= cap:
            return self.states[idx].copy()
        d = self.d[idx]
        deep = idx[d <= np.quantile(d, shell_width)]
        k_deep = min(len(deep), int(round(cap * shell)))
        a = self.rng.choice(deep, k_deep, replace=False)
        b = self.rng.choice(idx, cap - k_deep, replace=False)
        return self.states[np.concatenate([a, b])].copy()


def write_atomic(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def newest_eval(run_dir):
    """The newest greedy eval of a run: traj_<step>.jsonl with the largest step."""
    best = None
    for p in Path(run_dir).glob("traj_*.jsonl"):
        m = re.fullmatch(r"traj_(\d+)\.jsonl", p.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    return best


def build(map_path, field_path, envs, threads=None):
    if threads:
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    from surfgym.core import SurfCore, default_config
    from surfgym.goalfield import load_goal_field
    cfg = default_config(int(envs), lidar_w=0, lidar_h=0, max_episode_ticks=10 ** 8,
                         view_mode=2, yaw_rate_max_deg=10.0)
    cfg.phys.sv_maxvelocity = 4000.0
    core = SurfCore(str(map_path), cfg)
    core.set_teleport_fail(True)
    return core, load_goal_field(str(field_path))


def report(arch, d0):
    d = arch.d[~arch.is_root]
    if not len(d):
        return "no cells"
    q = np.quantile(d, [0.0, 0.01, 0.25, 0.5])
    return ("cells %d (roots %d)  d min %.0f (%.2f%%)  p1 %.0f  p25 %.0f  p50 %.0f"
            % (len(arch.d), int(arch.is_root.sum()), q[0], 100 * (d0 - q[0]) / d0, q[1], q[2], q[3]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_src_sidistic.bsp"))
    ap.add_argument("--field", default=None, help="goal field .npz (default <map>.goal_32.npz)")
    ap.add_argument("--roots", action="append", default=[],
                    help="greedy eval traj file(s) whose episodes' last --secs are the roots")
    ap.add_argument("--watch", default=None,
                    help="a run dir: re-root on each new greedy eval (traj_<step>.jsonl)")
    ap.add_argument("--out", required=True, help="STATE_DTYPE .npy for --spawn-states")
    ap.add_argument("--secs", type=float, default=6.0)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--cell", type=float, default=128.0)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--batches", type=int, default=150, help="batches before the first write")
    ap.add_argument("--write-every", type=int, default=50, help="batches between writes (loop)")
    ap.add_argument("--minutes", type=float, default=0.0, help="loop this long (0 = one pass)")
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d0", type=float, default=None, help="start geodesic, for the report only")
    a = ap.parse_args(argv)
    field_path = a.field or str(Path(a.map).with_suffix("")) + ".goal_32.npz"
    core, field = build(a.map, field_path, a.envs, a.threads)
    arch = FrontierArchive(core, field, cell=a.cell, horizon=a.horizon, seed=a.seed)
    seen = set()
    for p in a.roots:
        r = roots_from_traj(p, arch.template, a.secs, a.every)
        print(f"roots: {len(r)} states from {p} -> {arch.add_roots(r)} cells", flush=True)
        seen.add(str(Path(p).resolve()))
    if a.watch:
        ne = newest_eval(a.watch)
        if ne and str(ne[1].resolve()) not in seen:
            r = roots_from_traj(ne[1], arch.template, a.secs, a.every)
            print(f"roots: {len(r)} states from {ne[1]} -> {arch.add_roots(r)} cells", flush=True)
            seen.add(str(ne[1].resolve()))
    if not len(arch.states):
        raise SystemExit("no roots: pass --roots <traj.jsonl> (or --watch a run with an eval)")
    d0 = float(a.d0) if a.d0 else float(field.sample(arch.states["origin"][:1].astype(np.float64))[0])
    t0 = time.time()
    last_w = 0
    while True:
        add = arch.explore()
        if arch.batches % 10 == 0:
            print(f"batch {arch.batches}: +{add}  {report(arch, d0)}  "
                  f"{arch.sim_ticks / max(1e-9, time.time() - t0):,.0f} sim ticks/s", flush=True)
        if arch.batches >= a.batches and arch.batches - last_w >= (a.write_every if last_w else 0):
            out = arch.output(a.cap)
            write_atomic(a.out, out)
            last_w = arch.batches
            print(f"wrote {len(out):,} states to {a.out} at batch {arch.batches}", flush=True)
            if a.minutes <= 0:
                break
        if a.minutes > 0 and time.time() - t0 > 60.0 * a.minutes:
            break
        if a.watch and arch.batches % 10 == 0:
            ne = newest_eval(a.watch)
            if ne and str(ne[1].resolve()) not in seen:
                try:
                    r = roots_from_traj(ne[1], arch.template, a.secs, a.every)
                    print(f"roots: {len(r)} states from {ne[1].name} -> {arch.add_roots(r)} "
                          f"new cells", flush=True)
                    seen.add(str(ne[1].resolve()))
                except (OSError, ValueError) as e:     # a file still being written
                    print(f"roots: {ne[1].name} not readable yet ({e})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
