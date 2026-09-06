#!/usr/bin/env python3
"""diversity_bench.py - does a temperature T DIVERSIFY a checkpoint's rollouts,
what does it cost in progress, and which knob does it better?

The benchmark the unstuck mechanic (train_fast.py --unstuck, docs/unstuck.md)
is judged on before it is trusted with a plateau. N rollouts of ONE checkpoint
from ONE fixed spawn state, at a list of temperatures T, under a selectable
diversification knob applied at SAMPLING time in the eval wrapper
(train_fast.TemperedTorchPolicy - the same sample_padded / sample_view code
path the trainer's rollout takes under --unstuck, so "the eval at T" IS the
trainer's behaviour policy at T):

  sigma  the Gaussian view heads' sigma x (1 + T); softmax(logits / (1 + T))
         on the categorical heads (the keys; and the bins of a discrete
         checkpoint)
  eps    per decision, each head's sample replaced by a uniform draw with
         probability p = min(0.5, 0.05 T) (the planner's --eps rule)
  both   the two together

T = 0 is the plain sampled policy (byte-identical to SampledTorchPolicy), and
the greedy policy is reported alongside as the reference every rollout is
compared against (N identical rollouts by construction - and checked).

Metrics per (knob, T):

  1. progress    order-only corridor progress (tools/eval_honesty.py's
                 corridor_progress_ordered, window 16, corridor 1500) per
                 rollout: mean, max, finishes, share past 205,440 u (the
                 wall on surf_src_cannonball)
  2. spread      the rollouts are time-aligned from one spawn; at every
                 second of spawn time the RMS distance of the ALIVE
                 rollouts from their medoid. The curve, its values at
                 25 / 50 / 75 % of the median episode length and at 60 s
                 (the wall entry on the wall checkpoint)
  3. branches    single-linkage clusters at 512 u of the end positions, and
                 of the positions at 30 s and 60 s
  4. coverage    distinct 256 u position cells visited by the union of the
                 rollouts, and NOVEL cells = cells whose count in the
                 checkpoint's own novelty table (ck["int_counts"], summed
                 over its view / speed bins) is zero - the number the
                 unstuck mechanic actually wants to raise; `rare` = cells
                 under 100 visits
  5. same-line   share of rollouts that stay within 100 u of the greedy
                 trajectory for their whole life (spatially: max over the
                 life of the distance to the nearest greedy point; and
                 time-aligned)

    python tools/diversity_bench.py C:/RL_Surf_base/runs/research/cyABSV/ckpt_8002732032.pt \\
        --map C:/RL_Surf/maps/surf_src_cannonball.bsp \\
        --route C:/RL_Surf/maps/surf_src_cannonball.route.npz \\
        --knob sigma --knob eps --out runs/research/divbench/wall

Writes <out>/bench.csv (one row per knob x T, plus the greedy row), bench.md
(the table), bench.png, spread.json (the curves) and one rollouts_*.npz per
row (positions, end ticks, progress per rollout). `--from-spine spine.npy
--at-tick T` starts every rollout from that mid-route state instead of the
eval's spawn (`--seed`, the seed of core.reset as record_ckpt uses it).

Worktree trap (CLAUDE.md): pass the MAIN checkout's map and route paths; a
"bake" line in the output means a cache miss and a rebuild on your clock.
Needs the GPU for the lidar; 64 envs at 64x32 is light.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from surfgym import SurfCore, default_config  # noqa: E402
from surfgym.core import STATE_DTYPE  # noqa: E402
from surfgym.route import ArcProgress  # noqa: E402
from surfgym.view import view_mode_code  # noqa: E402
from train_fast import (GreedyTorchPolicy, HeadPacker, Policy,  # noqa: E402
                        SampledTorchPolicy, TemperedTorchPolicy)
from eval_honesty import load_route  # noqa: E402

WALL_U = 205440.0           # surf_src_cannonball: the 88.8 % wall, in route units
TICK_MS = 10.0
# the two knobs of the design plus their union, and three ATTRIBUTION knobs
# that temper ONE component of the sigma knob at a time (which one costs
# the flight?): yaw = the yaw head's sigma x (1+T) only, pitch = the pitch
# head's sigma only, keys = logits / (1+T) on the categorical heads only
KNOBS = ("sigma", "eps", "both", "yaw", "pitch", "keys")


def knob_policy(policy, packer, device, lidar, core, act_every, knob, T):
    """The eval wrapper for one (knob, T): the plain SampledTorchPolicy at
    T = 0 (byte-identical by construction), TemperedTorchPolicy otherwise."""
    temp = 1.0 + T if knob in ("sigma", "both") else 1.0
    eps = eps_of(T) if knob in ("eps", "both") else 0.0
    kw = {}
    if knob in ("yaw", "pitch", "keys") and T > 0.0:
        nz = int(getattr(policy, "n_z", 2))
        if knob == "yaw":
            kw["view_scale"] = [1.0 + T] * (nz - 1) + [1.0]
        elif knob == "pitch":
            kw["view_scale"] = [1.0] * (nz - 1) + [1.0 + T]
        else:
            kw["keys_temp"] = 1.0 + T
    if temp == 1.0 and eps == 0.0 and not kw:
        return SampledTorchPolicy(policy, packer, device, lidar, core,
                                  act_every, 1)
    return TemperedTorchPolicy(policy, packer, device, lidar, core, act_every,
                               1, temp=temp, eps=eps, **kw)

# config keys this tool does NOT mirror: a checkpoint that sets any of them
# changes what the policy sees or what an action means in a way that would
# need record_ckpt.py's extra machinery. Refused rather than mis-rolled.
UNSUPPORTED = ("route_file", "act_hist", "obs_compass", "priv_critic", "chunk",
               "frame_stack", "mask_forward_air", "jump_cooldown",
               "duck_air_mask", "yaw_cond", "fix_pitch", "pitch_fixed",
               "goals", "race_latch", "race_latch_frac", "obs_reward",
               "maps", "heldout_maps")


def eps_of(T: float) -> float:
    """The eps knob's schedule: p = min(0.5, 0.05 T)."""
    return min(0.5, 0.05 * float(T))


# --------------------------------------------------------------------------
# checkpoint -> core / lidar / policy (the record_ckpt.py subset)
# --------------------------------------------------------------------------
def check_supported(cfg: dict) -> None:
    bad = [k for k in UNSUPPORTED if cfg.get(k)]
    if cfg.get("rnn") not in (None, "none"):
        bad.append("rnn")
    if cfg.get("tick_ms") not in (None, 10, 10.0):
        bad.append("tick_ms")
    if cfg.get("reward") != "race":
        bad.append("reward")
    if bad:
        raise SystemExit("diversity_bench does not mirror "
                         + ", ".join(f"{k}={cfg.get(k)!r}" for k in bad)
                         + " - record it with tools/record_ckpt.py, or "
                         "extend build_core / build_policy here")


def core_kwargs(cfg: dict, ep_ticks: int) -> dict:
    fix_pitch, pitch_fixed = cfg.get("fix_pitch"), cfg.get("pitch_fixed")
    pitch_rate = (0.0 if (fix_pitch is not None or pitch_fixed is not None)
                  else float(cfg.get("pitch_rate", -1.0)))
    kw = dict(spawn_mode=2, max_episode_ticks=int(ep_ticks), water_fail=1,
              sv_maxvelocity=float(cfg.get("maxvel", 2000.0)),
              yaw_adaptive=1 if cfg.get("yaw_adaptive") else 0,
              yaw_blend=float(cfg.get("yaw_blend") or 1.0),
              side_hold_ticks=int(cfg.get("side_hold") or 0),
              lidar_w=0, lidar_h=0, pitch_rate_max_deg=pitch_rate)
    if cfg.get("view_absolute"):
        kw["view_mode"] = view_mode_code(cfg.get("view_absolute"))
    return kw


def build_core(map_path: str, cfg: dict, n: int, ep_ticks: int,
               yaw_jitter=None) -> SurfCore:
    kw = core_kwargs(cfg, ep_ticks)
    if yaw_jitter is not None:
        kw["yaw_jitter_deg"] = float(yaw_jitter)
    core = SurfCore(map_path, default_config(num_envs=n, **kw),
                    tick_ms=TICK_MS)
    core.set_teleport_fail(True)
    return core


def goal_field_and_box(core: SurfCore, cfg: dict):
    from surfgym.goalfield import EuclidField, build_goal_field
    from surfgym.mapfleet import map_tag
    from surfgym.vision import pick_cell
    from surfgym.zones import load_zones
    zones = load_zones(core.bsp_path)
    tag = map_tag(Path(core.bsp_path).stem)
    cells = dict(cfg.get("map_cells") or {})
    cell = float(cells.get(tag, cfg.get("lidar_cell") or pick_cell(core)))
    gcell = cell
    gcells, gc = cfg.get("goal_cells"), cfg.get("goal_cell")
    if isinstance(gcells, dict) and gcells:
        gcell = float(gcells.get(tag, gcell))
    elif gc and not (isinstance(gc, str) and "," in gc):
        gcell = float(gc)
    t0 = time.perf_counter()
    gf = (EuclidField(zones["end"]) if cfg.get("race_dist") == "euclid"
          else build_goal_field(core, zones["end"], cell=gcell))
    dt = time.perf_counter() - t0
    if dt > 20.0:
        print(f"WARNING: the goal field took {dt:.0f}s - that was a BAKE, "
              f"not a cache hit (worktree trap, CLAUDE.md)")
    return gf, zones, cell


def race_start_pool(core: SurfCore, gf):
    from surfgym.rewards import map_spawn_pool
    raw = map_spawn_pool(core)
    p = map_spawn_pool(core, yaw=gf.descent_yaw(raw["origin"]))
    p["pitch"] = -10.0
    return p


def build_lidar(core: SurfCore, cfg: dict, cell: float, device):
    from surfgym.vision import GpuLidar
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    return GpuLidar(core, lw, lh,
                    hfov_deg=float(cfg.get("lidar_hfov") or 120.0),
                    vfov_deg=float(cfg.get("lidar_vfov") or 90.0),
                    range_units=float(cfg.get("lidar_range", 2000.0)),
                    near_range=cfg.get("lidar_near"), cell=cell,
                    device=device, surf_mask=bool(cfg.get("surf_mask", 0)),
                    pinhole=bool(cfg.get("pinhole", 0)),
                    normals=bool(cfg.get("normals", 0)))


def build_policy(ck: dict, core: SurfCore, lidar, device) -> Policy:
    cfg = ck.get("config") or {}
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    policy = Policy(core.obs_dim + lw * lh * lidar.channels, lw, lh,
                    emb=int(cfg.get("emb", 256)),
                    hidden=int(cfg.get("hidden", 256)),
                    gps=bool(cfg.get("gps", True)),
                    trunk=str(cfg.get("trunk") or "plain"),
                    tower_depth=int(cfg.get("tower_depth") or 2),
                    conv_mult=int(cfg.get("conv_mult") or 1),
                    in_ch=lidar.channels,
                    view_continuous=bool(cfg.get("view_continuous")),
                    view_absolute=(cfg.get("view_absolute") or None)
                    ).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy


# --------------------------------------------------------------------------
# the novelty table's position cells (RaceReward._cells without the view /
# speed bins; tests/python/test_unstuck.py pins it against the reward)
# --------------------------------------------------------------------------
def cell_layout(core: SurfCore, cell: float):
    mins, maxs = core.map_bounds()
    mins = mins.astype(np.float64)
    dims = tuple(int(np.ceil((maxs[i] - mins[i]) / cell)) + 1 for i in range(3))
    return mins, dims


def pos_cells(xyz: np.ndarray, mins, dims, cell: float) -> np.ndarray:
    p = np.asarray(xyz, np.float64).reshape(-1, 3)
    ix = np.clip(((p[:, 0] - mins[0]) // cell).astype(np.int64), 0, dims[0] - 1)
    iy = np.clip(((p[:, 1] - mins[1]) // cell).astype(np.int64), 0, dims[1] - 1)
    iz = np.clip(((p[:, 2] - mins[2]) // cell).astype(np.int64), 0, dims[2] - 1)
    return ix + dims[0] * (iy + dims[1] * iz)


def position_counts(ck: dict, map_stem: str, dims, int_view: int,
                    int_speed: int):
    """ck['int_counts'] summed over its view x speed bins -> (n_pos,) int64,
    or None when the checkpoint carries no table for this map."""
    ic = ck.get("int_counts")
    if ic is None:
        return None
    if isinstance(ic, dict):
        ic = ic.get(map_stem)
        if ic is None:
            return None
    arr = np.asarray(ic)
    n_pos = int(dims[0] * dims[1] * dims[2])
    per = max(1, int_view) * max(1, int_speed)
    if arr.size != n_pos * per:
        raise SystemExit(f"int_counts has {arr.size:,} cells; expected "
                         f"{n_pos:,} position cells x {per} view/speed bins "
                         f"({dims}, int_view {int_view}, int_speed {int_speed})")
    return arr.reshape(n_pos, per).sum(axis=1, dtype=np.int64)


# --------------------------------------------------------------------------
# rollouts
# --------------------------------------------------------------------------
def roll(core: SurfCore, pol, seed: int, max_ticks: int, sample_seed: int):
    """N time-aligned rollouts from the core's one-entry spawn pool.
    -> pos (Tm, N, 3) float32 with NaN after an episode ends, end_tick (N,)
    (the tick the episode ended ON; -1 = still alive at Tm), fin (N,) bool
    (the core's goal_hits), kind (N,) 0 alive / 1 done / 2 trunc."""
    torch.manual_seed(int(sample_seed))
    obs = core.reset(int(seed))
    n = core.num_envs
    pos = np.full((max_ticks, n, 3), np.nan, np.float32)
    alive = np.ones(n, bool)
    end_tick = np.full(n, -1, np.int64)
    fin = np.zeros(n, bool)
    kind = np.zeros(n, np.int8)
    sv = core.states_view
    t = 0
    for t in range(max_ticks):
        o = np.asarray(sv["origin"])
        pos[t, alive] = o[alive]
        act = pol.act(obs)
        view = getattr(pol, "view", None)
        if view is None:
            obs, _r, done, trunc, _term = core.step(act)
        else:
            obs, _r, done, trunc, _term = core.step(act, view=view)
        d = np.asarray(done).astype(bool)
        tr = np.asarray(trunc).astype(bool)
        goal = np.asarray(core.goal_hits).astype(bool)
        ended = (d | tr) & alive
        fin |= goal & alive
        kind[ended & d] = 1
        kind[ended & ~d] = 2
        end_tick[ended] = t
        alive &= ~ended
        if not alive.any():
            break
    return pos[:t + 1], end_tick, fin, kind


def forward_fill(pos: np.ndarray) -> np.ndarray:
    """Hold every rollout at its last recorded position after it ends."""
    out = pos.copy()
    for i in range(pos.shape[1]):
        col = out[:, i, 0]
        bad = np.isnan(col)
        if bad.any():
            last = np.flatnonzero(~bad)
            if len(last) == 0:
                continue
            out[bad, i] = out[last[-1], i]
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def order_only_progress(pos: np.ndarray, pts, spacing: float,
                        corridor: float = 1500.0, window: int = 16):
    """eval_honesty.corridor_progress_ordered for every rollout at once:
    the same ArcProgress rule (local window, corridor gate, order-only),
    vectorised over the rollouts. pos is forward-filled, so an ended
    rollout sits still and its arc cannot move. -> (N,) best arc reached."""
    ap = ArcProgress(np.asarray(pts, np.float64), spacing, corridor=corridor,
                     window=window)
    p = np.asarray(pos, np.float64)
    ap.reset(p[0])
    best = ap.arc.copy()
    for t in range(1, len(p)):
        ap.advance(p[t])
        np.maximum(best, ap.arc, out=best)
    return best


def medoid_rms(P: np.ndarray) -> float:
    """RMS distance of the points from their medoid (0 for < 2 points)."""
    k = len(P)
    if k < 2:
        return 0.0
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    m = int(np.argmin(d.sum(1)))
    return float(np.sqrt((d[m] ** 2).mean()))


def single_linkage(P: np.ndarray, thresh: float) -> int:
    """Number of single-linkage clusters at ``thresh`` (union-find)."""
    k = len(P)
    if k == 0:
        return 0
    parent = list(range(k))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    ii, jj = np.nonzero(np.triu(d <= thresh, 1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(a) for a in range(k)})


def alive_at(end_tick: np.ndarray, t: int, tm: int) -> np.ndarray:
    """Rollouts whose episode had not ended before tick t."""
    return (end_tick < 0) | (end_tick >= t) if t < tm else np.zeros(len(end_tick), bool)


def spread_curve(pos: np.ndarray, end_tick: np.ndarray, every: int = 100):
    """-> (ticks (S,), rms (S,), n_alive (S,)) at every ``every`` ticks; the
    RMS is NaN wherever fewer than two rollouts are alive (no spread to
    speak of - the curve ends rather than dropping to zero)."""
    tm = pos.shape[0]
    ticks, rms, nal = [], [], []
    for t in range(0, tm, every):
        a = alive_at(end_tick, t, tm)
        P = pos[t, a]
        P = P[~np.isnan(P[:, 0])]
        ticks.append(t)
        nal.append(int(len(P)))
        rms.append(medoid_rms(P) if len(P) >= 2 else float("nan"))
    return np.asarray(ticks), np.asarray(rms), np.asarray(nal)


def branches_at(pos, end_tick, t, thresh=512.0):
    tm = pos.shape[0]
    if t >= tm:
        return 0
    a = alive_at(end_tick, t, tm)
    P = pos[t, a]
    P = P[~np.isnan(P[:, 0])]
    return single_linkage(P, thresh)


def end_positions(pos, end_tick):
    tm = pos.shape[0]
    idx = np.where(end_tick < 0, tm - 1, end_tick)
    return pos[idx, np.arange(pos.shape[1])]


def same_line_shares(pos, end_tick, gpos, gend, tol=100.0, pre=6000):
    """(spatial share, time-aligned share, spatial share over the first
    ``pre`` ticks) of rollouts within ``tol`` of the greedy trajectory.
    Spatial = the max over the life of the distance to the nearest greedy
    point; time-aligned = the max over the common life of the distance to
    the greedy position at the SAME tick, and the two lives within 1 s.
    The ``pre`` variant scores the approach to the wall only, so a rollout
    that rode the line and then fell differently still counts."""
    from scipy.spatial import cKDTree
    tm, n = pos.shape[:2]
    g_len = (gend + 1) if gend >= 0 else len(gpos)
    tree = cKDTree(gpos[:g_len])
    spatial = np.zeros(n, bool)
    timed = np.zeros(n, bool)
    early = np.zeros(n, bool)
    for i in range(n):
        L = (end_tick[i] + 1) if end_tick[i] >= 0 else tm
        P = pos[:L, i]
        ok = ~np.isnan(P[:, 0])
        P = P[ok]
        if len(P) == 0:
            continue
        dn = tree.query(P)[0]
        spatial[i] = float(dn.max()) <= tol
        early[i] = float(dn[:pre].max()) <= tol
        m = min(L, g_len)
        dt = np.sqrt(((pos[:m, i] - gpos[:m]) ** 2).sum(-1))
        timed[i] = (bool(np.nanmax(dt) <= tol) if m > 0 else False) \
            and abs(L - g_len) <= 100
    return float(spatial.mean()), float(timed.mean()), float(early.mean())


def score(tag, knob, T, pos, end_tick, fin, pts, spacing, mins, dims, cell,
          counts, greedy):
    """All five metric families for one batch of rollouts -> a flat dict."""
    tm, n = pos.shape[:2]
    posf = forward_fill(pos)
    prog = order_only_progress(posf, pts, spacing)
    lens = np.where(end_tick < 0, tm, end_tick + 1)
    med_len = int(np.median(lens))
    ticks, rms, nal = spread_curve(pos, end_tick)

    def rms_at(t):
        j = int(np.argmin(np.abs(ticks - t)))
        return float(rms[j]) if abs(int(ticks[j]) - t) <= 50 else float("nan")

    finite = rms[np.isfinite(rms)]

    ends = end_positions(posf, end_tick)
    cells_all = pos_cells(posf.reshape(-1, 3)[~np.isnan(pos.reshape(-1, 3)[:, 0])],
                          mins, dims, cell)
    uniq = np.unique(cells_all)
    if counts is not None:
        novel = int((counts[uniq] == 0).sum())
        rare = int((counts[uniq] < 100).sum())
    else:
        novel = rare = -1
    if greedy is not None:
        gpos, gend = greedy
        same_sp, same_t, same_pre = same_line_shares(posf, end_tick, gpos, gend)
    elif knob == "greedy":
        same_sp = same_t = same_pre = 1.0      # the reference, by definition
    else:
        same_sp = same_t = same_pre = float("nan")
    row = {
        "tag": tag, "knob": knob, "T": float(T),
        "temp": (1.0 + float(T) if knob in ("sigma", "both", "yaw", "pitch",
                                            "keys") else 1.0),
        "eps": eps_of(T) if knob in ("eps", "both") else 0.0,
        "n": int(n),
        "prog_mean": float(prog.mean()), "prog_max": float(prog.max()),
        "prog_min": float(prog.min()),
        "finishes": int(fin.sum()),
        "past_wall": float((prog > WALL_U).mean()),
        "len_med_s": med_len / 100.0, "len_mean_s": float(lens.mean()) / 100.0,
        "spread_25": rms_at(int(0.25 * med_len)),
        "spread_50": rms_at(int(0.50 * med_len)),
        "spread_75": rms_at(int(0.75 * med_len)),
        "spread_60s": rms_at(6000),
        "spread_max": float(finite.max()) if len(finite) else float("nan"),
        "alive_60s": int(alive_at(end_tick, 6000, tm).sum()),
        "branches_end": single_linkage(ends, 512.0),
        "branches_30s": branches_at(pos, end_tick, 3000),
        "branches_60s": branches_at(pos, end_tick, 6000),
        "cells": int(len(uniq)), "novel_cells": novel, "rare_cells": rare,
        "same_line": same_sp, "same_line_timed": same_t,
        "same_line_60s": same_pre,
    }
    curve = {"ticks": ticks.tolist(),
             "rms": [(round(float(v), 1) if v == v else None) for v in rms],
             "alive": nal.tolist()}
    return row, curve, prog


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
COLS = ["tag", "knob", "T", "temp", "eps", "n", "prog_mean", "prog_max",
        "prog_min", "finishes", "past_wall", "len_med_s", "len_mean_s",
        "spread_25", "spread_50", "spread_75", "spread_60s", "spread_max",
        "alive_60s", "branches_end", "branches_30s", "branches_60s", "cells",
        "novel_cells", "rare_cells", "same_line", "same_line_timed",
        "same_line_60s"]


def fmt(v):
    if isinstance(v, float):
        if v != v:
            return "n/a"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.3g}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def md_table(rows) -> str:
    head = ["knob", "T", "prog mean", "prog max", "past wall", "fin",
            "len med s", "spread 25/50/75 %", "spread 60 s", "branches "
            "end/30s/60s", "cells", "novel", "rare", "same-line life/60s"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join([
            r["tag"] if r["knob"] == "greedy" else r["knob"],
            fmt(r["T"]), fmt(r["prog_mean"]), fmt(r["prog_max"]),
            f"{100 * r['past_wall']:.0f}%", str(r["finishes"]),
            fmt(r["len_med_s"]),
            "/".join(fmt(r[k]) for k in ("spread_25", "spread_50", "spread_75")),
            fmt(r["spread_60s"]) + f" ({r['alive_60s']} alive)",
            f"{r['branches_end']}/{r['branches_30s']}/{r['branches_60s']}",
            str(r["cells"]), str(r["novel_cells"]), str(r["rare_cells"]),
            (f"{100 * r['same_line']:.0f}% / {100 * r['same_line_60s']:.0f}%"
             if r["same_line"] == r["same_line"] else "n/a")]) + " |")
    return "\n".join(out)


def plot(rows, curves, png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    knobs = [k for k in KNOBS if any(r["knob"] == k for r in rows)]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title)
    mk = {"sigma": "o-", "eps": "s--", "both": "^:", "yaw": "v-.",
          "pitch": "d-.", "keys": "x-."}
    for k in knobs:
        rr = sorted((r for r in rows if r["knob"] == k), key=lambda r: r["T"])
        Ts = [r["T"] for r in rr]
        ax[0, 0].plot(Ts, [r["prog_mean"] / 1000 for r in rr], mk[k],
                      label=f"{k} mean")
        ax[0, 0].plot(Ts, [r["prog_max"] / 1000 for r in rr], mk[k],
                      alpha=0.4, label=f"{k} max")
        ax[1, 0].plot(Ts, [r["branches_end"] for r in rr], mk[k],
                      label=f"{k} end")
        ax[1, 0].plot(Ts, [r["branches_60s"] for r in rr], mk[k], alpha=0.4,
                      label=f"{k} 60 s")
        ax[1, 1].plot(Ts, [r["novel_cells"] for r in rr], mk[k],
                      label=f"{k} novel")
        ax[1, 1].plot(Ts, [r["cells"] for r in rr], mk[k], alpha=0.4,
                      label=f"{k} distinct")
    g = [r for r in rows if r["knob"] == "greedy"]
    if g:
        ax[0, 0].axhline(g[0]["prog_mean"] / 1000, color="k", lw=0.8,
                         label="greedy")
    ax[0, 0].axhline(WALL_U / 1000, color="r", lw=0.8, ls=":", label="wall")
    ax[0, 0].set_xlabel("T"); ax[0, 0].set_ylabel("order-only progress, ku")
    ax[0, 0].set_title("progress vs T"); ax[0, 0].legend(fontsize=7)
    ls = {"sigma": "-", "eps": "--", "both": ":", "yaw": "-.", "pitch": "-.",
          "keys": "-."}
    cmap = plt.get_cmap("viridis")
    for key, c in curves.items():
        knob, T = key
        if knob == "greedy":
            continue
        col = cmap(min(1.0, float(T) / 4.0))
        ax[0, 1].plot(np.asarray(c["ticks"]) / 100.0,
                      [np.nan if v is None else v for v in c["rms"]],
                      ls[knob], color=col, lw=1.2, label=f"{knob} T={T:g}")
    ax[0, 1].axvline(60, color="r", lw=0.8, ls=":")
    ax[0, 1].set_xlabel("spawn time, s"); ax[0, 1].set_ylabel("RMS from medoid, u")
    ax[0, 1].set_title("spread vs time (alive rollouts)"); ax[0, 1].set_yscale("symlog")
    ax[0, 1].legend(fontsize=6, ncol=2)
    ax[1, 0].set_xlabel("T"); ax[1, 0].set_ylabel("single-linkage clusters @512 u")
    ax[1, 0].set_title("branches"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].set_xlabel("T"); ax[1, 1].set_ylabel("256 u position cells")
    ax[1, 1].set_title("coverage: distinct / NOVEL (count 0 in int_counts)")
    ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
def parse_temps(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--map", required=True, help="the MAIN checkout's .bsp")
    ap.add_argument("--route", required=True, help="route .npz next to it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64, help="rollouts per T")
    ap.add_argument("--temps", default="0,0.25,0.5,1,2,4")
    ap.add_argument("--knob", action="append", choices=KNOBS, default=None,
                    help="repeatable; default sigma and eps")
    ap.add_argument("--seed", type=int, default=0,
                    help="core.reset seed of the reference spawn (env 0 of "
                         "record_ckpt --seed N)")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="torch seed of the sampling noise; the SAME for "
                         "every T so the temperature is the only change")
    ap.add_argument("--ep-ticks", type=int, default=None,
                    help="episode cap (default: the checkpoint's)")
    ap.add_argument("--from-spine", default=None,
                    help="STATE_DTYPE .npy spine: start every rollout from "
                         "the state nearest --at-tick instead of the spawn")
    ap.add_argument("--at-tick", type=int, default=0)
    ap.add_argument("--greedy-only", action="store_true")
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--int-cell", type=float, default=256.0)
    args = ap.parse_args()
    knobs = args.knob or ["sigma", "eps"]
    temps = parse_temps(args.temps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    step = int(ck.get("global_step", 0))
    check_supported(cfg)
    ep_ticks = int(args.ep_ticks or cfg.get("ep_ticks", 12000))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_path = str(args.map)
    stem = Path(map_path).stem
    if cfg.get("map") and cfg["map"] != stem:
        print(f"WARNING: checkpoint map {cfg['map']!r} != --map {stem!r}")
    print(f"ckpt {Path(args.ckpt).name}  step {step:,}  map {stem}  "
          f"view {'abs ' + str(cfg.get('view_absolute')) if cfg.get('view_absolute') else ('delta' if cfg.get('view_continuous') else 'bins')}  "
          f"act_every {cfg.get('act_every', 1)}  ep_ticks {ep_ticks}  "
          f"device {device}")

    # -- the reference spawn: env 0 of record_ckpt's core at --seed ------
    ref = build_core(map_path, cfg, 1, ep_ticks)
    gf, zones, lcell = goal_field_and_box(ref, cfg)
    if args.from_spine:
        spine = np.load(args.from_spine)
        if spine.dtype != STATE_DTYPE:
            raise SystemExit(f"{args.from_spine} is not a STATE_DTYPE spine")
        j = int(np.argmin(np.abs(spine["tick"].astype(np.int64) - args.at_tick)))
        s0 = spine[j].copy()
        print(f"start: spine row {j} (tick {int(spine['tick'][j])} of the "
              f"recording, asked {args.at_tick}); origin "
              f"{np.round(s0['origin'], 1).tolist()}  |v| "
              f"{float(np.linalg.norm(s0['velocity'])):.0f} u/s")
    else:
        ref.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
        ref.set_spawn_pool(race_start_pool(ref, gf))
        ref.reset(int(args.seed))
        s0 = ref.get_states()[0].copy()
        print(f"start: the eval spawn at seed {args.seed}; origin "
              f"{np.round(s0['origin'], 1).tolist()}  yaw {float(s0['yaw']):.2f}"
              f"  pitch {float(s0['pitch']):.1f}")
    ref.close()

    # -- the bench core: N envs, NO yaw jitter, a one-entry pool ----------
    core = build_core(map_path, cfg, int(args.n), ep_ticks, yaw_jitter=0.0)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    core.set_spawn_pool(np.array([s0], dtype=STATE_DTYPE))
    core.reset(int(args.seed))
    sv = core.get_states()
    for f in ("origin", "velocity", "yaw", "pitch"):
        if not np.all(sv[f] == s0[f]):
            raise SystemExit(f"spawn is not fixed: {f} differs across envs")
    mins, dims = cell_layout(core, args.int_cell)
    counts = position_counts(ck, stem, dims, int(cfg.get("int_view") or 0),
                             int(cfg.get("int_speed") or 0))
    if counts is None:
        print("no int_counts in the checkpoint: novel cells not available")
    else:
        print(f"int_counts: {counts.size:,} position cells, "
              f"{int((counts > 0).sum()):,} visited, "
              f"{int(counts.sum()):,} visits")
    lidar = build_lidar(core, cfg, lcell, device)
    policy = build_policy(ck, core, lidar, device)
    packer = HeadPacker(device)
    act_every = int(cfg.get("act_every", 1))
    pts, spacing = load_route(args.route)
    print(f"route {len(pts)} pts x {spacing:g}u = {(len(pts) - 1) * spacing:,.0f}u; "
          f"setup {time.perf_counter() - t0:.0f}s")

    rows, curves = [], {}
    greedy = None

    def run_one(tag, knob, T, pol):
        t1 = time.perf_counter()
        pos, end_tick, fin, kind = roll(core, pol, args.seed, ep_ticks,
                                        args.sample_seed)
        row, curve, prog = score(tag, knob, T, pos, end_tick, fin, pts,
                                 spacing, mins, dims, args.int_cell, counts,
                                 greedy)
        np.savez_compressed(out / f"rollouts_{tag}.npz", pos=pos,
                            end_tick=end_tick, fin=fin, kind=kind, prog=prog)
        rows.append(row)
        curves[(knob, T)] = curve
        print(f"  {tag:<12} prog mean {row['prog_mean']:9,.0f}  max "
              f"{row['prog_max']:9,.0f}  wall {100 * row['past_wall']:3.0f}%  "
              f"fin {row['finishes']:2d}  len {row['len_med_s']:5.1f}s  "
              f"spread60 {row['spread_60s']:8,.0f}  branches "
              f"{row['branches_end']}/{row['branches_30s']}/{row['branches_60s']}"
              f"  cells {row['cells']:5d} novel {row['novel_cells']:4d}  "
              f"same-line {100 * row['same_line']:3.0f}%/"
              f"{100 * row['same_line_60s']:3.0f}%  "
              f"[{time.perf_counter() - t1:.0f}s]")
        return pos, end_tick

    if not args.no_greedy:
        pol = GreedyTorchPolicy(policy, packer, device, lidar, core,
                                act_every, 1)
        pos, end_tick = run_one("greedy", "greedy", 0.0, pol)
        # every rollout of a deterministic policy from one state is the
        # same rollout; if the core disagrees the benchmark is not what it
        # says it is
        L = (end_tick[0] + 1) if end_tick[0] >= 0 else pos.shape[0]
        same = np.all(end_tick == end_tick[0]) and np.allclose(
            np.nan_to_num(pos[:L]), np.nan_to_num(pos[:L, :1]), atol=1e-3)
        print(f"  greedy rollouts identical across the {args.n} envs: {same}")
        greedy = (pos[:, 0], int(end_tick[0]))
    if not args.greedy_only:
        for knob in knobs:
            for T in temps:
                pol = knob_policy(policy, packer, device, lidar, core,
                                  act_every, knob, T)
                run_one(f"{knob}_T{T:g}", knob, T, pol)

    with open(out / "bench.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})
    (out / "spread.json").write_text(json.dumps(
        {f"{k}|{T:g}": c for (k, T), c in curves.items()}, indent=0),
        encoding="utf-8")
    title = (f"{Path(args.ckpt).name} @ {step:,}  N={args.n}  spawn seed "
             f"{args.seed}" + (f"  spine tick {args.at_tick}" if args.from_spine else ""))
    md = (f"# diversity_bench: {title}\n\n"
          f"knob sigma = view sigma x (1+T), logits / (1+T); knob eps = each "
          f"head uniform with p = min(0.5, 0.05 T). Progress = order-only "
          f"corridor (window 16, corridor 1500) in route units; wall = "
          f"{WALL_U:,.0f} u; spread = RMS from the medoid of the alive "
          f"rollouts (u); branches = single-linkage clusters at 512 u; cells "
          f"= distinct 256 u position cells, novel = count 0 in the "
          f"checkpoint's int_counts, rare = under 100; same-line = within "
          f"100 u of the greedy trajectory for the whole life.\n\n"
          + md_table(rows) + "\n")
    (out / "bench.md").write_text(md, encoding="utf-8")
    try:
        plot(rows, curves, out / "bench.png", title)
    except Exception as exc:          # pragma: no cover - plotting is optional
        print(f"plot failed: {exc!r}")
    print("\n" + md_table(rows))
    print(f"\nwrote {out / 'bench.csv'}, bench.md, bench.png, spread.json  "
          f"[{time.perf_counter() - t0:.0f}s total]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
