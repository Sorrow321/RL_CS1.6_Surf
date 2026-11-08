#!/usr/bin/env python3
"""ge_beam.py - Go-Explore whose explore step is a POLICY-GUIDED BEAM (unr1b2, 2026-11-08).

The unified-recipe program's slot 2 (autoresearch/PROTOCOL.md "Your slot"): an archive of EXACT
simulator states (Go-Explore, Ecoffet et al. 2021), cells keyed by generic quantities (64 u
position x kinetic-energy height x velocity heading, tools/explore_phase1.CellHash), and an
explore step that restores M copies of the CURRENT policy at each selected cell and lets them
sample their own actions - a beam with no culling: EVERY new cell any copy reaches joins the
archive with the exact action rows that reached it (distance culling killed unitfarmer2's climb).

The knob between policy-guided and random is ONE number, ``--random-frac`` kappa: each run
(one copy from one restore, until it dies or the explore step ends) is, with probability kappa,
a Nature-paper burst - uniform random keys and a uniform view offset from the velocity heading
(the policy's own action frame, explore_phase1 --view-follow), the previous action repeated with
p = 0.95 per decision - and otherwise the policy sampling at --keys-temp / --view-temp with a
held yaw offset (--view-hold: coherent, not per-decision noise). kappa = 0 is the pure policy
beam, kappa = 1 the Nature explore step. Every cell records which kind of run found it, so the
status line says where on the knob the new cells come from.

Selection: W = 1 / sqrt(1 + C_seen) (Nature, Methods eq. 4; C_seen = runs that touched the
cell), plus a --front-frac share drawn uniformly from the --front-k deepest cells on the goal
field (return to the frontier; a generic progress quantity, no map constant). A cell keeps its
first arrival unless a later one reaches it --replace-secs sooner (Nature: the shorter
trajectory; a NEW node is made, so chains through the old one stay exact).

Restores go through the core's own reset (spawn pool of the selected cells, yaw jitter 0), so a
run from a cell is exactly what a 1-env replay of the same rows from the same cell produces: a
line is rebuilt link by link (tools/ge_replay's idea, stitched at each restore, as a backward
curriculum only RESETS into a line's states) and every link's end is checked against the
archived state. A line is exported when the deepest VIABLE cell (a run from it lived
>= --viable-secs, or it finishes) beats the last exported line by --gain u:
<out>/line_<k>.npy (STATE_DTYPE per tick, tick/stuck zeroed) and <out>/line.json, which
train_fast --ge-beam polls and turns into its backward curriculum.

No constant here comes from a map: cell sizes, seconds, a field-distance margin, budgets.
No demo, no record (CLAUDE.md 0, 0b): the explorer's proposals are the policy's own or uniform.

    python tools/ge_beam.py --ckpt runs/X/ge/policy.pt --map maps/M.bsp --out runs/X/ge \
        --serve --envs 512                       # beside the trainer (train_fast --ge-beam)
    python tools/ge_beam.py --ckpt ckpt.pt --map maps/M.bsp --out /tmp/geb --minutes 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE                                   # noqa: E402

G_ENERGY = 800.0


# ---------------------------------------------------------------------------------------------
# the archive
# ---------------------------------------------------------------------------------------------

class Archive:
    """Nodes: one exact state each, its parent node (the cell its run was restored from), the
    link (the run's engine rows from that restore: ``h0`` ticks of row 0 when the run began
    mid-hold, then K ticks per row, ``ticks`` ticks in all) and per-cell statistics. ``index``
    maps a cell key to its CURRENT node; a replaced node stays in the tree for its children."""

    COLS = (("key", np.int64, -1), ("parent", np.int64, -1), ("depth", np.int64, 0),
            ("d", np.float64, np.inf), ("ticks", np.int64, 0), ("h0", np.int64, 0),
            ("nrows", np.int64, 0), ("link", np.int64, -1), ("seen", np.int64, 0),
            ("chosen", np.int64, 0), ("surv", np.int64, 0), ("finish", np.bool_, False),
            ("mode", np.int8, -1), ("born", np.int64, 0), ("cur", np.bool_, False))

    def __init__(self, cap: int = 1 << 15):
        self.n = 0
        self.cap = 0
        self.state = np.zeros(0, STATE_DTYPE)
        for name, dt, _ in self.COLS:
            setattr(self, name, np.zeros(0, dt))
        self._grow(cap)
        self.index: dict = {}
        self.rows = np.zeros((1 << 16, 6), np.int8)
        self.views = np.zeros((1 << 16, 2), np.float32)
        self.nlink = 0

    def _grow(self, cap):
        if cap <= self.cap:
            return
        st = np.zeros(cap, STATE_DTYPE)
        st[:self.n] = self.state[:self.n]
        self.state = st
        for name, dt, fill in self.COLS:
            a = np.full(cap, fill, dt)
            a[:self.n] = getattr(self, name)[:self.n]
            setattr(self, name, a)
        self.cap = cap

    def add(self, key, state, parent, depth, d, ticks, h0, nrows, mode, born, finish=False):
        if self.n >= self.cap:
            self._grow(2 * self.cap)
        i = self.n
        self.n += 1
        self.key[i], self.state[i], self.parent[i], self.depth[i] = key, state, parent, depth
        self.d[i], self.ticks[i], self.h0[i], self.nrows[i] = d, ticks, h0, nrows
        self.mode[i], self.born[i], self.finish[i] = mode, born, finish
        if not finish:
            old = self.index.get(int(key))
            if old is not None:
                # the cell's statistics stay with the cell (W counts runs that touched it)
                self.seen[i], self.chosen[i] = self.seen[old], self.chosen[old]
                self.cur[old] = False
            self.index[int(key)] = i
            self.cur[i] = True
        return i

    @classmethod
    def load(cls, path):
        """An archive saved by :meth:`save` (a crash-resumed run continues its search)."""
        z = np.load(path, allow_pickle=False)
        n = int(len(z["state"]))
        A = cls(cap=max(1 << 15, n + 1))
        A.state[:n] = z["state"]
        for name, _, _ in cls.COLS:
            getattr(A, name)[:n] = z[name]
        A.n = n
        m = int(len(z["rows"]))
        cap = 1 << 16
        while cap < m + 1:
            cap *= 2
        A.rows = np.zeros((cap, 6), np.int8)
        A.views = np.zeros((cap, 2), np.float32)
        A.rows[:m], A.views[:m] = z["rows"], z["views"]
        A.nlink = m
        for i in np.flatnonzero(A.cur[:n]):
            A.index[int(A.key[i])] = int(i)
        return A

    def store_link(self, rows, views) -> int:
        m = len(rows)
        if self.nlink + m > len(self.rows):
            cap = len(self.rows)
            while cap < self.nlink + m:
                cap *= 2
            r = np.zeros((cap, 6), np.int8)
            v = np.zeros((cap, 2), np.float32)
            r[:self.nlink], v[:self.nlink] = self.rows[:self.nlink], self.views[:self.nlink]
            self.rows, self.views = r, v
        off = self.nlink
        self.rows[off:off + m] = rows
        self.views[off:off + m] = views
        self.nlink += m
        return off

    def chain(self, i) -> list:
        out = [int(i)]
        while self.parent[out[-1]] >= 0:
            out.append(int(self.parent[out[-1]]))
        return out[::-1]

    def current(self) -> np.ndarray:
        return np.flatnonzero(self.cur[:self.n])

    def select(self, rng, b: int, front_frac: float, front_k: int) -> np.ndarray:
        cur = self.current()
        nf = int(round(b * front_frac)) if len(cur) > 1 else 0
        w = 1.0 / np.sqrt(1.0 + self.seen[cur].astype(np.float64))
        out = [rng.choice(cur, b - nf, p=w / w.sum())]
        if nf:
            k = min(int(front_k), len(cur))
            deep = cur[np.argpartition(self.d[cur], k - 1)[:k]] if k < len(cur) else cur
            out.append(rng.choice(deep, nf))
        sel = np.concatenate(out).astype(np.int64)
        np.add.at(self.chosen, sel, 1)
        return sel

    def save(self, path):
        n, m = self.n, self.nlink
        tmp = str(path) + ".tmp.npz"
        np.savez(tmp, state=self.state[:n], rows=self.rows[:m], views=self.views[:m],
                 **{name: getattr(self, name)[:n] for name, _, _ in self.COLS})
        os.replace(tmp, path)


def link_row_index(t: np.ndarray | int, h0: int, k: int):
    """Row of a link applied at link tick t (0-based): row 0 for the first h0 ticks when the run
    began mid-hold (h0 > 0), then one row per K ticks."""
    t = np.asarray(t)
    if h0 > 0:
        return np.where(t < h0, 0, 1 + (t - h0) // k)
    return t // k


# ---------------------------------------------------------------------------------------------
# simulator + policy (checkpoint-faithful, tools/beam_tas.py's loading)
# ---------------------------------------------------------------------------------------------

def build_core(cfg, map_path, n, ep_cap, tick):
    """beam_tas.build_sim with the reset's yaw jitter OFF: a restore must be exact (a 1-env
    replay of a link resets into the same state as the explorer did)."""
    import beam_tas as bt
    from surfgym import default_config as _dc
    orig = bt.default_config

    def dc(**kw):
        kw["yaw_jitter_deg"] = 0.0
        return _dc(**kw)
    bt.default_config = dc
    try:
        return bt.build_sim(cfg, map_path, n, ep_cap, tick=tick)
    finally:
        bt.default_config = orig


def setup(ckpt_path, map_arg, n_envs, device, no_audit=False, with_policy=True):
    """-> a namespace with the cores (N envs and 1 env), the fields, the spawn pool, and (with
    ``with_policy``) the policy, its packer, lidar and the wrapper's feeds - mirrored exactly as
    beam_tas / record_ckpt build them."""
    import torch
    import beam_tas as bt
    import record_ckpt as _rc
    import train_fast as _tf
    from surfgym.bc import make_eval_feeds
    from surfgym.goalfield import build_goal_field
    from surfgym.mapfleet import map_tag
    from surfgym.rewards import map_spawn_pool
    from surfgym.tick import TickClock
    from surfgym.vision import GpuLidar, LidarPotential, pick_cell
    from surfgym.zones import load_zones

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck.get("config") or {})
    if cfg.get("reward") != "race":
        raise SystemExit("ge_beam needs a race checkpoint")
    bad = [k for k in bt.UNSUPPORTED if cfg.get(k)]
    if bad:
        raise SystemExit("ge_beam cannot carry the per-env inference state of: " + ", ".join(bad))
    if str(cfg.get("rnn") or "none") != "none" or cfg.get("obs_reward") \
            or cfg.get("curiosity_cond") or int(cfg.get("frame_stack") or 1) > 1:
        raise SystemExit("ge_beam does not mirror --rnn / --obs-reward / --curiosity-cond / "
                         "--frame-stack yet")
    _rc.audit_cfg(cfg, strict=not no_audit)
    keys = bool(cfg.get("keys_hold"))
    _tf.set_keys_hold(keys)
    tick = TickClock(float(cfg.get("tick_ms") or 10.0))
    if not tick.is_reference:
        raise SystemExit("ge_beam runs the 10 ms reference tick only (the program's physics)")
    K = int(cfg.get("act_every", 1))
    ep_cap = max(int(cfg.get("ep_ticks", 700)), 30000)
    map_path = str(map_arg) if str(map_arg).lower().endswith(".bsp") else bt.resolve_map(
        map_arg, cfg.get("map"))
    core1 = build_core(cfg, map_path, 1, ep_cap, tick)
    cell = float((cfg.get("map_cells") or {}).get(map_tag(Path(map_path).stem),
                                                  cfg.get("lidar_cell") or pick_cell(core1)))
    gcell = float(cfg.get("goal_cell") or cell) if not isinstance(cfg.get("goal_cell"), str) \
        else cell
    zones = load_zones(core1.bsp_path)
    t0 = time.time()
    gf = build_goal_field(core1, zones["end"], cell=gcell)
    ogf = gf
    if int(cfg.get("race_no_corner_cut") or 0):
        ogf = build_goal_field(core1, zones["end"], cell=gcell, no_corner_cut=True)
    dt = time.time() - t0
    print(f"goal fields @ cell {gcell:g} in {dt:.1f}s"
          + ("  ** WARNING: a RE-BAKE? wrong map path/mtime **" if dt > 30 else ""), flush=True)
    raw = map_spawn_pool(core1)
    pool = map_spawn_pool(core1, yaw=gf.descent_yaw(raw["origin"]))
    pool["pitch"] = -10.0
    if cfg.get("fix_pitch") is not None:
        pool["pitch"] = float(cfg["fix_pitch"])
    d0 = float(np.mean(gf.sample(raw["origin"])))

    def arm(core):
        core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
        if cfg.get("goal_hull"):
            from surfgym.zones import goal_trigger_model
            _gm = goal_trigger_model(str(core.bsp_path), zones["end"])
            if _gm >= 0:
                core.set_goal_model(_gm)
        core.set_teleport_fail(True)
        core.set_spawn_pool(pool)

    arm(core1)
    coreN = build_core(cfg, map_path, int(n_envs), ep_cap, tick)
    arm(coreN)
    ns = SimpleNamespace(cfg=cfg, K=K, tick=tick, map_path=map_path, core1=core1, coreN=coreN,
                         N=int(n_envs), gf=gf, ogf=ogf, pool=pool, d0=d0, zones=zones,
                         arm=arm, keys=keys, policy=None, view_abs=cfg.get("view_absolute"),
                         viewc=bool(cfg.get("view_continuous")), step=int(ck.get("global_step", 0)))
    if not with_policy:
        return ns
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    lidar = GpuLidar(core1, lw, lh,
                     hfov_deg=float(cfg.get("lidar_hfov") or 120.0),
                     vfov_deg=float(cfg.get("lidar_vfov") or 90.0),
                     range_units=float(cfg.get("lidar_range", 2000.0)),
                     near_range=cfg.get("lidar_near"), cell=cell, device=device,
                     surf_mask=bool(cfg.get("surf_mask", 0)), pinhole=bool(cfg.get("pinhole", 0)),
                     normals=bool(cfg.get("normals", 0)),
                     potential=LidarPotential.from_cfg(cfg, ogf, core1, device,
                                                       Path(map_path).stem,
                                                       d0=(d0 if ogf is gf else None)),
                     vision_clip=bool(int(cfg.get("vision_clip") or 0)),
                     depth_enc=str(cfg.get("depth_enc") or "legacy"),
                     ttc=bool(cfg.get("obs_ttc") or 0), edges=bool(cfg.get("obs_edges") or 0),
                     views=int(cfg.get("obs_views") or 1),
                     views_scale=int(cfg.get("obs_views_scale") or 1),
                     reach=bool(cfg.get("obs_reach") or 0),
                     texture=bool(cfg.get("obs_texture") or cfg.get("obs_normal")
                                  or cfg.get("obs_slope") or 0),
                     texture_mode=("normal" if cfg.get("obs_normal") else "slope"
                                   if cfg.get("obs_slope") else "rgb"),
                     depth_log_d0=float(cfg.get("depth_log_d0") or 200.0))
    n_latch = 1 if (float(cfg.get("race_latch") or 0.0) > 0.0
                    or float(cfg.get("race_latch_frac") or 0.0) > 0.0) else 0
    n_keys = _tf.keyshold.N_FEATURES if keys else 0
    XF = bt.XFeeds(cfg, ogf, tick.ms, K)
    if not ns.viewc:
        raise SystemExit("ge_beam drives a --view-continuous policy (the program's recipe)")
    policy = _tf.Policy(core1.obs_dim + n_latch + n_keys + XF.n_columns
                        + int(getattr(lidar, "frame_size", lw * lh * lidar.channels)),
                        lw, lh, emb=int(cfg.get("emb", 256)), hidden=int(cfg.get("hidden", 256)),
                        gps=bool(cfg.get("gps", True)), trunk=str(cfg.get("trunk") or "plain"),
                        tower_depth=int(cfg.get("tower_depth") or 2),
                        conv_mult=int(cfg.get("conv_mult") or 1), extra_feat=(),
                        obs_fourier=int(cfg.get("obs_fourier") or 0),
                        in_ch=int(getattr(lidar, "conv_channels", lidar.channels)),
                        views=int(cfg.get("obs_views") or 1),
                        views_scale=int(cfg.get("obs_views_scale") or 1),
                        drop_depth=bool(cfg.get("obs_no_depth") or 0),
                        simba=bool(cfg.get("simba") or 0),
                        split_trunk=bool(cfg.get("split_trunk") or 0),
                        n_codes=0, chunk=0, route_dim=n_latch + n_keys + XF.n_columns,
                        route_critic_only=bool(cfg.get("route_critic_only")),
                        view_continuous=True, view_absolute=ns.view_abs,
                        pitch_sigma_min=float(cfg.get("pitch_sigma_min") or 0.0)).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    slot, rf, lf = make_eval_feeds(cfg, gf, d0, K, tick_ms=tick.requested_ms)
    xkw, xb = XF.make(coreN)
    pol = make_knob_policy(policy, _tf.HeadPacker(device), device, lidar, coreN, K,
                           extra_slot=slot, extra_fn=rf, latch_fn=lf, keys_hold=keys, **xkw)
    pol.spawn_duck = bool(keys and cfg.get("keys_hold_spawn_duck"))
    ns.policy, ns.pol, ns.lidar, ns.device = policy, pol, lidar, device
    return ns


def make_knob_policy(*a, **kw):
    """The policy wrapper of the beam: per-env keys temperature, yaw-sigma multiplier and a held
    yaw-z offset (all None = SampledTorchPolicy's own draw)."""
    import torch
    import train_fast as _tf

    class KnobPolicy(_tf.SampledTorchPolicy):
        kt = None              # (N,) torch: logits / kt on the four key heads
        vt = None              # (N,) torch: sigma x vt on the yaw head
        voff = None            # (N,) torch: + voff on the yaw z (a held manoeuvre)

        @torch.inference_mode()
        def _decide(self, obs):
            logits, _ = self.policy(self._obs(obs))
            cat, mu = _tf.split_view(logits.float())
            padded = self._mask_padded(self.packer.pad(cat), obs)[:, _tf.N_VIEW:]
            if self.kt is not None:
                padded = padded / self.kt[:, None, None]
            u = torch.rand_like(padded).clamp_min_(1e-20)
            act = (padded - torch.log(-torch.log(u))).argmax(-1)
            sig = self.policy.log_std().exp().unsqueeze(0).expand_as(mu).clone()
            if self.vt is not None:
                sig[:, 0] = sig[:, 0] * self.vt
            z = mu + sig * torch.randn_like(mu)
            if self.voff is not None:
                z[:, 0] = z[:, 0] + self.voff
            return self._finish_view(act, z)

    return KnobPolicy(*a, **kw)


# ---------------------------------------------------------------------------------------------
# the explorer
# ---------------------------------------------------------------------------------------------

VIEW_FOLLOW = np.array([-90, -60, -30, -15, 0, 15, 30, 60, 90], np.float32)
MODE_POLICY, MODE_RANDOM, MODE_ROOT, MODE_BURST = 0, 1, 2, 3


class Explorer:
    def __init__(self, ctx, P, rng):
        from explore_phase1 import CellHash
        self.C, self.P, self.rng = ctx, P, rng
        mins, maxs = ctx.core1.map_bounds()
        self.hash = CellHash(mins, maxs, cell=P.cell, energy=P.cell_energy,
                             heading=P.cell_heading)
        self.A = Archive()
        self.K = ctx.K
        self.steps = 0
        self.runs = 0
        self.full = False
        self.new_by_mode = np.zeros(4, np.int64)
        self.runs_by_mode = np.zeros(4, np.int64)
        self.exported_d = np.inf
        self.exported_finish = False
        self.k_line = 0
        self.dfield = ctx.ogf
        rows = np.array(ctx.pool, copy=True)
        rows["tick"] = 0
        rows["stuck_ticks"] = 0
        self.d_root = float(np.mean(self.field_d(rows)))
        arc = Path(P.out) / "archive.npz"
        if getattr(P, "resume", False) and arc.exists():
            self.A = Archive.load(arc)
            lj = Path(P.out) / "line.json"
            if lj.exists():
                info = json.loads(lj.read_text(encoding="utf-8"))
                self.k_line = int(info.get("k", 0))
                self.exported_d = float(info.get("end_d", np.inf))
                self.exported_finish = bool(info.get("finish", False))
            print(f"--resume: {self.A.n:,} nodes / {len(self.A.index):,} cells from {arc}; "
                  f"line {self.k_line}", flush=True)
            rows = rows[:0]
        for q in range(len(rows)):
            self.A.add(int(self.hash.keys(rows[q:q + 1])[0]), rows[q], -1, 0,
                       float(self.field_d(rows[q:q + 1])[0]), 0, 0, 0, MODE_ROOT, 0)
            self.A.surv[q] = 1 << 40          # the true start is viable by definition
        neutral = np.asarray(__import__("train_fast").NEUTRAL_ACT, np.int32)
        self.neutral = neutral
        if P.random_frac > 0.0 and ctx.viewc and ctx.view_abs != "velocity":
            raise SystemExit("--random-frac acts in the velocity frame: a --view-absolute "
                             "velocity checkpoint")

    def field_d(self, states) -> np.ndarray:
        o = np.asarray(states["origin"], np.float64).reshape(-1, 3)
        d = np.asarray(self.dfield.sample(o), np.float64)
        return np.where(d < self.dfield.reach_max - 0.5 * self.dfield.cell, d, np.inf)

    def prog(self, d) -> float:
        return 100.0 * (1.0 - float(d) / self.d_root) if np.isfinite(d) else float("nan")

    # -- one explore step -------------------------------------------------------------------
    def _rand_rows(self, n):
        a = np.empty((n, 6), np.int32)
        a[:, 0], a[:, 1] = self.neutral[0], self.neutral[1]
        a[:, 2] = self.rng.integers(0, 3, n)
        a[:, 3] = self.rng.integers(0, 3, n)
        a[:, 4] = self.rng.integers(0, 2, n)
        a[:, 5] = self.rng.integers(0, 2, n)
        v = np.empty((n, 2), np.float32)
        v[:, 0] = self.rng.choice(VIEW_FOLLOW, n)
        from surfgym.view import PITCH_ABS_MID
        v[:, 1] = PITCH_ABS_MID
        return a, v

    def _run_modes(self, n):
        """Each new run's kind: a Nature burst run with probability --random-frac, else a
        burst-then-policy run when --burst > 0 (--burst decisions of held random actions from
        the restored cell, then the policy), else a policy run."""
        P = self.P
        m = np.where(self.rng.random(n) < float(P.random_frac), MODE_RANDOM,
                     MODE_BURST if int(P.burst) > 0 else MODE_POLICY).astype(np.int8)
        return m

    def step(self):
        import torch
        C, P, A, K = self.C, self.P, self.A, self.K
        N, core = C.N, C.coreN
        D = int(P.decisions)
        sel = A.select(self.rng, int(P.cells), float(P.front_frac), int(P.front_k))
        pool = np.array(A.state[sel], copy=True)
        pool["tick"] = 0
        pool["stuck_ticks"] = 0
        core.set_spawn_pool(pool)
        obs = np.array(core.reset(int(self.rng.integers(1 << 30))))
        lut = {}
        for j, r in zip(sel, pool):
            lut[np.concatenate([r["origin"], r["velocity"]]).astype(np.float32).tobytes()] = int(j)

        def match(st_rows):
            out = np.empty(len(st_rows), np.int64)
            for q, r in enumerate(st_rows):
                out[q] = lut.get(np.concatenate([r["origin"], r["velocity"]]).astype(
                    np.float32).tobytes(), -1)
            return out

        root = match(core.get_states())
        if (root < 0).any():
            raise RuntimeError(f"{int((root < 0).sum())} envs reset to a state that is not a "
                               "selected cell (yaw jitter on? pool changed?)")
        mode = self._run_modes(N)
        burst_left = np.where(mode == MODE_BURST, int(P.burst), 0).astype(np.int64)
        run_id = np.arange(N, dtype=np.int64) + self.runs
        self.runs += N
        np.add.at(self.runs_by_mode, mode, 1)
        run_ticks = np.zeros(N, np.int64)
        h0 = np.zeros(N, np.int64)
        nrows = np.zeros(N, np.int64)
        cap_rows = D + 2
        rows_buf = np.zeros((cap_rows, N, 6), np.int8)
        views_buf = np.zeros((cap_rows, N, 2), np.float32)
        pending = [[] for _ in range(N)]
        rand_a, rand_v = self._rand_rows(N)
        voff = (self.rng.normal(0.0, float(P.view_hold[0]), N).astype(np.float32)
                if P.view_hold[0] > 0 else np.zeros(N, np.float32))
        pol = C.pol if C.policy is not None else None
        if pol is not None:
            dev = C.device
            pol._tick = 0
            pol._held = None
            pol.kt = torch.full((N,), float(P.keys_temp), device=dev)
            pol.vt = torch.full((N,), float(P.view_temp), device=dev)
            pol.voff = torch.as_tensor(voff, device=dev)
        seen_r, seen_k = [], []
        cur_a = np.zeros((N, 6), np.int32)
        cur_v = np.zeros((N, 2), np.float32)
        n_new = 0
        n_fin = 0

        def end_run(i, survived):
            r = int(root[i])
            if survived > A.surv[r]:
                A.surv[r] = survived
            if pending[i]:
                off = A.store_link(rows_buf[:nrows[i], i], views_buf[:nrows[i], i])
                for nid in pending[i]:
                    A.link[nid] = off
                pending[i] = []

        for g in range(D * K):
            dec = g % K == 0
            if dec:
                kdec = g // K
                if P.view_hold[0] > 0 and kdec % max(1, int(P.view_hold[1])) == 0 and kdec:
                    voff = self.rng.normal(0.0, float(P.view_hold[0]), N).astype(np.float32)
                    if pol is not None:
                        pol.voff = torch.as_tensor(voff, device=C.device)
                rm = (mode == MODE_RANDOM) | (burst_left > 0)
                if rm.any():
                    redraw = rm & (self.rng.random(N) >= float(P.repeat))
                    if redraw.any():
                        na, nv = self._rand_rows(int(redraw.sum()))
                        rand_a[redraw], rand_v[redraw] = na, nv
            if pol is not None:
                a = np.array(pol.act(obs), np.int32, copy=True)
                v = np.array(pol.view, np.float32, copy=True)
            else:
                a = np.repeat(self.neutral[None], N, axis=0)
                v = np.zeros((N, 2), np.float32)
            if dec:
                rm = (mode == MODE_RANDOM) | (burst_left > 0)
                a[rm], v[rm] = rand_a[rm], rand_v[rm]
                burst_left[burst_left > 0] -= 1
                cur_a[:], cur_v[:] = a, v
                ok = nrows < cap_rows
                idx = np.flatnonzero(ok)
                rows_buf[nrows[idx], idx] = a[idx]
                views_buf[nrows[idx], idx] = v[idx]
                nrows[idx] += 1
            else:
                a, v = cur_a, cur_v
            pre = core.get_states()
            obs_, _r, done, trunc, _t = core.step(np.ascontiguousarray(a, np.int32),
                                                  view=np.ascontiguousarray(v, np.float32))
            obs = np.array(obs_)
            gh = np.asarray(core.goal_hits, bool)
            dead = np.asarray(done, bool) | np.asarray(trunc, bool)
            if gh.any():
                for i in np.flatnonzero(gh):
                    # the state just before the finishing tick, its link to there, flagged
                    nid = A.add(-1, pre[i], int(root[i]), int(A.depth[root[i]] + run_ticks[i]),
                                float(self.field_d(pre[i:i + 1])[0]), int(run_ticks[i]),
                                int(h0[i]), int(nrows[i]), int(mode[i]), self.steps, finish=True)
                    pending[i].append(nid)
                    n_fin += 1
            run_ticks += 1
            if dead.any():
                st = core.get_states()
                di = np.flatnonzero(dead)
                for i in di:
                    end_run(i, int(run_ticks[i]) - 1)
                nr = match(st[di])
                if (nr < 0).any():
                    raise RuntimeError("an auto-reset left the selected cells")
                root[di] = nr
                run_id[di] = np.arange(len(di)) + self.runs
                self.runs += len(di)
                mode[di] = self._run_modes(len(di))
                burst_left[di] = np.where(mode[di] == MODE_BURST, int(P.burst), 0)
                np.add.at(self.runs_by_mode, mode[di], 1)
                run_ticks[di] = 0
                rem = (K - ((g + 1) % K)) % K
                h0[di] = rem
                nrows[di] = 0
                na, nv = self._rand_rows(len(di))           # a new run's own first draw
                rand_a[di], rand_v[di] = na, nv
                if rem > 0:
                    rnd = di[(mode[di] == MODE_RANDOM) | (burst_left[di] > 0)]
                    if len(rnd):
                        cur_a[rnd], cur_v[rnd] = rand_a[rnd], rand_v[rnd]
                    if pol is not None:
                        pp = di[(mode[di] != MODE_RANDOM) & (burst_left[di] == 0)]
                        if len(pp):
                            cur_a[pp] = np.asarray(pol._held, np.int32)[pp]
                            cur_v[pp] = np.asarray(pol.view, np.float32)[pp]
                    rows_buf[0, di] = cur_a[di]
                    views_buf[0, di] = cur_v[di]
                    nrows[di] = 1
            if (g + 1) % K == 0:
                st = core.get_states()
                live = run_ticks > 0
                if live.any():
                    li = np.flatnonzero(live)
                    keys = self.hash.keys(st[li])
                    seen_r.append(run_id[li].copy())
                    seen_k.append(keys)
                    dd = None
                    for q, i in enumerate(li):
                        if A.n >= int(P.max_nodes):
                            self.full = True
                            break
                        kk = int(keys[q])
                        j = A.index.get(kk)
                        dep = int(A.depth[root[i]] + run_ticks[i])
                        if j is not None and dep >= A.depth[j] - int(P.replace_ticks):
                            continue
                        if dd is None:
                            dd = self.field_d(st[li])
                        nid = A.add(kk, st[i], int(root[i]), dep, float(dd[q]),
                                    int(run_ticks[i]), int(h0[i]), int(nrows[i]), int(mode[i]),
                                    self.steps)
                        pending[i].append(nid)
                        n_new += 1
                        self.new_by_mode[int(mode[i])] += 1
        for i in range(N):
            end_run(i, int(run_ticks[i]))
        if seen_r:
            r = np.concatenate(seen_r)
            k = np.concatenate(seen_k)
            pairs = np.unique(np.stack([r, k], axis=1), axis=0)
            uk, cnt = np.unique(pairs[:, 1], return_counts=True)
            for kk, c in zip(uk, cnt):
                j = A.index.get(int(kk))
                if j is not None:
                    A.seen[j] += int(c)
        self.steps += 1
        return n_new, n_fin

    # -- lines --------------------------------------------------------------------------------
    def best_viable(self):
        """The deepest cell a run from which lived >= --viable-secs (or a finish) -> node or -1."""
        A = self.A
        n = A.n
        fin = np.flatnonzero(A.finish[:n] & (A.link[:n] >= 0))
        if len(fin):
            return int(fin[np.argmin(A.depth[fin])])
        ok = np.flatnonzero(A.cur[:n] & (A.surv[:n] >= int(self.P.viable_ticks))
                            & np.isfinite(A.d[:n]) & (A.mode[:n] != MODE_ROOT))
        if not len(ok):
            return -1
        return int(ok[np.argmin(A.d[ok])])

    def replay_line(self, node) -> np.ndarray:
        """The node's chain replayed link by link on the 1-env core: per-tick states from the
        root's state to the node's, every link's end checked against the archived state."""
        C, A, K = self.C, self.A, self.K
        core = C.core1
        ch = A.chain(node)
        r0 = np.array(A.state[ch[0]], copy=True)
        out = [r0]
        for a, b in zip(ch[:-1], ch[1:]):
            ps = np.array(A.state[a:a + 1], copy=True)
            ps["tick"] = 0
            ps["stuck_ticks"] = 0
            core.set_spawn_pool(ps)
            core.reset(0)
            off, nr, h0, T = int(A.link[b]), int(A.nrows[b]), int(A.h0[b]), int(A.ticks[b])
            if off < 0:
                raise RuntimeError(f"node {b}: link not stored")
            rows = A.rows[off:off + nr].astype(np.int32)
            views = A.views[off:off + nr]
            ri = link_row_index(np.arange(T), h0, K)
            if T and int(ri[-1]) >= nr:
                raise RuntimeError(f"link {a}->{b}: {T} ticks need row {int(ri[-1])} of {nr}")
            for t in range(T):
                q = int(ri[t])
                o, _r, dn, tr, _tm = core.step(np.ascontiguousarray(rows[q:q + 1]),
                                               view=np.ascontiguousarray(views[q:q + 1]))
                if dn[0] or tr[0]:
                    raise RuntimeError(f"link {a}->{b} died at tick {t} of {T} on replay")
                out.append(core.get_states()[0].copy())
            # (a finish node's state is the one BEFORE its finishing tick: the same check)
            e = out[-1]
            err = float(np.abs(np.asarray(e["origin"], np.float64)
                               - np.asarray(A.state[b]["origin"], np.float64)).max())
            if err > 1e-3:
                raise RuntimeError(f"link {a}->{b}: replay ends {err:.4f} u off the archived "
                                   "state")
        line = np.array(out, dtype=STATE_DTYPE)
        line["tick"] = 0
        line["stuck_ticks"] = 0
        return line

    def maybe_export(self, out: Path):
        P, A = self.P, self.A
        b = self.best_viable()
        if b < 0:
            return None
        fin = bool(A.finish[b])
        d = float(A.d[b])
        if not ((fin and not self.exported_finish) or d < self.exported_d - float(P.gain)):
            return None
        t0 = time.time()
        try:
            line = self.replay_line(b)
        except RuntimeError as ex:
            print(f"!! line to node {b} does not replay: {ex}", flush=True)
            A.surv[b] = -1           # never the candidate again
            return None
        self.k_line += 1
        f = out / f"line_{self.k_line:03d}.npy"
        np.save(f, line)
        info = {"k": self.k_line, "line": str(f), "node": b, "end_d": d,
                "end_progress": round(self.prog(d), 3), "finish": fin, "n": int(len(line)),
                "secs": round(len(line) * self.C.tick.ms / 1000.0, 2),
                "links": len(A.chain(b)) - 1, "mode": int(A.mode[b]), "step": self.steps,
                "cells": len(A.index), "t": time.time(), "replay_s": round(time.time() - t0, 2),
                "policy_step": int(getattr(self, "policy_step", 0))}
        tmp = out / "line.json.tmp"
        tmp.write_text(json.dumps(info), encoding="utf-8")
        os.replace(tmp, out / "line.json")
        self.exported_d = min(self.exported_d, d)
        self.exported_finish = self.exported_finish or fin
        return info

    def status(self) -> dict:
        A = self.A
        n = A.n
        cur = A.current()
        dmin = float(A.d[cur].min()) if len(cur) else np.inf
        b = self.best_viable()
        m = A.mode[:n]
        return {"step": self.steps, "nodes": int(n), "cells": len(A.index), "runs": int(self.runs),
                "new_policy": int(self.new_by_mode[0]), "new_random": int(self.new_by_mode[1]),
                "new_burst": int(self.new_by_mode[3]),
                "runs_policy": int(self.runs_by_mode[0]),
                "runs_random": int(self.runs_by_mode[1]),
                "runs_burst": int(self.runs_by_mode[3]),
                "deepest_progress": round(self.prog(dmin), 3),
                "viable_progress": round(self.prog(A.d[b]), 3) if b >= 0 else None,
                "viable_mode": (int(A.mode[b]) if b >= 0 else None),
                "deep100_policy_share": (round(float(np.mean(
                    m[cur[np.argsort(A.d[cur])[:100]]] == MODE_POLICY)), 3) if len(cur) else None),
                "deep100_burst_share": (round(float(np.mean(
                    m[cur[np.argsort(A.d[cur])[:100]]] == MODE_BURST)), 3) if len(cur) else None),
                "new_by_mode": [int(x) for x in self.new_by_mode], "full": bool(self.full),
                "finishes": int(A.finish[:n].sum()), "exported_k": self.k_line,
                "exported_progress": round(self.prog(self.exported_d), 3)
                if np.isfinite(self.exported_d) else None}


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="checkpoint (or the trainer's ge/policy.pt)")
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--cells", type=int, default=64, help="cells restored per explore step")
    ap.add_argument("--decisions", type=int, default=150, help="decisions per explore step")
    ap.add_argument("--random-frac", type=float, default=0.0,
                    help="THE KNOB: share of runs that are Nature bursts (0 = policy beam)")
    ap.add_argument("--burst", type=int, default=0,
                    help="decisions of held random actions (--repeat) at the start of every "
                         "non-random run before the policy takes over (0 = pure policy runs): "
                         "the coherent random/policy mix in TIME")
    ap.add_argument("--repeat", type=float, default=0.95, help="random runs: action repeat p")
    ap.add_argument("--keys-temp", type=float, default=1.5,
                    help="policy runs: key heads at logits / T")
    ap.add_argument("--view-temp", type=float, default=1.0,
                    help="policy runs: yaw sigma x T")
    ap.add_argument("--view-hold", type=float, nargs=2, default=[0.3, 25.0],
                    metavar=("SIGMA", "DECISIONS"),
                    help="policy runs: a held yaw-z offset N(0, SIGMA) redrawn every DECISIONS")
    ap.add_argument("--front-frac", type=float, default=0.25,
                    help="share of the selected cells drawn from the --front-k deepest")
    ap.add_argument("--front-k", type=int, default=64)
    ap.add_argument("--cell", type=float, default=64.0)
    ap.add_argument("--cell-energy", type=float, default=128.0,
                    help="kinetic-energy-height bucket (u) of the cell key")
    ap.add_argument("--cell-heading", type=int, default=4,
                    help="velocity heading sectors (x rising / not) of the cell key")
    ap.add_argument("--max-nodes", type=int, default=12_000_000,
                    help="the archive stops growing at this many nodes (memory guard)")
    ap.add_argument("--replace-secs", type=float, default=0.25,
                    help="a cell's state is replaced by an arrival this many seconds sooner")
    ap.add_argument("--viable-secs", type=float, default=1.0,
                    help="a line may end only at a cell a run from which lived this long")
    ap.add_argument("--gain", type=float, default=100.0,
                    help="u a new line's end must beat the last exported one by")
    ap.add_argument("--resume", action="store_true",
                    help="continue <out>/archive.npz and <out>/line.json when they exist")
    ap.add_argument("--serve", action="store_true",
                    help="run until killed, reloading --ckpt whenever it changes")
    ap.add_argument("--minutes", type=float, default=0.0, help="stop after this long (0 = never)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--duty", type=float, default=1.0,
                    help="fraction of wall time spent exploring (sleeps the rest)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-policy", action="store_true",
                    help="random runs only, no network (tests): --random-frac is forced to 1")
    ap.add_argument("--save-every", type=float, default=600.0,
                    help="seconds between archive snapshots (archive.npz)")
    ap.add_argument("--no-config-audit", action="store_true")
    a = ap.parse_args(argv)
    a.view_hold = [float(a.view_hold[0]), float(a.view_hold[1])]
    return a


def main(argv=None) -> int:
    import torch
    a = parse(argv)
    if a.threads > 0:
        torch.set_num_threads(int(a.threads))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.no_policy:
        a.random_frac = 1.0
    ctx = setup(a.ckpt, a.map, a.envs, device, no_audit=a.no_config_audit,
                with_policy=not a.no_policy)
    tps = 1000.0 / ctx.tick.ms
    a.replace_ticks = int(round(a.replace_secs * tps))
    a.viable_ticks = int(round(a.viable_secs * tps))
    rng = np.random.default_rng(a.seed)
    ex = Explorer(ctx, a, rng)
    ex.policy_step = ctx.step
    print(f"ge_beam: {a.envs} envs, {a.cells} cells x {a.decisions} decisions per step (K "
          f"{ctx.K}), knob random-frac {a.random_frac:g}, policy keys T {a.keys_temp:g} / yaw "
          f"sigma x{a.view_temp:g} / held yaw z N(0,{a.view_hold[0]:g}) per "
          f"{a.view_hold[1]:g} decisions; cells {a.cell:g} u x energy {a.cell_energy:g} u x "
          f"heading {a.cell_heading}; burst {a.burst} decisions; max nodes {a.max_nodes:,}; "
          f"root d {ex.d_root:,.0f} u; ckpt step {ctx.step:,}",
          flush=True)
    t_start = time.time()
    mt = os.path.getmtime(a.ckpt)
    last_save = time.time()
    last_log = 0.0
    while True:
        t0 = time.time()
        n_new, n_fin = ex.step()
        info = ex.maybe_export(out)
        dt = time.time() - t0
        if info is not None:
            print(f"LINE {json.dumps(info)}", flush=True)
        if time.time() - last_log >= 30.0 or info is not None:
            last_log = time.time()
            s = ex.status()
            s.update(step_s=round(dt, 2), new=n_new, fin=n_fin,
                     elapsed_min=round((time.time() - t_start) / 60.0, 2))
            print("STATUS " + json.dumps(s), flush=True)
            tmp = out / "status.json.tmp"
            tmp.write_text(json.dumps(s), encoding="utf-8")
            os.replace(tmp, out / "status.json")
        if time.time() - last_save >= float(a.save_every):
            last_save = time.time()
            ex.A.save(out / "archive.npz")
        if a.max_steps and ex.steps >= a.max_steps:
            break
        if a.minutes and time.time() - t_start >= 60.0 * a.minutes:
            break
        if a.serve and ctx.policy is not None:
            try:
                m2 = os.path.getmtime(a.ckpt)
                if m2 != mt:
                    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
                    ctx.policy.load_state_dict(ck["policy"])
                    ex.policy_step = int(ck.get("global_step", 0))
                    mt = m2
                    print(f"weights reloaded: step {ex.policy_step:,}", flush=True)
            except Exception as e:                      # noqa: BLE001 - a half-written file
                print(f"weights reload deferred: {type(e).__name__}: {e}", flush=True)
        if a.duty < 1.0:
            time.sleep(dt * (1.0 / max(1e-3, a.duty) - 1.0))
    ex.A.save(out / "archive.npz")
    s = ex.status()
    print("FINAL " + json.dumps(s), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
