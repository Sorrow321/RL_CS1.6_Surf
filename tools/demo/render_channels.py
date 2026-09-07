#!/usr/bin/env python3
"""render_channels.py - a video of what an --obs-potential policy saw.

One recorded episode of a traj_*.jsonl, one frame per k-th physics tick: the
64x32 depth image the policy sees on top, the 64x32 potential channel
(``--obs-potential abs|rel|norm|logabs`` and ``--obs-potential-curtain``,
the run's own mode, scale and finish box) underneath,
both upscaled x10 nearest-neighbour, stacked with a thin separator and a
text strip (run, mode, spawn-clock time, tick, order-only progress, the
frame's channel range) plus a legend, encoded by ffmpeg (libx264, yuv420p)
at the recording's real-time rate (1000 / (tick_ms * every) fps).

The lidar is built exactly as tools/record_ckpt.py builds it for the
checkpoint - size, range, near range, cell, fov, the vision flags and the
potential channel through ``LidarPotential.from_cfg`` on the run's config
(run.json next to the trajectory, the trainer's own record of the config it
evaluated under; ``--ckpt`` cross-checks the checkpoint's copy) - and one
``GpuLidar.render`` per batch of recorded poses produces both channels, so
the frames ARE the observation, pixel for pixel. The trainer renders every
act_every-th tick; the ticks between are what the same renderer shows at
the recorded pose. The eye height uses the row's IN_DUCK button (the core's
own duck state is not in the recording; the difference is 5 u).

    python tools/demo/render_channels.py \\
        --traj C:/RL_Surf_cya/runs/cyPOTR/traj_1254096896.jsonl \\
        --episode best --every 2 \\
        --out C:/RL_Surf_base/runs/research/videos/potential_cyPOTR_rel.mp4

``--episode best`` picks the episode with the largest order-only corridor
progress (tools/eval_honesty.py --order-only 16: ``surfgym.route.ArcProgress``
at corridor 1,500 u, window 16, running max), and the same rule supplies the
running progress in the text strip.

Rollouts whose OWN checkpoint had no potential channel (the searched TAS
line, the human record) render under a REFERENCE config: ``--config-from
RUN_JSON`` (alias ``--run-json``) builds the lidar and the channel from that
run's config instead of a run.json next to the trajectory (cyPOTR's for rel,
cyPOTN's for norm), so the frames are what THAT policy would have seen at
these poses. ``--traj`` also accepts the HL demo frames ``.npz`` of
tools/demo/wr_scan.py (``simorg``, ``simvel``, ``uc_viewangles`` with the HL
pitch sign flipped to the simulator's positive = up, ``uc_buttons`` decoded
as wr_scan.decode_buttons, ``t`` the playback clock), converted in memory to
the same 15-column rows with the demo's own per-frame dt (7 / 8 ms on the
cannonball record). ``--t0 / --t1`` cut a window in seconds of the
recording's own clock (a traj row's spawn clock = the sum of the ticks
before it; the npz's ``t``); a negative value counts back from the
recording's END, so ``--t0 -12`` is its last 12 seconds. ``--clock-offset
S`` labels the on-screen time ``rec`` = clock - S (a record's timer: 1.81 s
behind the cannonball demo's playback clock).

    python tools/demo/render_channels.py \\
        --traj C:/RL_Surf_base/runs/research/tas_68.54/beam_best.jsonl \\
        --episode 0 --config-from C:/RL_Surf_cya/runs/cyPOTR/run.json \\
        --label tas_68.54 --t0 -12 --every 1 \\
        --out C:/RL_Surf_base/runs/research/videos/potential_finish_line_rel.mp4

Conventions on screen (fixed ranges, nothing is stretched per frame):
  * depth (top): the channel's own encoding - d / near inside near_range
    (2,000 u on these runs), 1 + 0.25 (1 - exp(-(d - near) / 2500)) beyond,
    ~1.244 for a ray clear at 11,500 u; brightness = 1 - value / 1.25, i.e.
    NEAR = BRIGHT, clear sky = black.
  * potential (bottom), the mode's clip range:
      rel  [-2, 2]  diverging, warm = positive = goal-ward, cool = leads
                    back; -2 (unreachable, or the eye unreachable) coolest.
      norm [-3, 3]  diverging with the SAME reading - warm = goal-ward -
                    which under norm is NEGATIVE (abs's sign, larger =
                    farther), so the colormap is flipped against the value;
                    +3 (unreachable) coolest; a frame under 8 honest pixels
                    is 0 = the neutral middle.
      abs  [0, 1.5] sequential, bright = 0 = at the finish, dark = 1 = at
                    the start, 1.5 (unreachable) darkest.
      logabs [0, 1.5] the same sequential reading on a LOG axis (0 = the
                    finish, 1 = the start, 1.5 unreachable): on cannonball
                    the wall region sits at 0.38 and the finish room at
                    0.07-0.13, where abs puts both under 0.04.
  With ``--obs-potential-curtain`` in the run's config the rays that cross
  the finish trigger read the GOAL - the brightest value under abs/logabs,
  the warmest under rel, the coolest end of norm's flipped map - so the
  finish appears as a patch in the potential channel where the depth
  channel shows only the wall behind it.

The map path must be the MAIN checkout's (the caches next to it are signed
against that bsp's mtime; a worktree copy re-bakes the goal field for 30
minutes - CLAUDE.md); a goal-field load over 30 s is reported as a re-bake.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import torch                                                    # noqa: E402

from surfgym import SurfCore, default_config                    # noqa: E402
from surfgym.goalfield import build_goal_field                  # noqa: E402
from surfgym.mapfleet import map_tag                            # noqa: E402
from surfgym.route import ArcProgress                           # noqa: E402
from surfgym.tick import step_seconds                           # noqa: E402
from surfgym.vision import GpuLidar, LidarPotential, pick_cell  # noqa: E402
from surfgym.zones import load_zones                            # noqa: E402

SCALE = 10                  # nearest-neighbour upscale of the 64x32 frames
SEP = 4                     # separator height, px
STRIP = 72                  # text strip height, px: three lines of LINE_H
LINE_H = 22                 # one text line, px (13 px Consolas + margins)
POT_RANGE = {"abs": (0.0, 1.5), "rel": (-2.0, 2.0), "norm": (-3.0, 3.0),
             "logabs": (0.0, 1.5)}
# the colormap per mode, and whether the VALUE axis is flipped so that warm
# always reads goal-ward (see the module docstring)
POT_CMAP = {"abs": ("viridis_r", False), "rel": ("RdBu_r", False),
            "norm": ("RdBu_r", True), "logabs": ("viridis_r", False)}
FONT_CANDIDATES = ("C:/Windows/Fonts/consola.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


# --------------------------------------------------------------- the config
VISION_KEYS = ("lidar_w", "lidar_h", "lidar_range", "lidar_near", "lidar_cell",
               "lidar_hfov", "lidar_vfov", "surf_mask", "pinhole", "normals",
               "goal_cell", "obs_potential", "obs_potential_d0",
               "obs_potential_curtain")


def load_cfg(traj: Path, run_json: str | None, ckpt: str | None) -> dict:
    rj = Path(run_json) if run_json else traj.parent / "run.json"
    if not rj.exists():
        raise SystemExit(f"no run.json next to {traj} (pass --config-from "
                         "RUN_JSON, a reference run's config)")
    cfg = json.loads(rj.read_text(encoding="utf-8")).get("config") or {}
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ccfg = ck.get("config") or {}
        bad = [k for k in VISION_KEYS if ccfg.get(k) != cfg.get(k)]
        if bad:
            raise SystemExit(f"{ckpt}: config differs from {rj} on {bad}")
        print(f"checkpoint config agrees with run.json on {len(VISION_KEYS)} "
              "vision keys")
    return cfg


def vision_cells(cfg: dict, core, map_stem: str):
    """The lidar cell and the goal-field cell exactly as record_ckpt.py
    resolves them (single-map runs: lidar_cell, goal_cell or the lidar cell)."""
    tag = map_tag(map_stem)
    cells = dict(cfg.get("heldout_map_cells") or {})
    cells.update(cfg.get("map_cells") or {})
    cell = float(cells.get(tag, cfg.get("lidar_cell") or pick_cell(core)))
    gcell = cell
    hgcells = cfg.get("heldout_goal_cells")
    gcells = cfg.get("goal_cells")
    gc = cfg.get("goal_cell")
    if isinstance(hgcells, dict) and tag in hgcells:
        gcell = float(hgcells[tag])
    elif isinstance(gcells, dict) and gcells:
        gcell = float(gcells.get(tag, gcell))
    elif isinstance(gc, str) and "," in gc:
        parts = [x.strip() for x in gc.split(",")]
        names = cfg.get("maps") or []
        idx = next((i for i, m in enumerate(names) if map_tag(m) == tag), None)
        if idx is not None and idx < len(parts) and parts[idx]:
            gcell = float(parts[idx])
    elif gc:
        gcell = float(gc)
    return cell, gcell


# -------------------------------------------------------------- the episode
def load_episodes(path: Path):
    """Every episode of a traj_*.jsonl as a full-width float array (all 15
    columns of surfgym.record: tick, x, y, z, vx, vy, vz, yaw, buttons,
    onground, progress, reward, pitch, fwd, side) plus its header dict.
    ``surfgym.route.episodes_from_traj`` keeps only the first 8 columns, and
    the render needs the pitch (12) and the IN_DUCK button (8). Same split
    rule: a header/footer dict, or the tick counter going backwards."""
    eps, hdrs, cur, cur_hdr, prev = [], [], [], None, None

    def flush():
        nonlocal cur, cur_hdr
        if len(cur) > 1:
            w = max(len(r) for r in cur)
            arr = np.zeros((len(cur), w), np.float64)
            for i, r in enumerate(cur):
                arr[i, :len(r)] = r
            eps.append(arr)
            hdrs.append(cur_hdr)
        cur, cur_hdr = [], None

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == "{":
                flush()
                prev = None
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    d = None
                cur_hdr = d if isinstance(d, dict) and "end" not in d else None
                continue
            row = json.loads(line)
            if not isinstance(row, list) or len(row) < 8:
                continue
            if prev is not None and row[0] <= prev:
                flush()                  # a header-less split: tick unknown
            prev = row[0]
            cur.append(row)
    flush()
    return eps, hdrs


DEMO_KEYS = ("t", "simorg", "simvel", "uc_viewangles", "uc_buttons", "onground")


def load_demo_npz(path: Path, map_name):
    """An HL demo's frames npz (tools/demo/wr_scan.py: parse_hldemo's
    per-frame arrays) as ONE episode in the traj row layout, plus a header,
    the per-row clock (the demo's playback ``t``, s) and the per-row dt.
    tick = frame index; pose = simorg; yaw = the usercmd yaw; pitch = MINUS
    the usercmd pitch (HL: positive = down; the simulator and the lidar:
    positive = up - wr_scan.load_record's rule); buttons = the usercmd's
    real low byte (bits 8..15 of the parsed ushort, wr_scan.decode_buttons),
    the IN_JUMP 2 / IN_DUCK 4 layout the recorder writes; fwd/side from the
    sign of the forward/side move. dt per row = the demo's own frame
    spacing, the last row's = the mean."""
    z = np.load(path)
    missing = [k for k in DEMO_KEYS if k not in z.files]
    if missing:
        raise SystemExit(f"{path}: not an HL demo frames npz, missing {missing}")
    t = np.asarray(z["t"], np.float64)
    n = len(t)
    dt = np.diff(t)
    if n < 2 or not np.all(dt > 0):
        raise SystemExit(f"{path}: the frame clock t is not increasing")
    ang = np.asarray(z["uc_viewangles"], np.float64)
    btn = (np.asarray(z["uc_buttons"], np.int64) >> 8) & 0xFF
    fsu = (np.asarray(z["uc_fsu"], np.float64) if "uc_fsu" in z.files
           else np.zeros((n, 3)))
    a = np.zeros((n, 15), np.float64)
    a[:, 0] = np.arange(n)
    a[:, 1:4] = np.asarray(z["simorg"], np.float64)
    a[:, 4:7] = np.asarray(z["simvel"], np.float64)
    a[:, 7] = ang[:, 1] % 360.0
    a[:, 8] = btn
    a[:, 9] = (np.asarray(z["onground"]) != 0)
    a[:, 12] = -ang[:, 0]
    a[:, 13] = np.where(fsu[:, 0] > 0, 2, np.where(fsu[:, 0] < 0, 0, 1))
    a[:, 14] = np.where(fsu[:, 1] > 0, 2, np.where(fsu[:, 1] < 0, 0, 1))
    dt = np.append(dt, dt.mean())
    header = {"map": map_name, "tick_ms": round(float(dt.mean() * 1000.0), 6),
              "source": "hldemo", "frames": int(n)}
    return a, header, t, dt


def order_only_progress(xyz: np.ndarray, route: ArcProgress) -> np.ndarray:
    """Running order-only corridor progress (u) per row: the rule of
    tools/eval_honesty.py::corridor_progress_ordered, kept per tick."""
    p = np.asarray(xyz, np.float64)
    route.reset(p[:1])
    out = np.empty(len(p), np.float64)
    best = float(route.arc[0])
    out[0] = best
    for k in range(1, len(p)):
        route.advance(p[k:k + 1])
        best = max(best, float(route.arc[0]))
        out[k] = best
    return out


def pick_episode(eps, hdrs, which: str, route: ArcProgress | None):
    if which != "best":
        i = int(which)
        if not 0 <= i < len(eps):
            raise SystemExit(f"episode {i} out of range (file has {len(eps)})")
        return i, None
    if route is None:
        raise SystemExit("--episode best needs the route file (--route)")
    scores = []
    for ep in eps:
        scores.append(float(order_only_progress(ep[:, 1:4], route)[-1]))
    i = int(np.argmax(scores))
    print("order-only progress per episode: "
          + "  ".join(f"ep{k} {s:,.0f}" for k, s in enumerate(scores)))
    return i, scores


# ---------------------------------------------------------------- the frame
def colormap_lut(name: str, flip: bool) -> np.ndarray:
    import matplotlib
    cm = matplotlib.colormaps[name]
    x = np.linspace(0.0, 1.0, 256)
    if flip:
        x = x[::-1]
    return (cm(x)[:, :3] * 255.0 + 0.5).astype(np.uint8)


def load_font(size: int):
    from PIL import ImageFont
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class FrameMaker:
    """Composes one RGB frame: depth (top), separator, potential, separator,
    text strip (a dynamic line over a static legend)."""

    def __init__(self, W: int, H: int, mode: str, enc_max: float,
                 rng_u: float, near_u: float, label: str):
        from PIL import Image, ImageDraw
        self.Image, self.ImageDraw = Image, ImageDraw
        self.W, self.H = W, H
        self.mode = mode
        self.enc_max = float(enc_max)
        self.vmin, self.vmax = POT_RANGE[mode]
        cmap, flip = POT_CMAP[mode]
        self.lut = colormap_lut(cmap, flip)
        self.fw, self.fh = W * SCALE, H * SCALE
        self.width = self.fw
        self.height = 2 * self.fh + 2 * SEP + STRIP
        self.font = load_font(13)
        self.y_pot = self.fh + SEP
        self.y_strip = 2 * self.fh + 2 * SEP
        self.label = label
        self.legend = self._legend(rng_u, near_u)

    def _legend(self, rng_u: float, near_u: float):
        """The static bottom two lines: the depth reading, and a colorbar
        for the potential in VALUE order (low value left) with its ends
        named."""
        img = self.Image.new("RGB", (self.width, STRIP - LINE_H), (24, 24, 24))
        d = self.ImageDraw.Draw(img)
        grey = (200, 200, 200)
        near_b = 1.0 - 1.0 / self.enc_max
        d.text((6, 3), f"depth: white = touching, {near_u:,.0f} u = "
                       f"{100 * near_b:.0f}% grey, black = clear at "
                       f"{rng_u:,.0f} u (fixed range)", fill=grey,
               font=self.font)
        lo, hi = self.vmin, self.vmax
        if self.mode == "rel":
            head = f"potential rel [{lo:g}, {hi:g}]:"
            left, right = f"{lo:+g} back / unreachable", f"{hi:+g} goal-ward"
        elif self.mode == "norm":
            head = f"norm [{lo:g}, {hi:g}] (warm = goal-ward):"
            left, right = f"{lo:+g} goal-ward", f"{hi:+g} far/unreachable"
        elif self.mode == "logabs":
            head = f"logabs [{lo:g}, {hi:g}] (log1p(d/1k) / log1p(d0/1k)):"
            left, right = f"{lo:g} finish", f"{hi:g} unreachable"
        else:
            head = f"potential abs [{lo:g}, {hi:g}] (d / d0):"
            left, right = f"{lo:g} finish", f"{hi:g} unreachable"
        y = LINE_H + 3
        x = 6
        d.text((x, y), head, fill=grey, font=self.font)
        x += d.textlength(head, font=self.font) + 8
        d.text((x, y), left, fill=grey, font=self.font)
        x += d.textlength(left, font=self.font) + 6
        bw, bh = 120, 12
        bar = self.lut[(np.arange(bw) * 255 // (bw - 1))]
        bar = np.repeat(bar[None, :, :], bh, 0)
        img.paste(self.Image.fromarray(bar), (int(x), y + 2))
        x += bw + 6
        d.text((x, y), right, fill=grey, font=self.font)
        return np.asarray(img)

    def depth_rgb(self, depth: np.ndarray) -> np.ndarray:
        g = 1.0 - np.clip(depth / self.enc_max, 0.0, 1.0)
        g8 = (g * 255.0 + 0.5).astype(np.uint8)
        g8 = np.repeat(np.repeat(g8, SCALE, 0), SCALE, 1)
        return np.repeat(g8[:, :, None], 3, 2)

    def pot_rgb(self, pot: np.ndarray) -> np.ndarray:
        u = (pot - self.vmin) / (self.vmax - self.vmin)
        idx = (np.clip(u, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        rgb = self.lut[idx]
        return np.repeat(np.repeat(rgb, SCALE, 0), SCALE, 1)

    def frame(self, depth: np.ndarray, pot: np.ndarray, line: str) -> bytes:
        out = np.full((self.height, self.width, 3), 24, np.uint8)
        out[:self.fh] = self.depth_rgb(depth)
        out[self.fh:self.fh + SEP] = 96
        out[self.y_pot:self.y_pot + self.fh] = self.pot_rgb(pot)
        out[self.y_pot + self.fh:self.y_strip] = 96
        out[self.y_strip + LINE_H:] = self.legend
        strip = self.Image.new("RGB", (self.width, LINE_H), (24, 24, 24))
        self.ImageDraw.Draw(strip).text((6, 4), line, fill=(235, 235, 235),
                                        font=self.font)
        out[self.y_strip:self.y_strip + LINE_H] = np.asarray(strip)
        return out.tobytes()


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traj", required=True)
    ap.add_argument("--episode", default="best",
                    help="index, or 'best' = largest order-only progress")
    ap.add_argument("--map", default=None,
                    help="bsp path (default C:/RL_Surf/maps/<header map>.bsp)")
    ap.add_argument("--route", default=None,
                    help="route npz for the progress (default next to the map)")
    ap.add_argument("--run-json", "--config-from", dest="run_json", default=None,
                    help="the config to build the lidar + channel from (default "
                         "run.json next to --traj); a REFERENCE run's for a "
                         "rollout whose own checkpoint had no potential channel")
    ap.add_argument("--ckpt", default=None,
                    help="cross-check the checkpoint's config against run.json")
    ap.add_argument("--label", default=None, help="run name on screen")
    ap.add_argument("--t0", type=float, default=None,
                    help="window start, s of the recording's own clock; "
                         "negative = counted back from its end")
    ap.add_argument("--t1", type=float, default=None,
                    help="window end, s (same clock, same rule)")
    ap.add_argument("--clock-offset", type=float, default=0.0,
                    help="show the time as rec = clock - S (a record's timer)")
    ap.add_argument("--every", type=int, default=2, help="render every k-th tick")
    ap.add_argument("--batch", type=int, default=256, help="ticks per render")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="debug cap")
    args = ap.parse_args()

    traj = Path(args.traj)
    cfg = load_cfg(traj, args.run_json, args.ckpt)
    mode = cfg.get("obs_potential")
    if mode not in POT_RANGE:
        raise SystemExit(f"{traj.parent / 'run.json'}: obs_potential is "
                         f"{mode!r}; this renders the potential channel")
    demo_dt = None                       # per-row dt of an npz input
    if traj.suffix.lower() == ".npz":
        a0, h0, t_rows, demo_dt = load_demo_npz(
            traj, Path(args.map).stem if args.map else cfg.get("map"))
        eps, hdrs = [a0], [h0]
    else:
        eps, hdrs = load_episodes(traj)
    if not eps:
        raise SystemExit(f"{traj}: no episodes")
    map_name = (hdrs[0] or {}).get("map") or cfg.get("map")
    map_path = args.map or f"C:/RL_Surf/maps/{map_name}.bsp"
    route_path = args.route or str(Path(map_path).with_suffix(".route.npz"))
    route = ArcProgress.load(route_path, corridor=1500.0, window=16) \
        if Path(route_path).exists() else None
    if route is None:
        print(f"no route at {route_path}: no progress readout")
    ep_i, scores = pick_episode(eps, hdrs, args.episode, route)
    a, header = eps[ep_i], hdrs[ep_i] or {}
    n = len(a)
    if demo_dt is not None:
        dt, t_s = demo_dt, t_rows                              # the demo's clock
    else:
        dt = np.asarray(step_seconds(header, n), np.float64)   # per-row s
        t_s = np.concatenate([[0.0], np.cumsum(dt)[:-1]])      # row -> s
    t_end = float(t_s[-1] + dt[-1])                            # after the last step
    prog = order_only_progress(a[:, 1:4], route) if route is not None else None
    rj = Path(args.run_json) if args.run_json else traj.parent / "run.json"
    own_cfg = rj.resolve().parent == traj.resolve().parent
    label = args.label or (json.loads(rj.read_text(encoding="utf-8")).get("label")
                           if own_cfg else None) or traj.stem
    print(f"{traj.name} episode {ep_i}/{len(eps)}: {n} ticks, clock "
          f"{t_s[0]:.3f}..{t_end:.3f} s"
          + (f", order-only {prog[-1]:,.0f} u ({100 * prog[-1] / route.length:.1f}%)"
             if prog is not None else "")
          + ("" if own_cfg else f"; config from {rj} (a reference run)"))

    # ---- the renderer, record_ckpt.py's way ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = SurfCore(map_path, default_config(num_envs=1, lidar_w=0, lidar_h=0))
    stem = Path(map_path).stem
    cell, gcell = vision_cells(cfg, core, stem)
    zones = load_zones(core.bsp_path)
    t0 = time.time()
    gf = build_goal_field(core, zones["end"], cell=gcell)
    dt_gf = time.time() - t0
    print(f"goal field @ cell {gcell:g} in {dt_gf:.1f}s"
          + ("  ** WARNING: that smells like a RE-BAKE - wrong map "
             "path/mtime? **" if dt_gf > 30 else ""))
    lw, lh = int(cfg.get("lidar_w") or 64), int(cfg.get("lidar_h") or 32)
    rng_u = float(cfg.get("lidar_range", 2000.0))
    near = cfg.get("lidar_near")
    pot = LidarPotential.from_cfg(cfg, gf, core, device, stem)
    lidar = GpuLidar(core, lw, lh,
                     hfov_deg=float(cfg.get("lidar_hfov") or 120.0),
                     vfov_deg=float(cfg.get("lidar_vfov") or 90.0),
                     range_units=rng_u, near_range=near, cell=cell,
                     device=device,
                     surf_mask=bool(cfg.get("surf_mask", 0)),
                     pinhole=bool(cfg.get("pinhole", 0)),
                     normals=bool(cfg.get("normals", 0)),
                     potential=pot)
    if lidar.channels != 2:
        raise SystemExit(f"lidar has {lidar.channels} channels, expected 2")
    print(pot.describe())
    enc_max = 1.25 if (near and float(near) < rng_u) else 1.0
    near_u = float(near) if near else rng_u
    print(f"depth channel: d/{near_u:g} inside {near_u:g} u"
          + (f", 1 + 0.25(1 - exp(-(d - {near_u:g})/2500)) beyond, clear at "
             f"{rng_u:g} u = "
             f"{1 + 0.25 * (1 - np.exp(-(rng_u - near_u) / 2500.0)):.4f}"
             if enc_max > 1.0 else f", 1.0 = clear at {rng_u:g} u")
          + f"; shown as brightness 1 - v/{enc_max:g} (near = bright)")

    # ---- the poses --------------------------------------------------------
    def _abs_t(v):
        return None if v is None else (t_end + v if v < 0 else v)
    w0, w1 = _abs_t(args.t0), _abs_t(args.t1)
    sel = np.ones(n, bool)
    if w0 is not None:
        sel &= t_s >= w0 - 1e-9
    if w1 is not None:
        sel &= t_s <= w1 + 1e-9
    kept = np.flatnonzero(sel)
    if len(kept) == 0:
        raise SystemExit(f"window [{w0}, {w1}] s selects no row of a recording "
                         f"on {t_s[0]:.3f}..{t_end:.3f} s")
    rows = kept[::max(1, int(args.every))]
    if w0 is not None or w1 is not None:
        print(f"window --t0 {args.t0} --t1 {args.t1} -> rows {kept[0]}..{kept[-1]} "
              f"of {n}, clock {t_s[kept[0]]:.3f}..{t_s[kept[-1]]:.3f} s "
              f"(+{dt[kept[-1]]:.4f} s step) of a recording ending {t_end:.3f} s"
              + (f"; shown as rec = clock - {args.clock_offset:g} s"
                 if args.clock_offset else ""))
    if args.max_frames > 0:
        rows = rows[:args.max_frames]
    # real-time rate: each frame stands for `every` ticks of the rendered
    # row's duration (exactly 1000 / (tick_ms * every) on a uniform tick)
    span = float(dt[rows].sum()) * max(1, int(args.every))
    fps = len(rows) / span if span > 0 else 50.0
    pos = a[rows, 1:4].astype(np.float32)
    yaw = a[rows, 7].astype(np.float32)
    pitch = (a[rows, 12] if a.shape[1] > 12 else np.zeros(len(rows))).astype(np.float32)
    duck = ((a[rows, 8].astype(np.int64) & 4) != 0).astype(np.int32)

    # ---- the video --------------------------------------------------------
    fm = FrameMaker(lw, lh, mode, enc_max, rng_u, near_u, label)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{fm.width}x{fm.height}",
           "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", str(args.crf), "-preset", "medium",
           "-movflags", "+faststart", str(out)]
    print(f"{len(rows)} frames (every {args.every} tick(s)) at {fps:.3f} fps "
          f"-> {fm.width}x{fm.height} {out}")
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    stats = {"pot_min": np.inf, "pot_max": -np.inf, "bad": 0, "px": 0,
             "depth_min": np.inf, "depth_max": -np.inf}
    bad_value = {"abs": 1.5, "rel": -2.0, "norm": 3.0, "logabs": 1.5}[mode]
    tname = "rec" if args.clock_offset else "t"
    t_start = time.time()
    try:
        for b0 in range(0, len(rows), max(1, int(args.batch))):
            b1 = min(len(rows), b0 + max(1, int(args.batch)))
            o = torch.as_tensor(pos[b0:b1], device=device)
            yw = torch.as_tensor(yaw[b0:b1], device=device)
            pt = torch.as_tensor(pitch[b0:b1], device=device)
            dk = torch.as_tensor(duck[b0:b1], device=device)
            img = lidar.render(o, yw, pt, dk)
            if img.ndim != 4 or img.shape[-1] != 2:
                raise SystemExit(f"render returned {tuple(img.shape)}")
            img = img.float().cpu().numpy()                    # (B,H,W,2)
            depth, ch = img[..., 0], img[..., 1]
            stats["pot_min"] = min(stats["pot_min"], float(ch.min()))
            stats["pot_max"] = max(stats["pot_max"], float(ch.max()))
            stats["depth_min"] = min(stats["depth_min"], float(depth.min()))
            stats["depth_max"] = max(stats["depth_max"], float(depth.max()))
            stats["bad"] += int((ch == bad_value).sum())
            stats["px"] += ch.size
            for j in range(b1 - b0):
                r = rows[b0 + j]
                line = (f"{label} {mode}  {tname} {t_s[r] - args.clock_offset:6.2f} s  "
                        f"tick {int(a[r, 0]):5d}")
                if prog is not None:
                    line += (f"  progress {prog[r]:8,.0f} u "
                             f"({100 * prog[r] / route.length:4.1f}%)")
                line += f"  pot {ch[j].min():+.2f}..{ch[j].max():+.2f}"
                enc.stdin.write(fm.frame(depth[j], ch[j], line))
            done = b1
            el = time.time() - t_start
            print(f"  {done}/{len(rows)} frames  {el:.0f}s", end="\r")
    finally:
        enc.stdin.close()
        rc = enc.wait()
    print()
    if rc != 0:
        raise SystemExit(f"ffmpeg exited {rc}")
    size = out.stat().st_size
    print(f"wrote {out}  {size / 1e6:.1f} MB  ({len(rows)} frames, "
          f"{len(rows) / fps:.2f} s at {fps:.3f} fps)")
    print(f"depth channel range {stats['depth_min']:.4f}..{stats['depth_max']:.4f}; "
          f"potential {mode} range {stats['pot_min']:+.3f}..{stats['pot_max']:+.3f}, "
          f"bad ({bad_value:+g}) pixels {100.0 * stats['bad'] / max(1, stats['px']):.3f}%")
    summary = {
        "traj": str(traj), "episode": ep_i, "episodes": len(eps),
        "source": header.get("source") or "traj",
        "config_from": str(rj), "own_config": bool(own_cfg), "label": label,
        "ticks": int(n), "seconds": float(t_end - t_s[0]),
        "clock_start": float(t_s[0]), "clock_end": t_end,
        "window": {"t0": w0, "t1": w1, "row_first": int(kept[0]),
                   "row_last": int(kept[-1]),
                   "clock_first": float(t_s[kept[0]]),
                   "clock_last": float(t_s[kept[-1]]),
                   "clock_offset": float(args.clock_offset)},
        "order_only_u": (float(prog[-1]) if prog is not None else None),
        "order_only_all": scores, "mode": mode, "every": int(args.every),
        "fps": fps, "frames": int(len(rows)), "size_bytes": int(size),
        "width": fm.width, "height": fm.height,
        "depth_range": [stats["depth_min"], stats["depth_max"]],
        "pot_range": [stats["pot_min"], stats["pot_max"]],
        "bad_share": stats["bad"] / max(1, stats["px"]),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
