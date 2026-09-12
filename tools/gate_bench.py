#!/usr/bin/env python3
"""gate_bench.py - the GATE benchmark: does a policy give up immediate
potential to pass a place where the field's own descent runs into a void?

The pattern (user, 2026-09-12): where a map needs the player to fly AGAINST
or ACROSS the goal potential for a while (cannonball's final room: turn
right and dive for the immediate potential, or turn left, take two ramps,
give up ~4 reward and keep the speed to finish; celestial's fork at 30%:
the field points east into open air, the real line flies sideways along a
contour for 2 s to the mirrored ramp), the greedy policy takes the paying
branch and dies. This benchmark starts rollouts a few seconds BEFORE such a
place, from a WINDOW of full core states cut out of a line that gets there
with the right speed and direction (a finisher's own greedy episode, or the
current policy's - it reaches the fork fine, it just turns wrong), and
counts how many rollouts pass.

Three subcommands:

  spine   cut the window out of a record_ckpt --dump-states dump and save it
          as a time-ordered STATE_DTYPE .npy (what train_fast --demo-file and
          record_ckpt --spawn-states take). Writes <out>_starts.png: where the
          starts are and with which speed, over the potential.
  run     roll a checkpoint from the window (record_ckpt --spawn-states,
          N episodes, greedy or --stochastic) and score it.
  score   score a recording: per episode, did it PASS (reach d < --d-pass
          alive, i.e. beyond where the void branch can get), did it go
          SIDEWAYS (|y - axis| > --side-min, the contour flight), did the
          map push back inside a --ramp-box, how much potential RISE it
          accepted before passing, and where it died. Writes <traj>_score.json,
          _score.csv and _ends.png (every rollout over the potential).

    python tools/gate_bench.py spine --dump runs/research/gate_bench/celestial_greedy_states.npz \
        --traj runs/research/gate_bench/celestial_greedy.jsonl --t0 6.0 --t1 8.5 \
        --map maps_pool/surf_src_celestial.bsp --goal-cell 48 --occ-cell 32 \
        --out runs/research/gate_bench/celestial_pregate.npy
    python tools/gate_bench.py run --ckpt runs/jt3ANCHU/ckpt_final.pt \
        --map maps_pool/surf_src_celestial.bsp --goal-cell 48 --occ-cell 32 \
        --spine runs/research/gate_bench/celestial_pregate.npy --episodes 64 --ep-ticks 1500 \
        --stochastic --d-pass 40000 --axis-y 552 --side-min 2000 --out-dir runs/research/gate_bench/celestial_jt3

The field and the occupancy come out of the baked caches next to the .bsp;
nothing bakes.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE                     # noqa: E402
from bev_potential import (load_goal, load_occ, load_episodes,   # noqa: E402
                           paint_potential, GRAVITY, SURF, INK, GRID)

C_PASS, C_FAIL, C_TRUNC, C_START = "#1baf7a", "#e0452b", "#8a8781", "#ffd23f"


def tick_ms_of(traj: Path | None, fallback: float) -> float:
    if traj is None:
        return float(fallback)
    with open(traj, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and line[0] == "{":
                h = json.loads(line)
                if "tick_ms" in h:
                    return float(h["tick_ms"])
                break
    return float(fallback)


def push_mask(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(t) < 3:
        return np.zeros(len(t), bool)
    az = np.gradient(v[:, 2], t)
    return np.abs(az + GRAVITY) > 150.0


def field_and_occ(a):
    gf = load_goal(Path(a.map), a.goal_cell)
    occ, omins, ocell = load_occ(Path(a.map), a.occ_cell)
    return gf, occ, omins, ocell


def bev_window(pts: np.ndarray, pad: float, override):
    if override:
        return tuple(override)
    x0, y0 = pts[:, :2].min(0) - pad
    x1, y1 = pts[:, :2].max(0) + pad
    return float(x0), float(x1), float(y0), float(y1)


def zslab(pts: np.ndarray, override):
    if override:
        return float(override[0]), float(override[1])
    return float(pts[:, 2].min()) - 64.0, float(pts[:, 2].max()) + 64.0


def draw_wr(ax, path: str | None, mirror_y):
    if not path:
        return
    eps, _h = load_episodes(Path(path))
    p = eps[0][:, 1:4]
    # a demo ends with a teleport back to the start: break the line there
    jump = np.where(np.linalg.norm(np.diff(p, axis=0), axis=1) > 600.0)[0]
    segs = np.split(p, jump + 1) if len(jump) else [p]
    for k, sgm in enumerate(segs):
        ax.plot(sgm[:, 0], sgm[:, 1], "-", color="#2a78d6", lw=1.4, alpha=0.9,
                label="human record" if k == 0 else None, zorder=3)
        if mirror_y is not None:
            ax.plot(sgm[:, 0], 2.0 * mirror_y - sgm[:, 1], "--", color="#7b3fb8", lw=1.2,
                    alpha=0.9, zorder=3,
                    label=f"record mirrored about y = {mirror_y:.0f}" if k == 0 else None)


# ------------------------------------------------------------------ spine --
def cmd_spine(a):
    z = np.load(a.dump)
    key = f"ep_{a.episode:04d}"
    if key not in z.files:
        raise SystemExit(f"{a.dump} has no {key}; keys: {z.files[:5]}...")
    S = np.asarray(z[key], STATE_DTYPE)
    tm = tick_ms_of(Path(a.traj) if a.traj else None, a.tick_ms)
    t = np.arange(len(S)) * tm / 1000.0
    m = (t >= a.t0) & (t <= a.t1)
    W = S[m][::max(1, a.every)]
    tw = t[m][::max(1, a.every)]
    if len(W) == 0:
        raise SystemExit(f"no states in [{a.t0}, {a.t1}] s (episode is {t[-1]:.2f} s long)")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, W)
    pos = W["origin"].astype(np.float64)
    vel = W["velocity"].astype(np.float64)
    spd = np.linalg.norm(vel, axis=1)
    print(f"spine {out}: {len(W)} states, t {tw[0]:.2f}..{tw[-1]:.2f} s of the dump "
          f"(tick {tm:.4g} ms), |v| {spd.min():.0f}..{spd.max():.0f} u/s "
          f"(mean {spd.mean():.0f}), z {pos[:, 2].min():.0f}..{pos[:, 2].max():.0f}, "
          f"onground {int(W['onground'].sum())}/{len(W)}")
    if not a.map:
        return
    gf, occ, omins, ocell = field_and_occ(a)
    d = gf.sample(pos)
    print(f"  potential d {d.min():.0f}..{d.max():.0f} u ({100 * d.min() / gf.sample(S['origin'][:1].astype(np.float64))[0]:.1f}"
          f"..{100 * d.max() / gf.sample(S['origin'][:1].astype(np.float64))[0]:.1f}% of the episode's start d)")

    # the whole greedy line for context, the window as arrows coloured by speed
    full = S["origin"].astype(np.float64)
    ctx = full[(t >= a.t0 - a.context) & (t <= a.t1 + a.context)]
    x0, x1, y0, y1 = bev_window(np.concatenate([ctx, pos]), a.pad, a.bev_window)
    zlo, zhi = zslab(np.concatenate([ctx, pos]), a.zslab)
    fig, ax = plt.subplots(figsize=(12.0, 10.0))
    paint_potential(fig, ax, gf, occ, omins, ocell, x0, x1, y0, y1, zlo, zhi,
                    quiver_step=a.quiver_step)
    draw_wr(ax, a.wr, a.mirror_y)
    ax.plot(full[:, 0], full[:, 1], "-", color="#eb6834", lw=1.6, alpha=0.9,
            label="the dumped greedy line (whole episode)", zorder=4)
    sc = ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], spd,
                   cmap="plasma", scale=40000, width=0.003, headwidth=4,
                   zorder=6, clim=(0, max(3000.0, spd.max())))
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08)
    cb.set_label("start speed |v| (u/s); arrow = velocity direction")
    for k in range(0, len(W), max(1, len(W) // 8)):
        ax.annotate(f"{tw[k]:.1f}s z{pos[k, 2]:.0f}", (pos[k, 0], pos[k, 1]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7,
                    color=INK, zorder=7)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("x (map units)"); ax.set_ylabel("y (map units)")
    ax.set_title(f"{Path(a.map).stem}: the {len(W)} benchmark start states "
                 f"(t {tw[0]:.1f}..{tw[-1]:.1f} s of the greedy line), "
                 f"|v| {spd.min():.0f}..{spd.max():.0f} u/s", loc="left", fontsize=10)
    ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.92)
    ax.grid(True, color=GRID, lw=0.4, zorder=1)
    fig.tight_layout()
    png = out.with_name(out.stem + "_starts.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"  {png}")


# ------------------------------------------------------------------ score --
def score_traj(a, traj: Path):
    eps, hdrs = load_episodes(traj)
    trailers = []
    with open(traj, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and line[0] == "{":
                dct = json.loads(line)
                if "end" in dct:
                    trailers.append(dct)
    gf, occ, omins, ocell = field_and_occ(a)
    boxes = [np.asarray(b, np.float64).reshape(2, 3).T for b in (a.ramp_box or [])]
    rows = []
    lines = []
    for i, (e, h) in enumerate(zip(eps, hdrs)):
        tm = float((h or {}).get("tick_ms", 10.0))
        t = e[:, 0] * tm / 1000.0
        t = t - t[0]
        p, v = e[:, 1:4].astype(np.float64), e[:, 4:7].astype(np.float64)
        d = gf.sample(p)
        push = push_mask(v, t)
        end = trailers[i]["end"] if i < len(trailers) else "?"
        if a.d_pass is not None:
            ok = d < a.d_pass
            if a.pass_zmin is not None:
                # a dive under the finish can read a small d in goal-adjacent
                # air (CLAUDE.md): the pass must happen at a legal height
                ok &= p[:, 2] >= a.pass_zmin
            passed = bool(ok.any())
            i_pass = int(np.argmax(ok)) if passed else len(d) - 1
            if passed and a.pass_hold > 0.0 and end == "fail" and t[-1] - t[i_pass] < a.pass_hold:
                # reached the number and died right after: the void route
                # flown a little further, not the far side of the gate
                passed = False
                i_pass = len(d) - 1
        else:
            passed = end == "done"
            i_pass = len(d) - 1
        run_min = np.minimum.accumulate(d[:i_pass + 1])
        rise = float(np.max(d[:i_pass + 1] - run_min))
        side = float(np.max(np.abs(p[:, 1] - a.axis_y))) if a.axis_y is not None else float("nan")
        sideways = bool(side > a.side_min) if a.axis_y is not None else False
        hit = False
        for b in boxes:
            inb = np.all((p >= b[:, 0]) & (p <= b[:, 1]), axis=1)
            hit |= bool(np.any(inb & push))
        n_contacts = int(np.sum(np.diff(push.astype(int)) == 1))
        rows.append(dict(ep=i, end=end, passed=passed, t_pass=(float(t[i_pass]) if passed else None),
                         d_min=float(d.min()), d_end=float(d[-1]), z_end=float(p[-1, 2]),
                         t_end=float(t[-1]), rise_before_pass=rise, side_max=side,
                         sideways=sideways, ramp_hit=hit, contacts=n_contacts,
                         d0=float(d[0]), speed0=float(np.linalg.norm(v[0])),
                         x0=float(p[0, 0]), y0=float(p[0, 1]), z0=float(p[0, 2])))
        lines.append((p, passed, end))
    n = len(rows)
    P = [r for r in rows if r["passed"]]
    F = [r for r in rows if not r["passed"]]
    summ = dict(
        traj=str(traj), episodes=n,
        pass_rate=len(P) / max(n, 1),
        sideways_rate=sum(r["sideways"] for r in rows) / max(n, 1),
        ramp_hit_rate=sum(r["ramp_hit"] for r in rows) / max(n, 1),
        # the recorder labels a finish crossing "fail" too (its bonus check);
        # a death is an unpassed episode the core ended
        death_rate=sum(r["end"] == "fail" and not r["passed"] for r in rows) / max(n, 1),
        trunc_rate=sum(r["end"] == "trunc" for r in rows) / max(n, 1),
        t_pass_mean=(float(np.mean([r["t_pass"] for r in P])) if P else None),
        rise_pass_mean=(float(np.mean([r["rise_before_pass"] for r in P])) if P else None),
        rise_fail_mean=(float(np.mean([r["rise_before_pass"] for r in F])) if F else None),
        d_min_fail_median=(float(np.median([r["d_min"] for r in F])) if F else None),
        d_end_fail_median=(float(np.median([r["d_end"] for r in F])) if F else None),
        z_end_fail_median=(float(np.median([r["z_end"] for r in F])) if F else None),
        t_end_fail_mean=(float(np.mean([r["t_end"] for r in F])) if F else None),
        speed0_mean=float(np.mean([r["speed0"] for r in rows])),
        d0_range=[float(min(r["d0"] for r in rows)), float(max(r["d0"] for r in rows))],
        d_pass=a.d_pass, axis_y=a.axis_y, side_min=a.side_min, ramp_boxes=len(boxes),
    )
    base = traj.with_suffix("")
    with open(f"{base}_score.json", "w", encoding="utf-8") as fh:
        json.dump(dict(summary=summ, episodes=rows), fh, indent=1)
    with open(f"{base}_score.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def f(x, nd=2):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
    print(f"{traj.name}: {n} episodes | PASS {100 * summ['pass_rate']:.0f}% | sideways "
          f"{100 * summ['sideways_rate']:.0f}% | ramp-box contact {100 * summ['ramp_hit_rate']:.0f}% | "
          f"deaths {100 * summ['death_rate']:.0f}% trunc {100 * summ['trunc_rate']:.0f}%")
    print(f"  passers: t_pass {f(summ['t_pass_mean'])} s, rise accepted before the pass "
          f"{f(summ['rise_pass_mean'], 0)} u | failers: d_min median {f(summ['d_min_fail_median'], 0)}, "
          f"d_end median {f(summ['d_end_fail_median'], 0)}, z_end median {f(summ['z_end_fail_median'], 0)}, "
          f"t_end {f(summ['t_end_fail_mean'])} s, rise {f(summ['rise_fail_mean'], 0)} u")
    print(f"  starts: |v| mean {summ['speed0_mean']:.0f} u/s, d {summ['d0_range'][0]:.0f}..{summ['d0_range'][1]:.0f}")

    # every rollout over the potential
    allp = np.concatenate([p for p, _ok, _e in lines])
    x0, x1, y0, y1 = bev_window(allp, a.pad, a.bev_window)
    zlo, zhi = zslab(allp, a.zslab)
    fig, ax = plt.subplots(figsize=(12.0, 10.0))
    paint_potential(fig, ax, gf, occ, omins, ocell, x0, x1, y0, y1, zlo, zhi,
                    quiver_step=a.quiver_step)
    draw_wr(ax, a.wr, a.axis_y if a.mirror_wr else None)
    for b in boxes:
        ax.add_patch(plt.Rectangle((b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0],
                                   fill=False, ec=C_START, lw=1.5, ls="--", zorder=5))
    for p, ok, end in lines:
        col = C_PASS if ok else (C_TRUNC if end == "trunc" else C_FAIL)
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=0.9, alpha=0.6, zorder=4)
        ax.plot(p[-1, 0], p[-1, 1], "x" if not ok else "o", color=col, ms=5, zorder=5)
    starts = np.asarray([[r["x0"], r["y0"]] for r in rows])
    ax.plot(starts[:, 0], starts[:, 1], ".", color=C_START, ms=6, zorder=6, label="starts")
    ax.plot([], [], "-", color=C_PASS, label=f"passed ({len(P)})")
    ax.plot([], [], "-", color=C_FAIL, label=f"died ({sum(r['end'] == 'fail' and not r['passed'] for r in rows)})")
    ax.plot([], [], "-", color=C_TRUNC, label=f"time cap ({sum(r['end'] == 'trunc' for r in rows)})")
    if a.axis_y is not None:
        ax.axhline(a.axis_y, color=INK, ls=":", lw=0.9, zorder=3)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("x (map units)"); ax.set_ylabel("y (map units)")
    ax.set_title(f"{Path(a.map).stem} gate benchmark: {traj.name} - pass {100 * summ['pass_rate']:.0f}% "
                 f"of {n}, sideways {100 * summ['sideways_rate']:.0f}%", loc="left", fontsize=10)
    ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.92)
    ax.grid(True, color=GRID, lw=0.4, zorder=1)
    fig.tight_layout()
    fig.savefig(f"{base}_ends.png", dpi=150)
    plt.close(fig)
    print(f"  {base}_score.json  {base}_ends.png")
    return summ


def cmd_score(a):
    for tr in a.traj:
        score_traj(a, Path(tr))


# -------------------------------------------------------------------- run --
def cmd_run(a):
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = a.name or (Path(a.ckpt).parent.name + ("_stoch" if a.stochastic else "_greedy"))
    traj = out_dir / f"{name}.jsonl"
    cmd = [sys.executable, str(ROOT / "tools" / "record_ckpt.py"), str(a.ckpt),
           "--spawn-states", str(a.spine), "--episodes", str(a.episodes),
           "--ep-ticks", str(a.ep_ticks), "--out", str(traj)]
    if a.map:
        cmd += ["--map", str(a.map)]
    if a.stochastic:
        cmd.append("--stochastic")
    if a.seed is not None:
        cmd += ["--seed", str(a.seed)]
    cmd += list(a.extra or [])
    print("+", " ".join(cmd))
    with open(out_dir / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = (out_dir / f"{name}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        raise SystemExit("record_ckpt failed:\n" + "\n".join(tail))
    return score_traj(a, traj)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--map", default=None, help="the .bsp; caches sit next to it")
        p.add_argument("--goal-cell", type=int, default=32)
        p.add_argument("--occ-cell", type=int, default=32)
        p.add_argument("--wr", default=None, help="record line .jsonl drawn for context")
        p.add_argument("--bev-window", nargs=4, type=float, default=None)
        p.add_argument("--zslab", nargs=2, type=float, default=None)
        p.add_argument("--pad", type=float, default=500.0)
        p.add_argument("--quiver-step", type=int, default=6)

    ps = sub.add_parser("spine", help="cut the start window out of a --dump-states dump")
    ps.add_argument("--dump", required=True, help="record_ckpt --dump-states .npz")
    ps.add_argument("--episode", type=int, default=0)
    ps.add_argument("--traj", default=None, help="the .jsonl recorded with the dump (its tick_ms)")
    ps.add_argument("--tick-ms", type=float, default=10.0, help="if no --traj")
    ps.add_argument("--t0", type=float, required=True)
    ps.add_argument("--t1", type=float, required=True)
    ps.add_argument("--every", type=int, default=1, help="keep every k-th tick")
    ps.add_argument("--context", type=float, default=6.0, help="seconds of the line drawn around the window")
    ps.add_argument("--mirror-y", type=float, default=None)
    ps.add_argument("--out", required=True, help=".npy")
    common(ps)
    ps.set_defaults(fn=cmd_spine)

    def scoring(p):
        p.add_argument("--d-pass", type=float, default=None,
                       help="PASS = reached d below this alive; default: the recorder's 'done'")
        p.add_argument("--pass-hold", type=float, default=3.0,
                       help="a pass must stay alive this many seconds after the passing tick "
                            "(a death sooner is the dead-end flown a little further)")
        p.add_argument("--pass-zmin", type=float, default=None,
                       help="a pass also needs z >= this at the passing tick (no dives under the finish)")
        p.add_argument("--axis-y", type=float, default=None, help="the fork's mirror plane")
        p.add_argument("--side-min", type=float, default=2000.0,
                       help="SIDEWAYS = |y - axis| exceeds this")
        p.add_argument("--ramp-box", nargs=6, type=float, action="append", default=None,
                       metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                       help="a ramp-contact box (repeatable)")
        p.add_argument("--mirror-wr", action="store_true", help="also draw the record mirrored about --axis-y")

    pc = sub.add_parser("score", help="score recordings")
    pc.add_argument("--traj", nargs="+", required=True)
    common(pc); scoring(pc)
    pc.set_defaults(fn=cmd_score)

    pr = sub.add_parser("run", help="roll a checkpoint from the spine and score it")
    pr.add_argument("--ckpt", required=True)
    pr.add_argument("--spine", required=True)
    pr.add_argument("--episodes", type=int, default=64)
    pr.add_argument("--ep-ticks", type=int, default=1500)
    pr.add_argument("--stochastic", action="store_true")
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--name", default=None)
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--extra", nargs=argparse.REMAINDER, help="passed to record_ckpt verbatim")
    common(pr); scoring(pr)
    pr.set_defaults(fn=cmd_run)

    a = ap.parse_args()
    if a.cmd != "spine" and not a.map:
        ap.error("--map is required")
    a.fn(a)


if __name__ == "__main__":
    main()
