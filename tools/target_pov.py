"""target_pov.py - a video of what the executor SEES along a recorded run: its depth image and the
target-ramp channel (surfgym.targetmask), side by side, in real time. The targets follow the
recording's own ramp sequence from the core's contact telemetry: the next ramp (+1, white) and the
one after it (-1, black) advance at each touch of the next one; the finish box is the last target.
MEASUREMENT / visualisation only - use our own policies' recordings (CLAUDE.md section 0).

    python tools/target_pov.py <mover ckpt> --map maps_pool/<m>.bsp --traj <run>.jsonl
        --touch <run>.npz --ramps-old <ramps3.npz> --mesh <ramps_mesh.npz> --out pov.mp4
        [--every 4] [--scale 10] [--unit face]

--touch: per tick the contact SETS in --ramps-old ids (tools' finisher_touch recording); each
old id is mapped to the face-extraction surface whose contact samples it overlaps most.
"""
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--map", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--touch", required=True)
    ap.add_argument("--ramps-old", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4, help="ticks per video frame (4 = 25 fps)")
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--unit", choices=("face", "object"), default="face")
    ap.add_argument("--schedule", choices=("takeoff", "touch"), default="takeoff",
                    help="takeoff (default): a target stays lit while it is ridden and fades out "
                         "over --fade s after the TAKEOFF, the next brightening 0.5 -> 1 and the "
                         "one after fading in 0 -> 0.5 (targetmask.takeoff_fade_values: "
                         "continuous); touch: the old +1 / -1 switch at the first touch")
    ap.add_argument("--fade", type=float, default=0.3, help="seconds of cross-fade")
    a = ap.parse_args(argv)
    import cv2
    import torch
    from scipy.spatial import cKDTree
    import record_ckpt
    from ramps import RampMap
    from surfgym.targetmask import FIN, NONE, TargetMask, takeoff_fade_values
    import edge_archive as ea

    rows = [json.loads(ln) for ln in open(a.traj, encoding="utf-8") if ln.startswith("[")]
    A = np.asarray([r[:13] for r in rows], np.float64)
    tz = np.load(a.touch, allow_pickle=True)
    sets = [str(s) for s in tz["sets"]]
    duck = np.asarray(tz["duck"], np.int64)
    n = min(len(A), len(sets))
    # old surface ids -> face-extraction surfaces (majority of the old surface's samples)
    old = RampMap(a.ramps_old)
    zm = np.load(a.mesh)
    tree = cKDTree(zm["points"])
    mcat = zm["cat"]
    rng = np.random.default_rng(0)
    o2m = {}
    for s in sorted({int(x) for ss in sets for x in ss.split(",") if x not in ("", "-4")}):
        pts = old.kp[old.members[s]]
        if len(pts) > 400:
            pts = pts[rng.choice(len(pts), 400, replace=False)]
        d, i = tree.query(pts)
        ids, c = np.unique(zm["surf"][i[d < 64]], return_counts=True)
        if len(ids):
            o2m[s] = int(ids[np.argmax(c)])
    # the ride sequence: target surfaces (floors, ramps) in touch order, consecutive repeats merged
    seq, first_t = [], []
    for t in range(n):
        for x in sets[t].split(","):
            if x in ("", "-4") or int(x) not in o2m:
                continue
            m = o2m[int(x)]
            if int(mcat[m]) not in (0, 1):
                continue
            if not seq or seq[-1] != m:
                if m in seq[-2:]:
                    continue            # a flicker back onto the previous surface
                seq.append(m)
                first_t.append(t)
    # the TAKEOFF from each ride: the first tick at which the agent has been off that surface
    # for DEPART_TICKS consecutive ticks after touching it (the ride contract's own rule)
    msets = [{o2m[int(x)] for x in s.split(",") if x not in ("", "-4") and int(x) in o2m}
             for s in sets[:n]]
    leave = []
    for i, (s, t0) in enumerate(zip(seq, first_t)):
        last = t0
        tl = np.inf
        for tt_ in range(t0, n):
            if s in msets[tt_]:
                last = tt_
            elif tt_ - last >= ea.DEPART_TICKS:
                tl = float(last + ea.DEPART_TICKS)
                break
        leave.append(tl)
    tgt_seq = list(seq) + [FIN]                 # the finish is the last target; never left
    leave_all = np.asarray(leave + [np.inf])
    fade_t = float(a.fade) * 100.0
    print(f"target_pov: {n} ticks, ride sequence of {len(seq)} target surfaces: "
          + " ".join(f"{s}@{t / 100:.1f}-{(lv / 100 if np.isfinite(lv) else float('nan')):.1f}s"
                     for s, t, lv in zip(seq, first_t, leave)))
    ctx = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--map", str(a.map)])
    lidar = ctx.pol.lidar
    dev = lidar.device
    tm = TargetMask(a.mesh, ctx.finish_box, dev, unit=a.unit)
    remap = (lambda s: int(tm.obj_of_surf[s])) if a.unit == "object" else (lambda s: s)
    S = int(a.scale)
    W, H = lidar.W * S, lidar.H * S
    bar = 70
    FW, FH = 2 * W + 30, H + bar + 34
    FW += FW % 2
    FH += FH % 2
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                           "gray", "-s", f"{FW}x{FH}", "-r", str(100 // int(a.every)), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt",
                           "yuv420p", str(a.out)], stdin=subprocess.PIPE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    k = -1
    ticks = list(range(0, n, int(a.every)))
    prev_ch, prev_dp, jumps = None, None, []
    B = 64
    for c0 in range(0, len(ticks), B):
        tt = ticks[c0:c0 + B]
        o = torch.as_tensor(A[tt, 1:4], dtype=torch.float32, device=dev)
        yaw = torch.as_tensor(A[tt, 7], dtype=torch.float32, device=dev)
        pitch = torch.as_tensor(A[tt, 12], dtype=torch.float32, device=dev)
        dk = torch.as_tensor(duck[tt], device=dev)
        out = lidar.render(o, yaw, pitch, dk)
        depth = (out[..., 0] if out.dim() == 4 else out).float().cpu().numpy()
        labels = []
        sl_ids, sl_vals = [], []
        for t in tt:
            while k + 1 < len(seq) and first_t[k + 1] <= t:
                k += 1
            cur = seq[k] if k >= 0 else None
            if a.schedule == "touch":
                nx = seq[k + 1] if k + 1 < len(seq) else FIN
                af = seq[k + 2] if k + 2 < len(seq) else (FIN if k + 1 < len(seq) else NONE)
                ids_ = [nx, af, NONE]
                vals_ = [1.0, -1.0, 0.0]
            else:
                v = takeoff_fade_values(leave_all, float(t), fade_t)
                top = [int(i) for i in np.argsort(-v)[:3] if v[i] > 1e-6]
                top.sort()
                ids_ = [tgt_seq[i] for i in top] + [NONE] * (3 - len(top))
                vals_ = [float(v[i]) for i in top] + [0.0] * (3 - len(top))
                lit = [(tgt_seq[i], float(v[i])) for i in top]
                nx = max(lit, key=lambda x: x[1])[0] if lit else NONE
                af = NONE
            sl_ids.append([(remap(s) if s >= 0 else s) for s in ids_])
            sl_vals.append(vals_)
            labels.append((cur, ids_, vals_))
        tm.set_slots(len(tt), np.asarray(sl_ids), np.asarray(sl_vals, np.float32))
        ch = tm.render(lidar, o, yaw, pitch, dk).cpu().numpy()
        dmax = float(np.percentile(depth, 99.5)) or 1.0
        dn = np.clip(depth / dmax, 0.0, 1.0)
        for j in range(len(tt)):
            if prev_ch is not None:
                dch = np.abs(ch[j] - prev_ch)
                ddp = np.abs(dn[j] - prev_dp)
                jumps.append((float(dch.mean()), float((dch > 0.25).mean()), float(ddp.mean()),
                              float((ddp > 0.25).mean()), float(tt[j])))
            prev_ch = ch[j]
            prev_dp = dn[j]
        for j, t in enumerate(tt):
            fr = np.full((FH, FW), 20, np.uint8)
            dimg = np.clip(depth[j] / dmax, 0.0, 1.0)
            dimg = (dimg * 235 + 20).astype(np.uint8)
            if a.schedule == "touch":
                cimg = np.full(ch[j].shape, 128, np.uint8)
                cimg[ch[j] > 0.5] = 255
                cimg[ch[j] < -0.5] = 0
            else:                              # 0 black, 0.5 grey, 1 white: brightness = order
                cimg = (np.clip(ch[j], 0.0, 1.0) * 235 + 20).astype(np.uint8)
            fr[34:34 + H, 10:10 + W] = cv2.resize(dimg, (W, H), interpolation=cv2.INTER_NEAREST)
            fr[34:34 + H, 20 + W:20 + 2 * W] = cv2.resize(cimg, (W, H),
                                                         interpolation=cv2.INTER_NEAREST)
            cv2.putText(fr, "DEPTH (dark = near)", (10, 24), font, 0.6, 230, 1, cv2.LINE_AA)
            cv2.putText(fr, ("TARGETS: WHITE = next ramp, BLACK = the one after"
                             if a.schedule == "touch" else
                             "TARGETS: WHITE = next / riding, GREY = the one after"),
                        (20 + W, 24), font, 0.6, 230, 1, cv2.LINE_AA)
            cur, ids_, vals_ = labels[j]
            sp = float(np.linalg.norm(A[t, 4:7]))

            def nm(s):
                if s is None:
                    return "-"
                if s == FIN:
                    return "FINISH"
                if s == NONE:
                    return "none"
                return ("R" if int(mcat[s]) == 1 else "F") + str(s)
            line1 = (f"t {t / 100:6.2f} s    speed {sp:5,.0f} u/s    last ramp touched: "
                     f"{nm(cur)}")
            line2 = ("targets: " + "   ".join(f"{nm(s)} {v:+.2f}" for s, v in zip(ids_, vals_)
                                              if s != NONE and v != 0.0)
                     + f"    ({a.unit}, {a.schedule} schedule)")
            cv2.putText(fr, line1, (10, 34 + H + 28), font, 0.62, 235, 1, cv2.LINE_AA)
            cv2.putText(fr, line2, (10, 34 + H + 56), font, 0.62, 235, 1, cv2.LINE_AA)
            ff.stdin.write(fr.tobytes())
    ff.stdin.close()
    ff.wait()
    J = np.asarray(jumps)
    np.savez(str(a.out) + ".jumps.npz", jumps=J, leave=np.asarray(leave), touch=np.asarray(first_t))
    print(f"target_pov: {len(ticks)} frames -> {a.out}; frame-to-frame change of the target "
          f"channel: mean |dv| {J[:, 0].mean():.4f} (max {J[:, 0].max():.3f}), pixels changing > "
          f"0.25: mean {100 * J[:, 1].mean():.2f}% (99th pct {100 * np.percentile(J[:, 1], 99):.1f}%, "
          f"max {100 * J[:, 1].max():.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
