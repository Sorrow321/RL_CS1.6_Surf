#!/usr/bin/env python3
"""end_bench.py - ANALYSIS ONLY (CLAUDE.md section 0: demo states for MEASUREMENT, never training).

Spawn a checkpoint from the WR demo's states along sidistic's final stretch and report which
piece of the finish it can fly: max field progress (old field, % of d0), finish, max z in the
room, whether it went up through a hole (z > 1,200 at x 2,490-2,700).

    python3 end_bench.py <ckpt> <demo.jsonl> <outdir> <set>[,<set>...] [n_greedy] [n_stoch] [keys_temp] [ep_ticks]
"""
import json, subprocess, sys, os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("RLS_ROOT", "/home/steam/ar/wt/b4"))
sys.path.insert(0, str(ROOT / "python"))
from surfgym.goalfield import GoalField  # noqa: E402
from surfgym.core import STATE_DTYPE  # noqa: E402

z = np.load(ROOT / "maps/surf_src_sidistic.goal_32.npz", allow_pickle=False)
GF = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
D0 = 97768.0

# demo time windows (s) of the final stretch
SETS = {
    "corr": (111.9, 112.3),     # corridor, descending toward the basement
    "base": (112.9, 113.2),     # basement, heading east, low
    "uturn": (113.6, 114.1),    # east-end U-turn
    "west": (114.45, 114.8),    # westward leg on +y, ~1,500 u/s, before the kick ramp
    "kick": (114.95, 115.05),   # just after the kick (vz ~ +1,000)
    "hole": (115.2, 115.3),     # in the +y hole, rising
    "room": (115.45, 115.75),   # in the room, rising, heading east
    "rim": (115.95, 116.25),    # above the rim, over the funnel
}


def demo_rows(path):
    rows = []
    for l in open(path):
        d = json.loads(l)
        if isinstance(d, dict):
            continue
        rows.append(d)
    return np.asarray(rows, np.float64)


def build(ck_path, demo, name, k=4):
    import torch
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    tmpl = np.asarray(ck["respawn_greedy"]["rows"])[0].copy()
    lo, hi = SETS[name]
    sel = demo[(demo[:, 0] / 100.0 >= lo) & (demo[:, 0] / 100.0 <= hi)]
    idx = np.linspace(0, len(sel) - 1, k).round().astype(int)
    out = np.zeros(k, dtype=STATE_DTYPE)
    for j, i in enumerate(idx):
        r = sel[i]
        s = tmpl.copy()
        s["origin"] = r[1:4]
        s["velocity"] = r[4:7]
        s["basevelocity"] = 0
        s["yaw"] = r[7]
        s["pitch"] = r[12] if len(r) > 12 else -10.0
        s["onground"] = -1 if int(r[9]) == 0 else 0
        s["ducked"] = 1 if (int(r[8]) & 4) else 0
        s["induck"] = 0
        s["oldbuttons"] = 0
        s["tick"] = 0
        s["stuck_ticks"] = 0
        out[j] = s
    return out


def episodes(path):
    eps, hdr, cur = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None:
                eps.append((hdr, np.asarray(cur, np.float64), d))
                hdr, cur = None, []
            else:
                hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return eps


def summarise(path):
    res = []
    for hdr, a, tr in episodes(path):
        p = a[:, 1:4]
        d = GF.sample(p).astype(np.float64)
        ok = d < GF.reach_max - 0.5 * GF.cell
        dmin = float(d[ok].min()) if ok.any() else float(d[0])
        end = tr.get("end")
        prog = 100.0 if end == "success" else 100 * (D0 - dmin) / D0
        hole = bool(np.any((p[:, 0] > 2480) & (p[:, 0] < 2700) & (p[:, 2] > 1200)))
        room = p[(p[:, 0] > 2460) & (p[:, 0] < 3420) & (np.abs(p[:, 1]) < 700)]
        zr = int(room[:, 2].max()) if len(room) else None
        res.append(dict(end=end, secs=round(len(a) / 100.0, 2), prog=round(prog, 2),
                        start=[int(x) for x in p[0]], hole=hole, zroom=zr,
                        last=[int(x) for x in p[-1]]))
    return res


def main():
    ck, demo_p, outd, names = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4].split(",")
    ng = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    nst = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    kt = sys.argv[7] if len(sys.argv) > 7 else "1.2"
    ept = sys.argv[8] if len(sys.argv) > 8 else "800"
    outd.mkdir(parents=True, exist_ok=True)
    demo = demo_rows(demo_p)
    for name in names:
        st = build(ck, demo, name, k=max(ng, 2))
        sp = outd / f"{name}.npy"
        np.save(sp, st)
        for mode, n in (("greedy", ng), ("stoch", nst)):
            if n <= 0:
                continue
            jl = outd / f"{name}_{mode}.jsonl"
            cmd = [sys.executable, str(ROOT / "tools/record_ckpt.py"), ck,
                   "--map", str(ROOT / "maps/surf_src_sidistic.bsp"), "--episodes", str(n),
                   "--seed", "0", "--out", str(jl), "--spawn-states", str(sp), "--ep-ticks", ept]
            if mode == "stoch":
                cmd += ["--stochastic", "--exec-keys-temp", kt]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
            if r.returncode != 0:
                print(name, mode, "FAILED", r.stdout[-1500:], r.stderr[-1500:], flush=True)
                continue
            rows = summarise(jl)
            fin = sum(1 for x in rows if x["end"] == "success")
            holes = sum(1 for x in rows if x["hole"])
            best = max(x["prog"] for x in rows)
            zr = [x["zroom"] for x in rows if x["zroom"] is not None]
            print(f"SET {name:6s} {mode:6s} n={len(rows)} finish={fin} hole={holes} best={best:.2f} "
                  f"zroom_max={max(zr) if zr else None}", flush=True)
            for x in rows:
                print("   ", json.dumps(x), flush=True)


if __name__ == "__main__":
    main()
