#!/usr/bin/env python3
"""sid_shaft_bench.py - MEASUREMENT ONLY: from the policy's OWN greedy states at sidistic's window /
shaft (the decision between the tunnel dive and the cavity), how often does the policy - sampled
at the trainer's temperature - enter the cavity, rise off its ramp, come back west through the lower
opening, reach a tunnel and the room, and pass the field-route limit (28.84%)? Map coordinates are
bench metrics (CLAUDE.md 0b), never a reward; the states are the checkpoint's own greedy recording.

    # 1. a greedy recording with full states (CPU is fine)
    python tools/record_ckpt.py CKPT --episodes 9 --seed 1 --ep-ticks 3500 \
        --dump-states OUT/greedy.npz --out OUT/greedy.jsonl
    # 2. cut the shaft windows, run the stochastic bench, summarise
    python autoresearch/analysis/sid_shaft_bench.py CKPT OUT [n_episodes] [exec_temp]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "autoresearch"))
sys.path.insert(0, str(REPO / "python"))
import score  # noqa: E402

START_D = 97768.0


def shaft_entry_ticks(r):
    """Tick index of each greedy episode's first shaft row after 22 s (None if never)."""
    t = r[:, 0] / 100.0
    x, z = r[:, 1], r[:, 3]
    m = (t > 22.0) & (x > 2208) & (x < 2512) & (z > 10700) & (z < 11520)
    i = np.flatnonzero(m)
    return int(i[0]) if len(i) else None


def cut(out: Path, before=1.0, after=0.8, every=5):
    dump = np.load(out / "greedy.npz")
    eps = score.episodes(out / "greedy.jsonl")
    rows = []
    for k, (hdr, r, tr) in enumerate(eps):
        i = shaft_entry_ticks(r)
        key = f"ep_{k:04d}"
        if i is None or key not in dump:
            continue
        st = dump[key]
        lo, hi = max(0, i - int(before * 100)), min(len(st), i + int(after * 100))
        rows.append(st[lo:hi:every])
    if not rows:
        raise SystemExit("no greedy episode reached the shaft: nothing to bench")
    cutp = out / "shaft_states.npy"
    np.save(cutp, np.concatenate(rows))
    print(f"cut {sum(len(x) for x in rows)} states from {len(rows)} greedy shaft passages -> {cutp}")
    return cutp


def flags(r):
    x, y, z, vz = r[:, 1], r[:, 2], r[:, 3], r[:, 6]
    cav = (x > 2576) & (x < 3040) & (z > 9500) & (z < 10760) & (np.abs(y) < 700)
    deep = (x > 1700) & (x < 3040) & (z > 9500) & (z < 10650)      # anywhere under the shaft floor
    out = {"C": bool(cav.any()), "R": bool((deep & (vz > 300)).any())}
    ic = np.flatnonzero(deep)
    back = False
    if len(ic):
        aft = np.arange(len(r)) > ic[0]
        back = bool((aft & (x < 2512) & (x > 2208) & (z > 10770) & (z < 11232)).any())
    out["B"] = back
    out["T"] = bool(((x > 1952) & (x < 2208) & (np.abs(y) > 800) & (z > 10900) & (z < 11232)).any())
    out["M"] = bool(((x > 1168) & (x < 1952) & (z > 10400) & (z < 11300)).any())
    return out


def summarise(path):
    f = score.field()
    res = []
    for hdr, r, tr in score.episodes(path):
        d = f.sample(r[:, 1:4]).astype(np.float64)
        ok = d < f.reach_max - 0.5 * f.cell
        dmin = float(d[ok].min()) if ok.any() else float(d[0])
        fl = flags(r)
        fl["prog"] = round(100.0 * (START_D - dmin) / START_D, 2)
        fl["secs"] = round(len(r) / 100.0, 2)
        fl["end"] = tr.get("end")
        res.append(fl)
    n = len(res)
    tally = {k: sum(1 for e in res if e[k]) for k in "CRBTM"}
    past = sum(1 for e in res if e["prog"] > 28.84)
    best = max(e["prog"] for e in res)
    print(f"{path.name}: {n} episodes | cavity {tally['C']}  rose {tally['R']}  back-west {tally['B']}  "
          f"tunnel {tally['T']}  room {tally['M']}  past 28.84% {past}  best {best:.2f}%  "
          f"median life {np.median([e['secs'] for e in res]):.2f} s")
    return res


def main():
    ckpt, out = sys.argv[1], Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    temp = sys.argv[4] if len(sys.argv) > 4 else "2.0"
    cutp = cut(out)
    bench = out / f"bench_T{temp}.jsonl"
    cmd = [sys.executable, str(REPO / "tools" / "record_ckpt.py"), ckpt, "--spawn-states", str(cutp),
           "--stochastic", "--exec-temp", temp, "--episodes", str(n), "--ep-ticks", "800",
           "--seed", "2", "--out", str(bench)]
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(1)
    res = summarise(bench)
    (out / f"bench_T{temp}.json").write_text(json.dumps(res, indent=0))


if __name__ == "__main__":
    main()
