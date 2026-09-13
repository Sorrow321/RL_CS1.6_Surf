"""uf2 ENTRY bench: which sub-skill is missing - the entry into the pit, the
pit surf, or the exit? Spawn the policies from three slices of the record's
dip window (uf2_pitwindow.npy, 548 states uniform over t 2.5-9 s):
  entry  t 2.5-3.5 s  (down the start ramp, before the dive)      idx   0- 84
  pit    t 3.5-6.0 s  (the dive and the pit face)                 idx  84-295
  exit   t 6.0-9.0 s  (the climb out)                             idx 295-548
Score: PASS = d < 29,000 (depth >= 1,589 u, the record at ~9.3 s) alive 3 s
later; pit-box contact; where they die. 8 sampled + 4 greedy, 15 s cap.
"""
import subprocess, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/RL_Surf")
GB = ROOT / "runs/research/gate_bench"
OUT = GB / "entry"
OUT.mkdir(exist_ok=True)
w = np.load(GB / "uf2_pitwindow.npy")
slices = {"entry": (0, 84), "pit": (84, 295), "exit": (295, 548)}
for k, (a, b) in slices.items():
    np.save(OUT / f"uf2_win_{k}.npy", w[a:b])
MAP = ROOT / "maps_pool/surf_unitfarmer2.bsp"
PIT = ["--ramp-box", "-2600", "-3400", "-1000", "-1700", "-1100", "-50"]
ckpts = [a for a in sys.argv[1:]] or ["uf2PITc", "uf2RPIT"]
rows = []
for run in ckpts:
    ck = ROOT / "runs" / run / "ckpt_final.pt"
    if not ck.exists():
        ck = ROOT / "runs" / run / "ckpt_latest.pt"
    for k in slices:
        for mode, eps in (("stoch", 8), ("greedy", 4)):
            name = f"{run}_{k}_{mode}"
            cmd = [sys.executable, str(ROOT / "tools/gate_bench.py"), "run", "--ckpt", str(ck),
                   "--spine", str(OUT / f"uf2_win_{k}.npy"), "--episodes", str(eps), "--ep-ticks", "1500",
                   "--name", name, "--out-dir", str(OUT), "--map", str(MAP), "--goal-cell", "48", "--occ-cell", "32",
                   "--d-pass", "29000", "--pass-hold", "3"] + PIT
            if mode == "stoch":
                cmd.append("--stochastic")
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
            lines = [l for l in (r.stdout + r.stderr).splitlines() if "episodes |" in l or "failers:" in l or "passers:" in l or "Error" in l or "failed" in l]
            print(f"== {name}")
            for l in lines[:4]:
                print("   " + l[:230])
            js = OUT / f"{name}_score.json"
            if js.exists():
                d = json.loads(js.read_text(encoding="utf-8"))
                rows.append((name, d))
print("\nsummary keys of one score json:", list(rows[0][1].keys())[:20] if rows else "none")
