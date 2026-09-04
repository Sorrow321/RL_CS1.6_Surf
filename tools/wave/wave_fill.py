"""wave_fill.py - keep racing small batches until every arm has a box.

    python wave_fill.py <arms.json> <out_dir> [--gpus RTX_5090,RTX_4090] [--rounds 4]

arms.json: {"base": "...", "arms": [{"name":..., "flags":...}, ...]}
Each round: the arms not yet launched (an OK entry in any <out_dir>/*/fleet.json)
are written to a temporary wave file for the next GPU type in the rotation
(race_extra 1) and wave_launch.py runs on it. Stops when nothing remains or
the rounds are exhausted. Writes <out_dir>/fleet_fill.json with every OK box.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SP = Path(__file__).resolve().parent


def launched(out_dir):
    ok = {}
    for f in Path(out_dir).glob("*/fleet.json"):
        try:
            for b in json.loads(f.read_text(encoding="ascii", errors="replace")):
                if b.get("ok"):
                    ok[b["run"]] = b
        except Exception:  # noqa: BLE001
            pass
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms")
    ap.add_argument("out_dir")
    ap.add_argument("--gpus", default="RTX_5090,RTX_4090")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--hours", type=float, default=4.0)
    a = ap.parse_args()
    spec = json.loads(Path(a.arms).read_text(encoding="utf-8"))
    gpus = a.gpus.split(",")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for r in range(a.rounds):
        done = launched(out)
        remaining = [x for x in spec["arms"] if x["name"] not in done]
        if not remaining:
            print("all arms launched", flush=True)
            break
        gpu = gpus[r % len(gpus)]
        wave = {"label": f"fill{r}-{gpu}", "gpu": gpu, "race_extra": 1,
                "base": spec["base"], "arms": remaining}
        wf = out / f"fill{r}.json"
        wf.write_text(json.dumps(wave, indent=1), encoding="ascii")
        print(f"round {r}: {gpu} for {[x['name'] for x in remaining]}", flush=True)
        subprocess.run([sys.executable, str(SP / "wave_launch.py"), str(wf), str(out / f"fill{r}"),
                        "--branch", "baseline", "--hours", str(a.hours)], cwd=r"C:\RL_Surf_base")
        time.sleep(5)
    ok = launched(out)
    fleet = [{"name": b["run"], "host": b["host"], "port": b["port"], "instance": b["instance"],
              "run": b["run"]} for b in ok.values()]
    (out / "fleet_fill.json").write_text(json.dumps(fleet, indent=1), encoding="ascii")
    print("launched:", [b["run"] for b in fleet], flush=True)
    print("missing:", [x["name"] for x in spec["arms"] if x["name"] not in ok], flush=True)


if __name__ == "__main__":
    main()
