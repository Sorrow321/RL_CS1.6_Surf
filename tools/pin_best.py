"""pin_best.py RUN - on the box: score each new greedy eval of runs/RUN; when one beats the best
so far, pin a checkpoint AT OR BEFORE it as ckpt_<step>.pt (the round's hand-on takes the newest
snapshot at or before the best eval). ckpt_latest.pt is rewritten every ~60 s, so a rolling copy
(ckpt_roll.pt) of each new ckpt_latest is kept: the pin is ckpt_latest if its step <= the eval's,
else the rolling copy if that one is, else ckpt_latest anyway. Polls every 20 s; nice 19."""
import json, re, shutil, subprocess, sys, time
from pathlib import Path
import torch
run = sys.argv[1]
rd = Path("/root/RL_Surf/runs") / run
seen, best = set(), -1.0
state = rd / "pin_best.json"
if state.exists():
    st = json.loads(state.read_text()); seen, best = set(st["seen"]), st["best"]
roll_step, last_mtime = None, None


def step_of(p):
    return int(torch.load(p, map_location="cpu", weights_only=False)["global_step"])


while True:
    ck = rd / "ckpt_latest.pt"
    try:
        mt = ck.stat().st_mtime
        if mt != last_mtime and time.time() - mt > 5:        # a finished write
            tmp = rd / "ckpt_roll.tmp"
            shutil.copy2(ck, tmp)
            s = step_of(tmp)
            tmp.replace(rd / "ckpt_roll.pt")
            roll_step, last_mtime = s, mt
    except Exception:  # noqa: BLE001
        pass
    for f in sorted(rd.glob("traj_*.jsonl")):
        if not re.fullmatch(r"traj_\d+\.jsonl", f.name) or f.name in seen:
            continue
        if time.time() - f.stat().st_mtime < 20:
            continue
        ev_step = int(f.name[5:-6])
        r = subprocess.run([sys.executable, "/root/RL_Surf/autoresearch/score.py", str(f)],
                           capture_output=True, text=True)
        m = re.search(r"score +([0-9.]+)", r.stdout)
        seen.add(f.name)
        if not m:
            continue
        sc = float(m.group(1))
        line = f"{time.strftime('%H:%M:%S')} {f.name} {sc:.3f}"
        if sc > best:
            best = sc
            try:
                ls = step_of(ck)
                src, s = (ck, ls)
                if ls > ev_step and roll_step is not None and roll_step <= ev_step:
                    src, s = (rd / "ckpt_roll.pt", roll_step)
                dst = rd / f"ckpt_{s}.pt"
                if not dst.exists():
                    shutil.copy2(src, dst)
                line += f"  NEW BEST -> pinned {dst.name} ({'at/before' if s <= ev_step else 'AFTER'} the eval)"
            except Exception as ex:  # noqa: BLE001
                line += f"  NEW BEST, pin failed: {ex}"
        with open(rd / "pin_best.log", "a") as fh:
            fh.write(line + "\n")
        state.write_text(json.dumps({"seen": sorted(seen), "best": best}))
    time.sleep(20)
