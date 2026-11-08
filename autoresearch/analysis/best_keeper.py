"""best_keeper.py - harvest helper for one run (unr1b3): keeps the checkpoint of every new best eval.

Reads only: the run's ckpt_latest.pt (copied into a small ring as soon as it changes, named by its
global_step), its finished eval trajectories (scored with autoresearch/score.py). Writes only into
runs/<run>_snaps/. When an eval beats every earlier one, the ring copy saved at that eval's step is
kept as ckpt_<step>.pt. Never touches the trainer; exits when the trainer's pid is gone.
Usage (from /root/RL_Surf): python3 best_keeper.py <run>
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch

RUN = sys.argv[1]
R = os.path.join("runs", RUN)
S = os.path.join("runs", RUN + "_snaps")
os.makedirs(S, exist_ok=True)
PID = int(open(os.path.join("runs", RUN + ".pid")).read().split()[0])
STATE = os.path.join(S, "keeper.json")
st = json.load(open(STATE)) if os.path.exists(STATE) else {"best": 0.0, "best_file": None, "seen": []}
ring = {}
for _p in glob.glob(os.path.join(S, "ring_*.pt")):      # a restarted keeper adopts its old ring
    try:
        ring[int(re.findall(r"ring_(\d+)\.pt", _p)[0])] = _p
    except (IndexError, ValueError):
        pass
last_mtime = 0.0


def log(msg):
    with open(os.path.join(S, "keeper.log"), "a") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " " + msg + "\n")


def alive(p):
    try:
        os.kill(p, 0)
        return True
    except OSError:
        return False


def step_of(name):
    return int(re.findall(r"traj_(\d+)\.jsonl", name)[0])


log(f"start: run {RUN} trainer pid {PID} best so far {st['best']}")
while alive(PID):
    ck = os.path.join(R, "ckpt_latest.pt")
    try:
        m = os.path.getmtime(ck)
    except OSError:
        m = 0.0
    if m and m != last_mtime and time.time() - m > 5:
        tmp = os.path.join(S, "ring_tmp.pt")
        try:
            shutil.copyfile(ck, tmp)
            gs = int(torch.load(tmp, map_location="cpu", weights_only=False)["global_step"])
            dst = os.path.join(S, f"ring_{gs}.pt")
            os.replace(tmp, dst)
            ring[gs] = dst
            last_mtime = m
        except Exception as e:  # a copy caught mid-write: retry on the next pass
            log(f"ring copy failed: {e!r}")
        for g in sorted(ring)[:-8]:
            try:
                os.remove(ring.pop(g))
            except OSError:
                ring.pop(g, None)
    for f in sorted(glob.glob(os.path.join(R, "traj_*.jsonl")), key=lambda p: step_of(os.path.basename(p))):
        name = os.path.basename(f)
        if name in st["seen"]:
            continue
        with open(f) as fh:
            if fh.read().count('"end"') < 9:
                continue
        st["seen"].append(name)
        out = subprocess.run(["python3", "autoresearch/score.py", f], capture_output=True, text=True).stdout.split()
        try:
            sc = float(out[2])
        except Exception:
            log(f"score failed {name}: {out[:6]}")
            continue
        step = step_of(name)
        if sc > st["best"]:
            st["best"], st["best_file"] = sc, name
            # ckpt_latest is saved on a ~1 min clock, not at evals: keep the ring copy AT OR
            # BEFORE the eval, unless the first copy after it is more than twice as near
            # (PROTOCOL's hand-on rule, applied to the run's own ckpt_latest saves)
            b = max([g for g in ring if g <= step], default=None)
            a = min([g for g in ring if g > step], default=None)
            if b is None or (a is not None and 2 * (a - step) < (step - b)):
                cand = a
            else:
                cand = b
            if cand is not None:
                shutil.copyfile(ring[cand], os.path.join(S, f"ckpt_{cand}.pt"))
                log(f"best {sc} {name} -> ckpt_{cand}.pt (eval step {step}, offset {cand - step:+d})")
            else:
                log(f"best {sc} {name} (no ring copy near {step}; ring {sorted(ring)})")
        json.dump(st, open(STATE, "w"))
    time.sleep(20)
log("trainer gone; keeper exits")
