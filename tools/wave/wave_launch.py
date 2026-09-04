"""wave_launch.py - rent, race, deploy and launch one wave of arms on vast.

    python wave_launch.py <wave.json> <out_dir> [--branch baseline] [--hours 4]

wave.json: {"arms": [{"name": "xW1CTL", "flags": "--goals 1 ..."}, ...],
            "gpu": "RTX_3090", "race_extra": 3}

Steps, all logged to <out_dir>/launch.log:
  1. offers: tools/vast_pick.py --gpu <gpu> (blocklist + physical-core
     filtered). Create len(arms) + race_extra instances at once.
  2. register every instance with tools/fleet_watchdog.py (deadline hours+1).
  3. after 70 s: ssh probe; instances not answering are blocklisted
     (--reason network) and released; healthy extras are released WITHOUT
     a blocklist entry (a loser that came up is not defective).
  4. per box, in parallel threads: clone <branch> over HTTP/1.1, build,
     background deps, wait for imports, gpu_health (thread caps), caches +
     bsp mtime pin (md5-verified), vastai key + on-box watchdog, run_arm.sh
     SCRATCH launch of the arm, dashboard.
  5. writes <out_dir>/fleet.json for wave_monitor.py.
Never edit repository files. PYTHONIOENCODING=utf-8 is set for children.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

SP = Path(__file__).resolve().parent
REPO = Path(r"C:\RL_Surf_base")
MAPS = Path(r"C:\RL_Surf\maps")
MAP = "surf_src_cannonball"
BSP_MTIME = 1776021647154187400
IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"
KEY = Path(r"C:\Users\bulti\.config\vastai\vast_api_key")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
LOG = None


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a", encoding="ascii", errors="replace") as fh:
            fh.write(line + "\n")


def run(cmd, timeout=600, check=False):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV, cwd=str(REPO))
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd}: {r.stderr[-400:]}")
    return r


def ssh(host, port, remote, timeout=600):
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ConnectTimeout=20", "-p", str(port), f"root@{host}", remote]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV)
    out = "\n".join(l for l in r.stdout.splitlines() if "Welcome to vast" not in l and "Have fun" not in l)
    return r.returncode, out, r.stderr


def scp(port, host, files, dest):
    cmd = ["scp", "-q", "-P", str(port)] + [str(f) for f in files] + [f"root@{host}:{dest}"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=ENV).returncode


def instances():
    r = run(["vastai", "show", "instances", "--raw"], timeout=120)
    try:
        return json.loads(r.stdout)
    except Exception:  # noqa: BLE001
        return []


def offers(gpu, n):
    r = run(["python", "tools/vast_pick.py", "--gpu", gpu, "-n", "60"], timeout=180)
    ids = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0].isdigit() and parts[1].isdigit():
            ids.append((int(parts[0]), float(parts[2]), line.strip()))
    log(f"{len(ids)} offers pass the filter")
    for _, _, l in ids[:n]:
        log("  " + l)
    return [i for i, _, _ in ids[:n]]


def create(offer):
    r = run(["vastai", "create", "instance", str(offer), "--image", IMAGE, "--disk", "60",
             "--ssh", "--direct", "--raw"], timeout=120)
    try:
        return int(json.loads(r.stdout)["new_contract"])
    except Exception:  # noqa: BLE001
        log(f"create {offer} failed: {r.stdout[-200:]} {r.stderr[-200:]}")
        return None


def deploy_and_launch(box, arm, branch, hours, out_dir):
    host, port, iid, name = box["ssh_host"], box["ssh_port"], box["id"], arm["name"]
    tag = f"[{name} {iid}]"
    try:
        log(f"{tag} clone+build")
        rc, out, err = ssh(host, port,
            "(setsid nohup bash -c 'DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg; "
            "pip install --break-system-packages scipy numpy opencv-python-headless numba pytest vastai' > /root/pip.log 2>&1 < /dev/null &); sleep 1; "
            f"git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/Sorrow321/RL_CS1.6_Surf /root/RL_Surf 2>&1 | tail -1; "
            f"cd /root/RL_Surf && git config http.version HTTP/1.1 && git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*' && "
            f"timeout 240 git fetch --depth 1 origin {branch} 2>&1 | tail -1; git checkout -q -B {branch} FETCH_HEAD && git log --oneline -1 && mkdir -p runs && bash build.sh 2>&1 | tail -1",
            timeout=900)
        log(f"{tag} {out.strip().splitlines()[-2:] if out.strip() else err[-200:]}")
        if "build OK" not in out:
            raise RuntimeError("build failed")
        for _ in range(60):
            rc, out, _ = ssh(host, port, "python3 -c 'import torch,triton,scipy,numba,cv2,vastai' 2>/dev/null && echo DEPS_OK", timeout=60)
            if "DEPS_OK" in out:
                break
            time.sleep(15)
        else:
            raise RuntimeError("deps never came up")
        rc, out, _ = ssh(host, port, "cd /root/RL_Surf && NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 python3 tools/gpu_health.py --all 2>&1 | tail -3", timeout=600)
        log(f"{tag} gpu_health: {' | '.join(out.strip().splitlines()[-2:])}")
        if "VERDICT: healthy" not in out:
            raise RuntimeError("gpu unhealthy")
        # caches + key + run_arm launch + dashboard + on-box watchdog: the
        # proven bash path (box_finish.sh), POSIX paths, md5 verified on the box
        deadline = int(time.time()) + int(hours * 3600)
        fin = subprocess.run(["bash", str(SP / "box_finish.sh"), str(port), host, str(iid), name, str(deadline)]
                             + arm["flags"].split(), capture_output=True, text=True, timeout=1800, env=ENV)
        fl = [l for l in fin.stdout.splitlines() if "Welcome to vast" not in l and "Have fun" not in l]
        (Path(out_dir) / f"finish_{name}.log").write_text(chr(10).join(fl) + chr(10) + fin.stderr[-2000:], encoding="ascii", errors="replace")
        log(f"{tag} " + " | ".join(l for l in fl if "ALIVE" in l or "watchdog up" in l or "!!" in l or "FAILED" in l)[:300])
        if "== ALIVE" not in fin.stdout:
            raise RuntimeError("trainer not alive after box_finish: " + (fl[-1] if fl else fin.stderr[-200:]))
        if "watchdog up" not in fin.stdout:
            raise RuntimeError("on-box watchdog did not start")
        box["run"] = name
        box["ok"] = True
    except Exception as e:  # noqa: BLE001
        log(f"{tag} FAILED: {e}")
        box["run"] = name
        box["ok"] = False
        box["error"] = str(e)


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("wave")
    ap.add_argument("out_dir")
    ap.add_argument("--branch", default="baseline")
    ap.add_argument("--hours", type=float, default=4.0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    LOG = out / "launch.log"
    wave = json.loads(Path(a.wave).read_text(encoding="utf-8"))
    arms = wave["arms"]
    for arm in arms:                       # "BASE" expands to the wave's shared flag set
        arm["flags"] = arm["flags"].replace("BASE", wave.get("base", "")).strip()
    need = len(arms)
    ids = offers(wave.get("gpu", "RTX_3090"), need + int(wave.get("race_extra", 3)))
    if len(ids) < need:
        log(f"only {len(ids)} offers for {need} arms - launching what fits")
    created = []
    t0 = time.time()
    for o in ids:
        iid = create(o)
        if iid:
            created.append(iid)
            run(["python", "tools/fleet_watchdog.py", "register", str(iid), "--minutes",
                 str(int(a.hours * 60 + 60)), "--label", wave.get("label", "wave")], timeout=60)
    log(f"created {created} at t=0")
    time.sleep(max(0, 72 - (time.time() - t0)))
    inst = {i["id"]: i for i in instances()}
    ready, losers = [], []
    seen_ep, seen_hn = set(), set()
    for iid in created:
        i = inst.get(iid)
        if not i:
            continue
        rc, probe, _ = ssh(i.get("ssh_host"), i.get("ssh_port"), "hostname; nvidia-smi --query-gpu=name --format=csv,noheader", timeout=40) if i.get("ssh_host") else (1, "", "")
        # a listing can hand two instances the SAME ssh endpoint while one is
        # still loading (seen 2026-09-04): the probe then answers for the wrong
        # box. Require a distinct endpoint AND a distinct container hostname.
        ep = (i.get("ssh_host"), i.get("ssh_port"))
        hn = probe.strip().splitlines()[0] if probe.strip() else ""
        dup = ep in seen_ep or (hn != "" and hn in seen_hn)
        if rc == 0 and "RTX" in probe and not dup:
            seen_ep.add(ep)
            seen_hn.add(hn)
            ready.append(i)
            log(f"ready {iid} machine {i.get('machine_id')} {i.get('ssh_host')}:{i.get('ssh_port')}")
        else:
            losers.append(i)
            log(f"not ready {iid} machine {i.get('machine_id')} ({i.get('actual_status')})")
    keep, extra = ready[:need], ready[need:]
    for i in losers:
        run(["python", "tools/vast_pick.py", "--block", str(i["id"]), "--reason", "network",
             "--detail", f"readiness: not answering ssh 72 s after create (wave {wave.get('label','')})"], timeout=60)
        run(["python", "tools/fleet_watchdog.py", "release", str(i["id"])], timeout=120)
    for i in extra:
        run(["python", "tools/fleet_watchdog.py", "release", str(i["id"])], timeout=120)
        log(f"released healthy extra {i['id']} (not blocklisted)")
    threads = []
    for box, arm in zip(keep, arms):
        t = threading.Thread(target=deploy_and_launch, args=(box, arm, a.branch, a.hours, out))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    fleet = [{"name": b["run"], "host": b["ssh_host"], "port": b["ssh_port"], "instance": b["id"],
              "run": b["run"], "ok": b.get("ok", False), "error": b.get("error", ""),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")} for b in keep]
    (out / "fleet.json").write_text(json.dumps(fleet, indent=1), encoding="ascii")
    for f in fleet:
        log(f"{f['run']:12s} {f['host']}:{f['port']} instance {f['instance']} {'OK' if f['ok'] else 'FAILED ' + f['error']}")
    unlaunched = arms[len(keep):]
    if unlaunched:
        log("NOT LAUNCHED (no box): " + ", ".join(x["name"] for x in unlaunched))
    for b in keep:
        if not b.get("ok"):
            log(f"box {b['id']} failed deploy - releasing")
            run(["python", "tools/fleet_watchdog.py", "release", str(b["id"])], timeout=120)


if __name__ == "__main__":
    main()
