"""wave_harvest.py - harvest every box of a wave, then destroy it.

    python wave_harvest.py <fleet.json> [--keep] [--only run1,run2]

Per box (parallel threads):
  1. ON THE BOX: tools/eval_honesty.py --order-only 16 over EVERY eval file
     -> runs/<run>/corridor.csv (step, mean, max, finishes, past_wall), so
     the matched-step reads need no trajectory transfer.
  2. tools/harvest_box.sh <port> <host> <run>  (progress.csv, run.json,
     newest periodic ckpt, last two trajs) into C:\RL_Surf_base\runs\research\<run>
     + corridor.csv + goals.csv + the launch log tail.
  3. md5 of the ckpt verified against the box.
  4. unless --keep: tools/fleet_watchdog.py release <instance> (destroys with
     -y and verifies).
Prints one line per box and writes <fleet dir>/harvest_report.txt.
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(r"C:\RL_Surf_base")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

CORRIDOR = r"""cd /root/RL_Surf && R={run} && out=runs/$R/corridor.csv && echo "step,mean,max,finishes,n,past_wall" > $out && \
for f in $(ls runs/$R/traj_[0-9]*[0-9].jsonl 2>/dev/null | sort); do \
  s=$(basename $f .jsonl | cut -c6-); \
  l=$(PYTHONIOENCODING=utf-8 timeout 300 python3 tools/eval_honesty.py --route maps/surf_src_cannonball.route.npz --order-only 16 $f 2>/dev/null | tr '\n' ' '); \
  m=$(echo "$l" | sed -n 's/.*ORDER-ONLY *mean \([0-9,]*\)u *max \([0-9,]*\)u.*/\1 \2/p' | tr -d ','); \
  fin=$(echo "$l" | sed -n 's/.*finishes \([0-9]*\)\/\([0-9]*\).*/\1 \2/p' | head -1); \
  pw=$(echo "$l" | sed -n 's/.*past 205,440u \([0-9]*\)\/[0-9]*.*/\1/p' | head -1); \
  echo "$s,$(echo $m | cut -d' ' -f1),$(echo $m | cut -d' ' -f2),$(echo $fin | cut -d' ' -f1),$(echo $fin | cut -d' ' -f2),$pw" >> $out; \
done; wc -l < $out; cp runs/$R/goals.csv /root/${{R}}_goals.csv 2>/dev/null; grep -v '^step' runs/${{R}}_launch.txt | tail -20 > /root/${{R}}_tail.txt; md5sum $(ls -t runs/$R/ckpt_[0-9]*.pt 2>/dev/null | head -1) 2>/dev/null | cut -d' ' -f1"""


def ssh(host, port, remote, timeout=1800):
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
           "-p", str(port), f"root@{host}", remote]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV)
    return r.returncode, "\n".join(l for l in r.stdout.splitlines() if "Welcome to vast" not in l and "Have fun" not in l), r.stderr


def harvest(box, keep, report):
    host, port, run, iid = box["host"], box["port"], box["run"], box["instance"]
    line = f"{run:12s} {host}:{port} {iid} "
    try:
        rc, out, err = ssh(host, port, CORRIDOR.format(run=run), timeout=2400)
        lines = out.strip().splitlines()
        n_rows, box_md5 = (lines[0].strip() if lines else "?"), (lines[-1].strip() if len(lines) > 1 else "")
        dest = REPO / "runs" / "research" / run
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["bash", "tools/harvest_box.sh", str(port), host, run], capture_output=True, text=True,
                           timeout=3600, env=dict(ENV, LOCAL_REPO="/c/RL_Surf_base"), cwd=str(REPO))
        ok_h = "done." in r.stdout
        for f, d in ((f"/root/RL_Surf/runs/{run}/corridor.csv", "corridor.csv"), (f"/root/{run}_goals.csv", "goals.csv"),
                     (f"/root/{run}_tail.txt", "launch_tail.txt")):
            subprocess.run(["scp", "-q", "-P", str(port), f"root@{host}:{f}", str(dest / d)], capture_output=True, timeout=600, env=ENV)
        cks = sorted(dest.glob("ckpt_[0-9]*.pt"), key=lambda p: p.stat().st_mtime)
        local_md5 = ""
        if cks:
            import hashlib
            local_md5 = hashlib.md5(cks[-1].read_bytes()).hexdigest()
        md5_ok = bool(box_md5) and local_md5 == box_md5
        line += f"corridor rows {n_rows} harvest {'ok' if ok_h else 'FAILED'} ckpt {'md5 OK' if md5_ok else 'MD5 MISMATCH/none'} "
        if not keep and ok_h and (md5_ok or not cks):
            rel = subprocess.run(["python", "tools/fleet_watchdog.py", "release", str(iid)], capture_output=True, text=True,
                                 timeout=300, env=ENV, cwd=str(REPO))
            line += "released: " + (rel.stdout.strip().splitlines()[-1] if rel.stdout.strip() else rel.stderr[-120:])
        elif not keep:
            line += "NOT released (harvest incomplete)"
    except Exception as e:  # noqa: BLE001
        line += f"ERROR {e}"
    print(line, flush=True)
    report.append(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fleet")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    fleet = json.loads(Path(a.fleet).read_text(encoding="utf-8"))
    only = set(x for x in a.only.split(",") if x)
    boxes = [b for b in fleet if not only or b["run"] in only]
    report = []
    threads = [threading.Thread(target=harvest, args=(b, a.keep, report)) for b in boxes]
    for t in threads:
        t.start()
        time.sleep(2)
    for t in threads:
        t.join()
    Path(a.fleet).with_name("harvest_report.txt").write_text("\n".join(report) + "\n", encoding="ascii", errors="replace")


if __name__ == "__main__":
    main()
