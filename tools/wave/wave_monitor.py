"""wave_monitor.py - zero-token fleet monitor for an overnight wave.

    python wave_monitor.py <fleet.json> <status_dir> [--every 1800] [--once]

fleet.json: [{"name": "xW1CTL", "host": "ssh9.vast.ai", "port": 33620,
              "instance": 49633620, "run": "xW1CTL", "started": "..."}]

Every `every` seconds, for each box: ssh once, read the trainer's pid file,
the last console step line, the count of eval files, and the order-only
corridor metric of the newest two evals (tools/eval_honesty.py ON THE BOX,
so nothing crosses the uplink). Writes <status_dir>/status.json (latest
snapshot per run) and appends one line per box per poll to
<status_dir>/history.jsonl, and rewrites <status_dir>/summary.txt - a
compact ASCII table the operator reads in one glance.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
       "-o", "ConnectTimeout=20"]

REMOTE = r"""cd /root/RL_Surf && R={run} && \
P=$(cat runs/$R.pid 2>/dev/null || echo 0); (kill -0 $P 2>/dev/null && echo ALIVE || echo DEAD); \
grep '^step' runs/${{R}}_launch.txt 2>/dev/null | tail -1 | cut -c1-300; \
echo "EVALS $(ls runs/$R/traj_[0-9]*[0-9].jsonl 2>/dev/null | wc -l)"; \
for f in $(ls -t runs/$R/traj_[0-9]*[0-9].jsonl 2>/dev/null | head -2 | sort); do \
  PYTHONIOENCODING=utf-8 timeout 240 python3 tools/eval_honesty.py --route maps/surf_src_cannonball.route.npz --order-only 16 $f 2>/dev/null | grep -E 'ORDER-ONLY|finishes' | tr '\n' ' ' | sed "s#^#EVAL $(basename $f .jsonl | cut -c6-16) #"; echo; done; \
grep -v '^step' runs/${{R}}_launch.txt 2>/dev/null | grep -iE 'Traceback|Error' | tail -2 | cut -c1-200; \
tail -1 /root/box_watchdog.log 2>/dev/null | cut -c1-100; \
df -h / | tail -1 | awk '{{print "DISK", $5}}'"""


def poll(box):
    cmd = SSH + ["-p", str(box["port"]), f"root@{box['host']}",
                 REMOTE.format(run=box["run"])]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=420)
        text = out.stdout
    except subprocess.TimeoutExpired:
        return {"run": box["run"], "ok": False, "error": "ssh timeout"}
    except Exception as e:  # noqa: BLE001
        return {"run": box["run"], "ok": False, "error": str(e)[:200]}
    lines = [l for l in text.splitlines() if l.strip() and "Welcome to vast" not in l and "Have fun" not in l]
    rec = {"run": box["run"], "ok": True, "alive": None, "step": None, "fps": None,
           "rew": None, "len": None, "goals": None, "mind": None, "evals": None,
           "eval_max": [], "eval_mean": [], "finishes": [], "errors": [], "watchdog": "", "disk": ""}
    for l in lines:
        if l in ("ALIVE", "DEAD"):
            rec["alive"] = (l == "ALIVE")
        elif l.startswith("step"):
            m = re.search(r"step\s+([\d,]+)", l)
            rec["step"] = int(m.group(1).replace(",", "")) if m else None
            for key, pat in (("rew", r"rew\s+(-?[\d.]+)"), ("len", r"len\s+(\d+)"),
                             ("fps", r"fps\s+([\d,]+)"), ("mind", r"mind\s+([\d.]+)%"),
                             ("goals", r"goals\s+([\d.]+)%")):
                m = re.search(pat, l)
                rec[key] = float(m.group(1).replace(",", "")) if m else None
        elif l.startswith("EVALS"):
            rec["evals"] = int(l.split()[1])
        elif l.startswith("EVAL "):
            m = re.search(r"mean ([\d,]+)u\s+max ([\d,]+)u", l)
            f = re.search(r"finishes (\d+)/(\d+)", l)
            if m:
                rec["eval_mean"].append(int(m.group(1).replace(",", "")))
                rec["eval_max"].append(int(m.group(2).replace(",", "")))
            if f:
                rec["finishes"].append(f"{f.group(1)}/{f.group(2)}")
        elif l.startswith("DISK"):
            rec["disk"] = l.split()[1]
        elif "watchdog" in l or "DESTROY" in l or "not alive" in l:
            rec["watchdog"] = l[:100]
        elif "Traceback" in l or "Error" in l:
            rec["errors"].append(l[:200])
    return rec


def summarize(status, path):
    rows = ["run          alive    step(M)   fps    rew   len  goals%  mind%  evals   corridor max (last 2)   mean        fin   note",
            "-" * 118]
    for r in status.values():
        if not r.get("ok"):
            rows.append(f"{r['run']:12s} ??       {r.get('error','')}")
            continue
        note = "; ".join(r["errors"][-1:]) if r["errors"] else ""
        if r["watchdog"] and ("DESTROY" in r["watchdog"] or "not alive" in r["watchdog"]):
            note = (r["watchdog"] + " " + note).strip()
        rows.append(
            f"{r['run']:12s} {'yes' if r['alive'] else 'NO ':5s} "
            f"{(r['step'] or 0)/1e6:9.0f} {(r['fps'] or 0)/1e3:5.0f}k "
            f"{(r['rew'] if r['rew'] is not None else float('nan')):6.1f} "
            f"{(r['len'] or 0):5.0f} {(r['goals'] if r['goals'] is not None else float('nan')):6.1f} "
            f"{(r['mind'] if r['mind'] is not None else float('nan')):6.1f} {(r['evals'] or 0):5d}   "
            f"{'/'.join(str(x) for x in r['eval_max']):22s} {'/'.join(str(x) for x in r['eval_mean']):11s} "
            f"{','.join(r['finishes']):5s} {note[:40]}")
    Path(path).write_text("\n".join(rows) + f"\n\nupdated {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="ascii", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fleet")
    ap.add_argument("status_dir")
    ap.add_argument("--every", type=int, default=1800)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    out = Path(a.status_dir)
    out.mkdir(parents=True, exist_ok=True)
    while True:
        fleet = json.loads(Path(a.fleet).read_text(encoding="utf-8"))
        status = {}
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        for box in fleet:
            rec = poll(box)
            rec["t"] = stamp
            status[rec["run"]] = rec
            with open(out / "history.jsonl", "a", encoding="ascii", errors="replace") as fh:
                fh.write(json.dumps(rec) + "\n")
        (out / "status.json").write_text(json.dumps(status, indent=1), encoding="ascii", errors="replace")
        summarize(status, out / "summary.txt")
        if a.once:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
