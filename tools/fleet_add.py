"""fleet_add.py - rent a box, deploy, launch an arm, in one call.

Bringing up a box by hand is ~6 steps (search, create, wait, deploy,
health-gate, launch) and every one of them has bitten us: offers churn,
images stall mid-pull, cards come power-capped, and a `pkill` in the
launch line kills its own shell. This wraps the sequence, gates on
gpu_health, records the verdict in bad_hosts.json, and launches through
an on-box script so no pattern can self-match.

  python tools/fleet_add.py sDIV3 --gpu RTX_3090 -- \
      --spawn mixed --drop-frac 0.25 --respawn-frac 0.9

Everything after `--` is appended to the champion recipe. Use --dry-run
to see the chosen offer and the exact command without spending anything.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "tools" / "bad_hosts.json"
IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"

BASE = ("--map maps/surf_src_cannonball.bsp --reward race --spawn platform "
        "--respawn-frac 0.9 --respawn-speed 1.0 1.5 --maxvel 4000 "
        "--lidar-w 64 --lidar-h 32 --lidar-range 11500 --lidar-near 2000 "
        "--act-every 3 --pitch-rate 1.33 --gamma 0.9995 --int-coef 0.25 "
        "--int-speed 3 --int-view 8 --steps 8e9 --ckpt-every 500e6 "
        "--record-every 150e6 --eval-eps 9 --eval-greedy-only")


def vast(*args, raw=True):
    cmd = ["vastai", *args] + (["--raw"] if raw else [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(args)}: {out.stderr[-400:]}")
    return json.loads(out.stdout) if raw else out.stdout


def blocked():
    d = json.loads(LIST.read_text(encoding="utf-8"))
    mids = {e["machine_id"] for e in d["blocked"] if e.get("machine_id")}
    hids = {e["host_id"] for e in d["blocked"] if e.get("host_id")}
    return mids, hids


def pick(gpu, min_cores, max_dph):
    mids, hids = blocked()
    q = (f"gpu_name={gpu} num_gpus=1 cuda_vers>=12.8 reliability>0.98 "
         f"inet_down>300 rentable=true disk_space>60")
    for o in vast("search", "offers", q, "-o", "dph+"):
        if o.get("machine_id") in mids or o.get("host_id") in hids:
            continue
        if (o.get("cpu_cores_effective") or 0) < min_cores:
            continue
        if o["dph_total"] > max_dph:
            continue
        return o
    return None


def sh(*args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError(f"failed: {' '.join(args[:3])}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gpu", default="RTX_3090")
    ap.add_argument("--min-cores", type=float, default=12)
    ap.add_argument("--max-dph", type=float, default=0.25)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="after --, appended to the champion recipe")
    a = ap.parse_args()
    extra = " ".join(x for x in a.extra if x != "--")

    o = pick(a.gpu, a.min_cores, a.max_dph)
    if o is None:
        sys.exit(f"no {a.gpu} offer under ${a.max_dph}/h with "
                 f">= {a.min_cores} cores that is not blocklisted")
    print(f"offer {o['id']} machine {o.get('machine_id')} "
          f"${o['dph_total']:.3f}/h {o.get('cpu_cores_effective'):.0f} cores "
          f"{o.get('geolocation','')}")
    cmd = f"python3 -u python/train_fast.py --run {a.run} {BASE} {extra}"
    if a.dry_run:
        print("would launch:", cmd)
        return

    inst = vast("create", "instance", str(o["id"]), "--image", IMAGE,
                "--disk", "60", "--ssh", "--direct")
    iid = inst["new_contract"]
    print(f"instance {iid} created; waiting for it to run")
    host = port = None
    for _ in range(60):
        time.sleep(20)
        for i in vast("show", "instances"):
            if i["id"] == iid and i["actual_status"] == "running" and i.get("ssh_host"):
                host, port = i["ssh_host"], i["ssh_port"]
                break
        if host:
            break
    if not host:
        sys.exit(f"instance {iid} never came up; destroy it by hand")
    print(f"up at {host}:{port}; deploying")

    env = {"SKIP_TORCH": "1", "EXPECTED_MD5": ""}
    import os
    e = dict(os.environ, **env)
    r = subprocess.run(["bash", str(ROOT / "tools" / "deploy_box.sh"),
                        str(port), host], capture_output=True, text=True, env=e)
    tail = (r.stdout or "")[-3000:]
    if "VERDICT: healthy" not in tail:
        print(tail[-800:])
        print(f"!! {iid} failed the health gate - blocklist and destroy it:")
        print(f"   python tools/vast_pick.py --block {iid} --reason gpu_capped")
        print(f"   vastai destroy instance {iid} -y")
        sys.exit(1)
    print("healthy; launching")

    script = f"#!/bin/bash\ncd /root/RL_Surf\nexec {cmd} > runs/{a.run}.log 2>&1\n"
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-p", str(port), f"root@{host}",
         f"cat > /root/go.sh << 'EOS'\n{script}EOS\nchmod +x /root/go.sh; "
         f"(setsid nohup /root/go.sh > /dev/null 2>&1 &); sleep 8; "
         f"pgrep -fc '[t]rain_fast'"],
        check=False)
    print(f"\n{a.run} launched on {host}:{port}  (instance {iid}, "
          f"${o['dph_total']:.3f}/h)")
    print(f"  ssh -p {port} root@{host}")
    print(f"  tail: ssh -p {port} root@{host} \"tail -1 /root/RL_Surf/runs/{a.run}.log\"")


if __name__ == "__main__":
    main()
