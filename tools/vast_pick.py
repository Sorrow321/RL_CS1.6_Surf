"""vast_pick.py - rent GPUs without re-renting the boxes that burned us.

A vast.ai listing cannot tell you the two things that actually cost us days:
a card whose clocks are pinned by a SW Power Cap (71% of a healthy 5090's
bf16 throughput, unfixable inside the container), and a host whose CPU path
collapses under the env-step + python-reward loop (scratch training is
CPU-bound; one 256-core host ran at 40% of a healthy box). Both are only
visible after deploying, so the only defence is remembering.

  python tools/vast_pick.py                       # ranked, blocklist-filtered
  python tools/vast_pick.py --gpu RTX_4090 -n 20
  python tools/vast_pick.py --block 48248684 --reason gpu_capped \
      --detail "166 TFLOPS vs 234 ref, SW Power Cap"
  python tools/vast_pick.py --good 48248695 --detail "824k steps/s scratch"

Offer ids churn between listings, so entries key on the stable machine_id /
host_id (ip is kept as a fallback for boxes only ever seen by ssh address).
Record the verdict BEFORE destroying an instance - `vastai show instances`
is where the identifiers come from, and they vanish with it.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

LIST = Path(__file__).with_name("bad_hosts.json")


def load():
    return json.loads(LIST.read_text(encoding="utf-8"))


def save(data):
    LIST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def vast(*args):
    out = subprocess.run(["vastai", *args, "--raw"], capture_output=True,
                         text=True)
    if out.returncode != 0:
        sys.exit(f"vastai {' '.join(args)} failed:\n{out.stderr[-500:]}")
    return json.loads(out.stdout)


def blocked_keys(data):
    mids, hids, ips = set(), set(), set()
    for e in data["blocked"]:
        if e.get("machine_id"):
            mids.add(e["machine_id"])
        if e.get("host_id"):
            hids.add(e["host_id"])
        if e.get("ip"):
            ips.add(e["ip"])
    return mids, hids, ips


def record(data, instance_id, bucket, reason, detail):
    # ORDER MATTERS: this looks the box up in `vastai show instances`, so
    # a destroyed instance can no longer be blocklisted - its machine_id
    # and host_id are gone with it. Always block THEN destroy. Losing that
    # order means the same bad host can be rented again tomorrow.
    inst = next((i for i in vast("show", "instances")
                 if int(i["id"]) == int(instance_id)), None)
    if inst is None:
        sys.exit(f"instance {instance_id} not found (already destroyed? "
                 "then add the entry by hand from your notes)")
    entry = {
        "ip": inst.get("public_ipaddr"),
        "machine_id": inst.get("machine_id"),
        "host_id": inst.get("host_id"),
        "date": None,
        "detail": detail or "",
        "run": "",
    }
    if bucket == "blocked":
        entry["reason"] = reason or "unspecified"
    data["in_flight"] = [e for e in data.get("in_flight", [])
                         if int(e.get("instance", -1)) != int(instance_id)]
    data.setdefault(bucket, []).append(entry)
    save(data)
    print(f"recorded machine_id {entry['machine_id']} "
          f"(host {entry['host_id']}, {entry['ip']}) under {bucket}")
    print("NOTE: set the date field by hand - the tool has no clock.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="RTX_5090")
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("--min-cores", type=float, default=12.0,
                    help="effective CPU cores; scratch runs are CPU-bound")
    ap.add_argument("--block", type=int, help="instance id to blocklist")
    ap.add_argument("--good", type=int, help="instance id to mark known-good")
    ap.add_argument("--reason", default=None,
                    choices=[None, "gpu_capped", "cpu_bound", "no_cuda",
                             "unreliable", "network"])
    ap.add_argument("--detail", default="")
    args = ap.parse_args()

    data = load()
    if args.block:
        return record(data, args.block, "blocked", args.reason, args.detail)
    if args.good:
        return record(data, args.good, "known_good", None, args.detail)

    mids, hids, ips = blocked_keys(data)
    q = (f"gpu_name={args.gpu} num_gpus=1 cuda_vers>=12.8 reliability>0.98 "
         f"inet_down>300 rentable=true disk_space>60")
    offers = vast("search", "offers", q, "-o", "dph+")
    rows, skipped = [], 0
    for o in offers:
        if (o.get("machine_id") in mids or o.get("host_id") in hids
                or o.get("public_ipaddr") in ips):
            skipped += 1
            continue
        if (o.get("cpu_cores_effective") or 0) < args.min_cores:
            continue
        rows.append(o)

    print(f"{len(offers)} offers, {skipped} blocklisted, "
          f"{len(rows)} pass (>= {args.min_cores:g} effective cores)\n")
    print(f"{'offer':>10} {'machine':>8} {'$/h':>6} {'rel':>6} {'cores':>6} "
          f"{'RAM':>5} {'net':>6}  location")
    for o in rows[:args.n]:
        print(f"{o['id']:>10} {o.get('machine_id', 0):>8} "
              f"{o['dph_total']:>6.3f} {o.get('reliability2', 0):>6.3f} "
              f"{o.get('cpu_cores_effective', 0):>6.0f} "
              f"{o.get('cpu_ram', 0) / 1024:>4.0f}G {o.get('inet_down', 0):>6.0f}"
              f"  {o.get('geolocation', '')}")
    if rows:
        print(f"\nvastai create instance {rows[0]['id']} \\\n"
              f"    --image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \\\n"
              f"    --disk 60 --ssh --direct")
        print("then: bash tools/deploy_box.sh <port> <host>   "
              "(gpu_health.py runs last and is the real gate)")


if __name__ == "__main__":
    main()
