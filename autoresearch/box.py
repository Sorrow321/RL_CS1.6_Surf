"""box.py - the ONLY way an autoresearch agent rents, uses and gives back a GPU box.

Every limit comes from autoresearch/config.json and is enforced HERE, before money is spent:

* ONE box per agent. The agent's box id lives in $AR_HOME/boxes/<agent>.json and in the fleet
  watchdog's registry (label = agent). A second `rent` while one is live is refused.
* At most config.max_boxes live boxes on the account.
* Consumer GPUs only, in the user's order: 3090 / 3090 Ti / 4080 / 5080 first (cheapest of
  those), then 4090, then 5090. Single GPU. Hard $/h cap per card (config.max_dph). No
  datacenter cards, ever.
* The budget governor: credit above config.budget_reserve_usd, and the live burn rate plus
  this box no faster than (credit - reserve) / hours left until config.budget_end_hours.
* Readiness: `running` within config.ready_running_s (the image pull), ssh within
  config.ready_ssh_s after that. Otherwise the host is blocklisted (tools/bad_hosts.json),
  the box destroyed, and the next offer tried (CLAUDE.md section 1).
* Every box is registered with tools/fleet_watchdog.py (deadline = --hours) the moment it
  exists, so the daemon destroys it if the agent forgets; `extend` pushes the deadline.
* The blocklist is the branch's tools/bad_hosts.json plus the fleet's $AR_HOME/bad_hosts.json
  (the orchestrator folds every branch's list into it), and config.cpu_exclude drops host CPUs
  that starve the physics step whatever clock vast reports for them.

    python autoresearch/box.py budget
    python autoresearch/box.py offers
    python autoresearch/box.py rent    --agent r1b2 --hours 2
    python autoresearch/box.py deploy  --agent r1b2 --branch ar/r1b2
    python autoresearch/box.py ssh     --agent r1b2 -- "nvidia-smi"
    python autoresearch/box.py launch  --agent r1b2 --run r1b2_x1 --steps 4e8 -- --extra-flag 1
    python autoresearch/box.py launch  ... --self-states "<how the file came from the policy's
                                       own states>" -- --spawn-states runs/x.npy
    python autoresearch/box.py pull    --agent r1b2 --run r1b2_x1 [--ckpt]
    python autoresearch/box.py extend  --agent r1b2 --hours 1
    python autoresearch/box.py release --agent r1b2
    python autoresearch/box.py status  [--agent r1b2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CFG = json.loads((REPO / "autoresearch" / "config.json").read_text(encoding="utf-8"))
AR_HOME = Path(os.environ.get("AR_HOME", str(Path.home() / "ar")))
BOXES = AR_HOME / "boxes"
FLEET_DIR = Path(os.environ.get("RLSURF_FLEET_DIR", str(AR_HOME / "fleet")))
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    try:
        AR_HOME.mkdir(parents=True, exist_ok=True)
        with open(AR_HOME / "box.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def die(msg: str, code: int = 2):
    log("REFUSED: " + msg)
    sys.exit(code)


# ------------------------------------------------------------------ vast ------
# the CLI lives in ~/.local/bin on the agent host, which only LOGIN shells put on PATH
VASTAI = shutil.which("vastai") or str(Path.home() / ".local" / "bin" / "vastai")


def vast(*args, raw=True, timeout=120, check=True):
    cmd = [VASTAI, *args] + (["--raw"] if raw else [])
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(args)} failed: {r.stderr.strip()[-400:]}")
    if not raw:
        return r.stdout
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"vastai {' '.join(args)}: not JSON: {r.stdout[:300]}")


def live_instances() -> list:
    out = vast("show", "instances")
    return out if isinstance(out, list) else out.get("instances", [])


def credit() -> float:
    u = vast("show", "user")
    return float(u.get("credit") or 0.0)


def gpu_cap(name: str):
    return CFG["max_dph"].get(name)


def gpu_tier(name: str):
    for t, names in enumerate(CFG["gpu_tiers"]):
        if name in names:
            return t
    return None


def norm_gpu(name: str) -> str:
    """'RTX 3090' (instance listings) -> 'RTX_3090' (offer queries)."""
    return (name or "").strip().replace(" ", "_")


# -------------------------------------------------------------- the lock ------
@contextmanager
def fleet_lock():
    """One renter at a time across every agent on this machine: two agents passing the
    budget and box-count checks at the same moment would otherwise both rent."""
    AR_HOME.mkdir(parents=True, exist_ok=True)
    fh = open(AR_HOME / "box.lock", "w")
    try:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except ImportError:          # Windows: tests only
            pass
        yield
    finally:
        fh.close()


# ------------------------------------------------------------ the records -----
def rec_path(agent: str) -> Path:
    return BOXES / f"{agent}.json"


def load_rec(agent: str):
    p = rec_path(agent)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_rec(agent: str, rec: dict) -> None:
    BOXES.mkdir(parents=True, exist_ok=True)
    rec_path(agent).write_text(json.dumps(rec, indent=1), encoding="utf-8")


def archive_rec(agent: str, rec: dict, why: str) -> None:
    rec = dict(rec, ended=time.time(), end_reason=why)
    with open(BOXES / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    rec_path(agent).unlink(missing_ok=True)


def watchdog(*args) -> subprocess.CompletedProcess:
    env = dict(os.environ, RLSURF_FLEET_DIR=str(FLEET_DIR), PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(REPO / "tools" / "fleet_watchdog.py"), *args],
                          capture_output=True, text=True, env=env, timeout=180)


# ------------------------------------------------------------- the budget -----
def budget_state(live=None) -> dict:
    live = live_instances() if live is None else live
    c = credit()
    burn = sum(float(i.get("dph_total") or 0.0) for i in live)
    start = float(os.environ.get("AR_START") or 0.0) or _ar_start()
    hours_left = max(1.0, CFG["budget_end_hours"] - (time.time() - start) / 3600.0)
    allowed = max(0.0, c - CFG["budget_reserve_usd"]) / hours_left
    return {"credit": round(c, 2), "burn_per_h": round(burn, 3), "boxes": len(live),
            "hours_left": round(hours_left, 1), "allowed_per_h": round(allowed, 3),
            "reserve": CFG["budget_reserve_usd"]}


def _ar_start() -> float:
    p = AR_HOME / "state.json"
    if p.exists():
        try:
            return float(json.loads(p.read_text(encoding="utf-8")).get("started") or time.time())
        except (ValueError, OSError):
            pass
    return time.time()


# ------------------------------------------------------------- the offers -----
def merged_blocklist(sources, lists=("blocked", "known_good")) -> dict:
    """The union of several blocklists (paths, or already-parsed dicts). The first readable
    source is kept exactly; a later one adds only the entries it does not already hold
    verbatim, so the same machine blocked twice for two reasons stays two entries (the
    history is the point of the file) while re-merging the same lists adds nothing."""
    out, seen = None, {k: set() for k in lists}
    for src in sources:
        if isinstance(src, dict):
            data = src
        else:
            try:
                data = json.loads(Path(src).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
        if out is None:
            out = json.loads(json.dumps(data))
            for k in lists:
                seen[k].update(json.dumps(e, sort_keys=True) for e in out.get(k, []))
            continue
        for k in lists:
            for e in data.get(k, []):
                key = json.dumps(e, sort_keys=True)
                if key not in seen[k]:
                    seen[k].add(key)
                    out.setdefault(k, []).append(e)
    if out is None:
        return {"blocked": []}
    out.setdefault("blocked", [])
    return out


def sync_shared_blocklist(paths=()) -> int:
    """Fold `paths` into $AR_HOME/bad_hosts.json, the list every agent's box.py reads on top of
    its own branch's: a machine one agent proved bad is skipped by all of them (two agents
    rented the same power-capped 3090 Ti on 2026-10-20). Returns the entry count."""
    shared = AR_HOME / "bad_hosts.json"
    data = merged_blocklist([shared, *paths])
    text = json.dumps(data, indent=2) + "\n"
    old = shared.read_text(encoding="utf-8") if shared.exists() else None
    if text != old:
        # a writer-unique temp name, then an atomic rename: a lost race only delays an entry to
        # the orchestrator's next pass, which re-reads every branch's list
        tmp = shared.with_name(f".bad_hosts.{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(shared)
    return len(data["blocked"])


def blocked_keys():
    # the branch's own list must parse; the fleet's shared one is best effort
    own = json.loads((REPO / "tools" / "bad_hosts.json").read_text(encoding="utf-8"))
    data = merged_blocklist([own, AR_HOME / "bad_hosts.json"])
    mids, hids, ips = set(), set(), set()
    for e in data.get("blocked", []):
        if e.get("machine_id"):
            mids.add(e["machine_id"])
        if e.get("host_id"):
            hids.add(e["host_id"])
        if e.get("ip"):
            ips.add(e["ip"])
    return mids, hids, ips


def offers(exclude_machines=()) -> list:
    """Every acceptable offer, best first: (tier, $/h). Nothing outside the whitelist, the
    caps, the blocklist or config.cpu_exclude is ever returned."""
    mids, hids, ips = blocked_keys()
    cpu_bad = [re.compile(p) for p in CFG.get("cpu_exclude", [])]
    out = []
    for tier, names in enumerate(CFG["gpu_tiers"]):
        for name in names:
            cap = gpu_cap(name)
            if cap is None:
                continue
            try:
                got = vast("search", "offers", f"gpu_name={name} {CFG['offer_query']}",
                           "-o", "dph+")
            except RuntimeError as e:
                log(f"offer search {name} failed: {e}")
                continue
            for o in got:
                if int(o.get("num_gpus") or 0) != 1:
                    continue
                if norm_gpu(o.get("gpu_name")) != name:
                    continue
                dph = float(o.get("dph_total") or 99.0)
                if dph > cap:
                    continue
                if (o.get("machine_id") in mids or o.get("host_id") in hids
                        or o.get("public_ipaddr") in ips
                        or o.get("machine_id") in exclude_machines):
                    continue
                cpu = (o.get("cpu_name") or "").strip()
                if any(p.search(cpu) for p in cpu_bad):
                    continue
                out.append({"tier": tier, "gpu": name, "dph": dph, "id": o["id"],
                            "machine_id": o.get("machine_id"), "host_id": o.get("host_id"),
                            "cores": o.get("cpu_cores_effective"), "ram_gb": round((o.get("cpu_ram") or 0) / 1024),
                            "inet_down": o.get("inet_down"), "where": o.get("geolocation", ""),
                            "cpu": cpu})
    out.sort(key=lambda o: (o["tier"], o["dph"]))
    return out


# --------------------------------------------------------------- the ssh ------
def ssh_target(rec: dict):
    return ["-p", str(rec["port"]), f"root@{rec['host']}"]


def ssh(rec: dict, cmd: str, timeout=600, check=False) -> subprocess.CompletedProcess:
    # stdin=DEVNULL: an ssh that inherits stdin swallows whatever the caller's shell had queued
    # (a `bash -s` heredoc lost every command after the first box.py call that way)
    r = subprocess.run(["ssh", *SSH_OPTS, *ssh_target(rec), cmd], capture_output=True,
                       stdin=subprocess.DEVNULL, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {r.stderr.strip()[-400:]}")
    return r


def ssh_endpoint(iid: int):
    """(host, port) from the instance record: direct ports first, the proxy otherwise."""
    inst = vast("show", "instance", str(iid))
    ports = (inst.get("ports") or {}).get("22/tcp") or []
    if inst.get("public_ipaddr") and ports:
        return inst["public_ipaddr"].strip(), int(ports[0]["HostPort"])
    if inst.get("ssh_host") and inst.get("ssh_port"):
        return inst["ssh_host"], int(inst["ssh_port"])
    return None, None


# --------------------------------------------------------------- commands -----
def cmd_budget(a):
    print(json.dumps(budget_state(), indent=1))


def cmd_offers(a):
    for o in offers()[:a.n]:
        print(f"tier {o['tier']}  {o['gpu']:12s} {o['dph']:.3f} $/h  offer {o['id']}  "
              f"machine {o['machine_id']}  {o['cores']} thr {o['cpu']}  {o['ram_gb']}G  "
              f"{o['inet_down']:.0f} Mb/s  {o['where']}")


def destroy(iid: int, why: str) -> bool:
    """Destroy and VERIFY (CLAUDE.md: without -y it silently aborts; re-issue until gone)."""
    for _ in range(6):
        vast("destroy", "instance", str(iid), "-y", raw=False, check=False)
        time.sleep(5)
        if all(int(i["id"]) != int(iid) for i in live_instances()):
            log(f"destroyed {iid} ({why})")
            return True
    log(f"!! {iid} STILL LIVE after 6 destroy attempts ({why})")
    return False


def cmd_rent(a):
    agent = a.agent
    with fleet_lock():
        rec = load_rec(agent)
        live = live_instances()
        live_ids = {int(i["id"]) for i in live}
        if rec and int(rec["id"]) in live_ids:
            die(f"{agent} already has box {rec['id']} ({rec['gpu']}, {rec['dph']} $/h). "
                "One box per agent: use it, or release it first.")
        if rec:
            archive_rec(agent, rec, "gone before rent")
        mine = [i for i in live if (i.get("label") or "") == agent]
        if mine:
            die(f"{agent}: instance(s) {[i['id'] for i in mine]} carry this label but are not "
                "in the record. Run `release --agent` first.")
        if len(live) >= CFG["max_boxes"]:
            die(f"{len(live)} boxes live, the cap is {CFG['max_boxes']}")
        b = budget_state(live)
        if b["credit"] <= b["reserve"]:
            die(f"credit ${b['credit']} is at the reserve ${b['reserve']}: no new boxes")
        tried = set()
        for attempt in range(3):
            cands = offers(exclude_machines=tried)
            cands = [o for o in cands if b["burn_per_h"] + o["dph"] <= b["allowed_per_h"] * 1.10]
            if not cands:
                die(f"no offer passes (whitelist, caps, blocklist) within the budget pace: "
                    f"burn {b['burn_per_h']} $/h, allowed {b['allowed_per_h']} $/h. Wait for a "
                    "box to be released or for prices to drop; do analysis meanwhile.", code=3)
            o = cands[0]
            tried.add(o["machine_id"])
            if a.dry_run:
                print(json.dumps({"would_rent": o, "budget": b}, indent=1))
                return
            log(f"{agent}: renting offer {o['id']} {o['gpu']} {o['dph']:.3f} $/h "
                f"(machine {o['machine_id']}, {o['where']})")
            try:
                res = vast("create", "instance", str(o["id"]), "--image", CFG["image"],
                           "--disk", str(CFG["disk_gb"]), "--ssh", "--direct",
                           "--label", agent)
            except RuntimeError as e:
                log(f"{agent}: create failed: {e}")
                continue
            iid = int(res.get("new_contract") or 0)
            if not iid:
                log(f"{agent}: create returned no contract: {res}")
                continue
            hours = float(a.hours or CFG["box_hours_default"])
            r = watchdog("register", str(iid), "--minutes", f"{hours * 60:.0f}",
                         "--label", agent, "--owner", "autoresearch")
            if r.returncode != 0:
                log(f"!! {agent}: watchdog register failed: {r.stderr[-300:]}; destroying {iid}")
                destroy(iid, "could not register")
                die("the fleet watchdog would not register the box")
            rec = {"agent": agent, "id": iid, "gpu": o["gpu"], "dph": o["dph"],
                   "machine_id": o["machine_id"], "created": time.time(),
                   "deadline_h": hours, "host": None, "port": None}
            save_rec(agent, rec)
            ok, why = wait_ready(rec)
            if ok:
                save_rec(agent, rec)
                log(f"{agent}: box {iid} READY {rec['gpu']} {rec['dph']} $/h  "
                    f"ssh -p {rec['port']} root@{rec['host']}")
                print(json.dumps(rec, indent=1))
                return
            log(f"{agent}: box {iid} not ready ({why}); blocklisting machine {o['machine_id']}")
            subprocess.run([sys.executable, str(REPO / "tools" / "vast_pick.py"), "--block",
                            str(iid), "--reason", "unreliable", "--detail",
                            f"autoresearch {agent}: {why}"], capture_output=True, text=True)
            sync_shared_blocklist([REPO / "tools" / "bad_hosts.json"])
            destroy(iid, why)
            watchdog("release", str(iid), "--no-harvest")
            archive_rec(agent, rec, why)
        die("three offers failed readiness in a row", code=4)


def wait_ready(rec: dict):
    t0 = time.time()
    while time.time() - t0 < CFG["ready_running_s"]:
        try:
            inst = vast("show", "instance", str(rec["id"]))
        except RuntimeError:
            inst = {}
        st = inst.get("actual_status") or ""
        if st == "running":
            break
        time.sleep(10)
    else:
        return False, (f"not running after {CFG['ready_running_s']} s (status "
                       f"{inst.get('actual_status')}, {(inst.get('status_msg') or '')[:120]})")
    t1 = time.time()
    while time.time() - t1 < CFG["ready_ssh_s"]:
        host, port = ssh_endpoint(rec["id"])
        if host:
            rec["host"], rec["port"] = host, port
            r = ssh(rec, "echo ok && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                    timeout=40)
            if r.returncode == 0 and "ok" in r.stdout:
                rec["nvidia"] = r.stdout.strip().splitlines()[-1]
                return True, ""
        time.sleep(8)
    return False, f"running but no ssh within {CFG['ready_ssh_s']} s"


def need_rec(agent: str) -> dict:
    rec = load_rec(agent)
    if not rec:
        die(f"{agent} has no box (rent one first)")
    return rec


def cmd_deploy(a):
    """deploy_box.sh with this agent's branch, from scratch, with sidistic's maps tarball;
    then the cache mtimes are restamped (tar drops sub-second mtimes) and, with --ckpt, a
    checkpoint is pushed (md5-verified on the box) as /root/RL_Surf/runs_ckpt.pt."""
    rec = need_rec(a.agent)
    tar = AR_HOME / "maps" / "sidistic_maps.tar.gz"
    if not tar.exists():
        die(f"no maps tarball at {tar}")
    # the pytorch/pytorch:*-cu128 image already carries torch (SKIP_TORCH=1 keeps it), and the
    # 10-minute test suite is skipped unless asked for: test YOUR change on the box instead
    env = dict(os.environ, BRANCH=a.branch, NO_CKPT="1", MAPS_TAR=str(tar),
               LOCAL_REPO=str(REPO), SKIP_TORCH=os.environ.get("SKIP_TORCH", "1"),
               SKIP_TESTS="0" if a.tests else "1")
    log(f"{a.agent}: deploying {a.branch} to box {rec['id']}")
    r = subprocess.run(["bash", str(REPO / "tools" / "deploy_box.sh"), str(rec["port"]),
                        rec["host"]], env=env, capture_output=True, text=True, timeout=3600,
                       stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-25:])
    print(tail)
    if r.returncode != 0:
        die(f"deploy_box.sh failed (exit {r.returncode}); see the tail above", code=5)
    ssh(rec, "cd /root/RL_Surf && python3 tools/restamp_maps.py", timeout=300, check=True)
    if a.ckpt:
        push_ckpt(rec, Path(a.ckpt))
    rec["branch"] = a.branch
    rec["deployed"] = time.time()
    save_rec(a.agent, rec)
    log(f"{a.agent}: box {rec['id']} deployed on {a.branch}")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def push_ckpt(rec: dict, ck: Path) -> None:
    if not ck.exists():
        die(f"no checkpoint at {ck}")
    want = md5(ck)
    subprocess.run(["scp", "-o", "BatchMode=yes", *SSH_OPTS, "-q", "-P", str(rec["port"]), str(ck),
                    f"root@{rec['host']}:/root/RL_Surf/runs_ckpt.pt"], check=True, timeout=1800,
                   stdin=subprocess.DEVNULL)
    got = ssh(rec, "md5sum /root/RL_Surf/runs_ckpt.pt | cut -d' ' -f1", check=True).stdout.strip()
    if got != want:
        die(f"checkpoint md5 on the box {got} != {want} (TRUNCATED transfer)")
    log(f"checkpoint {ck.name} on box {rec['id']} (md5 {want})")


def cmd_ssh(a):
    rec = need_rec(a.agent)
    cmd = list(a.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        die("box.py ssh --agent X -- <command>")
    r = ssh(rec, " ".join(cmd), timeout=a.timeout)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    sys.exit(r.returncode)


def cmd_launch(a):
    """The incumbent recipe from scratch (or --resume from runs_ckpt.pt) through
    tools/run_arm.sh, which gates the launch on the record check (CLAUDE.md section 4)."""
    rec = need_rec(a.agent)
    if not a.run.startswith(a.agent + "_"):
        die(f"run names start with the agent name: {a.agent}_<tag>")
    extra = list(a.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    recipe = [] if a.no_recipe else list(CFG["incumbent_recipe"])
    env = (f"MAP={shlex.quote(CFG['map'])} BUDGET={shlex.quote(str(int(float(a.steps))))} "
           f"RECORD_EVERY={shlex.quote(a.record_every)} ")
    if a.self_states is not None:
        # run_arm.sh refuses --spawn-states / --demo-file / --bc-file without SELF_STATES=1,
        # the declaration that the file came from the POLICY'S OWN states (CLAUDE.md section
        # 0); the declaration is recorded with the run, so it has to say how
        if len(a.self_states.strip()) < 20:
            die("--self-states takes the file's provenance: how it was built from the "
                "policy's own states (no demo), in a sentence")
        env += "SELF_STATES=1 "
    if a.resume:
        env += "CKPT=runs_ckpt.pt ARM_RESUME=1 "
        # a resume restores every shape-changing flag from the checkpoint and REFUSES a
        # disagreeing one, so none is passed (CLAUDE.md); --envs is not restored and the
        # trainer default (2048) broke torch.compile on a SimBa resume once
        recipe = ["--envs", "1024"]
    else:
        env += "SCRATCH=1 "
    flags = " ".join(shlex.quote(x) for x in recipe + extra)
    cmd = f"cd /root/RL_Surf && {env} bash tools/run_arm.sh {shlex.quote(a.run)} {flags}"
    log(f"{a.agent}: launch on {rec['id']}: {cmd}")
    r = ssh(rec, cmd, timeout=1500)
    print("\n".join((r.stdout + r.stderr).strip().splitlines()[-30:]))
    if r.returncode != 0:
        die(f"launch failed (exit {r.returncode})", code=6)
    run = {"run": a.run, "t": time.time(), "steps": a.steps, "flags": recipe + extra,
           "resume": bool(a.resume)}
    if a.self_states is not None:
        run["self_states"] = a.self_states
    rec.setdefault("runs", []).append(run)
    save_rec(a.agent, rec)


def cmd_pull(a):
    """Harvest a run to $AR_HOME/runs/<agent>/<run>/: run.json, progress.csv, the launch log,
    every greedy eval trajectory, with --ckpt the newest checkpoint and with --snapshot STEP
    the milestone ckpt_<STEP>.pt (repeatable; the round end hands on the snapshot at or
    before the best eval), each md5-verified."""
    rec = need_rec(a.agent)
    dst = AR_HOME / "runs" / a.agent / a.run
    dst.mkdir(parents=True, exist_ok=True)
    src = f"root@{rec['host']}:/root/RL_Surf/runs"
    # one tar stream: run.json, progress.csv, the launch log and every GREEDY eval trajectory
    # (traj_<step>.jsonl; the record gate's *_rec_* and the mirror evals are left behind)
    remote = (f"cd /root/RL_Surf/runs && tar czf - {shlex.quote(a.run + '_launch.txt')} "
              f"$(cd {shlex.quote(a.run)} && ls run.json progress.csv 2>/dev/null | sed "
              f"'s#^#{a.run}/#') $(cd {shlex.quote(a.run)} && ls traj_*.jsonl 2>/dev/null | "
              f"grep -E '^traj_[0-9]+[.]jsonl$' | sed 's#^#{a.run}/#')")
    with open(dst / "_pull.tgz", "wb") as fh:
        r = subprocess.run(["ssh", *SSH_OPTS, *ssh_target(rec), remote], stdout=fh,
                           stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1800)
    if r.returncode != 0:
        log(f"!! {a.agent}: pull of {a.run} failed: {r.stderr.decode(errors='replace')[-300:]}")
    import tarfile
    with tarfile.open(dst / "_pull.tgz") as tf:
        for m in tf.getmembers():
            name = Path(m.name).name
            if m.isfile() and (name in ("run.json", "progress.csv", a.run + "_launch.txt")
                               or (name.startswith("traj_") and name.endswith(".jsonl"))):
                with tf.extractfile(m) as fsrc:
                    data = fsrc.read()            # read whole before the old copy is replaced
                (dst / name).write_bytes(data)
    (dst / "_pull.tgz").unlink(missing_ok=True)
    cks = (["ckpt_final.pt" if a.final else "ckpt_latest.pt"] if a.ckpt else []) + \
        [f"ckpt_{s}.pt" for s in (a.snapshot or [])]
    for ck in cks:
        r = ssh(rec, f"md5sum /root/RL_Surf/runs/{a.run}/{ck} | cut -d' ' -f1")
        want = r.stdout.strip()
        if want:
            # into a .part first: a timed-out or dropped scp leaves a truncated file, and under
            # the real name it replaced the good copy pulled earlier, which the round end then
            # hands on as the next incumbent (r1b2's box ran at 31 KB/s on 2026-10-20)
            part = dst / (ck + ".part")
            part.unlink(missing_ok=True)
            try:
                subprocess.run(["scp", "-o", "BatchMode=yes", *SSH_OPTS, "-q", "-P",
                                str(rec["port"]), f"{src}/{a.run}/{ck}", str(part)],
                               stdin=subprocess.DEVNULL, timeout=1800)
            except subprocess.TimeoutExpired:
                log(f"!! {a.agent}: scp of {a.run}/{ck} timed out after 30 min")
            got = md5(part) if part.exists() else ""
            if got != want:
                part.unlink(missing_ok=True)
                kept = "the copy pulled earlier is kept" if (dst / ck).exists() else "none kept"
                log(f"!! {a.agent}: {ck} md5 {got} != {want} on the box (truncated pull); {kept}")
            else:
                part.replace(dst / ck)
                log(f"{a.agent}: pulled {a.run}/{ck} (md5 {want})")
    n = len(list(dst.glob("traj_*.jsonl")))
    log(f"{a.agent}: pulled {a.run} -> {dst} ({n} eval trajectories)")
    print(dst)


def cmd_extend(a):
    rec = need_rec(a.agent)
    r = watchdog("extend", str(rec["id"]), "--minutes", f"{float(a.hours) * 60:.0f}")
    print(r.stdout.strip() or r.stderr.strip())


def cmd_release(a):
    rec = load_rec(a.agent)
    ids = set()
    if rec:
        ids.add(int(rec["id"]))
    for i in live_instances():
        if (i.get("label") or "") == a.agent:
            ids.add(int(i["id"]))
    if not ids:
        print(f"{a.agent}: no box")
        if rec:
            archive_rec(a.agent, rec, "already gone")
        return
    for iid in ids:
        destroy(iid, f"release by {a.agent}")
        watchdog("release", str(iid), "--no-harvest")
    if rec:
        archive_rec(a.agent, rec, "released")
    print(f"{a.agent}: released {sorted(ids)}")


def cmd_status(a):
    live = {int(i["id"]): i for i in live_instances()}
    agents = [a.agent] if a.agent else sorted(p.stem for p in BOXES.glob("*.json"))
    for ag in agents:
        rec = load_rec(ag)
        if not rec:
            print(f"{ag}: no box")
            continue
        inst = live.get(int(rec["id"]))
        age = (time.time() - rec["created"]) / 3600.0
        print(f"{ag}: box {rec['id']} {rec['gpu']} {rec['dph']} $/h  "
              f"{(inst or {}).get('actual_status', 'GONE')}  age {age:.1f} h  "
              f"ssh -p {rec.get('port')} root@{rec.get('host')}  branch {rec.get('branch', '-')}"
              f"  runs {[r['run'] for r in rec.get('runs', [])]}")
    print(json.dumps(budget_state(list(live.values())), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("budget").set_defaults(fn=cmd_budget)
    p = sub.add_parser("offers")
    p.add_argument("-n", type=int, default=12)
    p.set_defaults(fn=cmd_offers)
    p = sub.add_parser("rent")
    p.add_argument("--agent", required=True)
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_rent)
    p = sub.add_parser("deploy")
    p.add_argument("--agent", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--ckpt", default=None, help="push this checkpoint as runs_ckpt.pt")
    p.add_argument("--tests", action="store_true", help="run the repo test suite on the box")
    p.set_defaults(fn=cmd_deploy)
    p = sub.add_parser("ssh")
    p.add_argument("--agent", required=True)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_ssh)
    p = sub.add_parser("launch")
    p.add_argument("--agent", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--steps", default="4e8")
    p.add_argument("--record-every", default="75e6")
    p.add_argument("--resume", action="store_true",
                   help="ARM_RESUME from runs_ckpt.pt (push it with deploy --ckpt)")
    p.add_argument("--no-recipe", action="store_true",
                   help="do not prepend config.incumbent_recipe")
    p.add_argument("--self-states", default=None, metavar="PROVENANCE",
                   help="SELF_STATES=1 for run_arm.sh: the --spawn-states (or similar) file "
                        "was built from the policy's OWN states; say how (it is recorded)")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_launch)
    p = sub.add_parser("pull")
    p.add_argument("--agent", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--final", action="store_true", help="ckpt_final.pt instead of ckpt_latest.pt")
    p.add_argument("--snapshot", type=int, action="append", metavar="STEP",
                   help="also pull the milestone ckpt_<STEP>.pt (repeatable)")
    p.set_defaults(fn=cmd_pull)
    p = sub.add_parser("extend")
    p.add_argument("--agent", required=True)
    p.add_argument("--hours", required=True)
    p.set_defaults(fn=cmd_extend)
    p = sub.add_parser("release")
    p.add_argument("--agent", required=True)
    p.set_defaults(fn=cmd_release)
    p = sub.add_parser("status")
    p.add_argument("--agent", default=None)
    p.set_defaults(fn=cmd_status)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
