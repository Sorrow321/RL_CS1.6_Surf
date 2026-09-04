#!/usr/bin/env python3
"""fleet_watchdog.py - rented boxes are running or deleted, never stale.

CLAUDE.md rule 1. An agent's attention is not a safety mechanism: sessions
end, contexts run out, ssh drops, a subagent dies mid-task. This is the
process that keeps costing-money-while-idle from happening anyway.

It polls `vastai show instances` and destroys, with `-y` (without which the
CLI silently aborts and the box keeps billing) and with verification:

  * anything NOT in the registry, once past a short grace window - an
    instance nobody claimed is an instance nobody is watching;
  * anything past its registered deadline;
  * anything that never came up (not "running" within the readiness window);
  * anything whose owning run has gone quiet past its deadline.

It also HARVESTS, for the same reason it destroys: a box that dies with its
results on it is the same failure as a box nobody destroyed, and both were
being held together by an agent being awake at the right minute. An entry
with a harvest spec is pulled 20 minutes before its deadline (and as soon as
its trainer is seen gone), by the daemon, with no agent involved. An entry
WITHOUT one behaves exactly as it always did.

Usage:

    # start it BEFORE renting anything (detached, survives the session)
    python tools/fleet_watchdog.py --daemon [--harvest-lead 20]

    # the moment `vastai create` returns an id, claim it
    python tools/fleet_watchdog.py register <id> --minutes 90 --label xROUTE

    # once the box is deployed and its ssh endpoint is known, say what to
    # save. Re-registering upserts: the `ready` latch and the deadline rules
    # are unchanged, and the ON-BOX deadline must be >= this one or the box
    # destroys itself before the harvest window opens.
    python tools/fleet_watchdog.py register <id> --minutes 245 --label xROUTE \
        --harvest "12345 ssh5.vast.ai xROUTE" --pid-file runs/xROUTE.pid

    # done with a box: harvest, then destroy, then drop the claim
    python tools/fleet_watchdog.py release <id> [--no-harvest]

    python tools/fleet_watchdog.py harvest <id>  # pull now, destroy nothing
    python tools/fleet_watchdog.py list
    python tools/fleet_watchdog.py reap-all      # emergency: destroy everything

Registry lives in runs/fleet.json so every agent and every session shares
one view of what is rented and who owns it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "runs" / "fleet.json"
LOG = ROOT / "runs" / "fleet_watchdog.log"

POLL_S = 30.0   # tighter, so READY_S=90 is detected at 90-120s
# An instance may exist for this long unclaimed: the window between
# `vastai create` returning and the agent registering the id.
GRACE_S = 240.0
# Not "running" within this long after creation = it never came up. The
# agent-facing rule is 60 s; this is the backstop for an agent that walked
# away, so it is deliberately looser.
#
# It applies ONLY to a box that has never been seen running. This cost a
# whole arm on 2026-08-23: instance 48446220 had been training for 16
# minutes when the vast API reported it "offline" for one poll (the same
# blip showed up locally as an ssh "connection abort" then "connection
# refused"), and this rule destroyed it as "never came up (status offline,
# age 28.1m)" - taking the checkpoint, progress.csv and every trajectory
# with it. Once a box has come up, the DEADLINE is what bounds it; a
# status blip is not evidence that anything is wrong, and destroying on
# one observation is unrecoverable while waiting is merely expensive.
# 420 s destroyed TWO healthy boxes mid-image-pull (rounds 22 and 23), each
# time losing the measurement they were rented for. This is the backstop for
# an ABANDONED box, not the readiness rule - the agent-facing rule is still
# 60 s - so it costs nothing to be generous, and a slow registry on a big
# image genuinely exceeds seven minutes.
READY_S = 90.0   # CLAUDE.md rule 1: ready in 60s or it dies. This
                 # was 1200 - a TWENTY MINUTE grace on a 60-SECOND
                 # rule, so the watchdog could never enforce it and
                 # a box that never opened ssh billed for 11 minutes
                 # while the agent waited. 90s = the rule plus one
                 # poll of slack. The `ready` latch (added after the
                 # watchdog killed a LIVE trainer) still gates this,
                 # so a box that ever reached `running` is never
                 # killed by readiness - only by its deadline.
MAX_BOXES = 12       # user-set 2026-08-23 (4 -> 6 -> 12); plus the local GPU

# ------------------------------------------------------------- harvesting ---
# A rented box takes its disk with it, so a result is only real once it is on
# the workstation. 2026-09-04: six 5090s and an expert box self-destructed on
# schedule with every checkpoint and trajectory still on them, because
# harvesting was a MANUAL step and an API rate limit took every agent out from
# 14:20 to 15:10 - the wave's own harvest window. Same class as the destroy
# button, and the same rule applies: the agent's attention is not a safety
# mechanism, so the daemon does it.
#
# The money rule still wins, in both directions. A pull runs in a background
# thread, so a 15-minute transfer cannot stall the sweep that enforces every
# OTHER box's deadline; and a box whose harvest failed is destroyed at its
# deadline anyway, loudly.
HARVEST_LEAD_S = 20 * 60.0      # open the harvest window this early
HARVEST_TIMEOUT_S = 15 * 60.0   # per box, per attempt
HARVEST_TRIES = 3               # attempts per trigger...
HARVEST_BACKOFF_S = 120.0       # ...at 2 min then 4, so all three fit the lead
PID_POLL_S = 300.0              # early-exit probe: cheap, every 5 minutes
PID_DEAD_NEEDED = 2             # ...and ONE observation is never evidence
SSH_TIMEOUT_S = 45.0

_REG_LOCK = threading.RLock()
_HARVESTING = set()             # instance ids with a pull in flight


def now() -> float:
    return time.time()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{stamp()}  {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def vast(*a, timeout=120):
    """Run the vast CLI. PYTHONIOENCODING is not optional: the user's console
    is cp1251 and the CLI's own output crashes on encode without it."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(["vastai", *a], capture_output=True, text=True,
                          env=env, timeout=timeout,
                          encoding="utf-8", errors="replace")


def instances():
    """-> list of dicts, or None if the API could not be reached.

    None and [] mean very different things here: [] authorises nothing to be
    destroyed and means the fleet is empty; None means we are blind, and a
    blind watchdog must not act (nor conclude the fleet is clean)."""
    p = vast("show", "instances", "--raw")
    if p.returncode != 0:
        log(f"WARN show instances failed rc={p.returncode}: "
            f"{(p.stderr or '').strip()[:200]}")
        return None
    txt = (p.stdout or "").strip()
    if not txt:
        return None
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        start = txt.find("[")
        try:
            d = json.loads(txt[start:]) if start >= 0 else None
        except json.JSONDecodeError:
            log(f"WARN unparseable instance list: {txt[:200]}")
            return None
    return d if isinstance(d, list) else None


def load_reg():
    try:
        return json.loads(REG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_reg(reg):
    REG.parent.mkdir(parents=True, exist_ok=True)
    tmp = REG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(REG)


def destroy(iid, reason: str) -> bool:
    """Destroy and VERIFY. `-y` matters: without it the CLI aborts silently
    and the instance keeps billing (tools/vast_pick.py documents the same)."""
    iid = str(iid)
    for attempt in range(1, 4):
        p = vast("destroy", "instance", iid, "-y")
        out = ((p.stdout or "") + (p.stderr or "")).strip()[:200]
        log(f"DESTROY {iid} ({reason}) attempt {attempt}: rc={p.returncode} {out}")
        time.sleep(4.0)
        live = instances()
        if live is None:
            continue
        if not any(str(i.get("id")) == iid for i in live):
            log(f"DESTROY {iid} confirmed gone")
            _merge_reg(drop=[iid])
            return True
    log(f"!! DESTROY {iid} FAILED after 3 attempts - destroy it by hand")
    return False


def _merge_reg(updates=None, drop=()):
    """Read-modify-write the registry under a lock, FIELD by field.

    The sweep and a harvest thread both write, and the thread's write lands
    minutes after the sweep took its snapshot. Saving a stale whole dict would
    silently drop `harvested_at` and re-run a pull that had already succeeded
    (and could resurrect a claim for a box destroyed meanwhile), so every
    writer merges fields. An entry that has vanished is never re-added.
    """
    with _REG_LOCK:
        cur = load_reg()
        for iid, fields in (updates or {}).items():
            if str(iid) in cur:
                cur[str(iid)].update(fields)
        for iid in drop:
            cur.pop(str(iid), None)
        save_reg(cur)


def _csv(s):
    """'a,b c' -> ['a', 'b', 'c'] - the list form every harvest flag takes."""
    return [x for x in str(s).replace(",", " ").split() if x]


def _posix_repo():
    """harvest_box.sh runs under Git Bash and writes to $LOCAL_REPO/runs/...,
    so hand it THIS checkout - a daemon started in a second clone must not
    harvest into the first."""
    p = str(ROOT).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = "/" + p[0].lower() + p[2:]
    return p


def _bash():
    """Absolute path to bash, or "" if there is none.

    MEASURED, not assumed: `bash` is NOT on the PATH a PowerShell-launched
    process inherits on this workstation, and the daemon is launched from
    PowerShell. Resolving it only at harvest time would turn every automatic
    pull into "no bash on PATH" 20 minutes before a deadline, which is the
    exact failure this whole mechanism exists to remove - so the daemon says
    so at startup instead.
    """
    import shutil
    p = shutil.which("bash")
    if p:
        return p
    for c in (r"C:\Program Files\Git\bin\bash.exe",
              r"C:\Program Files\Git\usr\bin\bash.exe",
              r"C:\Program Files (x86)\Git\bin\bash.exe",
              "/bin/bash", "/usr/bin/bash"):
        if Path(c).exists():
            return c
    return ""


def _exec(cmd, env, timeout):
    """The one place a harvest/ssh child is started (tests patch this)."""
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          timeout=timeout, cwd=str(ROOT),
                          encoding="utf-8", errors="replace")


def _ssh(host, port, remote, timeout=SSH_TIMEOUT_S):
    """-> (rc, stdout). Never raises: a failed ssh is not evidence of
    anything, so it must not look like an answer and must not kill a sweep."""
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ConnectTimeout=20", "-p", str(port), f"root@{host}", remote]
    try:
        p = _exec(cmd, dict(os.environ, PYTHONIOENCODING="utf-8"), timeout)
        return p.returncode, (p.stdout or "")
    except (subprocess.TimeoutExpired, OSError) as exc:
        return -1, f"<{exc!r}>"


def pid_state(host, port, pid_file):
    """'alive' | 'dead' | 'unknown' for the trainer behind runs/<run>.pid.

    A missing pid file is 'unknown', not 'dead': a run that has not written
    one yet has not exited. A failed ssh is 'unknown' too. Only a pid the box
    itself reports gone counts, and even that needs a second look
    (PID_DEAD_NEEDED) before anything acts on it.
    """
    q = str(pid_file).replace("'", "")
    rc, out = _ssh(host, port,
                   "cd /root/RL_Surf 2>/dev/null || exit 0; "
                   f"if [ ! -f '{q}' ]; then echo PID_NOFILE; "
                   f"elif kill -0 $(cat '{q}' 2>/dev/null) 2>/dev/null; "
                   "then echo PID_ALIVE; else echo PID_DEAD; fi")
    if rc != 0:
        return "unknown"
    if "PID_ALIVE" in out:
        return "alive"
    if "PID_DEAD" in out:
        return "dead"
    return "unknown"


def harvest_endpoint(spec, inst=None):
    """-> (host, port) for the pull.

    The STORED endpoint wins: it is the one the readiness probe verified and
    the deploy used. A raw listing has been seen handing two instances the
    same ssh endpoint while one was still loading (wave_launch.py,
    2026-09-04), so the API is the fallback, not the source of truth.
    """
    host, port = spec.get("host"), spec.get("port")
    if inst:
        host = host or inst.get("ssh_host")
        port = port or inst.get("ssh_port")
    try:
        port = int(port or 0)
    except (TypeError, ValueError):
        port = 0
    return (str(host) if host else ""), port


def _backoff(ent):
    n = max(1, int(ent.get("harvest_attempts") or 1))
    return HARVEST_BACKOFF_S * (2 ** (n - 1))


def run_harvest(iid, ent, reason, inst=None):
    """Pull one box's results NOW. Synchronous; True on success.

    It shells out to tools/harvest_box.sh - the same path a human uses - so
    there is exactly one implementation of what "harvested" means.
    """
    iid = str(iid)
    spec = dict(ent.get("harvest") or {})
    host, port = harvest_endpoint(spec, inst)
    runs = [str(r) for r in (spec.get("runs") or [])]
    if not host or not port:
        _merge_reg({iid: {"harvest_error": "no ssh endpoint known"}})
        log(f"!! HARVEST {iid} ({reason}): no ssh endpoint - register with "
            f'--harvest "<port> <host> <run>"')
        return False
    sh = _bash()
    if not sh:
        _merge_reg({iid: {"harvest_error": "no bash on PATH"}})
        log(f"!! HARVEST {iid} ({reason}): no bash found - install Git for "
            f"Windows or put bash on the daemon's PATH")
        return False
    cmd = [sh, str(ROOT / "tools" / "harvest_box.sh"), str(port), host] + runs
    env = dict(os.environ, PYTHONIOENCODING="utf-8", LOCAL_REPO=_posix_repo())
    if spec.get("extra"):
        env["HARVEST_EXTRA"] = ",".join(spec["extra"])
    if spec.get("newest"):
        env["HARVEST_NEWEST"] = ",".join(spec["newest"])
    if spec.get("only_extra"):
        env["HARVEST_ONLY_EXTRA"] = "1"
    log(f"HARVEST {iid} ({reason}): {' '.join(runs) or '<extras only>'} from "
        f"{host}:{port} -> {_posix_repo()}/runs/research/")
    # Hold the slot for one attempt's worth of time. `_HARVESTING` guards this
    # process; the registry is what stops a daemon restarted mid-pull from
    # starting a second copy of the same transfer. It is written HERE, not by
    # the sweep, so that a pull which fails immediately still owns the field
    # it wrote (the sweep's own merge lands after this one, and must not undo
    # the backoff).
    _merge_reg({iid: {"harvest_next_try": now() + HARVEST_TIMEOUT_S}})
    t0 = now()
    try:
        p = _exec(cmd, env, HARVEST_TIMEOUT_S)
        rc, out, err = p.returncode, (p.stdout or ""), (p.stderr or "")
    except subprocess.TimeoutExpired:
        rc, out, err = -9, "", f"timed out after {HARVEST_TIMEOUT_S / 60:.0f} min"
    except OSError as exc:                     # no bash on PATH, no such script
        rc, out, err = -1, "", repr(exc)
    dt = (now() - t0) / 60.0
    if rc == 0:
        _merge_reg({iid: {"harvested_at": stamp(), "harvest_error": ""}})
        log(f"HARVEST {iid} OK in {dt:.1f} min ({reason})")
        return True
    why = " ".join((err.strip() or out.strip() or "?").split())[-300:]
    _merge_reg({iid: {"harvest_error": f"rc={rc} {why}"[:400],
                      "harvest_next_try": now() + _backoff(ent)}})
    log(f"!! HARVEST {iid} FAILED rc={rc} after {dt:.1f} min ({reason}): {why}")
    return False


def _dispatch(fn, *args):
    """Run fn OFF the sweep thread: a pull can take 15 minutes and the sweep
    is what enforces every other box's deadline. Tests replace this with an
    inline call."""
    th = threading.Thread(target=fn, args=args, daemon=True)
    th.start()
    return th


def _harvest_worker(iid, ent, reason, inst):
    try:
        run_harvest(iid, ent, reason, inst)
    except Exception as exc:                      # noqa: BLE001 - never silent
        log(f"!! HARVEST {iid} raised {exc!r}")
    finally:
        _HARVESTING.discard(str(iid))


def _harvest_tick(iid, ent, inst, t, dl, upd):
    """Should this box's results be pulled now? If so, start the pull.

    Two triggers. The DEADLINE one opens at `deadline - HARVEST_LEAD_S` and is
    the one that matters: it is what makes the results independent of any
    agent being awake. The PID one is the early exit - a trainer that finishes
    hours early leaves the ON-BOX watchdog ten minutes before it destroys the
    box, and that is the other way a wave was lost.

    Bookkeeping goes into `upd`, which the sweep merges; `ent` is updated in
    step so the rest of the sweep sees the same numbers.
    """
    spec = ent.get("harvest") or {}
    if not spec or iid in _HARVESTING:
        return
    trigger = ent.get("harvest_trigger") or ""
    if ent.get("harvested_at") and trigger == "lead":
        return                        # the deadline pull is the last word
    attempts = int(ent.get("harvest_attempts") or 0)
    lead_open = bool(dl) and t >= dl - HARVEST_LEAD_S
    reason = ""
    if lead_open and trigger != "lead":
        # a new trigger gets its own attempts, even after an early pull
        attempts, trigger = 0, "lead"
        reason = f"deadline in {max(0.0, dl - t) / 60:.0f} min"
    elif (attempts and attempts < HARVEST_TRIES
          and t >= float(ent.get("harvest_next_try") or 0.0)):
        reason = (f"retry {attempts + 1}/{HARVEST_TRIES} after "
                  f"{str(ent.get('harvest_error') or '?')[:80]}")
    elif not attempts and spec.get("pid_file") and ent.get("ready"):
        # only a box that HAS come up can have a trainer to lose
        if t < float(ent.get("pid_next_poll") or 0.0):
            return
        host, port = harvest_endpoint(spec, inst)
        upd["pid_next_poll"] = ent["pid_next_poll"] = t + PID_POLL_S
        if not host or not port:
            return
        st = pid_state(host, port, spec["pid_file"])
        if st == "alive":
            if ent.get("pid_dead_polls"):
                upd["pid_dead_polls"] = ent["pid_dead_polls"] = 0
            return
        if st == "unknown":
            log(f"   {iid} ({ent.get('label', '?')}) pid probe inconclusive - "
                f"not evidence, looking again in {PID_POLL_S / 60:.0f} min")
            return
        n = int(ent.get("pid_dead_polls") or 0) + 1
        upd["pid_dead_polls"] = ent["pid_dead_polls"] = n
        if n < PID_DEAD_NEEDED:
            log(f"   {iid} ({ent.get('label', '?')}) trainer pid gone "
                f"({n}/{PID_DEAD_NEEDED}) - one observation is not evidence")
            return
        trigger = "pid"
        reason = f"trainer pid gone in {n} consecutive polls"
    if not reason:
        return
    attempts += 1
    upd["harvest_attempts"] = ent["harvest_attempts"] = attempts
    upd["harvest_trigger"] = ent["harvest_trigger"] = trigger
    upd["harvest_reason"] = ent["harvest_reason"] = reason
    ent["harvest_next_try"] = t + HARVEST_TIMEOUT_S   # run_harvest persists it
    _HARVESTING.add(iid)
    try:
        _dispatch(_harvest_worker, iid, dict(ent), reason, inst)
    except Exception:            # a thread that never started holds no slot
        _HARVESTING.discard(iid)
        raise


def created_at(inst) -> float:
    for k in ("start_date", "create_time", "start_time"):
        v = inst.get(k)
        if isinstance(v, (int, float)) and v > 1e9:
            return float(v)
    return 0.0


def sweep() -> int:
    live = instances()
    if live is None:
        return 0                      # blind: do nothing, try again next tick
    reg = load_reg()
    t = now()
    if len(live) > MAX_BOXES:
        log(f"!! {len(live)} instances live, cap is {MAX_BOXES}")
    killed = 0
    updates = {}
    for inst in live:
        iid = str(inst.get("id"))
        ent = reg.get(iid)
        age = t - created_at(inst) if created_at(inst) else 0.0
        status = (inst.get("actual_status") or inst.get("cur_state") or "?")
        if ent is None:
            if age and age < GRACE_S:
                continue              # still inside the claim window
            killed += destroy(iid, f"unregistered (age {age / 60:.1f}m, "
                                   f"status {status})")
            continue
        dl = float(ent.get("deadline", 0.0))
        if dl and t > dl:
            # The money rule wins over the evidence rule: an unharvested box
            # still dies on time. Say so loudly - this is the one line that
            # tells a reader results were lost and why.
            if ent.get("harvest") and not ent.get("harvested_at"):
                log(f"!! {iid} ({ent.get('label', '?')}) hits its deadline "
                    f"UNHARVESTED - destroying anyway: "
                    f"{ent.get('harvest_error') or 'no attempt completed'}")
            killed += destroy(iid, f"deadline passed by "
                                   f"{(t - dl) / 60:.1f}m "
                                   f"(label {ent.get('label', '?')})")
            continue
        upd = {}
        try:
            _harvest_tick(iid, ent, inst, t, dl, upd)
        except Exception as exc:                  # noqa: BLE001 - never a kill
            log(f"WARN harvest tick for {iid} raised {exc!r}")
        if upd:
            updates.setdefault(iid, {}).update(upd)
        if status == "running":
            if not ent.get("ready"):
                ent["ready"] = True          # latched: it DID come up
                updates.setdefault(iid, {})["ready"] = True
            continue
        if ent.get("ready"):
            # a box that came up and is momentarily not "running": a host
            # or API blip, not a box that never started. The deadline still
            # bounds it; say so and leave it alone.
            log(f"   {iid} ({ent.get('label', '?')}) reports {status} after "
                f"coming up - leaving it to its deadline")
            continue
        if age and age > READY_S:
            killed += destroy(iid, f"never came up (status {status}, "
                                   f"age {age / 60:.1f}m)")
    # registry entries whose instance is gone: drop the claim
    ids = {str(i.get("id")) for i in live}
    stale = [k for k in reg if k not in ids]
    if stale:
        for k in stale:
            del reg[k]
        log(f"registry: dropped {len(stale)} claim(s) for instances already gone")
    if updates or stale:
        _merge_reg(updates, stale)
    return killed


def cmd_daemon(a):
    log(f"watchdog up: poll {POLL_S:.0f}s, grace {GRACE_S:.0f}s, "
        f"ready {READY_S:.0f}s, cap {MAX_BOXES}, registry {REG}")
    log(f"harvest: lead {HARVEST_LEAD_S / 60:.0f} min, "
        f"{HARVEST_TRIES} tries, {HARVEST_TIMEOUT_S / 60:.0f} min timeout, "
        f"pid poll {PID_POLL_S / 60:.0f} min x{PID_DEAD_NEEDED}, "
        f"into {_posix_repo()}/runs/research/")
    sh = _bash()
    log(f"harvest: bash {sh}" if sh else
        "!! harvest: NO BASH FOUND - every automatic pull will fail. Install "
        "Git for Windows or start the daemon from a shell that has bash.")
    while True:
        try:
            sweep()
        except Exception as exc:                      # never let it die
            log(f"WARN sweep raised {exc!r}")
        time.sleep(POLL_S)


def _harvest_from_args(a, prev):
    """-> (spec or None, is_new).

    No harvest flag at all -> keep whatever the entry already had. A
    re-register (to relabel, or to pull the deadline in under the on-box one)
    must not silently drop the spec, exactly like the `ready` latch. Any
    harvest flag -> a new spec, and the bookkeeping resets with it, because a
    new spec means different files to pull.
    """
    flags = ("harvest", "harvest_runs", "harvest_extra", "harvest_newest",
             "pid_file", "harvest_only_extra")
    if not any(getattr(a, f, None) for f in flags):
        return prev.get("harvest"), False
    spec = dict(prev.get("harvest") or {})
    if getattr(a, "harvest", None):
        parts = str(a.harvest).split()
        if len(parts) < 2 or not parts[0].isdigit():
            raise SystemExit('--harvest wants "<port> <host> <run> [run...]"')
        spec["port"], spec["host"] = int(parts[0]), parts[1]
        if len(parts) > 2:
            spec["runs"] = parts[2:]
    if getattr(a, "harvest_runs", None):
        spec["runs"] = _csv(a.harvest_runs)
    for flag, key in (("harvest_extra", "extra"), ("harvest_newest", "newest")):
        v = getattr(a, flag, None)
        if v:
            if "'" in v:               # it is quoted into a remote shell
                raise SystemExit(f"--{flag.replace('_', '-')}: no quotes please")
            spec[key] = _csv(v)
    if getattr(a, "pid_file", None):
        spec["pid_file"] = a.pid_file
    if getattr(a, "harvest_only_extra", False):
        spec["only_extra"] = True
    spec.setdefault("runs", [])
    return spec, True


def cmd_register(a):
    reg = load_reg()
    iid = str(a.id)
    if iid not in reg and len(reg) >= MAX_BOXES:
        # Loud, but NOT a refusal. A hard error here is a footgun: the agent
        # that just rented a box would fail to claim it, the sweep would see
        # an unclaimed instance and destroy a box somebody is about to use.
        # Over-cap is the orchestrator's problem to fix by releasing one;
        # an unregistered box is the watchdog's problem and it kills it.
        log(f"!! OVER CAP: registry holds {len(reg)} boxes, cap is "
            f"{MAX_BOXES} - registering {iid} anyway, release one NOW")
    # Preserve the latched `ready` flag across a re-register. Dropping it
    # re-arms the readiness kill on a box that HAS come up, which is exactly
    # the failure that destroyed a live training box on 2026-08-23. Agents
    # re-register to relabel, and that must not be dangerous.
    prev = reg.get(iid) or {}
    spec, fresh = _harvest_from_args(a, prev)
    reg[iid] = {"label": a.label, "owner": a.owner,
                "registered": prev.get("registered") or stamp(),
                "ready": bool(prev.get("ready")),
                "deadline": now() + a.minutes * 60.0,
                "deadline_utc": datetime.fromtimestamp(
                    now() + a.minutes * 60.0, timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ")}
    if spec:
        reg[iid]["harvest"] = spec
        if not fresh:              # the same spec: carry its bookkeeping too
            for k in ("harvested_at", "harvest_attempts", "harvest_error",
                      "harvest_next_try", "harvest_trigger", "harvest_reason",
                      "pid_dead_polls", "pid_next_poll"):
                if k in prev:
                    reg[iid][k] = prev[k]
    save_reg(reg)
    log(f"registered {iid} label={a.label} owner={a.owner} "
        f"for {a.minutes:g} min (until {reg[iid]['deadline_utc']})")
    if spec:
        log(f"   harvest {' '.join(spec.get('runs') or []) or '<extras only>'}"
            f" from {spec.get('host', '?')}:{spec.get('port', '?')} at "
            f"deadline-{HARVEST_LEAD_S / 60:.0f}min"
            + (f", pid {spec['pid_file']}" if spec.get("pid_file") else "")
            + ("  (the ON-BOX deadline must be >= this one)"))
    elif not prev:
        log(f"   {iid} has NO harvest spec: its results die with it unless an "
            f'agent is awake. Re-register with --harvest "<port> <host> <run>"')


def cmd_extend(a):
    reg = load_reg()
    iid = str(a.id)
    if iid not in reg:
        raise SystemExit(f"{iid} is not registered")
    # MONOTONE: extend can only push a deadline OUT, never pull it in.
    # The old semantic was a bare `now() + minutes`, so "extend by 25" on a
    # box with 60 minutes left SHORTENED it to 25 - it cut three
    # mid-training boxes on 2026-08-23 and was caught by luck in 10 s.
    # A command named `extend` must never be able to kill a run early.
    want = now() + a.minutes * 60.0
    cur = float(reg[iid].get("deadline") or 0.0)
    if want < cur:
        log(f"   {iid}: extend to now+{a.minutes}m would SHORTEN it by "
            f"{(cur - want) / 60:.0f} min - keeping the later deadline")
    reg[iid]["deadline"] = max(cur, want)
    reg[iid]["deadline_utc"] = datetime.fromtimestamp(
        reg[iid]["deadline"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    save_reg(reg)
    log(f"extended {iid} to {reg[iid]['deadline_utc']}")


def _live_inst(iid):
    """The live instance dict for iid, or None. Only used to fill in an ssh
    endpoint the registry does not have."""
    for i in (instances() or []):
        if str(i.get("id")) == str(iid):
            return i
    return None


def cmd_release(a):
    """Harvest, THEN destroy. The pull is synchronous here: `release` is the
    manual path and the caller is waiting on it anyway."""
    ent = load_reg().get(str(a.id)) or {}
    if ent.get("harvest") and not getattr(a, "no_harvest", False):
        if ent.get("harvested_at"):
            log(f"release {a.id}: already harvested at {ent['harvested_at']}")
        elif not run_harvest(a.id, ent, "release", _live_inst(a.id)):
            log(f"!! release {a.id}: harvest FAILED - destroying anyway "
                f"(running or deleted, never stale). Results are LOST.")
    destroy(a.id, "released by owner")


def cmd_harvest(a):
    """Pull one box's results now, without destroying anything."""
    ent = load_reg().get(str(a.id))
    if ent is None:
        raise SystemExit(f"{a.id} is not registered")
    if not ent.get("harvest"):
        raise SystemExit(f"{a.id} carries no harvest spec - re-register it "
                         f'with --harvest "<port> <host> <run>"')
    if ent.get("harvested_at") and not a.force:
        log(f"{a.id} was harvested at {ent['harvested_at']} (--force to redo)")
        return
    if not run_harvest(a.id, ent, "manual", _live_inst(a.id)):
        raise SystemExit(1)


def cmd_list(a):
    reg = load_reg()
    live = instances()
    print(f"registry ({REG}):")
    for iid, e in sorted(reg.items()):
        left = (e.get("deadline", 0) - now()) / 60.0
        spec = e.get("harvest") or {}
        if not spec:
            hv = "harvest: NONE - results die with the box"
        elif e.get("harvested_at"):
            hv = f"harvested {e['harvested_at']}"
        else:
            hv = (f"harvest {','.join(spec.get('runs') or []) or 'extras'} in "
                  f"{left - HARVEST_LEAD_S / 60:+.0f} min")
            if e.get("harvest_error"):
                hv += f"  !! {str(e['harvest_error'])[:60]}"
        print(f"  {iid}  {e.get('label', '?'):12s} owner={e.get('owner', '?'):10s} "
              f"{left:+.0f} min left  {hv}")
    if live is None:
        print("live: <could not reach the vast API>")
        return
    print(f"live ({len(live)}):")
    for i in live:
        print(f"  {i.get('id')}  {i.get('actual_status')}  "
              f"{i.get('gpu_name')}  {i.get('ssh_host')}:{i.get('ssh_port')}  "
              f"${i.get('dph_total', 0):.3f}/h"
              + ("" if str(i.get("id")) in reg else "   <-- UNREGISTERED"))


def cmd_reap_all(a):
    live = instances()
    if live is None:
        raise SystemExit("could not reach the vast API - refusing to guess")
    if not live:
        print("nothing rented")
        return
    for i in live:
        destroy(i.get("id"), "reap-all")


def main():
    global HARVEST_LEAD_S
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--daemon", action="store_true", help="poll forever")
    ap.add_argument("--harvest-lead", type=float, default=HARVEST_LEAD_S / 60.0,
                    metavar="MIN",
                    help="pull results this many minutes before the deadline "
                         "(default 20)")

    r = sub.add_parser("register")
    r.add_argument("id")
    r.add_argument("--minutes", type=float, default=90.0)
    r.add_argument("--label", default="?")
    r.add_argument("--owner", default="?")
    r.add_argument("--harvest", metavar='"PORT HOST RUN [RUN...]"',
                   help="harvest these runs off this box before the deadline")
    r.add_argument("--harvest-runs", metavar="RUN,RUN",
                   help="runs only; the ssh endpoint is read from the vast "
                        "listing at harvest time")
    r.add_argument("--harvest-extra", metavar="PATH,PATH",
                   help="extra repo-relative files on the box (an expert "
                        "loop's runs/<name>/expert_summary.jsonl)")
    r.add_argument("--harvest-newest", metavar="GLOB,GLOB",
                   help="globs; each pulls its NEWEST match only "
                        "(runs/<name>/round_*/train/ckpt_final.pt)")
    r.add_argument("--harvest-only-extra", action="store_true",
                   help="skip the standard per-run pull (a box with no "
                        "runs/<run>/progress.csv)")
    r.add_argument("--pid-file", metavar="runs/<run>.pid",
                   help="poll this pid over ssh and harvest as soon as the "
                        "trainer has exited")
    r.set_defaults(fn=cmd_register)

    e = sub.add_parser("extend")
    e.add_argument("id")
    e.add_argument("--minutes", type=float, default=30.0)
    e.set_defaults(fn=cmd_extend)

    x = sub.add_parser("release")
    x.add_argument("id")
    x.add_argument("--no-harvest", action="store_true",
                   help="destroy without pulling the results first (the pull "
                        "is synchronous and can take 15 min, so a caller with "
                        "a short timeout of its own wants this)")
    x.set_defaults(fn=cmd_release)

    h = sub.add_parser("harvest")
    h.add_argument("id")
    h.add_argument("--force", action="store_true",
                   help="pull again even if harvested_at is already set")
    h.set_defaults(fn=cmd_harvest)

    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("sweep").set_defaults(fn=lambda a: sweep())
    sub.add_parser("reap-all").set_defaults(fn=cmd_reap_all)

    a = ap.parse_args()
    if getattr(a, "harvest_lead", None):
        HARVEST_LEAD_S = float(a.harvest_lead) * 60.0
    if a.daemon and not a.cmd:
        return cmd_daemon(a)
    if not getattr(a, "fn", None):
        ap.print_help()
        return 2
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
