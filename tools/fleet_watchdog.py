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

Usage:

    # start it BEFORE renting anything (detached, survives the session)
    python tools/fleet_watchdog.py --daemon

    # the moment `vastai create` returns an id, claim it
    python tools/fleet_watchdog.py register <id> --minutes 90 --label xROUTE

    # done with a box: this destroys it now and drops the claim
    python tools/fleet_watchdog.py release <id>

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
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "runs" / "fleet.json"
LOG = ROOT / "runs" / "fleet_watchdog.log"

POLL_S = 45.0
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
READY_S = 420.0
MAX_BOXES = 6        # user-set 2026-08-23 (was 4); plus the local GPU


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
            reg = load_reg()
            if iid in reg:
                del reg[iid]
                save_reg(reg)
            return True
    log(f"!! DESTROY {iid} FAILED after 3 attempts - destroy it by hand")
    return False


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
    dirty = False
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
            killed += destroy(iid, f"deadline passed by "
                                   f"{(t - dl) / 60:.1f}m "
                                   f"(label {ent.get('label', '?')})")
            continue
        if status == "running":
            if not ent.get("ready"):
                ent["ready"] = True          # latched: it DID come up
                dirty = True
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
        dirty = True
        log(f"registry: dropped {len(stale)} claim(s) for instances already gone")
    if dirty:
        save_reg(reg)
    return killed


def cmd_daemon(a):
    log(f"watchdog up: poll {POLL_S:.0f}s, grace {GRACE_S:.0f}s, "
        f"ready {READY_S:.0f}s, cap {MAX_BOXES}, registry {REG}")
    while True:
        try:
            sweep()
        except Exception as exc:                      # never let it die
            log(f"WARN sweep raised {exc!r}")
        time.sleep(POLL_S)


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
    reg[iid] = {"label": a.label, "owner": a.owner,
                "registered": prev.get("registered") or stamp(),
                "ready": bool(prev.get("ready")),
                "deadline": now() + a.minutes * 60.0,
                "deadline_utc": datetime.fromtimestamp(
                    now() + a.minutes * 60.0, timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ")}
    save_reg(reg)
    log(f"registered {iid} label={a.label} owner={a.owner} "
        f"for {a.minutes:g} min (until {reg[iid]['deadline_utc']})")


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


def cmd_release(a):
    destroy(a.id, "released by owner")


def cmd_list(a):
    reg = load_reg()
    live = instances()
    print(f"registry ({REG}):")
    for iid, e in sorted(reg.items()):
        left = (e.get("deadline", 0) - now()) / 60.0
        print(f"  {iid}  {e.get('label', '?'):12s} owner={e.get('owner', '?'):10s} "
              f"{left:+.0f} min left")
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--daemon", action="store_true", help="poll forever")

    r = sub.add_parser("register")
    r.add_argument("id")
    r.add_argument("--minutes", type=float, default=90.0)
    r.add_argument("--label", default="?")
    r.add_argument("--owner", default="?")
    r.set_defaults(fn=cmd_register)

    e = sub.add_parser("extend")
    e.add_argument("id")
    e.add_argument("--minutes", type=float, default=30.0)
    e.set_defaults(fn=cmd_extend)

    x = sub.add_parser("release")
    x.add_argument("id")
    x.set_defaults(fn=cmd_release)

    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("sweep").set_defaults(fn=lambda a: sweep())
    sub.add_parser("reap-all").set_defaults(fn=cmd_reap_all)

    a = ap.parse_args()
    if a.daemon and not a.cmd:
        return cmd_daemon(a)
    if not getattr(a, "fn", None):
        ap.print_help()
        return 2
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
