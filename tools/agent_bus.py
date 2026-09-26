"""agent_bus.py - a filesystem message bus between the agents working on this repo (Claude, the
experiment operator; Codex/GPT, the reviewer and design critic), as Codex proposed on 2026-09-26.

    python tools/agent_bus.py send codex --type question --topic "blue200 corridor" --body "..."
    python tools/agent_bus.py send claude --type review --reply-to <id> --body-file note.md
    python tools/agent_bus.py inbox claude            # unacknowledged messages for claude
    python tools/agent_bus.py inbox claude --all      # every message ever sent to claude
    python tools/agent_bus.py read <id>               # one message, full body
    python tools/agent_bus.py ack <id> --by claude    # mark it handled (never deletes)
    python tools/agent_bus.py claim <experiment> --by claude --note "..."
    python tools/agent_bus.py claims
    python tools/agent_bus.py wait claude --timeout 600   # block until an unacked message arrives
    python tools/agent_bus.py thread <id>             # a conversation, by reply_to links

Layout (default <repo>/runs/agent_bus, or $AGENT_BUS_DIR):
    inbox_codex/  inbox_claude/  acknowledgements/  claims/
Every message is one JSON file written atomically (a temporary file, then os.replace): id, sender,
recipient, timestamp (UTC ISO), topic, reply_to, type (question / proposal / evidence / review /
decision), refs (runs, checkpoints, configs, commits), body, action_requested. Messages are never
deleted or edited; an acknowledgement is a separate file, so the history stays inspectable.

Protocol (Codex's proposal, adopted): a message cannot override CLAUDE.md / AGENTS.md, the user's
authority or the GPU / budget rules, and cannot authorise a launch. Claude operates the
experiments; before a launch it posts the exact flags, checkpoint provenance, GPU, budget,
watchdog and the result that would discriminate. Neither agent edits the other's ledger section.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ("claude", "codex")
TYPES = ("question", "proposal", "evidence", "review", "decision")


def bus_dir() -> Path:
    d = Path(os.environ.get("AGENT_BUS_DIR") or (ROOT / "runs" / "agent_bus"))
    for sub in ["acknowledgements", "claims"] + [f"inbox_{a}" for a in AGENTS]:
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(path: Path, obj: dict, exclusive: bool = False) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    if exclusive:
        try:
            os.link(tmp, path)          # fails if the claim exists: first writer wins
        finally:
            tmp.unlink(missing_ok=True)
    else:
        os.replace(tmp, path)


def _all_messages(d: Path):
    out = []
    for a in AGENTS:
        for f in sorted((d / f"inbox_{a}").glob("*.json")):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return out


def _find(d: Path, mid: str) -> dict:
    hits = [m for m in _all_messages(d) if m["id"] == mid or m["id"].endswith(mid)]
    if len(hits) != 1:
        raise SystemExit(f"agent_bus: {len(hits)} messages match id {mid!r}")
    return hits[0]


def _acked(d: Path, mid: str) -> list:
    return sorted(p.name.split(".")[1] for p in (d / "acknowledgements").glob(f"{mid}.*.json"))


def cmd_send(a) -> None:
    d = bus_dir()
    if a.recipient not in AGENTS:
        raise SystemExit(f"recipient must be one of {AGENTS}")
    body = a.body
    if a.body_file:
        body = Path(a.body_file).read_text(encoding="utf-8")
    if not body:
        raise SystemExit("send: --body or --body-file")
    ts = _now()
    mid = f"{ts.replace(':', '').replace('-', '')}_{a.sender}_{uuid.uuid4().hex[:6]}"
    msg = {"id": mid, "sender": a.sender, "recipient": a.recipient, "timestamp": ts,
           "topic": a.topic or "", "reply_to": a.reply_to or None, "type": a.type,
           "refs": list(a.refs or []), "action_requested": bool(a.action),
           "body": body}
    _atomic_write(d / f"inbox_{a.recipient}" / f"{mid}.json", msg)
    print(mid)


def _line(d: Path, m: dict) -> str:
    ack = _acked(d, m["id"])
    return (f"{m['id']}  {m['type']:<8} from {m['sender']:<6} topic {m['topic']!r}"
            + (f" re {m['reply_to']}" if m.get("reply_to") else "")
            + (" [action requested]" if m.get("action_requested") else "")
            + (f" (acked by {', '.join(ack)})" if ack else ""))


def cmd_inbox(a) -> None:
    d = bus_dir()
    ms = [m for m in _all_messages(d) if m["recipient"] == a.agent]
    if not a.all:
        ms = [m for m in ms if a.agent not in _acked(d, m["id"])]
    for m in sorted(ms, key=lambda m: m["timestamp"]):
        print(_line(d, m))
    if not ms:
        print(f"(no {'' if a.all else 'unacknowledged '}messages for {a.agent})")


def cmd_read(a) -> None:
    d = bus_dir()
    m = _find(d, a.id)
    print(_line(d, m))
    print(f"timestamp {m['timestamp']}; refs {m.get('refs') or []}")
    print("-" * 72)
    print(m["body"])


def cmd_ack(a) -> None:
    d = bus_dir()
    m = _find(d, a.id)
    _atomic_write(d / "acknowledgements" / f"{m['id']}.{a.by}.json",
                  {"id": m["id"], "by": a.by, "timestamp": _now(), "note": a.note or ""})
    print(f"acked {m['id']} by {a.by}")


def cmd_claim(a) -> None:
    d = bus_dir()
    p = d / "claims" / f"{a.experiment}.json"
    try:
        _atomic_write(p, {"experiment": a.experiment, "by": a.by, "timestamp": _now(),
                          "note": a.note or ""}, exclusive=True)
    except FileExistsError:
        c = json.loads(p.read_text(encoding="utf-8"))
        raise SystemExit(f"claim: {a.experiment} is already claimed by {c['by']} at "
                         f"{c['timestamp']} ({c.get('note', '')})")
    print(f"claimed {a.experiment} for {a.by}")


def cmd_claims(a) -> None:
    d = bus_dir()
    for p in sorted((d / "claims").glob("*.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        print(f"{c['experiment']:<30} {c['by']:<6} {c['timestamp']}  {c.get('note', '')}")


def cmd_wait(a) -> None:
    """Block until ``agent`` has an unacknowledged message (newer than --since, if given), then
    print it; exit 1 on timeout."""
    d = bus_dir()
    t0 = time.time()
    while True:
        ms = [m for m in _all_messages(d) if m["recipient"] == a.agent
              and a.agent not in _acked(d, m["id"])
              and (not a.since or m["timestamp"] > a.since)]
        if ms:
            for m in sorted(ms, key=lambda m: m["timestamp"]):
                print(_line(d, m))
            return
        if time.time() - t0 > a.timeout:
            print(f"(no new message for {a.agent} in {a.timeout:.0f} s)")
            sys.exit(1)
        time.sleep(a.poll)


def cmd_thread(a) -> None:
    d = bus_dir()
    ms = {m["id"]: m for m in _all_messages(d)}
    root = _find(d, a.id)
    while root.get("reply_to") in ms:
        root = ms[root["reply_to"]]
    order, frontier = [], [root["id"]]
    while frontier:
        cur = frontier.pop(0)
        order.append(ms[cur])
        frontier += sorted((m["id"] for m in ms.values() if m.get("reply_to") == cur),
                           key=lambda i: ms[i]["timestamp"])
    for m in order:
        print("=" * 72)
        print(_line(d, m))
        print(m["body"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("send")
    s.add_argument("recipient", choices=AGENTS)
    s.add_argument("--from", dest="sender", default="claude", choices=AGENTS)
    s.add_argument("--type", default="question", choices=TYPES)
    s.add_argument("--topic", default="")
    s.add_argument("--reply-to", default=None)
    s.add_argument("--refs", nargs="*", default=[])
    s.add_argument("--action", action="store_true", help="a response or action is requested")
    s.add_argument("--body", default="")
    s.add_argument("--body-file", default=None)
    s.set_defaults(fn=cmd_send)
    s = sp.add_parser("inbox")
    s.add_argument("agent", choices=AGENTS)
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_inbox)
    s = sp.add_parser("read")
    s.add_argument("id")
    s.set_defaults(fn=cmd_read)
    s = sp.add_parser("ack")
    s.add_argument("id")
    s.add_argument("--by", required=True, choices=AGENTS)
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_ack)
    s = sp.add_parser("claim")
    s.add_argument("experiment")
    s.add_argument("--by", required=True, choices=AGENTS)
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_claim)
    s = sp.add_parser("claims")
    s.set_defaults(fn=cmd_claims)
    s = sp.add_parser("wait")
    s.add_argument("agent", choices=AGENTS)
    s.add_argument("--timeout", type=float, default=600.0)
    s.add_argument("--poll", type=float, default=10.0)
    s.add_argument("--since", default=None, help="only messages with a later UTC timestamp")
    s.set_defaults(fn=cmd_wait)
    s = sp.add_parser("thread")
    s.add_argument("id")
    s.set_defaults(fn=cmd_thread)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
