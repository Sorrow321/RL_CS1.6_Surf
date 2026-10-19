"""orchestrator.py - the fork-and-flush harness (Cai et al. 2026, arXiv 2610.07447).

A round forks config.branches_per_round branch agents from ONE workspace (a git commit + the
protected incumbent). Each is a fresh Claude Code session ("flush") in its own slot directory
(a git worktree, ~/ar/wt/b<k>) on its own branch ar/r<round>b<k>, with at most one GPU box.
They run for config.round_hours (12 h), uncoordinated. At the end every branch is scored with
autoresearch/score.py (the best greedy eval from the true start its runs produced). The winner's
branch head seeds the next round, and the losers are archived and pruned. The schedule is
fixed, not plateau-triggered (the paper: plateau triggers are hard to tune and rarely fired).

    python autoresearch/orchestrator.py run      # start, or resume after a crash (state.json)
    python autoresearch/orchestrator.py status
    python autoresearch/orchestrator.py wrapup   # end the current round now (graceful)
    python autoresearch/orchestrator.py stop     # release every box, end every session

Everything lives under $AR_HOME (default ~/ar): state.json, status.md, orchestrator.log,
wt/ (slots), runs/ (harvests), boxes/, incumbent/, archive/.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # the main checkout (~/RL_Surf)
CFG = json.loads((REPO / "autoresearch" / "config.json").read_text(encoding="utf-8"))
AR_HOME = Path(os.environ.get("AR_HOME", str(Path.home() / "ar")))
STATE = AR_HOME / "state.json"
TMUX = os.environ.get("AR_TMUX", "ar")
CLAUDE = os.environ.get("AR_CLAUDE") or shutil.which("claude") or str(
    Path.home() / ".local" / "bin" / "claude")
POLL_S = 60
TRAILERS = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
            "Claude-Session: https://claude.ai/code/session_01LCSvrZRGWfECSRc6zpr4x7")
# untracked files every slot needs from the main checkout (maps and caches are gitignored, the
# C core is built once): symlinked, so the bsp's st_mtime_ns - which every cache keys on - is
# the main checkout's and nothing re-bakes in a worktree (CLAUDE.md)
SHARED_GLOBS = ["maps/surf_src_sidistic.*", "build/*"]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    AR_HOME.mkdir(parents=True, exist_ok=True)
    with open(AR_HOME / "orchestrator.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sh(cmd, cwd=None, check=True, timeout=600):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                       shell=isinstance(cmd, str))
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd}: {r.stderr.strip()[-500:]}")
    return r


def git(*args, cwd=REPO, check=True):
    return sh(["git", *args], cwd=cwd, check=check)


def load_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else None


def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def agent_env(st, slot) -> dict:
    # `python` is the venv's (numpy, CPU torch, the built core): Ubuntu has no bare `python`,
    # and the system python3 has none of the packages
    path = (f"{AR_HOME / 'venv' / 'bin'}:{Path.home() / '.local' / 'bin'}:"
            f"{os.environ.get('PATH', '/usr/bin:/bin')}")
    return {"AR_HOME": str(AR_HOME), "AR_AGENT": slot["agent"], "AR_ROUND": str(st["round"]),
            "AR_START": str(st["started"]), "RLSURF_FLEET_DIR": str(AR_HOME / "fleet"),
            "PYTHONIOENCODING": "utf-8", "PATH": path}


# ------------------------------------------------------------------ slots -----
def slot_dir(k: int) -> Path:
    return AR_HOME / "wt" / f"b{k}"


def prepare_slot(st, k: int, base: str) -> dict:
    """Point slot k at a fresh branch ar/r<round>b<k> = base, clean, with the shared files."""
    d = slot_dir(k)
    agent = f"r{st['round']}b{k}"
    branch = f"ar/{agent}"
    if not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--force", "-B", branch, str(d), base)
    else:
        git("checkout", "-f", "-B", branch, base, cwd=d)
        git("clean", "-fdx", cwd=d)
    for pat in SHARED_GLOBS:
        for src in REPO.glob(pat):
            if not src.is_file():
                continue
            rel = src.relative_to(REPO)
            dst = d / rel
            if dst.exists() and not dst.is_symlink():
                continue                      # tracked in git: the branch's own copy wins
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
    now = time.time()
    info = {"agent": agent, "round": st["round"], "rounds": CFG["rounds"], "slot": k,
            "branch": branch, "base_commit": base, "started_at": now,
            "wrapup_at": st["round_end"] - CFG["wrapup_minutes"] * 60,
            "round_end": st["round_end"],
            "wrapup_at_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(
                st["round_end"] - CFG["wrapup_minutes"] * 60)),
            "round_end_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(st["round_end"])),
            "ar_home": str(AR_HOME), "incumbent": st.get("incumbent"),
            "runs_dir": str(AR_HOME / "runs" / agent)}
    (d / "AR_BRANCH.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    (AR_HOME / "runs" / agent).mkdir(parents=True, exist_ok=True)
    return {"k": k, "agent": agent, "branch": branch, "dir": str(d), "launched": 0,
            "restarts": 0, "wrapup_sent": False}


PROMPT = ("/loop You are fork-and-flush branch agent {agent} (round {round} of {rounds}) on "
          "surf_src_sidistic. Your identity, deadlines and the incumbent are in AR_BRANCH.json. "
          "On EVERY wakeup: re-read AR_BRANCH.json (if now >= wrapup_at, do only the Wrap-up), "
          "follow autoresearch/PROTOCOL.md exactly (read it and CLAUDE.md now if this is your "
          "first wakeup, then autoresearch/NOTES.md), keep RESULT.json and autoresearch/NOTES.md "
          "current, and schedule the next wakeup (20-30 min while a run trains).")
RESUME = ("/loop Your session was restarted. Continue as fork-and-flush branch agent {agent}: "
          "re-read AR_BRANCH.json, autoresearch/PROTOCOL.md, RESULT.json and the end of "
          "autoresearch/NOTES.md, check your box with box.py status, then carry on.")


def launch(st, slot, resume=False) -> None:
    d = Path(slot["dir"])
    text = (RESUME if resume else PROMPT).format(agent=slot["agent"], round=st["round"],
                                                 rounds=CFG["rounds"])
    (d / ".ar_prompt.txt").write_text(text, encoding="utf-8")
    env = " ".join(f"{k}={shlex.quote(v)}" for k, v in agent_env(st, slot).items())
    cont = "--continue " if resume else ""
    script = (f"#!/bin/bash\ncd {shlex.quote(str(d))}\nexport {env}\n"
              f"exec nice -n 10 {CLAUDE} -n ar-{slot['agent']} {cont}--permission-mode auto "
              f"--model {CFG['model']} --effort {CFG['effort']} "
              f"\"$(cat .ar_prompt.txt)\"\n")
    (d / ".ar_launch.sh").write_text(script, encoding="utf-8")
    os.chmod(d / ".ar_launch.sh", 0o755)
    win = f"b{slot['k']}"
    sh(["tmux", "kill-window", "-t", f"{TMUX}:{win}"], check=False)
    sh(["tmux", "new-window", "-d", "-t", f"{TMUX}:", "-n", win, "-c", str(d),
        f"bash {shlex.quote(str(d / '.ar_launch.sh'))}; echo SESSION ENDED; sleep 86400"])
    slot["launched"] = time.time()
    log(f"launched {slot['agent']} in {TMUX}:{win} ({'resume' if resume else 'fresh context'})")


def session_alive(slot) -> bool:
    r = sh(["tmux", "list-panes", "-t", f"{TMUX}:b{slot['k']}", "-F",
            "#{pane_current_command}"], check=False)
    return r.returncode == 0 and any(c in r.stdout for c in ("claude", "node"))


def send(slot, text: str) -> None:
    win = f"{TMUX}:b{slot['k']}"
    sh(["tmux", "send-keys", "-t", win, "-l", text], check=False)
    sh(["tmux", "send-keys", "-t", win, "Enter"], check=False)


def box_py(*args, env=None):
    e = dict(os.environ, AR_HOME=str(AR_HOME), RLSURF_FLEET_DIR=str(AR_HOME / "fleet"),
             PYTHONIOENCODING="utf-8", **(env or {}))
    return subprocess.run([sys.executable, str(REPO / "autoresearch" / "box.py"), *args],
                          capture_output=True, text=True, env=e, timeout=900)


# ------------------------------------------------------------------ rounds ----
def start_round(st, base: str) -> None:
    st["round"] += 1
    st["round_start"] = time.time()
    st["round_end"] = st["round_start"] + CFG["round_hours"] * 3600
    st["base"] = base
    st["slots"] = [prepare_slot(st, k, base) for k in range(1, CFG["branches_per_round"] + 1)]
    save_state(st)
    log(f"ROUND {st['round']} of {CFG['rounds']}: {len(st['slots'])} forks of {base[:10]} until "
        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(st['round_end']))}")
    for slot in st["slots"]:
        launch(st, slot)
        time.sleep(20)         # do not start four sessions in the same second
    save_state(st)


def score_branch(agent: str) -> dict:
    sys.path.insert(0, str(REPO / "autoresearch"))
    import score as scorer
    d = AR_HOME / "runs" / agent
    runs = [p for p in d.iterdir() if p.is_dir()] if d.exists() else []
    res = scorer.best_in(runs) if runs else {"best": None, "n_evals": 0, "n_valid": 0}
    return res


def end_round(st, final=False) -> None:
    r = st["round"]
    log(f"ROUND {r} ENDS")
    for slot in st["slots"]:
        rel = box_py("release", "--agent", slot["agent"])
        log(f"  {slot['agent']}: {rel.stdout.strip()[-160:] or rel.stderr.strip()[-160:]}")
        sh(["tmux", "kill-window", "-t", f"{TMUX}:b{slot['k']}"], check=False)
    results = []
    for slot in st["slots"]:
        d = Path(slot["dir"])
        arch = AR_HOME / "archive" / f"r{r}" / slot["agent"]
        arch.mkdir(parents=True, exist_ok=True)
        for f in ("RESULT.json", "autoresearch/NOTES.md", "AR_BRANCH.json"):
            if (d / f).exists():
                shutil.copy2(d / f, arch / Path(f).name)
        # the branch as the agent left it: commit whatever it did not, so the winner's
        # workspace is complete when the next round forks it
        git("add", "-A", cwd=d, check=False)
        git("commit", "-q", "-m", f"autoresearch {slot['agent']}: round-end snapshot" + TRAILERS,
            cwd=d, check=False)
        head = git("rev-parse", "HEAD", cwd=d).stdout.strip()
        sc = score_branch(slot["agent"])
        best = sc.get("best") or {}
        results.append({"agent": slot["agent"], "branch": slot["branch"], "head": head,
                        "score": best.get("score"), "best": best,
                        "n_evals": sc.get("n_evals"), "n_valid": sc.get("n_valid")})
        (arch / "score.json").write_text(json.dumps(results[-1], indent=1), encoding="utf-8")
        log(f"  {slot['agent']}: score {best.get('score')} ({sc.get('n_valid')}/{sc.get('n_evals')} "
            f"valid evals) head {head[:10]}")
    scored = [x for x in results if x["score"] is not None]
    win = max(scored, key=lambda x: x["score"]) if scored else None
    st.setdefault("history", []).append({"round": r, "results": results,
                                         "winner": win["agent"] if win else None,
                                         "ended": time.time()})
    if win:
        git("push", "-q", "origin", f"{win['branch']}:{win['branch']}", cwd=REPO, check=False)
        inc = st.get("incumbent") or {}
        if win["score"] > float(inc.get("score") or -1e9):
            new = AR_HOME / "incumbent" / f"r{r}"
            new.mkdir(parents=True, exist_ok=True)
            ev = Path(win["best"]["file"])
            shutil.copy2(ev, new / ev.name)
            ck = ev.parent / "ckpt_latest.pt"
            for name in ("ckpt_final.pt", "ckpt_latest.pt"):
                if (ev.parent / name).exists():
                    ck = ev.parent / name
                    break
            if ck.exists():
                shutil.copy2(ck, new / "ckpt.pt")
                link = AR_HOME / "incumbent" / "ckpt.pt"
                link.unlink(missing_ok=True)
                link.symlink_to(new / "ckpt.pt")
            for f in ("progress.csv", "run.json"):
                if (ev.parent / f).exists():
                    shutil.copy2(ev.parent / f, new / f)
            st["incumbent"] = {"score": win["score"], "agent": win["agent"], "round": r,
                               "eval": str(new / ev.name), "dir": str(new),
                               "ckpt": str(new / "ckpt.pt") if ck.exists() else None,
                               "commit": win["head"], "best": win["best"]}
            log(f"  NEW INCUMBENT {win['agent']} score {win['score']}")
        st["next_base"] = win["head"]
        log(f"  WINNER {win['agent']} ({win['score']}): round {r + 1} forks {win['head'][:10]}")
    else:
        st["next_base"] = st["base"]
        log("  no branch produced a valid eval: the next round forks the same base")
    st["slots"] = []
    save_state(st)


def write_status(st) -> None:
    lines = [f"# autoresearch status - {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    if st.get("round"):
        left = (st["round_end"] - time.time()) / 3600
        lines.append(f"round {st['round']} of {CFG['rounds']}, {left:.1f} h left in it; "
                     f"base {st.get('base', '')[:10]}")
    inc = st.get("incumbent") or {}
    lines.append(f"incumbent: score {inc.get('score')} ({inc.get('agent')}, round "
                 f"{inc.get('round')}) {inc.get('eval', '')}")
    b = box_py("status")
    lines += ["", "```", b.stdout.strip()[-3000:], "```", ""]
    for slot in st.get("slots", []):
        res = Path(slot["dir"]) / "RESULT.json"
        best, summ = "-", ""
        if res.exists():
            try:
                rj = json.loads(res.read_text(encoding="utf-8"))
                best = (rj.get("best") or {}).get("score")
                summ = (rj.get("summary") or "").strip().splitlines()[:2]
            except (ValueError, OSError):
                pass
        lines.append(f"- {slot['agent']}: alive {session_alive(slot)}, restarts "
                     f"{slot['restarts']}, best {best}  {' / '.join(summ) if summ else ''}")
    for h in st.get("history", []):
        lines.append(f"round {h['round']}: winner {h['winner']}  "
                     + ", ".join(f"{x['agent']} {x['score']}" for x in h["results"]))
    (AR_HOME / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_run(a):
    AR_HOME.mkdir(parents=True, exist_ok=True)
    (AR_HOME / "fleet").mkdir(exist_ok=True)
    st = load_state()
    if st is None:
        base = git("rev-parse", a.base or "HEAD").stdout.strip()
        st = {"started": time.time(), "round": 0, "base": base, "slots": [], "history": [],
              "incumbent": json.loads(Path(a.incumbent).read_text(encoding="utf-8"))
              if a.incumbent else None, "next_base": base}
        save_state(st)
        log(f"autoresearch starts: base {base[:10]}, {CFG['rounds']} rounds x "
            f"{CFG['round_hours']} h x {CFG['branches_per_round']} branches")
    last_status = 0.0
    while True:
        if not st.get("slots"):
            if st["round"] >= CFG["rounds"] or (AR_HOME / "STOP").exists():
                log("all rounds done (or STOP): the orchestrator exits")
                write_status(st)
                return
            start_round(st, st["next_base"])
        now = time.time()
        for slot in st["slots"]:
            if not slot["wrapup_sent"] and now >= st["round_end"] - CFG["wrapup_minutes"] * 60:
                send(slot, f"ROUND OVER for {slot['agent']}: do the Wrap-up in "
                           "autoresearch/PROTOCOL.md now (pull the best run with --ckpt, "
                           "final RESULT.json, NOTES.md, commit and push, release your box).")
                slot["wrapup_sent"] = True
                log(f"wrap-up sent to {slot['agent']}")
            elif (not session_alive(slot) and now < st["round_end"] - 120
                  and now - slot["launched"] > 120):
                slot["restarts"] += 1
                log(f"{slot['agent']}: session gone - restart {slot['restarts']}")
                launch(st, slot, resume=True)
        save_state(st)
        if (AR_HOME / "WRAPUP").exists():
            (AR_HOME / "WRAPUP").unlink()
            st["round_end"] = min(st["round_end"], time.time() + CFG["wrapup_minutes"] * 60)
            log("WRAPUP requested: the round ends in wrapup_minutes")
            save_state(st)
        if now >= st["round_end"]:
            end_round(st)
        if now - last_status > 300:
            try:
                write_status(st)
            except Exception as e:     # noqa: BLE001 - status is cosmetic
                log(f"status failed: {e}")
            last_status = now
        time.sleep(POLL_S)


def cmd_status(a):
    st = load_state()
    if not st:
        print("no state")
        return
    write_status(st)
    print((AR_HOME / "status.md").read_text(encoding="utf-8"))


def cmd_wrapup(a):
    (AR_HOME / "WRAPUP").touch()
    print("the orchestrator ends the current round within wrapup_minutes")


def cmd_stop(a):
    (AR_HOME / "STOP").touch()
    st = load_state() or {}
    for slot in st.get("slots", []):
        print(box_py("release", "--agent", slot["agent"]).stdout.strip())
        sh(["tmux", "kill-window", "-t", f"{TMUX}:b{slot['k']}"], check=False)
    print("STOP written: every box released, every branch session ended; the orchestrator "
          "exits at its next poll. Remove ~/ar/STOP and run it again to resume.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--base", default=None, help="commit or branch round 1 forks (default HEAD)")
    p.add_argument("--incumbent", default=None, help="INCUMBENT.json of round 0")
    p.set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("wrapup").set_defaults(fn=cmd_wrapup)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
