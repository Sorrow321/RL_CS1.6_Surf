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
import re
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
          "On EVERY wakeup: re-read AR_BRANCH.json (if now >= wrapup_at, do only the Wrap-up; "
          "before wrapup_at never wrap up or stop this loop: stopping a run does not end the "
          "round, so if RESULT.json says final, set it back to false and start your next idea), "
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
    """The window runs `bash .ar_launch.sh; echo SESSION ENDED; sleep 86400`, so the pane shows
    `sleep` exactly when the session has exited. The process NAME of a live session is useless:
    the native installer's binary is ~/.local/share/claude/versions/<version>."""
    r = sh(["tmux", "list-panes", "-t", f"{TMUX}:b{slot['k']}", "-F",
            "#{pane_current_command}"], check=False)
    if r.returncode != 0 or not r.stdout.strip():
        return False
    return r.stdout.strip().splitlines()[0] != "sleep"


def send(slot, text: str) -> None:
    win = f"{TMUX}:b{slot['k']}"
    sh(["tmux", "send-keys", "-t", win, "-l", text], check=False)
    sh(["tmux", "send-keys", "-t", win, "Enter"], check=False)


EARLY_S = 1800      # RESULT.json final this long before wrapup_at is an early wrap-up


def tell_early_wrapup(slot, now: float, wrapup_at: float) -> bool:
    """Whether to re-arm a branch that wrapped up early, recording that it was told.

    r1b3 (2026-10-20) stopped a run at 07:57, took that for the end of its round, did the
    Wrap-up and stopped its loop, and its slot sat idle for the 3.9 h left before wrapup_at.
    A branch is told once each time its RESULT.json goes final more than EARLY_S before
    wrapup_at; RESULT.json back at final false re-arms the check. Only the branch's own file
    counts: a new round's fork starts with the winner's committed RESULT.json, final and all."""
    try:
        res = json.loads((Path(slot["dir"]) / "RESULT.json").read_text(encoding="utf-8"))
        final = res.get("final") is True and res.get("agent") == slot["agent"]
    except (OSError, ValueError, AttributeError):
        return False
    if not final:
        slot["early_told"] = False
        return False
    if slot.get("early_told") or now >= wrapup_at - EARLY_S:
        return False
    slot["early_told"] = True
    return True


DEADLINE_WARN_S = 1200   # a branch hears about its box's registry deadline this long before it


def box_deadline_note(slot, now: float, round_end: float):
    """The note for a branch whose box dies within DEADLINE_WARN_S, or None; records it.

    The fleet watchdog destroys a box at its registry deadline, and a branch box has no harvest
    spec, so a run not pulled by then is lost. On 2026-10-20 r1b2's box was due at 08:47, 17 min
    into a 1e9-step run, and r1b3's at 11:50:38, the wrap-up minute; an operator's note saved
    both. A deadline is announced once, and an extension (a new deadline) re-arms it. A deadline
    after round_end needs no note: end_round releases the box anyway."""
    try:
        reg = json.loads((AR_HOME / "fleet" / "fleet.json").read_text(encoding="utf-8"))
        mine = sorted((r["deadline"], iid) for iid, r in reg.items()
                      if isinstance(r, dict) and r.get("label") == slot["agent"]
                      and isinstance(r.get("deadline"), (int, float)))
    except (OSError, ValueError, AttributeError):
        return None
    if not mine:
        return None
    dl, iid = mine[0]
    if dl >= round_end or not dl - DEADLINE_WARN_S <= now < dl or slot.get("deadline_told") == dl:
        return None
    slot["deadline_told"] = dl
    return (f"[orchestrator] Box deadline: your box {iid} expires at "
            f"{time.strftime('%H:%M', time.gmtime(dl))} UTC, {(dl - now) / 60:.0f} min from now. "
            "The fleet watchdog destroys it then, and whatever is not pulled dies with it. Extend "
            "it (box.py extend) if a run should keep training, or pull and release it. (A harness "
            "note, not a research instruction.)")


STALL_S = 75 * 60        # a wakeup is at most 60 min away, so a live loop is never this quiet
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


def session_quiet_s(slot, now: float):
    """Seconds since the branch's Claude session last wrote its transcript, or None."""
    proj = CLAUDE_PROJECTS / re.sub(r"[^A-Za-z0-9]", "-", str(slot["dir"]))
    try:
        newest = max(p.stat().st_mtime for p in proj.glob("*.jsonl"))
    except (OSError, ValueError):            # no project dir or no transcript yet
        return None
    return now - newest


def tell_stalled(slot, now: float, wrapup_at: float) -> bool:
    """Whether to re-arm a branch whose session went quiet, recording that it was told.

    A loop can die without a final RESULT.json: a wakeup that fires while the account is at its
    usage limit does not run (on 2026-09-04 a rate limit stopped every agent and its timers for
    50 min), or an agent stops scheduling. tmux still sees the session, so neither the restart
    nor the early wrap-up check fires. Told at most once per STALL_S, and never in a session's
    first STALL_S (a new round's session shares its project dir with the last round's)."""
    quiet = session_quiet_s(slot, now)
    if quiet is None or quiet < STALL_S or now >= wrapup_at:
        return False
    if now - slot.get("stall_told", 0) < STALL_S or now - slot.get("launched", 0) < STALL_S:
        return False
    slot["stall_told"] = now
    return True


def box_py(*args, env=None):
    e = dict(os.environ, AR_HOME=str(AR_HOME), RLSURF_FLEET_DIR=str(AR_HOME / "fleet"),
             PYTHONIOENCODING="utf-8", **(env or {}))
    return subprocess.run([sys.executable, str(REPO / "autoresearch" / "box.py"), *args],
                          capture_output=True, text=True, env=e, timeout=900)


# ----------------------------------------------------------------- harness ----
def slot_blocklists() -> list:
    return sorted((AR_HOME / "wt").glob("b*/tools/bad_hosts.json"))


def _box():
    p = str(REPO / "autoresearch")
    if p not in sys.path:
        sys.path.insert(0, p)
    import box
    return box


def sync_blocklists() -> int:
    """Every branch's tools/bad_hosts.json (and the main checkout's) into the fleet's shared
    $AR_HOME/bad_hosts.json, which box.py reads on top of the branch's own list."""
    return _box().sync_shared_blocklist([REPO / "tools" / "bad_hosts.json", *slot_blocklists()])


def harness_base(base: str, repo: Path = REPO, blocklists=(), since: str | None = None) -> str:
    """The commit a round forks: `base` (the winner's head) with config.harness_files restored
    from the main checkout's HEAD and the fleet's blocklist folded into tools/bad_hosts.json.

    The harness is the operator's, as in the paper: a branch cannot carry its own caps,
    grader or rules into the next round, and the operator's fixes reach every round instead
    of stopping at the round base they were not in. Git plumbing on a private index, so no
    working tree is touched; returns `base` itself when nothing differs.

    config.lineage_keys of config.json are the lineage's, not the operator's: the winner's
    value travels with it when the winner changed it since `since` (the round base it forked
    from). r1b1 made its scratch recipe the incumbent_recipe, and restoring the operator's
    config.json would have silently dropped those flags from every round-2 scratch launch."""
    box = _box()
    head = git("rev-parse", "HEAD", cwd=repo).stdout.strip()
    AR_HOME.mkdir(parents=True, exist_ok=True)
    idx = AR_HOME / f".harness.{os.getpid()}.index"
    env = dict(os.environ, GIT_INDEX_FILE=str(idx))

    def g(*args, check=True, stdin=None):
        r = subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True,
                           input=stdin, timeout=120)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[-300:]}")
        return r.stdout

    def entry(rev, path):        # (mode, blob) of path at rev, or None
        out = g("ls-tree", rev, "--", path).strip()
        return tuple(out.split("\t")[0].split()[::2]) if out else None

    try:
        g("read-tree", base)
        changed = []
        for path in CFG.get("harness_files", []):
            mine = entry(head, path)
            if mine is None or mine == entry(base, path):
                continue
            g("update-index", "--add", "--cacheinfo", f"{mine[0]},{mine[1]},{path}")
            changed.append(path)
        cfg = "autoresearch/config.json"
        if cfg in changed and CFG.get("lineage_keys"):
            def load(rev):
                try:
                    return json.loads(g("show", f"{rev}:{cfg}", check=False) or "{}")
                except ValueError:
                    return {}
            ours, theirs = load(head), load(base)
            was = load(since) if since else ours
            kept = [k for k in CFG["lineage_keys"]
                    if k in theirs and theirs[k] != was.get(k) and theirs[k] != ours.get(k)]
            if kept:
                ours.update({k: theirs[k] for k in kept})
                blob = g("hash-object", "-w", "--stdin",
                         stdin=json.dumps(ours, indent=2) + "\n").strip()
                g("update-index", "--add", "--cacheinfo", f"100644,{blob},{cfg}")
                if (entry(base, cfg) or ("", ""))[1] == blob:
                    changed.remove(cfg)              # that is the winner's config.json as is
                else:
                    changed[changed.index(cfg)] = f"{cfg} (the lineage's {', '.join(kept)} kept)"
        if blocklists:
            bh = "tools/bad_hosts.json"
            have = g("show", f"{base}:{bh}", check=False)
            try:
                theirs = [json.loads(have)] if have else []
            except ValueError:
                theirs = []
            text = json.dumps(box.merged_blocklist([*theirs, *blocklists]), indent=2) + "\n"
            if text != have:
                blob = g("hash-object", "-w", "--stdin", stdin=text).strip()
                g("update-index", "--add", "--cacheinfo", f"100644,{blob},{bh}")
                changed.append(bh)
        if not changed:
            return base
        tree = g("write-tree").strip()
        msg = (f"autoresearch: the operator's harness at {head[:10]} over {base[:10]}\n\n"
               + "\n".join(f"* {p}" for p in changed) + TRAILERS)
        return g("commit-tree", tree, "-p", base, "-m", msg).strip()
    finally:
        idx.unlink(missing_ok=True)


# ------------------------------------------------------------------ rounds ----
def start_round(st, base: str) -> None:
    try:
        sync_blocklists()
        fork = harness_base(base, blocklists=[AR_HOME / "bad_hosts.json"], since=st.get("base"))
    except Exception as e:      # noqa: BLE001 - the round still starts, on the plain base
        log(f"!! harness overlay failed ({e}); the round forks {base[:10]} as it is")
        fork = base
    if fork != base:
        log(f"harness overlay: {base[:10]} -> {fork[:10]}")
    st["round"] += 1
    st["round_start"] = time.time()
    st["round_end"] = st["round_start"] + CFG["round_hours"] * 3600
    st["base"] = fork
    st["slots"] = [prepare_slot(st, k, fork) for k in range(1, CFG["branches_per_round"] + 1)]
    save_state(st)
    log(f"ROUND {st['round']} of {CFG['rounds']}: {len(st['slots'])} forks of {base[:10]} until "
        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(st['round_end']))}")
    for slot in st["slots"]:
        launch(st, slot)
        time.sleep(20)         # do not start four sessions in the same second
    save_state(st)


def ckpt_step(path: Path):
    """The training step a checkpoint was saved at (its global_step), or None. The file came
    from a rented box, so it is never unpickled (and torch.load's weights_only mode refuses
    the trainer's checkpoints anyway, which hold numpy arrays): the step is the integer that
    follows the top-level "global_step" key in the pickle's opcode stream. Nothing executes."""
    import pickletools
    import zipfile
    ints = ("BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT", "LONG")
    try:
        with zipfile.ZipFile(path) as z:
            data = z.read(next(n for n in z.namelist() if n.split("/")[-1] == "data.pkl"))
        after_key = False
        for op, arg, _ in pickletools.genops(data):
            if after_key and op.name in ("BINPUT", "LONG_BINPUT", "MEMOIZE"):
                continue
            if after_key:
                return int(arg) if op.name in ints else None
            after_key = "UNICODE" in op.name and arg == "global_step"
    except Exception as e:                       # noqa: BLE001 - informational only
        log(f"  (cannot read the step of {path}: {type(e).__name__}: {str(e)[:120]})")
    return None


def ckpt_complete(path: Path) -> bool:
    """Every member of the checkpoint's zip is there and matches its CRC, and it names its
    step. A pull that timed out or dropped leaves a truncated file under the real name; this
    catches it without unpickling anything."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                return False
    except Exception:                            # noqa: BLE001 - any failure means incomplete
        return False
    return ckpt_step(path) is not None


def handon_ckpt(run_dir: Path, eval_step=None):
    """The checkpoint a run hands on: ckpt_final.pt, else ckpt_latest.pt, else its newest
    ckpt_<step>.pt milestone, whichever is the first COMPLETE one; None if none is.

    With eval_step (the step of the run's best eval), the newest complete milestone AT OR
    BEFORE that eval is handed on instead, unless the checkpoint above is more than twice as
    near the eval: those are the weights whose training led to the best eval, where the
    newest ones may have trained on past it. r7b2_bel (2026-10-23) finished once at
    25,448,939,520 inside a transient peak (greedy chamber entries 5.6% -> 1.9% -> 1.2% in the
    next 36-eval blocks, no other finish in 870M steps). Its snapshot 54M steps before the
    finish, resumed, finished again within 85M; its ckpt_latest was ~700M steps past it."""
    milestones = sorted(((int(m.group(1)), p) for p in run_dir.glob("ckpt_*.pt")
                         if (m := re.fullmatch(r"ckpt_(\d+)\.pt", p.name))), reverse=True)
    newest = None
    for p in [run_dir / "ckpt_final.pt", run_dir / "ckpt_latest.pt"] + [p for _, p in milestones]:
        if not p.exists():
            continue
        if ckpt_complete(p):
            newest = p
            break
        log(f"  !! {run_dir.name}/{p.name} is truncated or damaged: not handed on")
    if newest is None or eval_step is None:
        return newest
    n = ckpt_step(newest)
    if n is None or n <= eval_step:
        return newest
    for s, p in milestones:
        if s <= eval_step and p != newest and ckpt_complete(p):
            return p if eval_step - s <= 2 * (n - eval_step) else newest
    return newest


def score_branch(agent: str) -> dict:
    sys.path.insert(0, str(REPO / "autoresearch"))
    import score as scorer
    d = AR_HOME / "runs" / agent
    runs = [p for p in d.iterdir() if p.is_dir()] if d.exists() else []
    res = scorer.best_in(runs) if runs else {"best": None, "n_evals": 0, "n_valid": 0}
    return res


def incumbent_score(inc: dict, scorer=None):
    """The incumbent's score under the CURRENT grader: its stored eval re-scored, so a grader
    change between rounds (round 7: the no-corner-cut field, 9 episodes) compares like with
    like. The stored score when the eval is gone or no longer valid."""
    ev = inc.get("eval")
    if ev and Path(ev).exists():
        if scorer is None:
            sys.path.insert(0, str(REPO / "autoresearch"))
            import score as scorer
        s = scorer.score_file(Path(ev)).get("score")
        if s is not None:
            return s
    return inc.get("score")


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
        inc_now = incumbent_score(inc)
        if inc and inc_now != inc.get("score"):
            log(f"  incumbent {inc.get('agent')} re-scored with the current grader: "
                f"{inc.get('score')} -> {inc_now}")
            inc["score_before"], inc["score"] = inc.get("score"), inc_now
        if win["score"] > float(inc_now if inc_now is not None else -1e9):
            new = AR_HOME / "incumbent" / f"r{r}"
            new.mkdir(parents=True, exist_ok=True)
            ev = Path(win["best"]["file"])
            shutil.copy2(ev, new / ev.name)
            eval_step = int(ev.stem.split("_")[1])
            ck = handon_ckpt(ev.parent, eval_step)
            if ck is None:
                log(f"  !! no complete checkpoint in {ev.parent}: incumbent/ckpt.pt stays the "
                    "previous round's")
            else:
                shutil.copy2(ck, new / "ckpt.pt")
                link = AR_HOME / "incumbent" / "ckpt.pt"
                link.unlink(missing_ok=True)
                link.symlink_to(new / "ckpt.pt")
            for f in ("progress.csv", "run.json"):
                if (ev.parent / f).exists():
                    shutil.copy2(ev.parent / f, new / f)
            # the checkpoint handed on is the snapshot at or before the best eval when the run
            # has one (handon_ckpt), else its NEWEST one, usually later than the best eval
            # (r1b1_ratnovx: best eval 1,633M, ckpt_latest ~2.5B): say both steps, so the
            # next round does not expect the checkpoint to reproduce the score
            if ck is not None:
                ck_path, ck_from = str(new / "ckpt.pt"), f"{ev.parent.name}/{ck.name}"
            else:
                ck_path = inc.get("ckpt")
                ck_from = f"the previous incumbent's (no complete checkpoint in {ev.parent.name})"
            ck_step = ckpt_step(Path(ck_path)) if ck_path else None
            st["incumbent"] = {"score": win["score"], "agent": win["agent"], "round": r,
                               "eval": str(new / ev.name), "dir": str(new),
                               "ckpt": ck_path, "eval_step": eval_step,
                               "ckpt_from": ck_from, "ckpt_step": ck_step,
                               "commit": win["head"], "best": win["best"]}
            log(f"  NEW INCUMBENT {win['agent']} score {win['score']}: eval at step "
                f"{eval_step:,}; ckpt {ck_from} at step "
                f"{f'{ck_step:,}' if ck_step is not None else 'unknown'}")
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
        wrapup_at = st["round_end"] - CFG["wrapup_minutes"] * 60
        for slot in st["slots"]:
            prompt = PROMPT.format(agent=slot["agent"], round=st["round"], rounds=CFG["rounds"])
            if not slot["wrapup_sent"] and now >= wrapup_at:
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
                continue
            elif tell_early_wrapup(slot, now, wrapup_at):
                send(slot, prompt)
                log(f"{slot['agent']}: RESULT.json final {(wrapup_at - now) / 3600:.1f} h "
                    "before wrapup_at - its loop re-armed (the round is not over)")
            elif tell_stalled(slot, now, wrapup_at):
                send(slot, prompt)
                log(f"{slot['agent']}: session quiet {session_quiet_s(slot, now) / 60:.0f} min "
                    "before wrapup_at - its loop re-armed")
            try:                       # advisory: a failure here must not stop the round
                note = box_deadline_note(slot, now, st["round_end"])
                if note:
                    send(slot, note)
                    log(f"{slot['agent']}: told its box dies within {DEADLINE_WARN_S // 60} min")
            except Exception as e:     # noqa: BLE001
                log(f"box deadline check failed for {slot['agent']}: {e}")
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
            try:
                sync_blocklists()
            except Exception as e:     # noqa: BLE001 - the next pass retries
                log(f"blocklist sync failed: {e}")
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
