#!/usr/bin/env python3
"""xLOOP driver: self-improving spine loop on the local 5090.

Round k = 0..N-1, each a FRESH from-scratch train_fast (no --ckpt, so
weights AND optimizer are re-initialised) of ROUND_STEPS steps through
the canonical tools/run_arm.sh SCRATCH branch, map pointed at the MAIN
checkout (worktree-bake rule).

  round 0     : plain from-scratch, no demo file.
  end of each : record 20 GREEDY map-start evals from that round's final
                checkpoint (tools/record_ckpt.py --dump-states), pick the
                episode reaching the MINIMUM geodesic d, trim its fall
                (pick_selfline contact_cut, via tools/loop_spine.py), and
                write the surviving prefix as a spine .npy.
  round k+1   : every spawn drawn uniformly from that spine
                (--demo-file/--demo-window/--demo-rate 2.0/--demo-min-ep
                1e9/--respawn-frac 1.0 - the xDEMO50-validated set).

Artifacts per round in runs/<NAME>/round_<k>/: progress.csv, the eval
trajs + state dump, the chosen spine, pick.json. The machine-readable
result is runs/<NAME>/loop_summary.jsonl - one line per round.

A round is bounded by STEPS, with a wall-clock safety cap and the
csv-stale kill as backstops. Any failure to launch, train, record or
spine STOPS the loop: a skipped round would silently break the chain.

Env overrides (smoke): XLOOP_NAME, XLOOP_ROUNDS, XLOOP_STEPS,
XLOOP_EPISODES, XLOOP_EP_TICKS, XLOOP_WALL, XLOOP_CKPT_EVERY.
"""
import json
import os
import subprocess
import sys
import time

import psutil

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MAP = "C:/RL_Surf/maps/surf_src_cannonball.bsp"

NAME = os.environ.get("XLOOP_NAME", "xLOOP")
ROUNDS = int(os.environ.get("XLOOP_ROUNDS", "50"))
STEPS = os.environ.get("XLOOP_STEPS", "1e9")
EPISODES = os.environ.get("XLOOP_EPISODES", "20")
EP_TICKS = os.environ.get("XLOOP_EP_TICKS", "")      # "" = the ckpt's own
WALL = float(os.environ.get("XLOOP_WALL", "5400"))   # 90 min safety cap
CKPT_EVERY = os.environ.get("XLOOP_CKPT_EVERY", "")  # "" = run_arm default
# in-round eval cadence. run_arm's 75e6 default costs ~13 evals of 9
# episodes per 1e9-step round, which is real wall clock the loop does not
# need - the round's verdict comes from the 20 greedy evals at the END.
RECORD_EVERY = os.environ.get("XLOOP_RECORD_EVERY", "")
STALL_S = 900
GRACE_S = 1200

RUNROOT = os.path.join(WT, "runs", NAME)
os.makedirs(RUNROOT, exist_ok=True)
LOGF = open(os.path.join(RUNROOT, "driver.log"), "a", buffering=1,
            encoding="ascii", errors="replace")


def make_shim():
    """run_arm.sh invokes `python3`, which Windows python installs do not
    provide. Generate a one-line bash shim beside the run artifacts and put
    it first on PATH, so the committed driver depends on no session
    scratchpad."""
    d = os.path.join(RUNROOT, "bin")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "python3")
    with open(p, "w", encoding="ascii", newline="\n") as fh:
        fh.write('#!/bin/bash\nexec "%s" "$@"\n' % PY.replace("\\", "/"))
    os.chmod(p, 0o755)
    return d


SHIM = make_shim()


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    LOGF.write(line + "\n")


def find_bash():
    for p in (r"C:\Program Files\Git\bin\bash.exe",
              r"C:\Program Files\Git\usr\bin\bash.exe"):
        if os.path.exists(p):
            return p
    raise RuntimeError("git bash not found")


def find_trainer(run):
    """Processes running train_fast.py for exactly this --run value."""
    me = os.getpid()
    out = []
    for p in psutil.process_iter(["cmdline"]):
        try:
            cl = p.info["cmdline"] or []
            if (p.pid != me
                    and any("train_fast.py" in str(c) for c in cl)
                    and any(str(c) == run for c in cl)):
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def kill_all(procs):
    tokill = []
    for p in procs:
        try:
            tokill += p.children(recursive=True)
        except psutil.Error:
            pass
        tokill.append(p)
    for p in tokill:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(tokill, timeout=20)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass


def csv_stats(rundir):
    """(last step, best eval_progress, rows) from a round's progress.csv."""
    import csv as _csv
    f = os.path.join(rundir, "progress.csv")
    try:
        with open(f, encoding="utf-8", errors="replace") as fh:
            rows = list(_csv.DictReader(fh))
    except OSError:
        return None, None, 0
    if not rows:
        return None, None, 0
    step = int(rows[-1].get("time/total_timesteps") or 0)
    ev = [float(r["race/eval_progress"]) for r in rows
          if r.get("race/eval_progress")]
    return step, (max(ev) if ev else None), len(rows)


def run(cmd, timeout, tag):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        r = subprocess.run(cmd, cwd=WT, env=env, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"{tag}: TIMEOUT after {timeout}s")
        return None
    if r.returncode != 0:
        log(f"{tag}: rc={r.returncode}; stderr tail: "
            + (r.stderr or "")[-400:].replace("\n", " | "))
    for ln in (r.stdout or "").strip().splitlines()[-6:]:
        log(f"{tag}> {ln[:200]}")
    return r


def train_round(k, spine, spine_len):
    """Launch one round and wait for it to finish. -> run dir, or None."""
    run_name = f"{NAME}/round_{k}"
    rundir = os.path.join(WT, "runs", NAME, f"round_{k}")
    flags = ["--seed", "1"]
    if k > 0:
        flags += ["--demo-file", spine,
                  "--demo-window", str(spine_len),
                  "--demo-rate", "2.0",
                  "--demo-min-ep", "1e9",
                  "--respawn-frac", "1.0"]
    if EP_TICKS:
        flags += ["--ep-ticks", EP_TICKS]
    if CKPT_EVERY:
        flags += ["--ckpt-every", CKPT_EVERY]
    if RECORD_EVERY:
        # run_arm.sh puts its own --record-every in ARGS and appends "$@"
        # after it; argparse takes the LAST occurrence, so this overrides.
        flags += ["--record-every", RECORD_EVERY]

    env = dict(os.environ)
    env["PATH"] = SHIM + os.pathsep + env.get("PATH", "")
    env["SCRATCH"] = "1"
    env["MAP"] = MAP
    env["BUDGET"] = STEPS
    env["PYTHONIOENCODING"] = "utf-8"

    if find_trainer(run_name):
        log(f"r{k}: stale trainer for {run_name}; killing first")
        kill_all(find_trainer(run_name))
        time.sleep(10)
    log(f"r{k}: launching {run_name} steps={STEPS} "
        f"{'spine=' + os.path.basename(spine) if k else '(no spine)'}")
    try:
        r = subprocess.run([find_bash(), "tools/run_arm.sh", run_name]
                           + flags, cwd=WT, env=env, capture_output=True,
                           text=True, timeout=900)
        rc = r.returncode
        for ln in (r.stdout or "").strip().splitlines()[-3:]:
            log(f"r{k}: launcher> {ln[:200]}")
    except subprocess.TimeoutExpired:
        log(f"r{k}: launcher timed out")
        kill_all(find_trainer(run_name))
        return None

    t0 = time.time()
    csvf = os.path.join(rundir, "progress.csv")
    while True:
        procs = find_trainer(run_name)
        now = time.time()
        if not procs:
            break                       # finished its --steps (or died)
        if now - t0 > WALL:
            kill_all(procs)
            log(f"r{k}: WALL {WALL:.0f}s hit, killed")
            break
        try:
            stale = now - os.path.getmtime(csvf)
        except OSError:
            stale = 0.0 if now - t0 < GRACE_S else now - t0
        if now - t0 > GRACE_S and stale > STALL_S:
            kill_all(procs)
            log(f"r{k}: STALLED (csv stale {stale:.0f}s), killed")
            break
        if int(now - t0) % 900 < 60:
            st, bev, nr = csv_stats(rundir)
            log(f"r{k}: t+{(now - t0) / 60:5.1f}min step={st} "
                f"best_eval={bev} rows={nr}")
        time.sleep(60)

    ck = None
    for cand in ("ckpt_final.pt", "ckpt_latest.pt"):
        p = os.path.join(rundir, cand)
        if os.path.exists(p):
            ck = p
            break
    if ck is None:
        # a nonzero launcher rc with no checkpoint is a real launch failure;
        # rc!=0 WITH a checkpoint just means a short round outran the
        # launcher's own 90 s liveness probe
        log(f"r{k}: NO CHECKPOINT in {rundir} (launcher rc={rc}) - stopping")
        return None
    if rc != 0:
        log(f"r{k}: launcher rc={rc} but {os.path.basename(ck)} exists "
            f"(short round outran the liveness probe) - continuing")
    return rundir, ck


def main():
    log(f"=== {NAME} loop start: {ROUNDS} rounds x {STEPS} steps, "
        f"episodes={EPISODES} wall={WALL:.0f}s")
    os.makedirs(os.path.join(WT, "runs", NAME), exist_ok=True)
    summary_path = os.path.join(WT, "runs", NAME, "loop_summary.jsonl")
    spine, spine_len = None, 0

    for k in range(ROUNDS):
        t_round = time.time()
        # RESUME: a round whose spine already exists is complete - adopt it
        # and move on. A 50-round job is ~30 h of wall clock, so a restart
        # (crash, or a knob change like --record-every) must not redo work
        # that is already on disk. The spine IS the round's output.
        done_pick = os.path.join(WT, "runs", NAME, f"round_{k}", "pick.json")
        done_spine = os.path.join(WT, "runs", NAME, f"round_{k}", "spine.npy")
        if os.path.exists(done_pick) and os.path.exists(done_spine):
            with open(done_pick, encoding="utf-8") as fh:
                info = json.load(fh)
            spine, spine_len = done_spine, int(info.get("spine_len", 0))
            log(f"r{k}: RESUME - already complete "
                f"(spine_len={spine_len}); skipping")
            continue
        got = train_round(k, spine, spine_len)
        if got is None:
            log(f"r{k}: TRAIN FAILED - loop STOPPED")
            break
        rundir, ck = got
        step, best_eval, nrows = csv_stats(rundir)
        log(f"r{k}: trained to step={step} best_eval={best_eval} "
            f"rows={nrows}, ckpt={os.path.basename(ck)}")

        traj = os.path.join(rundir, "evals.jsonl")
        states = os.path.join(rundir, "evals_states.npz")
        cmd = [PY, "-u", "tools/record_ckpt.py", ck, "--map", MAP,
               "--episodes", EPISODES, "--out", traj,
               "--dump-states", states]
        if EP_TICKS:
            cmd += ["--ep-ticks", EP_TICKS]
        r = run(cmd, 5400, f"r{k}:record")
        if r is None or r.returncode != 0 or not os.path.exists(states):
            log(f"r{k}: RECORD FAILED - loop STOPPED")
            break

        nxt = os.path.join(rundir, "spine.npy")
        pick = os.path.join(rundir, "pick.json")
        r = run([PY, "-u", "tools/loop_spine.py", "--states", states,
                 "--traj", traj, "--ckpt", ck, "--map", MAP,
                 "--out", nxt, "--summary-out", pick], 1800, f"r{k}:spine")
        if r is None or r.returncode != 0 or not os.path.exists(pick):
            log(f"r{k}: SPINE FAILED - loop STOPPED")
            break
        with open(pick, encoding="utf-8") as fh:
            info = json.load(fh)
        if int(info.get("spine_len", 0)) < 1:
            log(f"r{k}: EMPTY SPINE - loop STOPPED")
            break

        row = {"round": k, "run": f"{NAME}/round_{k}", "steps": step,
               "csv_rows": nrows, "best_eval_progress": best_eval,
               "wall_s": round(time.time() - t_round, 1),
               "spawned_from": ("map start" if k == 0
                                else f"round_{k - 1} spine"),
               "prev_spine_len": spine_len, **info}
        with open(summary_path, "a", encoding="ascii", errors="replace") as fh:
            fh.write(json.dumps(row) + "\n")
        log(f"r{k}: SUMMARY min_d={info['chosen_min_d']:.0f} "
            f"corridor={info['chosen_corridor']:.0f} "
            f"finished={info['chosen_finished']} "
            f"spine_len={info['spine_len']} "
            f"dropped={info['trim_ticks_dropped']}t")
        spine, spine_len = nxt, int(info["spine_len"])

    log(f"=== {NAME} loop finished")
    with open(os.path.join(RUNROOT, "DONE"), "w") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log(f"DRIVER CRASH: {type(ex).__name__}: {ex}")
        raise
