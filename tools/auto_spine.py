#!/usr/bin/env python3
"""auto_spine.py - the next curriculum spine from the policy's OWN search (pbr3b4, 2026-10-24).

One step of the search -> curriculum loop that carried surf_princessburglar_b26 from 3.2% to
20.4% in round 2 (autoresearch/NOTES.md, "The method so far"), written down as a rule a trainer
can run by itself (train_fast --auto-spine) instead of an agent choosing cut points by eye:

1. BEAM: tools/beam_tas.py from the TRUE start on the newest weights - N stochastic copies of
   the policy, a greedy floor, the leaders by goal-field d cloned every 25 decisions,
   --save-frontier replays the frontier lineage open-loop into per-tick states.
2. CUT: the line ends at its best d (--cut best, the default; the --gain margin over the
   greedy frontier is what keeps a fatal fall from counting), or (--cut contact) at its best d
   at or before the LAST TICK THE MAP PUSHED BACK (tools/pick_selfline.py's contact_cut). The
   contact trim was the first default and failed on the gate it was built for: the lines that
   pass princessburglar's 20.4% catch make their last contact at the ramp landing (20.40%, level
   with the catch fall) and fly the rest (a U-turn climb to 21.4%) ballistically.
3. GO-EXPLORE when the beam does not beat the greedy frontier by --gain: tools/explore_phase1.py
   --exact-chains rooted at the line's own last --roots-secs seconds before its cut,
   tools/ge_replay.py replays the deepest chain exactly, the chain is spliced onto the line at
   its root and its views are put along the velocity (tools/spine_canon_view.py: the uniform
   explorer's views are arbitrary); then the same CUT.
4. The spine is written only if its end beats --greedy-d (the greedy eval's frontier) by
   --gain; the last stdout line is ``AUTO_SPINE {json}`` either way.

No constant here comes from a map: every threshold is a field distance margin, a time, or a
search budget, and the same flags run on any map. No demo or record data enters (CLAUDE.md
sections 0 and 0b): the beam's proposals are the policy's own, the explorer's actions are
uniform.

    python tools/auto_spine.py --ckpt runs/X/ckpt_latest.pt --map maps/M.bsp \\
        --field maps/M.goalc_32.npz --out runs/X/auto/k3 --greedy-d 45500 --beam-ticks 3200

    python tools/auto_spine.py --selftest          # the cut rules on synthetic lines
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

TICKS_PER_S = 100          # 10 ms; ckpt_tick() replaces it with the checkpoint's own tick


def ckpt_tick(ckpt) -> float:
    """The physics tick (ms) a checkpoint's weights were trained at: its config's realised
    mean tick (``tick_ms_eff``, under --tick-ms-schedule the ramp's current pattern), else
    ``tick_ms``, else 10. beam_tas runs that tick by itself (physics parity); the explorer is
    told it with --tick-ms, rounded to whole ms (explore_phase1's exact-chain rule)."""
    import torch
    try:
        c = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except Exception:                                    # noqa: BLE001
        return 10.0
    cf = c.get("config") or {}
    for k in ("tick_ms_eff", "tick_ms"):
        if cf.get(k) is not None:
            return float(cf[k])
    return 10.0


def load_field(path):
    from surfgym.goalfield import GoalField
    z = np.load(path, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def field_d(field, states) -> np.ndarray:
    """d of each state's origin on the field; off-field (the unreachable sentinel) -> inf."""
    o = np.asarray(states["origin"], np.float64).reshape(-1, 3)
    d = np.asarray(field.sample(o), np.float64)
    return np.where(d < field.reach_max - 0.5 * field.cell, d, np.inf)


def contact_cut_states(states, tol: float = 1.0) -> int:
    """The index of the last state the map pushed back on (pick_selfline.contact_cut on the
    states' vertical velocity: between contacts vz falls by one gravity step per tick)."""
    from pick_selfline import contact_cut
    v = np.asarray(states["velocity"], np.float64).reshape(-1, 3)
    rows = np.zeros((len(v), 8))
    rows[:, 4:7] = v
    return int(contact_cut(rows, tol)[0])


def cut_line(states, d, tol: float = 1.0, mode: str = "contact") -> tuple[int, float]:
    """-> (index of the line's end, its d). ``contact``: the best d at or before the last
    contact (a line that ends in a fatal fall ends where the track ended). ``best``: the best d
    anywhere - the --gain margin over the greedy frontier is then what keeps a fall from
    counting; on princessburglar's 20.4% gate the passing lines' last contact IS the ramp
    landing (20.40%, level with the catch fall's 20.41%) and everything that makes them
    passes (the U-turn climb to 21.4%) is ballistic after it (pbr3b4, 2026-10-24)."""
    if len(states) == 0:
        return -1, float("inf")
    if mode == "best":
        e = int(np.argmin(d))
        return e, float(d[e])
    c = contact_cut_states(states, tol)
    e = int(np.argmin(d[:c + 1]))
    return e, float(d[e])


def clean(states):
    out = np.array(states, copy=True)
    out["tick"] = 0
    out["stuck_ticks"] = 0
    return out


def run(cmd, log, timeout):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        try:
            r = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=str(ROOT), timeout=timeout)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    print(f"  exit {rc} after {time.time() - t0:.0f}s (log {log})", flush=True)
    return rc


def beam(a, out: Path, branch_tick: int = 0, tag: str = None):
    """beam_tas from the true start; ``branch_tick`` > 0 forks half the population there
    with held VIEW offsets (--branch-at, --branch-view-only: the policy's own keys) for
    --branch-burst-secs, protected --branch-protect generations - a different manoeuvre at the
    place the greedy line commits to its death, which the policy's own narrow proposals
    (a collapsed yaw sigma) never try."""
    tag = tag or ("beam" if branch_tick <= 0 else "beam_br")
    bd = out / tag
    bd.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", ROOT / "tools" / "beam_tas.py", a.ckpt, "--map", a.map,
           "--envs", a.envs, "--resample-every", 25, "--score", "d",
           "--greedy-envs", a.greedy_envs, "--allow-nonfinisher", "--save-frontier",
           "--skip-gate", "--route-file", "none", "--max-ticks", a.beam_ticks,
           "--torch-seed", a.seed, "--log-every", 5, "--out-dir", bd]
    if branch_tick > 0:
        cmd += ["--branch-at", f"t{int(branch_tick)}:{int(a.envs) // 2}",
                "--branch-burst", int(round(a.branch_burst_secs * TICKS_PER_S)),
                "--branch-hold", int(round(a.branch_hold_secs * TICKS_PER_S)),
                "--branch-jitter", int(round(a.branch_jitter)),
                "--branch-protect", int(a.branch_protect),
                "--branch-view-only"]
    rc = run(cmd, out / f"{tag}.log", a.timeout)
    f = bd / "beam_frontier.npz"
    if not f.exists():
        return None, {f"{tag}_rc": rc}
    z = np.load(f, allow_pickle=False)
    if "acts" in z.files and "act_every" in z.files:
        KEYS[tag] = keys_from_acts(z["acts"], int(z["act_every"]), len(z["states"]))
    return z["states"], {f"{tag}_rc": rc, f"{tag}_replay_ok": int(z["replay_ok"]),
                         f"{tag}_frontier_d": float(z["frontier_d"]),
                         f"{tag}_frontier_tick": int(z["frontier_tick"])}


# the held keys (fwd, side, duck; engine bins) each line's states were reached with, by line
# tag (beam / beam_br* / ge*): written next to the spine for train_fast --demo-keys, because a
# spine spawn that resets the held keys starts another state than the line's own (round 4)
KEYS: dict = {}
NEUTRAL_KEYS = (1, 1, 0)


def keys_from_acts(acts, k: int, n: int) -> np.ndarray:
    """(n, 3) held keys of a line's n states from its per-DECISION engine action table (k ticks a
    decision): state t was reached under decision (t-1)//k's keys; state 0 holds neutral."""
    acts = np.asarray(acts, np.int64)
    out = np.empty((n, 3), np.int32)
    out[:] = NEUTRAL_KEYS
    for t in range(1, n):
        a = acts[min((t - 1) // int(k), len(acts) - 1)]
        out[t] = (a[2], a[3], a[5])
    return out


def chain_keys(archive, field, n_states: int, cell: int = None):
    """(n_states - 1, 3) held keys of ge_replay's chain past its root: the per-tick link actions of
    every cell on the chain (explore_phase1 --exact-chains), as ge_replay walks it. ``cell`` = the
    chain's end (ge_replay's <line>.cell.json; a --pick-energy end is not the deepest), default the
    deepest cell on the field."""
    from ge_replay import chain_to
    z = np.load(archive, allow_pickle=False)
    if cell is None:
        d = np.asarray(field.sample(z["state"]["origin"].astype(np.float64)), np.float64)
        d[d >= field.reach_max - 0.5 * field.cell] = np.inf
        cell = int(np.argmin(d))
    ch = chain_to(z["parent"], int(cell))
    seq = np.concatenate([z["link_acts"][z["link_off"][c]:z["link_off"][c + 1]]
                          for c in ch[1:]]) if len(ch) > 1 else np.zeros((0, 6), np.int64)
    if len(seq) != n_states - 1:
        return None
    return np.asarray(seq[:, [2, 3, 5]], np.int32)


def stall_tick(d, margin: float) -> int:
    """The first tick at which the line came within ``margin`` u of its best d: where it
    stopped making progress (a line that crawls on after a crash keeps its best d for many
    seconds, and roots on that crawl explore a dead end - pbr3b4's first beam hopped on a ledge
    for 15 s after losing 1,000 u/s there)."""
    ok = np.flatnonzero(np.asarray(d) <= float(np.min(d)) + float(margin))
    return int(ok[0]) if len(ok) else int(np.argmin(d))


def boost_stages(spec: str) -> list:
    """--ge-boost '1.0,1.04;1.08,1.12' -> [[1.0, 1.04], [1.08, 1.12]]: the escalation ladder."""
    return [[float(b) for b in st.split(",") if b.strip()]
            for st in str(spec).split(";") if st.strip()]


def go_explore(a, out: Path, line, end: int, boosts=None, tag: str = "ge", line_keys=None):
    """GE --exact-chains rooted at the line's last --roots-secs before ``end``; the deepest
    chain replayed and spliced onto the line at its root -> (spliced states, info)."""
    from spine_canon_view import canon_view
    from surfgym.core import STATE_DTYPE
    lo = max(0, end - int(round(a.roots_secs * TICKS_PER_S)))
    idx = np.unique(np.linspace(lo, end, max(2, int(a.roots_n))).round().astype(int))
    roots = clean(line[idx])
    if boosts is None:
        boosts = boost_stages(a.ge_boost)[0]
    if boosts != [1.0]:
        # --ge-boost: each root also with its velocity scaled (the --respawn-front-perturb
        # rule: the line's own states with more or less energy); the chain's root is matched
        # back to the line by position, so a boosted root splices like any other
        rs = []
        for b in boosts:
            r = np.array(roots, copy=True)
            r["velocity"] = (r["velocity"].astype(np.float64) * b).astype(r["velocity"].dtype)
            rs.append(r)
        roots = np.concatenate(rs)
    gd = out / tag
    gd.mkdir(parents=True, exist_ok=True)
    np.save(gd / "roots.npy", roots)
    cmd = [sys.executable, "-u", ROOT / "tools" / "explore_phase1.py", "--map", a.map,
           "--out", gd, "--envs", a.envs, "--cell", a.ge_cell, "--cell-energy", a.ge_energy,
           "--decisions", 100, "--act-every", a.act_every, "--exact-chains",
           "--roots-spine", gd / "roots.npy", "--max-minutes", a.ge_minutes,
           "--goals", 1, "--seed", a.seed]
    if a.maxvel:
        cmd += ["--maxvel", a.maxvel]
    if a.threads:
        cmd += ["--threads", a.threads]
    if int(round(a.tick)) != 10:
        # the policy flies another tick (--tick-ms-schedule): so does the explorer, rounded to
        # whole ms (10 = the explorer's default, byte-identical)
        cmd += ["--tick-ms", int(round(a.tick))]
    if a.ge_view_follow and str(a.ge_view_follow).lower() != "none":
        # explore_phase1 --view-follow: the explorer's view holds offsets from the velocity
        # heading (the policy's frame), so a chain's held keys mean what they would mean to the
        # policy (round 4: gates 12 and 14 fell to such chains, unboosted)
        cmd += [f"--view-follow={a.ge_view_follow}"]
    rc = run(cmd, out / f"{tag}.log", a.ge_minutes * 60 + 600)
    info = {"ge_rc": rc, "ge_roots": [int(i) for i in idx], "ge_boost": boosts}
    if not (gd / "archive.npz").exists():
        return None, info
    rcmd = [sys.executable, "-u", ROOT / "tools" / "ge_replay.py", "--map", a.map,
            "--archive", gd / "archive.npz", "--field", a.field, "--out", gd / "line.npy"]
    if float(a.ge_pick_energy) > 0.0 and np.isfinite(a.greedy_d) and a.d0:
        # --ge-pick-energy: the chain end is the survivable cell past the spine's bar with the
        # highest training potential (progress + K x energy height), not the deepest one
        rcmd += ["--pick-max-d", f"{a.greedy_d - a.gain:.1f}", "--pick-energy",
                 f"{float(a.ge_pick_energy):g}", "--pick-d0", f"{float(a.d0):.1f}"]
    rc2 = run(rcmd, out / f"{tag}_replay.log", 1200)
    info["ge_replay_rc"] = rc2
    if not (gd / "line.npy").exists():
        return None, info
    gl = np.load(gd / "line.npy")
    if gl.dtype != STATE_DTYPE or len(gl) < 2:
        return None, info
    # the chain starts at one of the roots: find it on the line (roots are exact copies)
    ro = np.asarray(gl[0]["origin"], np.float64)
    dist = np.linalg.norm(np.asarray(line["origin"], np.float64) - ro, axis=1)
    j = int(np.argmin(dist))
    vr = float(np.linalg.norm(np.asarray(gl[0]["velocity"], np.float64))
               / max(1e-6, float(np.linalg.norm(np.asarray(line[j]["velocity"], np.float64)))))
    info.update(ge_root_index=j, ge_root_err=float(dist[j]), ge_chain_len=int(len(gl)),
                ge_root_boost=round(vr, 3))
    if dist[j] > 1.0:
        return None, info
    chain, _m = canon_view(gl[1:])
    if line_keys is not None:
        _cj = gd / "line.npy.cell.json"
        _cell = json.loads(_cj.read_text())["cell"] if _cj.exists() else None
        ck = chain_keys(gd / "archive.npz", load_field(a.field), len(gl), cell=_cell)
        if ck is not None:
            # the explorer's keys meant what they did relative to ITS view; the chain's views
            # are now along the velocity, so the keys move with them (spine_canon_keys)
            from spine_canon_keys import translate
            v = np.asarray(gl["velocity"][1:], np.float64)
            hd = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
            fast = np.hypot(v[:, 0], v[:, 1]) >= 100.0
            ck = ck.copy()
            ck[fast] = translate(ck[fast], np.asarray(gl["yaw"][1:], np.float64)[fast],
                                 hd[fast])
            KEYS[tag] = np.concatenate([np.asarray(line_keys[:j + 1], np.int32),
                                        ck.astype(np.int32)])
    return np.concatenate([np.asarray(line[:j + 1], STATE_DTYPE),
                           np.asarray(chain, STATE_DTYPE)]), info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt")
    ap.add_argument("--map")
    ap.add_argument("--field", help="the goal field the d of every rule is read on (the "
                                     "trainer's shaping field: <map>.goalc_32.npz)")
    ap.add_argument("--out", help="output directory (spine.npy, spine.json, logs)")
    ap.add_argument("--greedy-d", type=float, default=float("inf"),
                    help="the greedy eval's frontier d; a spine must end below it - --gain")
    ap.add_argument("--gain", type=float, default=200.0,
                    help="u of field distance a new spine must beat --greedy-d by")
    ap.add_argument("--beam-ticks", type=int, default=4000, help="beam_tas --max-ticks")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--greedy-envs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ge-minutes", type=float, default=10.0, help="0 = no GE fallback")
    ap.add_argument("--ge-cell", type=float, default=64.0)
    ap.add_argument("--ge-energy", type=float, default=32.0)
    ap.add_argument("--ge-boost", default="1.0,1.04;1.08;1.12",
                    help="the GE's escalation ladder: ';'-separated stages of comma-separated "
                         "speed factors; every root is tried with its velocity scaled by each "
                         "factor of a stage, and the next stage (--ge-minutes each) runs only "
                         "when one finds no line past the greedy frontier. One factor per stage "
                         "above 1.04: the deepest chain of a mixed stage is the most boosted "
                         "one, and a x1.12 root put a spine out of the greedy line's reach "
                         "(round 3, auto-spine search 3)")
    ap.add_argument("--ge-view-follow", default="-90,-60,-30,-15,0,15,30,60,90",
                    help="explore_phase1 --view-follow offsets (deg from the velocity heading) for "
                         "the Go-Explore stages; 'none' = the uniform yaw-rate explorer")
    ap.add_argument("--ge-pick-energy", type=float, default=0.0,
                    help="> 0: a Go-Explore stage's chain ends at the survivable archive cell past "
                         "--greedy-d - --gain with the highest -100 d / d0 + K h / 1000 (K = this, "
                         "the run's --energy-pot; h the energy height) instead of the deepest cell "
                         "(tools/ge_replay.py --pick-energy). 0 = the deepest (the old rule)")
    ap.add_argument("--d0", type=float, default=None,
                    help="the race reward's progress scale (the field at the true start); default "
                         "the beam line's first state")
    ap.add_argument("--roots-secs", type=float, default=4.0,
                    help="Go-Explore roots: the line's last seconds before it stalls")
    ap.add_argument("--stall-margin", type=float, default=500.0,
                    help="the line stalls where it first comes within this many u of its "
                         "best d")
    ap.add_argument("--roots-n", type=int, default=48)
    ap.add_argument("--act-every", type=int, default=4)
    ap.add_argument("--maxvel", type=float, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800.0, help="beam_tas wall cap (s)")
    ap.add_argument("--contact-tol", type=float, default=1.0)
    ap.add_argument("--cut", choices=["best", "contact"], default="best",
                    help="where a line ends: its best d (the --gain margin guards against a "
                         "fall), or the best d at or before its last map contact")
    ap.add_argument("--branch-tick", type=int, default=0,
                    help="> 0: when the plain beam falls short, a second beam forks half its "
                         "population at this tick (the greedy line's commitment point)")
    ap.add_argument("--branch-extra-secs", default="1,2",
                    help="comma list: further branched beams forking this many seconds before "
                         "--branch-tick (each ~20 s on a 5080), the deepest line kept")
    ap.add_argument("--branch-burst-secs", type=float, default=2.0)
    ap.add_argument("--branch-hold-secs", type=float, default=0.5)
    ap.add_argument("--branch-jitter", type=int, default=45,
                    help="deg: the held view offset is U(-J, J) around the policy's own aim")
    ap.add_argument("--branch-protect", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    for k in ("ckpt", "map", "field", "out"):
        if getattr(a, k) is None:
            ap.error(f"--{k} is required")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    global TICKS_PER_S
    a.tick = ckpt_tick(a.ckpt)
    TICKS_PER_S = 1000.0 / a.tick
    if int(round(a.tick)) != 10:
        print(f"checkpoint tick {a.tick:.4f} ms: the beam runs it (physics parity), the explorer "
              f"{int(round(a.tick))} ms; seconds convert at {TICKS_PER_S:.2f} ticks/s", flush=True)
    field = load_field(a.field)
    res = {"greedy_d": a.greedy_d, "gain": a.gain, "spine": None}
    line, info = beam(a, out)
    line_tag = "beam"
    res.update(info)
    best_src = None
    if line is not None and len(line):
        d = field_d(field, line)
        if a.d0 is None and np.isfinite(d[0]):
            a.d0 = float(d[0])                  # the beam starts at the true start
        e, de = cut_line(line, d, a.contact_tol, a.cut)
        res.update(beam_len=int(len(line)), beam_end=e, beam_end_d=de,
                   beam_best_d=float(d.min()))
        print(f"beam line: {len(line)} states, best d {d.min():,.0f}, cut at {e} "
              f"(last contact {contact_cut_states(line, a.contact_tol)}) d {de:,.0f}; greedy "
              f"frontier {a.greedy_d:,.0f}", flush=True)
        if de < a.greedy_d - a.gain:
            best_src = ("beam", line[:e + 1], de)
        # branched beams: the fork at --branch-tick and --branch-extra-secs earlier (a fork AT
        # the commitment point only perturbs what is already decided; each beam costs ~20 s)
        forks = [] if a.branch_tick <= 0 else [a.branch_tick] + [
            int(a.branch_tick - float(x) * TICKS_PER_S) for x in
            str(a.branch_extra_secs).split(",") if x.strip()]
        for k, ft in enumerate([f for f in forks if f > 0]):
            if best_src is not None:
                break
            tag = "beam_br" if k == 0 else f"beam_br{k + 1}"
            bl, bi = beam(a, out, ft, tag=tag)
            res.update(bi)
            if bl is None or not len(bl):
                continue
            bd_ = field_d(field, bl)
            be, bde = cut_line(bl, bd_, a.contact_tol, a.cut)
            pre = "br" if k == 0 else f"br{k + 1}"
            res.update({f"{pre}_len": int(len(bl)), f"{pre}_end": be, f"{pre}_end_d": bde,
                        f"{pre}_best_d": float(bd_.min()), f"{pre}_fork": int(ft)})
            print(f"branched beam line (fork at tick {ft}): {len(bl)} states, "
                  f"best d {bd_.min():,.0f}, cut at {be} d {bde:,.0f}", flush=True)
            if bde < a.greedy_d - a.gain:
                best_src = (tag, bl[:be + 1], bde)
            elif bde < de:
                line, d, e, de = bl, bd_, be, bde     # GE roots on the deepest line
                line_tag = tag
        if best_src is None and a.ge_minutes > 0:
            st = min(e, stall_tick(d, a.stall_margin))
            res.update(beam_stall=st)
            # the escalation ladder: a stage's speed factors only if the previous stage found
            # no line past the greedy frontier. The boost buys the uniform explorer the energy
            # its random actions waste; the trained policy has flown the route it found with
            # its own (round 3: gate 8's chain needed x1.04, the greedy passed unboosted)
            for k, boosts in enumerate(boost_stages(a.ge_boost)):
                tag = "ge" if k == 0 else f"ge{k + 1}"
                gl, gi = go_explore(a, out, line, st, boosts=boosts, tag=tag,
                                    line_keys=KEYS.get(line_tag))
                res.update({(kk if k == 0 else f"{tag}_{kk[3:]}" if kk.startswith("ge_")
                             else f"{tag}_{kk}"): vv for kk, vv in gi.items()})
                if gl is None:
                    continue
                gd_ = field_d(field, gl)
                ge_e, ge_de = cut_line(gl, gd_, a.contact_tol, a.cut)
                if (float(a.ge_pick_energy) > 0.0 and np.isfinite(gd_[-1])
                        and gd_[-1] < a.greedy_d - a.gain):
                    # --ge-pick-energy: the chain was replayed TO the picked cell; a deeper point
                    # earlier on it is where it had less energy left, so the spine ends at the pick
                    ge_e, ge_de = len(gl) - 1, float(gd_[-1])
                res.update({f"{tag}_len": int(len(gl)), f"{tag}_end": ge_e,
                            f"{tag}_end_d": ge_de, f"{tag}_best_d": float(gd_.min())})
                print(f"GE stage {k + 1} (speed x {boosts}): {len(gl)} states, best d "
                      f"{gd_.min():,.0f}, cut at {ge_e} d {ge_de:,.0f}", flush=True)
                if ge_de < a.greedy_d - a.gain:
                    best_src = (f"beam+{tag}", gl[:ge_e + 1], ge_de)
                    break
    if best_src is not None:
        src, sp, de = best_src
        sp = clean(sp)
        p = out / "spine.npy"
        np.save(p, sp)
        import hashlib
        res.update(spine=str(p), source=src, n=int(len(sp)), end_d=de,
                   md5=hashlib.md5(p.read_bytes()).hexdigest())
        kt = src.split("+")[-1]
        if KEYS.get(kt) is not None and len(KEYS[kt]) >= len(sp):
            # the spine's held keys, row for row (train_fast --demo-keys loads them on a swap)
            kp = out / "spine.keys.npy"
            np.save(kp, np.asarray(KEYS[kt][:len(sp)], np.int32))
            res.update(keys=str(kp))
        (out / "spine.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("AUTO_SPINE " + json.dumps(res), flush=True)
    return 0 if best_src is not None else 2


def selftest() -> int:
    """The cut on synthetic lines: a ride that ends in a free fall is cut at its last contact,
    and the end is the best d inside the ride, not in the fall."""
    from surfgym.core import STATE_DTYPE

    class F:                                   # d = x, reachable everywhere
        reach_max, cell = 1e9, 32.0

        def sample(self, o):
            return np.asarray(o, np.float64)[:, 0]
    n = 300
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = 1000.0 - np.arange(n) * 2.0          # d falls all the way
    vz = np.zeros(n)
    vz[:120] = np.sin(np.arange(120) / 7.0) * 50.0           # a ride: the map pushes back
    vz[120:] = vz[119] - 8.0 * np.arange(1, n - 119)          # then ballistic (8 u/s/tick)
    s["velocity"][:, 2] = vz
    d = field_d(F(), s)
    c = contact_cut_states(s)
    e, de = cut_line(s, d)
    ok = True
    for name, cond in (("contact cut at the ride's end", 115 <= c <= 120),
                       ("the end is the best d before the fall", e == c and de == d[c]),
                       ("the fall is not part of the line", e < 120)):
        print(("ok   " if cond else "FAIL ") + name)
        ok &= bool(cond)
    # a line that never leaves the ground keeps its best d
    s2 = np.zeros(50, STATE_DTYPE)
    s2["origin"][:, 0] = 500.0 - np.abs(np.arange(50) - 30.0)
    s2["velocity"][:, 2] = np.sin(np.arange(50))
    e2, de2 = cut_line(s2, field_d(F(), s2))
    cond = e2 == 0 or de2 == float(field_d(F(), s2).min())
    print(("ok   " if cond else "FAIL ") + "a grounded line ends at its best d")
    ok &= bool(cond)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
