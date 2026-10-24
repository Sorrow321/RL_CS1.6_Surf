"""autospine.py - the search -> curriculum loop of pbr2b3 (round 2), run by the trainer itself.

Round 2 crossed six gates of surf_princessburglar_b26 (3.2% -> 20.4%) with one loop, driven by
hand: when the greedy line from the true start reaches the end of the current spine,
``tools/beam_tas.py`` (a policy-guided population search: N stochastic copies of the CURRENT
weights from the true start, the leaders by goal-field d cloned every 25 decisions) finds the
policy's own line to the next frontier; the line is cut at its best and becomes the next
``--demo-file`` of the backward curriculum (Salimans & Chen), whose success line moves one gate
deeper. Each cycle waited for an agent wakeup (20-30 min) and a relaunch.

``--auto-spine`` runs that loop inside one launch:

* after every greedy eval the trainer calls :meth:`AutoSpine.note_eval`: when ``quorum`` of the
  eval's episodes reach the spine's success line (their min d on the shaping field is at or
  under ``--demo-success-d``'s line), a beam is wanted;
* :meth:`AutoSpine.start` runs beam_tas as a SUBPROCESS on a checkpoint of the current weights
  (the trainer keeps training meanwhile);
* :meth:`AutoSpine.poll` collects the frontier line (``beam_frontier.npz``, the exact replay of
  the deepest lineage), cuts it at its deepest CONTACT (the minimum d among the states where
  the map pushed back - never inside the fall that ends it), zeroes its clocks,
  puts the view along the velocity (``tools/spine_canon_view.py``'s rule) and accepts it only if
  its success line lies beyond what the quorum of greedy episodes already reaches; the trainer
  then swaps the curriculum onto it (tau back at the new spine's end).

Generic by construction (CLAUDE.md 0b): every input is the policy's own sampling in the
simulator from the map's true start and the shaping field the reward already uses; no map
coordinate, gate box or record enters. Every spine is a policy-derived SELF_STATES file and gets a
provenance sidecar (``autospine_<k>.json``)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import STATE_DTYPE

__all__ = ["AutoSpine", "SpineResult", "cut_line", "contact_mask", "canon_view",
           "episode_min_d"]


def canon_view(states: np.ndarray, min_speed: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """``tools/spine_canon_view.py``'s rule: a copy of ``states`` with yaw = the velocity heading
    (degrees, [0, 360)) on every row whose horizontal speed is >= ``min_speed`` (the core's
    absolute-view regime); returns (copy, mask of the rows changed)."""
    out = np.array(states, copy=True)
    v = out["velocity"].astype(np.float64)
    hs = np.hypot(v[:, 0], v[:, 1])
    m = hs >= float(min_speed)
    hd = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    out["yaw"][m] = hd[m].astype(out["yaw"].dtype)
    return out, m


def _field_d(field, origins) -> np.ndarray:
    """d on ``field`` per origin, the unreachable sentinel -> inf."""
    d = np.asarray(field.sample(np.asarray(origins, np.float64).reshape(-1, 3)), np.float64)
    return np.where(d < field.reach_max - 0.5 * field.cell, d, np.inf)


def contact_mask(states: np.ndarray, tol: float = 1.0) -> np.ndarray:
    """True on the states where the map pushed back: the tick's change of vertical velocity
    departs from the gravity step by more than ``tol`` (tools/pick_selfline.py's contact_cut, the
    rule CLAUDE.md names for trimming a policy line's fall). The gravity step is the MEDIAN of the
    per-tick change - a surf line is mostly ballistic (68% of the incumbent's greedy line), so it
    needs no engine constant. State 0 counts as a contact (the spawn)."""
    vz = np.asarray(states["velocity"], np.float64)[:, 2]
    m = np.zeros(len(vz), bool)
    if len(vz):
        m[0] = True
    if len(vz) >= 3:
        dvz = np.diff(vz)
        g = float(np.median(dvz))
        m[1:] = np.abs(dvz - g) > float(tol)
    return m


def cut_line(states: np.ndarray, field, canon: bool = True,
             contact: bool = True) -> tuple[np.ndarray, int, float]:
    """The spine a frontier line gives: its states up to and including the DEEPEST CONTACT - the
    state of minimum d on ``field`` among those where the map pushed back (:func:`contact_mask`).
    A line ends in the fall that killed it, and its deepest point is usually inside that fall
    (the incumbent's greedy line: last contact at 17.6%, then 1.35 s of free fall into a catch to
    19.9%); a spine ending there would pay the success bonus for diving deeper into the catch.
    ``contact=False``, or a line with no contact after its first state, cuts at the minimum d.
    Clocks zeroed (``tick``, ``stuck_ticks``, as every spine tool does), the view along the
    velocity when ``canon``. Returns (spine, cut index, d at the cut)."""
    s = np.asarray(states, STATE_DTYPE)
    if not len(s):
        raise ValueError("an empty line")
    d = _field_d(field, s["origin"])
    if not np.isfinite(d).any():
        raise ValueError("no state of the line is on the field")
    i = int(np.argmin(d))
    if contact:
        cm = contact_mask(s)
        cm[0] = False
        if cm.any() and np.isfinite(d[cm]).any():
            i = int(np.flatnonzero(cm)[np.argmin(d[cm])])
    sp = s[:i + 1].copy()
    sp["tick"] = 0
    sp["stuck_ticks"] = 0
    if canon:
        sp, _ = canon_view(sp)
    return sp, i, float(d[i])


def episode_min_d(traj_path, field) -> list[float]:
    """Each episode's minimum d on ``field`` in a record.py trajectory (a header line, the rows,
    a trailer {"end": ...}); positions are row columns 1:4, as score.py reads them."""
    out, rows, in_ep = [], [], False
    with open(traj_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == "{":
                rec = json.loads(line)
                if "end" in rec and in_ep:
                    if rows:
                        d = _field_d(field, np.asarray(rows, np.float64)[:, 1:4])
                        out.append(float(d.min()))
                    else:
                        out.append(float("inf"))
                    rows, in_ep = [], False
                else:
                    rows, in_ep = [], True
                continue
            if in_ep:
                rows.append(json.loads(line))
    return out


@dataclass
class SpineResult:
    states: np.ndarray        # the new spine (STATE_DTYPE), true start first
    path: Path                # where it was saved
    d_cut: float              # d of its last state on the field
    cut: int                  # the cut index on the beam's frontier line
    k: int                    # beam number (1-based)
    step: int                 # trainer step the beam's weights are from
    md5: str


class AutoSpine:
    """The --auto-spine state machine: idle -> wanted -> beam -> (accepted spine | GE -> replay ->
    (accepted spine | idle)).

    ``field``: the shaping field (what --demo-success-d's line is measured on). ``beam_cmd``: a
    callable (ckpt, out_dir, seed, max_ticks) -> argv, so a test can stand a stub in for
    beam_tas. ``margin``: the --demo-success-d margin, M (the success line is d(spine end) + M).
    ``quorum``: episodes of an eval that must reach the line to want a beam (0 = every eval).
    ``retry``: evals to wait after a cycle that found nothing deeper.

    The Go-Explore fallback (``ge_minutes`` > 0; round 2 needed it at three of its gates, where
    the policy's own proposals stalled): when a beam's line is not deep enough, Go-Explore phase
    1 with exact chains (``ge_cmd(roots_npy, out_dir, minutes)``: reward-free uniform bursts)
    is rooted at ``ge_roots_n`` states of that line's last ``ge_roots_ticks`` ticks before its
    best; ``replay_cmd(archive, field_npz, out_npy)`` replays the archive's deepest chain into a
    per-tick line; the beam line up to the chain's root + the chain is judged like a beam line.

    ``stall`` > 0: after that many evals without the quorum, a cycle runs anyway and its line only
    has to reach past the greedy's quorum-th best (not the success line in force), so a spine the
    greedy cannot be carried through is replaced by the current weights' own next line."""

    def __init__(self, out_dir, field, beam_cmd, margin: float, quorum: int = 5,
                 retry: int = 4, canon: bool = True, ep_ticks: int = 30000,
                 max_ticks: int = 0, ge_cmd=None, replay_cmd=None, ge_minutes: float = 0.0,
                 ge_roots_ticks: int = 500, ge_roots_n: int = 25, stall: int = 0,
                 log=print) -> None:
        self.out = Path(out_dir)
        self.field = field
        self.beam_cmd = beam_cmd
        self.margin = float(margin)
        self.quorum = int(quorum)
        self.retry = int(retry)
        self.canon = bool(canon)
        self.ep_ticks = int(ep_ticks)
        self.max_ticks = int(max_ticks)       # 0 = sized on the eval that wants the beam
        self.ge_cmd = ge_cmd
        self.replay_cmd = replay_cmd
        self.ge_minutes = float(ge_minutes)
        if self.ge_minutes > 0.0 and (ge_cmd is None or replay_cmd is None):
            raise ValueError("ge_minutes > 0 needs ge_cmd and replay_cmd")
        self.ge_roots_ticks = int(ge_roots_ticks)
        self.ge_roots_n = int(ge_roots_n)
        # stall: evals without the quorum (counted from the last accepted spine, or the last
        # stall cycle) after which a cycle runs anyway and is judged against the greedy's own
        # level only - a spine the curriculum cannot carry the greedy line through is re-cut
        # on the current weights (round 2's beam10 -> beam11); 0 = off
        self.stall = int(stall)
        self.since = 0
        self.want_stall = False
        # consecutive cycles that ended without a spine: each doubles the wait before the next
        # (retry x 1, 2, 4, 8 evals), so a gate that needs training rather than search does not
        # keep a beam + Go-Explore running beside the trainer; an accepted spine resets it
        self.fails = 0
        self.log = log
        self.k = 0                    # cycles (beams) started
        self.n_eval = 0               # evals seen
        self.retry_at = 0             # the eval count before which no beam is wanted
        self.want = False
        self.want_ticks = 0           # the beam's --max-ticks, sized on the eval that wanted it
        self.q_d = float("inf")       # the quorum-th best episode's min d in that eval
        self.proc = None
        self.cur = None               # the running stage: dict(stage, k, step, dir, lf, t0, ...)
        self.accepted = 0
        self.history = []
        self._field_npz = None

    # ---- the trigger ---------------------------------------------------------------------
    def note_eval(self, traj_path, step: int, end_d) -> str:
        """Read one greedy eval; set ``want`` when quorum of its episodes reached ``end_d`` (the
        success line in force). Returns a one-line note for the log."""
        self.n_eval += 1
        try:
            md = sorted(episode_min_d(traj_path, self.field))
        except (OSError, ValueError) as e:
            return f"auto-spine: eval unreadable ({e})"
        if not md:
            return "auto-spine: eval has no episodes"
        q = max(1, self.quorum)
        q_d = md[min(q, len(md)) - 1]
        n_in = int(sum(1 for x in md if end_d is not None and x <= end_d))
        note = (f"auto-spine: {n_in}/{len(md)} greedy episodes at or past the success line"
                + (f" (d <= {end_d:,.0f})" if end_d is not None else "")
                + f", quorum-th best d {q_d:,.0f}")
        if self.proc is not None:
            return note + f"; cycle {self.k} still running ({self.cur['stage']})"
        if self.n_eval < self.retry_at:
            return note + f"; next beam no earlier than eval {self.retry_at}"
        stalled = (self.stall > 0 and end_d is not None and self.quorum > 0
                   and n_in < self.quorum and self.n_eval - self.since >= self.stall)
        if end_d is None or self.quorum <= 0 or n_in >= self.quorum or stalled:
            self.want = True
            self.want_stall = bool(stalled)
            self.q_d = q_d
            # the beam runs until its population dies; the cap is twice the eval's longest
            # episode + 20 s, within the training episode cap
            try:
                n_rows = self._longest_episode(traj_path)
            except OSError:
                n_rows = self.ep_ticks
            self.want_ticks = (self.max_ticks if self.max_ticks > 0 else
                               int(min(self.ep_ticks, max(3000, 2 * n_rows + 2000))))
            note += (f" -> beam {self.k + 1} wanted (cap {self.want_ticks} ticks"
                     + (f"; STALL: {self.n_eval - self.since} evals without the quorum, judged "
                        f"against the greedy's level only" if stalled else "") + ")")
        return note

    @staticmethod
    def _longest_episode(traj_path) -> int:
        best, n = 0, 0
        with open(traj_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("{"):
                    best, n = max(best, n), 0
                elif line.strip():
                    n += 1
        return max(best, n)

    # ---- the stages ------------------------------------------------------------------------
    def _spawn(self, stage: str, argv, log_name: str, **extra) -> None:
        d = extra.pop("dir")
        lf = open(d / log_name, "w", encoding="utf-8")
        lf.write("# " + " ".join(str(a) for a in argv) + "\n")
        lf.flush()
        self.proc = subprocess.Popen([str(a) for a in argv], stdout=lf,
                                     stderr=subprocess.STDOUT)
        self.cur = {"stage": stage, "dir": d, "lf": lf, "t0": time.time(), **extra}

    def start(self, ckpt_path, step: int) -> None:
        self.want = False
        self.k += 1
        d = self.out / f"autobeam_{self.k:02d}"
        d.mkdir(parents=True, exist_ok=True)
        self._spawn("beam", self.beam_cmd(Path(ckpt_path), d, self.k - 1, self.want_ticks),
                    "beam.log", dir=d, k=self.k, step=int(step), q_d=self.q_d,
                    t_cycle=time.time(), stall=self.want_stall)
        self.log(f"auto-spine: beam {self.k} started on the step-{step:,} weights "
                 f"(pid {self.proc.pid}, cap {self.want_ticks} ticks) -> {d}")

    def running(self) -> bool:
        return self.proc is not None

    def _backoff(self, rec: dict, result: str, msg: str, cur=None) -> None:
        self.fails += 1
        self.retry_at = self.n_eval + self.retry * (2 ** min(self.fails - 1, 3))
        if cur is not None and cur.get("stall"):
            self.since = self.n_eval          # the next stall cycle waits a whole stall window
        self.history.append({**rec, "result": result})
        self.log(f"auto-spine: {msg}; next beam no earlier than eval {self.retry_at}")

    def field_npz(self) -> Path:
        """The shaping field as an npz tools/ge_replay.py reads (written once per run)."""
        if self._field_npz is None:
            p = self.out / "autospine_field.npz"
            f = self.field
            np.savez(p, grid=np.asarray(f.grid, np.float32), quant=np.float32(1.0),
                     mins=np.asarray(f.mins, np.float64), cell=np.float64(f.cell),
                     reach_max=np.float64(f.reach_max))
            self._field_npz = p
        return self._field_npz

    def _start_ge(self, cur: dict, line: np.ndarray, cut: int) -> None:
        lo = max(0, cut - self.ge_roots_ticks)
        idx = np.unique(np.linspace(lo, cut, max(1, self.ge_roots_n)).round().astype(int))
        roots = np.asarray(line, STATE_DTYPE)[idx].copy()
        roots["tick"] = 0
        roots["stuck_ticks"] = 0
        d = cur["dir"]
        rp = d / "ge_roots.npy"
        np.save(rp, roots)
        keep = {k: cur[k] for k in ("k", "step", "q_d", "t_cycle", "stall")}
        self._spawn("ge", self.ge_cmd(rp, d / "ge", self.ge_minutes), "ge.log", dir=d,
                    line=np.asarray(line, STATE_DTYPE), **keep)
        self.log(f"auto-spine: cycle {cur['k']}: Go-Explore ({self.ge_minutes:g} min, exact "
                 f"chains) rooted at {len(roots)} states of the beam line's ticks "
                 f"[{lo},{cut}] (pid {self.proc.pid})")

    def poll(self, end_d):
        """None while nothing is done; a SpineResult when a cycle produced a line deeper than
        both ``end_d`` (the line in force) and the wanting eval's quorum-th best (each by the
        margin); None (and a retry backoff) when a cycle ended without one."""
        if self.proc is None or self.proc.poll() is None:
            return None
        rc = self.proc.returncode
        cur = self.cur
        cur["lf"].close()
        self.proc, self.cur = None, None
        dt = time.time() - cur["t0"]
        k, d, stage = cur["k"], cur["dir"], cur["stage"]
        rec = {"k": k, "step": cur["step"], "stage": stage, "rc": rc, "secs": round(dt, 1)}
        if stage == "beam":
            fz = d / "beam_frontier.npz"
            if rc != 0 or not fz.is_file():
                self._backoff(rec, "failed", f"beam {k} FAILED (exit {rc}, {dt:.0f}s; "
                              f"{d / 'beam.log'})", cur)
                return None
            z = np.load(fz, allow_pickle=False)
            if "replay_ok" in z.files and not int(z["replay_ok"]):
                self._backoff(rec, "replay_failed",
                              f"beam {k}: the frontier line did not replay exactly", cur)
                return None
            line = np.asarray(z["states"], STATE_DTYPE)
            res, cut = self._judge(line, end_d, cur, rec, "beam")
            if res is not None or cut is None:
                return res
            if self.ge_minutes > 0.0:
                self._start_ge(cur, line, cut)
                return None
            self._backoff(rec, "no_gain", f"beam {k} ({dt:.0f}s) found nothing deeper", cur)
            return None
        if stage == "ge":
            arch = d / "ge" / "archive.npz"
            # explore_phase1 exits 1 when nothing finished: the normal end of a budgeted search
            if rc not in (0, 1) or not arch.is_file():
                self._backoff(rec, "ge_failed", f"cycle {k}: Go-Explore FAILED (exit {rc}; "
                              f"{d / 'ge.log'})", cur)
                return None
            self._spawn("replay", self.replay_cmd(arch, self.field_npz(), d / "ge_line.npy"),
                        "ge_replay.log", dir=d, line=cur["line"],
                        **{x: cur[x] for x in ("k", "step", "q_d", "t_cycle", "stall")})
            return None
        # stage == "replay": ge_replay exits 1 when a link diverged; its line stops there and
        # is still exact up to that link
        lp = d / "ge_line.npy"
        if not lp.is_file():
            self._backoff(rec, "replay_failed", f"cycle {k}: ge_replay wrote no line (exit {rc})",
                          cur)
            return None
        gl = np.asarray(np.load(lp, allow_pickle=False), STATE_DTYPE)
        line = cur["line"]
        o0 = gl["origin"][0].astype(np.float64)
        err = np.linalg.norm(line["origin"].astype(np.float64) - o0, axis=1)
        j = int(np.argmin(err))
        if not len(gl) or err[j] > 0.5:
            self._backoff(rec, "splice_failed", f"cycle {k}: the GE chain's root is not on the "
                          f"beam line (nearest {err[j]:.1f} u)", cur)
            return None
        spliced = np.concatenate([line[:j], gl])
        rec["splice_at"] = j
        res, cut = self._judge(spliced, end_d, cur, rec, "ge")
        if res is None and cut is not None:
            self._backoff(rec, "no_gain", f"cycle {k}: the beam line + the GE chain found "
                          f"nothing deeper either", cur)
        return res

    def _judge(self, line, end_d, cur, rec, source):
        """Cut ``line`` at its deepest contact and accept it if its success line is past the bar;
        returns (SpineResult or None, cut index or None when the line is unusable)."""
        k, step, q_d = cur["k"], cur["step"], cur["q_d"]
        try:
            sp, cut, d_cut = cut_line(line, self.field, canon=self.canon)
        except ValueError as e:
            self._backoff(rec, "no_line", f"cycle {k}: no usable line ({e})", cur)
            return None, None
        new_end = d_cut + self.margin
        # a stall cycle answers to the greedy's own level only (it may re-cut the spine SHORTER)
        bar = (float(q_d) if cur.get("stall") else
               min(float("inf") if end_d is None else float(end_d), float(q_d)))
        rec.update(d_cut=round(d_cut, 1), cut=cut, stall=bool(cur.get("stall")),
                   bar=(round(bar, 1) if np.isfinite(bar) else None))
        if not new_end < bar:
            self.log(f"auto-spine: cycle {k} ({source}): best d {d_cut:,.0f} (+{self.margin:g} "
                     f"= {new_end:,.0f}) is not past {bar:,.0f}")
            return None, cut
        p = self.out / f"autospine_{k:02d}.npy"
        np.save(p, sp)
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        how = ("beam_tas (population search over stochastic copies of the step-"
               f"{step:,} weights of this run, d-ranked) frontier lineage replayed exactly")
        if source == "ge":
            how += (f" up to tick {rec['splice_at']}, then the deepest chain of a Go-Explore "
                    "phase-1 archive (reward-free uniform bursts, exact chains) rooted at that "
                    "line's own states, replayed by tools/ge_replay.py")
        side = {"provenance": (f"the policy's OWN line from the map's true start: {how}; cut at "
                               f"its deepest contact ({d_cut:,.0f}u on the shaping field, index "
                               f"{cut})"
                               + ("; views along the velocity" if self.canon else "")
                               + ". No demo, no record data (--auto-spine)."),
                "tool": "surfgym.autospine (train_fast --auto-spine)", "rows": int(len(sp)),
                "md5": md5, "cycle_dir": str(cur["dir"]), "beam_ckpt_step": step,
                "source": source}
        p.with_suffix(".json").write_text(json.dumps(side, indent=1), encoding="utf-8")
        self.accepted += 1
        self.since = self.n_eval
        self.fails = 0
        self.history.append({**rec, "result": "accepted", "source": source, "spine": str(p),
                             "rows": len(sp)})
        self.log(f"auto-spine: cycle {k} ({source}, {time.time() - cur['t_cycle']:.0f}s) -> new "
                 f"spine {p.name}: {len(sp)} states to d {d_cut:,.0f} (the line in force "
                 f"{bar:,.0f}), md5 {md5}")
        return SpineResult(sp, p, d_cut, cut, k, step, md5), cut

    def close(self) -> None:
        """Stop a running stage (end of training)."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.cur is not None:
            self.cur["lf"].close()
        self.proc, self.cur = None, None

    def state_dict(self) -> dict:
        return {"k": self.k, "accepted": self.accepted, "fails": self.fails,
                "history": list(self.history)}


def beam_tas_argv(repo_root, map_path, envs: int, greedy_envs: int, python=None, tool=None):
    """The beam_tas command pbr2b3 ran by hand (NOTES.md: 4,096 copies, 32 greedy envs,
    d-ranked, resample every 25 decisions, --skip-gate --save-frontier --route-file none), as a
    beam_cmd for :class:`AutoSpine`. ``tool``: another script taking the same arguments (a test
    stub); default tools/beam_tas.py."""
    py = python or sys.executable
    tool = Path(tool) if tool else Path(repo_root) / "tools" / "beam_tas.py"
    if not tool.is_absolute():
        tool = Path(repo_root) / tool

    def argv(ckpt, out_dir, seed, max_ticks):
        return [py, "-u", str(tool), str(ckpt), "--map", str(map_path),
                "--envs", str(int(envs)), "--resample-every", "25", "--score", "d",
                "--greedy-envs", str(int(greedy_envs)), "--allow-nonfinisher",
                "--save-frontier", "--skip-gate", "--route-file", "none",
                "--max-ticks", str(int(max_ticks)), "--torch-seed", str(int(seed)),
                "--out-dir", str(out_dir)]
    return argv



def ge_argv(repo_root, map_path, maxvel: float, threads: int, envs: int = 4096,
            cell: float = 32.0, cell_energy: float = 16.0, act_every: int = 4, python=None):
    """The Go-Explore run pbr2b3 rooted at a stalled line's states by hand (NOTES.md: GE_g16,
    32 u x energy 16 cells, 4,096 envs, --exact-chains), as a ge_cmd for :class:`AutoSpine`."""
    py = python or sys.executable
    tool = Path(repo_root) / "tools" / "explore_phase1.py"

    def argv(roots, out_dir, minutes):
        return [py, "-u", str(tool), "--map", str(map_path), "--out", str(out_dir),
                "--envs", str(int(envs)), "--cell", f"{float(cell):g}",
                "--cell-energy", f"{float(cell_energy):g}", "--decisions", "100",
                "--act-every", str(int(act_every)), "--exact-chains",
                "--roots-spine", str(roots), "--maxvel", f"{float(maxvel):g}",
                "--max-minutes", f"{float(minutes):g}", "--threads", str(int(threads)),
                "--print-every", "100"]
    return argv


def replay_argv(repo_root, map_path, python=None):
    """tools/ge_replay.py on an archive's deepest chain, as a replay_cmd for :class:`AutoSpine`."""
    py = python or sys.executable
    tool = Path(repo_root) / "tools" / "ge_replay.py"

    def argv(archive, field_npz, out_npy):
        return [py, "-u", str(tool), "--map", str(map_path), "--archive", str(archive),
                "--field", str(field_npz), "--out", str(out_npy)]
    return argv
