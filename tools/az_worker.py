"""az_worker.py - the SEARCH WORKER of --plan-az: AlphaZero-style expert iteration for the learned
primitive planner (--goal-planner primlearn), decoupled from the trainer so it keeps its speed.

    python tools/az_worker.py runs/<run> [--sims 16] [--k 6] [--device cpu] [--searches N]

In a loop it (re)loads runs/<run>/ckpt_latest.pt - the executor and the planner - again whenever
the file moves, picks a ROOT state, runs the MCTS over primitives from it and writes ONE target
per search to runs/<run>/az/targets_<worker>_<pid>_<n>.npz, which the trainer's --plan-az reads.

The search is surfgym/goalsearch.PrimMCTS exactly as ``tools/record_ckpt.py --plan-mcts`` builds it
(record_ckpt.build: the same scratch core, fan line and greedy executor wrapper, the same config
audit): a node is an exact simulator state, an expansion flies --k of the planner's own primitives
with the checkpoint's greedy executor, edges pay the planner's reward, leaves are valued by its
value head, values back up by MAX, PUCT picks the path. Nothing about the map is written here: the
simulator is the model and the roots are the policy's own states.

ROOTS, --start-frac of them (default 25%) a map start, the rest states of the checkpoint's own
respawn reservoir - drawn, perturbed (speed x U(0.9, 1.1), pitch +-5 deg) and weighted (Go-Explore's
1/sqrt(1+N) under --plan-return) by the trainer's own RespawnBuffer.build_pool, then spawned
through the core's reset like a training respawn (the episode fields re-zeroed, the yaw
jittered). A root is a FRESH episode: the planner sees bank 0 and the executor starts with a fresh
wrapper, as at a real spawn. No reservoir in the checkpoint (or --start-frac 1): starts only.

A TARGET (the npz keys the trainer reads; the rest are diagnostics):
  x   (N_OBS,)  the planner's observation at the root: goalprimplan.observe with bank 0, the
                vector the root's candidates were drawn at (PrimMCTS stores it on the node)
  u   (K, D)    every root candidate's pre-squash numbers (K = --k, more after progressive
                widening at the root)
  pi  (K,)      their MCTS visit fractions (a search that spent no visit - --sims 1 - gives its
                committed choice, one-hot)
  z   ()        the value target: the root's VISIT-WEIGHTED Q, sum_i pi_i Q_i, where Q_i is edge
                i's max-backup value (r + gamma x the best continuation below it)
  q, n, died, fin, step, updates, kind (0 start / 1 reservoir), origin, expansions, depth, sims, k

Why visit-weighted and not the root's best Q: every leaf is valued by V itself, and the max over
K noisy leaf estimates is biased upward (the maximisation bias, van Hasselt 2010) - the MAX backup
already compounds that bias below the root, and a max again at the root would feed it straight
back into the V the next search values its leaves with (the trainer regresses V on z), a loop
that inflates both. The visit-weighted mean is the value of the policy the target trains toward
(pi and z describe the same improved policy, AlphaZero's pairing); PUCT concentrates the visits on
the best edges, so it tends to the best Q as the search deepens, but it does not jump to the
luckiest leaf on a small budget.

z is in the TRAINER's units: the search's edge rewards take the run's own plan_progress /
plan_finish_bonus from the checkpoint's planner state, the progress unit is the planner's, the
discount is PLAN_GAMMA per primitive (per nominal duration under --plan-smdp, --plan-mcts-time).
The search pays no novelty / coverage (the eval-time MCTS's choice), so z is the extrinsic part of
what V predicts; and it pays the plain refund rule (a death charged the face-value bank), so under
--plan-shaping pbrs / refund_i z is biased against the trainer's return (the worker warns).

Stops after --searches N (tests), or when the trainer is done: ckpt_final.pt newer than this
worker's start, the --trainer-pid gone (POSIX), or no new checkpoint for --idle-exit seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", help="runs/<run>: its ckpt_latest.pt is searched, its az/ gets "
                                    "the targets")
    ap.add_argument("--ckpt", default=None,
                    help="the checkpoint to search with (default <run_dir>/ckpt_latest.pt; "
                         "reloaded whenever it changes)")
    ap.add_argument("--out", default=None, help="the target directory (default <run_dir>/az)")
    ap.add_argument("--sims", type=int, default=16,
                    help="tree expansions per search (the root's own included)")
    ap.add_argument("--k", type=int, default=6, help="primitives per expansion")
    ap.add_argument("--explore", type=float, default=0.0,
                    help="a count-based novelty bonus in the tree (record_ckpt --plan-mcts-explore): "
                         "COEF / sqrt(1 + N) on every alive edge, N = the planner's end-cell counts. "
                         "0 = off")
    ap.add_argument("--searches", type=int, default=0,
                    help="stop after this many searches (0 = until the trainer is done)")
    ap.add_argument("--device", default="cpu",
                    help="torch device of the executor / planner / lidar (default cpu: the "
                         "trainer owns the GPU)")
    ap.add_argument("--start-frac", type=float, default=0.25,
                    help="share of the roots that are map starts (the rest: reservoir states)")
    ap.add_argument("--seed", type=int, default=None, help="default: from the pid")
    ap.add_argument("--worker-id", default="w", help="names this worker's target files")
    ap.add_argument("--threads", type=int, default=2, help="torch / numba threads")
    ap.add_argument("--poll", type=float, default=5.0,
                    help="seconds between looks for a checkpoint that is not there yet")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="a checkpoint is read once it has not been written for this long (the "
                         "trainer writes it in place)")
    ap.add_argument("--idle-exit", type=float, default=1800.0,
                    help="without --searches: exit when no new checkpoint arrived for this long")
    ap.add_argument("--trainer-pid", type=int, default=None,
                    help="POSIX: exit when this process is gone (ignored on Windows)")
    return ap.parse_args(argv)


def _alive(pid) -> bool:
    """Is the trainer still running? POSIX only - on Windows os.kill(pid, 0) TERMINATES pid."""
    if pid is None or os.name != "posix":
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Worker:
    POOL_SIZE = 4096

    def __init__(self, a):
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        self.a = a
        self.run = Path(a.run_dir)
        self.ck_path = Path(a.ckpt) if a.ckpt else self.run / "ckpt_latest.pt"
        self.out = Path(a.out) if a.out else self.run / "az"
        self.out.mkdir(parents=True, exist_ok=True)
        seed = int(a.seed) if a.seed is not None else (os.getpid() * 7919) & 0x7FFFFFFF
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.ctx = None
        self.sig = None              # the built checkpoint's config (a change -> full rebuild)
        self.loaded = None           # (mtime_ns, size) of the checkpoint the weights came from
        self.gen = None
        self.pool = None
        self.n_start = 0
        self.n = 0
        self.t0 = time.time()
        self.last_new = self.t0

    # ------------------------------------------------------------------ the checkpoint
    def _build(self, ck):
        """First load or a config change: record_ckpt's own construction, --plan-mcts on."""
        sys.path.insert(0, str(ROOT / "tools"))
        import record_ckpt
        a, cfg = self.a, ck.get("config") or {}
        if cfg.get("goal_planner") != "primlearn":
            raise SystemExit(f"az_worker: {self.ck_path} is not a --goal-planner primlearn "
                             f"checkpoint (goal_planner {cfg.get('goal_planner')!r})")
        if str(cfg.get("plan_shaping") or "refund") != "refund":
            # refund_i grows the bank by 1/gamma per primitive and pbrs pays gamma*Phi' - Phi;
            # PrimMCTS pays progress as it comes and charges a death the face-value bank
            print(f"az_worker: WARNING - plan_shaping {cfg.get('plan_shaping')!r}: the search's "
                  "edge rewards follow the plain refund rule (a death is charged the face-value "
                  "bank), so z is biased against the trainer's return", flush=True)
        argv = [str(self.ck_path), "--episodes", "1", "--plan-mcts", str(max(1, a.sims)),
                "--plan-mcts-k", str(max(2, a.k)), "--plan-mcts-noreuse"]
        if float(a.explore) > 0.0:
            argv += ["--plan-mcts-explore", str(float(a.explore))]
        if int(cfg.get("plan_smdp") or 0):
            # --plan-smdp: the trainer discounts a primitive by gamma ** (duration / nominal)
            argv.append("--plan-mcts-time")
        ctx = record_ckpt.build(argv, device=a.device)
        if ctx.planner is None or ctx.search is None:
            raise SystemExit("az_worker: the recorder built no primitive planner / search")
        self.ctx = ctx
        self.gen = self.torch.Generator(device=ctx.planner.device)
        self.gen.manual_seed(self.seed)
        # every root is a FRESH episode: the executor starts from a fresh wrapper (not the
        # recorder's), and each search builds its own tree
        ctx.search.real_policy = None
        ctx.search.reuse = False

    def _adopt(self, ck):
        """The weights, reward knobs and reservoir of ``ck`` into the built search."""
        np = self.np
        ctx = self.ctx
        ctx.policy.load_state_dict(ck["policy"])
        ctx.policy.eval()
        P = ctx.planner
        psd = ck.get("planner") or {}
        P.load_state_dict_all(psd)
        # the recorder's planner scores with PRIMLEARN_DEFAULTS; the targets' z must be in the
        # run's own reward units
        for k_ in ("plan_progress", "plan_finish_bonus"):
            if (psd.get("cfg") or {}).get(k_) is not None:
                P.cfg[k_] = float(psd["cfg"][k_])
        P.eval_unit = P.unit          # the TRAINING progress unit (--plan-units route)
        ctx.ck = ck
        ctx.step = int(ck.get("global_step", 0))
        # the roots: the trainer's own respawn draw over the checkpoint's reservoir
        import record_ckpt
        from surfgym.respawn import RespawnBuffer
        start = np.asarray(ctx.pool)
        pay = record_ckpt.reservoir_payload(ck, ctx.map_path, strict=False)
        states = None if pay is None else pay.get("states")
        frac = min(1.0, max(0.0, float(self.a.start_frac)))
        if states is None or len(states) == 0 or frac >= 1.0:
            self.pool, self.n_start = start, len(start)
            print(f"az_worker: roots = the {len(start)} map start(s) (no reservoir in the "
                  f"checkpoint)" if frac < 1.0 else "az_worker: roots = map starts only",
                  flush=True)
            return
        rb = RespawnBuffer(1, reservoir=len(states), map_id=str(pay.get("map_id", "")),
                           seed=int(self.rng.integers(1 << 31)))
        rb.load_state_dict(pay)
        if int(ctx.cfg.get("plan_return") or 0):
            rb.weight_fn = P.return_weights       # --plan-return: Go-Explore's return
        fresh = max(frac, 1.0 / self.POOL_SIZE)
        rsp = ctx.cfg.get("respawn_speed")
        self.pool = rb.build_pool(start, pool_size=self.POOL_SIZE, fresh_frac=fresh,
                                  vel_scale=(tuple(float(v) for v in rsp) if rsp
                                             else (0.9, 1.1)),
                                  pitch_jitter=(0.0 if ctx.cfg.get("fix_pitch") is not None
                                                else 5.0))
        self.n_start = max(1, int(round(self.POOL_SIZE * fresh)))    # build_pool's layout
        print(f"az_worker: roots = {fresh:.0%} map starts + {1 - fresh:.0%} of the "
              f"reservoir's {rb.size:,} states"
              + (" (--plan-return weights)" if rb.weight_fn is not None else ""), flush=True)

    def refresh(self) -> bool:
        """(Re)load the checkpoint when it has changed and settled -> is a search possible?"""
        try:
            s = self.ck_path.stat()
        except OSError:
            return self.ctx is not None
        key = (s.st_mtime_ns, s.st_size)
        if key == self.loaded:
            return self.ctx is not None
        if time.time() - s.st_mtime < float(self.a.settle):
            return self.ctx is not None           # still being written: the old weights
        torch = self.torch
        try:
            ck = torch.load(self.ck_path, map_location="cpu", weights_only=False)
            sig = json.dumps(ck.get("config") or {}, sort_keys=True, default=str)
            if self.ctx is None or sig != self.sig:
                self.sig = None
                self._build(ck)                   # the recorder reads the file again
                self.sig = sig
            self._adopt(ck)
        except Exception as e:                    # noqa: BLE001 - a torn read: next time
            # (a SystemExit - not a primlearn checkpoint, an unmirrored config key - is not
            # caught: the worker stops loudly)
            print(f"az_worker: {self.ck_path.name} did not load ({type(e).__name__}: {e}); "
                  "retrying", flush=True)
            return self.ctx is not None
        self.loaded = key
        self.last_new = time.time()
        print(f"az_worker: searching with {self.ck_path.name} @ step {self.ctx.step:,} "
              f"(planner updates {self.ctx.planner.updates})", flush=True)
        return True

    # ------------------------------------------------------------------ one search
    def root(self):
        """-> (state (1,) STATE_DTYPE, core observation (1, obs_dim), kind) of a fresh root."""
        np = self.np
        core = self.ctx.core
        j = int(self.rng.integers(len(self.pool)))
        kind = "start" if j < self.n_start else "reservoir"
        core.set_spawn_pool(self.pool[j:j + 1])
        obs = core.reset(int(self.rng.integers(1 << 31)))
        return core.get_states(), np.array(obs[0:1], np.float32, copy=True), kind

    def search_once(self) -> dict:
        from surfgym.goalprimplan import observe
        np, torch = self.np, self.torch
        ctx = self.ctx
        P, S = ctx.planner, ctx.search
        st, obs, kind = self.root()
        fin = np.asarray(P.finish, np.float64)
        bank = np.zeros(1)
        x = observe(P.caster, st["origin"].astype(np.float64), st["velocity"].astype(np.float64),
                    st["yaw"].astype(np.float64), fin, bank,
                    frame=getattr(P.prim, "frame", "velocity"))
        t = time.time()
        with torch.no_grad():
            _u, qs, info = S.choose(st, fin, bank, x, self.gen, obs=obs)
        dt = time.time() - t
        U = np.asarray(info["root_u"], np.float32)
        q = np.asarray(qs, np.float64).reshape(-1)
        nv = np.asarray(info["visits"], np.float64).reshape(-1)
        if nv.sum() > 0.0:
            pi = nv / nv.sum()
        else:
            pi = np.zeros(len(U))
            pi[int(info["best"])] = 1.0
        z = float((pi * q).sum())
        xr = info.get("root_x")
        xr = x[0] if xr is None else np.asarray(xr, np.float32).reshape(-1)
        self.n += 1
        name = f"targets_{self.a.worker_id}_{os.getpid()}_{self.n:06d}.npz"
        tmp = self.out / (name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, x=xr.astype(np.float32), u=U, pi=pi.astype(np.float32),
                     z=np.float32(z), q=q.astype(np.float32), n=nv.astype(np.int32),
                     died=np.asarray(info["died"]).reshape(-1),
                     fin=np.asarray(info["finished"]).reshape(-1),
                     step=np.int64(ctx.step), updates=np.int64(P.updates),
                     kind=np.int8(kind == "reservoir"),
                     origin=st["origin"][0].astype(np.float32),
                     expansions=np.int32(info.get("expansions", 1)),
                     depth=np.int32(info.get("depth", 1)),
                     sims=np.int32(self.a.sims), k=np.int32(self.a.k))
        os.replace(tmp, self.out / name)
        return {"name": name, "kind": kind, "secs": dt, "K": len(U), "z": z,
                "best": int(np.argmax(pi)), "visits": nv.astype(int).tolist(),
                "depth": int(info.get("depth", 1)), "died": int(np.sum(info["died"])),
                "fin": int(np.sum(info["finished"]))}

    # ------------------------------------------------------------------ the loop
    def done(self) -> str:
        a = self.a
        if a.searches and self.n >= a.searches:
            return f"{self.n} searches"
        if not _alive(a.trainer_pid):
            return f"the trainer (pid {a.trainer_pid}) is gone"
        fin = self.run / "ckpt_final.pt"
        try:
            if fin.stat().st_mtime > self.t0 and not a.ckpt:
                return "the trainer wrote ckpt_final.pt"
        except OSError:
            pass
        if (not a.searches and float(a.idle_exit) > 0.0
                and time.time() - self.last_new > float(a.idle_exit)):
            return f"no new checkpoint for {a.idle_exit:g} s"
        return ""

    def loop(self) -> int:
        waiting = False
        t_rate, n_rate = time.time(), 0
        while True:
            why = self.done()
            if why:
                print(f"az_worker: stop ({why}); {self.n} target(s) in {self.out}", flush=True)
                return 0
            if not self.refresh():
                if not waiting:
                    print(f"az_worker: waiting for {self.ck_path}", flush=True)
                    waiting = True
                time.sleep(float(self.a.poll))
                continue
            waiting = False
            r = self.search_once()
            n_rate += 1
            print(f"az_worker: #{self.n} {r['kind']:<9} {r['secs']:5.1f}s K {r['K']} visits "
                  f"{r['visits']} -> #{r['best']} z {r['z']:+.3f} depth {r['depth']} "
                  f"died {r['died']} fin {r['fin']}", flush=True)
            el = time.time() - t_rate
            if el >= 300.0:
                print(f"az_worker: {n_rate / el * 60.0:.1f} searches/min over the last "
                      f"{el / 60.0:.0f} min", flush=True)
                t_rate, n_rate = time.time(), 0


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.device == "cpu":
        # before torch is imported: the trainer owns the GPU ("-1", not "" - an empty value is
        # deleted on Windows and leaves the GPU visible)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    nt = str(max(1, int(a.threads)))
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, nt)
    sys.path.insert(0, str(ROOT / "python"))
    import torch
    torch.set_num_threads(max(1, int(a.threads)))
    w = Worker(a)
    print(f"az_worker: {w.ck_path} -> {w.out} (sims {a.sims}, k {a.k}, device {a.device}, "
          f"start share {a.start_frac:g}, seed {w.seed})", flush=True)
    return w.loop()


if __name__ == "__main__":
    sys.exit(main())
