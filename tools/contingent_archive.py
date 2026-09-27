"""contingent_archive.py - the archive AS A FEEDBACK POLICY over a stochastic option graph
(Claude + Codex, agent bus 2026-09-27 01:07-01:10Z; the user's direction: a planner that looks
ahead in the simulator).

    python tools/contingent_archive.py <ckpt> --out runs/research/contingent_<name>
        [--episodes 18] [--flights 4000] [--seed 0]

WHY. tools/edge_archive.py finds blue200's route from the map start in seconds, but a found chain
is one LUCKY sample of a stochastic graph: the executor samples its actions, a re-flown edge lands
in a cloud of states (3-12 distinct keys, ~20 u apart) and the corner edge of a typical chain dies
20-32 of 32 times from its exact parent. Committing the first move of a finishing sample, or a
fixed plan sequence, fails from the states real episodes reach (0-2/9). The decision must be made
on the DISTRIBUTION of outcomes: which move most probably ends in a finish before the deadline.

WHAT (every constant fixed once, the same on every map; no map location is named anywhere):
* The state is EXACT and CONTINUING: the real core tick, stuck_ticks, held keys and observation
  are carried through every simulated and committed flight. A flight that reaches the episode cap
  is a failure (P_finish counts only finishes BEFORE the deadline).
* FINE key (admission, exemplars): edge_archive's key (128 u XYZ x 6 horizontal-speed x 8 azimuth +
  stationary x 3 vertical x contact) + a remaining-time bin (TIME_BINS over the cap) + the held-keys
  state. Up to EXEMPLARS exact states per fine key, for expansion.
* COARSE key (statistical pooling): 256 u XYZ x 3 speed x azimuth x vertical x contact x time bin.
* CHANCE statistics per (fine key, choice) and per (coarse key, choice): deaths (or the cap),
  finishes, child keys.
* VALUES = P(finish before the deadline), by value iteration over the empirical graph: the coarse
  Q with a neutral prior (PRIOR, pseudo-count ALPHA); the fine Q shrunk toward its coarse Q with
  pseudo-count BETA; an unseen fine child takes its coarse parent's value.
* UNCERTAINTY by posterior sampling: each (fine key, choice)'s outcome distribution drawn from a
  Dirichlet over its counts + BETA mass on its coarse Q; POSTERIOR_SAMPLES value iterations; the
  root choices' UCB / LCB are the upper / lower QUANTILE.
* A DECISION (from the true state): ROOT_TRIALS flights of every choice from the exact root, then
  tranches of simulator flights allocated by optimism - the choice with the highest UCB, inside it
  the fine keys with the largest reach probability (under the plug-in policy) x 1/sqrt(1+trials) -
  until the best choice's LCB >= every other choice's UCB or the flight budget ends. COMMIT the
  choice with the highest LCB (never the optimistic number); an unresolved decision is logged. The
  committed move is flown ONCE for real, and that real outcome is also a sample.
* PERSISTENT across decisions and episodes (the statistics and exemplars accumulate); saved at the
  end (archive.pkl).

OUTPUT (<out>/): episodes.jsonl (per episode: end, seconds, decisions, per decision the position,
the plug-in Q / LCB / UCB per choice, the commitment, flights, resolved or not), summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from edge_archive import Archive, Flyer, keys_of, _ckpt_choices  # noqa: E402

TIME_BINS = 6
EXEMPLARS = 4
PRIOR = 0.5
ALPHA = 1.0             # coarse pseudo-count on the neutral prior
BETA = 4.0              # fine pseudo-count on the coarse Q
POSTERIOR_SAMPLES = 16
QUANTILE = 0.1          # LCB = the 10th percentile, UCB = the 90th
ROOT_TRIALS = 64
TRANCHE_KEYS = 32       # fine keys per tranche, x every choice x TRANCHE_REPS flights
TRANCHE_REPS = 2
REACH_DEPTH = 12        # plans the reach probability is propagated
VI_SWEEPS = 40
D_OUT, F_OUT = ("D",), ("F",)


class Graph:
    """The persistent chance graph: exemplars per fine key, outcome counts per (fine key, choice)
    and per (coarse key, choice)."""

    def __init__(self, C, mins, cap_ticks):
        self.C, self.mins, self.cap = int(C), np.asarray(mins, np.float64), int(cap_ticks)
        self.nodes = Archive()                 # exemplar storage (exact state, held keys, obs)
        self.ex = {}                           # fine key -> [node ids]
        self.coarse_of = {}                    # fine key -> coarse key
        self.cf = {}                           # (fine key, c) -> {outcome: count}
        self.cc = {}                           # (coarse key, c) -> {outcome: count}
        self.trials = {}                       # fine key -> flights started from it (all choices)

    # ------------------------------------------------------------------ keys
    def fine_key(self, st, ks):
        base = keys_of(np.array([st], dtype=st.dtype), self.mins)[0]
        rem = max(0, self.cap - int(st["tick"]))
        tb = min(TIME_BINS - 1, int(TIME_BINS * rem / max(1, self.cap)))
        held = tuple(int(v) for v in ks[0]) if ks is not None else ()
        return base + (tb,) + held

    @staticmethod
    def coarse(fk):
        cx, cy, cz, sb, az, vz, g, tb = fk[:8]
        s3 = 0 if sb <= 2 else (1 if sb <= 4 else 2)
        return (cx // 2, cy // 2, cz // 2, s3, az, vz, g, tb)

    # ------------------------------------------------------------------ admission / recording
    def admit(self, st, ks, obs):
        fk = self.fine_key(st, ks)
        self.coarse_of.setdefault(fk, self.coarse(fk))
        lst = self.ex.setdefault(fk, [])
        if len(lst) < EXEMPLARS:
            lst.append(self.nodes.add(st, ks, obs, fk, -1, -1, 0, int(st["tick"]), None))
        return fk

    def record(self, fk, c, outcome_fk):
        """One flight of choice c from fine key fk ended in outcome_fk (a fine key, D_OUT or
        F_OUT)."""
        d = self.cf.setdefault((fk, c), {})
        d[outcome_fk] = d.get(outcome_fk, 0) + 1
        ck = self.coarse_of.setdefault(fk, self.coarse(fk))
        oc = (outcome_fk if outcome_fk in (D_OUT, F_OUT)
              else self.coarse_of.setdefault(outcome_fk, self.coarse(outcome_fk)))
        e = self.cc.setdefault((ck, c), {})
        e[oc] = e.get(oc, 0) + 1

    def outcome_of(self, r):
        if r["fin"]:
            return F_OUT, None
        if r["died"] or r["end"] is None:
            return D_OUT, None
        return self.admit(r["end"], r["keys"], r["obs"]), r

    # ------------------------------------------------------------------ values
    def compile(self):
        """The graph as arrays (once per tranche): per layer the edge list (source index, choice,
        destination index or -1 death / -2 finish / -3 unseen, count) and the (source, choice)
        cells present."""
        C = self.C
        ckeys = list({ck for (ck, _c) in self.cc} | set(self.coarse_of.values()))
        cidx = {k: i for i, k in enumerate(ckeys)}
        fkeys = list(set(self.coarse_of) | {fk for (fk, _c) in self.cf})
        fidx = {k: i for i, k in enumerate(fkeys)}

        def layer(table, idx):
            src, ch, dst, n = [], [], [], []
            for (k, c), d in table.items():
                i = idx[k]
                for o, cnt in d.items():
                    src.append(i)
                    ch.append(c)
                    dst.append(-1 if o == D_OUT else (-2 if o == F_OUT else idx.get(o, -3)))
                    n.append(cnt)
            has = np.zeros((len(idx), C), bool)
            for (k, c) in table:
                has[idx[k], c] = True
            return (np.asarray(src, np.int64), np.asarray(ch, np.int64),
                    np.asarray(dst, np.int64), np.asarray(n, np.float64), has)
        cl = layer(self.cc, cidx)
        fl_ = layer(self.cf, fidx)
        f_c = np.array([cidx[self.coarse_of.get(k) or self.coarse(k)] for k in fkeys], np.int64)
        return {"C": C, "ckeys": ckeys, "cidx": cidx, "fkeys": fkeys, "fidx": fidx,
                "cl": cl, "fl": fl_, "f_c": f_c}

    @staticmethod
    def iterate(comp, rng=None, posterior=False):
        """Value iteration on a compiled graph -> (Vf (nF,), Qf (nF, C)). posterior: the outcome
        weights and the pseudo-masses are Gamma draws (one Dirichlet sample per cell)."""
        C = comp["C"]

        def weights(lay, pseudo, n_src):
            src, ch, dst, cnt, has = lay
            g = src * C + ch
            if posterior:
                w = rng.gamma(cnt + 1e-9)
                pm = rng.gamma(pseudo, size=n_src * C)
            else:
                w = cnt.copy()
                pm = np.full(n_src * C, pseudo)
            tot = np.bincount(g, weights=w, minlength=n_src * C) + pm
            return w / tot[g], (pm / tot).reshape(n_src, C)

        # an UNEXPLORED outcome (a destination with no statistics of its own) is worth PRIOR in
        # the plug-in solve; in a posterior sample it is a uniform draw - the value of territory
        # nobody has flown is unknown, so a choice that leads there has a WIDE LCB-UCB band and
        # the allocation must fly it before the decision can resolve
        def vi(lay, wn, pw, qprior, v0):
            src, ch, dst, cnt, has = lay
            V = v0.copy()
            Q = qprior.copy()
            unseen_v = (rng.uniform(0.0, 1.0, size=len(dst)) if posterior
                        else np.full(len(dst), PRIOR))
            for _ in range(VI_SWEEPS):
                vd = np.where(dst == -1, 0.0, np.where(dst == -2, 1.0,
                                                       np.where(dst >= 0, V[np.maximum(dst, 0)],
                                                                unseen_v)))
                Q = pw * qprior
                if len(src):
                    np.add.at(Q, (src, ch), wn * vd)
                Q = np.where(has, Q, qprior)
                Vn = Q.max(1) if len(Q) else V
                if np.allclose(Vn, V, atol=1e-6):
                    V = Vn
                    break
                V = Vn
            return V, Q
        nC, nF = len(comp["ckeys"]), len(comp["fkeys"])
        wc, pwc = weights(comp["cl"], ALPHA, nC)
        # the coarse prior: PRIOR in the plug-in solve, a uniform draw per (coarse key, choice) in
        # a posterior sample (the same reasoning: unexplored = unknown, not 0.5)
        q0 = (rng.uniform(0.0, 1.0, size=(nC, C)) if posterior else np.full((nC, C), PRIOR))
        Vc, Qc = vi(comp["cl"], wc, pwc, q0, q0.max(1) if nC else np.zeros(0))
        f_c = comp["f_c"]
        qprior = Qc[f_c] if (nC and nF) else np.full((nF, C), PRIOR)
        wf, pwf = weights(comp["fl"], BETA, nF)
        Vf, Qf = vi(comp["fl"], wf, pwf, qprior, qprior.max(1) if nF else np.zeros(0))
        return Vf, Qf

    def root_q(self, fk, rng):
        """-> (plug-in Q (C,), LCB (C,), UCB (C,), the compiled graph, its plug-in Qf) at fk."""
        comp = self.compile()
        Vf, Qf = self.iterate(comp)
        i = comp["fidx"].get(fk)
        q = Qf[i] if i is not None else np.full(self.C, PRIOR)
        samp = []
        for _ in range(POSTERIOR_SAMPLES):
            _v, Q2 = self.iterate(comp, rng=rng, posterior=True)
            samp.append(Q2[i] if i is not None else np.full(self.C, PRIOR))
        S = np.asarray(samp)
        return (q, np.quantile(S, QUANTILE, axis=0), np.quantile(S, 1.0 - QUANTILE, axis=0),
                comp, Qf)

    def reach(self, fk, c0, comp, Qf):
        """Probability mass reaching each fine key within REACH_DEPTH plans: choice c0 at fk,
        then the plug-in policy."""
        fidx = comp["fidx"]
        pol = Qf.argmax(1) if len(Qf) else np.zeros(0, np.int64)
        mass = {}
        frontier = {(fk, c0): 1.0}
        for _ in range(REACH_DEPTH):
            nxt = {}
            for (k, c), m in frontier.items():
                d = self.cf.get((k, c))
                if not d:
                    continue
                tot = float(sum(d.values()))
                for o, n in d.items():
                    if o in (D_OUT, F_OUT):
                        continue
                    pm = m * n / tot
                    if pm < 1e-6:
                        continue
                    mass[o] = mass.get(o, 0.0) + pm
                    j = fidx.get(o)
                    c2 = int(pol[j]) if j is not None else 0
                    nxt[(o, c2)] = nxt.get((o, c2), 0.0) + pm
            frontier = nxt
            if not frontier:
                break
        return mass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=None)
    ap.add_argument("--episodes", type=int, default=18)
    ap.add_argument("--flights", type=int, default=4000,
                    help="the simulator-flight budget of ONE decision (after the root trials)")
    ap.add_argument("--parents", type=int, default=64, help="scratch core size / choices")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--load", default=None, help="an archive.pkl to continue from")
    a = ap.parse_args(argv)
    import record_ckpt
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(a.seed))
    C = _ckpt_choices(a.ckpt)
    S = int(a.parents) * C
    rargv = [str(a.ckpt), "--episodes", "1", "--stochastic", "--plan-scratch", str(S)]
    if a.map:
        rargv += ["--map", str(a.map)]
    ctx = record_ckpt.build(rargv, device=a.device)
    fl = Flyer(ctx)
    P = ctx.planner
    fin = np.asarray(P.finish, np.float64)
    mins = np.asarray(ctx.core.map_bounds()[0], np.float64)
    tick_s = float(ctx.tick.ms) / 1000.0
    cap = int(getattr(getattr(ctx.core, "config", None), "max_episode_ticks", 0) or 0)
    if cap <= 0:
        cap = int(round(float((ctx.cfg or {}).get("ep_secs") or 30.0) / tick_s))
    G = Graph(C, mins, cap)
    if a.load:
        G = pickle.loads(Path(a.load).read_bytes())
    print(f"contingent_archive: {a.ckpt} on {Path(ctx.map_path).name}; {C} choices, cap {cap} ticks "
          f"({cap * tick_s:.0f} s); root trials {ROOT_TRIALS}/choice, {a.flights} flights per "
          f"decision, LCB/UCB = the {QUANTILE:.0%}/{1 - QUANTILE:.0%} posterior quantiles of "
          f"{POSTERIOR_SAMPLES} samples", flush=True)
    pool = np.asarray(ctx.pool)
    core1 = ctx.core
    epf = open(out / "episodes.jsonl", "w", encoding="utf-8")
    results = []
    for e in range(int(a.episodes)):
        j = e % max(1, len(pool))
        core1.set_spawn_pool(pool[j:j + 1])
        obs0 = core1.reset(int(a.seed) * 1000 + e)
        st = core1.get_states()[0].copy()
        ks, ob = fl.fresh_keys(), np.asarray(obs0)[0]
        dec = []
        end = "cap"
        t_ep = time.time()
        while True:
            fk = G.admit(st, ks, ob)
            root_id = G.nodes.add(st, ks, ob, fk, -1, -1, 0, int(st["tick"]), None)
            t0 = time.time()
            flights = 0
            # 1. the root trials: every choice ROOT_TRIALS times from the exact true state
            jobs = [(root_id, c) for c in range(C) for _ in range(ROOT_TRIALS)]
            for c0 in range(0, len(jobs), fl.S):
                chunk = jobs[c0:c0 + fl.S]
                for (nid, c), r in zip(chunk, fl.fly(chunk, G.nodes, fin, record_path=False,
                                                     keep_clock=True)):
                    o, _ = G.outcome_of(r)
                    G.record(fk, c, o)
                flights += len(chunk)
            # 2. tranches by optimism until the LCB separates or the budget ends
            resolved = False
            q, lcb, ucb, comp, Qf = G.root_q(fk, rng)
            while flights < ROOT_TRIALS * C + int(a.flights):
                best = int(np.argmax(lcb))
                if all(lcb[best] >= ucb[c] for c in range(C) if c != best):
                    resolved = True
                    break
                c_opt = int(np.argmax(ucb))
                reach = G.reach(fk, c_opt, comp, Qf)
                cand = [(m / math.sqrt(1.0 + G.trials.get(k, 0)), k) for k, m in reach.items()
                        if k in G.ex]
                cand.sort(reverse=True)
                keys = [k for _p, k in cand[:TRANCHE_KEYS]]
                if not keys:
                    # nothing downstream yet: more root trials of the optimistic choice
                    jobs = [(root_id, c_opt)] * fl.S
                    srcs = [fk] * fl.S
                else:
                    jobs, srcs = [], []
                    for i_, k in enumerate(keys):
                        exs = G.ex[k]
                        for c in range(C):
                            for rep in range(TRANCHE_REPS):
                                jobs.append((exs[(rep + i_) % len(exs)], c))
                                srcs.append(k)
                    jobs, srcs = jobs[:fl.S], srcs[:fl.S]
                for (nid, c), k, r in zip(jobs, srcs, fl.fly(jobs, G.nodes, fin,
                                                             record_path=False, keep_clock=True)):
                    o, _ = G.outcome_of(r)
                    G.record(k, c, o)
                    G.trials[k] = G.trials.get(k, 0) + 1
                flights += len(jobs)
                q, lcb, ucb, comp, Qf = G.root_q(fk, rng)
            move = int(np.argmax(lcb))
            # 3. the real flight (also a sample)
            r = fl.fly([(root_id, move)], G.nodes, fin, keep_clock=True)[0]
            o, _ = G.outcome_of(r)
            G.record(fk, move, o)
            dec.append({"t": round(float(st["tick"]) * tick_s, 2),
                        "pos": np.round(st["origin"].astype(np.float64), 0).tolist(),
                        "q": [round(float(v), 3) for v in q],
                        "lcb": [round(float(v), 3) for v in lcb],
                        "ucb": [round(float(v), 3) for v in ucb],
                        "move": move, "resolved": resolved, "flights": flights,
                        "secs": round(time.time() - t0, 2)})
            if r["fin"]:
                end = "finish"
                t_end = (int(st["tick"]) + r["ticks"]) * tick_s
                break
            if r["died"] or r["end"] is None:
                end = "died" if int(st["tick"]) + r["ticks"] < cap else "cap"
                t_end = (int(st["tick"]) + r["ticks"]) * tick_s
                break
            st, ks, ob = r["end"], r["keys"], r["obs"]
        rec = {"episode": e, "end": end, "secs": round(t_end, 2), "decisions": len(dec),
               "unresolved": sum(1 for d in dec if not d["resolved"]),
               "wall_secs": round(time.time() - t_ep, 1), "fine_keys": len(G.ex),
               "decisions_log": dec}
        epf.write(json.dumps(rec) + "\n")
        epf.flush()
        results.append(rec)
        print(f"episode {e}: {end.upper()} at {rec['secs']:.1f} s, {len(dec)} decisions "
              f"({rec['unresolved']} unresolved), {rec['wall_secs']:.0f} s wall; graph "
              f"{len(G.ex):,} fine keys, {len(G.cf):,} (key, choice) cells", flush=True)
    epf.close()
    fins = [r for r in results if r["end"] == "finish"]
    half = len(results) // 2
    summ = {"episodes": len(results), "finished": len(fins),
            "finished_first_half": sum(1 for r in results[:half] if r["end"] == "finish"),
            "finished_second_half": sum(1 for r in results[half:] if r["end"] == "finish"),
            "finish_secs": [r["secs"] for r in fins], "ckpt": str(a.ckpt),
            "map": Path(ctx.map_path).name, "flights_per_decision": int(a.flights),
            "constants": {"TIME_BINS": TIME_BINS, "EXEMPLARS": EXEMPLARS, "PRIOR": PRIOR,
                          "ALPHA": ALPHA, "BETA": BETA, "POSTERIOR_SAMPLES": POSTERIOR_SAMPLES,
                          "QUANTILE": QUANTILE, "ROOT_TRIALS": ROOT_TRIALS,
                          "TRANCHE_KEYS": TRANCHE_KEYS, "TRANCHE_REPS": TRANCHE_REPS,
                          "REACH_DEPTH": REACH_DEPTH}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    (out / "archive.pkl").write_bytes(pickle.dumps(G))
    print(f"contingent_archive: {len(fins)}/{len(results)} finished from the map start "
          f"(first half {summ['finished_first_half']}, second half "
          f"{summ['finished_second_half']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
