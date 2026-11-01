#!/usr/bin/env python3
"""cem_finish.py - per-decision CEM trajectory planning for a goal entry from the policy's OWN states.

tools/macro_finish.py samples a few HELD macros; a launch that has to turn a fast line up a ramp
and over an edge needs every decision. This plans the whole sequence: for each root (a state of
the policy's own frontier pool), a population of --pop action sequences of --horizon decisions
at the policy's own decision rate (act_every ticks, the checkpoint's), each decision a side key,
a forward key, duck and a yaw TARGET offset from the velocity heading (the policy's own action
frame, --view-absolute velocity). The sampling distribution is per decision: categoricals over
the keys and a Gaussian over the offset; each iteration keeps the --elite sequences that came
closest to the goal (macro_finish.goal_score: the field, the hull-inflated trigger boxes and the
origins the hull fits in) and refits the distribution to them (smoothed, sigma floored) - the
cross-entropy method (Rubinstein 1999) as a planner in the simulator. A goal entry is replayed on
a 1-env core (exact, same tick) and saved in the layout tools/finish_pool.py reads.

Nothing reads the map beyond the goal field and the finish triggers; the constants are search
scales set once.

    python tools/cem_finish.py CKPT POOL.npy --roots 47175,27925 --out runs/cem/a
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "autoresearch"))

import macro_finish as MF  # noqa: E402


class Dist:
    """Per-decision sampling distribution: side / fwd categoricals, duck Bernoulli, yaw Gaussian."""

    def __init__(self, H: int, yaw_sd: float, p_duck: float):
        self.ps = np.full((H, 3), 1.0 / 3.0)
        self.pf = np.full((H, 3), 1.0 / 3.0)
        self.pd = np.full(H, p_duck)
        self.mu = np.zeros(H)
        self.sd = np.full(H, yaw_sd)

    def sample(self, rng, n: int):
        H = len(self.mu)
        side = (rng.random((n, H, 1)) > np.cumsum(self.ps, 1)[None]).sum(-1)
        fwd = (rng.random((n, H, 1)) > np.cumsum(self.pf, 1)[None]).sum(-1)
        duck = rng.random((n, H)) < self.pd[None]
        yaw = self.mu[None] + self.sd[None] * rng.standard_normal((n, H))
        return np.minimum(side, 2), np.minimum(fwd, 2), duck, yaw

    def refit(self, side, fwd, duck, yaw, alpha: float, sd_min: float, p_min: float):
        H = len(self.mu)
        for k in range(3):
            self.ps[:, k] = (1 - alpha) * self.ps[:, k] + alpha * (side == k).mean(0)
            self.pf[:, k] = (1 - alpha) * self.pf[:, k] + alpha * (fwd == k).mean(0)
        for p in (self.ps, self.pf):
            np.maximum(p, p_min, out=p)
            p /= p.sum(1, keepdims=True)
        self.pd = np.clip((1 - alpha) * self.pd + alpha * duck.mean(0), p_min, 1 - p_min)
        self.mu = (1 - alpha) * self.mu + alpha * yaw.mean(0)
        self.sd = np.maximum(sd_min, (1 - alpha) * self.sd + alpha * yaw.std(0))
        return H


def run_pop(core, root_state, side, fwd, duck, yaw, K, gf, boxes, reach, d_every=4,
            perturb=None):
    """Roll every sequence from the root -> (fin tick (n,), min score (n,)). ``perturb`` =
    (rng, sigma_pos, sigma_vel, R): each env starts from the root with Gaussian noise on its
    origin and velocity, except every R-th (env % R == 0: the exact copy) - the --robust copies."""
    n, H = side.shape
    core.reset(0)
    for e in range(n):
        if perturb is None or e % perturb[3] == 0:
            core.set_state(e, root_state)
        else:
            rng, sp, sv, _R = perturb
            st = root_state.copy()
            st["origin"] += rng.normal(0.0, sp, 3).astype(np.float32)
            st["velocity"] += rng.normal(0.0, sv, 3).astype(np.float32)
            core.set_state(e, st)
    alive = np.ones(n, bool)
    fin = np.zeros(n, np.int32)
    best = np.full(n, np.inf)
    sv = core.states_view
    act = np.zeros((n, 6), np.int32)
    act[:, 0] = 7
    act[:, 1] = 3
    view = np.zeros((n, 2), np.float32)
    for t in range(H * K):
        j = t // K
        act[:, 2] = fwd[:, j]
        act[:, 3] = side[:, j]
        act[:, 5] = duck[:, j]
        view[:, 0] = yaw[:, j]
        _, _, done, trunc, _ = core.step(act, view=view)
        gh = np.asarray(core.goal_hits, bool) & alive
        fin[gh] = t + 1
        ended = ((np.asarray(done) | np.asarray(trunc)) != 0) & alive
        alive &= ~(ended | gh)
        if t % d_every == 0 or t == H * K - 1:
            o = np.asarray(sv["origin"], np.float64)
            d = MF.goal_score(o, sv["ducked"], gf, boxes, reach)
            best = np.where(alive, np.minimum(best, d), best)
    best[fin > 0] = 0.0
    return fin, best


def replay_seq(build, root_state, side, fwd, duck, yaw, K, fin_tick):
    core = build(1)
    core.reset(0)
    core.set_state(0, root_state)
    act = np.zeros((1, 6), np.int32)
    act[0, 0], act[0, 1] = 7, 3
    view = np.zeros((1, 2), np.float32)
    states, ok = [], False
    for t in range(int(fin_tick)):
        states.append(core.get_states()[0].copy())
        j = t // K
        act[0, 2], act[0, 3], act[0, 5] = fwd[j], side[j], duck[j]
        view[0, 0] = yaw[j]
        _, _, done, trunc, _ = core.step(act, view=view)
        if bool(np.asarray(core.goal_hits, bool)[0]):
            ok = (t + 1) == int(fin_tick)
            break
        if bool(np.asarray(done)[0]) or bool(np.asarray(trunc)[0]):
            break
    return np.array(states), ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("pool")
    ap.add_argument("--roots", required=True, help="comma-separated pool row indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pop", type=int, default=4096)
    ap.add_argument("--horizon", type=float, default=2.5, help="seconds")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--elite", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--yaw-sd", type=float, default=10.0)
    ap.add_argument("--sd-min", type=float, default=0.5)
    ap.add_argument("--p-min", type=float, default=0.02)
    ap.add_argument("--p-duck", type=float, default=0.1)
    ap.add_argument("--restarts", type=int, default=1, help="independent CEM runs per root")
    ap.add_argument("--max-fin", type=int, default=8, help="stop after this many finishes in all")
    ap.add_argument("--root-fin", type=int, default=0,
                    help="> 0: move on to the next root after this many finishes from one root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reach-cache", default=None)
    ap.add_argument("--init", default=None,
                    help="a fin_*.npz of the (single) root to start the distribution from")
    ap.add_argument("--init-p", type=float, default=0.8,
                    help="--init: the probability put on the line's own key bins")
    ap.add_argument("--robust", type=int, default=1,
                    help="> 1: score each sequence from this many noisy copies of the root")
    ap.add_argument("--robust-pos", type=float, default=2.0)
    ap.add_argument("--robust-vel", type=float, default=5.0)
    a = ap.parse_args(argv)

    import torch
    import beam_tas as BT
    from surfgym.tick import TickClock
    from surfgym.zones import load_zones
    import score as S

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    tick_ms = float(cfg.get("tick_ms") or 10.0)
    tick = TickClock(tick_ms)
    tick_s = tick_ms / 1000.0
    K = int(cfg.get("act_every") or 4)
    if int(cfg.get("view_continuous") or 0) != 1 or cfg.get("view_absolute") != "velocity":
        raise SystemExit("cem_finish: velocity-frame view targets need a --view-absolute "
                         "velocity checkpoint")
    map_path = str(ROOT / "maps" / f"{cfg['map']}.bsp")
    zones = load_zones(map_path)

    def build(n):
        c = BT.build_sim(cfg, map_path, n, int(cfg.get("ep_ticks") or 130000), tick=tick)
        c.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
        c.set_teleport_fail(True)
        return c

    gf = S.field()
    _end = zones["end"]
    boxes = [(b["mins"], b["maxs"]) for b in (_end.get("boxes") or [_end])]
    reach = MF.Reach(build(1), boxes, cache=Path(a.reach_cache) if a.reach_cache else None)
    pool = np.load(a.pool, allow_pickle=False).copy()
    pool["tick"] = 0
    pool["stuck_ticks"] = 0
    H = max(1, int(round(a.horizon / (K * tick_s))))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    core = build(a.pop)
    roots = [int(x) for x in a.roots.split(",") if x.strip()]
    print(f"cem_finish: roots {roots}, pop {a.pop}, horizon {H} decisions x {K} ticks "
          f"({H * K * tick_s:.2f}s), {a.iters} iterations, elite {a.elite}", flush=True)
    n_fin, t0 = 0, time.time()
    summary = []
    for r in roots:
        n_root = 0
        for rs in range(a.restarts):
            dist = Dist(H, a.yaw_sd, a.p_duck)
            if a.init:
                # --init: start from a line already found (a fin_*.npz of this root): the keys'
                # categoricals peaked on its keys, the yaw mean on its yaw, sigma --yaw-sd
                zi = np.load(a.init)
                if int(zi["root"]) != r:
                    raise SystemExit(f"--init {a.init} is a line of root {int(zi['root'])}, not {r}")
                n0 = min(H, len(zi["side"]))
                for arr, key in ((dist.ps, "side"), (dist.pf, "fwd")):
                    arr[:n0] = (1.0 - a.init_p) / 2.0
                    arr[np.arange(n0), np.asarray(zi[key][:n0], np.int64)] = a.init_p
                dist.pd[:n0] = np.where(np.asarray(zi["duck"][:n0]) > 0, a.init_p, 1 - a.init_p)
                dist.mu[:n0] = np.asarray(zi["yaw"][:n0], np.float64)
            rst = pool[r:r + 1]
            best_ever = np.inf
            for it in range(a.iters):
                side, fwd, duck, yaw = dist.sample(rng, a.pop // max(1, a.robust))
                if a.robust > 1:
                    # --robust M: every sequence runs from M copies of the root - copy 0 exact, the
                    # others with Gaussian noise on origin and velocity; its score is the MEAN of
                    # the copies' closest approaches (a finish counts 0), so the elite are the
                    # sequences that finish from a neighbourhood of the root. A finish of copy 0
                    # is an exact one and is replayed and saved like any other.
                    R = a.robust
                    sideR, fwdR, duckR, yawR = (np.repeat(x, R, axis=0)
                                                for x in (side, fwd, duck, yaw))
                    finR, scR = run_pop(core, rst, sideR, fwdR, duckR, yawR, K, gf, boxes,
                                        reach, perturb=(rng, a.robust_pos, a.robust_vel, R))
                    sc = scR.reshape(-1, R).mean(axis=1)
                    frac = (finR.reshape(-1, R) > 0).mean(axis=1)
                    fin = finR.reshape(-1, R)[:, 0].copy()
                    print(f"   robust: best finish share {frac.max():.0%}, "
                          f"{int((frac > 0).sum())} sequences finish from some copy", flush=True)
                else:
                    fin, sc = run_pop(core, rst, side, fwd, duck, yaw, K, gf, boxes, reach)
                order = np.argsort(sc)
                el = order[:a.elite]
                best_ever = min(best_ever, float(sc[order[0]]))
                hits = np.nonzero(fin > 0)[0]
                room = a.max_fin - n_fin
                if a.root_fin > 0:
                    room = min(room, a.root_fin - n_root)
                for e in hits[:max(0, room)]:
                    st, ok = replay_seq(build, rst, side[e], fwd[e], duck[e], yaw[e], K,
                                        int(fin[e]))
                    n_fin += 1
                    n_root += 1
                    print(f"FINISH #{n_fin}: root {r} restart {rs} iter {it} tick {int(fin[e])} "
                          f"({fin[e] * tick_s:.2f}s) replay {'ok' if ok else 'DIVERGED'}",
                          flush=True)
                    np.savez(out / f"fin_{n_fin:03d}.npz", states=st, finished=np.int32(1),
                             replay_ok=np.int32(1 if ok else 0), root=np.int64(r),
                             side=side[e], fwd=fwd[e], duck=duck[e], yaw=yaw[e],
                             act_every=np.int32(K), fin_tick=np.int32(fin[e]),
                             tick_ms=np.float64(tick_ms), pool=np.str_(str(a.pool)),
                             ckpt=np.str_(str(a.ckpt)))
                print(f"root {r} rs {rs} it {it:2d}: best {sc[order[0]]:7.1f} elite<= "
                      f"{sc[el[-1]]:7.1f} median {np.median(sc):7.1f} finishes {len(hits)} "
                      f"yaw sd {dist.sd.mean():5.2f} [{time.time() - t0:5.0f}s]", flush=True)
                if n_fin >= a.max_fin or (a.root_fin > 0 and n_root >= a.root_fin):
                    break
                dist.refit(side[el], fwd[el], duck[el], yaw[el], a.alpha, a.sd_min, a.p_min)
            summary.append({"root": r, "restart": rs, "best": best_ever, "finishes": n_root})
            (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
            if n_fin >= a.max_fin or (a.root_fin > 0 and n_root >= a.root_fin):
                break
        if n_fin >= a.max_fin:
            break
    print(f"cem_finish: {n_fin} finish(es) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
