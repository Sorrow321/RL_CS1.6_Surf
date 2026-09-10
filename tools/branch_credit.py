#!/usr/bin/env python3
"""branch_credit.py - one branch, per decision, with the critic and the
advantage the trainer would actually assign.

Takes a recording made by ``tools/record_ckpt.py --dump-value`` (which taps
the value head for exactly the row the policy acted on) and produces, per
DECISION, everything M2 asks for on a common time axis:

    position, speed, ridable support below
    geodesic d and the dip depth in reward units
    the shaping reward  scale * (d_prev - d)  and the time penalty
    cumulative and discounted-cumulative reward
    V(s_t)   - the checkpoint's own critic
    G_t      - the EMPIRICAL discounted return of the rest of the episode
    A_t      - the advantage, three ways (tools/credit_diag.py's arithmetic,
               which is pinned against train_fast.py's own GAE loop):
                 (a) the trainer's GAE at the run's lambda and n_steps, cut
                     at rollout-buffer boundaries, averaged over the buffer
                     phase the episode could have started at;
                 (b) lambda = 1 over the whole episode - the Monte-Carlo
                     advantage G_t - V_t;
                 (c) the run's lambda, whole episode, no truncation.

Two caveats stated rather than hidden. The reward is reconstructed from the
recorded POSITIONS (shaping + time penalty), because record_ckpt's core
carries no reward function and writes 0 in the reward column; the INTRINSIC
term (--int-coef) is therefore not included, and neither is the success
bonus on an episode that never finishes. And on a ``--race-ratchet``
checkpoint the ratchet form equals the stock form wherever d is monotone,
which is every decision of the petrus episodes measured here (0 dips).

    python tools/branch_credit.py --traj runs/.../ctl.jsonl \
        --value runs/.../ctl_V.npz --map C:/RL_Surf/maps/surf_petrus_lite.bsp \
        --d0 35636.65625 --episode 0 --out runs/.../credit_ctl.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from dip_report import load_field, load_traj                 # noqa: E402
from field_probe import Field                                 # noqa: E402
from credit_diag import (nonterm_mask, discounted_returns,    # noqa: E402
                         td_residuals, gae, gae_phase_mean)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--d0", type=float, required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--act-every", type=int, default=4)
    ap.add_argument("--tick-ms", type=float, default=10.0)
    ap.add_argument("--gamma-tick", type=float, default=0.9995)
    ap.add_argument("--gae", type=float, default=0.95)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--time-pen-tick", type=float, default=0.005)
    ap.add_argument("--cell", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gf = load_field(Path(a.map), a.cell)
    fp = Field(Path(a.map), a.cell)
    foot, hdr, rows = load_traj(a.traj)[a.episode]
    sub = rows[::a.act_every]
    d = gf.sample(sub[:, 1:4]).astype(np.float64)
    scale = 100.0 / float(a.d0)
    dt = a.act_every * a.tick_ms / 1000.0
    n = len(d)

    z = np.load(a.value)
    m = z["episode"] == a.episode
    V = np.asarray(z["value"], np.float64)[m]
    if len(V) < n:
        d, sub, n = d[:len(V)], sub[:len(V)], len(V)
    V = V[:n]

    shap = np.concatenate([[0.0], scale * (d[:-1] - d[1:])])
    tp = float(a.time_pen_tick) * int(a.act_every)
    # r_t is the reward earned BY decision t, i.e. between t and t+1
    r = np.concatenate([scale * (d[:-1] - d[1:]) - tp, [-tp]])
    g = float(a.gamma_tick) ** int(a.act_every)
    nt = nonterm_mask(n, ended_last=True)
    G = discounted_returns(r, g, nt)
    delta = td_residuals(r, V, g, nt)
    A_tr, A_tr_lo, A_tr_hi = gae_phase_mean(delta, g, a.gae, nt,
                                             a.n_steps)
    A_l1 = gae(delta, g, 1.0, nt, None)
    A_lam = gae(delta, g, a.gae, nt, None)

    b = np.minimum.accumulate(d)
    dip = (d - b) * scale
    spd = np.linalg.norm(sub[:, 4:7], axis=1)
    sup = []
    for p in sub[:, 1:4]:
        s = fp.support(p, reach=192.0, surf_only=True)
        sup.append(None if not np.isfinite(s) else float(s))

    out = dict(traj=str(a.traj), episode=int(a.episode), n=int(n), dt=dt,
               d0=float(a.d0), scale=scale, gamma_dec=g, lam=float(a.gae),
               n_steps=int(a.n_steps), time_pen_dec=tp,
               end=str(foot.get("end", "")), ticks=int(foot.get("ticks", 0)),
               t=(np.arange(n) * dt).tolist(),
               xyz=sub[:, 1:4].tolist(), speed=spd.tolist(),
               support=sup, d=d.tolist(), dip=dip.tolist(),
               shaping=shap.tolist(), reward=r.tolist(),
               cum=np.cumsum(r).tolist(),
               disc_cum=np.cumsum(r * (g ** np.arange(n))).tolist(),
               V=V.tolist(), G=G.tolist(), delta=delta.tolist(),
               A_trainer=A_tr.tolist(), A_trainer_lo=A_tr_lo.tolist(),
               A_trainer_hi=A_tr_hi.tolist(), A_lambda1=A_l1.tolist(),
               A_lam_untrunc=A_lam.tolist())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")

    print(f"{Path(a.traj).name} ep{a.episode}  {n} decisions "
          f"({n * dt:.2f} s)  end={out['end']}")
    print("   t     d      dip   r      cum     V       G      V-G    "
          "A_trainer A_lam1  A_lam   spd  supp")
    step = max(1, n // 40)
    for i in list(range(0, n, step)) + [n - 1]:
        s = sup[i]
        print(f"  {i*dt:5.2f} {d[i]:7.0f} {dip[i]:5.2f} {r[i]:+6.3f} "
              f"{np.cumsum(r)[i]:+7.3f} {V[i]:7.3f} {G[i]:7.3f} "
              f"{V[i]-G[i]:+7.3f} {A_tr[i]:+8.3f} {A_l1[i]:+7.3f} "
              f"{A_lam[i]:+7.3f} {spd[i]:5.0f} "
              f"{('%.0f' % s) if s is not None else 'NONE':>5s}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
