#!/usr/bin/env python3
"""transplant_view.py - a DISCRETE checkpoint -> a --view-continuous one.

The continuous policy (docs/contyaw.md, surfgym/view.py) keeps every tensor
of the discrete one - the trunk, both towers, action_head at its full
width, value_head - and adds two: ``view_head`` (the (yaw, pitch) mean of
the pre-tanh z, a Linear over the pi tower) and ``view_std.log_std`` (a
state-independent log sigma per head). This tool copies the shared weights
bit for bit and FITS the two new ones by distillation from the discrete
yaw/pitch heads, on observations the discrete policy itself visits:

1. roll the discrete checkpoint on N envs (envs [0, G) greedy, the rest
   sampling its own distribution) from a spawn pool that mixes the map
   start with the checkpoint's OWN respawn reservoir, so the rows cover the
   whole map and not only the first seconds; at every decision keep the pi
   tower's input and the six categorical heads' logits;
2. per row, the target z is the categorical head's MEAN in physical units
   mapped back through the warp: z_yaw = atanh(warp_inv(E[K])) with E[K]
   the expectation of the bin table under p(yaw), and z_pitch =
   atanh(E[pitch] / pitch_rate); the categorical SPREAD (the std of the
   bins' own z under p) seeds log sigma;
3. least squares (ridge) of z on the pi tower's output gives view_head; log
   sigma is the mean categorical spread per head, floored at exp(-3).

That linear readout is the design as specified, and it is NOT enough on the
reference checkpoint (measured 2026-09-06: R^2 0.80 / 0.71 in z, the
nearest bin of the fitted mean equals the greedy bin on 32% of rows, and
the greedy transplant dies within 8 s of the spawn 9/9). The argmax of 15
logits that are linear in the tower output is piecewise CONSTANT in it, and
no single Linear fits a staircase. So, as a documented deviation:

4. ``--finetune-steps N`` (recommended) DISTILS the actor tower: pi,
   action_head and view_head are trained together on the captured rows to
   minimise MSE(mu, z_target) + k_coef * MSE in PHYSICAL units (the applied
   K clipped to the +-3 surfing band, and the pitch in deg/tick, each over
   its own scale) + KL(discrete heads || new heads) over all six
   categorical heads (the four that stay live under the flag are what
   matters; the two dead ones ride along so the tensor stays coherent).
   The physical term exists because the warp is 4.6x steeper at K = +-1
   than at zero: a z error that is small at zero is a whole bin at the
   strafe optimum, and a z-only fit dies within 5 s of the spawn
   (measured). The conv trunk, the value tower, value_head and the seeded
   log sigma are untouched, bit for bit - the critic is exactly the
   source's - and the KL term keeps fwd/side/jump/duck at the discrete
   distributions.
5. ``--dagger-rounds R``: after the fit, roll the STUDENT (greedy envs +
   sampling envs) from the same pool, label its rows with the TEACHER's
   heads (the trunk is shared and frozen, so one captured trunk output
   serves both) and fine-tune again on the union - DAgger (Ross et al.
   2011) against the covariate shift a greedy mean inherits from a
   bang-bang teacher: the student's own drifted states are exactly the
   ones the teacher never visits and the first fit never sees.

The optimizer state is carried over with the two new parameters appended
to Adam's group with no moments (train_fast.widen_for_yawcond's rule; the
fine-tuned tensors keep their old moments, which are stale by the size of
the fine-tune); ``global_step``, the reservoir and the novelty counts are
copied as they are; the config gains ``view_continuous: 1`` plus a
``view_transplant`` provenance block with every number above.

    python tools/transplant_view.py <ckpt_in> <ckpt_out> \\
        --map C:/RL_Surf/maps/surf_src_cannonball.bsp --finetune-steps 4000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
import torch.nn.functional as F

from surfgym.bc import N_SCALAR, make_eval_feeds
from surfgym.core import PITCH_BINS, STATE_DTYPE, YAW_BINS
from surfgym.view import (K_BINS, U_CLIP, bin_view_moments, warp, warp_inv,
                          z_from_u)


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def say(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def pi_input(policy, x):
    """The pi tower's INPUT for a fused obs row batch (Policy.heads' f_pi,
    flat / no rnn / no priv): the trunk output plus the route-side scalar
    block unless the critic alone reads it."""
    scal, img = x[:, :policy.scal_dim], x[:, policy.scal_dim:]
    f = policy.features(scal, img)
    if policy.route_dim and not policy.route_critic_only:
        return torch.cat([f, scal[:, N_SCALAR:]], dim=1)
    return f


def capture_class():
    from train_fast import (N_VIEW, SampledTorchPolicy, sample_padded,
                            sample_view, split_view)

    class Capture(SampledTorchPolicy):
        """Rolls a driver - the discrete TEACHER (``student=None``) or the
        continuous STUDENT - with envs [0, n_greedy) greedy and the rest
        sampling, and keeps, per decision, the (shared, frozen) trunk's pi
        input and the TEACHER's flat logits of every env (CPU, float32)."""

        def __init__(self, *a, n_greedy: int = 0, student=None,
                     pitch_max: float = 10.0, **kw):
            super().__init__(*a, **kw)
            self._ng = int(n_greedy)
            self._student = student
            self._pm = float(pitch_max)
            self.rows_f, self.rows_lg, self.rows_bin = [], [], []

        @torch.inference_mode()
        def _decide(self, obs):
            x = self._obs(obs)
            f_pi = pi_input(self.policy, x)            # teacher trunk == student trunk
            logits = self.policy.action_head(self.policy.pi(f_pi))
            padded = self._mask_padded(self.packer.pad(logits.float()), obs)
            self.rows_f.append(f_pi.float().to("cpu"))
            self.rows_lg.append(logits.float().to("cpu"))
            self.rows_bin.append(padded[:, :N_VIEW, :].argmax(-1).to("cpu"))
            if self._student is None:
                act, _ = sample_padded(padded)
                if self._ng > 0:
                    act[:self._ng] = padded[:self._ng].argmax(-1)
                a = act.to("cpu").numpy().astype(np.int32)
                self._mask_note(a)
                self.view = None
                return a
            st = self._student
            t_s = st.pi(f_pi)
            cat, mu = split_view(torch.cat([st.action_head(t_s),
                                            st.view_head(t_s)], dim=-1).float())
            ps = self._mask_padded(self.packer.pad(cat), obs)
            act, z, _ = sample_view(ps, mu, st.log_std())
            if self._ng > 0:
                act[:self._ng, N_VIEW:] = ps[:self._ng, N_VIEW:].argmax(-1)
                z[:self._ng] = mu[:self._ng]
            return self._finish_view(act[:, N_VIEW:], z)

    return Capture


def mixed_pool(start_pool, reservoir, frac: float, cap: int, rng):
    """A spawn pool that draws a state from the checkpoint's reservoir with
    probability ``frac`` and from the map start otherwise (the core draws
    uniformly over entries, so the shares are set by replication)."""
    start = np.asarray(start_pool, STATE_DTYPE)
    if reservoir is None or len(reservoir) == 0 or frac <= 0.0:
        return start
    res = np.asarray(reservoir, STATE_DTYPE)
    if len(res) > cap:
        res = res[rng.choice(len(res), cap, replace=False)]
    n_start = max(len(start), int(round(len(res) * (1.0 - frac) / frac)))
    reps = max(1, int(round(n_start / len(start))))
    return np.concatenate([np.concatenate([start] * reps), res])


def fit_ridge(T, Z, ridge: float):
    """W, b minimising ||T W^T + b - Z||^2 + ridge ||W||^2 (float64)."""
    A = torch.cat([T.double(), torch.ones(T.shape[0], 1, dtype=torch.float64,
                                          device=T.device)], dim=1)
    G = A.T @ A
    lam = torch.full((A.shape[1],), float(ridge), dtype=torch.float64,
                     device=A.device)
    lam[-1] = 0.0                                   # do not shrink the bias
    Wb = torch.linalg.solve(G + torch.diag(lam), A.T @ Z.double())
    return Wb[:-1].T.contiguous(), Wb[-1].contiguous()   # (2, H), (2,)


def head_kl(lg_p, lg_q):
    """Sum over the six factored heads of KL(softmax(lg_p) || softmax(lg_q))
    on flat (B, sum(NVEC)) logits, mean over rows."""
    from train_fast import NVEC
    kl = 0.0
    o = 0
    for n in NVEC:
        lp = F.log_softmax(lg_p[:, o:o + n], dim=-1)
        lq = F.log_softmax(lg_q[:, o:o + n], dim=-1)
        kl = kl + (lp.exp() * (lp - lq)).sum(-1)
        o += n
    return kl.mean()


def transplant(ckpt_in, ckpt_out, map_path=None, envs: int = 256,
               ticks: int = 2400, n_greedy: int = 64, seed: int = 0,
               reservoir_frac: float = 0.7, reservoir_cap: int = 20000,
               ridge: float = 1e-3, min_log_std: float = -3.0,
               finetune_steps: int = 0, finetune_lr: float = 3e-4,
               finetune_batch: int = 4096, kl_coef: float = 1.0,
               k_coef: float = 4.0, dagger_rounds: int = 0,
               device: str = "auto", audit: bool = True) -> dict:
    import expert_dagger
    from train_fast import N_VIEW, Policy, split_view, warp_t

    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if device == "auto" else device)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    src_md5 = md5(ckpt_in)
    B = expert_dagger.load_bundle(ckpt_in, map_path, dev, audit=audit)
    cfg = dict(B["cfg"])
    if cfg.get("view_continuous"):
        raise SystemExit(f"{ckpt_in} is already a --view-continuous checkpoint")
    if cfg.get("yaw_cond"):
        raise SystemExit("--yaw-cond checkpoints cannot be transplanted: the "
                         "side key conditions on a yaw BIN")
    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    K = B["K"]
    policy_d = B["policy"]
    yaw_adaptive = bool(cfg.get("yaw_adaptive"))
    pitch_max = float(B["core1"].config.pitch_rate_max_deg)
    say(f"source {ckpt_in} md5 {src_md5} step {B['step']:,} act_every {K} "
        f"yaw_adaptive {yaw_adaptive} pitch rate {pitch_max:g} deg/tick "
        f"device {dev}")

    # ---- 1. observations the discrete policy visits --------------------
    res = None
    r = ck.get("respawn") or {}
    if isinstance(r, dict):
        res = r.get("states")
        if res is None:                       # multi-map layout
            for v in r.values():
                if isinstance(v, dict) and v.get("states") is not None:
                    res = v["states"]
                    break
    pool = mixed_pool(B["pool"], res, float(reservoir_frac),
                      int(reservoir_cap), rng)
    n = int(envs)
    core = expert_dagger.open_core(B, n)
    core.set_spawn_pool(pool)
    say(f"spawn pool: {len(pool)} entries "
        + ("(no reservoir)" if res is None else
           f"({len(res):,} reservoir states, share {reservoir_frac:g})"))
    Cap = capture_class()

    def rollout(student, rseed, tag):
        """One capture rollout: the teacher drives (student None) or the
        student does; every row carries the teacher's labels."""
        _slot, rf, lf = make_eval_feeds(cfg, B["gf"], B["d0"], K,
                                        tick_ms=B["tick"].requested_ms)
        pol = Cap(policy_d, B["packer"], dev, B["lidar"], core, K, 1,
                  extra_slot=B["slot"], extra_fn=rf, latch_fn=lf,
                  pitch_fixed=B["pitch_fixed"], n_greedy=int(n_greedy),
                  student=student, pitch_max=pitch_max)
        t0 = time.time()
        obs = core.reset(int(rseed))
        n_end = 0
        for _t in range(int(ticks)):
            a = pol.act(obs)
            v = pol.view
            obs, _rew, done, trunc, _term = (core.step(a) if v is None
                                             else core.step(a, view=v))
            n_end += int((np.asarray(done, bool)
                          | np.asarray(trunc, bool)).sum())
        FP = torch.cat(pol.rows_f, 0)                  # (R, feat + route)
        LG = torch.cat(pol.rows_lg, 0)                 # (R, sum(NVEC))
        BIN = torch.cat(pol.rows_bin, 0)               # (R, 2)
        say(f"rollout [{tag}]: {n} envs x {ticks} ticks ({ticks // K} "
            f"decisions) -> {int(FP.shape[0]):,} rows, {n_end} episode ends "
            f"({time.time() - t0:.0f}s)")
        return FP, LG, BIN

    FP, LG, BIN = rollout(None, int(seed), "teacher")
    R = int(FP.shape[0])

    # ---- 2. targets from the categorical heads --------------------------
    kbins = (K_BINS.astype(np.float64) if yaw_adaptive
             else 2.0 * YAW_BINS.astype(np.float64))

    def targets(LG):
        """Per row: the teacher's mean view in physical units (E[K],
        E[pitch] deg/tick), its z, and the categorical spread in z."""
        p_yaw = F.softmax(LG[:, :15], dim=-1).double().numpy()
        p_pit = F.softmax(LG[:, 15:22], dim=-1).double().numpy()
        e_k = p_yaw @ kbins                            # E[K] per row
        e_p = p_pit @ (PITCH_BINS.astype(np.float64) * (pitch_max / 10.0))
        z_yaw = z_from_u(warp_inv(e_k), U_CLIP)
        z_pit = (z_from_u(e_p / pitch_max, U_CLIP) if pitch_max > 0.0
                 else np.zeros(len(e_k)))
        probs = np.zeros((len(e_k), 6, 15), np.float32)
        probs[:, 0, :15] = p_yaw
        probs[:, 1, :7] = p_pit
        _m, sd_b = bin_view_moments(probs, yaw_adaptive, pitch_max)
        return (torch.as_tensor(np.stack([e_k, e_p], 1), dtype=torch.float64),
                torch.as_tensor(np.stack([z_yaw, z_pit], 1),
                                dtype=torch.float64),
                sd_b.astype(np.float64))

    V, Z, sd_b = targets(LG)
    e_k, z_yaw, z_pit = V[:, 0].numpy(), Z[:, 0].numpy(), Z[:, 1].numpy()
    spread = sd_b.mean(0)                              # per head
    log_std = np.maximum(np.log(np.maximum(spread, 1e-6)), float(min_log_std))
    say(f"targets: E[K] mean {e_k.mean():+.3f} sd {e_k.std():.3f}, "
        f"z_yaw sd {z_yaw.std():.3f}, z_pitch sd {z_pit.std():.3f}; "
        f"categorical spread in z {spread[0]:.3f} / {spread[1]:.3f} -> "
        f"log sigma {log_std[0]:.3f} / {log_std[1]:.3f}")

    def report(pred, tag, Z=Z, BIN=BIN):
        """RMSE / R^2 in z, and how often the fitted mean lands in the
        greedy yaw bin (the nearest bin of warp(tanh mu))."""
        resid = pred - Z.numpy()
        rmse = np.sqrt((resid ** 2).mean(0))
        r2 = 1.0 - (resid ** 2).mean(0) / np.maximum(Z.numpy().var(0), 1e-12)
        k_fit = warp(np.tanh(pred[:, 0]))
        near = np.abs(k_fit[:, None] - kbins[None, :]).argmin(1)
        agree = float((near == BIN[:, 0].numpy()).mean())
        k_arg = kbins[BIN[:, 0].numpy()]
        say(f"{tag}: rmse z {rmse[0]:.4f} / {rmse[1]:.4f}, R^2 {r2[0]:.3f} "
            f"/ {r2[1]:.3f}; nearest bin of warp(tanh mu) == greedy yaw bin "
            f"on {100 * agree:.1f}% of rows, mean |K_fit - K_greedy| "
            f"{np.abs(k_fit - k_arg).mean():.3f}")
        return [float(x) for x in rmse], [float(x) for x in r2], agree

    # ---- 3. the linear fit on the frozen tower --------------------------
    FPd = FP.to(dev)
    with torch.no_grad():
        Td = policy_d.pi(FPd)
    W, b = fit_ridge(Td, Z.to(dev), float(ridge))
    with torch.no_grad():
        pred0 = (Td.double() @ W.T + b).cpu().numpy()
    rmse0, r20, agree0 = report(pred0, "linear fit")

    # ---- 4. the continuous policy ---------------------------------------
    sd_d = ck["policy"]
    policy_c = Policy(policy_d.scal_dim - N_SCALAR + policy_d.lidar_w
                      * policy_d.lidar_h * policy_d.in_ch + N_SCALAR,
                      policy_d.lidar_w, policy_d.lidar_h,
                      emb=int(cfg.get("emb", 256)),
                      hidden=int(cfg.get("hidden", 256)),
                      gps=bool(cfg.get("gps", True)),
                      trunk=str(cfg.get("trunk") or "plain"),
                      tower_depth=int(cfg.get("tower_depth") or 2),
                      conv_mult=int(cfg.get("conv_mult") or 1),
                      extra_feat=(12,) if cfg.get("obs_reward") else (),
                      in_ch=policy_d.in_ch, n_codes=0, chunk=0,
                      route_dim=policy_d.route_dim,
                      route_critic_only=bool(cfg.get("route_critic_only")),
                      priv_dim=policy_d.priv_dim,
                      priv_hidden=int(cfg.get("priv_hidden") or 128),
                      view_continuous=True).to(dev)
    missing, unexpected = policy_c.load_state_dict(sd_d, strict=False)
    want = {"view_head.weight", "view_head.bias", "view_std.log_std"}
    if set(missing) != want or unexpected:
        raise SystemExit(f"unexpected state_dict delta: missing {missing} "
                         f"unexpected {unexpected}")
    with torch.no_grad():
        policy_c.view_head.weight.copy_(W.float())
        policy_c.view_head.bias.copy_(b.float())
        policy_c.view_std.log_std.copy_(torch.as_tensor(log_std,
                                                       dtype=torch.float32))

    # ---- 5. the actor-tower distillation (--finetune-steps) -------------
    ft = {"steps": int(finetune_steps), "dagger_rounds": int(dagger_rounds),
          "rounds": []}
    tuned = {"pi", "action_head", "view_head"}
    # physical-unit scales of the K-space term: a K error of 0.5 (one bin
    # around the strafe optimum) and a pitch error of 0.2 deg/tick (the
    # smallest pitch bin at the reference rate) each cost 1
    K_BAND, K_SCALE, P_SCALE = 3.0, 0.5, 0.2 * pitch_max / 10.0

    def finetune(FPd, LGd, Vd, Zd, BINr, tag):
        params = (list(policy_c.pi.parameters())
                  + list(policy_c.action_head.parameters())
                  + list(policy_c.view_head.parameters()))
        opt = torch.optim.Adam(params, lr=float(finetune_lr))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(finetune_steps), eta_min=float(finetune_lr) * 0.03)
        Rr = int(FPd.shape[0])
        g = torch.Generator(device=dev)
        g.manual_seed(int(seed) + 1 + len(ft["rounds"]))
        bs = max(64, min(int(finetune_batch), Rr))
        t1 = time.time()
        with torch.no_grad():
            kl0 = float(head_kl(LGd, policy_c.action_head(policy_c.pi(FPd))))
        Vt = Vd.float()
        # the yaw target of the physical term is the teacher's GREEDY bin's
        # K (what its argmax executes), the pitch target its mean
        Vt_k = torch.as_tensor(kbins, dtype=torch.float32, device=dev)[
            torch.as_tensor(np.asarray(BINr)[:, 0], device=dev)].clamp(-K_BAND, K_BAND)
        for it in range(int(finetune_steps)):
            idx = torch.randint(0, Rr, (bs,), generator=g, device=dev)
            t = policy_c.pi(FPd[idx])
            lg_new = policy_c.action_head(t)
            mu = policy_c.view_head(t)
            mse = (mu - Zd[idx]).pow(2).mean(0)
            u = torch.tanh(mu)
            k_pred = warp_t(u[:, 0]).clamp(-K_BAND, K_BAND)
            p_pred = u[:, 1] * pitch_max
            phys = (((k_pred - Vt_k[idx]) / K_SCALE).pow(2).mean()
                    + ((p_pred - Vt[idx, 1]) / P_SCALE).pow(2).mean())
            kl = head_kl(LGd[idx], lg_new)
            loss = mse.sum() + float(k_coef) * phys + float(kl_coef) * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (it + 1) % max(1, int(finetune_steps) // 5) == 0:
                say(f"  finetune[{tag}] {it + 1}/{finetune_steps}: mse z "
                    f"{float(mse[0]):.4f} / {float(mse[1]):.4f}  phys "
                    f"{float(phys):.4f}  kl {float(kl):.5f}")
        with torch.no_grad():
            t = policy_c.pi(FPd)
            pred1 = policy_c.view_head(t).double().cpu().numpy()
            lg1 = policy_c.action_head(t)
            kl1 = float(head_kl(LGd, lg1))
            # the four heads that stay live: argmax agreement with the source
            o, agree_c = 15 + 7, []
            for n_h in (3, 3, 2, 2):
                agree_c.append(float((lg1[:, o:o + n_h].argmax(-1)
                                      == LGd[:, o:o + n_h].argmax(-1))
                                     .float().mean()))
                o += n_h
        rmse1, r21, agree1 = report(pred1, f"after finetune[{tag}]",
                                    Z=Zd.double().cpu(), BIN=BINr)
        say(f"finetune[{tag}]: {Rr:,} rows, kl(disc || new) {kl0:.5f} -> "
            f"{kl1:.5f}; live heads argmax agreement fwd/side/jump/duck "
            + "/".join(f"{100 * a:.1f}%" for a in agree_c)
            + f" ({time.time() - t1:.0f}s)")
        ft["rounds"].append({"tag": tag, "rows": Rr, "kl_before": kl0,
                             "kl_after": kl1, "rmse_z": rmse1, "r2": r21,
                             "bin_agree": agree1, "live_head_agree": agree_c})

    if int(finetune_steps) > 0:
        ft.update(lr=float(finetune_lr), batch=int(finetune_batch),
                  kl_coef=float(kl_coef), k_coef=float(k_coef))
        FP_all, LG_all, BIN_all = FP, LG, BIN
        finetune(FPd, LG.to(dev), V.to(dev), Z.float().to(dev), BIN,
                 "teacher rows")
        for rd in range(int(dagger_rounds)):
            # the student's own states, the teacher's labels on them
            policy_c.eval()
            FPs, LGs, BINs = rollout(policy_c, int(seed) + 101 + rd,
                                     f"student round {rd + 1}")
            FP_all = torch.cat([FP_all, FPs], 0)
            LG_all = torch.cat([LG_all, LGs], 0)
            BIN_all = torch.cat([BIN_all, BINs], 0)
            Va, Za, _sd = targets(LG_all)
            finetune(FP_all.to(dev), LG_all.to(dev), Va.to(dev),
                     Za.float().to(dev), BIN_all, f"dagger round {rd + 1}")
    sd_c = {k: v.detach().cpu().clone() for k, v in policy_c.state_dict().items()}
    for k in sd_d:
        top = k.split(".")[0]
        if int(finetune_steps) > 0 and top in tuned:
            continue
        if not torch.equal(sd_c[k], sd_d[k]):
            raise SystemExit(f"shared tensor {k} changed in the transplant")

    # ---- 6. the checkpoint --------------------------------------------
    state = dict(ck)
    state["policy"] = sd_c
    opt_sd = state.get("optimizer")
    n_params = len(list(policy_c.parameters()))
    if opt_sd is not None:
        opt_sd = {"state": opt_sd.get("state", {}),
                  "param_groups": json.loads(json.dumps(
                      opt_sd.get("param_groups", [])))}
        for gr in opt_sd["param_groups"]:
            have = [int(i) for i in gr.get("params", [])]
            gr["params"] = have + [i for i in range(n_params)
                                   if i not in set(have)]
        state["optimizer"] = opt_sd
    cfg_out = dict(cfg)
    cfg_out["view_continuous"] = 1
    cfg_out["view_transplant"] = {
        "source": str(ckpt_in), "source_md5": src_md5,
        "rows": R, "envs": n, "ticks": int(ticks), "n_greedy": int(n_greedy),
        "reservoir_frac": float(reservoir_frac), "seed": int(seed),
        "ridge": float(ridge),
        "linear": {"rmse_z": rmse0, "r2": r20, "bin_agree": agree0},
        "finetune": ft,
        "log_std": [float(x) for x in log_std],
        "spread_z": [float(x) for x in spread],
        "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    state["config"] = cfg_out
    Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_out)
    say(f"wrote {ckpt_out} md5 {md5(ckpt_out)} ({n_params} params, "
        f"{len(sd_c)} tensors; optimizer groups extended to {n_params}"
        + ("" if int(finetune_steps) > 0 else
           "; every shared tensor bit-identical to the source") + ")")
    return cfg_out["view_transplant"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt_in")
    ap.add_argument("ckpt_out")
    ap.add_argument("--map", default=None,
                    help="ABSOLUTE .bsp in the main checkout (default: the "
                         "checkpoint's map resolved there)")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=2400,
                    help="rollout length per env in physics ticks")
    ap.add_argument("--greedy-envs", type=int, default=64,
                    help="envs [0, G) act greedily, the rest sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reservoir-frac", type=float, default=0.7,
                    help="share of spawns drawn from the ckpt's reservoir")
    ap.add_argument("--reservoir-cap", type=int, default=20000)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--min-log-std", type=float, default=-3.0)
    ap.add_argument("--finetune-steps", type=int, default=0,
                    help="distil the actor tower (pi + action_head + "
                         "view_head) for this many Adam steps: MSE on the "
                         "view means + KL to the discrete heads. 0 = the "
                         "linear readout alone (shared tensors untouched)")
    ap.add_argument("--finetune-lr", type=float, default=3e-4)
    ap.add_argument("--finetune-batch", type=int, default=4096)
    ap.add_argument("--kl-coef", type=float, default=1.0)
    ap.add_argument("--k-coef", type=float, default=4.0,
                    help="weight of the PHYSICAL-unit term (K within +-3, "
                         "pitch deg/tick) next to the z-space MSE")
    ap.add_argument("--dagger-rounds", type=int, default=0,
                    help="after the fit, roll the STUDENT this many times, "
                         "label its states with the teacher and refit on "
                         "the union (each round = one --ticks rollout + one "
                         "--finetune-steps pass)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-config-audit", action="store_true")
    a = ap.parse_args()
    transplant(a.ckpt_in, a.ckpt_out, map_path=a.map, envs=a.envs,
               ticks=a.ticks, n_greedy=a.greedy_envs, seed=a.seed,
               reservoir_frac=a.reservoir_frac, reservoir_cap=a.reservoir_cap,
               ridge=a.ridge, min_log_std=a.min_log_std,
               finetune_steps=a.finetune_steps, finetune_lr=a.finetune_lr,
               finetune_batch=a.finetune_batch, kl_coef=a.kl_coef,
               k_coef=a.k_coef, dagger_rounds=a.dagger_rounds,
               device=a.device, audit=not a.no_config_audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
