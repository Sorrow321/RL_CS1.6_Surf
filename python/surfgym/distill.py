"""distill.py - DAgger as online policy distillation (train_fast.py --distill, mm/distill).

The user, 2026-11-27, after the 20-map run: train separate policies, then "unite them into one
policy that will be basically supervised using this dagger method". DAgger (Ross, Gordon &
Bagnell 2011) queries the expert AT THE LEARNER'S OWN STATES, which removes the covariate shift
of plain behaviour cloning; with an expert that is a network, every state of the student's own
rollout can be labelled as it is flown, which is the online form of policy distillation (Rusu
et al. 2016; Czarnecki et al. 2019, "student-driven" distillation with the teacher's KL).

Two halves live in train_fast.py, both off without --distill:

* the UPDATE: every PPO minibatch row also goes through the frozen teacher, and the student fits
  KL(teacher || student) per head - the categorical heads exactly, the --view-continuous
  Gaussians in closed form with each side's own sigma - plus (V - V_teacher)^2 (terms() here);
  --distill-rl weighs the run's own PPO loss beside it, 0 = no reward gradient at all;
* the ROLLOUT (--distill-drive, DAgger's beta): the teacher's own sampled action replaces the
  student's on a fixed share of every map's envs (drive_mask() here), so the fleet's states and
  the respawn snapshots they leave cover the whole map while the student cannot fly it yet.

What the teacher sees must be what the student sees: the same rows go into both networks, so
every key in SEMANTIC (what the observation holds, what an action means, the physics) must
match between the teacher's checkpoint config and this run's (semantic_mismatch()); the
architecture is checked by a strict load of the teacher's weights into this run's own network.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# config keys that decide what an observation row holds, what an action means, or the physics
# the teacher learned under; compared when both configs carry the key (an older checkpoint
# predates the newer keys - ARCH_DEFAULTS below covers the ones that change the network)
SEMANTIC = (
    "act_every", "act_hist", "chunk", "codebook", "emb", "hidden", "trunk", "simba",
    "tower_depth", "conv_mult", "rnn", "rnn_size", "gps", "keys_hold", "lidar_cell", "lidar_h",
    "lidar_w", "lidar_hfov", "lidar_vfov", "lidar_near", "lidar_range", "max_step", "normals",
    "obs_compass", "obs_fourier", "obs_normal", "obs_pot_rate", "obs_potential",
    "obs_potential_curtain", "obs_reward", "obs_surf", "obs_views", "obs_views_scale",
    "obs_no_depth", "pinhole", "pitch_rate", "pitch_sigma_min", "yaw_sigma_max", "priv_critic",
    "race_latch", "race_no_corner_cut", "race_dist", "side_hold", "surf_mask", "teleport_fail",
    "tick_ms", "tick_pattern_ms", "view_absolute", "view_continuous", "vision_clip",
    "yaw_adaptive", "yaw_blend", "frame_stack", "fix_pitch", "pitch_fixed", "goal_obs", "goals",
    "int_split", "critic_pay_d0", "dae", "split_trunk", "conv_norm", "final_ln_affine",
    "mlp_act", "ramp_obs_vec", "ramp_obs_pass",
)
# the network keys newer than some checkpoints: an absent key is the network the code built
# before the flag existed
ARCH_DEFAULTS = {"conv_norm": 0, "final_ln_affine": 1, "mlp_act": "relu", "split_trunk": 0}
# per-map blocks: compared on the maps both runs have
PER_MAP = ("maxvel_maps", "goal_cells", "map_cells", "obs_potential_d0")


def _norm(k, v):
    if v is None and k in ARCH_DEFAULTS:
        return ARCH_DEFAULTS[k]
    if isinstance(v, bool):
        return int(v)
    return v


def semantic_mismatch(teacher_cfg: dict, cfg: dict) -> list:
    """[(key, teacher value, this run's value)] for every SEMANTIC key both configs carry (or
    that ARCH_DEFAULTS fills) and every PER_MAP entry on a map both have, where they differ."""
    out = []
    for k in SEMANTIC:
        if k not in teacher_cfg and k not in cfg:
            continue
        if (k not in teacher_cfg or k not in cfg) and k not in ARCH_DEFAULTS:
            continue
        a, b = _norm(k, teacher_cfg.get(k)), _norm(k, cfg.get(k))
        if isinstance(a, float) or isinstance(b, float):
            try:
                if abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a))):
                    continue
            except (TypeError, ValueError):
                pass
        if a != b:
            out.append((k, teacher_cfg.get(k), cfg.get(k)))
    for k in PER_MAP:
        ta, tb = teacher_cfg.get(k) or {}, cfg.get(k) or {}
        if not isinstance(ta, dict) or not isinstance(tb, dict):
            continue
        for tag in sorted(set(ta) & set(tb)):
            if abs(float(ta[tag]) - float(tb[tag])) > 1e-6 * max(1.0, abs(float(ta[tag]))):
                out.append((f"{k}[{tag}]", ta[tag], tb[tag]))
    return out


def slot_mismatch(teacher_cfg: dict, stem: str, tag: str, maxvel):
    """One map's teacher against the map it teaches -> (refusals, notes), lists of strings.
    Refused: a speed cap on this map other than the run's (``maxvel_maps[tag]`` of a --maps
    teacher, else its run-wide ``maxvel``): the physics the labels answer would not be this run's.
    Noted: a teacher that never trained on this map (``maps`` / ``map`` of its config) - its
    labels there are a zero-shot guess."""
    from pathlib import Path as _P
    maps = list(teacher_cfg.get("maps") or ([teacher_cfg["map"]] if teacher_cfg.get("map") else []))
    stems = [_P(str(m)).stem for m in maps]
    notes, refuse = [], []
    if stems and stem not in stems:
        notes.append(f"never trained on {stem} (its maps: {', '.join(stems)})")
    mvm = teacher_cfg.get("maxvel_maps") or {}
    tv = mvm.get(tag) if isinstance(mvm, dict) and tag in mvm else teacher_cfg.get("maxvel")
    if tv is not None and maxvel is not None and abs(float(tv) - float(maxvel)) > 1e-6:
        refuse.append(f"speed cap on {stem}: teacher {float(tv):g}, this run {float(maxvel):g}")
    return refuse, notes


def unseen_maps(teacher_cfg: dict, tags) -> list:
    """this run's map tags the teacher never trained on (its labels there are a zero-shot
    guess); [] when the teacher's config names no maps (a single-map checkpoint: its map)."""
    seen = set((teacher_cfg.get("maxvel_maps") or {}).keys())
    if not seen:
        return []
    return [t for t in tags if t not in seen]


def terms(pad_t, pad_s, v_t, v_s, mu_t=None, mu_s=None, ls_t=None, ls_s=None, per_row=False):
    """The distillation terms on one minibatch -> (kl_cat, kl_view, v_mse, agree), scalars.

    ``pad_*`` (B, heads, bins) categorical logits padded with HeadPacker.pad's -1e30 slots (the
    heads the policy acts with; the teacher's side ``_t`` is the fixed target and is detached);
    ``v_*`` (B,) or (B, k) values; ``mu_*`` (B, n_z) the view means and ``ls_*`` the two
    networks' log sigmas (n_z,) under --view-continuous, else None.

    kl_cat = sum over heads of KL(p_t || p_s), the pad slots contributing exactly 0 (p = 0 and
    both log-probs the same -1e30, as in augment.drac_terms); kl_view = the Gaussian closed form
    log(s_s / s_t) + (s_t^2 + (mu_t - mu_s)^2) / (2 s_s^2) - 1/2 summed over the view heads, so
    the student's sigma is fitted to the teacher's as well as its mean; agree = the share of
    (row, categorical head) pairs whose argmax agrees (a diagnostic, never the verdict).
    ``ls_t`` may also be (B, n_z): each row's own teacher's sigma (one teacher per map).
    ``per_row``: a fifth output, (B,) each row's KL_cat + KL_view, detached (the per-map log).
    """
    lt = F.log_softmax(pad_t.detach().float(), dim=-1)
    ls = F.log_softmax(pad_s.float(), dim=-1)
    kl_r = (lt.exp() * (lt - ls)).sum(-1).sum(-1)
    kl = kl_r.mean()
    agree = (pad_t.detach().argmax(-1) == pad_s.detach().argmax(-1)).float().mean()
    if mu_t is not None:
        lst, lss = ls_t.detach().float(), ls_s.float()
        kv_r = (lss - lst + ((2.0 * lst).exp() + (mu_t.detach().float() - mu_s.float()).pow(2))
                / (2.0 * (2.0 * lss).exp()) - 0.5).sum(-1)
        kv = kv_r.mean()
    else:
        kv_r = kv = torch.zeros((), device=kl.device)
    vm = (v_s.float() - v_t.detach().float()).pow(2).mean()
    if per_row:
        return kl, kv, vm, agree, (kl_r + kv_r).detach()
    return kl, kv, vm, agree


def drive_mask(slots, n_env: int, beta: float):
    """(n_env,) bool, True on the first round(beta x width) envs of every map slot (lo, hi):
    the envs whose actions the teacher draws. beta 0 = none, 1 = every env."""
    m = torch.zeros(int(n_env), dtype=torch.bool)
    for s in slots:
        lo, hi = int(s.lo), int(s.hi)
        m[lo:lo + int(round(float(beta) * (hi - lo)))] = True
    return m


def drive_at(step: float, beta0: float, steps) -> float:
    """--distill-drive's beta at env step ``step``: beta0 throughout without
    --distill-drive-steps, else linear from beta0 at step 0 to 0 at ``steps``."""
    if not steps:
        return float(beta0)
    return float(beta0) * max(0.0, 1.0 - float(step) / float(steps))
