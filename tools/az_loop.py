"""az_loop.py - the TREE-ONLY learner (azr1b4, slot 4 of the one-search series).

The user's endpoint (autoresearch/PROTOCOL.md, slot 4): "we didn't have one PPO and one search. We
had just one search. And the model kind of updates of the policy were made based on the trees that
we generated." No PPO, no rollouts, no reward: the network learns its policy and its value from
the search trees alone, and the search's roots are the network's own greedy line.

ONE process holds the network and repeats, from the network the run was started with:

  1. SEARCH   tools/mcts_search.run_mcts on the CURRENT weights (run 5's tree, every --mcts-*
              flag as given), with --mcts-labels: the tree comes back with its training rows;
  2. LABELS   every expanded node (the children that beat their siblings; the node's backed-up
              Q) + every decision of the best line from the true start (mcts_search._tree_labels);
              the last --az-buffer trees are kept;
  3. DISTIL   --az-steps supervised steps on them (distil_loss: the action heads' cross-entropy to
              the targets, the view heads by the trainer's own BC Gaussian form at the fixed
              reference sigma, the value head's MSE to Q). The network's log-sigma is not trained
              (the trainer's BC convention), so its random children keep the init spread;
  4. EVAL     --az-eval-eps greedy episodes from the TRUE start, side by side, with the trainer's
              eval rules (--az-eval-patience s without a 32 u gain ends an episode, the episode
              cap), written as traj_<total ticks>.jsonl in the record.py format score.py reads.

THE STEP COUNTER IS EVERY SIMULATOR TICK: the search's greedy pass + every wave's copies + the line
replay + every eval tick (all envs of the eval core). checkpoints carry it as global_step, so
autoresearch/ladder.py --total-is-step reads the evals directly.

Checkpoints: ckpt_latest.pt (atomic), ckpt_<total>.pt every --az-ckpt-every rounds and at every new
best eval; {"policy", "optimizer", "global_step", "config" (the init's, untouched: record_ckpt and
beam_tas load it), "az" (this loop's state)}. A relaunch with the same command resumes from
<out>/ckpt_latest.pt (its round, its tick count, its label files) - a crash-resume of the same run.
Nothing in this file names a map, a place, a speed or a progress value.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

LABEL_KEYS = ("scal", "pose", "probs", "vmu", "vsd", "value", "kind", "w")
#: optional label columns (absent in the files of an earlier commit: filled with 1.0)
LABEL_OPT = ("adv",)


def add_args(ap) -> None:
    """beam_tas.py's --az-* flags (all inert without --az-loop)."""
    ap.add_argument("--az-loop", action="store_true",
                    help="the tree-only learner (tools/az_loop.py): rounds of [MCTS on the current "
                         "weights -> the tree's labels -> supervised distillation -> greedy eval "
                         "from the true start]; needs --mcts. --out-dir is the run directory")
    ap.add_argument("--az-rounds", type=int, default=10 ** 6, help="--az-loop: stop after N rounds")
    ap.add_argument("--az-max-total", type=float, default=60e9,
                    help="--az-loop: stop once the run has spent this many simulator ticks")
    ap.add_argument("--az-steps", type=int, default=1000,
                    help="--az-loop: supervised gradient steps per round")
    ap.add_argument("--az-batch", type=int, default=1024, help="--az-loop: rows per step")
    ap.add_argument("--az-buffer", type=int, default=4,
                    help="--az-loop: the last N trees' labels are trained on")
    ap.add_argument("--az-z-clip", type=float, default=3.0,
                    help="--az-loop: view targets in pre-tanh z are clipped to +-Z (tanh 3 = "
                         "0.995: past it the executed steer barely moves); 0 = off")
    ap.add_argument("--az-recency", type=float, default=0.5,
                    help="--az-loop: a label row of a tree N rounds older than the newest is drawn "
                         "with weight RECENCY**N (1 = every buffered tree alike)")
    ap.add_argument("--az-node-adv", type=int, default=0, choices=(0, 1),
                    help="--az-loop: 1 = draw the expanded nodes' rows in proportion to their "
                         "target's ADVANTAGE (its Q minus the mean Q of the node's children, over "
                         "the tree's Q range; mcts_search.node_targets): a node where the named "
                         "move beat its siblings by much teaches more than one where a random "
                         "child won by a hair. 0 = every node row alike (as before)")
    ap.add_argument("--az-sigma-fit", type=int, default=0, choices=(0, 1),
                    help="--az-loop: 1 = the YAW head's log sigma is fitted to the tree's targets "
                         "(maximum likelihood, mu detached, never above its init 0.3): sigma becomes "
                         "the spread of the tree's preferred steers around the policy's own, and "
                         "the search's random children (drawn at a multiple of it) narrow as the "
                         "policy learns - in run 5 PPO narrowed it (0.30 -> 0.07 by 0.57B steps). "
                         "0 = sigma frozen at the init (the trainer's BC convention; as before)")
    ap.add_argument("--az-line-frac", type=float, default=0.25,
                    help="--az-loop: share of each batch drawn from the best lines' SEARCHED rows "
                         "(the tree's choices, decision by decision)")
    ap.add_argument("--az-prefix-frac", type=float, default=0.25,
                    help="--az-loop: share drawn from the best lines' greedy PREFIX rows (the "
                         "policy's own moves up to the root: the anchor); the rest of the batch "
                         "comes from the expanded nodes' rows. An empty pool's share goes to the "
                         "others in proportion")
    ap.add_argument("--az-lr", type=float, default=None,
                    help="--az-loop: Adam learning rate (default: the checkpoint's own lr)")
    ap.add_argument("--az-value-coef", type=float, default=0.25,
                    help="--az-loop: weight of the value MSE (MuZero's 0.25 against the "
                         "policy's 1)")
    ap.add_argument("--az-pitch-coef", type=float, default=0.0,
                    help="--az-loop: weight of the PITCH head in the view term (0: under "
                         "--view-absolute the pitch only aims the camera; the trainer's "
                         "--bc-pitch-coef 0)")
    ap.add_argument("--az-eval-eps", type=int, default=9, help="--az-loop: greedy eval episodes")
    ap.add_argument("--az-eval-patience", type=float, default=30.0,
                    help="--az-loop: an eval episode ends after SECS without a 32 u gain (0 = off)")
    ap.add_argument("--az-ckpt-every", type=int, default=5,
                    help="--az-loop: keep ckpt_<total>.pt every N rounds (and at every new best)")
    ap.add_argument("--az-run", default=None, help="--az-loop: the run name (default: the out dir)")


def _md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:            # pragma: no cover
        return ""


# --------------------------------------------------------------------------------------------
# the supervised loss (train_fast.bc_loss_fn's form for --view-continuous + --bc-target dist)
# --------------------------------------------------------------------------------------------
def distil_loss(policy, packer, scal, img, probs, vmu, vsd, value_t, w, view_w,
                value_coef: float, view_sigma: float, sigma_fit: bool = False):
    """-> (loss, stats dict of floats). ``probs`` (B, NACT, NPAD): the categorical targets on
    heads N_VIEW.. (view columns ignored); ``vmu`` / ``vsd`` (B, n_z): the view target moments in
    pre-tanh z; ``value_t`` (B,): the tree's Q; ``w`` (B,) row weights; ``view_w`` (n_z,) the
    per-view-head weights (pitch last). The view term is the Gaussian cross-entropy to
    N(vmu, vsd) at the FIXED sigma ``view_sigma`` (train_fast.BC_VIEW_SIGMA: the live sigma made
    the term a 1/sigma^2 drag on mu), exactly the trainer's nll_mm. ``sigma_fit`` (--az-sigma-fit)
    adds the maximum-likelihood term of the YAW head's own log sigma on the same targets with mu
    detached - log s + ((vmu - mu)^2 + vsd^2) / (2 s^2), minimised at s^2 = the mean squared gap
    between the policy's steer and the tree's preferred steer - so sigma only measures, and the
    fixed-sigma term above stays the one that moves mu."""
    import train_fast as _tf
    logits, value = policy.forward_split(scal, img)
    cat, mu = _tf.split_view(logits.float())
    padded = packer.pad(cat)
    lsm = F.log_softmax(padded, dim=-1)
    nv = int(_tf.N_VIEW)
    ce_cat = -(probs[:, nv:] * lsm[:, nv:]).sum(-1).sum(-1)
    ls = float(np.log(view_sigma))
    var2 = 2.0 * view_sigma * view_sigma
    nll_mm = (ls + 0.5 * _tf.LOG2PI + (vsd.pow(2) + (vmu - mu).pow(2)) / var2) * view_w
    wsum = w.sum().clamp_min(1e-6)
    loss_pi = ((ce_cat + nll_mm.sum(-1)) * w).sum() / wsum
    v = value.float().reshape(-1)
    v_mse = ((v - value_t).pow(2) * w).sum() / wsum
    loss = loss_pi + float(value_coef) * 0.5 * v_mse
    ls_y = policy.log_std()[0]
    if sigma_fit:
        gap2 = (vmu[:, 0] - mu[:, 0].detach()).pow(2) + vsd[:, 0].pow(2)
        sig_nll = ls_y + gap2 * torch.exp(-2.0 * ls_y) * 0.5
        loss = loss + (sig_nll * w).sum() / wsum
    with torch.no_grad():
        agree = (padded[:, nv:].argmax(-1) == probs[:, nv:].argmax(-1)).float()
        stats = torch.stack([loss.detach(), (ce_cat * w).sum() / wsum, agree.mean(),
                             agree.all(-1).float().mean(),
                             ((vmu[:, 0] - mu[:, 0]).pow(2) * w).sum() / wsum, v_mse.detach(),
                             ls_y.detach().exp()])
    return loss, stats


STAT_NAMES = ("loss", "ce", "acc", "joint_acc", "yaw_mse", "value_mse", "yaw_sigma")


class LabelBuffer:
    """The last ``keep`` trees' label rows, on the device, sampled by kind."""

    def __init__(self, keep: int, device, z_clip: float = 0.0):
        self.keep, self.device = max(1, int(keep)), device
        self.z_clip = float(z_clip or 0.0)
        self.sets = []           # [(round, {key: tensor})]

    def add(self, rnd: int, lab: dict) -> None:
        if lab is None or len(lab.get("kind", ())) == 0:
            return
        t = {}
        n_rows = len(np.asarray(lab["kind"]))
        for k in LABEL_KEYS + LABEL_OPT:
            a = (np.asarray(lab[k]) if k in lab else np.ones(n_rows, np.float32))
            if k == "kind":
                t[k] = torch.as_tensor(a.astype(np.int64), device=self.device)
            else:
                if k == "vmu" and self.z_clip > 0.0:
                    # a view target past tanh's saturation steers the same as one at it; in the
                    # z-space Gaussian term it would only cost a squared error the steer never sees
                    a = np.clip(a, -self.z_clip, self.z_clip)
                t[k] = torch.as_tensor(a.astype(np.float32), device=self.device)
        self.sets.append((int(rnd), t))
        self.sets = self.sets[-self.keep:]

    def rounds(self):
        return [r for r, _ in self.sets]

    def rows(self) -> int:
        return int(sum(len(t["kind"]) for _, t in self.sets))

    def _cat(self):
        return {k: torch.cat([t[k] for _, t in self.sets]) for k in LABEL_KEYS + LABEL_OPT}

    def sampler(self, batch: int, fracs: dict, gen: torch.Generator, recency: float = 1.0,
                adv_kinds=()):
        """-> (f() returning one batch dict, {kind: rows}). ``fracs`` {kind: share of the batch}
        (0 node rows, 1 the lines' searched rows, 2 their greedy prefixes); a kind with no rows
        hands its share to the others in proportion. Rows are drawn with replacement, a row of
        the tree ``age`` rounds older than the newest weighted ``recency ** age`` (1 = uniform):
        where the policy has moved on, the newer tree's labels win the disagreement. A kind in
        ``adv_kinds`` is drawn in proportion to its rows' ``adv`` as well (--az-node-adv: a node
        row by how much its target beat the node's other children; adv 0 is never drawn)."""
        allr = self._cat()
        kind = allr["kind"]
        newest = max(r for r, _ in self.sets)
        rw = torch.cat([torch.full((len(t["kind"]),), float(recency) ** (newest - r),
                                   device=self.device) for r, t in self.sets])
        pools = {}
        for kd in sorted(fracs):
            ix = torch.nonzero(kind == int(kd)).reshape(-1)
            wk = rw[ix] * (allr["adv"][ix].clamp_min(0.0) if int(kd) in adv_kinds else 1.0)
            if len(ix) and float(fracs[kd]) > 0.0 and float(wk.sum()) > 0.0:
                pools[kd] = (ix, wk, float(fracs[kd]))
        tot = sum(f for _i, _w, f in pools.values())
        counts, left = {}, int(batch)
        for j, kd in enumerate(sorted(pools)):
            n = (left if j == len(pools) - 1
                 else max(1, int(round(batch * pools[kd][2] / tot))))
            n = min(n, left)
            counts[kd] = n
            left -= n

        def draw():
            parts = [pools[kd][0][torch.multinomial(pools[kd][1], counts[kd], replacement=True,
                                                    generator=gen)]
                     for kd in sorted(pools) if counts[kd] > 0]
            ix = torch.cat(parts)
            return {k: v[ix] for k, v in allr.items()}
        return draw, {int(kd): int((kind == int(kd)).sum()) for kd in sorted(fracs)}


def render_img(lidar, pose):
    """[scal | image] row's image half from the stored pose (origin xyz, yaw, pitch, ducked) with
    the run's own lidar (the wrappers' _obs path), as a NORMAL tensor (the lidar runs under
    inference mode; a clone outside it can enter autograd)."""
    with torch.inference_mode():
        img = lidar.render(pose[:, 0:3].contiguous(), pose[:, 3].contiguous(),
                           pose[:, 4].contiguous(), pose[:, 5].round().to(torch.int32))
        img = img.reshape(pose.shape[0], -1).float()
    return img.clone()


def distil(policy, packer, lidar, opt, buf: LabelBuffer, a, gen, view_w) -> dict:
    import train_fast as _tf
    if buf.rows() == 0:
        return {"steps": 0}
    fr_line, fr_pre = float(a.az_line_frac), float(a.az_prefix_frac)
    draw, n_kind = buf.sampler(int(a.az_batch), {0: max(0.0, 1.0 - fr_line - fr_pre),
                                                 1: fr_line, 2: fr_pre}, gen,
                               float(a.az_recency),
                               adv_kinds=(0,) if int(a.az_node_adv) else ())
    policy.train()
    acc = None
    t0 = time.time()
    from surfgym.view import LOG_STD_INIT
    sigma_cap = LOG_STD_INIT
    for _s in range(int(a.az_steps)):
        b = draw()
        img = render_img(lidar, b["pose"])
        loss, st = distil_loss(policy, packer, b["scal"], img, b["probs"], b["vmu"], b["vsd"],
                               b["value"], b["w"], view_w, float(a.az_value_coef),
                               float(_tf.BC_VIEW_SIGMA), sigma_fit=bool(int(a.az_sigma_fit)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()
        if int(a.az_sigma_fit):
            # the yaw sigma may only NARROW from the init (its widest, before any tree): the
            # search's random children draw at a multiple of it, so a sigma that grew would make
            # the next tree wilder because this one was hard to fit
            with torch.no_grad():
                policy.view_std.log_std[0].clamp_(max=float(sigma_cap))
        acc = st if acc is None else acc + st
    policy.eval()
    s = (acc / max(1, int(a.az_steps))).tolist()
    out = {k: float(v) for k, v in zip(STAT_NAMES, s)}
    out.update(steps=int(a.az_steps), rows_node=n_kind.get(0, 0), rows_line=n_kind.get(1, 0),
               rows_prefix=n_kind.get(2, 0), secs=round(time.time() - t0, 1))
    return out


# --------------------------------------------------------------------------------------------
# the eval: greedy, from the true start, the trainer's rules
# --------------------------------------------------------------------------------------------
def evaluate(B: dict, a, path: Path, seed: int, rnd: int):
    """-> (ticks spent, per-episode summaries). B holds the beam_tas bundle."""
    import train_fast as _tf
    from surfgym.record import record_rollout_parallel
    n = int(a.az_eval_eps)
    core = B["eval_core"]
    es, ef, _st, lf = B["mk_feed"]()
    xkw, _xb = B["XF"].make(core)
    gpol = _tf.GreedyTorchPolicy(B["policy"], B["packer"], B["device"], B["lidar"], core, B["K"],
                                 B["stack"], extra_slot=es, extra_fn=ef, latch_fn=lf,
                                 cc_fn=B["cc_fnN"], keys_hold=B["KEYS"], **xkw)
    gpol.spawn_duck = B["SPAWN_DUCK"]
    pat_ticks = (B["TICK"].secs_to_ticks(float(a.az_eval_patience))
                 if float(a.az_eval_patience) > 0 else 0)
    hook = (_tf.make_eval_patience_hook_par(core, B["gf"], pat_ticks, n) if pat_ticks else None)
    count = {"t": 0}

    def on_tick(t, states, rewards, done, trunc):
        count["t"] += 1
        if hook is not None:
            hook(t, states, rewards, done, trunc)

    hdr = {"eval_stall": 0, "eval_parallel": 1, "az_round": int(rnd)}
    if pat_ticks:
        hdr["eval_patience"] = float(a.az_eval_patience)
    with torch.inference_mode():
        summ = record_rollout_parallel(core, gpol, path, n, seed=int(seed),
                                       max_ticks=int(B["ep_cap"]) + 1, on_tick=on_tick,
                                       header_extra=hdr)
    return int(count["t"]) * int(core.num_envs), summ


def score_eval(path: Path) -> dict:
    """autoresearch/score.py on one eval (the grader's own numbers), or {} when unavailable."""
    try:
        sys.path.insert(0, str(ROOT / "autoresearch"))
        import score as _score
        s = _score.score_file(path)
        return {k: s.get(k) for k in ("score", "best_progress_pct", "median_progress_pct",
                                       "max_cp", "finishes", "error") if k in s}
    except Exception as e:       # pragma: no cover
        return {"error": f"score.py: {e!r}"}


# --------------------------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------------------------
CSV_COLS = ("time/total_timesteps", "round", "wall_s", "search_s", "distil_s", "eval_s",
            "ticks_search", "ticks_line", "ticks_eval", "mcts_waves", "mcts_nodes", "mcts_cells",
            "mcts_roots", "mcts_best_pct", "mcts_greedy_pct", "mcts_raw_best_pct", "mcts_finished",
            "labels_node", "labels_line", "labels_prefix", "buf_rows", "loss", "ce", "acc",
            "joint_acc", "yaw_mse", "value_mse", "yaw_sigma", "eval_score", "eval_best_pct", "eval_median_pct", "eval_cp",
            "eval_finishes")


def run_az(B: dict, a) -> None:
    from mcts_search import run_mcts
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    policy, device = B["policy"], B["device"]
    cfg = B["cfg"]
    run_name = a.az_run or out.name
    if getattr(B["lidar"], "uses_velocity", False) or int(B["stack"]) > 1:
        raise SystemExit("--az-loop: a label row stores the image POSE only - not with --obs-ttc "
                         "(velocity channel) or a frame stack")
    a.mcts_labels = 1
    lr = float(a.az_lr if a.az_lr is not None else cfg.get("lr") or 3e-4)
    opt = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    import train_fast as _tf
    n_zv = int(getattr(policy, "n_z", 2))
    view_w = torch.ones(n_zv, device=device)
    view_w[n_zv - 1] = float(a.az_pitch_coef)
    buf = LabelBuffer(int(a.az_buffer), device, z_clip=float(a.az_z_clip))
    seed0 = int(a.torch_seed)
    st = {"round": 0, "total": 0, "ticks_search": 0, "ticks_line": 0, "ticks_eval": 0,
          "best_score": None, "best_step": None, "label_files": [], "started": time.time()}
    latest = out / "ckpt_latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["policy"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        st.update(ck.get("az") or {})
        st["total"] = int(ck["global_step"])
        for rr, f in st.get("label_files", [])[-int(a.az_buffer):]:
            fp = out / "labels" / f
            if fp.exists():
                z = np.load(fp, allow_pickle=False)
                buf.add(int(rr), {k: z[k] for k in LABEL_KEYS + LABEL_OPT if k in z.files})
        print(f"az: RESUMED {latest} at round {st['round']}, {st['total']:,} ticks, buffer "
              f"rounds {buf.rounds()} ({buf.rows():,} rows)", flush=True)
    else:
        init = Path(a.ckpt)
        meta = {"run": run_name, "kind": "az_loop (azr1b4: the tree only)",
                "argv": sys.argv, "commit": _git_head(), "init_ckpt": str(init),
                "init_md5": _md5(init), "init_step": int(B["ck"].get("global_step", 0)),
                "az": {k: v for k, v in vars(a).items()
                       if k.startswith("az_") or k.startswith("mcts_")
                       or k in ("envs", "torch_seed", "max_ticks", "seed")},
                "config": cfg, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (out / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"az: fresh start from {init} (step {meta['init_step']}, md5 {meta['init_md5']}); "
              f"run.json written", flush=True)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed0 * 1000003 + int(st["round"]))
    csv_path = out / "progress.csv"
    new_csv = not csv_path.exists()
    fcsv = open(csv_path, "a", newline="", encoding="utf-8")
    wcsv = csv.writer(fcsv)
    if new_csv:
        wcsv.writerow(CSV_COLS)
        fcsv.flush()

    def save(tag_steps: bool):
        state = {"policy": policy.state_dict(), "optimizer": opt.state_dict(),
                 "global_step": int(st["total"]), "config": cfg, "az": dict(st)}
        tmp = out / "ckpt_latest.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, latest)
        if tag_steps:
            snap = out / f"ckpt_{int(st['total']):010d}.pt"
            torch.save(state, snap)

    while int(st["round"]) < int(a.az_rounds) and int(st["total"]) < float(a.az_max_total):
        rnd = int(st["round"])
        t_r = time.time()
        a.torch_seed = seed0 + rnd
        torch.manual_seed(seed0 + rnd)
        # ---- 1. search + labels ------------------------------------------------------------
        t0 = time.time()
        res = run_mcts(B["ctx"], a)
        sm = res["summary"]
        lab = res.get("labels")
        res = None
        t_search = time.time() - t0
        st["total"] = int(st["total"]) + int(sm["ticks_total"])
        st["ticks_search"] = int(st["ticks_search"]) + int(sm["ticks_greedy"]) + int(
            sm["ticks_waves"])
        st["ticks_line"] = int(st["ticks_line"]) + int(sm.get("ticks_line", 0))
        fname = f"r{rnd:05d}.npz"
        if lab is not None and len(lab["kind"]):
            np.savez(out / "labels" / fname, **{k: lab[k] for k in LABEL_KEYS + LABEL_OPT
                                                if k in lab})
            st["label_files"] = (list(st.get("label_files", [])) + [[rnd, fname]])[
                -int(a.az_buffer):]
            for old in sorted((out / "labels").glob("r*.npz")):
                if old.name not in {f for _r, f in st["label_files"]}:
                    old.unlink()
        buf.add(rnd, lab)
        lab = None
        # ---- 2. distil ----------------------------------------------------------------------
        t0 = time.time()
        ds = distil(policy, B["packer"], B["lidar"], opt, buf, a, gen, view_w)
        t_distil = time.time() - t0
        # ---- 3. eval ------------------------------------------------------------------------
        t0 = time.time()
        tmp_path = out / "eval_tmp.jsonl"
        ev_ticks, _summ = evaluate(B, a, tmp_path, seed=seed0 + rnd, rnd=rnd)
        st["total"] = int(st["total"]) + int(ev_ticks)
        st["ticks_eval"] = int(st["ticks_eval"]) + int(ev_ticks)
        ev_path = out / f"traj_{int(st['total']):010d}.jsonl"
        os.replace(tmp_path, ev_path)
        sc = score_eval(ev_path)
        t_eval = time.time() - t0
        st["round"] = rnd + 1
        best = False
        if sc.get("score") is not None and (st["best_score"] is None
                                            or float(sc["score"]) > float(st["best_score"])):
            st["best_score"], st["best_step"] = float(sc["score"]), int(st["total"])
            best = True
        save(tag_steps=best or (int(a.az_ckpt_every) > 0 and (rnd + 1) % int(a.az_ckpt_every)
                                == 0))
        row = {"time/total_timesteps": int(st["total"]), "round": rnd,
               "wall_s": round(time.time() - t_r, 1), "search_s": round(t_search, 1),
               "distil_s": round(t_distil, 1), "eval_s": round(t_eval, 1),
               "ticks_search": int(sm["ticks_greedy"]) + int(sm["ticks_waves"]),
               "ticks_line": int(sm.get("ticks_line", 0)), "ticks_eval": int(ev_ticks),
               "mcts_waves": sm.get("waves"), "mcts_nodes": sm.get("nodes"),
               "mcts_cells": sm.get("cells"), "mcts_roots": sm.get("roots"),
               "mcts_best_pct": sm.get("best_pct"), "mcts_greedy_pct": sm.get("greedy_pct"),
               "mcts_raw_best_pct": sm.get("raw_best_pct"), "mcts_finished": int(
                   bool(sm.get("finished"))),
               "labels_node": sm.get("labels_node"), "labels_line": sm.get("labels_line"),
               "labels_prefix": sm.get("labels_prefix"),
               "buf_rows": buf.rows(), **{k: ds.get(k) for k in STAT_NAMES},
               "eval_score": sc.get("score"), "eval_best_pct": sc.get("best_progress_pct"),
               "eval_median_pct": sc.get("median_progress_pct"), "eval_cp": sc.get("max_cp"),
               "eval_finishes": sc.get("finishes")}
        wcsv.writerow([row.get(c) for c in CSV_COLS])
        fcsv.flush()
        print(f"[{int(st['total']):>14,}] az round {rnd}: search {t_search:.0f}s "
              f"(best {sm.get('best_pct')}% vs greedy {sm.get('greedy_pct')}%, "
              f"{sm.get('nodes'):,} nodes, {sm.get('cells'):,} cells), labels "
              f"{sm.get('labels_node')}/{sm.get('labels_line')}/{sm.get('labels_prefix')} "
              f"node/line/prefix, distil "
              f"{t_distil:.0f}s (loss {ds.get('loss', float('nan')):.3f}, acc "
              f"{ds.get('acc', float('nan')):.3f}, value mse "
              f"{ds.get('value_mse', float('nan')):.4f}, yaw sigma "
              f"{ds.get('yaw_sigma', float('nan')):.3f}), eval {t_eval:.0f}s: score "
              f"{sc.get('score')} best {sc.get('best_progress_pct')}% CP{sc.get('max_cp')} "
              f"fin {sc.get('finishes')}" + ("  NEW BEST" if best else ""), flush=True)
    fcsv.close()
    print(f"az: done at round {st['round']}, {int(st['total']):,} ticks", flush=True)
