"""Pre-launch fixes (2026-11-24, the Fable code and research reviews; the user: "Fix it", "lets
add them").

(a) a --wd-skip-norms checkpoint RESUMES. torch numbers a saved optimizer's parameters in
    param_groups order (weight matrices first, then biases / norm parameters / log-std), every
    warm-start helper read them in model order, and widen_for_route refused the first moment of
    the wrong shape - so the round-2 architecture could not resume at all (ledger, "known, not
    changed"). Now: opt_saved_groups names each saved index ("opt_names", or the checkpoint's
    own config for an older one), opt_state_to_model_order re-keys before the helpers and
    opt_state_to_run regroups by name before opt.load_state_dict: every parameter gets its own
    exp_avg / exp_avg_sq / step back, a widened resume still pads the right tensors, and a
    one-group checkpoint is not touched.
(b) --update-stats: train/clip_frac, train/grad_norm, train/grad_clipped and the zero-update
    numerics check (train/noise0_*), logging only.
(c) trainer smokes (CPU, recipe v14 at smoke scale on two maps, MM_TEST_MAPS): the round-2 arch
    with --update-stats trains, writes the columns, records opt_names equal to the re-derived
    order, and resumes - from its own checkpoint and from one without opt_names; the flags off
    are bit-identical to the PARENT commit (config, every eval trajectory, progress.csv minus fps,
    the final weights).
"""
from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(HERE))

import torch                                                   # noqa: E402

import train_fast as tf                                        # noqa: E402

PARENT = "b8630650"            # mm/algo head the fixes were built on (mm/prelaunch's base)
LW, LH, CH, RD = 64, 32, 2, 16


def make(**kw):
    torch.manual_seed(0)
    return tf.Policy(tf.N_SCALAR + RD + LW * LH * CH, LW, LH, emb=64, hidden=32, in_ch=CH,
                     simba=True, view_continuous=True, view_absolute="velocity", route_dim=RD,
                     conv_norm=True, final_ln_affine=False, mlp_act="leaky", **kw)


def grouped_opt(p, lr=3e-4):
    """the trainer's --wd 0.01 --wd-skip-norms 1 optimizer"""
    dec = [q for q in p.parameters() if q.ndim >= 2]
    nod = [q for q in p.parameters() if q.ndim < 2]
    return torch.optim.AdamW([{"params": dec, "weight_decay": 0.01},
                              {"params": nod, "weight_decay": 0.0}], lr=lr, eps=1e-5)


def stepped(p, opt, seed=3):
    g = torch.Generator().manual_seed(seed)
    for q in p.parameters():
        q.grad = torch.randn(q.shape, generator=g)
    opt.step()


def ckpt(p, opt, cfg, names=False):
    ck = {"policy": copy.deepcopy(p.state_dict()), "optimizer": copy.deepcopy(opt.state_dict()),
          "config": dict(cfg)}
    if names:
        nm = {id(q): n for n, q in p.named_parameters()}
        ck["opt_names"] = [nm[id(q)] for g in opt.param_groups for q in g["params"]]
    return ck


def resume(ck, p2, opt2, helpers=()):
    """the trainer's resume order: name the saved groups, re-key, the helpers, regroup, load"""
    saved = tf.opt_saved_groups(ck, p2)
    rk = saved is not None and tf.opt_state_to_model_order(ck, p2, saved)
    for h in helpers:
        h(ck, p2)
    grouped = len(opt2.param_groups) > 1
    if rk or grouped:
        tf.opt_state_to_run(ck, p2, opt2, saved)
    p2.load_state_dict(ck["policy"])
    opt2.load_state_dict(ck["optimizer"])
    return rk


def same_state(a, b, name):
    return all(torch.equal(a[k], b[k]) for k in ("exp_avg", "exp_avg_sq", "step")), name


# ==========================================================================
# (a) the optimizer state by name
# ==========================================================================
def test_the_old_reading_refused_a_grouped_checkpoint():
    p = make()
    opt = grouped_opt(p)
    stepped(p, opt)
    ck = ckpt(p, opt, {"wd": 0.01, "wd_skip_norms": 1})
    with pytest.raises(SystemExit, match="cannot warm-start this checkpoint"):
        tf.widen_for_route(ck, make())          # the pre-fix resume path, as it was


@pytest.mark.parametrize("names", [False, True])
def test_a_grouped_checkpoint_resumes_with_every_moment_on_its_own_parameter(names):
    p = make()
    opt = grouped_opt(p)
    stepped(p, opt)
    ck = ckpt(p, opt, {"wd": 0.01, "wd_skip_norms": 1}, names=names)
    p2 = make()
    opt2 = grouped_opt(p2, lr=1e-3)
    assert resume(ck, p2, opt2, helpers=(tf.widen_for_route,))
    a = {n: opt.state[q] for n, q in p.named_parameters()}
    b = {n: opt2.state[q] for n, q in p2.named_parameters()}
    assert set(a) == set(b)
    for n in a:
        assert same_state(a[n], b[n], n)[0], n
    # the checkpoint's group hyperparameters, as load_state_dict keeps them (lr is re-set after)
    assert [g["weight_decay"] for g in opt2.param_groups] == [0.01, 0.0]
    assert [g["lr"] for g in opt2.param_groups] == [3e-4, 3e-4]


def test_the_rule_derives_the_names_a_new_checkpoint_records():
    p = make()
    opt = grouped_opt(p)
    stepped(p, opt)
    ck = ckpt(p, opt, {"wd": 0.01, "wd_skip_norms": 1}, names=True)
    derived = tf.opt_saved_groups({k: v for k, v in ck.items() if k != "opt_names"}, make())
    assert [n for _, ns in derived for n in ns] == ck["opt_names"]
    # --dae-pi-lr's split on top: [dec & pi, dec - pi, nod & pi, nod - pi]
    named = [(n, q.ndim) for n, q in p.named_parameters()]
    rule = tf.opt_group_rule(named, 0.01, 1, 0.5)
    assert len(rule) == 4 and all(n.startswith(tf.OPT_PI_PREFIX) for n in rule[0] + rule[2])
    assert not any(n.startswith(tf.OPT_PI_PREFIX) for n in rule[1] + rule[3])
    assert sorted(n for g in rule for n in g) == sorted(n for n, _ in named)


def test_a_one_group_checkpoint_is_not_touched():
    p = make()
    opt = torch.optim.Adam(p.parameters(), lr=3e-4, eps=1e-5)
    stepped(p, opt)
    ck = ckpt(p, opt, {})
    before = copy.deepcopy(ck["optimizer"])
    saved = tf.opt_saved_groups(ck, make())
    assert len(saved) == 1
    assert not tf.opt_state_to_model_order(ck, make(), saved)
    assert ck["optimizer"]["param_groups"] == before["param_groups"]
    assert ck["optimizer"]["state"].keys() == before["state"].keys()
    for i in before["state"]:
        assert ck["optimizer"]["state"][i] is not None
        assert same_state(ck["optimizer"]["state"][i], before["state"][i], i)[0]


def make_flat_view(**kw):
    torch.manual_seed(0)
    return tf.Policy(tf.N_SCALAR + RD + LW * LH * CH, LW, LH, emb=64, hidden=32, in_ch=CH,
                     simba=True, route_dim=RD, conv_norm=True, final_ln_affine=False,
                     mlp_act="leaky", **kw)


def test_a_widened_resume_onto_a_grouped_checkpoint_pads_the_right_moments():
    """--priv-critic onto a --wd-skip-norms checkpoint: value_head grows trailing zero columns
    (its moments too), the fresh priv MLP joins both groups with no moments. (Bins view: under
    --view-continuous the view head is registered AFTER the priv MLP and widen_for_priv refuses
    such a checkpoint with or without groups - pre-existing, unchanged.)"""
    p = make_flat_view()
    opt = grouped_opt(p)
    stepped(p, opt)
    ck = ckpt(p, opt, {"wd": 0.01, "wd_skip_norms": 1})
    p2 = make_flat_view(priv_dim=8, priv_hidden=16)
    opt2 = grouped_opt(p2)
    resume(ck, p2, opt2, helpers=(tf.widen_for_priv,))
    a = {n: opt.state[q] for n, q in p.named_parameters()}
    for n, q in p2.named_parameters():
        st = opt2.state.get(q)
        if n.startswith("priv_mlp."):
            assert not st, n                     # fresh: no moments
            continue
        if n == "value_head.weight":
            w = a[n]["exp_avg"].shape[1]
            assert torch.equal(st["exp_avg"][:, :w], a[n]["exp_avg"])
            assert torch.equal(st["exp_avg"][:, w:], torch.zeros_like(st["exp_avg"][:, w:]))
            continue
        assert same_state(a[n], st, n)[0], n


def test_a_one_group_checkpoint_regroups_into_a_grouped_run_with_the_runs_own_decay():
    """an Adam (no --wd) checkpoint resumed with --wd 0.01 --wd-skip-norms 1 PASSED: the moments
    by name, the weight decay this run's flags give (load_state_dict died on the group count)"""
    p = make()
    opt = torch.optim.Adam(p.parameters(), lr=3e-4, eps=1e-5)
    stepped(p, opt)
    ck = ckpt(p, opt, {})
    p2 = make()
    opt2 = grouped_opt(p2)
    resume(ck, p2, opt2)
    a = {n: opt.state[q] for n, q in p.named_parameters()}
    for n, q in p2.named_parameters():
        assert same_state(a[n], opt2.state[q], n)[0], n
    assert [g["weight_decay"] for g in opt2.param_groups] == [0.01, 0.0]


def test_group_sizes_that_do_not_fit_the_config_are_refused():
    p = make()
    opt = grouped_opt(p)
    stepped(p, opt)
    ck = ckpt(p, opt, {})                        # two groups, but a config that says one
    with pytest.raises(SystemExit, match="cannot be matched to parameters by name"):
        tf.opt_saved_groups(ck, make())


# ==========================================================================
# (c) trainer smokes
# ==========================================================================
import v14_smoke as V                                          # noqa: E402

KINDS5 = ("goal", "goalc", "sdf", "occ", "slabocc")            # no face-id: --obs-normal 0
MAPS = V.MM_MAPS[:2]
needs_two = pytest.mark.skipif(
    not (V.DLL.exists() and len(MAPS) == 2 and all(
        m.exists() and all(m.with_suffix(f".{k}_32.npz").exists() for k in KINDS5)
        for m in MAPS)), reason="MM_TEST_MAPS: two maps with their cell-32 caches")
SM = ["--obs-normal", "0", "--envs", "64", "--maxvel", "4000"]
ARCH = ["--wd", "0.01", "--wd-skip-norms", "1", "--conv-norm", "1", "--final-ln-affine", "0",
        "--mlp-act", "leaky"]
UCOLS = ["train/clip_frac", "train/grad_norm", "train/grad_clipped", "train/noise0_logp",
         "train/noise0_logp_max", "train/noise0_clip"]


def _maps():
    return ["--maps", ",".join(str(m) for m in MAPS)]


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


@needs_two
def test_update_stats_and_a_grouped_resume():
    r, d = V.run_line(V.ROOT, "pre_on", _maps(), steps=8 * 64 * 12,
                      extra=SM + ARCH + ["--update-stats", "1"], timeout=5400)
    out = r.stdout + r.stderr
    names = ("pre_on", "pre_re", "pre_re_old")
    try:
        assert r.returncode == 0, out[-5000:]
        assert "update: clip_frac" in out and "at zero update" in out
        cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
        assert cfg["update_stats"] == 1 and cfg["wd_skip_norms"] == 1
        sys.path.insert(0, str(V.ROOT / "tools"))
        import record_ckpt
        record_ckpt.audit_cfg(cfg)
        head, rows = _csv(d / "progress.csv")
        assert head[-len(UCOLS):] == UCOLS and len(rows) >= 3
        for c in UCOLS:
            vals = [float(row[head.index(c)]) for row in rows]
            assert all(np.isfinite(vals)), (c, vals)
        col = {c: [float(row[head.index(c)]) for row in rows] for c in UCOLS}
        assert all(0.0 <= v <= 1.0 for c in ("train/clip_frac", "train/grad_clipped",
                                              "train/noise0_clip") for v in col[c])
        assert all(v > 0.0 for v in col["train/grad_norm"])
        # fp32 on CPU, the rollout's own weights: the stored log-prob is the update's to
        # rounding - a temperature / mask / observation mismatch would show here in nats
        assert max(col["train/noise0_logp"]) < 1e-3, col["train/noise0_logp"]
        # the checkpoint names its optimizer order, and the config rule re-derives it
        ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
        on = set(ck["opt_names"])
        named = [(k, v.dim()) for k, v in ck["policy"].items() if k in on]
        assert [n for g in tf.opt_group_rule(named, 0.01, 1) for n in g] == ck["opt_names"]
        # resume: from the checkpoint as written, and from one written before "opt_names"
        old = d / "ckpt_old.pt"
        torch.save({k: v for k, v in ck.items() if k != "opt_names"}, old)
        os.environ["SELF_STATES"] = "1"            # the run's own spine is policy-derived
        for name, src in (("pre_re", d / "ckpt_final.pt"), ("pre_re_old", old)):
            r2, d2 = V.run_line(V.ROOT, name, _maps(), steps=8 * 64 * 16,
                                extra=SM + ["--ckpt", str(src)], timeout=5400)
            o2 = r2.stdout + r2.stderr
            assert r2.returncode == 0, (name, o2[-5000:])
            assert "matched to this run's parameters BY NAME" in o2, name
            assert "update_stats=1" in o2 and "wd_skip_norms=1" in o2, name
    finally:
        os.environ.pop("SELF_STATES", None)
        for name in (() if os.environ.get("PRELAUNCH_KEEP") else names):
            shutil.rmtree(V.ROOT / "runs" / name, ignore_errors=True)


def _parent_tree(dst):
    pre = os.environ.get("PRELAUNCH_PARENT_TREE")
    if pre:
        return Path(pre)
    dst.mkdir(parents=True, exist_ok=True)
    tar = dst / "base.tar"
    subprocess.run(["git", "-C", str(V.ROOT), "archive", "--format=tar", "-o", str(tar), PARENT,
                    "python"], check=True, capture_output=True)
    with tarfile.open(tar) as t:
        t.extractall(dst)
    return dst


@needs_two
def test_flags_off_are_bit_identical_to_the_parent_commit(tmp_path):
    base = _parent_tree(tmp_path / "base")
    r0, d0 = V.run_line(base, "pre_id", _maps(), extra=SM + ARCH)
    assert r0.returncode == 0, (r0.stdout + r0.stderr)[-4000:]
    r1, d1 = V.run_line(V.ROOT, "pre_id", _maps(), extra=SM + ARCH)
    assert r1.returncode == 0, (r1.stdout + r1.stderr)[-4000:]
    try:
        ca = json.loads((d0 / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((d1 / "run.json").read_text(encoding="utf-8"))["config"]
        for c in (ca, cb):
            c.pop("demo_file", None)
        assert ca == cb, sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
        assert "update_stats" not in cb
        ta, tb = sorted(d0.glob("traj_*.jsonl")), sorted(d1.glob("traj_*.jsonl"))
        assert len(ta) >= 4 and [p.name for p in ta] == [p.name for p in tb]
        for x, y in zip(ta, tb):
            assert x.read_bytes() == y.read_bytes(), x.name
        ha, ra = _csv(d0 / "progress.csv")
        hb, rb = _csv(d1 / "progress.csv")
        assert ha == hb
        fps = ha.index("time/fps")
        assert len(ra) == len(rb) >= 3
        for x, y in zip(ra, rb):
            assert x[:fps] + x[fps + 1:] == y[:fps] + y[fps + 1:]
        sa = torch.load(d0 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(d1 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        assert sa["policy"].keys() == sb["policy"].keys()
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
        assert "opt_names" not in sa and sb["opt_names"]
        assert set(sb["opt_names"]) <= set(sb["policy"])
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        if not os.environ.get("PRELAUNCH_PARENT_TREE"):
            shutil.rmtree(d0, ignore_errors=True)
