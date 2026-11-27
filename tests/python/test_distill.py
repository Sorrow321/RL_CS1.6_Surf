"""test_distill.py - train_fast.py --distill (mm/distill, 2026-11-27): DAgger as online policy
distillation from a frozen teacher checkpoint (surfgym/distill.py).

(a) the terms: KL(teacher || student) over the padded categorical heads is the exact categorical
    KL (the -1e30 pad slots contribute 0), the view term is the Gaussian closed form with each
    side's own sigma, the value term the MSE; the teacher's side carries no gradient;
(b) the rollout's drive mask (DAgger's beta) and its linear decay; the semantic check of the
    teacher's config against the run's;
(c) trainer smokes (CPU, v14 at smoke scale on two maps, MM_TEST_MAPS): with the flags off the
    run is bit-identical to the parent commit; a student resumed from its own teacher's weights
    under pure distillation gets a zero gradient and stays the teacher (KL exactly 0, agreement
    1) - the same rows reach both networks and no PPO term leaks in; a fresh student closes on
    its teacher, the drive share follows its schedule, a driven rollout's stored actions are not
    the student's own draws, the config / record gate / bare resume carry the flags; and the
    refusals (another architecture, other physics, drive with a PPO share, flags without
    --distill).
"""
from __future__ import annotations

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
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))

from surfgym import distill as D      # noqa: E402

PARENT = "2519d7f6"            # mm/m20 head mm/distill was built on

NEG = -1e30


def _pad(logits, widths, npad):
    """(B, sum(widths)) -> (B, heads, npad) with HeadPacker.pad's -1e30 slots"""
    out = torch.full((logits.shape[0], len(widths), npad), NEG)
    o = 0
    for h, w in enumerate(widths):
        out[:, h, :w] = logits[:, o:o + w]
        o += w
    return out


# ==========================================================================
# (a) the terms
# ==========================================================================
def test_the_terms_are_the_exact_kls_and_the_teacher_carries_no_gradient():
    g = torch.Generator().manual_seed(0)
    widths, npad, B = (4, 4, 2, 3), 15, 32
    lt = torch.randn(B, sum(widths), generator=g) * 2.0
    ls = torch.randn(B, sum(widths), generator=g, requires_grad=True)
    pt, ps = _pad(lt, widths, npad), _pad(ls, widths, npad)
    mu_t, mu_s = torch.randn(B, 2, generator=g), torch.randn(B, 2, generator=g).requires_grad_()
    lst = torch.tensor([-1.6, -0.9])
    lss = torch.tensor([-1.2, -0.7], requires_grad=True)
    vt, vs = torch.randn(B, generator=g), torch.randn(B, generator=g).requires_grad_()
    kl, kv, vm, ag = D.terms(pt, ps, vt, vs, mu_t, mu_s, lst, lss)
    # the categorical KL, head by head, by torch.distributions on the UNPADDED logits
    want, o = torch.zeros(B), 0
    for w in widths:
        want += torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=lt[:, o:o + w]),
            torch.distributions.Categorical(logits=ls[:, o:o + w]))
        o += w
    assert torch.isfinite(kl) and kl.item() == pytest.approx(want.mean().item(), rel=1e-5)
    wv = torch.distributions.kl_divergence(
        torch.distributions.Normal(mu_t, lst.exp()),
        torch.distributions.Normal(mu_s, lss.exp())).sum(-1).mean()
    assert kv.item() == pytest.approx(wv.item(), rel=1e-5)
    assert vm.item() == pytest.approx((vs - vt).pow(2).mean().item(), rel=1e-6)
    o, hits = 0, []
    for w in widths:
        hits.append(lt[:, o:o + w].argmax(-1) == ls[:, o:o + w].argmax(-1))
        o += w
    assert ag.item() == pytest.approx(torch.stack(hits).float().mean().item())
    # gradients reach the student (logits, view means, its sigma, its value) and stay finite
    (kl + kv + vm).backward()
    for t in (ls, mu_s, lss, vs):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    # the teacher's side is a fixed target even when it requires grad
    pt2 = pt.detach().clone().requires_grad_()
    vt2 = vt.detach().clone().requires_grad_()
    ps2 = ps.detach().clone().requires_grad_()
    vs2 = vs.detach().clone().requires_grad_()
    k2, _, v2, _ = D.terms(pt2, ps2, vt2, vs2)
    (k2 + v2).backward()
    assert pt2.grad is None and vt2.grad is None
    assert ps2.grad is not None and vs2.grad is not None


def test_a_student_equal_to_its_teacher_scores_zero_everywhere():
    g = torch.Generator().manual_seed(1)
    widths, npad, B = (4, 4, 2, 3), 15, 16
    lt = torch.randn(B, sum(widths), generator=g)
    p = _pad(lt, widths, npad)
    mu = torch.randn(B, 2, generator=g)
    lsd = torch.tensor([-1.3, -0.8])
    v = torch.randn(B, generator=g)
    kl, kv, vm, ag = D.terms(p, p.clone().requires_grad_(), v, v.clone(), mu, mu.clone(),
                             lsd, lsd.clone())
    assert kl.item() == 0.0 and abs(kv.item()) < 1e-7 and vm.item() == 0.0 and ag.item() == 1.0
    # the discrete policy's form: no view term
    kl, kv, vm, ag = D.terms(p, p.clone(), v, v.clone())
    assert kv.item() == 0.0 and kl.item() == 0.0


# ==========================================================================
# (b) the drive mask, its decay, the semantic check
# ==========================================================================
class _S:
    def __init__(self, lo, hi, tag="m"):
        self.lo, self.hi, self.tag = lo, hi, tag


def test_the_drive_mask_takes_the_first_share_of_every_map_and_decays():
    slots = [_S(0, 8), _S(8, 16), _S(16, 20)]
    m = D.drive_mask(slots, 20, 0.25)
    assert m.dtype == torch.bool and m.tolist() == (
        [True] * 2 + [False] * 6 + [True] * 2 + [False] * 6 + [True] + [False] * 3)
    assert not D.drive_mask(slots, 20, 0.0).any() and D.drive_mask(slots, 20, 1.0).all()
    assert D.drive_at(123.0, 0.5, None) == 0.5
    assert D.drive_at(0.0, 0.5, 100.0) == 0.5
    assert D.drive_at(50.0, 0.5, 100.0) == pytest.approx(0.25)
    assert D.drive_at(150.0, 0.5, 100.0) == 0.0


def test_the_semantic_check_names_what_differs():
    t = {"act_every": 4, "keys_hold": True, "tick_ms": 7.63, "hidden": 448,
         "maxvel_maps": {"a": 3500.0, "b": 4000.0}, "only_teacher": 1}
    s = dict(t, maxvel_maps={"a": 3500.0, "b": 4000.0, "c": 3500.0}, only_student=2,
             mlp_act="relu", conv_norm=0)
    assert D.semantic_mismatch(t, s) == []
    s2 = dict(s, act_every=3, maxvel_maps={"a": 3500.0, "b": 3500.0}, mlp_act="leaky")
    got = {k for k, _, _ in D.semantic_mismatch(t, s2)}
    assert got == {"act_every", "maxvel_maps[b]", "mlp_act"}
    assert D.unseen_maps(t, ["a", "b", "c"]) == ["c"]
    assert D.unseen_maps({}, ["a"]) == []


# ==========================================================================
# (c) trainer smokes
# ==========================================================================
import v14_smoke as V                                          # noqa: E402

KINDS5 = ("goal", "goalc", "sdf", "occ", "slabocc")            # no face-id: --obs-normal 0
MAPS = V.MM_MAPS[:2]
NO_TWO = not (V.DLL.exists() and len(MAPS) == 2 and all(
    m.exists() and all(m.with_suffix(f".{k}_32.npz").exists() for k in KINDS5) for m in MAPS))
needs_two = pytest.mark.skipif(NO_TWO, reason="MM_TEST_MAPS: two maps with their cell-32 caches")
SM = ["--obs-normal", "0", "--envs", "64", "--maxvel", "4000"]
IT = 64 * 8 * 4                                                # env steps per iteration
STEPS = IT * 6


def _maps():
    return ["--maps", ",".join(str(m) for m in MAPS)]


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    dup = sorted({h for h in rows[0] if rows[0].count(h) > 1})
    assert not dup, f"duplicate CSV columns: {dup}"
    return rows[0], rows[1:]


def _col(head, rows, name):
    i = head.index(name)
    return [float(r[i]) for r in rows if r[i] != ""]


def _parent_tree(dst):
    pre = os.environ.get("DISTILL_PARENT_TREE")
    if pre:
        return Path(pre)
    dst.mkdir(parents=True, exist_ok=True)
    tar = dst / "base.tar"
    subprocess.run(["git", "-C", str(V.ROOT), "archive", "--format=tar", "-o", str(tar), PARENT,
                    "python"], check=True, capture_output=True)
    with tarfile.open(tar) as t:
        t.extractall(dst)
    return dst


DCOLS = ["distill/kl", "distill/kl_view", "distill/v_mse", "distill/agree", "distill/drive"]


@needs_two
def test_flags_off_is_bit_identical_to_the_parent(tmp_path):
    base = _parent_tree(tmp_path / "base")
    name = "ds_id"
    r0, d0 = V.run_line(base, name, _maps(), steps=STEPS, extra=SM)
    assert r0.returncode == 0, (r0.stdout + r0.stderr)[-4000:]
    r1, d1 = V.run_line(V.ROOT, name, _maps(), steps=STEPS, extra=SM)
    assert r1.returncode == 0, (r1.stdout + r1.stderr)[-4000:]
    try:
        V.same_run(d0, d1, n_traj=2, drop_cfg=("demo_file",))
        cb = json.loads((d1 / "run.json").read_text(encoding="utf-8"))["config"]
        assert not [k for k in cb if k.startswith("distill")]
        sa = torch.load(d0 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(d1 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        assert sa["policy"].keys() == sb["policy"].keys()
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
        assert "--distill" not in r1.stdout
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        if not os.environ.get("DISTILL_PARENT_TREE"):
            shutil.rmtree(d0, ignore_errors=True)


@pytest.fixture(scope="module")
def teacher():
    """a smoke teacher with one weight average (--eval-ema), its ckpt_final.pt; trained at a
    high learning rate so its policy is far from the uniform one a fresh network starts at"""
    if NO_TWO:
        pytest.skip("MM_TEST_MAPS: two maps with their cell-32 caches")
    r, d = V.run_line(V.ROOT, "ds_teacher", _maps(), steps=STEPS,
                      extra=SM + ["--eval-ema", "4096", "--lr", "3e-3", "--epochs", "4"])
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    yield d / "ckpt_final.pt"
    if not os.environ.get("DISTILL_KEEP"):
        shutil.rmtree(d, ignore_errors=True)


def _block(head):
    """the five distill/* columns: present, contiguous, in order (a --maps run's per-map block
    follows them)"""
    i = head.index(DCOLS[0])
    assert head[i:i + len(DCOLS)] == DCOLS, head[i:i + len(DCOLS)]


@needs_two
def test_a_student_that_is_its_teacher_scores_zero(teacher):
    """the teacher's run resumed with --lr 0 and distilled from its own raw weights: the two
    networks are the same weights for the whole run, so every minibatch's KL, view KL and value
    term is EXACTLY 0 and the agreement 1 - which only holds if both networks get the
    same rows and no PPO term reaches the loss: the gradient is rounding only (a resume carries
    Adam's moments, hence lr 0, so the weights alone could not show it)"""
    os.environ["SELF_STATES"] = "1"            # the teacher run's own spine is policy-derived
    try:
        r, d = V.run_line(V.ROOT, "ds_self", _maps(), steps=STEPS + IT * 3,
                          extra=SM + ["--ckpt", str(teacher), "--distill", str(teacher),
                                      "--lr", "0", "--update-stats", "1"])
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-5000:]
        assert "--distill: teacher " in out and "(pure distillation: no reward gradient)" in out
        assert "--distill: the teacher's config agrees with this run's" in out
        head, rows = _csv(d / "progress.csv")
        _block(head)
        kl, kv, vm, ag = (_col(head, rows, c) for c in DCOLS[:4])
        assert len(kl) >= 3
        assert all(x == 0.0 for x in kl) and all(x == 0.0 for x in vm), (kl, vm)
        assert all(x == 0.0 for x in kv) and all(x == 1.0 for x in ag), (kv, ag)
        assert all(x == 0.0 for x in _col(head, rows, "distill/drive"))
        # the BACKWARDED loss has no gradient but rounding: a softmax sums to 1 only to an ulp,
        # so log_softmax's backward leaves p (sum p - 1) ~ 1e-8 per logit; a PPO term would put it
        # at a training run's scale (train/loss is the PPO terms' sum, logged for the record and
        # never backwarded under --distill-rl 0)
        gn = _col(head, rows, "train/grad_norm")
        assert len(gn) >= 3 and max(gn) < 1e-4, gn
        assert any(x != 0.0 for x in _col(head, rows, "train/loss"))
        a = torch.load(teacher, map_location="cpu", weights_only=False)["policy"]
        b = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)["policy"]
        for k in a:
            assert torch.equal(a[k], b[k]), k
    finally:
        os.environ.pop("SELF_STATES", None)
        shutil.rmtree(V.ROOT / "runs" / "ds_self", ignore_errors=True)


@needs_two
def test_a_fresh_student_closes_on_its_teacher(teacher):
    names = ("ds_on", "ds_re")
    try:
        r, d = V.run_line(V.ROOT, "ds_on", _maps(), steps=IT * 12,
                          extra=SM + ["--seed", "1", "--lr", "1e-3", "--epochs", "4",
                                      "--update-stats", "1",
                                      "--distill", str(teacher), "--distill-ema", "0",
                                      "--distill-drive", "0.5",
                                      "--distill-drive-steps", str(IT * 6)])
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-5000:]
        assert "its weight average 0 (half-life 4,096 steps)" in out
        assert "the teacher draws the actions of 50% of every map's envs" in out
        cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
        assert cfg["distill"] == str(teacher) and cfg["distill_ema"] == 0
        assert cfg["distill_drive"] == 0.5 and cfg["distill_drive_steps"] == IT * 6
        assert cfg["distill_rl"] == 0.0 and cfg["distill_coef"] == 1.0
        assert len(cfg["distill_md5"]) == 32 and cfg["distill_step"] > 0
        import record_ckpt
        record_ckpt.audit_cfg(cfg)
        head, rows = _csv(d / "progress.csv")
        _block(head)
        kl = _col(head, rows, "distill/kl")
        ag = _col(head, rows, "distill/agree")
        dr = _col(head, rows, "distill/drive")
        assert len(kl) >= 10 and all(np.isfinite(kl)) and all(0.0 <= x <= 1.0 for x in ag)
        # closes on the teacher: the late KL well under the first
        assert kl[0] > 1e-3 and np.mean(kl[-3:]) < 0.7 * kl[0], kl
        # beta: 0.5 at step 0, linear to 0 at IT * 6, then 0 (16 of the 32 envs per map at 0.5)
        assert dr[0] == 0.5 and dr[-1] == 0.0 and all(a >= b for a, b in zip(dr, dr[1:])), dr
        # a driven env's STORED action is the teacher's draw, not the student's: at zero update
        # the student's log-prob of it is far from the stored one (its own draw's) while beta >
        # 0, and the two agree to rounding once the student flies every env
        n0 = _col(head, rows, "train/noise0_logp")
        k = sum(1 for x in dr if x >= 0.25)
        assert k >= 2 and np.mean(n0[:k]) > 0.05 and np.mean(n0[-3:]) < 1e-3, (n0, dr)
        # a bare resume carries the flags on (the teacher path, its average, the drive)
        os.environ["SELF_STATES"] = "1"
        r2, d2 = V.run_line(V.ROOT, "ds_re", _maps(), steps=IT * 14,
                            extra=SM + ["--ckpt", str(d / "ckpt_final.pt")])
        o2 = r2.stdout + r2.stderr
        assert r2.returncode == 0, o2[-5000:]
        assert f"distill={teacher}" in o2 and "distill_ema=0" in o2
        assert json.loads((d2 / "run.json").read_text(encoding="utf-8"))["config"][
            "distill_md5"] == cfg["distill_md5"]
    finally:
        os.environ.pop("SELF_STATES", None)
        for name in (() if os.environ.get("DISTILL_KEEP") else names):
            shutil.rmtree(V.ROOT / "runs" / name, ignore_errors=True)


@needs_two
def test_the_refusals(teacher):
    def fails(name, extra, needle):
        r, _ = V.run_line(V.ROOT, name, _maps(), steps=IT * 2, extra=SM + extra, timeout=900)
        out = r.stdout + r.stderr
        shutil.rmtree(V.ROOT / "runs" / name, ignore_errors=True)
        assert r.returncode != 0 and needle in out, out[-3000:]

    # another network: the strict load is the architecture check
    fails("ds_x1", ["--distill", str(teacher), "--hidden", "32"],
          "the teacher is not this run's architecture (strict load)")
    # other physics / action timing: the semantic check names the key
    fails("ds_x2", ["--distill", str(teacher), "--act-every", "3"], "act_every: teacher 4")
    # a driven rollout under a PPO share
    fails("ds_x3", ["--distill", str(teacher), "--distill-drive", "0.5", "--distill-rl", "1"],
          "--distill-drive with --distill-rl > 0")
    # the knobs without a teacher
    fails("ds_x4", ["--distill-coef", "2"], "--distill-* without --distill")
    fails("ds_x5", ["--distill", str(teacher) + ".missing"], "no such checkpoint")
    fails("ds_x6", ["--distill", str(teacher), "--distill-ema", "3"],
          "the teacher carries 1 weight average(s)")
