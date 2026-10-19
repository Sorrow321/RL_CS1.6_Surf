"""CPU smoke of ``--aug-mirror`` / ``--eval-mirror`` / ``--aug-drop`` / ``--aug-noise`` /
``--aug-shift`` through the real trainer, the recorder's ``--mirror``, and the flag-off
bit-identity against the trainer before any of them existed.

The unit half (surfgym/mirror.py, surfgym/augment.py, the physics claim on the core, the
eval wrapper) is tests/python/test_mirror_aug.py. This file is what those cannot see:
the rollout's per-episode flags, the truncation bootstrap's reflected s_T, the update's
DrAC branch inside mb_step, the extra eval, the CSV block, run.json, and the recorder's
TRAIN_ONLY audit - all on the CPU harness the other trainer smokes use (CUDA hidden,
64 envs, a 16x8 camera, a few thousand steps).

    python -m pytest tests/python/test_mirror_aug_smoke.py -q
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

from test_curiosity_cond import _csv, _run, _train            # noqa: E402
from test_view_continuous import needs_run                     # noqa: E402

RECORD = ROOT / "tools" / "record_ckpt.py"
# the from-scratch recipe's observation and action space (CLAUDE.md section 2 + the
# 2026-10-06 normals / SimBa rule): absolute continuous view, held keys, the potential
# channel with its curtain, the ego-frame normal - so the mirror's channel bookkeeping
# (depth, potential, nx, ny LEFT, nz) is the one the experiment will run
RECIPE = ["--view-continuous", "--view-absolute", "velocity", "--keys-hold",
          "--obs-potential", "norm", "--obs-potential-curtain", "--obs-normal", "1",
          "--simba", "1"]
# 64-tick episodes: every env truncates inside the run, so flags are redrawn and the
# bootstrap rebuilds (and reflects) terminal rows; an eval after every iteration
SHORT = ["--ep-ticks", "64", "--record-every", "2048"]
AUG = ["--aug-mirror", "0.5", "--eval-mirror", "1", "--aug-drop", "0.25",
       "--aug-noise", "0.05", "--aug-shift", "2"]
AUG_COLS = ["aug/mirror_frac", "aug/kl", "aug/v_mse", "race/eval_progress_mirror",
            "race/eval_finishes_mirror", "race/map_pct_mirror"]
AUG_KEYS = ("aug_mirror", "eval_mirror", "aug_drop", "aug_noise", "aug_shift", "aug_coef")


def _header(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


@needs_run
def test_mirror_and_drac_train_eval_and_record_end_to_end():
    run = "cya_mirror_aug_smoke"
    r = _train(run, RECIPE + SHORT + AUG)
    out = r.stdout
    assert "--aug-mirror: 50% of training episodes fly the MIRROR IMAGE" in out
    assert "channels [3] negated; scalar columns [1, 10] negated" in out
    assert "DrAC image augmentation in the update (shift +-2 px" in out
    assert "greedy[MIRROR]:" in out and "drac kl " in out
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["aug_mirror"] == 0.5 and cfg["eval_mirror"] == 1
    assert cfg["aug_drop"] == 0.25 and cfg["aug_noise"] == 0.05
    assert cfg["aug_shift"] == 2 and cfg["aug_coef"] == 0.1
    rows = _csv(run)
    assert len(rows) == 3
    assert list(rows[0])[-6:] == AUG_COLS
    for x in rows:
        assert 0.2 <= float(x["aug/mirror_frac"]) <= 0.8
        kl, vm = float(x["aug/kl"]), float(x["aug/v_mse"])
        assert math.isfinite(kl) and kl >= 0.0 and math.isfinite(vm) and vm >= 0.0
        assert math.isfinite(float(x["train/loss"]))
    ev = [x for x in rows if x["race/eval_progress_mirror"] != ""]
    assert ev, "no row carries the mirrored eval"
    for x in ev:
        assert math.isfinite(float(x["race/eval_progress_mirror"]))
        assert float(x["race/eval_finishes_mirror"]) >= 0.0
        assert math.isfinite(float(x["race/map_pct_mirror"]))
    mt = sorted(d.glob("traj_*_mirror.jsonl"))
    plain = [p for p in d.glob("traj_*.jsonl") if not p.name.endswith("_mirror.jsonl")]
    assert len(mt) >= 2 and len(plain) == len(mt)
    assert _header(mt[0]).get("mirror") == 1 and "mirror" not in _header(plain[0])
    # the recorder: all six keys are TRAIN_ONLY (a plain recording is the real map) and
    # --mirror flies the mirror image, saying so in the header
    ck = d / "ckpt_final.pt"
    for extra, want in (([], None), (["--mirror"], 1)):
        dst = d / f"rec{'_mirror' if extra else ''}.jsonl"
        rr = _run([sys.executable, str(RECORD), str(ck), "--episodes", "1",
                   "--out", str(dst)] + extra, 900)
        assert rr.returncode == 0, rr.stdout[-3000:] + rr.stderr[-3000:]
        assert _header(dst).get("mirror") == want
        if extra:
            assert "--mirror: the policy flies the MIRROR IMAGE of the map" in rr.stdout
    shutil.rmtree(d, ignore_errors=True)


@needs_run
def test_mirror_refuses_what_it_cannot_reflect():
    run = "cya_mirror_refuse"
    try:
        r = _run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"), "--run", run]
                 + __import__("test_view_continuous").SMOKE_FLAGS
                 + ["--steps", "2048", "--aug-mirror", "0.5"], 900)
        assert r.returncode != 0
        assert "--aug-mirror: not with the discrete yaw bins" in r.stdout + r.stderr
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _pre_trainer(dst: Path):
    """train_fast.py of the newest first-parent commit WITHOUT --aug-mirror (HEAD while
    this change is uncommitted), as BYTES (the console is cp1251, the file UTF-8)."""
    try:
        r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                           capture_output=True, text=True, cwd=str(ROOT), timeout=60)
        refs = r.stdout.split() if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        refs = []
    for ref in refs or ["HEAD"]:
        r = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if r.returncode != 0 or b"--aug-mirror" in r.stdout:
            continue
        dst.write_bytes(r.stdout)
        return ref
    return None


@needs_run
def test_flags_off_are_bit_identical_to_the_trainer_before_them():
    """Every flag off: config, progress.csv (fps excluded, header included), the eval
    trajectories, the weights and Adam's moments all match the pre-flag trainer."""
    import torch
    old = ROOT / "python" / "train_fast_premirror.py"
    if _pre_trainer(old) is None:
        pytest.skip("no pre-mirror train_fast.py in the first-parent history")
    try:
        _train("cya_mirror_new", RECIPE + SHORT)
        _train("cya_mirror_old", RECIPE + SHORT, script=old)
    finally:
        old.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "cya_mirror_new", ROOT / "runs" / "cya_mirror_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert not any(k in ca for k in AUG_KEYS) and ca == cb
    ra, rb = _csv(a.name), _csv(b.name)
    assert len(ra) == len(rb) == 3 and list(ra[0]) == list(rb[0])
    assert not any(c in ra[0] for c in AUG_COLS)
    for x, y in zip(ra, rb):
        for k in x:
            if k != "time/fps":
                assert x[k] == y[k], k
    ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
    assert ta and [p.name for p in ta] == [p.name for p in tb]
    for p, q in zip(ta, tb):
        assert p.read_bytes() == q.read_bytes(), p.name
    sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert set(sa["policy"]) == set(sb["policy"])
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    oa, ob = sa["optimizer"]["state"], sb["optimizer"]["state"]
    for i in oa:
        for k in oa[i]:
            if torch.is_tensor(oa[i][k]):
                assert torch.equal(oa[i][k], ob[i][k]), (i, k)
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
