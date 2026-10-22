"""--race-no-corner-cut 1 (2026-10-22, r5b4): the shaping field is the goal graph WITHOUT corner
cutting (surfgym.goalfield no_corner_cut), baked once per map into <map>.goalc_<cell>.npz.

* the trainer shapes on it (a different start geodesic), says so, writes it to the config, a
  bare resume restores it, and record_ckpt mirrors it for the observation channels;
* it is refused with --goal-field-file;
* off: the trainer before the change and after it give the same config, progress.csv (fps
  aside), eval trajectory bytes and weights.

The CPU tests need the bake in the cache (maps/surf_src_sidistic.goalc_32.npz): a full-map
bake is minutes on a GPU and hours on 4 CPU threads, so they skip without it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ez_viewc import SMOKE, _csv, _run, _train, needs_run  # noqa: E402

BAKED = ROOT / "maps" / "surf_src_sidistic.goalc_32.npz"
needs_bake = pytest.mark.skipif(not BAKED.exists(),
                                reason="needs the no-corner-cut bake in the map cache")


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
@needs_bake
def test_the_trainer_shapes_on_it_and_a_resume_and_the_recorder_follow():
    run = "nocut_on"
    r = _train(run, ["--race-ratchet", "--race-no-corner-cut", "1"], steps="2048")
    out = r.stdout
    assert "--race-no-corner-cut: the shaping field is the goal graph without corner cutting" in out
    assert "start geodesic 99675u (standard 97768u)" in out
    assert _cfg(run)["race_no_corner_cut"] == 1
    ck = ROOT / "runs" / run / "ckpt_final.pt"
    r2 = _run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"), "--ckpt", str(ck),
               "--run", run + "_res", "--steps", str(2048 + 2048), "--envs", "64",
               "--record-every", "1e12", "--ckpt-every", "1e9", "--eval-eps", "1",
               "--eval-greedy-only"])
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert "race_no_corner_cut=1" in r2.stdout
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"), str(ck),
               "--episodes", "1", "--ep-ticks", "120",
               "--out", str(ROOT / "runs" / run / "rec.jsonl")])
    assert r3.returncode == 0, r3.stdout[-3000:] + r3.stderr[-3000:]
    assert "--race-no-corner-cut mirrored" in r3.stdout
    for dd in (run, run + "_res"):
        shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)


@needs_run
def test_it_is_refused_with_a_field_file():
    r = _run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"), "--run", "nocut_bad",
              "--steps", "2048"] + SMOKE
             + ["--race-no-corner-cut", "1", "--goal-field-file", str(ROOT / "nope.npz")])
    assert r.returncode != 0
    assert "not with --race-ground, --race-dist euclid or --goal-field-file" in r.stdout + r.stderr
    shutil.rmtree(ROOT / "runs" / "nocut_bad", ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--race-no-corner-cut" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    pre = ROOT / "python" / "_train_fast_pre_nocut.py"
    extra = ["--race-ratchet", "--record-every", "2048", "--eval-eps", "2"]
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("nocut_off_new", extra)
        _train("nocut_off_old", extra, script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "nocut_off_new", ROOT / "runs" / "nocut_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "race_no_corner_cut" not in ca
    ra, rb = _csv(a.name), _csv(b.name)
    assert ra and len(ra) == len(rb) and list(ra[0]) == list(rb[0])
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
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    for dd in (a, b):
        shutil.rmtree(dd, ignore_errors=True)
