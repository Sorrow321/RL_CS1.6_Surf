"""--eval-dump-states (pbr8b3, 2026-10-27): the trainer's greedy eval also writes the FULL
per-tick states of every episode to traj_<step>.states.npz (record_ckpt --dump-states' layout),
side by side (--eval-parallel) and sequential alike. TRAIN_ONLY in record_ckpt; the states match
the trajectory row for row; flag off bit-identical to the trainer before the change."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"
EVAL = ["--record-every", "2048", "--eval-eps", "2", "--ep-ticks", "300"]


def test_train_only():
    import record_ckpt
    assert "eval_dump_states" in record_ckpt.TRAIN_ONLY


def _rows(path):
    eps, cur = [], None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("{"):
            h = json.loads(line)
            if "episode" in h and "end" not in h:
                cur = []
                eps.append(cur)
        elif line.startswith("[") and cur is not None:
            cur.append(json.loads(line))
    return eps


@needs_pb
@pytest.mark.parametrize("par", ["1", "0"])
def test_the_states_match_the_trajectory(tmp_path, par):
    run = f"eds_smoke{par}"
    try:
        r = _run_demo(run, EVAL + ["--eval-parallel", par, "--eval-dump-states"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        d = ROOT / "runs" / run
        trajs = sorted(d.glob("traj_*.jsonl"))
        assert trajs
        for tp in trajs:
            sp = tp.with_name(tp.stem + ".states.npz")
            assert sp.exists(), sp
            z = np.load(sp)
            rows = _rows(tp)
            assert sorted(z.files) == [f"ep_{i:04d}" for i in range(len(rows))]
            for i, er in enumerate(rows):
                s = z[f"ep_{i:04d}"]
                assert len(s) == len(er), (tp.name, i, len(s), len(er))
                o = np.asarray([row[1:4] for row in er], np.float64)
                assert np.allclose(np.asarray(s["origin"], np.float64), o, atol=0.011)
        cfg = json.loads((d / "run.json").read_text())["config"]
        assert cfg["eval_dump_states"] == 1
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--eval-dump-states" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_eds.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("eds_off_new", TRAIN), ("eds_off_old", pre)):
            r = _run_demo(run, EVAL + ["--eval-parallel", "1"], tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "eds_off_new", ROOT / "runs" / "eds_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "eval_dump_states" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        assert not list(a.glob("traj_*.states.npz"))
        for ta, tb in zip(sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))):
            assert ta.read_bytes() == tb.read_bytes()
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
