"""--auto-beam N (pbr3b3, 2026-10-24): the search -> curriculum loop inside the run. After a greedy
eval with enough episodes past the --demo-file spine's --demo-success-d line, the trainer saves its
weights, runs tools/beam_tas.py from the map start with the policy's own proposals, cuts the
deepest lineage at its deepest point on the shaping field and, when that carries the success line
far enough, makes it the curriculum's spine (runs/<run>/auto_spine_<k>.npy; the config's
demo_file follows it so a resume continues on it).

CPU, princessburglar, tiny scratch model: the swap happens and is logged, the spine file and the
search's frontier exist, the checkpoint names the new spine; without --demo-success-d it is
refused; flag off is bit-identical to the trainer before the flag."""
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
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _spine  # noqa: E402
from test_ge_minutes import SMOKE, _csv, _env, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


def _smoke():
    s = list(SMOKE)
    s[s.index("--ep-ticks") + 1] = "400"      # short greedy episodes -> a short search
    return s


def _run(run, extra, tmp_path, script=TRAIN, steps="2048"):
    sp = _spine(tmp_path)
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    env = _env()
    env["SELF_STATES"] = "1"
    return subprocess.run([sys.executable, "-u", str(script), "--run", run] + _smoke()
                          + ["--steps", steps, "--race-ratchet", "--demo-file", str(sp),
                             "--demo-window", "5"] + list(extra),
                          capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=1800,
                          encoding="utf-8", errors="replace")


AB = ["--auto-beam", "8", "--auto-beam-cross", "0.5", "--auto-beam-secs", "1",
      "--auto-beam-margin", "-1", "--auto-beam-args",
      "--resample-every 5 --score d --greedy-envs 2"]


@needs_pb
def test_the_spine_is_swapped_for_the_searched_line(tmp_path):
    run = "ab_on"
    try:
        # the success line 5,000 u ABOVE the spine's end: every greedy episode starts past it
        r = _run(run, ["--demo-success-d", "5000", "--demo-success-bonus", "50"] + AB, tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-beam 8: when 50% of a greedy eval's episodes pass" in out
        assert "--auto-beam #1: 2/2 greedy episodes past" in out, out[-6000:]
        assert "--auto-beam #1: SPINE SWAP -> auto_spine_01.npy" in out, out[-6000:]
        d = ROOT / "runs" / run
        sp = np.load(d / "auto_spine_01.npy")
        assert sp.dtype.names is not None and len(sp) >= 1
        z = np.load(d / "auto_beam_01" / "beam_frontier.npz")
        assert int(z["replay_ok"]) == 1
        # the cut is the frontier line's deepest point: a prefix of its states
        st = z["states"]
        assert len(sp) <= len(st) and sp.tobytes() == st[:len(sp)].tobytes()
        cfg = json.loads((d / "run.json").read_text())["config"]
        assert cfg["auto_beam"]["envs"] == 8
        import torch
        ck = torch.load(d / "ckpt_latest.pt", map_location="cpu", weights_only=False)
        assert ck["config"]["demo_file"].endswith("auto_spine_01.npy")
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def test_the_cut_drops_the_final_fall():
    from surfgym.core import STATE_DTYPE
    from train_fast import last_contact_cut
    S = np.zeros(60, STATE_DTYPE)
    S["onground"] = -1
    vz = np.zeros(60)
    vz[:20] = 50.0                         # on a ramp: vz held by the surface
    vz[20:] = 50.0 - 8.0 * np.arange(1, 41)  # then free fall at one gravity step per tick
    S["velocity"][:, 2] = vz
    assert last_contact_cut(S, 59, 8.0) == 19         # the take-off state (19 -> 20 is ballistic)
    assert last_contact_cut(S, 10, 8.0) == 10         # contact all the way: the deepest point
    S["onground"][45] = 0
    assert last_contact_cut(S, 59, 8.0) == 45         # standing counts as contact
    T = np.zeros(10, STATE_DTYPE)
    T["velocity"][:, 2] = -8.0 * np.arange(10)
    T["onground"] = -1
    assert last_contact_cut(T, 9, 8.0) == 9           # no contact at all: keep the whole line


@needs_pb
def test_it_needs_the_success_line(tmp_path):
    r = _run("ab_bad", AB, tmp_path)
    shutil.rmtree(ROOT / "runs" / "ab_bad", ignore_errors=True)
    assert r.returncode != 0 and "--auto-beam moves the --demo-file spine's --demo-success-d" in (
        r.stdout + r.stderr)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--auto-beam" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_ab.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("ab_off_new", TRAIN), ("ab_off_old", pre)):
            r = _run(run, ["--demo-success-d", "50"], tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "ab_off_new", ROOT / "runs" / "ab_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "auto_beam" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
