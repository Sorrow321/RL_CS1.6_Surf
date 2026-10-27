"""--auto-spine-cut-secs (pbr8b2, 2026-10-27): the loop's curriculum spine is the stored line cut
SECS of line past the greedy frontier - the launch spine at the first eval, rolled on at every
quorum, a deeper search line replacing the stored one. The cut row rule and the milestone
de-cluster on synthetic lines; CPU princessburglar: a smoke run in which the first eval cuts (or
declines to cut) the launch spine and the key is dumped, the refusals, TRAIN_ONLY in
record_ckpt, flag off bit-identical to the trainer before the change."""
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
from test_auto_spine_async import LOOP  # noqa: E402
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


def _line(n, step=10.0):
    """n rows moving +x by ``step`` u a row; the field below reads d = 10,000 - x."""
    from surfgym.respawn import STATE_DTYPE
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = step * np.arange(n)
    return s


def _d(p):
    return 10000.0 - np.asarray(p, np.float64)[:, 0]


def test_cut_row_rule():
    from train_fast import spine_cut_ahead
    s = _line(1000)                                   # d 10,000 -> 10 over the rows
    # the frontier d 9,000 is first reached at row 100; 50 ticks past it -> keep rows 0..150
    assert spine_cut_ahead(s, _d, 9000.0, 50) == 150
    assert spine_cut_ahead(s, _d, 9000.0, 49.6) == 150   # ticks round
    # a line that ends before the cut row, never reaches the frontier, or no frontier yet
    assert spine_cut_ahead(s, _d, 9000.0, 899) is None
    assert spine_cut_ahead(s, _d, 9000.0, 898) == 998
    assert spine_cut_ahead(s, _d, 5.0, 10) is None
    assert spine_cut_ahead(s, _d, None, 10) is None
    # the FIRST row at the frontier counts, even when the line comes back out of it later
    s2 = _line(300)
    s2["origin"][150:, 0] = 0.0                       # back to d 10,000 after row 150
    assert spine_cut_ahead(s2, _d, 9000.0, 20) == 120


def test_milestone_declusters():
    from train_fast import milestone_clear
    assert milestone_clear([20000.0, 15000.0], 12000.0, [11500.0], 100.0)
    assert not milestone_clear([20000.0, 12050.0], 12000.0, [11500.0], 100.0)
    assert not milestone_clear([20000.0], 12000.0, [11920.0], 100.0)
    assert milestone_clear([], 12000.0, [], 100.0)


def test_train_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    assert "auto_spine_cut_secs" in record_ckpt.TRAIN_ONLY
    assert "auto_spine_cut_beyond" in record_ckpt.TRAIN_ONLY


@needs_pb
def test_the_first_eval_cuts_the_launch_spine(tmp_path):
    run = "aspc_smoke"
    try:
        r = _run_demo(run, LOOP + ["--auto-spine-cut-secs", "0.05"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-spine-cut-secs 0.05: the curriculum's spine is the stored line" in out
        assert "is the stored line, cut at the first greedy eval" in out
        # the first eval decides: a cut (the spine reaches the frontier and runs 5 ticks past it)
        # or none (it does not)
        assert ("--auto-spine-cut-secs: the launch spine CUT at row" in out
                or "--auto-spine-cut-secs: the stored line is in use to its end; not cut" in out
                or "; not cut (greedy frontier" in out), out[-6000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine_cut_secs"] == 0.05
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    for name, extra, msg in (
            ("aspc_bad1", ["--demo-success-d", "50", "--auto-spine-cut-secs", "1"],
             "--auto-spine-cut-secs cuts --auto-spine's curriculum spine"),
            ("aspc_bad2", LOOP + ["--auto-spine-cut-secs", "-1"], "must be >= 0")):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr), name
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


STUB = '''import json, shutil, sys
from pathlib import Path
import os
a = sys.argv
out = Path(a[a.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
shutil.copyfile(os.environ["STUB_SPINE"], out / "spine.npy")
res = {"spine": str(out / "spine.npy"), "source": "stub"}
if os.environ.get("STUB_KEYS"):
    shutil.copyfile(os.environ["STUB_KEYS"], out / "spine.keys.npy")
    res["keys"] = str(out / "spine.keys.npy")
print("AUTO_SPINE " + json.dumps(res))
'''


@needs_pb
def test_a_search_line_replaces_the_stored_line_only_when_deeper(tmp_path, monkeypatch):
    """A stub search (RLSURF_AUTO_SPINE_TOOL) returns 1,000 rows of the policy's own k04c line:
    deeper than the launch spine, so the first result becomes the stored line, cut past the
    greedy frontier; the same line again is not deeper than itself, so the stored line stays."""
    k = ROOT / "autoresearch" / "spines" / "pbr7b2_k04c.npy"
    if not k.exists():
        pytest.skip("no k04c spine")
    line = tmp_path / "line.npy"
    np.save(line, np.load(k)[:1000])
    keys = tmp_path / "line.keys.npy"
    np.save(keys, np.load(k.with_name("pbr7b2_k04c.keys.npy"))[:1000])
    k40 = tmp_path / "spine40.keys.npy"            # held keys for the helper's 40-row spine
    np.save(k40, np.zeros((40, 3), np.int32))
    stub = tmp_path / "stub_auto_spine.py"
    stub.write_text(STUB)
    monkeypatch.setenv("RLSURF_AUTO_SPINE_TOOL", str(stub))
    monkeypatch.setenv("STUB_SPINE", str(line))
    monkeypatch.setenv("STUB_KEYS", str(keys))
    run = "aspc_swap"
    try:
        r = _run_demo(run, LOOP + ["--auto-spine-cut-secs", "0.05", "--auto-spine-milestones",
                                   "--demo-keys", str(k40), "--steps", "8192"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-spine-cut-secs: search 1's line (1000 rows" in out, out[-6000:]
        assert "is the stored line, cut at row" in out
        assert "NEW SPINE from stub" in out
        assert "--demo-keys: the new spine's held keys from" in out      # sliced to the cut
        assert "auto-spine search 2: its line ends at d" in out, out[-6000:]
        assert "the stored line stays" in out
        cur = json.loads((ROOT / "runs" / run / "auto" / "current.json").read_text())
        assert cur["stored_rows"] == 1000 and 0 < cur["cut_rows"] < 1000
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--auto-spine-cut-secs" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    """The loop (a search that fires and ends) and every progress.csv number, against the
    trainer before the change."""
    pre = ROOT / "python" / "_train_fast_pre_aspc.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("aspc_off_new", TRAIN), ("aspc_off_old", pre)):
            r = _run_demo(run, LOOP, tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "aspc_off_new", ROOT / "runs" / "aspc_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "auto_spine_cut_secs" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        assert (a / "auto" / "k01" / "auto_spine.log").exists()
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)


def _old_respawn():
    """surfgym/respawn.py as it was before --auto-spine-cut-beyond (first-parent history)."""
    import importlib.util
    import tempfile
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/surfgym/respawn.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"beyond_frac" not in s.stdout:
            p = Path(tempfile.mkdtemp()) / "respawn_old.py"
            p.write_bytes(s.stdout)
            spec = importlib.util.spec_from_file_location("surfgym.respawn_old", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    return None


def _demo(mod, n=200):
    s = _line(n)
    return mod.DemoCurriculum(s, window=5, rate=0.3, grow=5, front_frac=0.8, seed=7), s


def test_beyond_draws_and_flag_off_identity():
    import surfgym.respawn as new
    old = _old_respawn()
    if old is None:
        pytest.skip("no pre-change respawn.py in the first-parent history")
    start = _line(3)
    a, _ = _demo(new)
    b, _ = _demo(old)
    for _ in range(3):                                # off: byte-identical pools, same rng stream
        pa, pb = a.build_pool(start, 512), b.build_pool(start, 512)
        assert pa.tobytes() == pb.tobytes()
    c, _ = _demo(new)
    c.beyond_frac = 0.25                              # a share with nothing beyond the cut: off
    assert c.build_pool(start, 512).tobytes() == _demo(new)[0].build_pool(start, 512).tobytes()
    far = _line(50, step=-10.0)                       # stand-in rows past the cut (x <= 0)
    c.beyond = far
    p = c.build_pool(start, 512)
    n_fresh = max(1, int(round(512 * 0.10)))
    n_beyond = int(round((512 - n_fresh) * 0.25))
    assert len(p) == 512
    tail = p[-n_beyond:]
    assert np.all(tail["origin"][:, 0] <= 0.0) and len(np.unique(tail["origin"][:, 0])) > 10
    body = p[n_fresh:-n_beyond]
    assert np.all(body["origin"][:, 0] >= 0.0)       # the spine draws stay on the spine


@needs_pb
def test_the_beyond_share_follows_the_cut(tmp_path):
    k = ROOT / "autoresearch" / "spines" / "pbr7b2_k04c.npy"
    if not k.exists():
        pytest.skip("no k04c spine")
    line = tmp_path / "line.npy"
    np.save(line, np.load(k)[:1000])
    run = "aspc_beyond"
    try:
        r = _run_demo(run, LOOP + ["--auto-spine-cut-secs", "0.05", "--auto-spine-cut-beyond",
                                   "0.2", "--steps", "4096", "--demo-file", str(line)], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-spine-cut-beyond 0.2: that share of the curriculum's spine draws" in out
        assert "the launch spine CUT at row" in out, out[-6000:]
        assert "--auto-spine-cut-beyond: 20% of the spine draws over the" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine_cut_beyond"] == 0.2
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("aspc_bad3", LOOP + ["--auto-spine-cut-beyond", "0.2"], tmp_path)
    assert bad.returncode != 0 and "give it" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "aspc_bad3", ignore_errors=True)
