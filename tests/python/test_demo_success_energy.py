"""--demo-success-energy (pbr5b1, 2026-10-25): the spine's success line counts only when the
crossing carries the spine end's energy height (minus a margin). The rule is a pure helper
(train_fast.demo_energy_gate); flag off, the trainer takes its old branch unchanged."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import train_fast as tf  # noqa: E402


def test_energy_height_of_states():
    st = np.zeros(2, dtype=[("origin", "<f4", 3), ("velocity", "<f4", 3)])
    st["origin"][:, 2] = [100.0, 200.0]
    st["velocity"][1] = [0.0, 400.0, 0.0]
    h = tf.energy_height(st, 800.0)
    assert np.allclose(h, [100.0, 200.0 + 400.0 ** 2 / 1600.0])


def test_gate_pays_only_energetic_crossings_once():
    end_d, end_h = 1000.0, 500.0
    rec = np.array([1200.0, 990.0, 990.0, 900.0, 990.0])     # rows 1, 2, 4 cross now
    paid = np.array([False, False, False, True, False])       # row 3 crossed earlier
    ended = np.array([False, False, False, False, True])      # row 4 just ended (respawned)
    h = np.array([900.0, 600.0, 400.0, 900.0, 900.0])
    pay, paid2, ok = tf.demo_energy_gate(rec, paid, ended, end_d, h, end_h)
    assert pay.tolist() == [False, True, False, False, False]   # row 2 is short of the energy
    assert ok.tolist() == pay.tolist()
    # row 2 was judged (paid) at its crossing: a later tick with more energy is not paid
    assert paid2.tolist() == [False, True, True, True, True]
    pay3, paid3, _ = tf.demo_energy_gate(rec, paid2, np.zeros(5, bool), end_d,
                                         np.full(5, 900.0), end_h)
    assert not pay3.any()
    # an ended row is re-marked by where it respawned (past the line = judged)
    assert paid3[4]


def test_gate_matches_the_plain_rule_when_energy_is_ample():
    rng = np.random.default_rng(0)
    rec = rng.uniform(800.0, 1200.0, 64)
    paid = rng.random(64) < 0.3
    ended = rng.random(64) < 0.1
    pay, paid2, _ = tf.demo_energy_gate(rec, paid, ended, 1000.0, np.full(64, 1e9), 0.0)
    hit = (~ended) & (~paid) & (rec <= 1000.0)
    assert pay.tolist() == hit.tolist()
    want = paid | hit
    want[ended] = rec[ended] <= 1000.0
    assert paid2.tolist() == want.tolist()


def test_flag_default_off_and_train_only():
    ap_src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--demo-success-energy", type=float, default=None' in ap_src
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"demo_success_energy"' in rc


def test_a_spawn_past_the_line_holds_it_with_the_energy():
    # a curriculum band at the spine's end starts past the success line: the old rule counts it
    # at once (record <= line); the energy rule does too, if the spawn carries the energy
    ok = tf.demo_energy_spawn_ok([990.0, 990.0, 1010.0], 1000.0, [600.0, 400.0, 900.0], 500.0)
    assert ok.tolist() == [True, False, False]


# ---- trainer smoke (CPU, princessburglar): on = message + key; off = bit-identical ------------
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

import pytest  # noqa: E402

sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


@needs_pb
def test_the_energy_gate_is_on_and_dumped(tmp_path):
    run = "dse_on"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--demo-success-energy", "150"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-success-energy 150: the success line counts only with energy height" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_energy"] == 150.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dse_bad", ["--demo-success-energy", "150"], tmp_path)
    assert bad.returncode != 0 and "--demo-success-energy gates the --demo-success-d line" in (
        bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "dse_bad", ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--demo-success-energy" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_dse.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("dse_off_new", TRAIN), ("dse_off_old", pre)):
            r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50"], tmp_path,
                          script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "dse_off_new", ROOT / "runs" / "dse_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_success_energy" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
