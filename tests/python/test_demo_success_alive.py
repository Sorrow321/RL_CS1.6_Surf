"""--demo-success-alive SECS (pbr6b4, 2026-10-26): a crossing of the spine's success line (with the
energy, under --demo-success-energy) counts for the curriculum's outcome, the bonus and the
auto-spine quorum only once its episode has stayed alive SECS past it, or it finishes. (a) the
training-side gate (alive_gate) and the eval-side quorum rule (crossed_and_alive); (b) a CPU
trainer run on princessburglar prints the rule and dumps the key, the refusals; (c) flag off is
bit-identical to the trainer before the change; (d) record_ckpt knows the flag."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))


def test_alive_gate_pays_a_survived_crossing_and_fails_a_dive():
    from train_fast import alive_gate
    F, T = False, True
    none = np.zeros(4, bool)
    # env 0 crosses on call 1 and survives; env 1 crosses on call 1 and dies on call 2; env 2
    # starts past the line (counts from its spawn, never paid); env 3 never crosses
    st = (None, None, None)
    pay, *st, boot = alive_gate([T, T, F, F], none, [F, F, T, F], *st, 1, 3)
    assert not pay.any() and not boot.any()
    pay, *st, boot = alive_gate(none, [F, T, F, F], [F, F, T, F], *st, 1, 3)
    assert not pay.any() and not boot[1]            # the dive ended before its 3 ticks: a failure
    assert st[0][1] == -1 and not st[1][1]          # re-anchored on a spawn short of the line
    pay, *st, boot = alive_gate(none, none, [F, F, T, F], *st, 1, 3)
    assert not pay.any() and boot.tolist() == [F, F, T, F]   # the spawn past it: ok, unpaid
    pay, *st, boot = alive_gate(none, none, [F, F, T, F], *st, 1, 3)
    assert pay.tolist() == [T, F, F, F] and boot.tolist() == [T, F, T, F]
    pay, *st, boot = alive_gate(none, none, [F, F, T, F], *st, 1, 3)
    assert not pay.any()                            # paid once
    pay, *st, boot = alive_gate(none, [T, F, F, F], [F, F, F, F], *st, 1, 3)
    assert boot[0] and not st[1][0] and st[0][0] == -1   # its outcome, then re-anchored


def test_the_quorum_counts_the_crossings_that_were_survived():
    from train_fast import crossed_and_alive
    d = np.linspace(1000.0, 0.0, 101)               # reaches 500 at row 50
    h = np.full(101, 300.0)
    assert crossed_and_alive(d[:61], h[:61], 500.0, None, 10, 1.0)         # 10 rows past it
    assert not crossed_and_alive(d[:60], h[:60], 500.0, None, 10, 1.0)     # 9: a dive
    assert not crossed_and_alive(d[:40], h[:40], 500.0, None, 10, 1.0)     # never crossed
    assert crossed_and_alive(d, h, 5.0, None, 10, 1.0)                     # ends at the finish
    assert not crossed_and_alive(d[:61], h[:61], 500.0, 301.0, 10, 1.0)    # without the energy
    assert crossed_and_alive(d[:61], h[:61], 500.0, 300.0, 10, 1.0)


def test_the_trainer_runs_the_rule_and_refuses_it_alone(tmp_path):
    from test_demo_success_d import _run_demo
    from test_ge_minutes import PB
    if not PB.exists():
        pytest.skip("no princessburglar")
    run = "dsa_smoke"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--demo-success-alive", "0.5"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--demo-success-alive 0.5: a crossing of the success line counts" in out
        assert "stayed alive 50 ticks past it" in out, out[-6000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_alive"] == 0.5
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    for extra, msg in ((["--demo-success-d", "50", "--demo-success-alive", "0.5"],
                        "give both"),
                       (["--demo-success-d", "50", "--demo-success-bonus", "50",
                         "--demo-success-alive", "-1"], "must be >= 0")):
        bad = _run_demo("dsa_bad", extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr), (
            bad.stdout + bad.stderr)[-3000:]
        shutil.rmtree(ROOT / "runs" / "dsa_bad", ignore_errors=True)


def test_the_loop_counts_survived_crossings_for_its_quorum(tmp_path):
    from test_demo_success_d import _run_demo
    from test_ge_minutes import PB
    if not PB.exists():
        pytest.skip("no princessburglar")
    run = "dsa_loop"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--auto-spine", "--auto-spine-quorum", "9",
                            "--auto-spine-energy", "150", "--auto-spine-ge-minutes", "0",
                            "--demo-success-alive", "0.5", "--record-every", "2048",
                            "--eval-eps", "2", "--ep-ticks", "300", "--steps", "8192"],
                      tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "greedy episodes stay alive 50 ticks past their crossing" in out, out[-6000:]
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def test_flag_off_is_bit_identical_to_the_trainer_before_the_change(tmp_path):
    import subprocess
    import torch
    from test_demo_success_d import TRAIN, _run_demo
    from test_ge_minutes import PB, _csv
    if not PB.exists():
        pytest.skip("no princessburglar")
    pre = ROOT / "python" / "_train_fast_pre_dsa.py"
    ref = None
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=50", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for c in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{c}:python/train_fast.py"], capture_output=True,
                           cwd=str(ROOT))
        if s.returncode == 0 and b"--demo-success-alive" not in s.stdout:
            pre.write_bytes(s.stdout)
            ref = c
            break
    if ref is None:
        pytest.skip("no pre-change trainer in the first-parent history")
    flags = ["--demo-success-d", "50", "--demo-success-bonus", "50", "--record-every", "2048"]
    try:
        for run, script in (("dsa_off_new", TRAIN), ("dsa_off_old", pre)):
            rr = _run_demo(run, flags, tmp_path, script=script)
            assert rr.returncode == 0, rr.stdout[-4000:] + rr.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "dsa_off_new", ROOT / "runs" / "dsa_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_success_alive" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)


def test_record_ckpt_knows_the_flag():
    import record_ckpt
    assert "demo_success_alive" in record_ckpt.TRAIN_ONLY
