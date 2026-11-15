"""--demo-success-bonus-energy-only (ab/bonus-energy, 2026-11-15): the curriculum's success bonus
pays only while the held line is an ENERGY spine.

v12 put the energy pair (--auto-spine-energy 150, which needs --demo-success-bonus 50) on every
map. That bonus pays reaching the end of WHATEVER line the loop holds: from scratch, unitfarmer2
never dove by 1.0B (the trough line outpaid the novelty-driven dive; v7 dove at 378M) and
celestial sat at its fork at 30-31.6% to 1.41B (v5 finished at 529M). With the flag, a depth line
is paid by the field alone, as in v7, and an energy spine (the energy stage's, or one taught with
--demo-success-energy) keeps the bonus its energy rule needs.

(a) bonus_live: the gate both payment sites read;
(b) the trainer: both payment sites read it, the flag is dumped, restored and refused without the
    bonus or a way to hold an energy line, and the recorder lists it TRAIN_ONLY;
(c) princessburglar CPU runs: a loop whose energy stage hands back nothing (no search) decides
    bit-identically to the same loop without the energy pair (progress rows and greedy evals);
    on a success line the smoke's episodes cross, the bare bonus pays, the scoped one does not
    while the line is a depth line, and does once the same line is an energy spine.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402


def test_the_gate():
    from train_fast import bonus_live
    # no bonus: never
    for e in (False, True):
        for h in (None, 3524.0):
            assert not bonus_live(0.0, e, h)
    # the bonus without the flag: always, as before
    assert bonus_live(50.0, False, None) and bonus_live(50.0, False, 3524.0)
    # with the flag: only while an energy floor is armed
    assert not bonus_live(50.0, True, None)
    assert bonus_live(50.0, True, 3524.0)


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--demo-success-bonus-energy-only"' in src
    # both payment sites go through the gate, none pays on the bare bonus any more
    assert src.count("bonus_live(DEMO_BONUS, BONUS_E_ONLY, DEMO_H_MIN)") == 2
    assert "if DEMO_BONUS and r is not None:" not in src
    assert not re.search(r"\n\s*if DEMO_BONUS:\n\s*r_dec = _demo_bonus_pay", src)
    assert 'meta["config"]["demo_success_bonus_energy_only"] = 1' in src
    assert 'ck_cfg.get("demo_success_bonus_energy_only")' in src
    rec = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    block = rec[rec.index("TRAIN_ONLY = frozenset({"):]
    assert '"demo_success_bonus_energy_only"' in block[:block.index("})")]


_BASE = ["--demo-success-d", "50", "--demo-grow", "5", "--demo-front-frac", "0.67",
         "--steps", "8192", "--record-every", "4096", "--eval-eps", "2",
         "--auto-spine", "--auto-spine-delay", "1e12"]


@needs_pb
def test_the_refusals(tmp_path):
    for name, extra, msg in (
            ("dsbe_bad0", ["--demo-success-d", "50", "--demo-success-bonus-energy-only",
                           "--demo-success-energy", "75"], "scopes the --demo-success-bonus"),
            ("dsbe_bad1", ["--demo-success-d", "50", "--demo-success-bonus", "50",
                           "--demo-success-bonus-energy-only"], "pays only an ENERGY line")):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr), name
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


@needs_pb
def test_no_energy_line_decides_as_without_the_pair(tmp_path):
    runs = (("dsbe_v11", _BASE),
            ("dsbe_v13", _BASE + ["--auto-spine-energy", "150", "--demo-success-bonus", "50",
                                  "--demo-success-bonus-energy-only"]))
    try:
        outs = {}
        for run, extra in runs:
            r = _run_demo(run, extra, tmp_path)
            outs[run] = r.stdout + r.stderr
            assert r.returncode == 0, outs[run][-4000:]
        assert "(the starting spine: a depth line, unpaid)" in outs["dsbe_v13"]
        assert "first payment" not in outs["dsbe_v13"]
        a, b = ROOT / "runs" / "dsbe_v11", ROOT / "runs" / "dsbe_v13"
        cb = json.loads((b / "run.json").read_text())["config"]
        assert cb["demo_success_bonus_energy_only"] == 1 and cb["demo_success_bonus"] == 50.0
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y.get(k), k
        ta = sorted(p.name for p in a.glob("traj_*.jsonl"))
        assert ta and ta == sorted(p.name for p in b.glob("traj_*.jsonl"))
        for n in ta:
            assert (a / n).read_bytes() == (b / n).read_bytes(), n
    finally:
        for run, _ in runs:
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


@needs_pb
def test_only_an_energy_spine_is_paid(tmp_path):
    # a success line 50 u past the spine's end, which the smoke's episodes do cross: the bare
    # bonus pays it (the control), the scoped one does not while the line is a depth line, and
    # does once the same line is an energy spine (a 10,000 u margin: any crossing has the energy)
    line = ["--demo-success-d", "-50", "--demo-success-bonus", "50", "--steps", "8192"]
    runs = (("dsbe_pay_bare", line),
            ("dsbe_pay_depth", line + ["--auto-spine", "--auto-spine-delay", "1e12",
                                       "--auto-spine-energy", "150",
                                       "--demo-success-bonus-energy-only"]),
            ("dsbe_pay_energy", line + ["--demo-success-energy", "10000",
                                        "--demo-success-bonus-energy-only"]))
    try:
        outs = {}
        for run, extra in runs:
            r = _run_demo(run, extra, tmp_path)
            outs[run] = r.stdout + r.stderr
            assert r.returncode == 0, outs[run][-4000:]
        assert "--demo-success-bonus: first payment at step" in outs["dsbe_pay_bare"]
        assert "(the starting spine: a depth line, unpaid)" in outs["dsbe_pay_depth"]
        assert "first payment" not in outs["dsbe_pay_depth"]
        assert "(the starting spine: an energy spine, paid)" in outs["dsbe_pay_energy"]
        assert "--demo-success-bonus: first payment at step" in outs["dsbe_pay_energy"]
    finally:
        for run, _ in runs:
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
