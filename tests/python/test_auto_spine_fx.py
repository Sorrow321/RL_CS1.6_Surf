"""fx/sib (2026-11-12): the search loop's field-free stage and its bookkeeping.

The frozen recipe's three nulls (princessburglar's pit, blue200's void, unitfarmer2's pit) share
one cause: a stretch against the field, while the policy and the tree search both follow the
field. The pieces, all generic and all off by default:

* auto_spine.py ``--ge-held 1``: the Go-Explore stage keeps the plain archive and marks the cells
  whose state the map holds (explore_phase1 --cell-held, tests/python/test_cell_held.py); its line
  is the chain to the deepest HELD cell and ends there, never in a flight;
* with ``--ge-cell-contact 1`` as well every stage runs once per archive kind, the held one first
  (auto_spine.ge_kinds): a flight longer than one burst needs flight cells as stepping stones, a
  long walk beside a void needs an archive the void's air cannot fill;
* auto_spine.py ``--current-d`` (train_fast ``--auto-spine-current-ref``): a new line is measured
  against the deeper of the greedy frontier and the current line's end; a tree / beam line that
  only beats the frontier is the FALLBACK, handed back when no stage finds a deeper one;
* train_fast ``--auto-spine-args``: the stage's knobs as one string of the run's flag set;
* train_fast ``--auto-spine-patience-tau``: the patience counts from the curriculum's last new low;
* train_fast ``--auto-spine-finish-stop``: no search while the greedy finishes the map.

CPU, princessburglar: the refusals; a smoke run with every piece on whose Go-Explore stage runs
the held-marked archive and then the contact archive, both with the plain cell key;
auto_spine.py's three verdicts on that run's checkpoint
(fallback after a stage, fallback with no stage possible, the rule without --current-d); the
finishing-quorum stop (the trainer's own lines on stand-ins: no smoke episode finishes a map);
an adoption under the new clock; all flags off bit-identical to the trainer before the change.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, _env, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"
GE_ARGS = ("--ge-cell-heading 0 --ge-view-follow none --ge-boost 1.0 --ge-held 1 "
           "--ge-cell-contact 1")
NEW_KEYS = ("auto_spine_args", "auto_spine_current_ref", "auto_spine_patience_tau",
            "auto_spine_finish_stop")


def _loop(ge="0", quorum="1", gain=None, patience="1"):
    out = ["--demo-success-d", "50", "--demo-success-bonus", "50",
           "--auto-spine", "--auto-spine-quorum", quorum,
           "--auto-spine-envs", "32", "--auto-spine-ge-minutes", ge,   # 32: the beam's own floor
           "--auto-spine-extra-secs", "1", "--auto-spine-retry", "1",
           "--record-every", "2048", "--eval-eps", "2", "--ep-ticks", "300"]
    if patience is not None:
        out += ["--auto-spine-patience", patience]
    if gain is not None:
        out += ["--auto-spine-gain", gain]
    return out


FX = ["--auto-spine-args", GE_ARGS, "--auto-spine-current-ref", "--auto-spine-patience-tau",
      "--auto-spine-finish-stop", "--demo-match-band"]


def test_train_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    for k in NEW_KEYS:
        assert k in record_ckpt.TRAIN_ONLY, k


def test_the_trainer_wires_it():
    src = TRAIN.read_text(encoding="utf-8")
    # a finishing quorum and a new low of tau are read before the quorum / patience decide
    i_tau = src.index('AUTO["tau_low"] = int(demo.tau)')
    i_fin = src.index('AUTO["fin"] = _nf >= int(args.auto_spine_quorum)')
    i_go = src.index("go = (quorum or pat) and global_step >= AUTO[\"retry_at\"]")
    assert i_tau < i_fin < i_go
    # a search that ends after a finishing eval is discarded before the curriculum is touched
    i_disc = src.index('if args.auto_spine_finish_stop and AUTO.get("fin"):')
    i_load = src.index('sp = np.load(res["spine"])')
    assert i_go < i_disc < i_load < src.index("demo.reset_spine(sp)", i_load)
    # the trainer's own auto_spine.py flags cannot be repeated through the pass-through
    from train_fast import AUTO_SPINE_OWN_FLAGS
    cmd = src[src.index('cmd = [sys.executable, "-u", str(Path(__file__).resolve().parents[1] '
                        '/ "tools"'):src.index("t0 = time.time()\n        if args.auto_spine_async")]
    assert set(re.findall(r'"(--[a-z-]+)"', cmd)) == AUTO_SPINE_OWN_FLAGS


@needs_pb
def test_refusals(tmp_path):
    for k, (extra, msg) in enumerate((
            (["--auto-spine-args", GE_ARGS], "--auto-spine-args passes flags to --auto-spine"),
            (_loop() + ["--auto-spine-args", "--ge-boost 1.0 --greedy-d 5"],
             "--auto-spine-args: --greedy-d is set by the trainer itself"),
            (_loop() + ["--auto-spine-args", "--ge-boost '1.0"], "--auto-spine-args does not parse"),
            (["--auto-spine-current-ref"], "--auto-spine-current-ref is a rule of --auto-spine"),
            (_loop(patience=None) + ["--auto-spine-patience-tau"],
             "--auto-spine-patience-tau moves --auto-spine-patience's clock"),
            (["--auto-spine-finish-stop"], "--auto-spine-finish-stop is a rule of --auto-spine"))):
        run = f"asfx_bad{k}"
        try:
            r = _run_demo(run, extra, tmp_path)
            assert r.returncode != 0 and msg in (r.stdout + r.stderr), (
                msg, (r.stdout + r.stderr)[-3000:])
        finally:
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _auto(cmd, out_dir, greedy_d, current_d, gain="200"):
    """One auto_spine.py call built from the trainer's own command line: (result, log)."""
    cmd = list(cmd)
    for flag, val in (("--out", str(out_dir)), ("--greedy-d", greedy_d), ("--gain", gain)):
        cmd[cmd.index(flag) + 1] = val
    i = cmd.index("--current-d")
    cmd[i:i + 2] = [] if current_d is None else ["--current-d", current_d]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(ROOT), env=_env(), timeout=1800)
    log = r.stdout + r.stderr
    res = None
    for ln in r.stdout.splitlines()[::-1]:
        if ln.startswith("AUTO_SPINE "):
            res = json.loads(ln[len("AUTO_SPINE "):])
            break
    assert res is not None, log[-6000:]
    return res, log


def _ge_cmds(log):
    return [ln for ln in log.splitlines() if ln.startswith("+ ") and "explore_phase1.py" in ln]


@needs_pb
def test_the_stage_runs_the_held_archive_and_the_fallback_rule(tmp_path):
    run = "asfx_smoke"
    try:
        # a gain no 4 s line reaches: the beam's line fails, the Go-Explore stage gets its turn
        r = _run_demo(run, _loop(ge="0.05", gain="5000") + FX, tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        m = re.search(r"auto-spine search 1: (\S+ -u \S+auto_spine\.py .*)", out)
        assert m, out[-6000:]
        cmd = m.group(1).split()
        assert cmd[-len(GE_ARGS.split()):] == GE_ARGS.split(), cmd
        assert float(cmd[cmd.index("--current-d") + 1]) > 0
        assert "NO new spine" in out, out[-6000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine_args"] == GE_ARGS
        assert [cfg[k] for k in NEW_KEYS[1:]] == [1, 1, 1] and cfg["demo_match_band"] == 1
        log = (ROOT / "runs" / run / "auto" / "k01" / "auto_spine.log").read_text()
        ge = _ge_cmds(log)
        # --ge-boost 1.0 with both archive kinds: the held archive, then the contact one
        assert len(ge) == 2, log[-6000:]
        assert " --cell-held" in ge[0] and "--cell-contact" not in ge[0]
        assert " --cell-contact" in ge[1] and "--cell-held" not in ge[1]
        assert "/ge2 " in ge[1] and "/ge " in ge[0]
        for g in ge:
            assert "--cell-heading" not in g and "--view-follow" not in g
        res = json.loads(log[log.rindex("AUTO_SPINE ") + len("AUTO_SPINE "):].splitlines()[0])
        assert res["spine"] is None and "fallback" not in res
        assert res["ref_d"] == min(res["greedy_d"], res["current_d"])
        assert (res["ge_kind"], res["ge2_kind"]) == ("held", "contact")
        assert "ge_pick" in res and "ge2_pick" not in res      # the contact archive: its deepest

        # the same search on the run's checkpoint, three references
        # (a) the line beats the greedy frontier only; a stage can still pass; none does
        res, log = _auto(cmd, tmp_path / "fb", "1000000", "50000")
        assert res["spine"] and res["fallback"] == 1 and res["source"] == "beam"
        assert res["ref_d"] == 50000.0 and res["end_d"] > 50000.0 - 200.0
        assert "kept as the fallback" in log and "the fallback is handed back" in log
        assert len(_ge_cmds(log)) == 2
        # (b) the current line ends at the goal: no stage can pass, none runs
        res, log = _auto(cmd, tmp_path / "fb0", "1000000", "100")
        assert res["spine"] and res["fallback"] == 1
        assert "no Go-Explore stage: nothing can end below" in log and not _ge_cmds(log)
        # (c) no --current-d: the rule before the change
        res, log = _auto(cmd, tmp_path / "leg", "1000000", None)
        assert res["spine"] and "fallback" not in res and "ref_d" not in res
        assert "current line's end" not in log and not _ge_cmds(log)
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def test_a_finishing_quorum_starts_no_search():
    """No smoke episode finishes a map, so the trainer's own lines are run on stand-ins."""
    src = TRAIN.read_text(encoding="utf-8")
    a = src.index("        if args.auto_spine_finish_stop:\n            _nf = ")
    b = src.index("        quorum = m >= int(args.auto_spine_quorum)", a)
    fn = ("def f(args, AUTO, _eval_finishes, ev_path, eps, med, global_step):\n"
          + textwrap.indent(textwrap.dedent(src[a:b]), "    ") + "    return 'search rule'\n")
    ns = {}
    exec(fn, ns)
    args = SimpleNamespace(auto_spine_finish_stop=True, auto_spine_quorum=5)
    for n_fin, fin in ((0, False), (4, False), (5, True), (9, True)):
        AUTO = {"last": 7}
        got = ns["f"](args, AUTO, lambda p: (n_fin, 44.0), "ev", [0] * 9, 0.0, 1000)
        # a finishing quorum: the latch, the patience restarts, the search rule is not reached
        assert (got is None) == fin and AUTO["fin"] is fin
        assert AUTO["last"] == (1000 if fin else 7)
    # the flag off: the lines do nothing
    AUTO = {"last": 7}
    args.auto_spine_finish_stop = False
    assert ns["f"](args, AUTO, None, "ev", [0] * 9, 0.0, 1000) == "search rule" and AUTO == {"last": 7}


@needs_pb
def test_an_adoption_under_the_new_clock(tmp_path):
    run = "asfx_adopt"
    try:
        # a gain every line clears: search 1's line is adopted, the tau watch restarts on it
        r = _run_demo(run, _loop(gain="-100000") + ["--auto-spine-current-ref",
                                                     "--auto-spine-patience-tau"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "auto-spine search 1: NEW SPINE from beam" in out, out[-6000:]
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--auto-spine-current-ref" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flags_off_is_bit_identical(tmp_path):
    """The blocking loop (a search that fires and ends) and its numbers, against the trainer
    before the change."""
    pre = ROOT / "python" / "_train_fast_pre_asfx.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("asfx_off_new", TRAIN), ("asfx_off_old", pre)):
            r = _run_demo(run, _loop(), tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "asfx_off_new", ROOT / "runs" / "asfx_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and not any(k in ca for k in NEW_KEYS)
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        la = (a / "auto" / "k01" / "auto_spine.log").read_text()
        assert "--current-d" not in la and "current line's end" not in la
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
