"""--greedy-tail-frac: spawn a share of training episodes on the policy's OWN greedy line, just
before it fails (2026-10-21, autoresearch r3b2; surfgym/greedy_tail.py).

(a) TailCapture splits env 0's per-tick states into episodes and labels each end like record.py
    (trunc / done by the finish bonus / fail); an episode still open is dropped.
(b) cut_tails keeps only FAILED episodes, the [end - secs, end - skip) window every k ticks, drops
    states under the speed floor and episodes over max_secs.
(c) sidistic CPU smoke: with eval-stall kills (so a fresh policy's eval episodes FAIL) the trainer
    cuts tails at every eval and says so on the eval line, the config carries the keys, a flagless
    resume restores them, --spawn-states together with it is refused, a knob alone is refused.
(d) flag off: the trainer before the flag and after it agree on the config dump and every
    non-timing progress.csv column.
"""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.core import STATE_DTYPE                         # noqa: E402
from surfgym.greedy_tail import TailCapture, cut_tails       # noqa: E402
from test_ez_viewc import SMOKE, TRAIN, _csv, _run, _train, needs_run   # noqa: E402


def _states(n, speed=1500.0, tick0=0):
    st = np.zeros(n, STATE_DTYPE)
    st["tick"] = np.arange(tick0, tick0 + n)
    st["velocity"][:, 0] = speed
    return st


def _feed(cap, st, end):
    """Feed one episode tick by tick; `end` labels the last tick (None = left open)."""
    for k in range(len(st)):
        last = k == len(st) - 1 and end is not None
        done = np.array([last and end in ("fail", "done")])
        trunc = np.array([last and end == "trunc"])
        rew = np.array([50.0 if (last and end == "done") else 0.01])
        cap(k, st[k:k + 1], rew, done, trunc)


def test_capture_splits_and_labels_episodes():
    cap = TailCapture()
    _feed(cap, _states(30), "fail")
    _feed(cap, _states(20), "done")
    _feed(cap, _states(10), "trunc")
    _feed(cap, _states(7), None)            # still open when the recording stops
    assert [e for _, e in cap.episodes] == ["fail", "done", "trunc"]
    assert [len(s) for s, _ in cap.episodes] == [30, 20, 10]


def test_cut_keeps_failed_tails_in_the_window_and_applies_the_filters():
    hz = 100.0
    eps = [(_states(1000), "fail"), (_states(800), "trunc"), (_states(900), "done"),
           (_states(500, speed=100.0), "fail"), (_states(20000), "fail")]
    rows, notes = cut_tails(eps, hz, secs=3.0, skip=0.25, every=5, max_secs=120.0,
                            min_speed=300.0)
    # only episode 0: ticks [700, 975) every 5 -> 55 rows; ep 3 is under the speed floor,
    # ep 4 is longer than max_secs (200 s)
    assert len(rows) == 55
    assert rows["tick"][0] == 700 and rows["tick"][-1] == 970
    assert any("trunc - skipped" in s for s in notes) and any("done - skipped" in s for s in notes)
    assert any("> 120 s - skipped" in s for s in notes)
    none, _ = cut_tails([(_states(50), "trunc")], hz)
    assert len(none) == 0


# --------------------------------------------------------------------------- (c) the smoke
GT = ["--record-every", "2048", "--eval-eps", "3", "--eval-stall", "1", "--stall-secs", "1",
      "--ep-ticks", "1200", "--greedy-tail-frac", "0.2", "--greedy-tail-secs", "0.8",
      "--greedy-tail-min-speed", "0"]


@needs_run
def test_trainer_cuts_tails_at_every_eval_and_resumes():
    run = "gtail_on"
    r = _train(run, GT, steps="8192")
    assert "--greedy-tail-frac 0.2: that share of every training spawn pool" in r.stdout
    assert "  gtail " in r.stdout, r.stdout[-3000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["greedy_tail_frac"] == 0.2 and cfg["greedy_tail_secs"] == 0.8
    assert cfg["greedy_tail_min_speed"] == 0.0 and cfg["greedy_tail_every"] == 5
    run2 = "gtail_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt", str(d / "ckpt_final.pt")]
              + SMOKE + ["--steps", "10240", "--record-every", "2048", "--eval-stall", "1",
                         "--stall-secs", "1", "--ep-ticks", "1200"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "greedy_tail_frac=0.2" in r2.stdout and "  gtail " in r2.stdout
    for dd in (d, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
def test_refusals():
    npy = ROOT / "runs" / "gtail_states.npy"
    npy.parent.mkdir(exist_ok=True)
    np.save(npy, _states(10))
    import os
    env_ok = dict(os.environ, SELF_STATES="1")
    bad = subprocess.run([sys.executable, "-u", str(TRAIN), "--run", "gtail_bad"] + SMOKE
                         + ["--steps", "2048", "--greedy-tail-frac", "0.2", "--spawn-states",
                            str(npy), "--spawn-states-frac", "0.1"],
                         capture_output=True, text=True, cwd=str(ROOT), env=env_ok)
    assert bad.returncode != 0 and "both take the same share" in (bad.stdout + bad.stderr)
    alone = _run([sys.executable, "-u", str(TRAIN), "--run", "gtail_bad2"] + SMOKE
                 + ["--steps", "2048", "--greedy-tail-secs", "2"])
    assert alone.returncode != 0 and "--greedy-tail-secs without --greedy-tail-frac" in \
        (alone.stdout + alone.stderr)
    npy.unlink(missing_ok=True)
    for x in ("gtail_bad", "gtail_bad2"):
        shutil.rmtree(ROOT / "runs" / x, ignore_errors=True)


# ------------------------------------------------ (d) flag-off identity vs the pre-flag trainer
def _pre_flag_trainer() -> Path | None:
    r = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-n", "40", "--",
                        "python/train_fast.py"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    for ref in r.stdout.split():
        s = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if s.returncode == 0 and s.stdout and "--greedy-tail-frac" not in s.stdout:
            p = ROOT / "python" / "_train_fast_pre_greedy_tail.py"
            p.write_text(s.stdout, encoding="utf-8")
            return p
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_pre_flag_trainer():
    old = _pre_flag_trainer()
    if old is None:
        pytest.skip("no pre-flag train_fast.py in the history")
    a, b = "gtail_id_now", "gtail_id_old"
    extra = ["--record-every", "2048", "--eval-eps", "2"]
    try:
        _train(a, extra, steps="6144")
        _train(b, extra, script=old, steps="6144")
        ca = json.loads((ROOT / "runs" / a / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((ROOT / "runs" / b / "run.json").read_text(encoding="utf-8"))["config"]
        assert ca == cb, "the flag-off config dump changed"
        ra, rb = _csv(a), _csv(b)
        assert list(ra[0]) == list(rb[0]) and len(ra) == len(rb) > 0
        for xa, xb in zip(ra, rb):
            for k in xa:
                if not k.startswith("time/"):
                    assert xa[k] == xb[k], (k, xa[k], xb[k])
        ta = sorted((ROOT / "runs" / a).glob("traj_*.jsonl"))
        tb = sorted((ROOT / "runs" / b).glob("traj_*.jsonl"))
        assert [x.name for x in ta] == [x.name for x in tb] and ta
        for x, y in zip(ta, tb):
            assert x.read_bytes() == y.read_bytes(), x.name
    finally:
        old.unlink(missing_ok=True)
        for dd in (a, b):
            shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)
