"""ab/pmax (2026-11-14): --auto-spine-patience-max N - a held line gets a patience search N steps
after its adoption (or the last search's start) without a quorum, whatever its walk-back does.

--auto-spine-patience-tau restarts the patience on every new low of tau, so a walk-back that
creeps holds its line indefinitely: unitfarmer2's v5 run ab7c_ud held a trough line 1.98B steps
(0.44 -> 2.42B). The longest adoption-to-quorum gap of a LEARNED line in the finishing runs read
for this (fix round blue200, celestial, petrus, cannonball, sidistic; ab6c, ab7c_1, ab7b_1,
ab4u_pet5, ab4u_b5; live ab8s_1, ab8n_1) is 0.66B, so N = 1e9 interrupts none of them.

(a) the decision with patience_max_due: with N = 0 it is the decision as before on a grid of
    inputs (bit for bit); with N it fires after N steps on the line without a quorum, never with
    one, never before the retry; (b) the trainer: the cap is one term of the decision, the clock
    restarts at an
    adoption, at a search's start and when the greedy finishes, the flag's refusal, the config
    dump, TRAIN_ONLY in the recorder.
"""
from __future__ import annotations

import itertools
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))


def _decide(step, quorum, last, patience, retry_at, t_hold, n):
    """The trainer's decision, as written in train_fast.py with the cap."""
    from train_fast import patience_max_due
    pat = (float(patience) > 0 and step - last >= float(patience))
    pmax = patience_max_due(step, t_hold, n, quorum)
    pat = pat or pmax
    return (quorum or pat) and step >= retry_at, pat, pmax


def test_off_is_the_old_decision():
    for step, quorum, last, patience, retry_at, t_hold in itertools.product(
            (0, 2.9e8, 3e8, 1.2e9, 2.5e9), (False, True), (0, 1e8, 2.2e9), (0.0, 3e8),
            (0, 1e9, 3e9), (None, 0, 4e8)):
        old_pat = float(patience) > 0 and step - last >= float(patience)
        old_go = (quorum or old_pat) and step >= retry_at
        for n in (0.0, None):
            go, pat, pmax = _decide(step, quorum, last, patience, retry_at, t_hold, n)
            assert (go, pat, pmax) == (old_go, old_pat, False)


def test_the_cap():
    # ab7c_ud: line 1 adopted at 0.438B; its walk-back keeps making new lows, so the patience
    # clock (last) is always recent
    held, n = 0.438e9, 1e9
    go, pat, pmax = _decide(1.40e9, False, 1.39e9, 3e8, 0, held, n)
    assert not go
    go, pat, pmax = _decide(1.45e9, False, 1.44e9, 3e8, 0, held, n)
    assert go and pat and pmax                       # 1.012B on the line: a patience search
    # a quorum is a quorum search, never the cap's
    go, pat, pmax = _decide(1.45e9, True, 1.44e9, 3e8, 0, held, n)
    assert go and not pmax
    # not before the retry
    go, pat, pmax = _decide(1.45e9, False, 1.44e9, 3e8, 1.5e9, held, n)
    assert pmax and not go
    # the clock restarts at a search's start: 0.5B later it is quiet again
    go, pat, pmax = _decide(1.95e9, False, 1.94e9, 3e8, 0, 1.45e9, n)
    assert not go and not pmax


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--auto-spine-patience-max"' in src
    assert 'raise SystemExit("--auto-spine-patience-max N is a rule of --auto-spine' in src
    assert 'meta["config"]["auto_spine_patience_max"] = float(args.auto_spine_patience_max)' in src
    # the cap is one term of the decision, after the finish stop, before go
    i_fin = src.index('AUTO["fin"] = _nf >= int(args.auto_spine_quorum)')
    i_pmax = src.index("pmax = patience_max_due(global_step, AUTO[\"t_hold\"],")
    i_or = src.index("pat = pat or pmax", i_pmax)
    i_go = src.index('go = (quorum or pat) and global_step >= AUTO["retry_at"]', i_or)
    assert i_fin < i_pmax < i_or < i_go
    # the clock: an adoption, a search's start, a finishing greedy
    assert src.count('AUTO["t_hold"] = int(global_step)') == 4         # + the first eval's
    i_k = src.index('AUTO["k"] += 1')
    assert src.index('AUTO["t_hold"] = int(global_step)', i_k) < src.index("kdir = adir", i_k)
    import record_ckpt
    assert "auto_spine_patience_max" in record_ckpt.TRAIN_ONLY


try:
    from test_demo_success_d import _run_demo  # noqa: E402
    from test_ge_minutes import needs_pb  # noqa: E402
except Exception:                                      # noqa: BLE001
    _run_demo = None
    needs_pb = pytest.mark.skip(reason="the princessburglar smoke helpers are unavailable")


@needs_pb
def test_refusal(tmp_path):
    run = "aspmax_bad0"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--auto-spine-patience-max", "1e9"],
                      tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode != 0 and "--auto-spine-patience-max N is a rule" in out, out[-3000:]
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
