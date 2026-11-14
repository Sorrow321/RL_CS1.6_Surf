"""ab/rh (2026-11-14): --demo-rehearse - a learned line keeps a share of the curriculum when the
search replaces it with a line through its end.

ab4u_1 (unitfarmer2, recipe v1): line 5 taught the pit exit (the greedy out at 3.47-3.85B); search
6, a QUORUM search, handed back line 6 (11,100 rows) which passes line 5's end 27 u away at its
row 1,658; the walk-back restarted at line 6's far end, line 5's rows left the curriculum, and
the greedy was back in the pit from 3.93B to the 7.07B read while the walk-back reached row 2,234.
With the flag, F of the frontier-band draws stay on the replaced line's walked rows until the new
walk-back reaches the junction.

(a) DemoCurriculum: off = the same draws; on = the first band draws come from the rehearsal set,
    they match no row of the new spine, and the rehearsal ends at the junction row; tau_min;
(b) the trainer: the flag, its refusal, the trigger read before the search, the junction measured
    before the swap and the rehearsal set after it, the config dump, TRAIN_ONLY in the recorder.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))


def _spine(n, x0=0.0):
    from surfgym.core import STATE_DTYPE
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = x0 + np.arange(n) * 10.0
    return s


def _cur(spine, seed=3):
    from surfgym.respawn import DemoCurriculum
    return DemoCurriculum(spine, window=5, rate=0.2, grow=5, front_frac=0.5, seed=seed,
                          min_ep=1, cooldown=1)


def test_off_draws_are_unchanged():
    start = _spine(4, x0=-1e5)
    a, b = _cur(_spine(200)), _cur(_spine(200))
    b.set_rehearsal(None, 0.25, -1)
    for _ in range(5):
        pa, pb = a.build_pool(start, 400, 0.1), b.build_pool(start, 400, 0.1)
        assert np.array_equal(pa["origin"], pb["origin"])
    assert np.array_equal(a.draw(64)["origin"], b.draw(64)["origin"])


def test_on_the_first_band_draws_rehearse_the_old_line():
    start = _spine(4, x0=-1e5)
    old = _spine(50, x0=1e5)                 # the replaced line's walked rows
    c = _cur(_spine(200))
    c.set_rehearsal(old, 0.25, 40)
    pool = c.build_pool(start, 400, 0.1)
    n_fresh, n_demo = 40, 360
    n_front = 180
    n_reh = int(round(n_front * 0.25))
    demo_part = pool[n_fresh:]
    xs = demo_part["origin"][:, 0]
    assert np.all(xs[:n_reh] >= 1e5)         # the rehearsal rows lead the band draws
    assert np.all(xs[n_reh:] < 1e5)
    # they never count in the new spine's outcomes
    assert np.all(c.match(demo_part["origin"][:n_reh]) == -1)
    assert np.all(c.match(demo_part["origin"][n_reh:n_reh + 5]) >= 0)
    assert len(c.draw(64)) == 64


def test_the_rehearsal_ends_at_the_junction(capsys):
    start = _spine(4, x0=-1e5)
    c = _cur(_spine(200))
    c.set_rehearsal(_spine(50, x0=1e5), 0.25, 190)
    c.tau = 190                               # the walk-back has reached the junction row
    pool = c.build_pool(start, 400, 0.1)
    assert np.all(pool["origin"][40:, 0] < 1e5)
    assert c.reh is None and "rehearsal over" in capsys.readouterr().out


def test_tau_min_and_reset():
    c = _cur(_spine(200))
    assert c.tau_min == 199
    c.ep[:] = 10.0
    c.win[:] = 10.0                           # every band passes: tau walks back
    for _ in range(4):
        c._move()
    assert c.tau < 199 and c.tau_min == c.tau
    c.set_rehearsal(_spine(10), 0.25, 5)
    c.reset_spine(_spine(300))
    assert c.tau_min == 299 and c.reh is None


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--demo-rehearse"' in src and '"--demo-rehearse-r"' in src
    assert 'raise SystemExit("--demo-rehearse keeps a learned line' in src
    assert 'meta["config"]["demo_rehearse"] = float(args.demo_rehearse)' in src
    i_go = src.index('go = (quorum or pat) and global_step >= AUTO["retry_at"]')
    i_trig = src.index('AUTO["trig"] = "quorum" if quorum else "patience"')
    assert i_go < i_trig
    i_j = src.index('if REHEARSE and AUTO.get("trig") == "quorum" and demo.n > 2:')
    i_reset = src.index("demo.reset_spine(sp)", i_j)
    i_set = src.index("demo.set_rehearsal(", i_reset)
    assert i_j < i_reset < i_set
    import record_ckpt
    assert {"demo_rehearse", "demo_rehearse_r"} <= set(record_ckpt.TRAIN_ONLY)


try:
    from test_demo_success_d import _run_demo  # noqa: E402
    from test_ge_minutes import needs_pb  # noqa: E402
except Exception:                                      # noqa: BLE001
    _run_demo = None
    needs_pb = pytest.mark.skip(reason="the princessburglar smoke helpers are unavailable")


@needs_pb
def test_refusals(tmp_path):
    for k, (extra, msg) in enumerate((
            (["--demo-success-d", "50", "--demo-rehearse", "0.25"],
             "--demo-rehearse keeps a learned line"),
            (["--demo-success-d", "50", "--auto-spine", "--auto-spine-envs", "32",
              "--demo-rehearse", "1.5"], "--demo-rehearse F needs 0 < F < 1"))):
        run = f"asrh_bad{k}"
        try:
            r = _run_demo(run, extra, tmp_path)
            out = r.stdout + r.stderr
            assert r.returncode != 0 and msg in out, (msg, out[-3000:])
        finally:
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
