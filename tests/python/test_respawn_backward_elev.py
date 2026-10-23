"""--respawn-backward-elev LO HI (2026-10-23, autoresearch r7b2): the elevation band the
--respawn-backward / --respawn-backward-mix rows' velocity is clamped to (default -60 30).

The rows fly along the field's own descent; where that descent climbs - out of a pit, up a ramp,
up through a hole in a ceiling - the default +30 makes every row flatter than the way on.

(a) The rows (BackwardSpawnSampler on sidistic at 5% of d0): with (-60, 60) every elevation is
    inside the band and some are steeper than the default's +30; the default stays (-60, 30).
(b) The trainer (CPU, sidistic, a seeded reservoir, --respawn-backward-mix on): the startup line,
    the config key, a bare resume restores it, the refusal; flag off is bit-identical to the
    trainer before the change with --respawn-backward-mix on.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.respawn import BackwardSpawnSampler                    # noqa: E402

from test_ez_viewc import SMOKE, TRAIN, _run, needs_run              # noqa: E402
from test_respawn_backward_mix import NOVEL_SMOKE, _cfg, _core, _csv, _field  # noqa: E402

BMIX = ["--respawn-backward-mix", "0.2", "--respawn-backward-shell", "0"]


def _elev(rows):
    v = rows["velocity"].astype(np.float64)
    return np.degrees(np.arcsin(v[:, 2] / np.linalg.norm(v, axis=1)))


@needs_run
def test_the_rows_respect_the_band_and_a_wider_one_lets_them_climb():
    fld, core = _field(), _core(1)
    d0 = 97768.0
    kw = dict(start_frac=0.05, speed=(1000.0, 2500.0), floor=256.0, shell_frac=0.0, seed=5)
    base = BackwardSpawnSampler(core, fld, d0, **kw).sample_states(400, None)
    wide = BackwardSpawnSampler(core, fld, d0, elev_range=(-60.0, 60.0), **kw).sample_states(400, None)
    eb, ew = _elev(base), _elev(wide)
    assert eb.max() <= 30.0 + 1e-3 and eb.min() >= -60.0 - 1e-3
    assert ew.max() <= 60.0 + 1e-3 and ew.min() >= -60.0 - 1e-3
    # the same positions (the elevation draw comes after them), some rows steeper than +30
    assert np.allclose(base["origin"], wide["origin"])
    assert (ew > 30.0 + 1e-3).sum() >= 5, np.sort(ew)[-10:]


@needs_run
def test_the_trainer_carries_restores_and_refuses_it():
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("belev_base")
    run = "belev_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + NOVEL_SMOKE + BMIX
             + ["--steps", "10240", "--respawn-backward-elev", "-60", "60"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "(elevation clamped to [-60, 60] deg)" in r.stdout, r.stdout[-3000:]
    assert _cfg(run)["respawn_backward_elev"] == [-60.0, 60.0]
    run2 = "belev_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "14336"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_backward_elev=-60-60" in r2.stdout
    assert _cfg(run2)["respawn_backward_elev"] == [-60.0, 60.0]
    rec = ROOT / "runs" / run / "rec_greedy.jsonl"
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(rec),
               "--episodes", "1", "--ep-ticks", "300"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", "belev_bad"] + SMOKE
              + ["--steps", "2048", "--respawn-backward-elev", "40", "30"])
    assert r4.returncode != 0 and "needs -90 <= LO < HI <= 90" in r4.stdout + r4.stderr
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2, ROOT / "runs" / "belev_bad"):
        shutil.rmtree(dd, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--respawn-backward-elev" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    pre = ROOT / "python" / "_train_fast_pre_belev.py"
    d0, seeded = _seeded_ckpt("belev_off_base")
    runs = {"belev_off_new": TRAIN, "belev_off_old": pre}
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in runs.items():
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run, "--ckpt", str(seeded)]
                     + SMOKE + GREEDY_SMOKE + NOVEL_SMOKE + BMIX + ["--steps", "10240"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "belev_off_new", ROOT / "runs" / "belev_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "respawn_backward_elev" not in ca
    ra, rb = _csv(a.name), _csv(b.name)
    assert ra and len(ra) == len(rb) and list(ra[0]) == list(rb[0])
    for x, y in zip(ra, rb):
        for k in x:
            if k != "time/fps":
                assert x[k] == y[k], k
    ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
    assert ta and [p.name for p in ta] == [p.name for p in tb]
    for p, q in zip(ta, tb):
        assert p.read_bytes() == q.read_bytes(), p.name
    sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    for dd in (d0, a, b):
        shutil.rmtree(dd, ignore_errors=True)
