"""--respawn-front-perturb F (2026-10-22, autoresearch r6b2): F of the non-start pool rows are the
policy's OWN reservoir states from its frontier band (field progress >= (1 - B) x the reservoir's
deepest), kept in place, heading (velocity + view yaw) turned by U(-T, T) deg and velocity scaled by
U(LO, HI); clocks zeroed; real episodes.

(a) The rows (FrontierPerturbRows on the sidistic field): drawn only from the band, positions kept,
    velocity turned with the yaw and scaled within the range, clocks zeroed; an empty reservoir gives
    none; bad settings are refused.
(b) The trainer (CPU, sidistic, a seeded reservoir): the startup line, the periodic line, the config
    keys and a bare resume; refusals; flag off is bit-identical to the trainer before the change.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.core import STATE_DTYPE                               # noqa: E402
from surfgym.goalfield import GoalField                            # noqa: E402
from surfgym.respawn import FrontierPerturbRows                    # noqa: E402

from test_ez_viewc import SID_FIELD, SMOKE, TRAIN, _run, needs_run    # noqa: E402


class _Res:
    def __init__(self, rows):
        self._store = rows
        self.size = len(rows)


def _field():
    z = np.load(SID_FIELD, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def _reservoir(fld, n=2000, seed=0):
    """n states at reachable voxel centres, random velocities and clocks."""
    rng = np.random.default_rng(seed)
    g = fld.grid
    iz, iy, ix = np.nonzero(g < float(fld.reach_max) - 0.5 * float(fld.cell))
    pick = rng.integers(0, len(ix), n)
    p = np.asarray(fld.mins) + (np.stack([ix[pick], iy[pick], iz[pick]], 1) + 0.5) * fld.cell
    rows = np.zeros(n, STATE_DTYPE)
    rows["origin"] = p.astype(np.float32)
    rows["velocity"] = rng.normal(0.0, 900.0, (n, 3)).astype(np.float32)
    rows["yaw"] = rng.uniform(0.0, 360.0, n).astype(np.float32)
    rows["tick"] = rng.integers(1, 9000, n)
    rows["stuck_ticks"] = 3
    return rows


@needs_run
def test_rows_come_from_the_band_turned_scaled_and_clocks_zeroed():
    fld = _field()
    res = _Res(_reservoir(fld))
    d = np.asarray(fld.sample(res._store["origin"]), np.float64)
    d0 = 97768.0
    fp = FrontierPerturbRows(fld, d0, band=0.2, turn=60.0, speed=(1.2, 1.4), seed=3)
    out = fp.rows(res, 500)
    assert len(out) == 500
    p_max = d0 - d.min()
    d_thr = d0 - 0.8 * p_max
    dout = np.asarray(fld.sample(out["origin"]), np.float64)
    assert np.all(dout <= d_thr + 1e-3) and fp.last["src"] == int((d <= d_thr).sum())
    # each output row is a band row moved in nothing but its velocity and yaw
    key = {tuple(np.round(o, 3)): i for i, o in enumerate(res._store["origin"].astype(np.float64))}
    src = np.array([key[tuple(np.round(o, 3))] for o in out["origin"].astype(np.float64)])
    v0 = res._store["velocity"][src].astype(np.float64)
    v1 = out["velocity"].astype(np.float64)
    r = np.linalg.norm(v1, axis=1) / np.linalg.norm(v0, axis=1)
    assert r.min() >= 1.2 - 1e-4 and r.max() <= 1.4 + 1e-4
    a_v = np.degrees(np.arctan2(v1[:, 1], v1[:, 0]) - np.arctan2(v0[:, 1], v0[:, 0]))
    a_y = out["yaw"].astype(np.float64) - res._store["yaw"][src].astype(np.float64)
    wrap = lambda x: (x + 180.0) % 360.0 - 180.0
    assert np.allclose(wrap(a_v - a_y), 0.0, atol=2e-2)
    assert np.all(np.abs(wrap(a_y)) <= 60.0 + 1e-2)
    assert np.allclose(v1[:, 2] / v0[:, 2], r, rtol=1e-3)
    assert np.all(out["tick"] == 0) and np.all(out["stuck_ticks"] == 0)
    assert len(fp.rows(_Res(np.zeros(0, STATE_DTYPE)), 10)) == 0


@pytest.mark.parametrize("kw,msg", [
    ({"band": 0.0}, "--respawn-front-perturb-band B needs 0 < B < 1"),
    ({"turn": 200.0}, "--respawn-front-perturb-turn T needs 0 <= T <= 180"),
    ({"speed": (1.5, 1.0)}, "--respawn-front-perturb-speed LO HI needs 0 < LO <= HI"),
])
@needs_run
def test_bad_settings_are_refused_by_the_rows(kw, msg):
    with pytest.raises(ValueError, match=re.escape(msg)):
        FrontierPerturbRows(_field(), 97768.0, **kw)


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_mixes_logs_saves_restores_and_refuses():
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("fpert_base")
    run = "fpert_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + ["--steps", "10240", "--respawn-front-perturb", "0.2",
                                       "--respawn-front-perturb-band", "0.3"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    out = r.stdout
    assert "respawn FRONT-PERTURB: 20% of the non-start pool" in out
    m = re.findall(r"front-perturb: ([0-9,]+) rows from ([0-9,]+) frontier-band", out)
    assert m and int(m[0][0].replace(",", "")) == int(round((4096 - 410) * 0.2)), out[-3000:]
    cfg = _cfg(run)
    assert cfg["respawn_front_perturb"] == 0.2 and cfg["respawn_front_perturb_band"] == 0.3
    assert cfg["respawn_front_perturb_turn"] == 90.0
    assert cfg["respawn_front_perturb_speed"] == [1.0, 1.5]
    run2 = "fpert_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "14336"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_front_perturb=0.2" in r2.stdout
    assert _cfg(run2)["respawn_front_perturb_band"] == 0.3
    rec = ROOT / "runs" / run / "rec_greedy.jsonl"
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(rec),
               "--episodes", "1", "--ep-ticks", "300"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", "fpert_bad"] + SMOKE
              + ["--steps", "2048", "--respawn-front-perturb", "1.5"])
    assert r4.returncode != 0 and "--respawn-front-perturb F needs 0 < F < 1" in r4.stdout + r4.stderr
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2, ROOT / "runs" / "fpert_bad"):
        shutil.rmtree(dd, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--respawn-front-perturb" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    from test_respawn_backward_mix import NOVEL_SMOKE, _csv
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    pre = ROOT / "python" / "_train_fast_pre_fpert.py"
    d0, seeded = _seeded_ckpt("fpert_off_base")
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("fpert_off_new", TRAIN), ("fpert_off_old", pre)):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run, "--ckpt", str(seeded)]
                     + SMOKE + GREEDY_SMOKE + NOVEL_SMOKE
                     + ["--steps", "10240", "--respawn-greedy", "0.25"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "fpert_off_new", ROOT / "runs" / "fpert_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "respawn_front_perturb" not in ca
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
