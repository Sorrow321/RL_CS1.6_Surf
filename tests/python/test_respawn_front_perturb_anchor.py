"""--respawn-front-perturb-anchor N (2026-10-23, autoresearch r7b4): the front-perturb band hangs off
the reservoir's N-th DEEPEST state instead of its single deepest one (N = 1, the default, is the
shipped rule).

Why: a real finish harvests its whole chain, so one finishing episode puts a state in the finish
funnel; with the band anchored on the single deepest state (d <= d_min + B x the span) every row
then starts in the funnel and finishes trivially, and the practice where the line actually is
stops (sidistic round 6, tp3 at 21.118B: 100% front-perturb finishes for ~25M steps, then 0 for
the next 1.06B). A count anchor needs N such states before the band moves.

(a) The rows: N = 1 draws exactly the shipped rows; with N = k the band threshold follows the k-th
    smallest distance, so a few outlier states do not move it; N < 1 is refused.
(b) The trainer (CPU, sidistic, a seeded reservoir): the config key and a bare resume; flag off
    (N = 1) is bit-identical to the trainer before the change with --respawn-front-perturb on.
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

from surfgym.respawn import FrontierPerturbRows                    # noqa: E402

from test_ez_viewc import SMOKE, TRAIN, _run, needs_run              # noqa: E402
from test_respawn_front_perturb import _Res, _field, _reservoir      # noqa: E402


def _band_rows(fld, lo, hi, n, seed=0):
    """n states at reachable voxel centres whose field distance lies in [lo, hi]."""
    from surfgym.core import STATE_DTYPE
    rng = np.random.default_rng(seed)
    g = fld.grid
    iz, iy, ix = np.nonzero((g >= lo) & (g <= hi) & (g < float(fld.reach_max) - 0.5 * fld.cell))
    pick = rng.integers(0, len(ix), n)
    p = np.asarray(fld.mins) + (np.stack([ix[pick], iy[pick], iz[pick]], 1) + 0.5) * fld.cell
    rows = np.zeros(n, STATE_DTYPE)
    rows["origin"] = p.astype(np.float32)
    rows["velocity"] = rng.normal(0.0, 900.0, (n, 3)).astype(np.float32)
    rows["yaw"] = rng.uniform(0.0, 360.0, n).astype(np.float32)
    return rows


@needs_run
def test_anchor_one_is_the_shipped_rule_and_k_follows_the_kth_deepest():
    fld = _field()
    rows = _reservoir(fld)
    d = np.asarray(fld.sample(rows["origin"]), np.float64)
    d0 = 97768.0
    a = FrontierPerturbRows(fld, d0, band=0.2, seed=3)
    b = FrontierPerturbRows(fld, d0, band=0.2, seed=3, anchor=1)
    ra, rb = a.rows(_Res(rows), 300), b.rows(_Res(rows), 300)
    assert np.array_equal(ra, rb) and a.last == {k: v for k, v in b.last.items() if k in a.last}
    k = 50
    c = FrontierPerturbRows(fld, d0, band=0.2, seed=3, anchor=k)
    c.rows(_Res(rows), 300)
    ds = np.sort(d[d < fld.reach_max - 0.5 * fld.cell])
    assert c.last["d_thr"] == pytest.approx(d0 - 0.8 * (d0 - ds[k - 1]), abs=1e-6)
    # a bulk of states at d 40-60k plus 5 outliers near the goal (a finishing chain's harvest):
    # the shipped band collapses onto the outliers, the anchored one stays on the bulk
    bulk = _band_rows(fld, 40000.0, 60000.0, 2000, seed=4)
    outl = _band_rows(fld, 0.0, 1000.0, 5, seed=5)
    mixed = np.concatenate([bulk, outl])
    a2 = FrontierPerturbRows(fld, d0, band=0.01, seed=3)
    c2 = FrontierPerturbRows(fld, d0, band=0.01, seed=3, anchor=k)
    a2.rows(_Res(mixed), 100)
    c2.rows(_Res(mixed), 100)
    assert a2.last["src"] <= 5 and a2.last["d_thr"] < 2000.0
    assert c2.last["src"] > 5 and c2.last["d_thr"] > 40000.0


@needs_run
def test_a_bad_anchor_is_refused():
    with pytest.raises(ValueError, match=re.escape("--respawn-front-perturb-anchor N needs N >= 1")):
        FrontierPerturbRows(_field(), 97768.0, anchor=0)


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_writes_restores_and_refuses_it():
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("fpanc_base")
    run = "fpanc_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + ["--steps", "10240", "--respawn-front-perturb", "0.2",
                                       "--respawn-front-perturb-band", "0.3",
                                       "--respawn-front-perturb-anchor", "40"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "of the reservoir's 40th deepest state" in r.stdout
    assert re.search(r"front-perturb: ([0-9,]+) rows from ([0-9,]+) frontier-band", r.stdout)
    assert _cfg(run)["respawn_front_perturb_anchor"] == 40
    run2 = "fpanc_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "14336"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_front_perturb_anchor=40" in r2.stdout
    assert _cfg(run2)["respawn_front_perturb_anchor"] == 40
    rec = ROOT / "runs" / run / "rec_fpanc.jsonl"
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(rec),
               "--episodes", "1", "--ep-ticks", "300"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", "fpanc_bad"] + SMOKE
              + ["--steps", "2048", "--respawn-front-perturb-anchor", "5"])
    assert r4.returncode != 0 and "needs --respawn-front-perturb" in r4.stdout + r4.stderr
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2, ROOT / "runs" / "fpanc_bad"):
        shutil.rmtree(dd, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if (s.returncode == 0 and b"--respawn-front-perturb-anchor" not in s.stdout
                and b"--respawn-front-perturb" in s.stdout):
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    from test_respawn_backward_mix import _csv
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    pre = ROOT / "python" / "_train_fast_pre_fpanc.py"
    d0, seeded = _seeded_ckpt("fpanc_off_base")
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("fpanc_off_new", TRAIN), ("fpanc_off_old", pre)):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run, "--ckpt", str(seeded)]
                     + SMOKE + GREEDY_SMOKE
                     + ["--steps", "10240", "--respawn-front-perturb", "0.2",
                        "--respawn-front-perturb-band", "0.3"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "fpanc_off_new", ROOT / "runs" / "fpanc_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "respawn_front_perturb_anchor" not in ca
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
