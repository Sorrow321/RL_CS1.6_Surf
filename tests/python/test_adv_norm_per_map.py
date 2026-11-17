"""--adv-norm-per-map and the per-map advantage / critic columns (mm/advmap, 2026-11-17).

* the estimator: each map's rows of a minibatch standardised by their own moments (exactly the
  single-map estimator on those rows); a map under ADV_MAP_MIN_ROWS rows keeps the pooled ones;
* the flag needs --maps with two or more maps;
* flag OFF, a --maps run of recipe v14 is byte-identical to dcbfc05a (config dump, every eval
  trajectory, every progress.csv field) except the new train/<name>.<tag> columns, appended LAST;
  the single-map invariants are tests/python/test_mm_identity.py (single == fea35b06 ==
  dcbfc05a, --maps A == --map A), untouched by these columns (they exist only under --maps);
* flag ON, the complete P1 line (petrus_lite, edgeflow_blue200, src_celestial; v14 at CPU
  smoke scale, --obs-normal 0: petrus has no face-id grid on a CPU) runs clean, every map's
  rows leave the estimator at unit std, and record_ckpt's config audit accepts the key.

Maps: MM_P1_MAPS (three .bsp, in that order) and MM_TEST_MAPS (two maps, as test_mm_s1), each
with its cell-32 caches; SURFCORE_DLL. MM_BASE_TREE_DCBF reuses an extracted dcbfc05a python/.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "python"))
import v14_smoke as V  # noqa: E402

P1 = [Path(p) for p in os.environ.get("MM_P1_MAPS", "").split(",") if p]
KINDS5 = ("goal", "goalc", "sdf", "occ", "slabocc")          # no face-id: --obs-normal 0


def _have(maps, kinds=V.KINDS):
    return V.DLL.exists() and bool(maps) and all(
        m.exists() and all(m.with_suffix(f".{k}_32.npz").exists() for k in kinds)
        for m in maps)


needs_two = pytest.mark.skipif(not _have(V.MM_MAPS[:2]), reason="MM_TEST_MAPS: two maps")
needs_p1 = pytest.mark.skipif(len(P1) != 3 or not _have(P1, KINDS5), reason="MM_P1_MAPS")


def test_estimator_is_the_single_map_estimator_per_map():
    import torch
    import train_fast as TF
    g = torch.Generator().manual_seed(3)
    n_env, per, t = 63, 21, 8
    adv = torch.randn(t, n_env, generator=g) * torch.tensor(
        [3.0, 0.03, 0.3]).repeat_interleave(per) + torch.tensor([1.0, -2.0, 0.0]).repeat_interleave(per)
    idx = torch.randperm(t * n_env, generator=g)[:252]
    a, sl = adv.reshape(-1)[idx], (idx % n_env) // per
    out = TF.adv_norm_per_map(a, sl, 3)
    for k in range(3):
        rows = a[sl == k]
        assert len(rows) >= TF.ADV_MAP_MIN_ROWS
        ref = (rows - rows.mean()) / (rows.std() + 1e-8)       # the single-map line, alone
        assert torch.allclose(out[sl == k], ref, atol=1e-5, rtol=1e-5), k
        assert abs(float(out[sl == k].std()) - 1.0) < 1e-5
    # a map under the floor keeps the pooled moments; a map with no row is never indexed
    few = TF.adv_norm_per_map(a, sl, 4, min_rows=10_000)
    assert torch.allclose(few, (a - a.mean()) / (a.std() + 1e-8), atol=1e-6)
    assert out.dtype == a.dtype and torch.isfinite(TF.adv_norm_per_map(a[:5], sl[:5], 3)).all()


@needs_two
def test_the_flag_needs_two_maps():
    r, _d = V.run_line(V.ROOT, "advmap_one", ["--map", str(V.MM_MAPS[0])], steps=2048,
                       extra=["--adv-norm-per-map", "1"])
    assert r.returncode != 0 and "--adv-norm-per-map standardises each MAP" in r.stdout + r.stderr
    shutil.rmtree(V.ROOT / "runs" / "advmap_one", ignore_errors=True)


def _base_tree(dst, ref="dcbfc05a"):
    pre = os.environ.get("MM_BASE_TREE_DCBF")
    if pre:
        return Path(pre)
    dst.mkdir(parents=True, exist_ok=True)
    tar = dst / "base.tar"
    subprocess.run(["git", "-C", str(V.ROOT), "archive", "--format=tar", "-o", str(tar), ref,
                    "python"], check=True, capture_output=True)
    with tarfile.open(tar) as tf:
        tf.extractall(dst)
    return dst


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


@needs_two
def test_flag_off_maps_run_is_dcbfc05a_plus_the_new_columns(tmp_path):
    maps = ["--maps", ",".join(str(m) for m in V.MM_MAPS[:2])]
    mv = ["--maxvel", ",".join("3500" if "unitfarmer2" in m.stem else "4000"
                               for m in V.MM_MAPS[:2])]
    base = _base_tree(tmp_path / "base")
    r0, d0 = V.run_line(base, "advmap_id", maps, extra=mv)
    assert r0.returncode == 0, (r0.stdout + r0.stderr)[-4000:]
    r1, d1 = V.run_line(V.ROOT, "advmap_id", maps, extra=mv)
    assert r1.returncode == 0, (r1.stdout + r1.stderr)[-4000:]
    try:
        ca = json.loads((d0 / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((d1 / "run.json").read_text(encoding="utf-8"))["config"]
        for c in (ca, cb):
            c.pop("demo_file", None)                 # the spine path names the checkout
        assert ca == cb, sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
        ta, tb = sorted(d0.glob("traj_*.jsonl")), sorted(d1.glob("traj_*.jsonl"))
        assert len(ta) >= 4 and [p.name for p in ta] == [p.name for p in tb]
        for x, y in zip(ta, tb):
            assert x.read_bytes() == y.read_bytes(), x.name
        ha, ra = _csv(d0 / "progress.csv")
        hb, rb = _csv(d1 / "progress.csv")
        tags = [s.split(".", 1)[1] for s in ha if s.startswith("unstuck/T.")]
        new = [f"train/{c}.{t}" for t in tags for c in
               ("adv_std", "adv_std_norm", "explained_var", "vf_err", "ret_std", "vloss_share")]
        assert hb == ha + new                         # the new columns, LAST, nothing else
        fps = ha.index("time/fps")
        assert len(ra) == len(rb) >= 5
        for x, y in zip(ra, rb):
            assert x[:fps] + x[fps + 1:] == y[:len(ha)][:fps] + y[:len(ha)][fps + 1:]
            assert all(v != "" for v in y[len(ha):][0::6])          # adv_std is logged
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        if not os.environ.get("MM_BASE_TREE_DCBF"):
            shutil.rmtree(d0, ignore_errors=True)


def _p1(run, flag, iters=12):
    steps = iters * 63 * 32
    return V.run_line(V.ROOT, run, ["--maps", ",".join(str(m) for m in P1)], steps=steps,
                      extra=["--obs-normal", "0", "--envs", "63", "--maxvel", "4000"]
                      + (["--adv-norm-per-map", "1"] if flag else []), timeout=7200)


@needs_p1
def test_flag_on_p1_line_runs_and_every_map_leaves_at_unit_std():
    r, d = _p1("advmap_p1_on", True)
    out = r.stdout + r.stderr
    try:
        assert r.returncode == 0, out[-4000:]
        assert "--adv-norm-per-map: every minibatch standardises each map's advantages" in out
        cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
        assert cfg["adv_norm_per_map"] == 1
        sys.path.insert(0, str(V.ROOT / "tools"))
        import record_ckpt
        record_ckpt.audit_cfg(cfg)                    # the record gate's audit takes the key
        head, rows = _csv(d / "progress.csv")
        cols = [i for i, h in enumerate(head) if h.startswith("train/adv_std_norm.")]
        assert len(cols) == 3 and len(rows) >= 10
        for row in rows:
            for i in cols:
                assert abs(float(row[i]) - 1.0) < 0.02, (head[i], row[i])
    finally:
        shutil.rmtree(d, ignore_errors=True)
