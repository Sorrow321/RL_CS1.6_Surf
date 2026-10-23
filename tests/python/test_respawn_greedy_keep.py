"""--respawn-greedy-keep K (2026-10-23, autoresearch r7b4): the greedy-line pool also keeps the
windows of the K DEEPEST failed greedy episodes seen so far (ranked by each episode's best field
progress), ACROSS evals, drawn beside the last eval's own windows.

(a) The pool: the K deepest episodes survive later, shallower evals and are replaced by deeper
    ones; the kept rows from EARLIER evals sit at the pool's end (n_arch) and an episode of the
    current eval is never added twice; the rank function decides, the windows do not change;
    mix() reports the drawn kept rows; the state round-trips; K = 0 keeps the pool exactly the
    last eval's windows; K < 0 is refused.
(b) The trainer (CPU, sidistic, a seeded reservoir): the startup line, the config key, the
    gkeep/* columns, a bare resume restores K and the kept episodes, the recorder reads the
    checkpoint; refused without --respawn-greedy; flag off is bit-identical to the trainer
    before the change with --respawn-greedy on.
"""
from __future__ import annotations

import csv
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
from surfgym.greedy_spawn import GreedyLinePool, SpawnOutcomeCounter   # noqa: E402

from test_ez_viewc import SMOKE, TRAIN, _run, needs_run              # noqa: E402


def _ep(x_best, n=600, y=0.0):
    """One greedy episode of n ticks flying +x from 0 to x_best (its best progress at its end)."""
    e = np.zeros(n, STATE_DTYPE)
    e["origin"][:, 0] = np.linspace(0.0, x_best, n).astype(np.float32)
    e["origin"][:, 1] = np.float32(y)
    e["velocity"][:, 0] = 100.0
    e["tick"] = np.arange(n)
    return e


def _d(o):
    return 10_000.0 - np.asarray(o, np.float64)[:, 0]


def _build(pool, xs, rank_fn=None, y=0.0):
    pool._par = None
    pool._eps = [_ep(x, y=y) for x in xs]
    pool._cur = []
    return pool.build([{"end": "fail"}] * len(xs), 10.0, d_fn=_d, rank_fn=rank_fn)


def _xs_best(rows):
    """The episodes behind a set of rows, by their window's largest x (rounded)."""
    return sorted({round(float(r), -1) for r in rows["origin"][:, 0]})


def test_the_k_deepest_survive_later_evals_and_deeper_ones_replace_them():
    p = GreedyLinePool(0.25, window=(0.25, 2.0), seed=3, keep=2)
    _build(p, [100.0, 300.0, 200.0])
    assert p.info["fail"] == 3 and p.n_arch == 0          # the kept two are this eval's own
    assert p.size == 3 * 175 and p.info["kept_d"] == [9700.0, 9800.0]
    _build(p, [50.0, 150.0])                               # a shallower eval
    assert p.info["kept_d"] == [9700.0, 9800.0]
    assert p.n_arch == 2 * 175 and p.size == 4 * 175       # B's two + A's kept two, at the end
    tail = p.rows[p.size - p.n_arch:]
    assert tail["origin"][:, 0].max() == pytest.approx(300.0 * (600 - 25 - 1) / 599, abs=1.0)
    assert np.all(p.rows["tick"] == 0) and np.all(p.rows["stuck_ticks"] == 0)
    _build(p, [400.0])                                     # a deeper one replaces the shallower
    assert p.info["kept_d"] == [9600.0, 9700.0]
    assert p.n_arch == 175 and p.size == 2 * 175
    assert "kept 2/2 deepest (best d 9,600..9,700 u)" in p.line()


def test_the_rank_function_decides_which_are_kept():
    p = GreedyLinePool(0.25, window=(0.25, 2.0), seed=3, keep=1)
    # ranked on -x instead: the SHALLOWEST of these episodes wins
    _build(p, [100.0, 300.0], rank_fn=lambda o: 10_000.0 + np.asarray(o, np.float64)[:, 0])
    assert p.info["kept_d"] == [10_000.0]
    q = GreedyLinePool(0.25, window=(0.25, 2.0), seed=3, keep=1)
    _build(q, [100.0, 300.0])
    assert np.array_equal(p.rows, q.rows)                  # the windows are the same


def test_mix_reports_the_kept_rows_it_drew():
    p = GreedyLinePool(0.5, window=(0.25, 2.0), seed=3, keep=1)
    _build(p, [300.0], y=7.0)
    _build(p, [100.0], y=-7.0)
    assert p.n_arch == 175
    out = p.mix(np.zeros(400, STATE_DTYPE), 40)
    tail = out[len(out) - p.mixed:]
    kept = tail["origin"][:, 1] == 7.0
    assert 0 < kept.sum() < p.mixed
    assert len(p.last_arch) == int(kept.sum())
    assert np.all(p.last_arch["origin"][:, 1] == 7.0)


def test_state_round_trip_and_off_is_the_last_eval_only():
    p = GreedyLinePool(0.25, window=(0.25, 2.0), seed=3, keep=2)
    _build(p, [300.0])
    _build(p, [100.0])
    q = GreedyLinePool(0.25, window=(0.25, 2.0), seed=3, keep=2)
    assert q.load_state_dict(p.state_dict())
    assert q.n_arch == p.n_arch and q._n_eval == p._n_eval and np.array_equal(q.rows, p.rows)
    assert [d for d, _, _ in q._arch] == [d for d, _, _ in p._arch]
    _build(q, [50.0])
    assert q.info["kept_d"] == [9700.0, 9900.0]
    off = GreedyLinePool(0.25, window=(0.25, 2.0), seed=3)
    _build(off, [300.0])
    _build(off, [100.0])
    assert off.n_arch == 0 and off.size == 175 and "kept_d" not in off.info
    assert "kept" not in off.state_dict()
    assert off.mix(np.zeros(400, STATE_DTYPE), 40) is not None and len(off.last_arch) == 0
    # a pool restored from a checkpoint saved without the kept episodes starts with none
    r = GreedyLinePool(0.25, window=(0.25, 2.0), seed=3, keep=2)
    assert r.load_state_dict(off.state_dict()) and r.n_arch == 0 and r._arch == []
    with pytest.raises(ValueError, match=re.escape("--respawn-greedy-keep K needs K >= 0")):
        GreedyLinePool(0.25, keep=-1)


class _Field:
    reach_max, cell = 1e9, 32.0

    def sample(self, p):
        return _d(p)


def test_the_outcome_counter_tags_spawns_by_origin():
    c = SpawnOutcomeCounter(_Field())
    rows = np.zeros(2, STATE_DTYPE)
    rows["origin"] = [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
    c.remember(rows)
    st = np.zeros(4, STATE_DTYPE)
    st["origin"] = [[10.0, 0.0, 0.0], [5.0, 0.0, 0.0], [20.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    ended = np.array([1, 1, 1, 0], bool)
    c.note_outcomes(ended, np.zeros(4, bool))
    c.note_spawns(st, ended)
    assert list(c.env) == [True, False, True, False]
    c.observe(np.array([[500.0, 0, 0], [900.0, 0, 0], [700.0, 0, 0], [0.0, 0, 0]]))
    c.note_outcomes(np.array([1, 1, 0, 0], bool), np.array([1, 1, 0, 0], bool))
    e, w, d = c.pop_outcomes()
    assert (e, w, d) == (1, 1, 9300.0)
    assert c.pop_outcomes() == (0, 0, float("inf"))


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_wires_restores_records_and_refuses_it():
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    import torch
    d0, seeded = _seeded_ckpt("gkeep_base")
    run = "gkeep_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + ["--steps", "18432", "--respawn-greedy", "0.25",
                                       "--respawn-greedy-keep", "3"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "the windows of the 3 deepest failed greedy episodes so far, kept across evals" in r.stdout
    assert re.search(r"kept [123]/3 deepest \(best d", r.stdout), r.stdout[-3000:]
    assert re.search(r"  gkeep fin \d+/\d+", r.stdout), r.stdout[-3000:]
    assert _cfg(run)["respawn_greedy_keep"] == 3
    with open(ROOT / "runs" / run / "progress.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][-3:] == ["gkeep/ended", "gkeep/finished", "gkeep/best_prog"]
    assert all(len(x) == len(rows[0]) for x in rows[1:])
    ck = torch.load(ROOT / "runs" / run / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert len(ck["respawn_greedy"]["kept"]) >= 1
    run2 = "gkeep_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "22528"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_greedy_keep=3" in r2.stdout and "restored --respawn-greedy state" in r2.stdout
    assert _cfg(run2)["respawn_greedy_keep"] == 3
    rec = ROOT / "runs" / run / "rec_gkeep.jsonl"
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(rec),
               "--episodes", "1", "--ep-ticks", "400"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", "gkeep_bad"] + SMOKE
              + ["--steps", "2048", "--respawn-greedy-keep", "2"])
    assert r4.returncode != 0 and "it needs --respawn-greedy F > 0" in r4.stdout + r4.stderr
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2, ROOT / "runs" / "gkeep_bad"):
        shutil.rmtree(dd, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--respawn-greedy-keep" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    from test_respawn_backward_mix import _csv
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    pre = ROOT / "python" / "_train_fast_pre_gkeep.py"
    d0, seeded = _seeded_ckpt("gkeep_off_base")
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("gkeep_off_new", TRAIN), ("gkeep_off_old", pre)):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run, "--ckpt", str(seeded)]
                     + SMOKE + GREEDY_SMOKE + ["--steps", "10240", "--respawn-greedy", "0.25"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "gkeep_off_new", ROOT / "runs" / "gkeep_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "respawn_greedy_keep" not in ca
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
