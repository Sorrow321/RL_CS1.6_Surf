"""--int-mode edge (directed-transition novelty), --int-rare and the
survivor-gated predecessor archive (cross-review 2026-09-13, mechanisms 1-2).

(a) edge mode pays only when the POSITION cell changes: turning the view
    or crossing a speed bin inside one cell pays nothing (cell mode with
    yaw sectors pays); A -> B and B -> A are different edges; two envs
    entering the same edge on one tick are ranked (1/sqrt(1), 1/sqrt(2))
    and the table counts both; --int-rare flags entries below the count;
    decay_counts leaves the edge table alone. The novelty is isolated as
    the difference against a twin reward with int_coef 0 driven through
    the same states (the shaping term moves with the position too).
(b) PredecessorArchive: a rare entry freezes the run-up; survival past the
    hold commits it in chronological order, a death inside the hold discards
    it, a timeout after the rare entry commits it; mix_pool replaces the
    requested share; state_dict round-trips.
(c) trainer smoke (CPU, cannonball toy set): --int-mode edge --int-rare
    --archive-frac run, the config carries the keys, progress.csv gains the
    archive columns, the archive holds rows by the end.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.archive import PredecessorArchive                  # noqa: E402
from surfgym.core import STATE_DTYPE                            # noqa: E402
from test_curiosity_cond import FakeCore, _race, _step          # noqa: E402
from test_unstuck import ABS, TRAIN, _csv, _run, _train         # noqa: E402
from test_view_continuous import needs_run                      # noqa: E402

CELL = 256.0
START = (-11000.0, 7487.0 + 3000.0, -1000.0)


class Twin:
    """A novelty reward and an int_coef = 0 twin driven through the same
    states; `step` returns the novelty actually paid per env."""

    def __init__(self, n, **kw):
        kw.setdefault("int_coef", 1.0)
        kw.setdefault("int_cell", CELL)
        self.core = FakeCore(n)
        self.core0 = FakeCore(n)
        for c in (self.core, self.core0):
            for i in range(n):
                c.states_view["origin"][i] = START
        self.rw = _race(**kw)
        kw0 = dict(kw, int_coef=0.0)
        kw0.pop("int_mode", None); kw0.pop("int_rare", None); kw0.pop("int_edge_bits", None)
        self.rw0 = _race(**kw0)
        self.rw.on_reset(self.core)
        self.rw0.on_reset(self.core0)

    def move(self, i, dx=0.0, dy=0.0, yaw=None, speed=None):
        for c in (self.core, self.core0):
            o = c.states_view["origin"][i]
            c.states_view["origin"][i] = (o[0] + dx, o[1] + dy, o[2])
            if yaw is not None:
                c.states_view["yaw"][i] = yaw
            if speed is not None:
                c.states_view["velocity"][i] = (speed, 0.0, 0.0)

    def step(self):
        r = np.asarray(_step(self.rw, self.core), np.float64)
        r0 = np.asarray(_step(self.rw0, self.core0), np.float64)
        return r - r0


# ---------------------------------------------------------------- (a) edge mode
def test_edge_pays_only_on_position_cell_change_and_ignores_view_and_speed():
    t = Twin(1, int_mode="edge", int_view=8, int_speed=3)
    t.step()
    t.move(0, yaw=170.0, speed=3500.0)        # turn + speed bin, same cell
    assert abs(float(t.step()[0])) < 1e-6
    t.move(0, dx=CELL)                         # a new position cell
    assert float(t.step()[0]) == pytest.approx(1.0, abs=1e-5)


def test_cell_mode_still_pays_for_a_view_sector_change_in_place():
    t = Twin(1, int_mode="cell", int_view=8)
    t.step()
    t.move(0, yaw=170.0)
    assert float(t.step()[0]) > 0.5           # the trap the review named


def test_edges_are_directed():
    t = Twin(1, int_mode="edge")
    t.step()
    t.move(0, dx=CELL); r_ab = float(t.step()[0])       # A -> B, first
    t.move(0, dx=-CELL); r_ba = float(t.step()[0])      # B -> A, first
    t.move(0, dx=CELL); r_ab2 = float(t.step()[0])      # A -> B, second
    assert r_ab == pytest.approx(1.0, abs=1e-5)
    assert r_ba == pytest.approx(1.0, abs=1e-5)         # a fresh edge
    assert r_ab2 == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-5)


def test_same_edge_on_one_tick_is_ranked_and_counted_per_env():
    t = Twin(3, int_mode="edge", int_rare=2)
    t.step()
    for i in range(3):
        t.move(i, dx=CELL)                     # all three take the same edge now
    paid = np.sort(t.step())[::-1]
    assert paid == pytest.approx([1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(3.0)], abs=1e-4)
    assert int(t.rw._counts.max()) == 3        # the table saw three visits
    assert int(t.rw.rare_entry.sum()) == 2     # ranks 0 and 1 are below 2, rank 2 is not
    t.step()
    assert int(t.rw.rare_entry.sum()) == 0     # per tick


def test_edge_table_is_never_decayed():
    t = Twin(1, int_mode="edge")
    t.step()
    t.move(0, dx=CELL); t.step()
    before = t.rw._counts.copy()
    assert t.rw.decay_counts(0.5) == int(np.count_nonzero(before))
    assert np.array_equal(t.rw._counts, before)


# ------------------------------------------------------------- (b) the archive
def _st(n, x):
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = x
    return s


def test_archive_commits_the_run_up_on_survival_and_discards_on_death():
    n = 2
    ar = PredecessorArchive(n, window_ticks=8, hold_ticks=4, capacity=100, snap_every=2)
    z = np.zeros(n, bool)
    # ticks 0..7: both envs advance one unit per tick (snapshots every 2 ticks)
    for t in range(8):
        ar.observe(_st(n, float(t)), z, z, z)
    rare = np.array([True, True])
    ar.observe(_st(n, 8.0), rare, z, z)        # both take a rare edge at tick 8
    assert len(ar.pending) == 2 and ar.pending[0]["origin"][:, 0].tolist() == [0.0, 2.0, 4.0, 6.0]
    # env 1 dies two ticks later (inside the hold), env 0 survives the hold
    ar.observe(_st(n, 9.0), z, z, z)
    ended = np.array([False, True]); died = np.array([False, True])
    ar.observe(_st(n, 10.0), z, ended, died)
    st = ar.pop_stats()
    assert st["discards"] == 1 and st["commits"] == 0 and st["rare"] == 2
    for t in range(11, 14):
        ar.observe(_st(n, float(t)), z, z, z)
    st = ar.pop_stats()
    assert st["commits"] == 1 and ar.size == 4
    assert ar.store[:4]["origin"][:, 0].tolist() == [0.0, 2.0, 4.0, 6.0]


def test_archive_timeout_after_rare_entry_commits_and_pool_mix_replaces_share():
    n = 1
    ar = PredecessorArchive(n, window_ticks=4, hold_ticks=100, capacity=10, snap_every=1)
    z = np.zeros(n, bool)
    for t in range(4):
        ar.observe(_st(n, float(t)), z, z, z)
    ar.observe(_st(n, 4.0), np.array([True]), z, z)
    ar.observe(_st(n, 5.0), z, np.array([True]), np.array([False]))   # timeout, not death
    assert ar.pop_stats()["commits"] == 1 and ar.size == 4
    pool = _st(20, -1.0)
    mixed = ar.mix_pool(pool, 0.25)
    assert int((mixed["origin"][:, 0] >= 0).sum()) == 5
    assert int((pool["origin"][:, 0] >= 0).sum()) == 0            # the input is untouched
    d = ar.state_dict()
    ar2 = PredecessorArchive(n, 4, 100, capacity=10)
    ar2.load_state_dict(d)
    assert ar2.size == 4 and ar2.store[:4]["origin"][:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


# --------------------------------------------------------------- (c) the smoke
@needs_run
def test_edge_archive_smoke_runs_and_logs():
    chk = _run([sys.executable, "-c",
                "import torch; assert not torch.cuda.is_available()"], 300)
    assert chk.returncode == 0, chk.stderr
    run = "cya_edge"
    r = _train(run, ABS + ["--int-coef", "1.0", "--int-mode", "edge", "--int-rare", "4",
                           "--archive-frac", "0.2", "--archive-window", "0.2",
                           "--archive-hold", "0.1", "--ep-ticks", "48"], steps="10240")
    assert "--archive-frac 0.2:" in r.stdout
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["int_mode"] == "edge" and cfg["int_rare"] == 4 and cfg["archive_frac"] == 0.2
    rows = _csv(run)
    head = list(rows[0])
    assert "archive/rows" in head and "archive/commits" in head
    rare = sum(int(x["archive/rare"]) for x in rows)
    assert rare > 0
    ck = __import__("torch").load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert "archive" in ck
    shutil.rmtree(d, ignore_errors=True)
