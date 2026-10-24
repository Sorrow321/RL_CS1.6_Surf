"""--respawn-greedy-temp T (pbr2b3, 2026-10-24): an env whose episode started on a greedy-line row
(--respawn-greedy) samples the keys at 1 + T and the yaw sigma x --respawn-greedy-temp-view for its
first --respawn-greedy-temp-ticks ticks (the --front-temp per-row slots).

(a) GreedyLinePool.is_row: a spawn is a greedy row iff its origin is one of the pool's (0.1 u),
    and it follows the pool when the rows are rebuilt;
(b) the trainer (CPU, princessburglar): it says so, the config dumps the keys; it is refused
    without --respawn-greedy. Flag off: tests/python/test_demo_front_temp.py's bit-identity
    tests run the trainer with every new flag absent against the trainer before them.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import needs_pb  # noqa: E402


def test_is_row_follows_the_pool():
    from surfgym.core import STATE_DTYPE
    from surfgym.greedy_spawn import GreedyLinePool
    g = GreedyLinePool(0.2, window=(0.25, 2.0), speed=(1.0, 1.05))
    rows = np.zeros(3, STATE_DTYPE)
    rows["origin"] = [[1.0, 2.0, 3.0], [10.04, 20.0, 30.0], [-5.0, 0.0, 7.5]]
    assert not g.is_row(rows["origin"]).any()          # an empty pool knows no row
    g.rows = rows
    q = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [10.04, 20.0, 30.0], [0, 0, 0]],
                 np.float32)
    assert g.is_row(q).tolist() == [True, True, True, False]   # 10.04 rounds to 10.0
    pool = np.zeros(10, STATE_DTYPE)
    out = g.mix(pool, 2)                                # the mixed rows keep their origins
    assert g.is_row(out["origin"][-g.mixed:]).all()
    g.rows = rows[:1]                                   # rebuilt: the keys follow
    assert g.is_row(q).tolist() == [True, False, False, False]


@needs_pb
def test_on_dumped_and_refused(tmp_path):
    run = "rgt_on"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--respawn-greedy", "0.2",
                            "--respawn-greedy-temp", "1", "--respawn-greedy-temp-ticks", "60",
                            "--respawn-greedy-temp-view", "6"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert ("--respawn-greedy-temp 1: an env whose episode started on a greedy-line row "
                "samples the keys at 2 and the yaw sigma x 6 for its first 60 ticks") in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["respawn_greedy_temp"] == 1.0 and cfg["respawn_greedy_temp_ticks"] == 60
        assert cfg["respawn_greedy_temp_view"] == 6.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("rgt_bad", ["--respawn-greedy-temp", "1"], tmp_path)
    assert bad.returncode != 0 and "it needs --respawn-greedy F > 0" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "rgt_bad", ignore_errors=True)
