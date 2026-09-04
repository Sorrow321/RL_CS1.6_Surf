"""xLOOP selection + the demo curriculum the loop spawns from.

Two things had no test at all and both decide what a loop round trains on:

* ``tools/loop_spine.py`` picks the round's line by MINIMUM GEODESIC d.
  That rule is blind the moment a round finishes the map - every finisher
  has ``min_d == 0``, so ``min()`` returns whichever one the recorder
  happened to write first. ``--pick fastest`` ranks the finishers by their
  tick count, which is the metric CLAUDE.md section 3 names for runs that
  finish, and falls back to the old rule when nothing finished.
* ``surfgym.respawn.DemoCurriculum`` under the flag set both xDEMO50 and
  the loop use (``--demo-window <whole spine> --demo-rate 2.0
  --demo-min-ep 1e9 --respawn-frac 1.0``), whose intended behaviour is
  "every spawn drawn uniformly from the whole spine, forever". Each of
  those three numbers is load-bearing and none was pinned.
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE                       # noqa: E402
from surfgym.respawn import DemoCurriculum                 # noqa: E402
from loop_spine import choose_episode                      # noqa: E402


def _ep(i, ticks, min_d, finished, corridor=0.0):
    return {"ep": i, "ticks": ticks, "min_d": min_d,
            "finished": finished, "corridor": corridor}


# ---------------------------------------------------------------------------
# choose_episode
# ---------------------------------------------------------------------------

def test_deepest_is_the_old_rule_exactly():
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(1, 12))
        rows = [_ep(i, int(rng.integers(100, 9999)),
                    float(rng.choice([0.0, 1.0, 2.0, 3.0]) * 1000.0),
                    bool(rng.integers(0, 2))) for i in range(n)]
        assert (choose_episode(rows, "deepest")
                is min(rows, key=lambda r: r["min_d"]))


def test_fastest_ranks_finishers_by_time_where_min_d_cannot():
    # every finisher has min_d 0: the old rule sees a 9-way tie and takes
    # the first, the new one takes the fastest
    rows = [_ep(0, 9683, 0.0, True), _ep(1, 9735, 0.0, True),
            _ep(2, 9702, 0.0, True), _ep(3, 9649, 0.0, True),
            _ep(4, 9816, 0.0, True)]
    assert choose_episode(rows, "deepest")["ep"] == 0
    assert choose_episode(rows, "fastest")["ep"] == 3


def test_fastest_ignores_a_shorter_non_finisher():
    # a 3.8 s death is the shortest episode in the file and must never win
    rows = [_ep(0, 494, 195_074.0, False), _ep(1, 9612, 0.0, True),
            _ep(2, 9341, 6_177.0, False), _ep(3, 9664, 0.0, True)]
    assert choose_episode(rows, "fastest")["ep"] == 1
    # ... while the old rule agrees here only because the finishers are
    # also the deepest
    assert choose_episode(rows, "deepest")["ep"] == 1


def test_fastest_falls_back_to_deepest_with_no_finisher():
    rows = [_ep(0, 400, 40_000.0, False), _ep(1, 7000, 3_761.0, False),
            _ep(2, 9000, 18_700.0, False)]
    assert choose_episode(rows, "fastest") is choose_episode(rows, "deepest")
    assert choose_episode(rows, "fastest")["ep"] == 1


def test_ties_are_deterministic_and_by_episode_index():
    rows = [_ep(0, 900, 5.0, True), _ep(1, 900, 5.0, True),
            _ep(2, 900, 5.0, True)]
    assert choose_episode(rows, "fastest")["ep"] == 0
    assert choose_episode(rows, "deepest")["ep"] == 0
    rev = list(reversed(rows))
    assert choose_episode(rev, "fastest")["ep"] == 0     # index, not order
    assert choose_episode(rev, "deepest")["ep"] == 0


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        choose_episode([], "deepest")
    with pytest.raises(ValueError):
        choose_episode([_ep(0, 1, 0.0, True)], "shallowest")


# ---------------------------------------------------------------------------
# the spawn distribution the loop actually installs
# ---------------------------------------------------------------------------

def _spine(n=500):
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(n, dtype=np.float32) * 24.31   # ~1 tick
    s["origin"][:, 2] = 1000.0
    s["velocity"][:, 0] = 3000.0
    s["yaw"] = 90.0
    s["onground"] = -1
    s["ducked"][::5] = 1                    # 20%, as the real line is
    return s


def test_loop_flag_set_freezes_the_window_over_the_whole_spine():
    s = _spine()
    # --demo-window <spine_len> --demo-rate 2.0 --demo-min-ep 1e9
    d = DemoCurriculum(s, window=len(s), rate=2.0, min_ep=1e9)
    assert (d._lo(), d.tau) == (0, len(s) - 1)
    for _ in range(50):
        d._move()
        assert (d._lo(), d.tau) == (0, len(s) - 1), "the window moved"
    # a rate of 2.0 can never be reached, so even a 100% finish rate and a
    # reachable min_ep leave tau alone
    d2 = DemoCurriculum(s, window=len(s), rate=2.0, min_ep=1.0)
    d2.note_outcomes(np.arange(len(s)), np.ones(len(s)))
    d2._move()
    assert d2.tau == len(s) - 1
    # and the rule DOES move under the paper's own constant, so the test
    # above is pinning the flag set rather than a broken curriculum
    d3 = DemoCurriculum(s, window=len(s), rate=0.2, min_ep=1.0)
    d3.note_outcomes(np.arange(len(s)), np.ones(len(s)))
    d3._move()
    assert d3.tau == len(s) - 2


def test_respawn_frac_1_gives_a_pool_of_spine_states():
    s = _spine()
    d = DemoCurriculum(s, window=len(s), rate=2.0, min_ep=1e9)
    start = np.zeros(3, STATE_DTYPE)
    start["origin"][:, 1] = -99999.0                 # marks a fresh state
    pool = d.build_pool(start, pool_size=4096, fresh_frac=0.0)
    assert len(pool) == 4096
    fresh = int((pool["origin"][:, 1] == -99999.0).sum())
    assert fresh == 1, fresh          # max(1, 0): one fresh, 4,095 demo
    # every demo entry is a whole state, ducked flag and all
    assert pool.dtype == STATE_DTYPE
    assert (pool["ducked"] != 0).any()
    # ... and they span the spine rather than clustering at the end
    xs = pool["origin"][1:, 0]
    assert xs.min() < s["origin"][10, 0] and xs.max() > s["origin"][-10, 0]


def test_match_reidentifies_every_spine_row():
    # the C reset never reports which pool entry it drew, so the curriculum
    # re-identifies a spawn by its origin rounded to 0.1u. A spine whose
    # rows collide there would silently mis-credit outcomes.
    s = _spine()
    d = DemoCurriculum(s, window=len(s), rate=2.0, min_ep=1e9)
    idx = d.match(np.asarray(s["origin"], np.float64))
    assert np.array_equal(idx, np.arange(len(s)))
    assert d.match(np.array([[1e6, 1e6, 1e6]]))[0] == -1


def test_traj_to_spine_replays_a_recording_exactly():
    """The round-0 spine is REPLAYED out of a trajectory rather than
    reconstructed with defaults; its selftest records a random-action
    episode through the core and replays it back from the .jsonl alone."""
    import traj_to_spine
    if not (ROOT / "maps" / "surf_src_cannonball.bsp").exists():
        pytest.skip("no map in this checkout")
    assert traj_to_spine.selftest() == 0


def test_the_round0_spine_has_no_origin_collisions():
    """The artifact this arm ships, if it is present."""
    p = ROOT / "runs" / "research" / "xLOOP131" / "spine_r0.npy"
    if not p.exists():
        pytest.skip("round-0 spine not built in this checkout")
    s = np.load(p)
    assert s.dtype == STATE_DTYPE
    d = DemoCurriculum(s, window=len(s), rate=2.0, min_ep=1e9)
    assert len(d._key) == len(s), "two spine rows share a 0.1u origin key"
    assert np.array_equal(d.match(np.asarray(s["origin"], np.float64)),
                          np.arange(len(s)))


# ---------------------------------------------------------------------------
# the driver's wiring (XLOOP_FLAGS / XLOOP_SPINE0 / XLOOP_PICK)
# ---------------------------------------------------------------------------

def _driver(monkeypatch, **env):
    """Import tools/loop_driver.py under a throwaway XLOOP_NAME."""
    import importlib
    name = "_pytest_loop"
    for k in ("XLOOP_FLAGS", "XLOOP_SPINE0", "XLOOP_PICK", "XLOOP_EP_TICKS",
              "XLOOP_CKPT_EVERY", "XLOOP_RECORD_EVERY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("XLOOP_NAME", name)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("loop_driver", None)
    mod = importlib.import_module("loop_driver")
    return mod


@pytest.fixture(autouse=True)
def _cleanup_driver_runroot():
    yield
    sys.modules.pop("loop_driver", None)
    shutil.rmtree(ROOT / "runs" / "_pytest_loop", ignore_errors=True)


def test_driver_default_flags_are_unchanged(monkeypatch):
    m = _driver(monkeypatch)
    assert m.round_flags(None, 0) == ["--seed", "1"]
    assert m.round_flags("s.npy", 1234) == [
        "--seed", "1", "--demo-file", "s.npy", "--demo-window", "1234",
        "--demo-rate", "2.0", "--demo-min-ep", "1e9",
        "--respawn-frac", "1.0"]
    assert m.PICK == "deepest" and m.FLAGS == [] and m.SPINE0 == ""


def test_driver_flags_come_last_so_they_win(monkeypatch):
    flags = ("--tick-ms 7.63 --ent 0.001 --ep-secs 120 --respawn-margin 2 "
             "--respawn-binned 1 --respawn-bins 128 --eval-stall 1")
    m = _driver(monkeypatch, XLOOP_FLAGS=flags, XLOOP_PICK="fastest",
                XLOOP_RECORD_EVERY="75e6")
    got = m.round_flags("s.npy", 9612)
    assert got[-len(flags.split()):] == flags.split()
    assert got.index("--record-every") < got.index("--tick-ms")
    assert got.index("--demo-file") < got.index("--tick-ms")
    assert m.PICK == "fastest"


def test_driver_seed_spine_reads_its_own_length(monkeypatch, tmp_path):
    s = _spine(321)
    p = tmp_path / "seed.npy"
    np.save(p, s)
    m = _driver(monkeypatch, XLOOP_SPINE0=str(p))
    assert m.SPINE0 == str(p)
    assert int(len(np.load(m.SPINE0))) == 321
    assert m.round_flags(m.SPINE0, 321)[3] == str(p)


def test_driver_rejects_a_missing_seed_spine(monkeypatch, tmp_path):
    m = _driver(monkeypatch, XLOOP_SPINE0=str(tmp_path / "nope.npy"))
    with pytest.raises(SystemExit):
        m.main()
    assert not os.path.exists(ROOT / "runs" / "_pytest_loop" / "DONE")
