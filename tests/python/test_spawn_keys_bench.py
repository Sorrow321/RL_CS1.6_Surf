"""record_ckpt --spawn-keys / band_bench --keys (pbr6b1, 2026-10-26): a bench spawn on a keyed
spine row holds that row's keys, as train_fast --demo-keys does in training. CPU, no map: the
greedy wrapper's key sync on a fake core, band_bench's key lookup, record_ckpt's refusal."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


class _Core:
    def __init__(self, tick, origin, ducked):
        self.states_view = {"tick": np.asarray(tick, np.int32),
                            "origin": np.asarray(origin, np.float32),
                            "ducked": np.asarray(ducked, np.int32)}


def _wrapper(core, spawn_keys=None, spawn_duck=False):
    import train_fast
    w = object.__new__(train_fast._TorchPolicyBase)
    w.keys, w._keys_tick, w._period, w.core = None, None, 4, core
    if spawn_keys is not None:
        w.spawn_keys = spawn_keys
    if spawn_duck:
        w.spawn_duck = True
    return w


ROWS = np.array([[0.0, 0.0, 100.0], [64.0, 0.0, 100.0], [128.0, 0.0, 100.0]])
KEYS = np.array([[1, 0, 1], [1, 2, 1], [2, 1, 0]], np.int32)


def test_a_spawn_holds_its_rows_keys():
    core = _Core([0, 0, 0], [ROWS[1], ROWS[2], [999.0, 0.0, 0.0]], [1, 0, 0])
    w = _wrapper(core, (ROWS, KEYS))
    w._keys_sync(3)
    assert w.keys.state.tolist() == [[1, 2, 1], [2, 1, 0], [1, 1, 0]]   # no row: neutral
    # one decision later env 0 is still flying (tick +4: not fresh) and env 1 respawned on row 0
    w.keys.state[0] = (0, 0, 0)
    core.states_view["tick"][:] = (4, 0, 4)
    core.states_view["origin"][1] = ROWS[0]
    w._keys_sync(3)
    assert w.keys.state.tolist() == [[0, 0, 0], [1, 0, 1], [1, 1, 0]]


def test_without_spawn_keys_a_spawn_is_neutral_as_before():
    core = _Core([0, 0], [ROWS[0], ROWS[1]], [1, 1])
    w = _wrapper(core)
    w._keys_sync(2)
    assert w.keys.state.tolist() == [[1, 1, 0], [1, 1, 0]]
    w = _wrapper(_Core([0, 0], [ROWS[0], ROWS[1]], [1, 0]), spawn_duck=True)
    w._keys_sync(2)
    assert w.keys.state.tolist() == [[1, 1, 1], [1, 1, 0]]


def test_band_bench_finds_the_spines_keys(tmp_path):
    import band_bench
    sp = tmp_path / "line.npy"
    np.save(sp, np.zeros(3))
    assert band_bench.spine_keys(str(sp), None, 3) is None            # no keys file: neutral
    np.save(tmp_path / "line.keys.npy", KEYS)
    assert band_bench.spine_keys(str(sp), None, 3).tolist() == KEYS.tolist()
    assert band_bench.spine_keys(str(sp), "none", 3) is None
    with pytest.raises(SystemExit):
        band_bench.spine_keys(str(sp), None, 4)


def test_record_ckpt_refuses_keys_without_states():
    import record_ckpt
    with pytest.raises(SystemExit, match="--spawn-states rows"):
        record_ckpt.main(["no_such.pt", "--spawn-keys", "k.npy"])
