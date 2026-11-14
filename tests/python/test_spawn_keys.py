"""ab/sil (2026-11-14): record_ckpt --spawn-keys / band_bench '+k' - a bench spawn holds the keys
its line held at that row.

The trainer's spine spawns start NEUTRAL unless --demo-keys is given, and the loop's own spines
never carry keys in the recipe (DEMO_KEYS stays None under --auto-spine-scratch). The unitfarmer2
exit line holds a strafe on 2,583 of its 2,600 rows, so a neutral spawn on it is a state the line
never visited. To measure what that costs, the eval wrapper's held-key reset can now look the
fresh episode's row up in its --spawn-states pool (nearest origin) and hold that row's keys.

(a) the wrapper: a fresh env holds its nearest row's keys; without the attribute nothing changes;
(b) the recorder refuses --spawn-keys without --spawn-states and parses it; band_bench's '+k'
    modes map to the same sampling flags as their plain twins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


class _Core:
    def __init__(self, origins, ticks):
        self.states_view = {"tick": np.asarray(ticks, np.int64),
                            "origin": np.asarray(origins, np.float64),
                            "ducked": np.zeros(len(ticks), np.int32)}


def _wrapper(core, spawn_keys=None):
    import train_fast
    w = object.__new__(train_fast._TorchPolicyBase)
    w.keys, w.core, w._period, w._keys_tick = None, core, 4, None
    if spawn_keys is not None:
        w.spawn_keys = spawn_keys
    return w


def test_a_fresh_spawn_holds_its_rows_keys():
    from surfgym.keyshold import NEUTRAL
    pool_o = np.array([[0.0, 0, 0], [100.0, 0, 0], [200.0, 0, 0]])
    pool_k = np.array([[1, 2, 0], [1, 0, 0], [2, 2, 1]], np.int32)
    # three envs at their first decision, each a few units from one row
    core = _Core([[198.0, 3, 0], [2.0, 0, 1], [103.0, -2, 0]], [0, 0, 0])
    w = _wrapper(core, (pool_o, pool_k))
    w._keys_sync(3)
    assert np.array_equal(w.keys.state, pool_k[[2, 0, 1]])
    # next decision: env 1 respawned (its tick fell back), the others carry on and keep theirs
    w.keys.state[:] = np.asarray(NEUTRAL)
    core.states_view["tick"] = np.array([4, 1, 4])
    core.states_view["origin"] = np.array([[260.0, 0, 0], [99.0, 1, 0], [150.0, 0, 0]])
    w._keys_sync(3)
    assert np.array_equal(w.keys.state[1], pool_k[1])
    assert np.array_equal(w.keys.state[[0, 2]], np.tile(np.asarray(NEUTRAL), (2, 1)))


def test_without_the_attribute_the_spawn_is_neutral_as_before():
    from surfgym.keyshold import NEUTRAL
    core = _Core([[0.0, 0, 0], [100.0, 0, 0]], [0, 0])
    w = _wrapper(core)
    w._keys_sync(2)
    assert np.array_equal(w.keys.state, np.tile(np.asarray(NEUTRAL), (2, 1)))
    assert w._keys_tick is not None


def test_band_bench_plus_k_modes():
    import band_bench
    for m in ("greedy", "stoch", "keys2", "view4", "all2"):
        assert band_bench.mode_flags(m + "+k") == band_bench.mode_flags(m)
    src = (ROOT / "tools" / "band_bench.py").read_text(encoding="utf-8")
    assert '"--spawn-keys", str(kf)' in src


def test_the_recorder_wires_it():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"--spawn-keys"' in src
    assert 'raise SystemExit("--spawn-keys needs --spawn-states and a --keys-hold checkpoint")' \
        in src
    assert "_pol.spawn_keys = (" in src


def test_the_trainer_flag_is_wired():
    """--auto-spine-keys: the first spine starts with neutral rows (so the swaps load a spine's
    keys), a spine without keys falls back to neutral instead of dropping the keys for good,
    config only when on, TRAIN_ONLY in the recorder."""
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--auto-spine-keys"' in src
    assert ('raise SystemExit("--auto-spine-keys sets the held keys of --auto-spine\'s spine "'
            in src)
    assert 'meta["config"]["auto_spine_keys"] = 1' in src
    i_set = src.index("if DEMO_KEYS[0] is None and args.demo_file:")
    i_swap = src.index("if DEMO_KEYS[0] is not None:\n            # the new spine's held keys")
    i_fb = src.index("elif args.auto_spine_keys:", i_swap)
    i_drop = src.index("DEMO_KEYS[0] = None", i_swap)
    assert i_set < i_swap < i_fb < i_drop
    import record_ckpt
    assert "auto_spine_keys" in record_ckpt.TRAIN_ONLY
