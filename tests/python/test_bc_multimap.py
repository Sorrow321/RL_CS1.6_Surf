"""--bc-file under --maps (tools/exit_mm.py, the seven-map expert loop): one planner file per
map, every map an equal share of each BC batch, each block of rows rendered by its OWN map's
lidar. What could be silently wrong: a block rendered with another map's geometry, a share that
moves between calls (the compiled BC step needs fixed shapes), files built for different layouts
mixed into one batch, a map tag that names no map, a given spine's held keys of the wrong shape."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import bc  # noqa: E402
from surfgym.core import STATE_DTYPE  # noqa: E402


def _write(path, n, value, probs=False):
    meta = {"n_latch": 0, "obs_reward": False, "keys_hold": 1}
    st = np.zeros(n, STATE_DTYPE)
    st["origin"][:, 0] = value                     # the pose says which file a row came from
    bc.save_bc_dataset(path, st, np.full((n, 15), float(value), np.float32),
                       np.zeros(n, np.float32), np.zeros((n, 6), np.int64),
                       np.ones(n, np.float32), np.zeros(n, np.int32), meta,
                       keys=np.zeros((n, 7), np.float32),
                       probs=(_onehot(n) if probs else None))


def _onehot(n):
    p = np.zeros((n, bc.NACT, bc.NPAD), np.float32)
    p[:, :, 0] = 1.0                                         # every head's bin 0
    return p


class _Lidar:
    """a map's renderer stand-in: every pixel of its image is the map's id"""

    def __init__(self, ident):
        self.ident = float(ident)

    def render(self, origin, yaw, pitch, ducked):
        import torch
        return torch.full((origin.shape[0], 2, 3), self.ident)


def _ds(path, seed):
    return bc.BCDataset(path, "cpu", n_latch=0, obs_reward=False, keys_hold=True, seed=seed,
                        xlayout={})


def test_every_map_takes_a_fixed_share_rendered_by_its_own_lidar(tmp_path):
    torch = pytest.importorskip("torch")
    _write(tmp_path / "a.npz", 40, 1.0)
    _write(tmp_path / "b.npz", 7, 2.0)                       # a much shorter line
    _write(tmp_path / "c.npz", 13, 3.0)
    m = bc.MultiMapBC([("a", _ds(tmp_path / "a.npz", 1)), ("b", _ds(tmp_path / "b.npz", 2)),
                       ("c", _ds(tmp_path / "c.npz", 3))], [_Lidar(10), _Lidar(20), _Lidar(30)])
    assert m.n == 60 and m.split(2048) == [683, 683, 682] and m.split(5) == [2, 2, 1]
    for n in (2048, 5, 3):
        out = m.sample_all(n)
        assert len(out) == 8 and out[0].shape == (n, 15 + 7)
        sizes = m.split(n)
        scal, pose = out[0][:, 0], out[1]
        img = m.render(None, pose, torch.float32)           # the trainer's lidar is ignored
        assert img.shape == (n, 6)
        o = 0
        for i, s in enumerate(sizes):
            assert torch.all(scal[o:o + s] == float(i + 1))  # block i is map i's rows ...
            assert torch.all(pose[o:o + s, 0] == float(i + 1))
            assert torch.all(img[o:o + s] == 10.0 * (i + 1))  # ... drawn by map i's lidar
            o += s
    assert "[a]" in m.describe() and "[c]" in m.describe()


def test_files_built_for_different_layouts_are_refused(tmp_path):
    pytest.importorskip("torch")
    _write(tmp_path / "a.npz", 4, 1.0, probs=True)
    _write(tmp_path / "b.npz", 4, 2.0, probs=False)
    with pytest.raises(SystemExit):
        bc.MultiMapBC([("a", _ds(tmp_path / "a.npz", 1)), ("b", _ds(tmp_path / "b.npz", 2))],
                      [_Lidar(1), _Lidar(2)])


def test_map_files_name_the_runs_maps(tmp_path):
    for f in ("x.npz", "y.npz"):
        (tmp_path / f).write_bytes(b"0")
    tags = ["sidistic", "celestial", "edgeflow_blue200"]
    x, y = tmp_path / "x.npz", tmp_path / "y.npz"
    assert bc.parse_map_files(f"celestial={x}", tags, "--bc-file") == {1: str(x)}
    # the full stem names its slot too, and spaces around items are fine
    got = bc.parse_map_files(f" surf_edgeflow_blue200={x} , sidistic={y}", tags, "--bc-file")
    assert got == {2: str(x), 0: str(y)}
    for bad in (str(x),                                   # no tag
                f"petrus_lite={x}",                       # not one of the run's maps
                f"celestial={x},celestial={y}",           # twice
                f"celestial={tmp_path / 'missing.npz'}",  # no such file
                ""):
        with pytest.raises(SystemExit):
            bc.parse_map_files(bad, tags, "--bc-file")


def test_a_given_spine_spawns_holding_its_own_keys(tmp_path):
    sp = tmp_path / "spine.npy"
    np.save(sp, np.zeros(5, STATE_DTYPE))
    assert bc.resume_spine_keys(sp, 5) is None               # no keys file: neutral spawns
    k = np.array([[2, 1, 0]] * 5)
    np.save(tmp_path / "spine.keys.npy", k)
    assert bc.resume_spine_keys(sp, 5).tolist() == k.tolist()
    with pytest.raises(SystemExit):
        bc.resume_spine_keys(sp, 6)                          # not the spine's rows
    np.save(tmp_path / "spine.keys.npy", np.array([[3, 1, 0]] * 5))
    with pytest.raises(SystemExit):
        bc.resume_spine_keys(sp, 5)                          # fwd bin 3 does not exist


def test_the_loop_overrides_a_recipe_token_list():
    sys.path.insert(0, str(ROOT / "tools"))
    from exit_mm import set_flag
    rec = ["--envs", "1029", "--demo-grow", "5", "--demo-grow-max", "40",
           "--respawn-greedy-window", "0.25", "2.0", "--auto-spine-args",
           "--ge-cell-heading 0 --mc-envs 4096", "--demo-success-bonus", "50",
           "--demo-success-bonus-energy-only", "--render-graph"]
    t = set_flag(rec, "--demo-grow", 0)
    assert t[-2:] == ["--demo-grow", "0"] and t.count("--demo-grow") == 1
    t = set_flag(t, "--demo-grow-max", None)                 # removed with its value
    assert "--demo-grow-max" not in t and "40" not in t
    t = set_flag(t, "--respawn-greedy-window", 0.5, 1.0)     # both values replaced
    assert t[-3:] == ["--respawn-greedy-window", "0.5", "1.0"] and "2.0" not in t
    t = set_flag(t, "--demo-success-bonus", None)
    t = set_flag(t, "--demo-success-bonus-energy-only", None)  # a bare switch, removed
    assert "--demo-success-bonus" not in t and "--demo-success-bonus-energy-only" not in t
    # a value that is itself a flag string survives untouched, as does a bare switch
    i = t.index("--auto-spine-args")
    assert t[i + 1] == "--ge-cell-heading 0 --mc-envs 4096" and "--render-graph" in t
