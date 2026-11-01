"""tools/finish_beams.py roots and tools/finish_pool.py mixing (CPU, no core)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402
import finish_beams as FB  # noqa: E402
import finish_pool as FP  # noqa: E402


def _pool(lengths, step=10.0, gap=1000.0):
    rows = np.zeros(sum(lengths), STATE_DTYPE)
    i, x0 = 0, 0.0
    for n in lengths:
        rows["origin"][i:i + n, 0] = x0 + step * np.arange(n)
        i += n
        x0 += step * n + gap
    rows["tick"] = 7
    rows["stuck_ticks"] = 3
    return rows


def test_episode_starts_and_roots():
    p = _pool([5, 7, 3])
    assert FB.episode_starts(p, 200.0) == [0, 5, 12]
    # offsets in seconds at a 1 s tick: 0 -> every start, 4 -> start + 4 where it fits
    assert FB.plan_roots(p, 200.0, [0.0, 4.0], 1.0) == [0, 5, 12, 4, 9]


def test_mix_fraction_and_clocks(tmp_path):
    base = _pool([100])
    line = _pool([30])
    np.savez(tmp_path / "ok.npz", states=line, finished=np.int32(1), replay_ok=np.int32(1))
    np.savez(tmp_path / "nofin.npz", states=line, finished=np.int32(0), replay_ok=np.int32(1))
    np.savez(tmp_path / "bad.npz", states=line, finished=np.int32(1), replay_ok=np.int32(0))
    paths = sorted(str(p) for p in tmp_path.glob("*.npz"))
    tails = FP.finishing_tails(paths, secs=10.0, tick_s=1.0)
    assert len(tails) == 1 and len(tails[0][1]) == 10
    # the tail is the LAST 10 states of the line
    assert np.array_equal(tails[0][1]["origin"], line["origin"][-10:])
    rows = FP.mix(base, tails, 0.5)
    n_t = len(rows) - len(base)
    assert n_t == 100 and rows.dtype == STATE_DTYPE
    assert (rows["tick"] == 0).all() and (rows["stuck_ticks"] == 0).all()
    assert np.array_equal(rows["origin"][:100], base["origin"])
    # no finished line: the base comes back unchanged
    assert np.array_equal(FP.mix(base, [], 0.5)["origin"], base["origin"])


def test_main_writes_atomically(tmp_path):
    base = _pool([40])
    np.save(tmp_path / "base.npy", base)
    np.savez(tmp_path / "l.npz", states=_pool([20]), finished=np.int32(1),
             replay_ok=np.int32(1))
    out = tmp_path / "sp" / "pool.npy"
    FP.main(["--base", str(tmp_path / "base.npy"), "--lines", str(tmp_path / "*.npz"),
             "--secs", "0.07", "--frac", "0.2", "--tick-ms", "7", "--out", str(out)])
    got = np.load(out)
    assert got.dtype == STATE_DTYPE and len(got) == 40 + 10
    assert not list(out.parent.glob("*.tmp*"))
