"""tools/stitch_spine.py (pbr5b2, 2026-10-25): a spine from alternating segments - the policy's beam
line, an exact Go-Explore chain rooted on the line, the policy's own rollout from the line's end -
with per-row held keys. Synthetic data on a field whose d falls along x."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402


def _field(tmp_path):
    nx, ny, nz, cell = 128, 4, 4, 32.0
    grid = np.zeros((nz, ny, nx), np.float32)
    grid[:] = (10000.0 - (np.arange(nx) + 0.5) * cell)[None, None, :]
    fp = tmp_path / "field.npz"
    np.savez(fp, grid=grid, quant=np.float64(1.0), mins=np.zeros(3), cell=np.float64(cell),
             reach_max=np.float64(20000.0))
    return fp


def _line(x0, x1, n, vx=500.0):
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.linspace(x0, x1, n)
    s["origin"][:, 1:] = 50.0
    s["velocity"][:, 0] = vx
    s["yaw"] = 30.0                     # the explorer's arbitrary view
    return s


def test_stitch_beam_ge_roll(tmp_path):
    import stitch_spine
    fp = _field(tmp_path)
    beam = _line(100.0, 1000.0, 40)
    acts = np.zeros((10, 6), np.int64)
    acts[:, 2], acts[:, 3], acts[:, 5] = 2, 0, 1          # fwd W, side A, duck
    np.savez(tmp_path / "beam.npz", states=beam, acts=acts, act_every=np.int64(4))
    # a GE archive: root = beam state 30, two links of 5 and 4 ticks
    chain = _line(float(beam["origin"][30, 0]), 1600.0, 10, vx=550.0)
    chain[0] = beam[30]
    st = np.zeros(3, STATE_DTYPE)
    st[0], st[1], st[2] = chain[0], chain[5], chain[9]
    la = np.zeros((9, 6), np.int64)
    la[:, 2], la[:, 3], la[:, 5] = 1, 2, 0                # side D, no duck
    np.savez(tmp_path / "arch.npz", state=st, parent=np.array([-1, 0, 1]),
             link_acts=la, link_off=np.array([0, 0, 5, 9]))
    np.save(tmp_path / "chain.npy", chain)
    # the policy's rollout from the chain's end, its keys in the jsonl
    roll = _line(1600.0, 2400.0, 6, vx=600.0)
    roll[0] = chain[-1]
    np.savez(tmp_path / "roll.npz", ep_0000=roll)
    with open(tmp_path / "roll.jsonl", "w") as f:
        f.write(json.dumps({"episode": 0}) + "\n")
        for t in range(6):
            f.write(json.dumps([t, 0, 0, 0, 0, 0, 0, 0, 4 if t % 2 else 0, 0, 0, 0, 0, 0, 2]) + "\n")
        f.write(json.dumps({"end": "fail"}) + "\n")
    out = tmp_path / "spine.npy"
    rc = stitch_spine.main(["--out", str(out), "--field", str(fp),
                            f"beam:{tmp_path / 'beam.npz'}",
                            f"ge:{tmp_path / 'arch.npz'}:{tmp_path / 'chain.npy'}",
                            f"roll:{tmp_path / 'roll.npz'}:{tmp_path / 'roll.jsonl'}"])
    assert rc == 0
    sp = np.load(out)
    keys = np.load(tmp_path / "spine.keys.npy")
    assert len(sp) == 31 + 9 + 5 == len(keys)
    assert np.allclose(sp["origin"][:31], beam["origin"][:31])
    assert np.allclose(sp["origin"][31:40], chain["origin"][1:])
    assert np.allclose(sp["origin"][40:], roll["origin"][1:])
    # beam keys from its action table, the chain's link actions (views put along the velocity:
    # heading 0 vs the explorer's 30 deg view -> side D at 30 deg stays the nearest key), the
    # rollout's keys from its jsonl rows (state t+1 under row t)
    assert tuple(keys[1]) == (2, 0, 1) and tuple(keys[0]) == (1, 1, 0)
    assert np.all(sp["yaw"][31:40] == 0.0)
    assert np.all(keys[31:40, 2] == 0)
    assert [tuple(k) for k in keys[40:]] == [(0, 2, 0), (0, 2, 1), (0, 2, 0), (0, 2, 1), (0, 2, 0)]
    info = json.loads((tmp_path / "spine.json").read_text())
    assert [s["seg"] for s in info["segments"]] == ["beam", "ge", "roll"]
    assert info["segments"][1]["root"] == 30 and info["segments"][1]["cell"] == 2


def test_refuses_a_chain_off_the_line(tmp_path):
    import stitch_spine
    import pytest
    fp = _field(tmp_path)
    beam = _line(100.0, 1000.0, 40)
    np.savez(tmp_path / "beam.npz", states=beam)
    chain = _line(5000.0, 6000.0, 5)
    np.savez(tmp_path / "arch.npz", state=chain[[0, 4]], parent=np.array([-1, 0]),
             link_acts=np.zeros((4, 6), np.int64), link_off=np.array([0, 0, 4]))
    np.save(tmp_path / "chain.npy", chain)
    with pytest.raises(SystemExit):
        stitch_spine.main(["--out", str(tmp_path / "s.npy"), "--field", str(fp),
                           f"beam:{tmp_path / 'beam.npz'}",
                           f"ge:{tmp_path / 'arch.npz'}:{tmp_path / 'chain.npy'}"])
