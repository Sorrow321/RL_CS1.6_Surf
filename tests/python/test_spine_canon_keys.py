"""tools/spine_canon_keys.py: the wish direction survives putting the view along the velocity."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import spine_canon_keys as sck  # noqa: E402


def test_wish_angles():
    assert sck.wish_angle(2, 1) == 0.0          # forward
    assert sck.wish_angle(1, 0) == 90.0         # left strafe (sidemove -400)
    assert sck.wish_angle(1, 2) == -90.0        # right strafe
    assert sck.wish_angle(0, 1) == 180.0        # back
    assert sck.wish_angle(2, 0) == 45.0         # forward + left
    assert np.isnan(sck.wish_angle(1, 1))       # nothing held


def test_translate_keeps_the_wish_direction():
    # forward with the view 89 deg LEFT of the velocity = a left strafe along it
    k = sck.translate(np.array([[2, 1, 0]]), yaw=[189.0], heading=[100.0])
    assert k.tolist() == [[1, 0, 0]]
    # forward + left with the view 46 deg left (round 4's GE14 ramp) -> left strafe
    k = sck.translate(np.array([[2, 0, 1]]), yaw=[243.0], heading=[197.0])
    assert k.tolist() == [[1, 0, 1]]            # duck untouched
    # already canonical: unchanged; nothing held: unchanged
    k = sck.translate(np.array([[1, 2, 0], [1, 1, 0]]), yaw=[30.0, 30.0], heading=[30.0, 30.0])
    assert k.tolist() == [[1, 2, 0], [1, 1, 0]]
    # wrap-around: view 350, heading 10 -> offset -20; forward -> forward
    k = sck.translate(np.array([[2, 1, 0]]), yaw=[350.0], heading=[10.0])
    assert k.tolist() == [[2, 1, 0]]


def test_auto_spine_keys_from_a_decision_table():
    import auto_spine
    acts = np.array([[7, 3, 2, 0, 1, 0], [7, 3, 1, 2, 0, 1]])     # two decisions of k = 2 ticks
    k = auto_spine.keys_from_acts(acts, 2, 5)
    # state 0 neutral; states 1-2 under decision 0; states 3-4 under decision 1
    assert k.tolist() == [[1, 1, 0], [2, 0, 0], [2, 0, 0], [1, 2, 1], [1, 2, 1]]


def test_auto_spine_chain_keys_follow_the_deepest_chain(tmp_path):
    import auto_spine
    sys.path.insert(0, str(ROOT / "python"))
    from surfgym.core import STATE_DTYPE
    st = np.zeros(3, STATE_DTYPE)
    st["origin"][:, 0] = [300.0, 200.0, 100.0]          # cell 2 is the deepest (d = x)
    acts = np.array([[0, 3, 2, 0, 1, 0], [0, 3, 2, 0, 1, 0],
                     [0, 3, 1, 2, 0, 1], [0, 3, 1, 2, 0, 1], [0, 3, 0, 1, 0, 0]], np.int8)
    np.savez(tmp_path / "archive.npz", state=st, parent=np.array([-1, 0, 1]),
             link_acts=acts, link_off=np.array([0, 0, 2, 5]))

    class F:
        reach_max, cell = 1e9, 32.0

        def sample(self, o):
            return np.asarray(o, np.float64)[:, 0]
    k = auto_spine.chain_keys(tmp_path / "archive.npz", F(), 6)   # root + 5 ticks
    assert k.tolist() == [[2, 0, 0], [2, 0, 0], [1, 2, 1], [1, 2, 1], [0, 1, 0]]
    assert auto_spine.chain_keys(tmp_path / "archive.npz", F(), 4) is None   # length mismatch
