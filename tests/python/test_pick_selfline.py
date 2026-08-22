"""pick_selfline.py - trimming a NON-finishing recording into a reference line.

Round 19's xARC and xAUTO both paid arc length along a line taken from a
recording of a FINISHER. The follow-up (xSELF) builds the line from the stuck
checkpoint's OWN best failure, which means the source recording ends in a
fall, and a line that includes that fall PAYS FOR FALLING - measured on the
real recordings, an untrimmed line lets the checkpoint's own fallers reach
99.3% of it while episodes that finish the map reach only 96.9%.

These tests cover the trim: the last tick at which the map pushed back, i.e.
the last tick whose vertical acceleration departs from the constant free-fall
step. CPU-only, no map, no DLL, no GPU, no torch.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.route import ArcProgress, episodes_from_traj  # noqa: E402


def _mod():
    spec = importlib.util.spec_from_file_location(
        "pick_selfline_mod", ROOT / "tools" / "pick_selfline.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


def _ep(n=400, last_contact=300, g=-8.0, bounce_every=50, drift=0.0):
    """Contacts every ``bounce_every`` ticks up to ``last_contact``, then a
    pure ballistic fall that drifts sideways at ``drift`` units per tick."""
    t = np.arange(n, dtype=np.float64)
    vz, v = np.empty(n), 0.0
    for k in range(n):
        vz[k] = v
        v += g
        if k < last_contact and (k + 1) % bounce_every == 0:
            v = 0.0
    z = np.concatenate(([0.0], np.cumsum(vz[:-1])))
    y = drift * np.maximum(t - last_contact, 0.0)
    return np.stack([t, t * 10.0, y, z, np.full(n, 10.0),
                     np.zeros(n), vz, np.zeros(n)], 1)


def test_cut_is_the_last_contact_and_is_kept():
    cut, g = M.contact_cut(_ep())
    assert cut == 300 and abs(g + 8.0) < 1e-12


def test_gravity_is_read_from_the_data_not_a_constant():
    for g in (-8.0, -3.5, -20.0):
        cut, got = M.contact_cut(_ep(g=g))
        assert abs(got - g) < 1e-12 and cut == 300


def test_a_longer_fall_does_not_move_the_cut():
    """The whole point: the tail's LENGTH must not decide where the line ends."""
    assert M.contact_cut(_ep(n=400))[0] == M.contact_cut(_ep(n=2000))[0] == 300


def test_a_ballistic_episode_is_not_trimmed():
    """No contact anywhere -> keep everything. Trimming must fail SAFE."""
    assert M.contact_cut(_ep(last_contact=0))[0] == 399


def test_short_and_degenerate_inputs_do_not_raise():
    for n in (0, 1, 2):
        cut, _g = M.contact_cut(np.zeros((n, 8)))
        assert cut == max(n - 1, -1) or cut == n - 1


def test_tolerance_widens_monotonically():
    e = _ep()
    cuts = [M.contact_cut(e, tol=t)[0] for t in (0.5, 1.0, 4.0, 1e9)]
    assert cuts[0] == cuts[1] == cuts[2] == 300
    assert cuts[3] == len(e) - 1          # everything looks like free fall


def test_ranking_on_the_trimmed_path_inverts_the_raw_one():
    """An episode that leaves the track early and then falls a long way has
    the LONGEST raw path and one of the shortest trimmed ones."""
    good, bad = _ep(n=900, last_contact=300), _ep(n=900, last_contact=100)
    assert M.path_length(bad[:, 1:4]) > M.path_length(good[:, 1:4])
    cg = M.contact_cut(good)[0]
    cb = M.contact_cut(bad)[0]
    assert M.path_length(bad[:cb + 1, 1:4]) < M.path_length(good[:cg + 1, 1:4])


def test_written_episode_round_trips_through_build_routes_reader(tmp_path):
    e = _ep()
    cut = M.contact_cut(e)[0]
    f = M.write_episode(tmp_path / "e.jsonl", e[:cut + 1], note="unit test")
    back = episodes_from_traj(f)
    assert len(back) == 1 and back[0].shape == (cut + 1, 8)
    assert np.allclose(back[0][:, 1:4], e[:cut + 1, 1:4], rtol=1e-5)


def test_trimming_removes_the_pay_for_falling(tmp_path):
    """The property the trim exists for, end to end through ArcProgress.

    A line built from an UNTRIMMED failure ends in the fall, so a faller can
    out-earn an agent that stays on the track. Trimmed, nobody can pass the
    line's end.
    """
    src = _ep(n=700, last_contact=300, drift=6.0)
    cut = M.contact_cut(src)[0]
    for pts, trimmed in ((src[:, 1:4], False), (src[:cut + 1, 1:4], True)):
        line = np.asarray(pts, np.float64)
        arc = ArcProgress(line, spacing=float(np.linalg.norm(line[1] - line[0])),
                          corridor=1500.0, window=16)

        def reach(xyz):
            arc.reset(xyz[:1])
            best = float(arc.arc[0])
            for k in range(1, len(xyz)):
                arc.advance(xyz[k:k + 1])
                best = max(best, float(arc.arc[0]))
            return best

        faller = reach(src[:, 1:4])
        # an agent that keeps flying the track instead of falling
        onward = _ep(n=700, last_contact=650)[:, 1:4]
        stayer = reach(onward)
        if trimmed:
            assert faller <= stayer + 1e-6, (faller, stayer)
        else:
            assert faller > stayer, (faller, stayer)


def test_cli_selftest_passes():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "pick_selfline.py"),
                        "--selftest"], capture_output=True, text=True)
    assert r.returncode == 0 and "selftest OK" in r.stdout, r.stderr


def test_cli_refuses_without_out():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "pick_selfline.py"),
                        "nope.jsonl"], capture_output=True, text=True)
    assert r.returncode != 0 and "--out" in (r.stderr + r.stdout)
