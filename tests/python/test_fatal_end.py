"""ab/v3 (2026-11-14): auto_spine.py --end-survive-secs - a line must not end in a fatal flight.

The unitfarmer2 Monte Carlo 'pit exits' (abuf2_all final weights, ledger c9a3f657 and after) end
mid-flight at z 504-612 with the deepest d of 4,096 rollouts, and every one of them touches the
teleport over the pit 0.1-0.4 s later - under the race rule (teleport_fail, as every recipe run
trains) a death. The loop handed such a tip back as a line and the curriculum's success line sat
on it. With the flag, a tree / beam / Monte Carlo line whose end is airborne and whose neutral-key
flight from there dies within S seconds is cut back to its last map contact.

CPU, the real unitfarmer2 map: the probe on a tip state (dies) and on the spawn (lands); the cut
on a synthetic line that rides, launches and flies into that teleport; off = untouched; a held end
is never probed; and auto_spine's decision with the probe stood in: the Monte Carlo tip no longer
becomes the fallback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

UF2 = ROOT / "maps" / "surf_unitfarmer2.bsp"
try:
    from test_ge_minutes import DLL
except Exception:                                      # noqa: BLE001
    DLL = None
needs_uf2 = pytest.mark.skipif(not (UF2.exists() and DLL is not None and Path(DLL).exists()),
                               reason="needs the built core and surf_unitfarmer2")

TICK = 7.666667
TIP = ((-1730.0, -2771.0, 504.0), (-636.0, 807.0, 126.0))     # a Monte Carlo line's last state


def _state(origin, velocity, onground=-1):
    from surfgym.core import STATE_DTYPE
    s = np.zeros(1, STATE_DTYPE)
    s["origin"][0] = origin
    s["velocity"][0] = velocity
    s["onground"][0] = onground
    s["yaw"][0] = float(np.degrees(np.arctan2(velocity[1], velocity[0])))
    return s


def _flying_line(n_ride=40, n_fly=60):
    """A line that rides a surface (vz changes that are not a gravity step), launches and flies
    ballistically into the pit's teleport: its last state is the measured tip. The flight is the
    majority of rows, as in a real line's tail, because contact_cut reads the gravity step as
    the median vz change."""
    from surfgym.core import STATE_DTYPE
    s = np.zeros(n_ride + n_fly, STATE_DTYPE)
    g = 800.0 * TICK / 1000.0
    o, v = np.array(TIP[0]), np.array(TIP[1])
    for i in range(n_ride + n_fly - 1, -1, -1):         # backwards from the tip
        s["origin"][i] = o
        s["velocity"][i] = v
        s["onground"][i] = -1
        if i >= n_ride:
            v = v + np.array([0.0, 0.0, g])               # ballistic: vz one gravity step higher
        else:
            v = v + np.array([0.0, 0.0, (0.3 + 0.5 * (i % 3)) * g])   # riding: the map pushes
        o = o - v * TICK / 1000.0
    return s


@needs_uf2
def test_the_probe_on_the_real_map():
    import auto_spine as A
    assert A.flight_fate(UF2, _state(*TIP), TICK, 3500.0, 1.0) == "dies"
    spawn = _state((-2176.0, -2272.0, 1192.0), (0.0, 0.0, 0.0), onground=0)
    assert A.flight_fate(UF2, spawn, TICK, 3500.0, 1.0) == "lands"


@needs_uf2
def test_the_cut():
    import auto_spine as A
    line = _flying_line()
    c_true = A.contact_cut_states(line)
    assert 35 <= c_true <= 42
    a = SimpleNamespace(end_survive_secs=1.0, contact_tol=1.0, act_every=4, map=str(UF2),
                        tick=TICK, maxvel=3500.0)
    res = {}
    c = A.fatal_end_cut(a, line, len(line) - 1, "mc", res)
    assert c == c_true and res["mc_fatal_end"] == 1 and res["mc_end_flight"] == "dies"
    # off: nothing is probed
    a.end_survive_secs = 0.0
    assert A.fatal_end_cut(a, line, len(line) - 1, "mc", {}) is None
    # a held end (the last row is the last contact) is never probed
    a.end_survive_secs = 1.0
    res = {}
    assert A.fatal_end_cut(a, line, c_true, "mc", res) is None and "mc_end_flight" not in res


def test_the_decision(monkeypatch, tmp_path):
    """abuf2_all search 5's numbers with the Monte Carlo stand-in ending in a fatal flight: cut
    to its last contact (d 31,370) it is no fallback, and the tree's line (30,100) is."""
    import auto_spine as A
    from test_auto_spine_mc import _Field, GE_ARGS
    from surfgym.core import STATE_DTYPE

    def line_to(d_end, n, airborne_tail=0, d_contact=None):
        s = np.zeros(n, STATE_DTYPE)
        s["origin"][:, 0] = np.linspace(d_end + 2000.0, d_end, n)
        if airborne_tail:
            # ride (non-gravity vz changes), then a ballistic tail; the contact end at d_contact
            c = n - airborne_tail
            s["origin"][:c, 0] = np.linspace(d_contact + 500.0, d_contact, c)
            vz = np.zeros(n)
            vz[:c] = np.sin(np.arange(c) / 3.0) * 40.0
            vz[c:] = vz[c - 1] - 6.13 * np.arange(1, airborne_tail + 1)
            s["velocity"][:, 2] = vz
        return s
    monkeypatch.setattr(A, "ckpt_tick", lambda _c: TICK)
    monkeypatch.setattr(A, "load_field", lambda _p: _Field())
    monkeypatch.setattr(A, "beam", lambda a, out, branch_tick=0, tag=None: (
        line_to(30099.9, 300), {"beam_rc": 0, "beam_replay_ok": 1, "beam_finished": 0}))
    monkeypatch.setattr(A, "mc", lambda a, out, tag="mc": (
        line_to(29834.0, 400, airborne_tail=260, d_contact=31370.0),
        {"mc_rc": 0, "mc_replay_ok": 1, "mc_finished": 0}))
    monkeypatch.setattr(A, "go_explore", lambda *a, **k: (None, {"ge_rc": 1}))
    monkeypatch.setattr(A, "flight_fate", lambda *a, **k: "dies")
    argv = ["--ckpt", "x.pt", "--map", "m.bsp", "--field", "f.npz", "--out", str(tmp_path),
            "--greedy-d", "30300.6", "--gain", "200", "--beam-ticks", "6127", "--envs", "2048",
            "--ge-minutes", "5", "--explorer", "mcts", "--current-d", "28936.6",
            "--mc-envs", "4096", "--mc-keys-temp", "1.5", "--end-survive-secs", "1.0"] + GE_ARGS
    A.KEYS.clear()
    assert A.main(argv) == 0
    res = json.loads((tmp_path / "spine.json").read_text())
    assert res["mc_fatal_end"] == 1 and res["mc_end_d"] == pytest.approx(31370.0)
    assert res["source"] == "beam" and res["fallback"] == 1
    assert res["end_d"] == pytest.approx(30099.9)
