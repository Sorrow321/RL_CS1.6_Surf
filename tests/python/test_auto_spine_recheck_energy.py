"""--auto-spine-recheck with --auto-spine-energy (ab/recheck-energy, 2026-11-15): the recheck must
not discard an ENERGY spine for failing the depth gain.

abegy_1 search 2 (princessburglar, sync, 4,900,519,936): no deeper line existed, so the energy
stage handed back a line keeping +278 u of energy height at 6.76% (the drop surfed at 1,097 u/s
where the greedy lands at 670-940). Its end d 53,284 can never beat the greedy frontier 53,362 by
200 u - an energy spine is no deeper than the greedy by construction - and the trainer printed
"STALE spine discarded". recheck_spine judges such a spine by its own rule: stale only when a quorum
of the latest greedy eval's episodes already cross its success line (end d + --demo-success-d) with
its energy (end h - G/2).

(a) a depth spine's verdict is spine_is_fresh's, so runs without --auto-spine-energy decide the
    same as before;
(b) search 2's case, replayed on that eval's own rows (tests/python/data/recheck_energy_cases.npz,
    9 episodes: every one crosses d <= 53,384 in the air with h 3,306-3,322 < the floor 3,524):
    the depth rule discards it, the energy rule keeps it; with the floor reached by 5 of 9 it is
    stale;
(c) the trainer: the energy rule only for a spine the energy stage marked, the eval rows kept
    only under --auto-spine-energy, the check before the curriculum is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
DATA = ROOT / "tests" / "python" / "data" / "recheck_energy_cases.npz"


def _case():
    z = np.load(DATA, allow_pickle=False)
    dh = [(z[f"d{k}"].astype(np.float64), z[f"h{k}"].astype(np.float64)) for k in range(9)]
    return dh, float(z["spine_end_d"]), float(z["spine_end_h"]), float(z["greedy_med"])


def test_a_depth_spine_is_judged_as_before():
    from train_fast import recheck_spine, spine_is_fresh
    for end_d in (2552.0, 2553.0, 4376.0, 53284.0, 53161.5, 53162.0):
        for med in (None, 2752.0, 53361.5, 1e9):
            for gain in (0.0, 200.0, 500.0):
                assert recheck_spine(end_d, med, gain) == (spine_is_fresh(end_d, med, gain), None)
    # the energy arguments are ignored for a depth spine (energy_rule None)
    dh, _, _, _ = _case()
    assert recheck_spine(53284.0, 53361.5, 200.0, None, dh, 5) == (False, None)


def test_search_2s_energy_spine_is_kept():
    from train_fast import recheck_spine, spine_energy_rule
    dh, end_d, end_h, med = _case()
    assert abs(end_d - 53284.1) < 1.0 and abs(end_h - 3599.4) < 1.0
    # the bug: the depth rule discards it
    fresh, n = recheck_spine(end_d, med, 200.0)
    assert not fresh and n is None
    # the energy rule (the trainer's: success line end d + 100, floor end h - G/2 with G 150)
    st = np.zeros(1, dtype=[("origin", "<f4", (3,)), ("velocity", "<f4", (3,))])
    st["origin"][0] = (0.0, 0.0, end_h)                          # h = z when |v| = 0
    h_min = spine_energy_rule({"energy": True}, st[0], 150.0)
    assert abs(h_min - (end_h - 75.0)) < 1e-3
    fresh, n = recheck_spine(end_d, med, 200.0, (end_d + 100.0, h_min), dh, 5)
    assert fresh and n == 0
    # every episode crosses the line, none with the energy (the greedy lands on the drop's floor)
    for d, h in dh:
        k = np.flatnonzero(d <= end_d + 100.0)
        assert len(k) and h[k[0]] < h_min


def test_an_energy_spine_the_greedy_already_flies_is_stale():
    from train_fast import recheck_spine
    dh, end_d, end_h, med = _case()
    h_min = end_h - 75.0
    lifted = [(d, h + (400.0 if i < 5 else 0.0)) for i, (d, h) in enumerate(dh)]
    assert recheck_spine(end_d, med, 200.0, (end_d + 100.0, h_min), lifted, 5) == (False, 5)
    assert recheck_spine(end_d, med, 200.0, (end_d + 100.0, h_min), lifted[:4] + dh[4:], 5) \
        == (True, 4)
    # no eval read yet: the search's own verdict stands
    assert recheck_spine(end_d, med, 200.0, (end_d + 100.0, h_min), None, 5) == (True, 0)


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    i_load = src.index('sp = np.load(res["spine"])')
    i_rule = src.index('if args.auto_spine_energy is not None and res.get("energy"):', i_load)
    i_chk = src.index("recheck_spine(_de_new", i_load)
    i_reset = src.index("demo.reset_spine(sp)", i_load)
    assert i_load < i_rule < i_chk < i_reset
    # the eval's rows are kept only under --auto-spine-energy (no work, no state otherwise)
    i_keep = src.index('AUTO["eval_dh"] = eval_dh')
    assert src[src.rindex("\n", 0, src.rindex("\n", 0, i_keep)):i_keep].strip() \
        == "if args.auto_spine_energy is not None:"
    assert "eval_dh.append((d," in src
