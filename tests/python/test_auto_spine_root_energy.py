"""--auto-spine-root-energy M (pbr7b3, 2026-10-26): a depth spine the loop's search spliced onto a
Go-Explore chain from a BOOSTED root (x1.08 / x1.12 ladder stages) is held back, and the trainer
first teaches an energy spine - the line up to the chain's first rows, the root boosted as the
chain flew it, success only on arriving there with the chain's energy height - M; on that energy
spine's quorum the held depth spine goes in, without a new search. (princessburglar's gates 15,
17 and 19: a depth curriculum stalled at a x1.08 / x1.12 root until the energy was taught.)
(a) the rule (root_energy_spine, root_boost_of); (b) a CPU trainer run whose search (a stand-in
tools/auto_spine.py in a copy of the tree) returns a x1.12 spine: held, taught for energy, then
released on the quorum; refused without --auto-spine-energy; (c) record_ckpt knows the flag."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))


def _spine(n=60):
    from surfgym.core import STATE_DTYPE
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(n, dtype=np.float32) * 10.0   # x 0 .. 590
    s["origin"][:, 2] = 100.0
    s["velocity"][:, 0] = 400.0                                  # h = 100 + 100 = 200
    return s


def _sample(p):
    """d = 10,000 - x: the spine runs toward the goal."""
    return 10000.0 - np.asarray(p, np.float64)[:, 0]


def test_root_energy_spine_rule():
    from train_fast import root_energy_spine
    sp = _spine()
    sp["velocity"][21:, 0] = 448.0           # the chain flies from row 20 x1.12: h 225.44 ...
    sp["velocity"][30, 0] = 430.0            # ... with one dip on its first rows (h 215.56)
    r = root_energy_spine(sp, 20, 1.12, _sample, 50.0, 10.0)
    # rows up to the root reach d 9,800; the first row past it whose d is 50 u deeper: row 26
    # (d 9,740 + 50 < 9,800), so the line is reached on the chain's own rows only
    assert r["n"] == 27 and r["root"] == 20
    assert r["line"] == pytest.approx(9740.0 + 50.0)
    e = r["spine"]
    assert len(e) == 27 and float(e[20]["velocity"][0]) == pytest.approx(448.0)   # 400 x 1.12
    assert float(sp[20]["velocity"][0]) == pytest.approx(400.0)                   # not mutated
    assert r["h_root"] == pytest.approx(100.0 + 448.0 ** 2 / 1600.0)
    assert r["h_min"] == pytest.approx(100.0 + 448.0 ** 2 / 1600.0 - 10.0)        # dip past c
    r2 = root_energy_spine(sp, 20, 1.12, _sample, 95.0, 0.0)      # c = 30: the dip is inside
    assert r2["n"] == 31 and r2["h_min"] == pytest.approx(100.0 + 430.0 ** 2 / 1600.0)
    assert root_energy_spine(sp, 20, 1.0, _sample, 50.0, 10.0) is None     # not boosted
    assert root_energy_spine(sp, 0, 1.12, _sample, 50.0, 10.0) is None     # no line before it
    assert root_energy_spine(sp, 58, 1.12, _sample, 50.0, 10.0) is None    # no chain after it
    assert root_energy_spine(sp, 20, 1.12, _sample, 5000.0, 10.0) is None  # never 5,000 deeper


def test_root_boost_of():
    from train_fast import root_boost_of
    assert root_boost_of({"source": "beam+ge2", "ge2_root_index": 412,
                          "ge2_root_boost": 1.08}) == (412, 1.08)
    assert root_boost_of({"source": "beam+ge", "ge_root_index": 7, "ge_root_boost": 1.04}) \
        == (7, 1.04)
    assert root_boost_of({"source": "beam"}) is None
    assert root_boost_of({"source": "beam_br2"}) is None
    assert root_boost_of({"source": "beam+ge_energy", "ge_energy_root_index": 3,
                          "ge_energy_root_boost": 1.04, "energy": True}) is None
    assert root_boost_of({"source": "beam+ge3"}) is None       # no root keys
    assert root_boost_of(None) is None


FAKE_SEARCH = r'''
import json, sys
from pathlib import Path
import numpy as np
a = sys.argv
out = Path(a[a.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
sp = np.load(SPINE)
np.save(out / "spine.npy", sp)
print("AUTO_SPINE " + json.dumps({"spine": str(out / "spine.npy"), "source": "beam+ge2",
                                  "ge2_root_index": ROOT_ROW, "ge2_root_boost": 1.12,
                                  "n": int(len(sp)), "end_d": 0.0}), flush=True)
'''


def _walk(tmp_path):
    """Real states: from 160 u past the true spawn on its floor (x 1,020, where the goal field is
    ~160 u farther), walking toward the goal (-x) past the spawn - a 'search line' whose first rows
    lie behind the eval's start, so an eval episode starts past a line drawn there."""
    from surfgym.core import SurfCore, default_config
    from surfgym.rewards import map_spawn_pool
    from test_ge_minutes import PB
    core = SurfCore(str(PB), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    core.set_teleport_fail(True)
    core.reset(0)
    st = map_spawn_pool(core)[0].copy()
    st["origin"][0] = 1020.0
    st["yaw"] = 180.0
    core.set_state(0, st)
    rows = []
    a = np.zeros((1, 6), np.int32)
    a[0, 0], a[0, 1], a[0, 2] = 7, 3, 2                 # no turn, neutral pitch, forward
    for _ in range(120):
        rows.append(core.get_states()[0].copy())
        core.step(a)
    rows = np.array(rows)
    p = tmp_path / "walk.npy"
    np.save(p, rows)
    return p, rows


def _tree(tmp_path, spine_path, root_row):
    """A copy of the trainer in a tree of its own whose tools/auto_spine.py is the stand-in."""
    t = tmp_path / "tree"
    (t / "python").mkdir(parents=True)
    (t / "tools").mkdir()
    shutil.copy2(ROOT / "python" / "train_fast.py", t / "python" / "train_fast.py")
    for name in ("surfgym",):
        os.symlink(ROOT / "python" / name, t / "python" / name)
    for name in ("maps", "build"):
        if (ROOT / name).exists():
            os.symlink(ROOT / name, t / name)
    (t / "tools" / "auto_spine.py").write_text(
        f"SPINE = {str(spine_path)!r}\nROOT_ROW = {int(root_row)}\n" + FAKE_SEARCH)
    return t


def test_the_trainer_holds_and_releases(tmp_path):
    from test_demo_success_d import _spine as _start_spine
    from test_ge_minutes import PB, SMOKE, _env
    if not PB.exists():
        pytest.skip("no princessburglar")
    walk_p, walk = _walk(tmp_path)
    from surfgym.goalfield import GoalField
    z = np.load(ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz", allow_pickle=False)
    f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                  float(z["cell"]), float(z["reach_max"]))
    d = np.asarray(f.sample(np.asarray(walk["origin"], np.float64)), np.float64)
    d_start = float(f.sample(np.array([[860.0, -3005.0, 3653.0]]))[0])
    assert d[0] > d_start + 120.0 and d[-1] < d_start, (d[0], d[-1], d_start)
    root_row = int(np.flatnonzero(d < d_start + 110.0)[0])     # ~50 u past the eval's line
    tree = _tree(tmp_path, walk_p, root_row)
    sp = _start_spine(tmp_path)
    run = "asre_smoke"
    env = _env()
    env["SELF_STATES"] = "1"
    cmd = [sys.executable, "-u", str(tree / "python" / "train_fast.py"), "--run", run] + SMOKE + [
        "--demo-file", str(sp), "--demo-window", "5", "--race-ratchet",
        "--demo-success-d", "50", "--demo-success-bonus", "50",
        "--auto-spine", "--auto-spine-quorum", "1", "--auto-spine-patience", "1",
        "--auto-spine-energy", "150", "--auto-spine-root-energy", "100",
        "--auto-spine-ge-minutes", "0", "--record-every", "2048", "--eval-eps", "2",
        "--ep-ticks", "300", "--steps", "8192"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tree),
                       timeout=1800, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-6000:]
    assert "--auto-spine-root-energy 100: a depth spine from a Go-Explore root boosted" in out
    assert f"the chain's root is row {root_row} x1.12" in out, out[-6000:]
    assert "--auto-spine-root-energy: the depth spine is HELD" in out, out[-6000:]
    assert "-> the HELD depth spine" in out, out[-6000:]
    assert "--auto-spine-root-energy: the held depth spine from beam+ge2 (120 states" in out
    # held at the first search, released on the next eval's quorum (patience 1 searches again)
    assert out.index("the depth spine is HELD") < out.index("-> the HELD depth spine")
    cur = json.loads((tree / "runs" / run / "auto" / "current.json").read_text())
    assert cur["root_energy"] == "released" or cur["root_energy"]["root"] == root_row
    cfg = json.loads((tree / "runs" / run / "run.json").read_text())["config"]
    assert cfg["auto_spine_root_energy"] == 100.0
    bad = subprocess.run([sys.executable, "-u", str(tree / "python" / "train_fast.py"), "--run",
                          "asre_bad"] + SMOKE + [
        "--demo-file", str(sp), "--demo-window", "5", "--race-ratchet",
        "--demo-success-d", "50", "--demo-success-bonus", "50", "--auto-spine",
        "--auto-spine-root-energy", "100", "--steps", "4096"],
        capture_output=True, text=True, env=env, cwd=str(tree), timeout=1800,
        encoding="utf-8", errors="replace")
    assert bad.returncode != 0 and "give --auto-spine and --auto-spine-energy" in (
        bad.stdout + bad.stderr), (bad.stdout + bad.stderr)[-3000:]


def test_flag_off_is_bit_identical_to_the_trainer_before_the_change(tmp_path):
    """The loop on (--auto-spine --auto-spine-energy, its quorum hook read at every eval), the
    flag off: config, every progress.csv number but time/*, and the final weights are those of
    the trainer before --auto-spine-root-energy."""
    import torch
    from test_demo_success_d import TRAIN, _run_demo
    from test_ge_minutes import PB, _csv
    if not PB.exists():
        pytest.skip("no princessburglar")
    pre = ROOT / "python" / "_train_fast_pre_asre.py"
    ref = None
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=50", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for c in (r.stdout.split() if r.returncode == 0 else ()):
        g = subprocess.run(["git", "show", f"{c}:python/train_fast.py"], capture_output=True,
                           cwd=str(ROOT))
        if g.returncode == 0 and b"--auto-spine-root-energy" not in g.stdout:
            pre.write_bytes(g.stdout)
            ref = c
            break
    if ref is None:
        pytest.skip("no pre-change trainer in the first-parent history")
    flags = ["--demo-success-d", "50", "--demo-success-bonus", "50", "--record-every", "2048",
             "--auto-spine", "--auto-spine-quorum", "9", "--auto-spine-energy", "150",
             "--auto-spine-ge-minutes", "0", "--eval-eps", "2", "--ep-ticks", "300"]
    try:
        for run, script in (("asre_off_new", TRAIN), ("asre_off_old", pre)):
            rr = _run_demo(run, flags, tmp_path, script=script)
            assert rr.returncode == 0, rr.stdout[-4000:] + rr.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "asre_off_new", ROOT / "runs" / "asre_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "auto_spine_root_energy" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)


def test_record_ckpt_knows_the_flag():
    import record_ckpt
    assert "auto_spine_root_energy" in record_ckpt.TRAIN_ONLY
