"""beam_tas --start-state FILE[:INDEX] (pbr3b3, 2026-10-24): the population search starts from one
of the policy's own states, spawned through a one-row spawn pool the way the trainer spawns a
spine row, instead of from the map spawn - so a search a few seconds before a wall costs those
seconds, and it works on a --keys-hold checkpoint (which refuses --greedy-prefix / --prefix-line).

CPU. The loader is tested on its own; the search tests need the built core, princessburglar and
a princessburglar checkpoint (BEAM_CKPT, default the autoresearch incumbent) and are skipped
without them: the start state is the spawn of the saved frontier line (clocks zeroed), its replay
reaches the search's frontier, and with the flag off the tool is byte-identical to the tool before
the flag (frontier file and summary, timing keys aside)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

BEAM = ROOT / "tools" / "beam_tas.py"
CKPT = Path(os.environ.get("BEAM_CKPT", "/home/steam/ar_pb/incumbent/ckpt.pt"))


def _states(n=5):
    from surfgym.core import STATE_DTYPE
    a = np.zeros(n, STATE_DTYPE)
    a["origin"][:, 0] = np.arange(n) * 10.0
    a["tick"] = np.arange(n) + 100
    return a


def test_loader_reads_npy_npz_and_indices(tmp_path):
    from beam_tas import load_start_state
    a = _states()
    np.save(tmp_path / "s.npy", a)
    np.savez(tmp_path / "d.npz", ep_0000=a, ep_0001=a[:2])
    row, desc = load_start_state(str(tmp_path / "s.npy"))
    assert row.shape == (1,) and row[0]["origin"][0] == 40.0 and "[4] of 5" in desc
    row, _ = load_start_state(str(tmp_path / "s.npy") + ":1")
    assert row[0]["origin"][0] == 10.0
    row, _ = load_start_state(str(tmp_path / "s.npy") + ":-2")
    assert row[0]["origin"][0] == 30.0
    row, _ = load_start_state(str(tmp_path / "d.npz") + ":2")
    assert row[0]["origin"][0] == 20.0
    with pytest.raises(SystemExit):
        load_start_state(str(tmp_path / "s.npy") + ":5")
    np.save(tmp_path / "bad.npy", np.zeros(3))
    with pytest.raises(SystemExit):
        load_start_state(str(tmp_path / "bad.npy"))


def test_keys_hold_does_not_refuse_it():
    src = BEAM.read_text(encoding="utf-8")
    raw = src[src.index("_raw = [f for f, on in"):src.index('raise SystemExit("--keys-hold checkpoint: "')]
    assert "start_state" not in raw


try:
    from test_ge_minutes import needs_pb  # noqa: E402
except Exception:                          # pragma: no cover
    needs_pb = pytest.mark.skip(reason="no princessburglar test fixtures")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason=f"no checkpoint at {CKPT}")

BASE = ["--envs", "8", "--resample-every", "25", "--score", "d", "--greedy-envs", "2",
        "--save-frontier", "--route-file", "none", "--max-ticks", "200"]


def _beam(script, out, extra=()):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    r = subprocess.run([sys.executable, str(script), str(CKPT), *BASE, *extra,
                        "--out-dir", str(out)], capture_output=True, text=True, env=env,
                       cwd=str(ROOT), timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    return r


@needs_pb
@needs_ckpt
def test_the_search_starts_at_the_given_state(tmp_path):
    from beam_tas import load_start_state
    from surfgym.core import STATE_DTYPE
    # a state of the checkpoint's own greedy line, written by record_ckpt --dump-states
    rec = tmp_path / "rec"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "record_ckpt.py"), str(CKPT),
                        "--episodes", "1", "--ep-ticks", "400", "--out", str(rec) + ".jsonl",
                        "--dump-states", str(rec)], capture_output=True, text=True, env=env,
                       cwd=str(ROOT), timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    npz = Path(str(rec) + ".npz")
    want, _ = load_start_state(f"{npz}:300")
    out = tmp_path / "beam"
    res = _beam(BEAM, out, ["--start-state", f"{npz}:300"])
    assert "--start-state " in res.stdout and "gate skipped" in res.stdout
    z = np.load(out / "beam_frontier.npz")
    sp = z["spawn_state"].astype(STATE_DTYPE)[0]
    assert np.allclose(sp["origin"], want[0]["origin"]) and int(sp["tick"]) == 0
    assert np.allclose(sp["velocity"], want[0]["velocity"])
    assert int(z["replay_ok"]) == 1
    assert json.loads((out / "summary.json").read_text())["start_state"].endswith(":300")


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:tools/beam_tas.py"], capture_output=True,
                           cwd=str(ROOT))
        if s.returncode == 0 and b"--start-state" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
@needs_ckpt
def test_flag_off_is_byte_identical(tmp_path):
    pre = ROOT / "tools" / "_beam_tas_pre_start_state.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change beam_tas in the first-parent history")
        a, b = tmp_path / "new", tmp_path / "old"
        _beam(BEAM, a, ["--skip-gate", "--allow-nonfinisher"])
        _beam(pre, b, ["--skip-gate", "--allow-nonfinisher"])
    finally:
        pre.unlink(missing_ok=True)
    za, zb = np.load(a / "beam_frontier.npz"), np.load(b / "beam_frontier.npz")
    assert za.files == zb.files
    for k in za.files:
        assert za[k].tobytes() == zb[k].tobytes(), k
    sa = json.loads((a / "summary.json").read_text())
    sb = json.loads((b / "summary.json").read_text())
    for k in set(sa) | set(sb):
        if k not in ("env_steps_per_s", "search_wall_s"):
            assert sa.get(k) == sb.get(k), k


def test_pareto_dominance_count_matches_brute_force():
    """--score pareto ranks by how many lineages dominate one on (d lower, energy higher)."""
    from beam_tas import pareto_dominated
    rng = np.random.default_rng(0)
    d, h = rng.uniform(0, 1, 2500), rng.uniform(0, 1, 2500)
    c = pareto_dominated(d, h, chunk=700)
    for i in range(0, 2500, 61):
        bf = int(((d <= d[i]) & (h >= h[i]) & ((d < d[i]) | (h > h[i]))).sum())
        assert c[i] == bf
    assert list(pareto_dominated([1., 2., 3., 2., 5.], [1., 3., 2., 3., 0.])) == [0, 0, 2, 0, 4]
