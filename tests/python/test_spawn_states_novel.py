"""--spawn-states-novel (2026-10-20, autoresearch r2b1): --spawn-states rows drawn only where the
policy's own episodes have not been (the --respawn-novel visit gate), their episodes novel.

(a) The sampler, on sidistic's real field and geometry: pick_unvisited returns only rows whose
    3x3x3 visit block is unmarked, nothing when every block is marked; add_keys makes an episode
    that starts from such a row a NOVEL episode (tagged, so the reservoir discards it).
(b) The trainer (CPU smoke on sidistic): with the flag the pool draws the file's unvisited rows
    and the step log reports them, the config carries the key and a flagless resume restores
    it; the flag without --respawn-novel is refused.
(c) Flag off: the trainer before this change and after it give the same config, progress.csv
    (fps aside), eval trajectory bytes and weights on a --spawn-states + --respawn-novel smoke.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.core import STATE_DTYPE                               # noqa: E402
from surfgym.goalfield import GoalField                            # noqa: E402
from surfgym.respawn import NovelSpawnSampler, RespawnBuffer       # noqa: E402
from test_ez_viewc import SMOKE, SID, SID_FIELD, TRAIN, _csv, _env, needs_run  # noqa: E402


def _field():
    z = np.load(SID_FIELD, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def _core(n=4):
    from surfgym import SurfCore, default_config
    return SurfCore(str(SID), default_config(num_envs=n, spawn_mode=2, max_episode_ticks=200,
                                             sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))


def _reachable_rows(core, field, n, seed=0):
    rng = np.random.default_rng(seed)
    nz, ny, nx = field.grid.shape
    lim = field.reach_max - 0.5 * field.cell
    out = []
    while len(out) < n:
        ix, iy, iz = (rng.integers(0, m, 20000) for m in (nx, ny, nz))
        ok = field.grid[iz, iy, ix] < lim
        p = np.asarray(field.mins) + (np.stack([ix, iy, iz], 1)[ok] + 0.5) * field.cell
        for q in p:
            if not core.trace(q, q, hull=0).startsolid:
                out.append(q)
            if len(out) >= n:
                break
    rows = np.zeros(n, STATE_DTYPE)
    rows["origin"] = np.asarray(out, np.float32)
    rows["velocity"] = rng.normal(0.0, 900.0, (n, 3)).astype(np.float32)
    rows["onground"] = -1
    return rows


needs_sid = pytest.mark.skipif(not (SID.exists() and SID_FIELD.exists()),
                               reason="needs sidistic and its prebaked goal field")


@needs_sid
def test_pick_unvisited_draws_only_unmarked_blocks_and_tags_their_episodes():
    core, field = _core(), _field()
    s = NovelSpawnSampler(core, field, 4, seed=3)
    rows = _reachable_rows(core, field, 400)
    # mark the cells of the first half: those rows (and their 3x3x3 neighbours) are visited
    ix = s._cells(np.asarray(rows["origin"][:200], np.float64))
    s.visit[ix[:, 2], ix[:, 1], ix[:, 0]] = True
    near = s._near_visited(np.asarray(rows["origin"], np.float64))
    assert near[:200].all() and (~near[200:]).sum() > 50
    got, nc = s.pick_unvisited(rows, 300)
    assert len(got) == 300 and nc == int((~near).sum())
    assert not s._near_visited(np.asarray(got["origin"], np.float64)).any()
    # every block marked: nothing is drawn
    s.visit[:] = True
    none, nc0 = s.pick_unvisited(rows, 10)
    assert len(none) == 0 and nc0 == 0
    s.visit[:] = False
    # an episode starting from an added row is a NOVEL episode, and the reservoir drops it
    pool = np.zeros(64, STATE_DTYPE)
    pool = s.mix(pool, 4, 0, None)                  # no novel rows this iteration
    pick, _ = s.pick_unvisited(rows, 2)
    s.add_keys(pick)
    res = RespawnBuffer(4, reservoir=1000, margin_ticks=0, snap_every=1,
                        map_id="surf_src_sidistic")
    res.skip = s.novel_env
    st = np.zeros(4, STATE_DTYPE)
    st[0], st[1] = pick[0], pick[1]
    st["origin"][2:] = np.array([1.0e6, 0.0, 0.0], np.float32)
    ended = np.ones(4, bool)
    res.observe(st, ended)
    s.note_spawns(st, ended)
    assert s.novel_env.tolist() == [True, True, False, False]
    # the keys survive one more mix (a reset can still draw from the previous pool)
    s.mix(pool, 4, 0, None)
    s.note_spawns(st, np.array([True, False, False, False]))
    assert s.novel_env[0]


def _states_file(tmp_path, n=600):
    core, field = _core(), _field()
    rows = _reachable_rows(core, field, n, seed=5)
    p = tmp_path / "own_states.npy"
    np.save(p, rows)
    return p


def _run(cmd, timeout=1800, self_states=True):
    env = _env()
    if self_states:
        env["SELF_STATES"] = "1"
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT),
                          timeout=timeout, encoding="utf-8", errors="replace")


# a 4 s cap and a 0.1 s harvest margin fill the reservoir inside the smoke, so the novel
# sampler (which mixes only once the reservoir holds 2,000 states) and its grid are live
FILL = ["--ep-ticks", "400", "--respawn-margin", "0.1", "--respawn-novel"]


def _train(run, extra, steps="40960", script=TRAIN):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE + FILL
             + ["--steps", steps] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


@needs_run
def test_the_flag_draws_unvisited_rows_logs_them_and_resumes(tmp_path):
    f = _states_file(tmp_path)
    run = "ssnovel_on"
    r = _train(run, ["--spawn-states", str(f), "--spawn-states-frac", "0.5",
                     "--spawn-states-novel"])
    assert "--spawn-states-novel: the spawn-states rows are drawn only" in r.stdout
    m = re.findall(r"--spawn-states-novel: ([0-9,]+) of ([0-9,]+) file states", r.stdout)
    assert m and all(int(b.replace(",", "")) == 600 for _, b in m), r.stdout[-3000:]
    assert any(int(a.replace(",", "")) > 0 for a, _ in m)
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["spawn_states_novel"] == 1 and cfg["respawn_novel"] == 1
    run2 = "ssnovel_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--steps", "45056"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "spawn_states_novel" in r2.stdout
    bad = _run([sys.executable, "-u", str(TRAIN), "--run", "ssnovel_bad"] + SMOKE
               + ["--steps", "2048", "--spawn-states", str(f), "--spawn-states-frac", "0.5",
                  "--spawn-states-novel"])
    assert bad.returncode != 0 and "needs --respawn-novel" in (bad.stdout + bad.stderr)
    for dd in (d, ROOT / "runs" / run2, ROOT / "runs" / "ssnovel_bad"):
        shutil.rmtree(dd, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"spawn-states-novel" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change(tmp_path):
    import torch
    f = _states_file(tmp_path)
    extra = ["--spawn-states", str(f), "--spawn-states-frac", "0.5"]
    pre = ROOT / "python" / "_train_fast_pre_ssnovel.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("ssnovel_off_new", extra, steps="20480")
        _train("ssnovel_off_old", extra, steps="20480", script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "ssnovel_off_new", ROOT / "runs" / "ssnovel_off_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb and "spawn_states_novel" not in ca
    ra, rb = _csv(a.name), _csv(b.name)
    assert ra and len(ra) == len(rb) and list(ra[0]) == list(rb[0])
    for x, y in zip(ra, rb):
        for k in x:
            if k != "time/fps":
                assert x[k] == y[k], k
    ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
    assert ta and [p.name for p in ta] == [p.name for p in tb]
    for p, q in zip(ta, tb):
        assert p.read_bytes() == q.read_bytes(), p.name
    sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    for dd in (a, b):
        shutil.rmtree(dd, ignore_errors=True)
