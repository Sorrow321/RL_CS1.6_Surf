"""--respawn-novel (user, 2026-10-19): the policy's own reservoir states MOVED
to where it has never been, never deeper than its real best, with the
velocity, view and keys they had (surfgym.respawn.NovelSpawnSampler).

(a) The sampler, on the real petrus field and geometry:
    * every row passes every gate: finite potential, no deeper than the
      ceiling, a cell one cell clear of every visited one, the standing
      hull fits, and moved [MIN, MAX] from the source it keeps the
      velocity, view and duck state of;
    * the deep shell supplies the rows it is asked for;
    * the ceiling moves only with REAL envs;
    * a novel episode is tagged by its spawn origin, and the reservoir
      discards it at its end;
    * the real C reset puts an env on the pool row's origin byte for byte,
      which is what the tag relies on;
    * the visit grid and the ceiling survive a checkpoint.
(b) The trainer (CPU, cannonball toy set):
    * a resume onto a filled reservoir builds novel rows, logs them, saves
      the state, and record_ckpt reads the checkpoint;
    * the conflicting spawn sources are refused.
"""
from __future__ import annotations

import json
import os
import shutil
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

PETRUS = ROOT / "maps" / "surf_petrus_lite.bsp"


def _field_npz(stem: str):
    cands = [ROOT / "maps", Path("C:/RL_Surf/maps"), Path("/root/RL_Surf/maps")]
    env = os.environ.get("RLSURF_MAPS")
    if env:
        cands.insert(0, Path(env))
    for d in cands:
        p = d / f"{stem}.goal_32.npz"
        if p.is_file():
            return p
    return None


FIELD_NPZ = _field_npz("surf_petrus_lite")
needs_petrus = pytest.mark.skipif(not (PETRUS.is_file() and FIELD_NPZ is not None),
                                  reason="needs surf_petrus_lite.bsp and its goal_32 field")


def _load_field(path) -> GoalField:
    z = np.load(path, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


@pytest.fixture(scope="module")
def field():
    return _load_field(FIELD_NPZ)


@pytest.fixture(scope="module")
def core():
    from surfgym import SurfCore, default_config
    return SurfCore(str(PETRUS), default_config(
        num_envs=4, spawn_mode=2, max_episode_ticks=200,
        sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))


def _reachable_rows(core, field, n, seed=0):
    """``n`` states at reachable, hull-clear points of the field, each with a
    velocity nobody else has (so a moved row can be matched to its source)."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = field.grid.shape
    lim = field.reach_max - 0.5 * field.cell
    out = []
    while len(out) < n:
        ix = rng.integers(0, nx, 20000)
        iy = rng.integers(0, ny, 20000)
        iz = rng.integers(0, nz, 20000)
        ok = field.grid[iz, iy, ix] < lim
        p = (np.asarray(field.mins) + (np.stack([ix, iy, iz], 1)[ok] + 0.5) * field.cell)
        for q in p:
            if not core.trace(q, q, hull=0).startsolid:
                out.append(q)
            if len(out) >= n:
                break
    rows = np.zeros(n, STATE_DTYPE)
    rows["origin"] = np.asarray(out, np.float32)
    rows["velocity"] = rng.normal(0.0, 900.0, (n, 3)).astype(np.float32)
    rows["yaw"] = rng.uniform(-180.0, 180.0, n).astype(np.float32)
    rows["pitch"] = rng.uniform(-30.0, 30.0, n).astype(np.float32)
    rows["ducked"] = rng.integers(0, 2, n).astype(np.int32)
    rows["onground"] = 3
    return rows


@pytest.fixture(scope="module")
def reservoir(core, field):
    res = RespawnBuffer(4, reservoir=8000, map_id="surf_petrus_lite")
    res.push_many(_reachable_rows(core, field, 3000))
    return res


def _sampler(core, field, n=4, **kw):
    kw.setdefault("seed", 5)
    return NovelSpawnSampler(core, field, n, **kw)


def _mark(s, origins):
    ix = s._cells(np.asarray(origins, np.float64))
    ok = np.all((ix >= 0) & (ix < s.dims), axis=1)
    s.visit[ix[ok, 2], ix[ok, 1], ix[ok, 0]] = True


# ==========================================================================
# (a) the sampler
# ==========================================================================
@needs_petrus
def test_every_row_passes_every_gate(core, field, reservoir):
    s = _sampler(core, field)
    store = reservoir._store[:reservoir.size]
    # a tenth of the reservoir marked as visited (all 3,000 points, one cell of clearance
    # each, cover petrus's ~4,400 reachable 128 u cells and leave nothing to move to)
    _mark(s, store["origin"][:300])
    d_src = field.sample(np.asarray(store["origin"], np.float64))
    s.d_best = float(np.quantile(d_src, 0.2))        # a frontier inside the reservoir
    rows = s.sample_states(600, reservoir)
    assert 100 <= len(rows) <= 600, len(rows)
    p = np.asarray(rows["origin"], np.float64)
    d = field.sample(p)
    assert (d < field.reach_max - 0.5 * field.cell).all(), "a row off the reachable set"
    assert (d >= s.d_best - 1e-3).all(), "a row deeper than the ceiling"
    assert not s._near_visited(p).any(), "a row within a cell of a visited one"
    assert all(not core.trace(q, q, hull=0).startsolid for q in p), "a row in solid"
    assert (rows["onground"] == -1).all()
    # each row is ONE source state moved: same velocity, view and duck state
    key = {store["velocity"][i].tobytes(): i for i in range(len(store))}
    for r in rows:
        i = key[np.ascontiguousarray(r["velocity"]).tobytes()]
        moved = float(np.linalg.norm(np.asarray(r["origin"], np.float64)
                                     - np.asarray(store["origin"][i], np.float64)))
        assert s.r_lo - 0.01 <= moved <= s.r_hi + 0.01, moved
        assert r["yaw"] == store["yaw"][i] and r["pitch"] == store["pitch"][i]
        assert r["ducked"] == store["ducked"][i]


@needs_petrus
def test_the_deep_shell_is_where_it_is_asked_for(core, field, reservoir):
    store = reservoir._store[:reservoir.size]
    d_src = field.sample(np.asarray(store["origin"], np.float64))
    deep = float(np.quantile(d_src, 0.25))
    key = {store["velocity"][i].tobytes(): i for i in range(len(store))}
    shares = {}
    for frac in (0.0, 0.9):
        s = _sampler(core, field, shell_frac=frac, shell_width=0.25)
        s.d_best = float(d_src.min())
        rows = s.sample_states(400, reservoir)
        src = np.array([key[np.ascontiguousarray(r["velocity"]).tobytes()] for r in rows])
        shares[frac] = float((d_src[src] <= deep).mean())
    # flattened over progress, the deepest quarter of the range holds a share of the sources
    # set by the reservoir's own spread; asked for 90 %, it holds most of them
    assert shares[0.9] >= 0.75 and shares[0.9] > shares[0.0] + 0.3, shares


@needs_petrus
def test_the_ceiling_moves_only_with_real_envs(core, field, reservoir):
    s = _sampler(core, field, n=3)
    store = reservoir._store[:reservoir.size]
    d_src = field.sample(np.asarray(store["origin"], np.float64))
    order = np.argsort(d_src)
    deep, mid, far = (np.asarray(store["origin"][order[k]], np.float64)
                      for k in (0, len(order) // 2, len(order) - 1))
    s.novel_env[:] = [True, False, False]            # env 0 flies a novel episode
    s.observe(np.stack([deep, mid, far]))
    assert s.d_best == pytest.approx(float(field.sample(mid[None])[0]))
    s.novel_env[:] = False                           # the same point, now a real episode
    s.observe(np.stack([deep, mid, far]))
    assert s.d_best == pytest.approx(float(field.sample(deep[None])[0]))
    # only REAL envs mark their cells: the first observe had env 0 novel
    t = _sampler(core, field, n=3)
    t.novel_env[:] = [True, False, False]
    t.observe(np.stack([deep, mid, far]))
    assert t._near_visited(np.stack([mid, far])).all()
    assert not t.visit[tuple(t._cells(deep[None])[0][::-1])], "a novel env marked its cell"


@needs_petrus
def test_novel_episodes_are_tagged_and_never_harvested(core, field, reservoir):
    s = _sampler(core, field, n=4)
    s.d_best = 0.0
    pool = np.zeros(64, STATE_DTYPE)
    pool = s.mix(pool, 4, 60, reservoir)
    assert len(s._keys) >= 30
    novel = pool[4:6]
    res = RespawnBuffer(4, reservoir=1000, margin_ticks=0, snap_every=1,
                        map_id="surf_petrus_lite")
    res.skip = s.novel_env
    st = np.zeros(4, STATE_DTYPE)
    st[0], st[1] = novel[0], novel[1]                # envs 0, 1 spawn on novel rows
    st["origin"][2:] = np.array([1.0e6, 0.0, 0.0], np.float32)
    ended = np.ones(4, bool)
    res.observe(st, ended)
    s.note_spawns(st, ended)
    assert s.novel_env.tolist() == [True, True, False, False]
    assert s.pop_share() == pytest.approx(0.5) and s.pop_share() != s.pop_share()
    for _ in range(5):                               # five ticks of snapshots each
        res.observe(st, np.zeros(4, bool))
    res.observe(st, ended)                           # every episode ends
    rows, _, envs = res.drain_harvest()
    assert len(rows) and set(envs.tolist()) == {2, 3}, envs
    # the keys of the pool before last still count (a reset can draw from it)
    s.mix(pool, 4, 0, reservoir)
    st2 = np.zeros(4, STATE_DTYPE)
    st2[0] = novel[0]
    s.note_spawns(st2, np.array([True, False, False, False]))
    assert s.novel_env[0]


@needs_petrus
def test_the_real_reset_lands_on_the_pool_origin_byte_for_byte(core, field, reservoir):
    s = _sampler(core, field, n=4)
    s.d_best = 0.0
    from surfgym.rewards import map_spawn_pool
    fresh = map_spawn_pool(core)[:1]
    pool = s.mix(np.repeat(fresh, 8), 1, 7, reservoir)
    core.set_spawn_pool(pool)
    core.reset()
    act = np.zeros((4, 6), np.int32)
    core.step(act)
    core.force_fail(np.ones(4, bool))
    core.step(act)
    sv = core.states_view
    s.note_spawns(sv, np.ones(4, bool))
    want = [np.ascontiguousarray(sv["origin"][i], np.float32).tobytes() in s._keys
            for i in range(4)]
    assert s.novel_env.tolist() == want
    on_pool = {np.ascontiguousarray(r, np.float32).tobytes() for r in pool["origin"]}
    assert all(np.ascontiguousarray(sv["origin"][i], np.float32).tobytes() in on_pool
               for i in range(4))


@needs_petrus
def test_the_window_forgets_what_the_policy_stopped_visiting(core, field, reservoir):
    s = _sampler(core, field, window=100.0)
    p = np.asarray(reservoir._store[:1]["origin"], np.float64)
    assert not s.roll(0)                             # anchors the first window
    _mark(s, p)
    assert s._near_visited(p).all()
    assert not s.roll(50)                            # inside the window
    assert s.roll(100)                               # live -> previous
    assert s._near_visited(p).all(), "remembered for one more window"
    assert s.roll(200)                               # nothing new was entered
    assert not s._near_visited(p).any(), "forgotten after two windows"
    # window 0 = ever visited: never forgets
    e = _sampler(core, field)
    _mark(e, p)
    assert not e.roll(10 ** 12) and e._near_visited(p).all()


@needs_petrus
def test_the_grid_and_the_ceiling_survive_a_checkpoint(core, field, reservoir):
    s = _sampler(core, field, window=100.0)
    s.roll(0)
    _mark(s, reservoir._store[:500]["origin"])
    s.roll(100)
    _mark(s, reservoir._store[500:900]["origin"])
    s.d_best = 1234.5
    t = _sampler(core, field, window=100.0)
    assert t.load_state_dict(s.state_dict())
    assert np.array_equal(t.visit, s.visit) and t.d_best == 1234.5
    assert np.array_equal(t.visit_prev, s.visit_prev) and t._w_start == 100
    u = _sampler(core, field, cell=64.0)
    assert not u.load_state_dict(s.state_dict()), "a different grid is not restored"


def test_the_recorder_treats_the_flags_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    for k in ("respawn_novel", "respawn_novel_frac", "respawn_novel_shell",
              "respawn_novel_shell_width", "respawn_novel_radius",
              "respawn_novel_cell", "respawn_novel_tries", "respawn_novel_window"):
        assert f'"{k}"' in src, k


# ==========================================================================
# (b) the trainer
# ==========================================================================
from test_unstuck import ABS, TRAIN, _run, _train                  # noqa: E402
from test_view_continuous import CANNONBALL, SMOKE_FLAGS, needs_run  # noqa: E402


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_a_resume_onto_a_full_reservoir_builds_logs_saves_and_records():
    chk = _run([sys.executable, "-c",
                "import torch; assert not torch.cuda.is_available()"], 300)
    assert chk.returncode == 0, chk.stderr
    import torch
    from surfgym import SurfCore, default_config
    run0 = "cya_novel_base"
    _train(run0, ABS, steps="2048")
    d0 = ROOT / "runs" / run0
    ck = torch.load(d0 / "ckpt_final.pt", map_location="cpu", weights_only=False)
    # a reservoir of 3,000 own-looking states near the cannonball start (the trainer
    # mixes only once the reservoir holds 2,000)
    cb = SurfCore(str(CANNONBALL), default_config(num_envs=1, spawn_mode=2,
                                                  max_episode_ticks=200,
                                                  sv_maxvelocity=4000.0,
                                                  lidar_w=0, lidar_h=0))
    fcb = _load_field(_field_npz("surf_src_cannonball"))
    rows = _reachable_rows(cb, fcb, 3000, seed=1)
    ck["respawn"] = {"states": rows, "map_id": ck["respawn"]["map_id"]}
    # and a sampler state as deep as that reservoir: a fresh run's real envs set the
    # ceiling by flying, which a 12-iteration smoke never gets far enough to do
    ns = NovelSpawnSampler(cb, fcb, 64)
    ns.d_best = float(fcb.sample(np.asarray(rows["origin"], np.float64)).min())
    ck["respawn_novel"] = {"cannonball": ns.state_dict()}
    seeded = d0 / "ckpt_seeded.pt"
    torch.save(ck, seeded)
    run = "cya_novel"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    # --stall-secs 2: episodes end inside the smoke, so resets draw from the pool
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE_FLAGS + ["--steps", "24576", "--respawn-reservoir", "5000",
                              "--stall-secs", "2", "--respawn-novel"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "respawn NOVEL: 50% of the non-start pool" in r.stdout
    assert "restored --respawn-novel state" in r.stdout
    assert "novel: ceiling" in r.stdout and "novel ceil" in r.stdout
    cfg = _cfg(run)
    assert cfg["respawn_novel"] == 1 and cfg["respawn_novel_radius"] == [256.0, 2048.0]
    lines = (ROOT / "runs" / run / "progress.csv").read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    assert head[-6:] == ["novel/ceiling", "novel/rows", "novel/prog_med",
                         "novel/moved_med", "novel/share", "novel/visited"]
    vals = [ln.split(",") for ln in lines[1:]]
    assert max(int(v[head.index("novel/rows")] or 0) for v in vals) > 500
    assert max(int(v[head.index("novel/visited")] or 0) for v in vals) > 0
    # episodes really started from novel rows: the per-tick tag found them
    assert max(float(v[head.index("novel/share")] or 0) for v in vals) > 0.0
    ck2 = torch.load(ROOT / "runs" / run / "ckpt_final.pt", map_location="cpu",
                     weights_only=False)
    assert "respawn_novel" in ck2 and ck2["respawn_novel"]
    # a bare resume restores the sampler and its state
    run2 = "cya_novel_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE_FLAGS
              + ["--steps", "26624", "--respawn-reservoir", "5000"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_novel" in r2.stdout and "restored --respawn-novel state" in r2.stdout
    # the recorder reads the checkpoint (the record gate's own backend)
    out = ROOT / "runs" / run / "rec_novel.jsonl"
    r5 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(out),
               "--episodes", "1", "--ep-ticks", "400"])
    assert r5.returncode == 0, r5.stdout[-4000:] + r5.stderr[-4000:]
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--respawn-frac", "0"], "--respawn-novel needs the reservoir"),
    (["--respawn-frontier"], "--respawn-novel is not wired with --respawn-frontier"),
])
def test_conflicting_spawn_sources_are_refused(extra, msg):
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "cya_novel_bad"] + SMOKE_FLAGS
             + ["--steps", "2048", "--respawn-novel"] + extra)
    shutil.rmtree(ROOT / "runs" / "cya_novel_bad", ignore_errors=True)
    assert r.returncode != 0 and msg in r.stdout + r.stderr, r.stdout[-2000:]
