"""``--heldout-maps``: EVAL-ONLY maps - evaluated at every eval, never trained on.

The generalisation probe. A policy that learned "a ramp can be surfed" makes
progress on a map it has never seen; a map-memoriser does not. What is
pinned here, in the order it would hurt if it broke:

  * **a held-out slot never contributes a rollout row.** It owns no env
    range, it is not in ``MapFleet.slots``, and every hot-path method of the
    fleet (step, reward, respawn, novelty counts, render, bootstrap) is
    driven with a held-out slot whose core and reward object RAISE if
    touched. ``--dump-invariants`` prints ``heldout_envs`` and it is 0.
  * **its eval columns appear, LAST, and never in an aggregate.**
    ``race/heldout_progress_field.<tag>`` etc. sit after every other column
    (the progress.csv header-migration rule), and ``race/map_pct`` is the
    mean over the TRAINING maps alone.
  * **the honest corridor MAX matches tools/eval_honesty.py --order-only 16**
    where a route line exists.
  * **the recorder agrees with the trainer** on which cells a held-out map's
    field and SDF were built at (``heldout_goal_cells`` / ``heldout_map_cells``
    in run.json), so ``record_ckpt.py --map <held-out>`` is the same probe.

The CPU runs at the bottom use three tiny pool maps with prebaked caches
(surf_bom, surf_d as training, surf_0way held out) and hide the GPU with
``CUDA_VISIBLE_DEVICES=-1``; they skip when the pool or the DLL is absent.

    python -m pytest tests/python/test_heldout.py -q
"""
from __future__ import annotations

import csv
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

from surfgym.core import STATE_DTYPE                            # noqa: E402
from surfgym.mapfleet import HeldoutSlot, MapFleet, MapSlot      # noqa: E402
from train_fast import (HELD_COLS, HELD_CORR_COL, corridor_max,  # noqa: E402
                        heldout_columns, heldout_csv_values)

TRAIN_SRC = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
REC_SRC = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
RUN_ARM = (ROOT / "tools" / "run_arm.sh").read_text(encoding="utf-8")

# the map pool with prebaked caches. From a worktree the main checkout's
# copy is the one whose cache signatures match (CLAUDE.md: absolute paths
# into the main checkout); a box unpacks the pool over maps/.
POOL = next((p for p in (ROOT / "maps_pool", Path("C:/RL_Surf/maps_pool"),
                         ROOT / "maps")
             if (p / "surf_bom.goal_48.npz").exists()
             and (p / "surf_d.goal_48.npz").exists()
             and (p / "surf_0way.goal_48.npz").exists()), None)
TRAIN = ["surf_bom", "surf_d"]
HELD = "surf_0way"


# --------------------------------------------------------------------------
# DLL-free fakes
# --------------------------------------------------------------------------
class FakeCore:
    OBS = 15

    def __init__(self, n):
        self.num_envs = int(n)
        self.obs_dim = self.OBS
        self.states_view = np.zeros(n, dtype=STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self.seen = None
        self.reset_seed = None
        self.failed = None

    def map_bounds(self):
        return (np.full(3, -1e4, np.float32), np.full(3, 1e4, np.float32))

    def reset(self, seed=0):
        self.reset_seed = seed
        return np.zeros((self.num_envs, self.OBS), np.float32)

    def step(self, actions):
        self.seen = actions.copy()
        n = self.num_envs
        return (np.zeros((n, self.OBS), np.float32), np.zeros(n, np.float32),
                np.zeros(n, np.uint8), np.zeros(n, np.uint8),
                np.zeros((n, self.OBS), np.float32))

    def force_fail(self, mask):
        self.failed = np.asarray(mask).copy()


class TouchyCore(FakeCore):
    """The held-out map's core: any hot-path call is a test failure."""

    def reset(self, seed=0):
        raise AssertionError("held-out core reset by the fleet")

    def step(self, actions):
        raise AssertionError("held-out core stepped by the fleet")

    def force_fail(self, mask):
        raise AssertionError("held-out core force_fail'd by the fleet")


class QuietReward:
    """A training reward with every method the fleet calls."""

    int_coef = 0.0

    def __init__(self, n):
        self.n = n
        self.resets = 0

    def on_reset(self, core):
        self.resets += 1

    def __call__(self, prev, obs, term, base, done, trunc, core, goal=None):
        return np.ones(len(done), np.float32)

    def latch_flags(self):
        return None

    def pop_stats(self):
        return {"success_rate": 0.5, "finish_s": float("nan"),
                "episodes": 2, "int_per_ep": float("nan")}

    def stats_vector(self):
        return np.zeros(6, np.float64)

    def clear_stats(self):
        pass

    def set_step(self, s):
        pass


class NeverCalled:
    """The held-out map's reward object: exists for the eval mirrors only."""

    field = None
    int_coef = 1.0
    stall_ticks = 1500
    stall_eps = 32.0
    every = 1

    def _boom(self, *a, **k):
        raise AssertionError("held-out reward_fn touched by the fleet")

    __call__ = on_reset = latch_flags = pop_stats = _boom
    stats_vector = clear_stats = set_step = counts_delta_sparse = _boom
    pop_stall_mask = stagnant_mask = apply_counts_delta_sparse = _boom


def _training_slots(sizes):
    out, lo = [], 0
    for i, n in enumerate(sizes):
        s = MapSlot(f"surf_map{i}", f"/x/surf_map{i}.bsp", FakeCore(n),
                    lo, lo + n)
        s.reward_fn = QuietReward(n)
        s.map_center = np.array([1000.0 * i, 0.0, 0.0], np.float32)
        out.append(s)
        lo += n
    return out


def _heldout(name="surf_probe", at=8):
    h = HeldoutSlot(name, f"/x/{name}.bsp", TouchyCore(1), at)
    h.reward_fn = NeverCalled()
    h.map_center = np.array([9e3, 9e3, 9e3], np.float32)
    h.d0 = h.rf_d0 = 12345.0
    return h


# --------------------------------------------------------------------------
# 1. the slot and the fleet
# --------------------------------------------------------------------------
def test_a_heldout_slot_owns_no_env_rows():
    h = _heldout(at=8)
    assert h.heldout is True
    assert h.n == 0
    assert h.sl == slice(8, 8)
    assert np.zeros(10)[h.sl].size == 0
    assert MapSlot("surf_a", "/x/a.bsp", FakeCore(4), 0, 4).heldout is False
    with pytest.raises(ValueError):          # its core is the 1-env eval core
        HeldoutSlot("surf_b", "/x/b.bsp", FakeCore(4), 0)


def test_the_fleet_never_touches_a_heldout_slot_on_the_hot_path():
    slots = _training_slots([4, 4])
    h = _heldout(at=8)
    f = MapFleet(slots, heldout=[h])
    assert f.n_envs == 8 and f.n_maps == 2
    assert h not in f.slots
    assert f.eval_slots == slots + [h]
    assert f.heldout == [h]
    # the whole hot path, exactly as train_fast drives it
    obs = f.reset(3)
    assert obs.shape == (8, 15)
    f.on_reset()
    acts = np.zeros((8, 6), np.int32)
    o2, base, done, trunc, term = f.step(acts)
    assert o2.shape == (8, 15) and base.shape == (8,)
    r = f.reward(obs, o2, term, base, done, trunc)
    assert r.shape == (8,) and float(r.sum()) == 8.0
    assert f.goal_hits().shape == (8,)
    assert f.apply_stall_kills() == 0
    assert f.stagnant_mask() is None
    assert f.latch_flags().shape == (8,)
    f.observe_respawn(np.zeros(8, bool))
    f.track_start_bins(np.zeros(8, bool), np.zeros(8, bool),
                       np.full(8, -1, np.int64))
    f.set_step(10)
    assert f.pop_stats()["episodes"] == 4
    assert f.stats_vector().shape == (6,)
    f.clear_stats()
    assert f.reservoir_size() == 0
    assert f.reservoir_min_depth() != f.reservoir_min_depth()   # NaN
    dt = np.dtype([("slot", np.int32), ("cell", np.int64), ("inc", np.int64)])
    assert len(f.counts_delta_sparse(dt)) == 0
    assert f.map_centers(np.arange(8)).shape == (8, 3)
    vis = np.zeros((8, 6), np.float32)
    f.fill_pose(vis)
    # the held-out core and reward were never reached
    assert h.core.seen is None and h.core.reset_seed is None
    assert h.core.failed is None
    for s in slots:
        assert s.core.seen is not None and s.reward_fn.resets == 1


def test_the_fleet_refuses_a_heldout_that_could_train():
    slots = _training_slots([4, 4])
    # a plain MapSlot is not a held-out slot, whatever its name
    with pytest.raises(ValueError):
        MapFleet(slots, heldout=[MapSlot("surf_z", "/x/z.bsp", FakeCore(0),
                                         8, 8)])
    # the same map on both sides is a contradiction, not a configuration
    with pytest.raises(ValueError):
        MapFleet(slots, heldout=[_heldout("surf_map0")])
    # a reservoir on a held-out map would harvest states to train from
    h = _heldout()
    h.respawn = object()
    with pytest.raises(ValueError):
        MapFleet(slots, heldout=[h])
    # and a held-out slot in the TRAINING list is refused too
    with pytest.raises(ValueError):
        MapFleet(slots + [_heldout(at=8)])


def test_tags_stay_unique_across_training_and_heldout_maps():
    slots = _training_slots([4])
    slots[0].name = "surf_x"
    h = _heldout("surf_src_x", at=4)          # would also tag as "x"
    f = MapFleet(slots, heldout=[h])
    f.retag()
    assert slots[0].tag == "surf_x" and h.tag == "surf_src_x"
    assert f.tags == ["surf_x"]


# --------------------------------------------------------------------------
# 2. the CSV block
# --------------------------------------------------------------------------
class _Tagged:
    def __init__(self, tag, route=None):
        self.tag, self.route = tag, route


def test_heldout_columns_are_the_field_metric_plus_corridor_where_routed():
    a, b = _Tagged("alpha"), _Tagged("beta", route=Path("beta.route.npz"))
    cols = heldout_columns([a, b])
    assert cols == ["race/heldout_progress_field.alpha",
                    "race/heldout_finish_s.alpha",
                    "race/heldout_finishes.alpha",
                    "race/heldout_pct_field.alpha",
                    "race/heldout_progress_field.beta",
                    "race/heldout_finish_s.beta",
                    "race/heldout_finishes.beta",
                    "race/heldout_pct_field.beta",
                    "race/heldout_corridor_max.beta"]
    assert all("field" in c for c in cols if c.startswith(HELD_COLS[0][:20])
               and "progress" in c)
    assert HELD_CORR_COL == "race/heldout_corridor_max"
    nan = float("nan")
    vals = heldout_csv_values([a, b], {"alpha": (nan, nan, nan, nan),
                                       "beta": (1234.56, 21.234, 2.0, 55.5555)},
                              {"beta": 9876.54})
    assert vals == ["", "", "", "", 1234.6, 21.23, 2, 55.556, 9876.5]
    assert len(vals) == len(cols)


def _write_traj(path, episodes):
    """episodes: list of (T, 3) position arrays -> a record.py-shaped file."""
    with open(path, "w", encoding="utf-8") as f:
        for ep, xyz in enumerate(episodes):
            f.write(json.dumps({"map": "synthetic", "episode": ep}) + "\n")
            for t, p in enumerate(xyz):
                row = [t, float(p[0]), float(p[1]), float(p[2]),
                       0.0, 0.0, 0.0, 0.0, 0.0]
                f.write(json.dumps(row) + "\n")
            f.write(json.dumps({"end": "fail", "ticks": len(xyz)}) + "\n")


def test_corridor_max_is_eval_honesty_order_only_at_its_maximum(tmp_path):
    from eval_honesty import corridor_progress_ordered
    # a straight route along +x, 20,000 u at the default 128 u spacing
    pts = np.stack([np.arange(0.0, 20001.0, 128.0),
                    np.zeros(157), np.zeros(157)], axis=1).astype(np.float32)
    rp = tmp_path / "synthetic.route.npz"
    np.savez(rp, route=pts, spacing=np.float32(128.0))
    # episode A: 3,000 u along the line, then a fall 6,000 u off it that
    # drifts alongside the route's far end (the off-route dive the order-only
    # rule must NOT credit). episode B: 7,000 u along the line, on it.
    a_on = np.stack([np.linspace(0, 3000, 60), np.zeros(60), np.zeros(60)], 1)
    a_off = np.stack([np.linspace(15000, 19000, 40), np.full(40, 6000.0),
                      np.full(40, -3000.0)], 1)
    ep_a = np.concatenate([a_on, a_off])
    ep_b = np.stack([np.linspace(0, 7000, 140), np.zeros(140),
                     np.zeros(140)], 1)
    tp = tmp_path / "traj_0000000001_synthetic.jsonl"
    _write_traj(tp, [ep_a, ep_b])
    got = corridor_max(tp, rp)
    ref = max(corridor_progress_ordered(ep, pts, 128.0, 1500.0, 16)[0]
              for ep in (ep_a, ep_b))
    assert got == pytest.approx(ref)
    assert 6800.0 <= got <= 7200.0         # B's frontier, not A's dive
    # an episode-less recording scores NaN rather than 0
    empty = tmp_path / "traj_0000000002_synthetic.jsonl"
    empty.write_text(json.dumps({"map": "synthetic"}) + "\n", encoding="utf-8")
    assert corridor_max(empty, rp) != corridor_max(empty, rp)


# --------------------------------------------------------------------------
# 3. run.json / resume / recorder agreement / launcher (source-level)
# --------------------------------------------------------------------------
def test_run_json_records_the_heldout_list_and_the_ckpt_restores_it():
    assert '"heldout_maps": (list(HSTEMS) if NHELD else None)' in TRAIN_SRC
    assert '"heldout_goal_cells": ({s.tag: s.goal_cell' in TRAIN_SRC
    assert '"heldout_map_cells": ({s.tag: s.cell for s in heldout}' in TRAIN_SRC
    assert 'flag_given("--heldout-maps") and ck_cfg.get("heldout_maps")' \
        in TRAIN_SRC
    assert 'flag_given("--heldout-goal-cell")' in TRAIN_SRC
    # the held-out block is appended LAST, after the PPO hygiene columns
    hyg = TRAIN_SRC.index('"race/crawl_frac", "train/ret_mean", "train/ret_std"]')
    assert TRAIN_SRC.index("CSV_COLS += heldout_columns(heldout)") > hyg
    # and never enters the aggregate: the training table is rows :NMAPS
    assert "ev = ev_all[:NMAPS]" in TRAIN_SRC
    assert 'eval_aggregate(ev, [s.finish_kind for s in slots], n_rec)' \
        in TRAIN_SRC


def test_the_recorder_reads_the_heldout_cells_the_trainer_wrote():
    # tools/record_ckpt.py --map <held-out map> must build the field at the
    # cell the trainer's held-out eval used, or the zero-shot probe there is
    # a different measurement from race/heldout_*.<tag>
    for key in ("heldout_maps", "heldout_goal_cells", "heldout_map_cells",
                "heldout_goal_cell"):
        assert f'"{key}"' in REC_SRC, key
    assert 'hgcells = cfg.get("heldout_goal_cells")' in REC_SRC
    assert '_cells = dict(cfg.get("heldout_map_cells") or {})' in REC_SRC


def test_run_arm_multimap_branch_passes_heldout_and_runs_one_gpu_direct():
    assert 'HELD=(--heldout-maps "$HELDOUT_MAPS")' in RUN_ARM
    assert 'HELD+=(--heldout-goal-cell "$HELDOUT_CELLS")' in RUN_ARM
    assert '${HELD[@]+"${HELD[@]}"} "$@"' in RUN_ARM
    # MULTIMAP=1: a single process, no torchrun, no warm-caches pre-pass
    assert 'if [ "$NGPU" = "1" ]; then' in RUN_ARM
    assert 'nohup python3 -u python/train_fast.py --run "$RUN" "${ARGS[@]}"' \
        in RUN_ARM


# --------------------------------------------------------------------------
# 4. CLI guards (fast: they fire before any core, field or SDF is loaded)
# --------------------------------------------------------------------------
def _train(*argv, env=None):
    e = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, str(ROOT / "python" / "train_fast.py"), *argv],
        capture_output=True, text=True, timeout=900, env=e)


needs_pool = pytest.mark.skipif(POOL is None, reason="no map pool with "
                                "prebaked caches (surf_bom/surf_d/surf_0way)")


def _bsp(stem):
    return str(POOL / f"{stem}.bsp")


@needs_pool
def test_a_heldout_map_may_not_also_be_a_training_map():
    p = _train("--reward", "race", "--envs", "8",
               "--maps", ",".join(_bsp(m) for m in TRAIN),
               "--heldout-maps", _bsp(TRAIN[1]))
    assert p.returncode != 0
    assert "overlaps the TRAINING maps" in p.stdout + p.stderr


@needs_pool
def test_heldout_needs_the_race_objective():
    p = _train("--reward", "path", "--envs", "8", "--map", _bsp(TRAIN[0]),
               "--heldout-maps", _bsp(HELD))
    assert p.returncode != 0
    assert "--heldout-maps needs --reward race" in p.stdout + p.stderr


@needs_pool
def test_heldout_refuses_single_map_objects():
    p = _train("--reward", "race", "--envs", "8", "--map", _bsp(TRAIN[0]),
               "--heldout-maps", _bsp(HELD), "--goals", "1")
    assert p.returncode != 0
    assert "cannot combine with --goals" in p.stdout + p.stderr


@needs_pool
def test_heldout_goal_cells_must_match_the_map_count():
    p = _train("--reward", "race", "--envs", "8", "--map", _bsp(TRAIN[0]),
               "--heldout-maps", _bsp(HELD), "--heldout-goal-cell", "48,32")
    assert p.returncode != 0
    assert "lists 2 cells for 1 held-out maps" in p.stdout + p.stderr


# --------------------------------------------------------------------------
# 5. the CPU runs: 2 training maps + 1 held-out, GPU hidden
# --------------------------------------------------------------------------
def _have_dll():
    try:
        from surfgym import SurfCore, default_config
        SurfCore(_bsp(TRAIN[0]), default_config(num_envs=1, lidar_w=0,
                                                 lidar_h=0)).close()
        return True
    except Exception:
        return False


needs_sim = pytest.mark.skipif(POOL is None or not _have_dll(),
                               reason="needs the map pool and a built "
                                      "surfcore library")

COMMON = ["--reward", "race", "--spawn", "platform",
          "--lidar-w", "64", "--lidar-h", "32", "--lidar-cell", "32",
          "--goal-cell", "48", "--heldout-goal-cell", "48",
          "--envs", "8", "--n-steps", "8", "--minibatches", "1",
          "--epochs", "1"]


@needs_sim
def test_cpu_dump_invariants_shows_the_heldout_slot_contributes_nothing(tmp_path):
    p = _train("--run", str(tmp_path / "dry"),
               "--maps", ",".join(_bsp(m) for m in TRAIN),
               "--heldout-maps", _bsp(HELD), *COMMON, "--dump-invariants")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    line = [l for l in p.stdout.splitlines() if l.startswith("{")][-1]
    inv = json.loads(line)
    assert inv["maps"] == TRAIN and inv["heldout"] == [HELD]
    assert HELD not in inv["maps"]
    assert inv["n_heldout"] == 1
    assert inv["heldout_envs"] == 0                    # THE invariant
    assert inv["heldout_in_fleet_slots"] == 0
    assert inv["train_envs"] == inv["envs_per_rank"] == 8
    assert inv["envs_per_slot"] * inv["n_maps"] == inv["train_envs"]
    assert inv["heldout_d0"]["0way"] > 1000.0
    assert "0way" not in inv["map_d0"]
    # the startup banner says what the slot is
    assert "EVAL-ONLY (never trained)" in p.stdout
    # the CSV header already exists: held-out columns LAST, none of the
    # training-block columns for the held-out tag
    head = (tmp_path / "dry" / "progress.csv").read_text(
        encoding="utf-8").splitlines()[0].split(",")
    # The held-out block goes after every column the TRAINING maps write.
    # It is not the tail of the header, though, and has not been since
    # --tick-ms-schedule appended tick/tick_ms after it (baseline b6cc661
    # already fails a head[-4:] pin) - every later feature appends at the
    # end, which is the strict-prefix rule that makes a resumed pre-feature
    # progress.csv migrate instead of breaking. So: a CONTIGUOUS block,
    # positioned after everything tagged with a training map.
    _want = [f"{c}.0way" for c in HELD_COLS]
    _i = head.index(_want[0])
    assert head[_i:_i + len(_want)] == _want, head[_i:_i + len(_want)]
    assert not [h for h in head[:_i] if h.endswith(".0way")]
    assert "race/eval_progress.0way" not in head
    assert "race/map_pct.0way" not in head
    assert "race/eval_progress.bom" in head and "race/map_pct.d" in head
    assert head.index("train/ret_std") < head.index(
        "race/heldout_progress_field.0way")


@needs_sim
def test_cpu_run_evaluates_the_heldout_map_and_keeps_it_out_of_the_aggregate(
        tmp_path):
    out = tmp_path / "tiny"
    p = _train("--run", str(out),
               "--maps", ",".join(_bsp(m) for m in TRAIN),
               "--heldout-maps", _bsp(HELD), *COMMON,
               "--ep-ticks", "300", "--eval-eps", "1", "--eval-greedy-only",
               "--steps", "300", "--record-every", "1e9",
               "--ckpt-every", "1e9", "--eval-stall", "1",
               "--stall-secs", "1", "--respawn-frac", "0.9",
               "--respawn-margin", "2", "--respawn-binned", "1",
               "--respawn-bins", "128")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    # its own greedy eval, its own trajectory, named like a training map's
    trajs = sorted(q.name for q in out.glob("traj_*.jsonl"))
    assert any(t.endswith("_0way.jsonl") for t in trajs), trajs
    assert any(t.endswith("_bom.jsonl") for t in trajs), trajs
    assert "greedy[HELDOUT 0way]" in p.stdout
    assert "HELD-OUT  (never trained" in p.stdout
    # run.json: the list and the cells the recorder will read back
    cfg = json.loads((out / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["maps"] == TRAIN and cfg["heldout_maps"] == [HELD]
    assert cfg["heldout_goal_cells"] == {"0way": 48.0}
    assert cfg["heldout_map_cells"] == {"0way": 32.0}
    assert cfg["heldout_goal_cell"] == "48"
    # the CSV row: held-out values present, aggregate over training maps only
    rows = list(csv.DictReader(open(out / "progress.csv", encoding="utf-8")))
    ev = [r for r in rows if r["race/heldout_pct_field.0way"] != ""]
    assert ev, rows
    r = ev[0]
    assert r["race/heldout_finishes.0way"] == "0"
    assert r["race/heldout_progress_field.0way"] != ""
    trained = [float(r[f"race/map_pct.{t}"]) for t in ("bom", "d")]
    assert float(r["race/map_pct"]) == pytest.approx(
        round(sum(trained) / len(trained), 3), abs=2e-3)
    # the held-out reservoir does not exist: only the two training maps'
    # reservoirs are reported
    assert "heldout" not in r["race/map_pct"]
    # the held-out eval was stall-killed like a training eval (--eval-stall)
    hdr = json.loads(next(open(out / next(
        q for q in out.glob("traj_*_0way.jsonl")), encoding="utf-8")))
    assert hdr.get("eval_stall") == 1
