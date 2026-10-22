"""--respawn-backward-mix F (2026-10-22, autoresearch r6b2): goal-anchored rows MIXED into the
reservoir pool next to the --respawn-novel rows.

The --respawn-backward sampler (the map's own geodesic field, no demo: distance to the goal in
[floor, start x d0], speed U(lo, hi) along the field's descent) supplies F of the non-start rows,
placed right after the novel quota and before the --respawn-greedy tail, at a FIXED band. Their
episodes are tagged through the novel sampler's keys, so they are never harvested, mark no visits,
never lower the novel ceiling and earn no novelty under --int-real-only; they are also counted on
their own (ended / finished).

(a) The tags (NovelSpawnSampler on sidistic): a backward row's episode is tagged novel AND
    backward, the outcome count reads the ended backward episodes before the retag, the keys live
    for the live and the previous pool, and without backward keys nothing changes.
(b) The rows (BackwardSpawnSampler on sidistic at 5% of d0): inside the band, at the asked speed,
    rising no steeper than the sampler's clamp.
(c) The trainer (CPU, sidistic, a seeded reservoir): the startup line, the step line and the
    bmix/* columns, the config key and a bare resume; the refusals; flag off is bit-identical to
    the trainer before the change on the same seeded checkpoint with --respawn-novel on.
"""
from __future__ import annotations

import csv
import json
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
from surfgym.respawn import BackwardSpawnSampler, NovelSpawnSampler  # noqa: E402

from test_ez_viewc import SID, SID_FIELD, SMOKE, TRAIN, _run, _train, needs_run  # noqa: E402


def _field():
    z = np.load(SID_FIELD, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def _core(n=4):
    from surfgym import SurfCore, default_config
    return SurfCore(str(SID), default_config(num_envs=n, spawn_mode=2, max_episode_ticks=200,
                                             sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))


def _rows(xs):
    r = np.zeros(len(xs), STATE_DTYPE)
    r["origin"][:, 0] = np.asarray(xs, np.float32)
    r["origin"][:, 1] = 7.0
    r["origin"][:, 2] = 11.0
    return r


@needs_run
def test_backward_rows_are_tagged_novel_and_counted_on_their_own():
    fld, core = _field(), _core(4)
    nov = NovelSpawnSampler(core, fld, 4, d0=97768.0)
    pool = _rows([1.0, 2.0, 3.0, 4.0])
    nov.mix(pool, 1, 0, None)                    # no novel rows this pool
    nov.add_back_keys(_rows([101.0, 103.0]))
    spawned = _rows([101.0, 5.0, 103.0, 6.0])     # envs 0 and 2 from backward rows
    ended = np.ones(4, bool)
    nov.note_spawns(spawned, ended)
    assert nov.novel_env.tolist() == [True, False, True, False]
    assert nov.back_env.tolist() == [True, False, True, False]
    assert nov.back_spawned == 2
    # the outcome count reads the ENDED backward episodes before the retag
    nov.note_back_outcomes(np.array([True, True, False, False]),
                           np.array([True, False, False, False]))
    assert nov.pop_back() == (1, 1) and nov.pop_back() == (0, 0)
    nov.note_back_outcomes(np.array([False, False, True, True]), np.zeros(4, bool))
    assert nov.pop_back() == (1, 0)
    # the keys live for the live and the previous pool: a later pool without backward rows
    # still recognises the previous one, the one after that does not
    nov.mix(pool, 1, 0, None)
    nov.add_back_keys(np.zeros(0, STATE_DTYPE))
    nov.note_spawns(spawned, ended)
    assert nov.back_env.tolist() == [True, False, True, False]
    nov.mix(pool, 1, 0, None)
    nov.add_back_keys(np.zeros(0, STATE_DTYPE))
    nov.note_spawns(spawned, ended)
    assert nov.back_env.tolist() == [False] * 4 and nov.novel_env.tolist() == [False] * 4


@needs_run
def test_without_backward_keys_the_novel_tags_are_unchanged():
    fld, core = _field(), _core(4)
    nov = NovelSpawnSampler(core, fld, 4, d0=97768.0)
    nov.mix(_rows([1.0, 2.0, 3.0, 4.0]), 1, 0, None)
    nov._keys = frozenset({_rows([5.0])["origin"][0].tobytes()})   # one novel row
    nov.note_spawns(_rows([5.0, 6.0, 5.0, 7.0]), np.ones(4, bool))
    assert nov.novel_env.tolist() == [True, False, True, False]
    assert not nov.back_env.any() and nov.back_spawned == 0
    nov.note_back_outcomes(np.ones(4, bool), np.ones(4, bool))
    assert nov.pop_back() == (0, 0)


@needs_run
def test_backward_rows_sit_in_the_band_at_the_asked_speed():
    fld, core = _field(), _core(1)
    d0 = 97768.0
    b = BackwardSpawnSampler(core, fld, d0, start_frac=0.05, speed=(1000.0, 2500.0),
                             floor=256.0, shell_frac=0.0, seed=5)
    rows = b.sample_states(300, None)
    assert len(rows) >= 250, len(rows)
    d = np.asarray(fld.sample(rows["origin"].astype(np.float32)), np.float64)
    assert d.min() >= 256.0 - 1.0 and d.max() <= 0.05 * d0 + 32.0, (d.min(), d.max())
    spd = np.linalg.norm(rows["velocity"].astype(np.float64), axis=1)
    assert spd.min() >= 1000.0 - 1e-3 and spd.max() <= 2500.0 + 1e-3
    elev = np.degrees(np.arcsin(rows["velocity"][:, 2] / spd))
    assert elev.max() <= 30.0 + 1e-3 and elev.min() >= -60.0 - 1e-3
    # a FIXED band: nothing moves W without observe / advance
    assert b.W == pytest.approx(0.05 * d0)


# ==========================================================================
# (c) the trainer
# ==========================================================================
def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


def _csv(run):
    with open(ROOT / "runs" / run / "progress.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


NOVEL_SMOKE = ["--respawn-novel", "--respawn-novel-frac", "0.3", "--respawn-novel-tries", "4"]


@needs_run
def test_the_trainer_mixes_logs_saves_restores_and_refuses():
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("bmix_base")
    run = "bmix_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + NOVEL_SMOKE
             + ["--steps", "10240", "--respawn-backward-mix", "0.2",
                "--respawn-backward-shell", "0", "--respawn-greedy", "0.3"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    out = r.stdout
    assert "respawn BACKWARD-MIX: 20% of the non-start pool" in out
    rows = [int(m.replace(",", "")) for m in re.findall(r"  bmix rows ([0-9,]+) fin", out)]
    want = int(round((4096 - 410) * 0.2))
    assert rows and max(rows) == want, (rows, want)
    assert re.search(r"backward: W [0-9,]+u = 5\.0% of d0", out), out[-3000:]
    cfg = _cfg(run)
    assert cfg["respawn_backward_mix"] == 0.2 and cfg["respawn_backward_shell"] == 0.0
    rr = _csv(run)
    assert rr and "bmix/rows" in rr[0] and "bmix/finished" in rr[0]
    assert max(int(x["bmix/rows"] or 0) for x in rr) == want
    # a bare resume restores the flag
    run2 = "bmix_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "14336"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_backward_mix=0.2" in r2.stdout
    assert "respawn BACKWARD-MIX: 20%" in r2.stdout
    assert _cfg(run2)["respawn_backward_mix"] == 0.2
    # the recorder reads the checkpoint (the record gate's own backend)
    rec = ROOT / "runs" / run / "rec_greedy.jsonl"
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(rec),
               "--episodes", "1", "--ep-ticks", "300"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--respawn-backward-mix", "0.2"], "--respawn-backward-mix needs --respawn-novel"),
    (["--respawn-backward-mix", "1.2", "--respawn-novel"],
     "--respawn-backward-mix F needs 0 < F < 1"),
    (["--respawn-backward-mix", "0.2", "--respawn-novel", "--respawn-backward"],
     "--respawn-backward-mix is exclusive with --respawn-backward"),
])
def test_bad_settings_are_refused(extra, msg):
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "bmix_bad"] + SMOKE
             + ["--steps", "2048"] + extra)
    shutil.rmtree(ROOT / "runs" / "bmix_bad", ignore_errors=True)
    assert r.returncode != 0 and msg in r.stdout + r.stderr, r.stdout[-2000:]


def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this change."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--respawn-backward-mix" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    pre = ROOT / "python" / "_train_fast_pre_bmix.py"
    d0, seeded = _seeded_ckpt("bmix_off_base")
    runs = {"bmix_off_new": TRAIN, "bmix_off_old": pre}
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in runs.items():
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run, "--ckpt", str(seeded)]
                     + SMOKE + GREEDY_SMOKE + NOVEL_SMOKE + ["--steps", "10240"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "bmix_off_new", ROOT / "runs" / "bmix_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "respawn_backward_mix" not in ca
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
    for dd in (d0, a, b):
        shutil.rmtree(dd, ignore_errors=True)
