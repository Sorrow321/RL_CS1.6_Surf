"""--eval-parallel (2026-10-21, autoresearch r4b4): the greedy eval's episodes run side by
side on an --eval-eps-env eval core (surfgym.record.record_rollout_parallel) instead of one
after another on one env.

Why: on sidistic the sequential eval took 45-49% of the box's wall clock once greedy
episodes lived to the 300 s cap (r3b3_gwx2: 324k steps/s while training, 168k overall).

(a) The recorder on the real core: episode i of a parallel recording is, line for line,
    the sequential recording of one episode from env i's spawn state (same policy, same
    physics), and the trailers / summaries match.
(b) The greedy-line pool's per-env hook: env i's states until ITS episode ends (the
    autoreset tail is ignored), so build() keeps exactly what the sequential hook keeps
    from the same episodes; the capture is the recorded trajectory row for row.
(c) The trainer (CPU, sidistic): evals write --eval-eps episodes from an --eval-eps-env
    core, the greedy-line pool is built from them, the config carries the flag, a bare
    resume restores it, record_ckpt reads the checkpoint; the refusals.
(d) Flag off: the trainer before this change and after it give the same config,
    progress.csv (fps aside), eval trajectory bytes and weights on the same smoke.
"""
from __future__ import annotations

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
from surfgym.greedy_spawn import GreedyLinePool                    # noqa: E402

from test_ez_viewc import (SID, SMOKE, TRAIN, _csv, _run, _train,   # noqa: E402
                           needs_run)


def _core(n, pool=None, ticks=240):
    from surfgym import SurfCore, default_config
    # no spawn yaw jitter: it is drawn from the env's own RNG, so env i of an n-env core
    # and the only env of a 1-env core would start 5 deg apart from the same pool state
    kw = dict(num_envs=n, max_episode_ticks=ticks, sv_maxvelocity=4000.0, lidar_w=0,
              lidar_h=0, yaw_jitter_deg=0.0)
    if pool is not None:
        kw["spawn_mode"] = 2
    c = SurfCore(str(SID), default_config(**kw))
    if pool is not None:
        c.set_spawn_pool(pool)
    return c


def _pool(k=6):
    """k distinct spawn states: the map's own spawn, each with its own heading and speed."""
    c = _core(1)
    c.reset(0)
    s0 = c.get_states()[0].copy()
    rows = np.repeat(s0[None], k).astype(STATE_DTYPE)
    ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    spd = np.linspace(200.0, 900.0, k)
    rows["velocity"][:, 0] = (spd * np.cos(ang)).astype(np.float32)
    rows["velocity"][:, 1] = (spd * np.sin(ang)).astype(np.float32)
    rows["yaw"] = np.degrees(ang).astype(np.float32)
    return rows


class _Pol:
    """A deterministic function of each env's OWN observation row, so env i of an n-env
    core and the only env of a 1-env core act alike from the same state."""

    def act(self, obs):
        o = np.asarray(obs, np.float64)
        a = np.zeros((o.shape[0], 6), np.int32)
        a[:, 0] = (np.floor(np.abs(o[:, 0]) * 997.0).astype(np.int64)
                   + np.floor(np.abs(o[:, 1]) * 113.0).astype(np.int64)) % 15
        a[:, 1] = 7
        a[:, 2] = 2
        a[:, 3] = np.where(o[:, 2] > 0.0, 2, 0)
        a[:, 4] = (np.floor(np.abs(o[:, 3]) * 31.0).astype(np.int64) % 5 == 0)
        return a


def _episodes(path):
    """[(header, [row lines], trailer)] as raw strings / dicts."""
    out, hdr, rows = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("{"):
            d = json.loads(line)
            if "end" in d and hdr is not None:
                out.append((hdr, rows, d))
                hdr, rows = None, []
            else:
                hdr, rows = d, []
            continue
        rows.append(line)
    return out


# ==========================================================================
# (a) the recorder
# ==========================================================================
@needs_run
def test_each_parallel_episode_is_the_sequential_episode_from_the_same_spawn(tmp_path):
    from surfgym.record import record_rollout, record_rollout_parallel
    pool = _pool(6)
    n, seed = 6, 11
    twin = _core(n, pool)
    twin.reset(seed)
    spawns = twin.get_states().copy()          # env i's spawn draw under this seed
    pp = tmp_path / "par.jsonl"
    summ = record_rollout_parallel(_core(n, pool), _Pol(), pp, episodes=n, seed=seed,
                                  header_extra={"eval_stall": 0})
    par = _episodes(pp)
    assert len(par) == len(summ) == n
    ends = set()
    for i in range(n):
        sp = tmp_path / f"seq{i}.jsonl"
        ssum = record_rollout(_core(1, spawns[i:i + 1]), _Pol(), sp, episodes=1,
                              max_ticks=10_000, seed=seed, header_extra={"eval_stall": 0})
        (h1, r1, t1), = _episodes(sp)
        h0, r0, t0 = par[i]
        assert h0["episode"] == i and h1["episode"] == 0
        h0 = dict(h0, episode=0)
        assert h0 == h1
        assert r0 == r1, f"episode {i}: rows differ"
        assert t0 == t1 and summ[i] == ssum[0]
        ends.add(t0["end"])
    # the spawns differ, so the episodes do too
    assert len({tuple(e[1][-1:]) for e in par}) > 1
    assert ends <= {"fail", "trunc", "done"}


@needs_run
def test_a_wall_budget_truncates_the_open_episodes(tmp_path):
    from surfgym.record import record_rollout_parallel
    pp = tmp_path / "par.jsonl"
    summ = record_rollout_parallel(_core(3, _pool(3), ticks=5000), _Pol(), pp,
                                   episodes=2, max_ticks=50, seed=1)
    eps = _episodes(pp)
    assert len(eps) == 2 and [e[0]["episode"] for e in eps] == [0, 1]
    for (h, rows, tr), s in zip(eps, summ):
        assert len(rows) == tr["ticks"] <= 50
        if tr["ticks"] == 50:
            assert tr["end"] == "trunc"
        assert s["end"] == tr["end"]
    with pytest.raises(ValueError):
        record_rollout_parallel(_core(2, _pool(2)), _Pol(), pp, episodes=3)


# ==========================================================================
# (b) the greedy-line pool's per-env hook
# ==========================================================================
def _ep(n, tag):
    rows = np.zeros(n, STATE_DTYPE)
    rows["origin"][:, 0] = tag * 1e5 + np.arange(n)
    rows["tick"] = np.arange(n) + 7
    rows["stuck_ticks"] = 3
    return rows


def test_the_per_env_hook_keeps_what_the_sequential_hook_keeps():
    eps = [_ep(900, 1), _ep(300, 2), _ep(40, 3), _ep(700, 4)]
    summ = [{"end": "fail"}, {"end": "trunc"}, {"end": "fail"}, {"end": "done"}]
    seq = GreedyLinePool(0.2, window=(0.5, 5.0))
    hook = seq.hook()
    t = 0
    for ep in eps:
        for i in range(len(ep)):
            hook(t, ep[i:i + 1], np.zeros(1), np.array([i == len(ep) - 1]),
                 np.array([False]))
            t += 1
    par = GreedyLinePool(0.2, window=(0.5, 5.0))
    hp = par.hook_parallel(4)
    L = max(len(e) for e in eps) + 25
    for t in range(L):
        st = np.zeros(4, STATE_DTYPE)
        done = np.zeros(4, bool)
        for k, ep in enumerate(eps):
            if t < len(ep):
                st[k] = ep[t]
                done[k] = t == len(ep) - 1
            else:
                st[k]["origin"][0] = -1.0          # the autoreset tail: must be ignored
        hp(t, st, np.zeros(4), done, np.zeros(4, bool))

    def d_fn(o):
        return 1e6 - np.asarray(o)[:, 0] % 1e5    # progress grows along each episode
    a = seq.build(summ, tick_ms=10.0, d_fn=d_fn)
    b = par.build(summ, tick_ms=10.0, d_fn=d_fn)
    assert a == b and seq.size == par.size > 0
    assert np.array_equal(seq.rows, par.rows)
    assert (par.rows["origin"][:, 0] >= 0).all()


@needs_run
def test_the_per_env_capture_is_the_recorded_trajectory_row_for_row(tmp_path):
    from surfgym.record import record_rollout_parallel
    p = GreedyLinePool(0.2, window=(0.0, 1.0))
    path = tmp_path / "t.jsonl"
    hp = p.hook_parallel(3)
    summ = record_rollout_parallel(_core(3, _pool(3)), _Pol(), path, episodes=3, seed=4,
                                   on_tick=hp)
    cur, fin = p._par
    assert all(f is not None for f in fin)
    for ep, tr, (h, rows, t) in zip(fin, summ, _episodes(path)):
        js = np.asarray([json.loads(r) for r in rows])
        assert len(ep) == tr["ticks"] == len(js)
        np.testing.assert_allclose(ep["origin"], js[:, 1:4], atol=0.006)
        np.testing.assert_allclose(ep["velocity"], js[:, 4:7], atol=0.006)


# ==========================================================================
# (c) the trainer
# ==========================================================================
def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


PAR_SMOKE = ["--respawn-reservoir", "5000", "--record-every", "4096", "--eval-eps", "3",
             "--eval-parallel", "1", "--unstuck", "--unstuck-reach", "eval",
             "--obs-compass", "1", "--obs-surf", "1", "--surf-bonus", "0",
             "--obs-pot-rate", "1", "--race-ratchet", "--time-pen", "0",
             "--ep-ticks", "600"]


@needs_run
def test_the_trainer_runs_parallel_evals_builds_the_greedy_pool_and_resumes():
    import torch
    from test_respawn_greedy import _seeded_ckpt
    d0, seeded = _seeded_ckpt("evpar_base")
    run = "evpar_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + PAR_SMOKE + ["--steps", "16384", "--respawn-greedy", "0.25"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    out = r.stdout
    assert "--eval-parallel: the greedy eval's 3 episodes run side by side" in out
    built = re.findall(r"greedy-line pool: ([0-9,]+) own states from (\d+)/(\d+) failed", out)
    assert built and all(int(e) == 3 for _, _, e in built), built
    trajs = sorted((ROOT / "runs" / run).glob("traj_*.jsonl"))
    trajs = [t for t in trajs if re.fullmatch(r"traj_\d+\.jsonl", t.name)]
    assert len(trajs) >= 2
    for t in trajs:
        eps = _episodes(t)
        assert [e[0]["episode"] for e in eps] == [0, 1, 2]
        for h, rows, tr in eps:
            assert len(rows) == tr["ticks"] <= 600 and tr["end"] in ("fail", "trunc", "done")
    assert _cfg(run)["eval_parallel"] == 1
    ck = torch.load(ROOT / "runs" / run / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    assert ck["respawn_greedy"]["rows"].dtype == STATE_DTYPE
    # a bare resume restores the flag
    run2 = "evpar_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE
              + ["--record-every", "4096", "--eval-eps", "3", "--steps", "20480"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "eval_parallel=1" in r2.stdout
    assert "--eval-parallel: the greedy eval's 3 episodes" in r2.stdout
    assert _cfg(run2)["eval_parallel"] == 1
    # the recorder reads the checkpoint (the record gate's own backend)
    rec = ROOT / "runs" / run / "rec_par.jsonl"
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(rec),
               "--episodes", "1", "--ep-ticks", "300"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--eval-stall", "1"], "--eval-parallel is not implemented with --eval-stall"),
    (["--eval-mirror", "1"], "--eval-parallel is not implemented with --eval-mirror"),
])
def test_untested_eval_paths_are_refused(extra, msg):
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "evpar_bad"] + SMOKE
             + ["--steps", "2048", "--eval-parallel", "1"] + extra)
    shutil.rmtree(ROOT / "runs" / "evpar_bad", ignore_errors=True)
    assert r.returncode != 0 and msg in r.stdout + r.stderr, r.stdout[-2000:]


# ==========================================================================
# (d) flag off
# ==========================================================================
def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this change."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--eval-parallel" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    pre = ROOT / "python" / "_train_fast_pre_evpar.py"
    extra = ["--record-every", "2048", "--eval-eps", "2"]
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("evpar_off_new", extra)
        _train("evpar_off_old", extra, script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "evpar_off_new", ROOT / "runs" / "evpar_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "eval_parallel" not in ca
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
