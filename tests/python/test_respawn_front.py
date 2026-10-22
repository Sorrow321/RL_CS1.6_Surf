"""--respawn-front F (2026-10-22, autoresearch r6b3): a share of the training spawn pool drawn from
the approach to the deepest places REAL training episodes reach - the seconds before each new
episode record within --respawn-front-band x d0 of the ceiling, kept even when the episode dies
right after (surfgym.frontier_rows.FrontierRowPool).

(a) The pool, on synthetic states: a real record inside the band after a full window keeps the
    HI..LO window (oldest first, clocks zeroed); no capture without a full window, for a non-real
    env, outside the band, or inside the cooldown; an episode end restarts the ring; the FIFO cap;
    mix writes the block right before the greedy tail and nothing below min_rows; checkpoint round
    trip; refusals; the recorder lists the flags as training-only.
(b) The trainer (CPU, sidistic): the flag reaches the pool (startup line, step line, config) and a
    bare resume restores it; it is refused without the reservoir.
(c) Flag off: the trainer before this change and after it, with --respawn-greedy on, give the same
    config, progress.csv (fps aside), eval trajectory bytes and weights.
"""
from __future__ import annotations

import json
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
from surfgym.frontier_rows import FrontierRowPool                  # noqa: E402

from test_ez_viewc import SMOKE, TRAIN, _csv, _run, needs_run       # noqa: E402

# 10 ms ticks, a snapshot every 4 ticks = 25 per second: HI 0.4 s = 10 snapshots back
TICK, EVERY = 10.0, 4


def _pool(n=3, **kw):
    kw = dict(dict(frac=0.2, band=0.01, d0=10000.0, tick_ms=TICK, every=EVERY,
                   window=(0.0, 0.4), cooldown=0.2, cap=1000, min_rows=5), **kw)
    return FrontierRowPool(n, **kw)


class _Drive:
    """Feeds the pool tick by tick: env e's origin x = its own snapshot counter, tagged by e,
    and d(x) from a per-env schedule."""

    def __init__(self, pool, n):
        self.p, self.n = pool, n
        self.k = np.zeros(n, np.int64)              # ticks into the current episode
        self.dsched = [lambda k: 5000.0] * n
        self.real = np.ones(n, bool)

    def tick(self, ended=None, ceiling=None):
        ended = np.zeros(self.n, bool) if ended is None else np.asarray(ended, bool)
        self.k[ended] = 0
        st = np.zeros(self.n, STATE_DTYPE)
        st["origin"][:, 0] = self.k
        st["origin"][:, 1] = np.arange(self.n)
        st["tick"] = self.k + 3
        st["stuck_ticks"] = 2
        d = np.array([self.dsched[e](self.k[e]) for e in range(self.n)])
        out = self.p.observe(st, lambda o: d, self.real, ended, ceiling=ceiling)
        self.k += 1
        return out


def test_a_record_in_the_band_keeps_the_window_before_it():
    p = _pool()
    dr = _Drive(p, 3)
    # env 0 dives: d = 5,000 - 80 k enters the band (ceiling 1,000 + 0.01 x 10,000) at k 49; the
    # snapshots fall on k = 3, 7, ... (every 4th tick), so the first one inside is k 51
    dr.dsched[0] = lambda k: 5000.0 - 80.0 * k
    for _ in range(51):
        dr.tick(ceiling=1000.0)
    assert p.size == 0
    dr.tick(ceiling=1000.0)
    assert p.captured == 1 and p.size == 11
    assert (p.rows["origin"][:, 1] == 0).all()          # env 0's states only


def test_capture_window_contents():
    p = _pool(n=1, cooldown=10.0)
    dr = _Drive(p, 1)
    dr.dsched[0] = lambda k: 2000.0 if k < 80 else 1050.0 - k   # in the band from k 80
    for _ in range(84):                                  # the snapshot at k 83 is the record
        dr.tick(ceiling=1000.0)
    assert p.captured == 1 and p.size == 11
    x = p.rows["origin"][:, 0].astype(int).tolist()
    assert x == list(range(43, 84, 4))       # HI 0.4 s = 10 snapshots back to the record, oldest first
    assert (p.rows["tick"] == 0).all() and (p.rows["stuck_ticks"] == 0).all()
    assert "11 own states (1 captures" in p.line()


def test_no_capture_without_full_window_non_real_outside_band_or_in_cooldown():
    # spawned inside the band: the first full window comes 10 snapshots later, and only a NEW
    # record after it captures (d keeps falling here, so it does - once, then the cooldown)
    p = _pool(n=1, cooldown=10.0)
    dr = _Drive(p, 1)
    dr.dsched[0] = lambda k: 1050.0 - 0.01 * k
    for _ in range(36):                      # 9 snapshots: no full window yet
        dr.tick(ceiling=1000.0)
    assert p.captured == 0
    for _ in range(8):
        dr.tick(ceiling=1000.0)
    assert p.captured == 1
    for _ in range(200):                     # cooldown 10 s: nothing more
        dr.tick(ceiling=1000.0)
    assert p.captured == 1
    # non-real
    q = _pool(n=1)
    dq = _Drive(q, 1)
    dq.real[:] = False
    dq.dsched[0] = lambda k: 2000.0 - 10.0 * k
    for _ in range(200):
        dq.tick(ceiling=1000.0)
    assert q.captured == 0
    # outside the band (ceiling 1,000 + 0.01 x 10,000 = 1,100; d stays >= 1,200)
    r = _pool(n=1)
    drr = _Drive(r, 1)
    drr.dsched[0] = lambda k: max(1200.0, 3000.0 - 10.0 * k)
    for _ in range(400):
        drr.tick(ceiling=1000.0)
    assert r.captured == 0


def test_an_episode_end_restarts_the_ring_and_the_record():
    p = _pool(n=1, cooldown=0.04)
    dr = _Drive(p, 1)
    dr.dsched[0] = lambda k: 1050.0 - k          # in the band, falling
    for _ in range(44):
        dr.tick(ceiling=1000.0)
    c0 = p.captured
    assert c0 >= 1
    dr.tick(ended=[True], ceiling=1000.0)        # new episode: ring and record restart
    for _ in range(36):                          # 9 more snapshots: no full window
        dr.tick(ceiling=1000.0)
    assert p.captured == c0


def test_fifo_cap_and_mix_block_and_min_rows():
    p = _pool(n=1, cooldown=0.04, cap=30, min_rows=25)
    dr = _Drive(p, 1)
    dr.dsched[0] = lambda k: 1050.0 - k
    pool = np.zeros(100, STATE_DTYPE)
    pool["origin"][:, 2] = np.arange(100)
    for _ in range(44):
        dr.tick(ceiling=1000.0)
    assert 0 < p.size < 25
    out = p.mix(pool, 10, 20)
    assert out is pool and p.mixed == 0          # below min_rows: untouched
    for _ in range(400):
        dr.tick(ceiling=1000.0)
    assert p.size == 30 and p.added > 30         # FIFO at the cap
    out = p.mix(pool, 10, 20)
    n = int(round(90 * 0.2))
    assert p.mixed == n == 18
    # head (start rows + the rest) and the greedy tail untouched; the block before the tail drawn
    assert out["origin"][:62, 2].tolist() == list(range(62))
    assert out["origin"][80:, 2].tolist() == list(range(80, 100))
    blk = out[62:80]
    assert np.isin(blk["origin"][:, 0], p.rows["origin"][:, 0]).all()
    assert (blk["origin"][:, 2] == 0).all()


def test_checkpoint_round_trip_and_refusals():
    p = _pool(n=1, cooldown=0.04)
    dr = _Drive(p, 1)
    dr.dsched[0] = lambda k: 1050.0 - k
    for _ in range(100):
        dr.tick(ceiling=1000.0)
    q = _pool(n=1)
    assert q.load_state_dict(p.state_dict())
    assert q.rows.tobytes() == p.rows.tobytes() and q.captured == p.captured
    assert not q.load_state_dict({"rows": np.zeros(3, np.float32)})
    for kw in (dict(frac=0.0), dict(frac=1.0), dict(band=0.0), dict(window=(2.0, 1.0)),
               dict(window=(-1.0, 1.0)), dict(d0=0.0)):
        with pytest.raises(ValueError):
            _pool(**kw)


def test_the_recorder_treats_the_flags_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    for k in ("respawn_front", "respawn_front_band", "respawn_front_window"):
        assert f'"{k}"' in src


# ==========================================================================
# (b) the trainer
# ==========================================================================
def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_wires_restores_and_refuses_it():
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("front_base")
    run = "front_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + GREEDY_SMOKE + ["--steps", "10240", "--respawn-greedy", "0.25",
                                       "--respawn-front", "0.1", "--respawn-front-band", "0.02"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "respawn FRONT: 10% of the non-start pool" in r.stdout
    assert "  front " in r.stdout
    cfg = _cfg(run)
    assert cfg["respawn_front"] == 0.1 and cfg["respawn_front_band"] == 0.02
    assert cfg["respawn_front_window"] == [0.0, 2.0]
    run2 = "front_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "14336"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_front=0.1" in r2.stdout
    assert _cfg(run2)["respawn_front_band"] == 0.02
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "front_bad"] + SMOKE
              + ["--steps", "2048", "--respawn-frac", "0", "--respawn-front", "0.1"])
    assert r3.returncode != 0 and "--respawn-front needs the reservoir" in r3.stdout + r3.stderr
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2, ROOT / "runs" / "front_bad"):
        shutil.rmtree(dd, ignore_errors=True)


# ==========================================================================
# (c) flag off
# ==========================================================================
def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--respawn-front\"" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    pre = ROOT / "python" / "_train_fast_pre_front.py"
    d0, seeded = _seeded_ckpt("front_off_base")
    runs = {"front_off_new": TRAIN, "front_off_old": pre}
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in runs.items():
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run, "--ckpt", str(seeded)]
                     + SMOKE + GREEDY_SMOKE + ["--steps", "14336", "--respawn-greedy", "0.25"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "front_off_new", ROOT / "runs" / "front_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "respawn_front" not in ca
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
    assert "respawn_front" not in sa
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    for dd in (d0, a, b):
        shutil.rmtree(dd, ignore_errors=True)
