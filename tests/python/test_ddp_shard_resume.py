"""--ddp-maps shard (mm/ddp): the checkpoint is the UNION of the ranks' per-map state, and a
resume restores each process's own maps by tag - so a run moves between world sizes freely.

Real trainer, CPU (gloo), recipe v14 at smoke scale (tests/python/v14_smoke.py) with
--adv-norm-per-map 1, three maps (2 + 1 on two ranks: unequal shards). A source run trains a
few iterations at world size 2 (and one at 1); every resume runs ZERO iterations (--steps =
the checkpoint's step) and writes its ckpt_final.pt straight from what it restored, so each
round trip must give back every map's state exactly:

    W=2 -> W=2,   W=2 -> W=1,   W=1 -> W=2

Needs the built core (SURFCORE_DLL) and three maps with their six cell-32 caches
(DDP_SHARD_MAPS = comma list of .bsp paths; default the repo's maps/ copies of blue200,
unitfarmer2, celestial). ~10 minutes on the droplet.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v14_smoke as V  # noqa: E402

ROOT = V.ROOT
MAPS = [Path(p) for p in os.environ.get("DDP_SHARD_MAPS", ",".join(
    str(ROOT / "maps" / f"{s}.bsp") for s in ("surf_edgeflow_blue200", "surf_unitfarmer2",
                                               "surf_src_celestial"))).split(",") if p]
needs = pytest.mark.skipif(
    not (V.have_maps(MAPS) and torch.distributed.is_available()
         and torch.distributed.is_gloo_available()),
    reason="needs the built core, three maps with their six cell-32 caches "
           "(DDP_SHARD_MAPS) and gloo")
STEPS = 6144                       # 4 iterations at 48 envs x 8 decisions x 4 ticks

PER_MAP = ("int_counts", "unstuck", "auto_spine_failed", "auto_spine_spines",
           "int_coef_progress", "int_share_max", "respawn_greedy", "respawn")


def _line(steps):
    return V.BASE + V.V14 + V.MCTS + V.smoke(steps, envs=48) + ["--adv-norm-per-map", "1"]


def run(run_name, nproc, extra=(), steps=STEPS, timeout=1800):
    shutil.rmtree(ROOT / "runs" / run_name, ignore_errors=True)
    shutil.rmtree(ROOT / "runs" / f"{run_name}_warm", ignore_errors=True)
    head = ([sys.executable, "-m", "torch.distributed.run", "--standalone",
             f"--nproc-per-node={nproc}"] if nproc > 1 else [sys.executable, "-u"])
    cmd = (head + [str(ROOT / "python" / "train_fast.py")]
           + (["--run-name", run_name, "--ddp-maps", "shard"] if nproc > 1
              else ["--run", run_name])
           + list(extra) + _line(steps))
    env = dict(V.env(), SELF_STATES="1")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT),
                       timeout=timeout, encoding="utf-8", errors="replace")
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    return r, ROOT / "runs" / run_name


def same(a, b, where=""):
    """deep equality of checkpoint payloads (dicts, lists, arrays, tensors, NaN == NaN)"""
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{where}: keys {list(a)} != {list(b)}"
        for k in a:
            same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), f"{where}: length"
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{where}[{i}]")
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{where}: tensor differs"
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, f"{where}: dtype/shape"
        assert a.tobytes() == b.tobytes(), f"{where}: array differs"
    elif isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        pass
    else:
        assert a == b, f"{where}: {a!r} != {b!r}"


def load(d, tag="final"):
    return torch.load(d / f"ckpt_{tag}.pt", map_location="cpu", weights_only=False)


def check_round_trip(src, dst):
    a, b = load(src), load(dst)
    assert a["global_step"] == b["global_step"] == STEPS
    stems = [p.stem for p in MAPS]
    assert a["config"]["maps"] == b["config"]["maps"] == stems
    for k in PER_MAP:
        assert k in a and k in b, k
        assert len(a[k]) == len(MAPS), f"{k}: {list(a[k])} is not every map"
        same(a[k], b[k], k)
    same(a["policy"], b["policy"], "policy")


@pytest.fixture(scope="module")
def sources():
    _, w2 = run("ddpres_w2", 2, extra=["--maps", ",".join(str(p) for p in MAPS)])
    _, w1 = run("ddpres_w1", 1, extra=["--maps", ",".join(str(p) for p in MAPS)])
    yield {"w2": w2, "w1": w1}
    for d in (w2, w1):
        shutil.rmtree(d, ignore_errors=True)


@needs
def test_sharded_checkpoint_is_every_maps_state_in_map_order(sources):
    c = load(sources["w2"])
    tags = list(c["config"]["map_shard"])
    assert c["config"]["ddp_maps"] == "shard" and c["config"]["world_size"] == 2
    assert c["config"]["map_shard"] == dict(zip(tags, [0, 1, 0]))     # round-robin
    assert c["config"]["envs_per_rank"] == [32, 16] and c["config"]["envs"] == 48
    for k in PER_MAP:
        keys = list(c[k])
        assert keys in (tags, [p.stem for p in MAPS]), (k, keys)       # the run's map order
    w1 = load(sources["w1"])
    assert list(w1["unstuck"]) == list(c["unstuck"])                    # one process's layout


@needs
@pytest.mark.parametrize("src,nproc", [("w2", 2), ("w2", 1), ("w1", 2)])
def test_resume_restores_every_map_at_any_world_size(sources, src, nproc):
    name = f"ddpres_{src}_to_w{nproc}"
    # (--maps given again, the same list in the same order: a checkpoint names map STEMS, which
    # resolve under maps/ - these test copies live elsewhere)
    r, d = run(name, nproc, extra=["--ckpt", str(sources[src] / "ckpt_final.pt"),
                                   "--maps", ",".join(str(p) for p in MAPS)],
               steps=STEPS)
    try:
        assert f"resumed {sources[src] / 'ckpt_final.pt'} at step {STEPS:,}" in r.stdout
        check_round_trip(sources[src], d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
