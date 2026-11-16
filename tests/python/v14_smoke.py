"""Recipe v14 at CPU smoke scale, for the multi-map build's identity tests (mm/s1, 2026-11-16).

V14 is the incumbent recipe's 160 tokens as launched on princessburglar (ab/share fea35b06;
--maxvel 3500, the unitfarmer2 / princessburglar value), BASE the tools/run_arm.sh SCRATCH pins
in front of it, MCTS the launch extra. SMOKE shrinks only sizes (envs, lidar, network, rollout)
and, for identity runs, keeps the search from starting (--auto-spine-delay 1e12): the async
search is timing-dependent, so a run with one is not reproducible byte for byte.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V14 = [
    "--envs", "1024", "--obs-normal", "1", "--simba", "1", "--ep-ticks", "30000", "--unstuck",
    "--respawn-margin", "2", "--unstuck-temp-heads", "keys", "--obs-surf", "1", "--surf-bonus",
    "0", "--obs-compass", "1", "--obs-pot-rate", "1", "--race-ratchet", "--race-no-corner-cut",
    "1", "--time-pen", "0.001", "--stall-eps", "5", "--pitch-sigma-min", "0.05",
    "--eval-patience", "30", "--eval-parallel", "1", "--front-temp", "1", "--front-band", "0.01",
    "--energy-pot", "3", "--energy-pot-front", "0.01", "--energy-pot-front-bonus", "1",
    "--maxvel", "3500", "--auto-spine", "--auto-spine-scratch", "--auto-spine-async",
    "--auto-spine-recheck", "--auto-spine-banked", "--auto-spine-explorer", "mcts",
    "--auto-spine-mcts-minutes", "6", "--auto-spine-envs", "2048", "--auto-spine-ge-minutes", "5",
    "--auto-spine-patience", "3e8", "--auto-spine-retry", "3e8", "--demo-success-d", "100",
    "--demo-window", "5", "--demo-grow", "5", "--demo-grow-max", "40", "--demo-cooldown", "5",
    "--demo-rate", "0.2", "--demo-min-ep", "50", "--demo-front-frac", "0.5", "--auto-spine-args",
    "--ge-cell-heading 0 --ge-view-follow none --ge-boost 1.0;E1.0 --ge-held 1 --ge-cell-contact 1 --mc-envs 4096 --mc-keys-temp 1.5 --end-survive-secs 1.0",
    "--auto-spine-current-ref", "--auto-spine-patience-tau", "--auto-spine-finish-stop",
    "--demo-match-band", "--int-coef", "2.5", "--int-view", "0", "--int-speed", "8",
    "--int-speed-weight", "3", "--int-speed-weight-up", "--respawn-greedy", "0.25",
    "--respawn-greedy-stuck", "200", "--respawn-greedy-window", "0.25", "2.0",
    "--respawn-greedy-speed", "1.0", "1.3", "--respawn-greedy-turn", "180",
    "--respawn-greedy-temp", "1.0", "--respawn-greedy-temp-ticks", "100",
    "--respawn-front-perturb", "0.25", "--respawn-front-perturb-band", "0.005",
    "--respawn-front-perturb-speed", "1.0", "1.3", "--respawn-front-perturb-turn", "90",
    "--unstuck-max", "0.5", "--unstuck-patience", "2e7", "--unstuck-period", "2e7",
    "--unstuck-reach", "res", "--yaw-sigma-max", "0.2", "--tick-ms", "7.63",
    "--auto-spine-archive-bar", "--auto-spine-archive-ban", "128", "--auto-spine-failed-fallback",
    "--auto-spine-dead-end-ban", "--auto-spine-keys", "--int-coef-progress", "2.5:0.25:0.1",
    "--demo-rehearse", "0.25", "--demo-rehearse-r", "256", "--auto-spine-delay", "3e8",
    "--auto-spine-patience-max", "1e9", "--demo-front-temp", "0.5", "--demo-front-yaw-abs",
    "0.066", "--demo-front-ticks", "40", "--auto-spine-energy", "150", "--demo-success-bonus",
    "50", "--demo-success-bonus-energy-only", "--int-share-max", "0.25",
]
assert len(V14) == 160
MCTS = ["--auto-spine-mcts-args",
        "--mcts-mix-rule sib --mcts-mix-floor 0.05 --mcts-mix-horizon 256 --mcts-mix-warm 2"]
BASE = ["--reward", "race", "--envs", "2048", "--spawn", "platform", "--lidar-w", "64",
        "--lidar-h", "32", "--lidar-cell", "32", "--lidar-range", "11500", "--lidar-near", "2000",
        "--emb", "512", "--hidden", "448", "--act-every", "4", "--pitch-rate", "1.33",
        "--teleport-fail", "--lr", "3e-4", "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2",
        "--vf", "0.5", "--ent", "0.005", "--n-steps", "128", "--epochs", "4", "--minibatches", "16",
        "--ep-ticks", "12000", "--time-pen", "0.005", "--success-bonus", "50", "--finish-k", "0",
        "--stall-secs", "15", "--race-dist", "geodesic", "--maxvel", "4000", "--train-stride", "1",
        "--yaw-adaptive", "--respawn-frac", "0.9", "--respawn-margin", "10",
        "--respawn-reservoir", "100000", "--int-coef", "0.25", "--int-view", "8", "--int-speed",
        "3", "--steps", "60000000000", "--ckpt-every", "1e9", "--record-every", "75e6",
        "--eval-eps", "9", "--eval-greedy-only", "--view-continuous", "--view-absolute",
        "velocity", "--keys-hold", "--obs-potential", "norm", "--obs-potential-curtain"]


def smoke(steps=12288, envs=64, search=False):
    s = ["--envs", str(envs), "--lidar-w", "16", "--lidar-h", "8", "--emb", "64", "--hidden",
         "64", "--n-steps", "8", "--epochs", "1", "--minibatches", "2",
         "--respawn-reservoir", "1000", "--steps", str(int(steps)), "--record-every", "2048",
         "--eval-eps", "2", "--ep-ticks", "600", "--unstuck-patience", "4096",
         "--unstuck-period", "4096", "--auto-spine-envs", "16", "--auto-spine-mcts-minutes",
         "0.05", "--auto-spine-ge-minutes", "0.05", "--auto-spine-patience", "1",
         "--auto-spine-retry", "1", "--auto-spine-extra-secs", "1", "--no-compile", "--fp32",
         "--seed", "0"]
    # identity runs hold the search off; a search smoke lets it start at once (v14's own
    # --auto-spine-delay 3e8 would keep any CPU-sized run from ever searching)
    s += ["--auto-spine-delay", "1e12" if not search else "0"]
    if search:
        # a CPU search at smoke scale: the MC stage's 4096 envs -> 64 (v14's other search args)
        i = V14.index("--auto-spine-args")
        s += ["--auto-spine-args", V14[i + 1].replace("--mc-envs 4096", "--mc-envs 64")]
    return s


DLL = Path(os.environ.get("SURFCORE_DLL", str(ROOT / "build" / "libsurfcore.so")))
MAP = Path(os.environ.get("V14_SMOKE_MAP", str(ROOT / "maps" / "surf_unitfarmer2.bsp")))
KINDS = ("goal", "goalc", "sdf", "occ", "slabocc", "faceid")


def have_map(m=MAP) -> bool:
    return DLL.exists() and m.exists() and all(
        m.with_suffix(f".{k}_32.npz").exists() for k in KINDS)


def env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
                OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", SURFCORE_DLL=str(DLL))


def run(root, run_name, map_args, extra=(), steps=12288, search=False, timeout=3600):
    root = Path(root)
    shutil.rmtree(root / "runs" / run_name, ignore_errors=True)
    cmd = ([sys.executable, "-u", str(root / "python" / "train_fast.py")] + list(map_args)
           + ["--run", run_name] + BASE + V14 + MCTS + smoke(steps, search=search) + list(extra))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env(), cwd=str(root),
                       timeout=timeout, encoding="utf-8", errors="replace")
    return r, root / "runs" / run_name


def base_tree(dst, ref="fea35b06"):
    """<dst>/python as of ``ref`` (V14_BASE_TREE reuses an extracted one)"""
    pre = os.environ.get("V14_BASE_TREE")
    if pre:
        return Path(pre)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    tar = dst / "base.tar"
    r = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "-o", str(tar), ref,
                        "python"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git archive {ref}: {r.stderr[-300:]}")
    with tarfile.open(tar) as tf:
        tf.extractall(dst)
    return dst


def rows(path, drop=("time/fps",)):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    keep = [i for i, h in enumerate(head) if h not in drop]
    return [[r.split(",")[i] for i in keep] for r in lines[1:]]


def same_run(a, b, n_traj=3, drop_cfg=()):
    """config dict, every eval trajectory byte for byte, progress rows (minus time/fps)"""
    a, b = Path(a), Path(b)
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    for k in drop_cfg:
        ca.pop(k, None)
        cb.pop(k, None)
    diff = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
    assert not diff, f"config differs in {diff}"
    ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
    assert len(ta) >= n_traj and [p.name for p in ta] == [p.name for p in tb], (ta, tb)
    for x, y in zip(ta, tb):
        assert x.read_bytes() == y.read_bytes(), f"{x.name} differs"
    assert rows(a / "progress.csv") == rows(b / "progress.csv")


# --- multi-map runs of the v14 line, minus the families not ported yet --------------------------

STRVAL = {"--auto-spine-args", "--auto-spine-mcts-args"}     # one value that starts with --
FAMILIES = {
    "front_temp": ["--front-temp", "--front-band"],
    "demo_front_temp": ["--demo-front-temp", "--demo-front-yaw-abs", "--demo-front-ticks"],
    "greedy_temp": ["--respawn-greedy-temp", "--respawn-greedy-temp-ticks"],
    "energy_front": ["--energy-pot-front", "--energy-pot-front-bonus"],
    "int_gate": ["--int-coef-progress", "--int-share-max"],
    "unstuck": ["--unstuck", "--unstuck-temp-heads", "--unstuck-max", "--unstuck-patience",
                "--unstuck-period", "--unstuck-reach"],
    "front_perturb": ["--respawn-front-perturb", "--respawn-front-perturb-band",
                      "--respawn-front-perturb-speed", "--respawn-front-perturb-turn"],
    "greedy": ["--respawn-greedy", "--respawn-greedy-stuck", "--respawn-greedy-window",
               "--respawn-greedy-speed", "--respawn-greedy-turn"],
    "loop": [t for t in V14 if t.startswith("--auto-spine") or t.startswith("--demo-")
             if t not in ("--demo-front-temp", "--demo-front-yaw-abs", "--demo-front-ticks")]
            + ["--auto-spine-mcts-args", "--auto-spine-envs", "--auto-spine-mcts-minutes",
               "--auto-spine-ge-minutes", "--auto-spine-patience", "--auto-spine-retry",
               "--auto-spine-extra-secs", "--auto-spine-delay"],
    "obs_normal": ["--obs-normal"],
}


def strip(tokens, flags):
    flags = set(flags)
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in flags:
            i += 1
            if t in STRVAL:
                i += 1
            else:
                while i < len(tokens) and not tokens[i].startswith("--"):
                    i += 1
            continue
        out.append(t)
        i += 1
    return out


def run_line(root, run_name, map_args, drop=(), extra=(), steps=12288, search=False,
             timeout=3600):
    """the v14 line with the families in ``drop`` removed (FAMILIES keys)"""
    root = Path(root)
    shutil.rmtree(root / "runs" / run_name, ignore_errors=True)
    flags = [f for k in drop for f in FAMILIES[k]]
    line = strip(BASE + V14 + MCTS + smoke(steps, search=search), flags)
    cmd = ([sys.executable, "-u", str(root / "python" / "train_fast.py")] + list(map_args)
           + ["--run", run_name] + line + list(extra))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env(), cwd=str(root),
                       timeout=timeout, encoding="utf-8", errors="replace")
    return r, root / "runs" / run_name


MM_MAPS = [Path(p) for p in os.environ.get("MM_TEST_MAPS", ",".join(
    str(ROOT / "maps" / f"{s}.bsp") for s in ("surf_edgeflow_blue200",
                                                 "surf_unitfarmer2"))).split(",") if p]


def have_maps(maps=None) -> bool:
    return all(have_map(m) for m in (maps or MM_MAPS))
