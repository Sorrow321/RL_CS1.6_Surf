"""A crash-resume of a THREE-map v14 run through the launcher path (mm/s1, 2026-11-16).

1. a 3-map v14 run trains (CPU smoke sizes, the search held off) and leaves ckpt_latest.pt;
2. box.py deploy's push reads that checkpoint's config (scp and ssh stubbed), and
   `box.py launch --multimap --resume` builds the box command (ssh stubbed);
3. that command runs in a tree laid out like /root/RL_Surf (python/, tools/run_arm.sh, maps/,
   runs_ckpt.pt) with DRY_RUN=1: run_arm.sh's resume branch prints the trainer's argv;
4. the trainer runs with exactly that argv in that tree: every map's state comes back by its
   tag (held spine, failed-line state, novelty gate and cap, greedy pool, unstuck schedule),
   the maps from the checkpoint (no --map / --maps on the line), the run's own --envs and
   per-map --maxvel, and every v14 flag of the first run's config.

The resume's command-line extra is the smoke SIZES only (a box resume passes none: the recipe
and config multimap's extra come from the config); the maps come from MM3_TEST_MAPS (three
.bsp paths, each with its six caches).
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "autoresearch"))
sys.path.insert(0, str(ROOT / "python"))
import v14_smoke as V  # noqa: E402

MAPS3 = [Path(p) for p in os.environ.get("MM3_TEST_MAPS", ",".join(
    str(ROOT / "maps" / f"{s}.bsp") for s in ("surf_edgeflow_blue200", "surf_unitfarmer2",
                                               "surf_src_celestial"))).split(",") if p]
MAXVEL = [4000.0, 3500.0, 4000.0]
ENVS = 66                                   # a multiple of three maps at CPU scale
needs3 = pytest.mark.skipif(not (len(MAPS3) == 3 and V.have_maps(MAPS3)),
                            reason="needs MM3_TEST_MAPS: three maps with their six caches")

# config keys that are the launch, not the run (the resume's own line and run_arm.sh's)
LAUNCH_KEYS = {"run", "steps", "ckpt", "ckpt_every", "eval_greedy_only", "seed"}


def _resume_extra(steps):
    """the smoke sizes for the resumed line: no --envs (the launcher's), no --steps (run_arm's
    budget), no --seed (the run's)"""
    line = V.smoke(steps)
    out, i = [], 0
    while i < len(line):
        if line[i] in ("--envs", "--steps", "--seed"):
            i += 2
            continue
        out.append(line[i])
        i += 1
    return out


def _tree(dst: Path, ck: Path) -> Path:
    """a /root/RL_Surf look-alike: the code under test, run_arm.sh, the maps, the checkpoint"""
    shutil.copytree(ROOT / "python", dst / "python",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "tools").mkdir(parents=True)
    shutil.copy2(ROOT / "tools" / "run_arm.sh", dst / "tools" / "run_arm.sh")
    (dst / "maps").mkdir()
    for m in MAPS3:
        for f in m.parent.glob(f"{m.stem}.*"):
            if f.suffix in (".bsp", ".json") or f.name.endswith("_32.npz"):
                shutil.copy2(f, dst / "maps" / f.name)     # copy2: the mtime the caches key on
    shutil.copy2(ck, dst / "runs_ckpt.pt")
    return dst


@needs3
def test_a_three_map_crash_resume_through_box_py_restores_every_maps_state(monkeypatch,
                                                                           tmp_path):
    import box
    import torch
    from surfgym.mapfleet import map_tags
    tags = map_tags([m.stem for m in MAPS3])
    r1, d1 = V.run_line(V.ROOT, "mmres3_a", ["--maps", ",".join(str(m) for m in MAPS3)],
                        extra=["--envs", str(ENVS), "--maxvel",
                               ",".join(f"{v:g}" for v in MAXVEL)], steps=8192)
    try:
        assert r1.returncode == 0, (r1.stdout + r1.stderr)[-4000:]
        # the end of the first run (its evals have moved every map's state); a crash would leave
        # ckpt_latest.pt, the same state dict at an earlier step
        ck1 = d1 / "ckpt_final.pt"
        cfg1 = json.loads((d1 / "run.json").read_text())["config"]
        tree = _tree(tmp_path / "RL_Surf", ck1)

        # 2. the push reads the checkpoint, the launch builds the box command
        monkeypatch.setattr(box, "AR_HOME", tmp_path / "ar")
        monkeypatch.setattr(box, "BOXES", tmp_path / "ar" / "boxes")
        # the v14 config as a P1 box carries it: the 160 tokens + config multimap (its extra =
        # v14's launch extra)
        monkeypatch.setitem(box.CFG, "incumbent_recipe", list(V.V14))
        monkeypatch.setitem(box.CFG, "multimap", {"maps": [f"maps/{m.name}" for m in MAPS3],
                                                  "maxvel": MAXVEL, "envs": ENVS,
                                                  "extra": list(V.MCTS)})
        want_md5 = box.md5(ck1)
        sent = []

        def fake_ssh(rec, cmd, **k):
            sent.append(cmd)
            out = want_md5 + "\n" if cmd.startswith("md5sum") else ""
            return subprocess.CompletedProcess(cmd, 0, out, "")
        monkeypatch.setattr(box, "ssh", fake_ssh)
        real_run = box.subprocess.run
        monkeypatch.setattr(box.subprocess, "run", lambda argv, **k: (
            subprocess.CompletedProcess(argv, 0, "", "") if argv[0] == "scp"
            else real_run(argv, **k)))
        rec = {"agent": "mmres3", "id": 1, "gpu": "RTX_5080", "dph": 0.4, "created": 0,
               "host": "h", "port": 1}
        box.push_ckpt(rec, ck1)
        assert rec["ckpt_cfg"]["maps"] == [m.stem for m in MAPS3]
        box.save_rec("mmres3", rec)
        extra = _resume_extra(8192)
        box.main(["launch", "--agent", "mmres3", "--run", "mmres3_b", "--multimap", "--resume",
                  "--steps", "8192", "--record-every", "2048", "--", *extra])
        cmd = sent[-1]
        assert cmd.startswith("cd /root/RL_Surf && ")
        assert "CKPT=runs_ckpt.pt ARM_RESUME=1" in cmd and "MAPS=" not in cmd

        # 3. run_arm.sh's resume branch, dry: the trainer's argv
        sh = cmd.replace("cd /root/RL_Surf && ", "", 1)
        env = dict(V.env(), DRY_RUN="1", CKF=str(tmp_path / "ck_step"), PYTHON=sys.executable)
        r = subprocess.run(["bash", "-c", sh], cwd=str(tree), env=env, capture_output=True,
                           text=True, timeout=600)
        assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
        argv = [ln[len("DRY_RUN_ARG "):] for ln in r.stdout.splitlines()
                if ln.startswith("DRY_RUN_ARG ")]
        assert argv[:2] == ["--ckpt", "runs_ckpt.pt"] and "--map" not in argv
        assert "--maps" not in argv
        assert argv.count("--envs") == 1 and argv.count("--maxvel") == 1
        assert argv[argv.index("--envs") + 1] == str(ENVS)
        assert argv[argv.index("--maxvel") + 1] == ",".join(f"{v:g}" for v in MAXVEL)
        assert argv.count("--auto-spine-scratch") == 1 and argv.count("--auto-spine-mcts-args") == 1
        assert int(argv[argv.index("--steps") + 1]) == int(torch.load(
            ck1, map_location="cpu", weights_only=False)["global_step"]) + 8192

        # 4. the trainer, exactly that argv, in the box-like tree
        r2 = subprocess.run([sys.executable, "-u", "python/train_fast.py"] + argv,
                            cwd=str(tree), env=V.env(), capture_output=True, text=True,
                            timeout=3600, encoding="utf-8", errors="replace")
        out = r2.stdout + r2.stderr
        assert r2.returncode == 0, out[-4000:]
        assert "maps=" + ",".join(m.stem for m in MAPS3) in out          # restored, not passed
        ck = torch.load(ck1, map_location="cpu", weights_only=False)
        for t in tags:
            _c = float(ck["int_coef_progress"][t]["coef"])
            assert f"--int-coef-progress[{t}]: restored the novelty coefficient {_c:.4g}" in out
            assert f"--auto-spine-scratch[{t}]: resumes on its held spine" in out, t
            assert f"auto-spine[{t}]: failed-line state restored from the checkpoint" in out, t
            assert f"--int-coef-progress[{t}]: restored the novelty coefficient" in out, t
            assert f"--int-share-max[{t}]: restored the cap's multiplier" in out, t
            assert f"restored --unstuck state [{t}]" in out, t
            assert f"restored --respawn-greedy state[{t}]" in out, t
        d2 = tree / "runs" / "mmres3_b"
        for t in tags:
            assert list(d2.glob(f"traj_*_{t}.jsonl")), t
        cfg2 = json.loads((d2 / "run.json").read_text())["config"]
        assert cfg2["maps"] == cfg1["maps"] and cfg2["envs"] == ENVS
        assert cfg2["maxvel_maps"] == cfg1["maxvel_maps"] == dict(zip(tags, MAXVEL))
        # every flag of the first run's config comes back (but the launch's own)
        diff = {k: (cfg1.get(k), cfg2.get(k)) for k in sorted(set(cfg1) | set(cfg2))
                if k not in LAUNCH_KEYS and cfg1.get(k) != cfg2.get(k)}
        assert not diff, diff
    finally:
        shutil.rmtree(d1, ignore_errors=True)
