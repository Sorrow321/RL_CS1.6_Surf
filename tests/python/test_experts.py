"""test_experts.py - what mm/experts adds to the trainer (2026-11-27, the per-map expert farm and its
expert loop):

* --demo-success-bonus-energy-only takes an explicit 0. A recipe-v14 checkpoint restores the flag
  on a resume, and the expert loop's warm rounds (tools/exit_local.py) pass --demo-success-bonus 0,
  which the restored flag refused ("scopes the --demo-success-bonus: give it"): the first loop on
  the farm's first expert (latebra) died in its first training round. The bare flag stays 1
  (recipe v14's token), so a run that passes it is unchanged.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v14_smoke as V                                          # noqa: E402

KINDS5 = ("goal", "goalc", "sdf", "occ", "slabocc")
MAPS = V.MM_MAPS[:1]
NO_MAP = not (V.DLL.exists() and len(MAPS) == 1 and all(
    m.exists() and all(m.with_suffix(f".{k}_32.npz").exists() for k in KINDS5) for m in MAPS))
needs_map = pytest.mark.skipif(NO_MAP, reason="MM_TEST_MAPS: one map with its cell-32 caches")
SM = ["--obs-normal", "0", "--envs", "64", "--maxvel", "3500"]
IT = 64 * 8 * 4


@needs_map
def test_an_explicit_zero_switches_the_restored_energy_only_flag_off():
    names = ("ex_seed", "ex_bad", "ex_ok")
    try:
        r, d = V.run_line(V.ROOT, "ex_seed", ["--map", str(MAPS[0])], steps=IT * 4, extra=SM)
        assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
        ck = d / "ckpt_final.pt"
        os.environ["SELF_STATES"] = "1"          # the run's own spine is policy-derived
        # a bare resume line with the loop's --demo-success-bonus 0: the restored flag refuses it
        base = [sys.executable, "-u", str(V.ROOT / "python" / "train_fast.py"), "--map",
                str(MAPS[0]), "--ckpt", str(ck), "--steps", str(IT * 6), "--envs", "64",
                "--obs-normal", "0", "--no-compile", "--fp32", "--record-every", "1e12",
                "--ckpt-every", "1e12", "--demo-success-bonus", "0"]
        import subprocess
        bad = subprocess.run(base + ["--run", "ex_bad"], capture_output=True, text=True,
                             env=V.env(), cwd=str(V.ROOT), timeout=1800, encoding="utf-8",
                             errors="replace")
        out = bad.stdout + bad.stderr
        assert bad.returncode != 0 and "scopes the --demo-success-bonus" in out, out[-3000:]
        # the same line with the explicit 0 starts, and its config says the flag is off
        ok = subprocess.run(base + ["--run", "ex_ok", "--demo-success-bonus-energy-only", "0"],
                            capture_output=True, text=True, env=V.env(), cwd=str(V.ROOT),
                            timeout=1800, encoding="utf-8", errors="replace")
        o2 = ok.stdout + ok.stderr
        assert ok.returncode == 0, o2[-4000:]
        assert "demo_success_bonus_energy_only" not in o2.split("restored from checkpoint config")[-1].split("\n")[0]
        import json
        cfg = json.loads((V.ROOT / "runs" / "ex_ok" / "run.json").read_text(encoding="utf-8"))["config"]
        assert "demo_success_bonus_energy_only" not in cfg
        seed_cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
        assert seed_cfg.get("demo_success_bonus_energy_only") == 1   # the bare v14 token: on
    finally:
        os.environ.pop("SELF_STATES", None)
        for n in names:
            shutil.rmtree(V.ROOT / "runs" / n, ignore_errors=True)
