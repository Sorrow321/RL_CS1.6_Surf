"""--yaw-sigma-min S (pbr2b2, 2026-10-24): the --pitch-sigma-min floor on the yaw head(s).
CPU, princessburglar: a smoke run with the flag floors the yaw sigma (printed and dumped), it is
refused out of range; flag off is covered by test_demo_view_velocity's bit-identity check (the
same pre-change trainer, which has neither flag)."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_ge_minutes import _train, needs_pb  # noqa: E402


@needs_pb
def test_the_floor_is_on_and_dumped():
    run = "ysm_on"
    try:
        r = _train(run, ["--yaw-sigma-min", "0.05", "--pitch-sigma-min", "0.05",
                         "--yaw-entropy", "0"])
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        m = re.search(r"--yaw-sigma-min 0.05: the yaw head's sigma is floored at 0.05; sig now "
                      r"([0-9.]+)/([0-9.]+)", out)
        assert m, out[-3000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["yaw_sigma_min"] == 0.05 and cfg["pitch_sigma_min"] == 0.05
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _train("ysm_bad", ["--yaw-sigma-min", "5"])
    assert bad.returncode != 0 and "--yaw-sigma-min must lie in" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "ysm_bad", ignore_errors=True)
