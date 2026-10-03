"""--min-speed-kill / --min-speed-grace (the user, 2026-10-03: maps with no death - "kill the agent
when ... the speed drops below some value"): a training episode slower than the floor past the
grace ends as a FAIL through the stall kill's force_fail (counted in race/stall_frac); off, nothing
changes and the config carries no key."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_obs_potential import ABS, _train                          # noqa: E402
from test_view_continuous import needs_run                          # noqa: E402


@needs_run
def test_the_speed_floor_kills_slow_episodes():
    on = _train("cya_msk_on", ABS + ["--min-speed-kill", "500", "--min-speed-grace", "1"],
                steps="24576")
    assert on.returncode == 0, on.stdout[-3000:] + on.stderr[-3000:]
    d = ROOT / "runs" / "cya_msk_on"
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert (cfg["min_speed_kill"], cfg["min_speed_grace"]) == (500.0, 1.0)
    rows = list(csv.DictReader(open(d / "progress.csv", encoding="utf-8")))
    # a fresh policy on the start platform is slow: every episode ends just past the 1 s grace
    lens = [float(r["rollout/ep_len_mean"] or 0) for r in rows]
    assert max(lens) > 0 and all(L < 130 for L in lens if L > 0)
    assert any(float(r["race/stall_frac"] or 0) > 0 for r in rows)
    off = _train("cya_msk_off", ABS, steps="24576")
    assert off.returncode == 0, off.stdout[-3000:] + off.stderr[-3000:]
    cfg0 = json.loads((ROOT / "runs" / "cya_msk_off" / "run.json").read_text(encoding="utf-8"))
    assert "min_speed_kill" not in cfg0["config"]
    for n in ("cya_msk_on", "cya_msk_off"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
