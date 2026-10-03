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
    # --min-speed-secs: only a sustained second below the floor kills - episodes run ~1 s longer
    win = _train("cya_msk_win", ABS + ["--min-speed-kill", "500", "--min-speed-grace", "1",
                                       "--min-speed-secs", "1"], steps="24576")
    assert win.returncode == 0, win.stdout[-3000:] + win.stderr[-3000:]
    wrows = list(csv.DictReader(open(ROOT / "runs" / "cya_msk_win" / "progress.csv",
                                     encoding="utf-8")))
    wl = [float(r["rollout/ep_len_mean"] or 0) for r in wrows if float(r["rollout/ep_len_mean"] or 0)]
    assert wl and min(wl) >= 190, wl
    off = _train("cya_msk_off", ABS, steps="24576")
    assert off.returncode == 0, off.stdout[-3000:] + off.stderr[-3000:]
    cfg0 = json.loads((ROOT / "runs" / "cya_msk_off" / "run.json").read_text(encoding="utf-8"))
    assert "min_speed_kill" not in cfg0["config"]
    for n in ("cya_msk_on", "cya_msk_win", "cya_msk_off"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)


@needs_run
def test_sv_friction_reaches_training_and_recording():
    """--sv-friction: the server's ground friction (stock 4); a slide server is 0. The config
    carries it only when changed; record_ckpt rebuilds the core with it."""
    from test_obs_potential import CANNONBALL, RECORD, _run
    r = _train("cya_fric", ABS + ["--sv-friction", "0"], steps="6144")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = ROOT / "runs" / "cya_fric"
    assert json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]["sv_friction"] == 0.0
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"), "--map",
                str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-3000:] + rec.stderr[-3000:]
    hdr = json.loads((d / "rec.jsonl").read_text(encoding="utf-8").splitlines()[0])
    phys = hdr.get("phys") or {}
    assert float(phys.get("sv_friction", -1)) == 0.0, phys
    shutil.rmtree(d, ignore_errors=True)
