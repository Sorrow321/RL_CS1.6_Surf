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


@needs_run
def test_jump_penalties_reach_training_and_recording():
    """--bhop-cap 0 / --stamina 0: CS 1.6's jump penalties off (a no-slowdown server). The config
    carries them only when off; record_ckpt rebuilds the core without them."""
    from test_obs_potential import CANNONBALL, RECORD, _run
    r = _train("cya_jpen", ABS + ["--bhop-cap", "0", "--stamina", "0"], steps="6144")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = ROOT / "runs" / "cya_jpen"
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["bhop_cap"] == 0 and cfg["stamina"] == 0, cfg
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"), "--map",
                str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-3000:] + rec.stderr[-3000:]
    hdr = json.loads((d / "rec.jsonl").read_text(encoding="utf-8").splitlines()[0])
    phys = hdr.get("phys") or {}
    assert int(phys.get("enable_bhop_cap", -1)) == 0 and int(phys.get("enable_stamina", -1)) == 0, phys
    shutil.rmtree(d, ignore_errors=True)


def test_jump_penalties_physics():
    """The cap does what the help says, on the cannonball spawn floor: a jump at 1,200 u/s leaves
    240 u/s with it and keeps its speed without it."""
    import numpy as np
    from surfgym import SurfCore, default_config
    from surfgym.core import SurfState
    from test_obs_potential import CANNONBALL
    out = {}
    for cap in (1, 0):
        core = SurfCore(str(CANNONBALL), default_config(num_envs=1, lidar_w=0, lidar_h=0,
                                                        sv_friction=0.0, enable_bhop_cap=cap))
        core.reset(0)
        st = core.get_states()[0]
        s = SurfState()
        for k in range(3):
            s.origin[k] = float(st["origin"][k])
        for _ in range(60):                               # settle on the spawn floor
            core.pm_step_usercmd(s, 0.0, 0.0, 0.0, 0.0, 0, 10)
        assert s.onground != -1
        s.velocity[0], s.velocity[1] = 1200.0, 0.0
        core.pm_step_usercmd(s, 0.0, 0.0, 0.0, 0.0, 2, 10)      # IN_JUMP
        out[cap] = float(np.hypot(s.velocity[0], s.velocity[1]))
    assert abs(out[1] - 240.0) < 1.0, out
    assert out[0] > 1150.0, out
