"""autoresearch/mm_cut.py - the P1 cut rule (mm/s1): STOP early on throughput or on no finish by
3.0B fleet steps, CUT at 6.0B or once every map has finished and has 5 evals after its first
finish, CONTINUE otherwise. The rule on synthetic rows; end to end on v14's petrus / blue200 /
celestial evals renamed as one --maps run (SCORE_MAPS_BUNDLES, SCORE_MAPS_ROOT as in
test_score_maps.py)."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "autoresearch"))
sys.path.insert(0, str(ROOT / "python"))
import mm_cut  # noqa: E402

A = SimpleNamespace(cut_steps=6.0e9, after_finish_evals=5, min_fps=150e3, fps_after_s=1800.0,
                    no_finish_by=3.0e9)


def _row(name, fins):
    """a map with one eval per 0.1B fleet steps; fins[i] = finishes at eval i"""
    return {"map": name, "evals": [(int((i + 1) * 1e8), f, 9, 50.0, None) for i, f in
                                   enumerate(fins)], "first_finish_own_step": None}


def test_the_rule():
    no = [_row("a", [0] * 10), _row("b", [0] * 10)]
    # throughput: only past 30 min of training
    assert mm_cut.decide(no, {"step": int(1e8), "fps": 100e3}, A)[0] == mm_cut.CONTINUE
    code, why, _ = mm_cut.decide(no, {"step": int(5e8), "fps": 100e3}, A)
    assert code == mm_cut.STOP and "steps/s" in why[0]
    # no map finished by 3.0B
    assert mm_cut.decide(no, {"step": int(2.9e9), "fps": 4e5}, A)[0] == mm_cut.CONTINUE
    code, why, _ = mm_cut.decide(no, {"step": int(3.0e9), "fps": 4e5}, A)
    assert code == mm_cut.STOP and "no map finished" in why[0]
    # one map finished: not hopeless; every map + 5 evals after its first finish: CUT
    one = [_row("a", [0] * 3 + [1] * 7), _row("b", [0] * 10)]
    assert mm_cut.decide(one, {"step": int(3.5e9), "fps": 4e5}, A)[0] == mm_cut.CONTINUE
    four = [_row("a", [0] * 3 + [1] * 7), _row("b", [0] * 5 + [2] * 5)]     # b: 4 after
    assert mm_cut.decide(four, {"step": int(1e9), "fps": 4e5}, A)[0] == mm_cut.CONTINUE
    five = [_row("a", [0] * 3 + [1] * 7), _row("b", [0] * 4 + [2, 0, 0, 0, 0, 0])]
    code, why, _ = mm_cut.decide(five, {"step": int(1e9), "fps": 4e5}, A)
    assert code == mm_cut.CUT and "every map finished" in why[0]
    # the step cut
    code, why, _ = mm_cut.decide(one, {"step": int(6.0e9), "fps": 4e5}, A)
    assert code == mm_cut.CUT and "6,000,000,000" in why[0]


BUNDLES = {k: Path(v) for k, v in (x.split("=", 1) for x in
                                   os.environ.get("SCORE_MAPS_BUNDLES", "").split(",") if "=" in x)}
MAPS_ROOT = os.environ.get("SCORE_MAPS_ROOT")


@pytest.mark.skipif(not (BUNDLES and MAPS_ROOT), reason="needs SCORE_MAPS_BUNDLES and _ROOT")
def test_end_to_end_on_v14s_evals(tmp_path):
    import score
    from surfgym.mapfleet import map_tag
    keep = dict(score.CFG)
    run = tmp_path / "mm"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"config": {"maps": list(BUNDLES), "envs": 1023}}))
    files = {stem: score.eval_files([b]) for stem, b in BUNDLES.items()}
    try:
        def stage(n):
            for p in run.glob("traj_*"):
                p.unlink()
            last = 0
            for stem, fs in files.items():
                for f in fs[:n]:
                    step = int(f.name.split("_")[1].split(".")[0])
                    last = max(last, step)
                    gz = ".gz" if f.name.endswith(".gz") else ""
                    shutil.copy(f, run / f"traj_{step:010d}_{map_tag(stem)}.jsonl{gz}")
            (run / "progress.csv").write_text(
                f"time/total_timesteps,time/fps\n{last},400000\n")
        # the first 8 evals of each: petrus has 3 after its first finish, the others fewer
        stage(8)
        assert mm_cut.main([str(run), "--root", MAPS_ROOT]) == mm_cut.CONTINUE
        # all of them: every map finished with >= 5 evals after its first finish
        stage(10 ** 6)
        assert mm_cut.main([str(run), "--root", MAPS_ROOT]) == mm_cut.CUT
    finally:
        score.use_config(keep)


M7 = SimpleNamespace(cut_steps=0.0, after_finish_evals=0, min_fps=120e3, fps_after_s=1800.0,
                     no_finish_by=0.0, credit=None, credit_floor=10.0, dead=None)


def test_the_mm7_rules_stop_only_a_broken_run():
    done = [_row("a", [1] * 10), _row("b", [1] * 10)]
    none = [_row("a", [0] * 10), _row("b", [0] * 10)]
    # no step cap, no success cut, no "nothing finished" stop
    assert mm_cut.decide(done, {"step": int(9e9), "fps": 3e5}, M7)[0] == mm_cut.CONTINUE
    assert mm_cut.decide(none, {"step": int(50e9), "fps": 3e5}, M7)[0] == mm_cut.CONTINUE
    # a broken run: throughput, a dead trainer, the credit floor
    assert mm_cut.decide(none, {"step": int(5e8), "fps": 1e5}, M7)[0] == mm_cut.STOP
    dead = SimpleNamespace(**{**vars(M7), "dead": "the trainer is gone"})
    assert mm_cut.decide(none, {"step": int(1e9), "fps": 3e5}, dead)[0] == mm_cut.STOP
    poor = SimpleNamespace(**{**vars(M7), "credit": 9.5})
    code, why, _ = mm_cut.decide(none, {"step": int(1e9), "fps": 3e5}, poor)
    assert code == mm_cut.STOP and "credit" in why[0]


def test_a_dead_trainer_is_seen_on_the_box(tmp_path):
    import time
    run = tmp_path / "mm7_1"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"config": {"maps": ["surf_petrus_lite"],
                                                         "envs": 147}}))
    (run / "progress.csv").write_text("time/total_timesteps,time/fps\n1000000,300000\n")
    pid = tmp_path / "mm7_1.pid"
    pid.write_text(str(os.getpid()))                    # alive
    assert mm_cut.main([str(run), "--mm7", "--pid-file", str(pid)]) == mm_cut.CONTINUE
    pid.write_text("999999999")                          # gone
    assert mm_cut.main([str(run), "--mm7", "--pid-file", str(pid)]) == mm_cut.STOP
    old = time.time() - 11 * 60                          # not written for 11 minutes
    os.utime(run / "progress.csv", (old, old))
    assert mm_cut.main([str(run), "--mm7"]) == mm_cut.STOP
    assert mm_cut.main([str(run), "--mm7", "--stale-min", "0"]) == mm_cut.CONTINUE
    assert mm_cut.main([str(run), "--mm7", "--stale-min", "0", "--credit", "8"]) == mm_cut.STOP


def test_the_queue_waits_from_the_launch_log(tmp_path):
    log = tmp_path / "l.txt"
    log.write_text(
        "[        6,144] auto-spine[a]: quorum request queued (queue: a:quorum)\n"
        "[        6,144] auto-spine[b]: quorum request queued (queue: a:quorum, b:quorum)\n"
        "[        6,144] auto-spine[a] search 1: running beside the training (pid 1, log x)\n"
        "[       10,240] auto-spine[b]: quorum request already queued (queue: b:quorum)\n"
        "[       22,528] auto-spine[b] search 1: running beside the training (pid 2, log y)\n"
        "[       26,624] auto-spine[a]: patience request queued (queue: a:patience)\n")
    q = mm_cut.queue_waits(log, 2)
    assert q["a"]["searches"] == 1 and q["a"]["waited"] == 0 and q["a"]["pending_since"] == 26624
    assert q["b"] == {"searches": 1, "waited": 22528 - 6144, "max_wait": 22528 - 6144}


def test_the_read_against_v14_alone():
    refs = json.loads((ROOT / "autoresearch" / "map_configs" / "v14_alone.json").read_text())
    pb = refs["surf_princessburglar_b26"]
    assert [m[0] for m in pb["milestones"]] == ["room exit", "past the x -1,005 wall", "the shaft",
                                                "the finish"]
    # a seven-map run: pb leaves the room at fleet 7.0B (1.0B own) and passes the wall
    ev = [(int(7.0e9), 0, 9, 3.2, None), (int(14.0e9), 0, 9, 15.0, None)]
    lines = mm_cut.against_refs([{"map": "surf_princessburglar_b26", "evals": ev}], refs, 7)
    assert "room exit (best > 1%): 1,000,000,000 own | v14 alone 1,283,981,312 | x0.78" in lines[1]
    assert "past the x -1,005 wall (best > 7.5%): 2,000,000,000 own" in lines[2]
    assert "the shaft (best > 56%): not yet" in lines[3] and "the finish (a finish): not yet" in lines[4]
