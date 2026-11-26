"""test_m20.py - what the 20-map run adds (mm/m20, 2026-11-25; the user: "keep world record timers
somewhere and have a metric like which % of maps in training beat the WR", and the late lr +
entropy anneal of the 2026-11-24 review):

(a) surfgym.wrclock: the record clock runs from the LAST touch of the start trigger (n - i_s
    ticks), grade() counts a finish as autoresearch/score.py counts it (a goal test that says no
    removes it), summarise() counts only comparable maps;
(b) tools/leaderboard.py snapshot freezes the boards' records into one json;
(c) trainer smokes (CPU, v14 at smoke scale on two maps, MM_TEST_MAPS): --wr-file writes the
    record columns and the checkpoint's wr_best, --lr-final with --anneal-start moves train/lr
    linearly from the anneal start to --steps, a bare resume restores all three; with the flags
    off the run is bit-identical to the parent commit, and so is an --ent-final schedule (its
    start defaults to step 0, as it always ran).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))

PARENT = "f0c8193c"            # mm/prelaunch head mm/m20 was built on


# ==========================================================================
# (a) the record clock
# ==========================================================================
@pytest.fixture
def no_bsp(monkeypatch):
    """StartZone without a map file: no brush model matches, so every box is a plain box"""
    import surfgym.goalhull as gh
    monkeypatch.setattr(gh.GoalHull, "match", classmethod(lambda cls, bsp, box, tol=0.5: None))


def _rows(xs):
    return np.array([[i, x, 50.0, 50.0, 0.0, 0.0, 0.0] for i, x in enumerate(xs)], np.float64)


def test_the_record_clock_starts_at_the_last_touch_of_the_start_zone(no_bsp):
    from surfgym.wrclock import StartZone
    sz = StartZone("none.bsp", {"mins": [0, 0, 0], "maxs": [4, 100, 100]})
    assert sz.trigger and not sz.hulls and len(sz.boxes) == 1
    # grown by the standing hull: x in [-16, 20]
    xs = [-100 + 10 * i for i in range(50)]               # inside at i = 9 .. 12
    assert list(np.flatnonzero(sz.touch(_rows(xs)[:, 1:4]))) == [9, 10, 11, 12]
    assert sz.record_ticks(_rows(xs)) == 50 - 12
    # back through the zone later: the timer restarts, the clock runs from the later touch
    back = xs[:30] + [0.0, 5.0] + [100 + 10 * i for i in range(18)]
    assert sz.record_ticks(_rows(back)) == 50 - 31
    # never touched: no record clock at all (the human timer would never have started)
    assert sz.record_ticks(_rows([500 + i for i in range(20)])) is None
    # a button start is not a record start
    assert not StartZone("none.bsp", {"mins": [0, 0, 0], "maxs": [4, 4, 4],
                                      "from": "func_button"}).trigger


def test_grade_counts_a_finish_as_the_grader_does(no_bsp):
    from surfgym.wrclock import StartZone, grade, summarise
    sz = StartZone("none.bsp", {"mins": [0, 0, 0], "maxs": [4, 100, 100]})
    hdr = {"tick_ms": 7.666667, "tick_pattern_ms": [8, 8, 7]}
    xs = [-100 + 10 * i for i in range(50)]
    eps = [(hdr, _rows(xs), {"end": "done"}),               # 50 rows, record 38 ticks
           (hdr, _rows(xs[:40]), {"end": "done"}),          # 40 rows, record 28: the goal says no
           (hdr, _rows(xs[:30]), {"end": "fail"}),
           (hdr, _rows([500 + i for i in range(45)]), {"end": "done"})]   # never at the start

    class Goal:
        def finished(self, rows, tick_s):
            assert tick_s == pytest.approx(7.666667e-3)
            return len(rows) != 40

    g = grade(eps, sz, Goal())
    assert (g["n"], g["done"], g["verified"], g["touched"]) == (4, 3, 2, 1)
    assert g["best_spawn_s"] == pytest.approx(45 * 7.666667e-3)
    assert g["best_record_s"] == pytest.approx(38 * 7.666667e-3)
    assert grade(eps, None, None)["best_record_s"] is None
    s = summarise({"a": 10.0, "b": 30.0, "c": None, "d": 5.0},
                  {"a": 11.0, "b": 20.0, "c": 50.0},
                  {"a": True, "b": True, "c": True, "d": False})
    assert (s["n_comparable"], s["n_beat"], s["beat"], s["n_finished"]) == (3, 1, ["a"], 2)
    assert s["ratio_median"] == pytest.approx(np.median([10 / 11, 30 / 20]))


def test_the_snapshot_freezes_the_boards(tmp_path):
    import leaderboard
    out = tmp_path / "rec.json"
    leaderboard.main(["snapshot", "--maps", "surf_src_cannonball,surf_no_such_map",
                      "--out", str(out)])
    d = json.loads(out.read_text(encoding="utf-8"))
    board = json.loads((ROOT / "docs" / "leaderboards" / "surf_src_cannonball.json")
                       .read_text(encoding="utf-8"))
    assert d["maps"]["surf_src_cannonball"]["wr_s"] == board["wr_s"]
    assert d["maps"]["surf_no_such_map"]["wr_s"] is None
    from surfgym.wrclock import load_records
    assert load_records(out) == {"surf_src_cannonball": board["wr_s"]}


# ==========================================================================
# (c) trainer smokes
# ==========================================================================
import v14_smoke as V                                          # noqa: E402

KINDS5 = ("goal", "goalc", "sdf", "occ", "slabocc")            # no face-id: --obs-normal 0
MAPS = V.MM_MAPS[:2]
needs_two = pytest.mark.skipif(
    not (V.DLL.exists() and len(MAPS) == 2 and all(
        m.exists() and all(m.with_suffix(f".{k}_32.npz").exists() for k in KINDS5)
        for m in MAPS)), reason="MM_TEST_MAPS: two maps with their cell-32 caches")
SM = ["--obs-normal", "0", "--envs", "64", "--maxvel", "4000"]
ARCH = ["--wd", "0.01", "--wd-skip-norms", "1", "--conv-norm", "1", "--final-ln-affine", "0",
        "--mlp-act", "leaky"]
STEPS = 8 * 64 * 24                                            # 12,288 env steps
A0 = 8 * 64 * 12                                               # the anneal starts half way


def _maps():
    return ["--maps", ",".join(str(m) for m in MAPS)]


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    dup = sorted({h for h in rows[0] if rows[0].count(h) > 1})
    assert not dup, f"duplicate CSV columns (a DictReader keeps the last): {dup}"
    return rows[0], rows[1:]


def _tag(m):
    from surfgym.mapfleet import map_tag
    return map_tag(Path(m).stem)


@needs_two
def test_wr_columns_lr_schedule_and_a_resume(tmp_path):
    rec = tmp_path / "records.json"
    # the first map's record is unbeatable, the second map's is beaten by any finish at all
    rec.write_text(json.dumps({"maps": {MAPS[0].stem: {"wr_s": 0.01},
                                        MAPS[1].stem: {"wr_s": 1e6}}}), encoding="utf-8")
    r, d = V.run_line(V.ROOT, "m20_on", _maps(), steps=STEPS,
                      extra=SM + ARCH + ["--wr-file", str(rec), "--lr-final", "1e-4",
                                         "--anneal-start", str(A0)], timeout=5400)
    out = r.stdout + r.stderr
    try:
        assert r.returncode == 0, out[-5000:]
        assert "--wr-file: " in out and "] WR: " in out
        cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
        assert cfg["wr_file"] == str(rec) and cfg["lr_final"] == 1e-4
        assert cfg["anneal_start"] == A0
        import record_ckpt
        record_ckpt.audit_cfg(cfg)
        head, rows = _csv(d / "progress.csv")
        tags = [_tag(m) for m in MAPS]
        want = (["race/maps_wr", "race/maps_wr_ever", "race/wr_ratio_med"]
                + [f"race/eval_rec_s.{t}" for t in tags] + ["train/lr", "train/ent_coef"])
        assert head[-len(want):] == want and len(rows) >= 6
        st = [float(x[head.index("time/total_timesteps")]) for x in rows]
        lr = [float(x[head.index("train/lr")]) for x in rows]
        for s, v in zip(st, lr):
            f = min(1.0, max(0.0, s - A0) / (STEPS - A0))
            assert v == pytest.approx(3e-4 + (1e-4 - 3e-4) * f, rel=1e-5), (s, v)
        assert lr[0] == pytest.approx(3e-4) and lr[-1] < 2e-4
        # the record columns: shares in [0, 1] once an eval ran; the unbeatable map never counts
        mw = [x[head.index("race/maps_wr")] for x in rows if x[head.index("race/maps_wr")]]
        assert mw and all(0.0 <= float(v) <= 0.5 for v in mw)
        ck = __import__("torch").load(d / "ckpt_final.pt", map_location="cpu",
                                      weights_only=False)
        assert set(ck["wr_best"]) == set(tags)
        # a bare resume restores the three flags and carries the schedule on
        os.environ["SELF_STATES"] = "1"            # the run's own spine is policy-derived
        r2, d2 = V.run_line(V.ROOT, "m20_re", _maps(), steps=STEPS + 8 * 64 * 4,
                            extra=SM + ["--ckpt", str(d / "ckpt_final.pt")], timeout=5400)
        o2 = r2.stdout + r2.stderr
        assert r2.returncode == 0, o2[-5000:]
        assert "lr_final=0.0001" in o2 and f"anneal_start={A0}" in o2
        assert f"wr_file={rec}" in o2
    finally:
        os.environ.pop("SELF_STATES", None)
        for name in (() if os.environ.get("M20_KEEP") else ("m20_on", "m20_re")):
            shutil.rmtree(V.ROOT / "runs" / name, ignore_errors=True)


def _parent_tree(dst):
    pre = os.environ.get("M20_PARENT_TREE")
    if pre:
        return Path(pre)
    dst.mkdir(parents=True, exist_ok=True)
    tar = dst / "base.tar"
    subprocess.run(["git", "-C", str(V.ROOT), "archive", "--format=tar", "-o", str(tar), PARENT,
                    "python"], check=True, capture_output=True)
    with tarfile.open(tar) as t:
        t.extractall(dst)
    return dst


@needs_two
@pytest.mark.parametrize("extra,name", [([], "m20_id"),
                                        (["--ent-final", "0.001"], "m20_ident")])
def test_flags_off_and_ent_final_are_bit_identical_to_the_parent(tmp_path, extra, name):
    import torch
    base = _parent_tree(tmp_path / "base")
    r0, d0 = V.run_line(base, name, _maps(), extra=SM + ARCH + extra)
    assert r0.returncode == 0, (r0.stdout + r0.stderr)[-4000:]
    r1, d1 = V.run_line(V.ROOT, name, _maps(), extra=SM + ARCH + extra)
    assert r1.returncode == 0, (r1.stdout + r1.stderr)[-4000:]
    try:
        ca = json.loads((d0 / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((d1 / "run.json").read_text(encoding="utf-8"))["config"]
        for c in (ca, cb):
            c.pop("demo_file", None)
        assert ca == cb, sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
        assert not {"lr_final", "anneal_start", "wr_file"} & set(cb)
        ta, tb = sorted(d0.glob("traj_*.jsonl")), sorted(d1.glob("traj_*.jsonl"))
        assert len(ta) >= 4 and [p.name for p in ta] == [p.name for p in tb]
        for x, y in zip(ta, tb):
            assert x.read_bytes() == y.read_bytes(), x.name
        ha, ra = _csv(d0 / "progress.csv")
        hb, rb = _csv(d1 / "progress.csv")
        # --ent-final now logs the scheduled values in two columns after everything else
        add = ["train/lr", "train/ent_coef"] if extra else []
        assert hb == ha + add
        fps = ha.index("time/fps")
        assert len(ra) == len(rb) >= 3
        for x, y in zip(ra, rb):
            y = y[:len(ha)]
            assert x[:fps] + x[fps + 1:] == y[:fps] + y[fps + 1:]
        if extra:
            ent = [float(y[hb.index("train/ent_coef")]) for y in rb]
            assert ent[-1] < ent[0]                  # the old schedule, from step 0
        sa = torch.load(d0 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(d1 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        assert sa["policy"].keys() == sb["policy"].keys()
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
        assert "wr_best" not in sb
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        if not os.environ.get("M20_PARENT_TREE"):
            shutil.rmtree(d0, ignore_errors=True)


# ==========================================================================
# (d) --resume-search-state: a resume (or a fork) goes on with every map's search loop
# ==========================================================================
@needs_two
def test_resume_search_state_keeps_the_clocks_and_the_held_keys(tmp_path):
    """every --maps --auto-spine checkpoint stores each map's search clocks; a resume with
    --resume-search-state 1 under another run name (a fork) takes them back - the delay is NOT
    re-armed - and the held spine's spawns hold its own keys (<spine>.keys.npy); a plain resume
    re-arms the delay from the resume step, as it always did"""
    import torch
    delay = ["--auto-spine-delay", "1e9"]
    r, d = V.run_line(V.ROOT, "m20_cl", _maps(), steps=STEPS, extra=SM + delay, timeout=5400)
    out = r.stdout + r.stderr
    names = ("m20_cl", "m20_rs", "m20_rp")
    try:
        assert r.returncode == 0, out[-5000:]
        ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
        tags = [_tag(m) for m in MAPS]
        clocks = ck["auto_search_clocks"]
        assert set(clocks) == set(tags)
        for t in tags:
            assert clocks[t]["retry_at"] == int(1e9) and clocks[t]["k"] == 0, clocks[t]
            assert set(clocks[t]) == {"last", "retry_at", "t_hold", "tau_low", "k"}
        # the held spines (each map's spawn spine) get keys of their own: a fork must spawn on them
        held = ck["auto_spine_spines"]
        for t in tags:
            sp = Path(held[t])
            assert sp.exists(), sp
            n = len(np.load(sp, mmap_mode="r"))
            np.save(sp.with_suffix(".keys.npy"), np.tile(np.array([2, 0, 1], np.int64), (n, 1)))
        os.environ["SELF_STATES"] = "1"            # the run's own spine is policy-derived
        more = STEPS + 8 * 64 * 4
        r2, d2 = V.run_line(V.ROOT, "m20_rs", _maps(), steps=more,
                            extra=SM + delay + ["--ckpt", str(d / "ckpt_final.pt"),
                                                "--resume-search-state", "1"], timeout=5400)
        o2 = r2.stdout + r2.stderr
        assert r2.returncode == 0, o2[-5000:]
        line = next(x for x in o2.splitlines() if x.startswith("--resume-search-state: "))
        for t in tags:
            assert f"{t} retry_at 1,000,000,000" in line and "held keys" in line, line
        assert o2.count("its spawns hold their rows' keys (spine0.keys.npy)") == len(tags)
        cfg2 = json.loads((d2 / "run.json").read_text(encoding="utf-8"))["config"]
        assert cfg2["resume_search_state"] == 1
        import record_ckpt
        record_ckpt.audit_cfg(cfg2)
        ck2 = torch.load(d2 / "ckpt_final.pt", map_location="cpu", weights_only=False)
        assert all(ck2["auto_search_clocks"][t]["retry_at"] == int(1e9) for t in tags)
        # a plain resume: the delay re-armed from the resume step, neutral keys, no config key
        r3, d3 = V.run_line(V.ROOT, "m20_rp", _maps(), steps=more,
                            extra=SM + delay + ["--ckpt", str(d / "ckpt_final.pt")],
                            timeout=5400)
        o3 = r3.stdout + r3.stderr
        assert r3.returncode == 0, o3[-5000:]
        assert "--resume-search-state: " not in o3 and "spine0.keys.npy" not in o3
        rearm = [x for x in o3.splitlines() if x.startswith("--auto-spine-delay: no search before")]
        assert rearm and "1,000,000,000 " not in rearm[0] + " ", rearm
        cfg3 = json.loads((d3 / "run.json").read_text(encoding="utf-8"))["config"]
        assert "resume_search_state" not in cfg3
        # and the flag refuses a run that is not a resume
        r4, _ = V.run_line(V.ROOT, "m20_rx", _maps(), steps=STEPS,
                           extra=SM + ["--resume-search-state", "1"], timeout=600)
        assert r4.returncode != 0 and "give --ckpt" in (r4.stdout + r4.stderr)
    finally:
        os.environ.pop("SELF_STATES", None)
        for name in (() if os.environ.get("M20_KEEP") else names + ("m20_rx",)):
            shutil.rmtree(V.ROOT / "runs" / name, ignore_errors=True)
