"""score.py per map (mm/s1, S2): --config / use_config, the tag-aware eval file filter, .jsonl.gz,
and autoresearch/score_maps.py over a --maps run (each map graded with its own config).

The end-to-end check renames single-map bundles' evals as one --maps run and compares the grades
(SCORE_MAPS_BUNDLES, SCORE_MAPS_ROOT; see below).
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "autoresearch"))
sys.path.insert(0, str(ROOT / "python"))
import score  # noqa: E402


def _touch(d, name, text="{}\n"):
    p = d / name
    if name.endswith(".gz"):
        with gzip.open(p, "wt") as fh:
            fh.write(text)
    else:
        p.write_text(text)
    return p


def test_the_eval_files_of_one_map(tmp_path):
    for n in ("traj_0000000100.jsonl", "traj_0000000100_petrus_lite.jsonl",
              "traj_0000000100_edgeflow_blue200.jsonl", "traj_0000000100_rec_greedy.jsonl",
              "traj_0000000100_petrus_lite_stoch.jsonl", "traj_0000000200_petrus_lite.jsonl.gz",
              "traj_0000000300_petrus_lite.jsonl", "traj_0000000300_petrus_lite.jsonl.gz"):
        _touch(tmp_path, n)
    got = [p.name for p in score.eval_files([tmp_path], "petrus_lite")]
    assert got == ["traj_0000000100.jsonl", "traj_0000000100_petrus_lite.jsonl",
                   "traj_0000000200_petrus_lite.jsonl.gz", "traj_0000000300_petrus_lite.jsonl"]
    assert [p.name for p in score.eval_files([tmp_path], None)] == ["traj_0000000100.jsonl"]


def test_the_eval_files_come_in_step_order_past_10b(tmp_path):
    """the trainer pads steps to 10 digits: an 11-digit step must not sort before a 10-digit one"""
    for n in ("traj_9661579264_x.jsonl", "traj_10718543872_x.jsonl", "traj_1359476736_x.jsonl",
              "traj_0000524288_x.jsonl"):
        _touch(tmp_path, n)
    got = [p.name for p in score.eval_files([tmp_path], "x")]
    assert got == ["traj_0000524288_x.jsonl", "traj_1359476736_x.jsonl",
                   "traj_9661579264_x.jsonl", "traj_10718543872_x.jsonl"]


def test_gz_reads_like_plain(tmp_path):
    text = ('{"map": "m", "episode": 0}\n[0, 1.0, 2.0, 3.0]\n[1, 1.5, 2.0, 3.0]\n'
            '{"end": "fail"}\n')
    a, b = _touch(tmp_path, "a.jsonl", text), _touch(tmp_path, "b.jsonl.gz", text)
    ea, eb = score.episodes(a), score.episodes(b)
    assert len(ea) == len(eb) == 1 and (ea[0][1] == eb[0][1]).all() and ea[0][2] == eb[0][2]


def test_use_config_switches_everything_and_drops_the_caches(tmp_path):
    keep = dict(score.CFG)
    try:
        score._FIELD, score._CPS, score._GOAL = "x", "y", "z"
        cfg = {"map": "maps/surf_petrus_lite.bsp", "field": "maps/f.npz", "start_d": 35940.0,
               "start_d_span": [35916.0, 35940.0], "score_episodes": 9, "time_cap_s": 120.0}
        p = tmp_path / "c.json"
        p.write_text(json.dumps(cfg))
        score.use_config(p, root=tmp_path)
        assert score.START_D == 35940.0 and score.START_SPAN == (35916.0, 35940.0)
        assert score.SCORE_EPISODES == 9 and score.EP_CAP_S == 120.0
        assert score._FIELD is None and score._CPS is None and score._GOAL is None
        assert score.map_tag_of_config() == "petrus_lite"
        assert score._in_root("maps/f.npz") == tmp_path / "f.npz"       # a flat staging dir
    finally:
        score.use_config(keep)


# {stem: a single-map run bundle with its greedy evals traj_<step>.jsonl[.gz]}, e.g.
# SCORE_MAPS_BUNDLES=surf_petrus_lite=runs/ab7c/ab7c_p14,surf_src_celestial=runs/ab7c/ab7c_l14
# SCORE_MAPS_ROOT = a flat directory with each map's grading field (<stem>.goalc_32.npz + bsp)
BUNDLES = {k: Path(v) for k, v in (x.split("=", 1) for x in
                                   os.environ.get("SCORE_MAPS_BUNDLES", "").split(",") if "=" in x)}
MAPS_ROOT = os.environ.get("SCORE_MAPS_ROOT")


@pytest.mark.skipif(not (BUNDLES and MAPS_ROOT), reason="needs SCORE_MAPS_BUNDLES and _ROOT")
def test_a_maps_run_is_graded_per_map_exactly_as_its_single_map_files(tmp_path):
    """the bundles' evals renamed as one --maps run's traj_<step>_<tag> files: score_maps grades
    each map on its own config (picked from run.json) exactly as score.py grades the originals"""
    import shutil
    import score_maps
    from surfgym.mapfleet import map_tag
    keep = dict(score.CFG)
    run = tmp_path / "mm"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"config": {"maps": list(BUNDLES)}}))
    want = {}
    try:
        for stem, b in BUNDLES.items():
            files = score.eval_files([b])
            assert files, b
            for f in files:
                step = f.name.split("_")[1].split(".")[0]
                gz = ".gz" if f.name.endswith(".gz") else ""
                shutil.copy(f, run / f"traj_{step}_{map_tag(stem)}.jsonl{gz}")
            score.use_config(score_maps.MAP_CONFIGS / f"{stem}.json", root=MAPS_ROOT)
            want[stem] = score.best_in(files)
        rows = score_maps.grade(run, score_maps.configs_of_run(run), MAPS_ROOT)
    finally:
        score.use_config(keep)
    assert [r["map"] for r in rows] == list(BUNDLES)
    for r in rows:
        w = want[r["map"]]
        assert (r["n_evals"], r["n_valid"]) == (w["n_evals"], w["n_valid"]), r["map"]
        assert w["best"] is not None and r["best"] is not None, r
        for k in ("score", "best_progress_pct", "finishes", "best_finish_s", "episodes"):
            assert r["best"][k] == w["best"][k], (r["map"], k)
        assert Path(r["best"]["file"]).name.startswith(Path(w["best"]["file"]).name[:15])
