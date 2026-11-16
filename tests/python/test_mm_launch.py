"""The v14 multi-map launch path (mm/s0, 2026-11-16), not used yet: run_arm.sh SCRATCH=1 with
MAPS=a.bsp,b.bsp,... (the scratch base with --maps in place of --map) and box.py launch
--multimap (config "multimap": maps, one --maxvel per map, --envs).

* run_arm.sh: without MAPS the trainer receives exactly the tokens the fea35b06 script passed
  (a stub interpreter records them); with MAPS the only difference is --maps for --map;
  DRY_RUN=1 prints the same tokens and starts nothing.
* box.py: the recipe keeps every token but --envs and --maxvel; MAPS replaces MAP; bad
  configs and --resume are refused.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "autoresearch"))

RECIPE_TAIL = ["--envs", "1024", "--keys-hold", "--maxvel", "3500", "--tick-ms", "7.63",
               "--auto-spine-args", "--ge-held 1 --mc-envs 4096"]


def _stub(tmp_path):
    """an interpreter that records its argv and exits (the launcher then reports a dead
    trainer and exits 1, after the arguments are on disk)"""
    p = tmp_path / "stubpy"
    p.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$ARGS_OUT\"\nexit 0\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _script_tree(tmp_path, name, text):
    d = tmp_path / name
    (d / "tools").mkdir(parents=True)
    s = d / "tools" / "run_arm.sh"
    s.write_text(text)
    return s


def _launch(script, env_extra, args_out):
    env = dict(os.environ, SCRATCH="1", BUDGET="1000", RECORD_EVERY="75e6",
               ARGS_OUT=str(args_out), **env_extra)
    r = subprocess.run(["bash", str(script), "mmtest_run"] + RECIPE_TAIL, capture_output=True,
                       text=True, env=env, timeout=120)
    return r


def _old_script():
    r = subprocess.run(["git", "-C", str(ROOT), "show", "fea35b06:tools/run_arm.sh"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("fea35b06 is not reachable from this checkout")
    return r.stdout


def test_one_map_passes_the_tokens_it_always_did(tmp_path):
    stub = _stub(tmp_path)
    old = _script_tree(tmp_path, "old", _old_script())
    new = _script_tree(tmp_path, "new", (ROOT / "tools" / "run_arm.sh").read_text())
    a_old, a_new = tmp_path / "a_old.txt", tmp_path / "a_new.txt"
    _launch(old, {"PYTHON": str(stub), "MAP": "maps/x.bsp"}, a_old)
    _launch(new, {"PYTHON": str(stub), "MAP": "maps/x.bsp"}, a_new)
    t_old = a_old.read_text().splitlines()
    t_new = a_new.read_text().splitlines()
    assert t_old[:4] == ["-u", "python/train_fast.py", "--map", "maps/x.bsp"]
    assert t_new == t_old
    assert t_new[-len(RECIPE_TAIL):] == RECIPE_TAIL


def test_maps_replaces_map_and_nothing_else(tmp_path):
    stub = _stub(tmp_path)
    new = _script_tree(tmp_path, "new", (ROOT / "tools" / "run_arm.sh").read_text())
    one, two = tmp_path / "one.txt", tmp_path / "two.txt"
    _launch(new, {"PYTHON": str(stub), "MAP": "maps/x.bsp"}, one)
    r = _launch(new, {"PYTHON": str(stub), "MAP": "maps/x.bsp",
                      "MAPS": "maps/a.bsp,maps/b.bsp"}, two)
    t1, t2 = one.read_text().splitlines(), two.read_text().splitlines()
    assert t2[2:4] == ["--maps", "maps/a.bsp,maps/b.bsp"] and "--map" not in t2
    assert t2[4:] == t1[4:]
    assert "maps: maps/a.bsp,maps/b.bsp (one shared policy; MAP is ignored)" in r.stdout
    assert "== liveness (300s)" in r.stdout


def test_dry_run_prints_the_tokens_and_starts_nothing(tmp_path):
    stub = _stub(tmp_path)
    new = _script_tree(tmp_path, "new", (ROOT / "tools" / "run_arm.sh").read_text())
    rec = tmp_path / "rec.txt"
    _launch(new, {"PYTHON": str(stub), "MAPS": "maps/a.bsp,maps/b.bsp"}, rec)
    (tmp_path / "new" / "runs" / "mmtest_run.pid").unlink()   # the recording launch's
    dry = tmp_path / "dry.txt"
    r = _launch(new, {"PYTHON": str(stub), "MAPS": "maps/a.bsp,maps/b.bsp", "DRY_RUN": "1"},
                dry)
    assert r.returncode == 0 and not dry.exists()
    printed = [ln[len("DRY_RUN_ARG "):] for ln in r.stdout.splitlines()
               if ln.startswith("DRY_RUN_ARG ")]
    assert printed == rec.read_text().splitlines()[2:]
    assert not (tmp_path / "new" / "runs" / "mmtest_run.pid").exists()


# --- box.py ---------------------------------------------------------------------------------------

V14ISH = ["--envs", "1024", "--obs-normal", "1", "--maxvel", "3500", "--tick-ms", "7.63",
          "--auto-spine-args", "--ge-held 1 --mc-envs 4096", "--int-share-max", "0.25"]
MM = {"maps": ["maps/surf_src_sidistic.bsp", "maps/surf_unitfarmer2.bsp",
               "maps/surf_princessburglar_b26.bsp"],
      "maxvel": [4000, 3500, 3500], "envs": 1029}


def test_the_recipe_keeps_every_token_but_envs_and_maxvel():
    import box
    maps, out = box.multimap_recipe(MM, V14ISH)
    assert maps == MM["maps"]
    want = list(V14ISH)
    want[1] = "1029"
    want[5] = "4000,3500,3500"
    assert out == want and len(out) == len(V14ISH)
    for bad in ({**MM, "envs": 1024}, {**MM, "maxvel": [4000, 3500]}, {**MM, "maps": MM["maps"][:1],
                                                                        "maxvel": [4000]},
                {**MM, "maps": MM["maps"][:2] + MM["maps"][:1]}):
        with pytest.raises(SystemExit):
            box.multimap_recipe(bad, V14ISH)
    with pytest.raises(SystemExit):
        box.multimap_recipe(MM, ["--envs", "1", "--envs", "2"])


def test_launch_multimap(monkeypatch, tmp_path):
    import box
    monkeypatch.setattr(box, "AR_HOME", tmp_path)
    monkeypatch.setattr(box, "BOXES", tmp_path / "boxes")
    monkeypatch.setitem(box.CFG, "incumbent_recipe", list(V14ISH))
    monkeypatch.setitem(box.CFG, "multimap", dict(MM))
    box.save_rec("mm1", {"agent": "mm1", "id": 42, "gpu": "RTX_5090", "dph": 0.5, "created": 0,
                         "host": "h", "port": 1})
    sent = []
    monkeypatch.setattr(box, "ssh", lambda rec, cmd, **k: sent.append(cmd) or
                        box.subprocess.CompletedProcess(cmd, 0, "", ""))
    box.main(["launch", "--agent", "mm1", "--run", "mm1_p1", "--multimap", "--steps", "3e9"])
    cmd = sent[-1]
    assert "MAPS=maps/surf_src_sidistic.bsp,maps/surf_unitfarmer2.bsp," in cmd
    assert " MAP=" not in cmd and "SCRATCH=1 " in cmd
    assert "--envs 1029" in cmd and "--maxvel 4000,3500,3500" in cmd
    assert box.load_rec("mm1")["runs"][-1]["maps"] == MM["maps"]
    # without the flag: the single-map launch it always was
    box.main(["launch", "--agent", "mm1", "--run", "mm1_p2", "--steps", "3e9"])
    assert "MAP=" in sent[-1] and "MAPS=" not in sent[-1] and "--envs 1024" in sent[-1]
    assert "maps" not in box.load_rec("mm1")["runs"][-1]
    with pytest.raises(SystemExit):
        box.main(["launch", "--agent", "mm1", "--run", "mm1_p3", "--multimap", "--resume"])
