"""The viewer must resolve a trajectory's map from the FILE, not from run.json.

Why this test exists: a --maps run trains one policy on several maps, and
run.json records only one of them. If the viewer trusts run.json, every
petrus recording renders against cannonball's geometry - which looks like
"the map is broken in the visualizer" and has been reported by the user
repeatedly. The fix is three lines in viewer/app.js, and it has been LOST
TWICE by branching from a branch that predates it (the arm branches carry
their own app.js). A static check catches that on any branch, in CI, without
a browser.

It also pins the POV path, which regressed the same way: the client must use
the path the server returns, or a stale depth-only render is shown in place
of a --surf-mask one.
"""
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "viewer" / "app.js"


def _src():
    return APP.read_text(encoding="utf-8")


def test_trajectory_header_map_overrides_run_json():
    s = _src()
    assert "if (hdr && hdr.map) cfg.map = hdr.map;" in s, (
        "viewer/app.js no longer takes the map from the trajectory's own "
        "header line; multi-map runs will render every trajectory against "
        "run.json's single map")


def test_loadmapforrun_receives_the_resolved_config():
    s = _src()
    assert "loadMapForRun(cfg);" in s, "loadMapForRun must get the RESOLVED cfg"
    assert "loadMapForRun(runCfg);" not in s, (
        "loadMapForRun(runCfg) is the bug: runCfg is run.json's config and "
        "names only one map of a --maps run")


def test_header_is_parsed_before_the_mesh_is_requested():
    """Resolution must precede loadMapForRun, or two meshes get fetched and
    the wrong one can win the race."""
    s = _src()
    i_hdr = s.index("if (hdr && hdr.map) cfg.map = hdr.map;")
    i_load = s.index("loadMapForRun(cfg);")
    assert i_hdr < i_load, "the header is parsed AFTER loadMapForRun is called"


def test_newline_split_does_not_use_a_literal_newline_in_a_string():
    """A literal newline inside a JS string is a parse error and takes the
    whole viewer down (this exact break shipped once)."""
    s = _src()
    assert "String.fromCharCode(10)" in s, (
        "the header split must use String.fromCharCode(10); a literal "
        "newline inside the string literal breaks parsing")


def test_pov_uses_the_server_provided_path():
    s = _src()
    assert re.search(r"j\.pov\s*\|\|", s), (
        "the POV panel must use the path the server returns; guessing "
        "'.pov.mp4' serves a stale depth-only render for --surf-mask runs")
