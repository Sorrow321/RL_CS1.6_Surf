"""tools/leaderboard.py: parsing, rank arithmetic, and the `run` table.

NO NETWORK. The parsing tests run against a trimmed capture of a real
speedrun.eu records page committed under tests/fixtures/, and the fetch test
substitutes that capture for the HTTP layer, so the pagination and top-N
logic is exercised without touching the site.

The three things most likely to rot, and why each has a test:

* the rank column is MEDAL EMOJI for ranks 1-3 and a number from 4 on, so a
  parser that trusts the cell would lose the world record itself;
* our eval clock is the SPAWN clock and a human record is the start-trigger
  clock, and a map whose finish is a BUTTON is not comparable at all
  (CLAUDE.md 4b) - the `run` table must say so and must not count a button
  map as a beaten world record;
* a joint run writes `race/eval_finish_s.<tag>` and a single-map run writes
  `race/eval_finish_s`, and an empty cell means "no finish in that eval",
  not zero.
"""
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import leaderboard as L                                 # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
PAGE1 = (FIXTURES / "speedrun_records_page1.html").read_text(encoding="utf-8")
EMPTY = (FIXTURES / "speedrun_records_empty.html").read_text(encoding="utf-8")


# --------------------------------------------------------------- parsing

def test_parse_time_formats():
    assert L.parse_time("00:27.74") == pytest.approx(27.74)
    assert L.parse_time("01:02.50") == pytest.approx(62.5)
    assert L.parse_time("1:02:03.45") == pytest.approx(3723.45)
    assert L.parse_time("27.74") == pytest.approx(27.74)
    assert L.parse_time("") is None
    assert L.parse_time("n/a") is None
    assert L.parse_time(None) is None


def test_parse_page_entries():
    got = L.parse_records_page(PAGE1)
    e = got["entries"]
    assert len(e) == 5
    assert [x["time_s"] for x in e] == pytest.approx(
        [27.74, 27.75, 27.76, 28.27, 3723.45])
    assert [x["player"] for x in e] == [
        "SHALL WE?", "Den0", "shrL", "Fizz & co", "Suzuki"]
    assert [x["date"] for x in e] == [
        "2022-11-08", "2022-10-06", "2025-06-21", "2024-09-10", "2022-10-03"]


def test_parse_page_medal_ranks_are_not_numbers():
    """Ranks 1-3 are medal emoji; only 4+ carry a digit in the cell."""
    got = L.parse_records_page(PAGE1)
    assert [x["rank"] for x in got["entries"]] == [None, None, None, 4, 5]


def test_parse_page_totals_and_pagination():
    got = L.parse_records_page(PAGE1)
    assert got["total_records"] == 48
    assert (got["page"], got["pages"]) == (1, 4)
    assert got["no_records"] is False


def test_parse_page_with_no_records():
    got = L.parse_records_page(EMPTY)
    assert got["entries"] == []
    assert got["total_records"] is None
    assert got["no_records"] is True


# ----------------------------------------------------------------- fetch

def _offline(monkeypatch, pages):
    """Serve `pages` (a dict cache-key -> html) instead of speedrun.eu."""
    seen = []

    def fake(url, cache, key, delay, refresh, tries=4, log=print):
        seen.append(key)
        return pages.get(key, EMPTY)

    monkeypatch.setattr(L, "http_get_cached", fake)
    return seen


def test_fetch_map_stops_at_the_last_page(monkeypatch, tmp_path):
    seen = _offline(monkeypatch, {"surf_x__p1": PAGE1})
    board = L.fetch_map("surf_x", 100, tmp_path, 0.0, False, log=lambda *a: None)
    assert board["status"] == "ok"
    assert board["wr_s"] == pytest.approx(27.74)
    assert board["wr_player"] == "SHALL WE?"
    assert board["total_records"] == 48          # from the page, not len()
    assert len(board["entries"]) == 5
    # every entry gets a rank even though 1-3 were medals
    assert [e["rank"] for e in board["entries"]] == [1, 2, 3, 4, 5]
    # page 2 was asked for (the footer said 4 pages) and came back empty
    assert seen[:2] == ["surf_x__p1", "surf_x__p2"]


def test_fetch_map_clamps_to_top(monkeypatch, tmp_path):
    _offline(monkeypatch, {"surf_x__p1": PAGE1})
    board = L.fetch_map("surf_x", 3, tmp_path, 0.0, False, log=lambda *a: None)
    assert len(board["entries"]) == 3
    assert board["total_records"] == 48


def test_fetch_map_not_found_tries_variants_once(monkeypatch, tmp_path):
    seen = _offline(monkeypatch, {})
    board = L.fetch_map("surf_a-b", 100, tmp_path, 0.0, False,
                        log=lambda *a: None)
    assert board["status"] == "not_found"
    assert board["entries"] == [] and board["wr_s"] is None
    assert board["variants_tried"] == ["surf_a-b", "surf_a_b"]
    assert seen == ["surf_a-b__p1", "surf_a_b__p1"]
    assert board["total_records"] == 0


# ---------------------------------------------------------------- ranking

BOARD = {"map": "surf_x", "total_records": 48, "wr_s": 27.74,
         "wr_player": "SHALL WE?",
         "entries": [{"rank": 1, "player": "SHALL WE?", "time_s": 27.74,
                      "date": "2022-11-08"},
                     {"rank": 2, "player": "Den0", "time_s": 27.75,
                      "date": "2022-10-06"},
                     {"rank": 3, "player": "shrL", "time_s": 27.76,
                      "date": "2025-06-21"},
                     {"rank": 4, "player": "Fizz", "time_s": 28.27,
                      "date": "2024-09-10"}]}


def test_rank_is_one_plus_the_faster_entries():
    assert L.rank_for_time(BOARD, 27.755)["rank"] == 3
    assert L.rank_for_time(BOARD, 28.00)["rank"] == 4
    assert L.rank_for_time(BOARD, 27.73)["rank"] == 1


def test_rank_ties_share_the_rank():
    """Equal is not FASTER: matching rank 2's time ranks 2nd, not 3rd."""
    assert L.rank_for_time(BOARD, 27.75)["rank"] == 2
    assert L.rank_for_time(BOARD, 27.74)["rank"] == 1


def test_rank_beats_wr_and_delta():
    r = L.rank_for_time(BOARD, 27.00)
    assert r["beats_wr"] is True
    assert r["delta_wr"] == pytest.approx(-0.74)
    assert r["rank"] == 1
    slow = L.rank_for_time(BOARD, 30.0)
    assert slow["beats_wr"] is False
    assert slow["delta_wr"] == pytest.approx(2.26)


def test_rank_past_the_fetched_window_is_not_exact():
    """48 records exist but we hold 4; slower than all 4 means '>4'."""
    r = L.rank_for_time(BOARD, 99.0)
    assert r["exact"] is False
    assert L.fmt_rank(r) == ">4 of 48"
    inside = L.rank_for_time(BOARD, 27.755)
    assert inside["exact"] is True
    assert L.fmt_rank(inside) == "3 of 48"


def test_rank_exact_when_we_hold_every_record():
    board = dict(BOARD, total_records=4)
    assert L.rank_for_time(board, 99.0)["exact"] is True
    assert L.fmt_rank(L.rank_for_time(board, 99.0)) == "5 of 4"


# ------------------------------------------------------------ finish kind

def _zones(repo, stem, end, **extra):
    d = repo / "maps"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("%s.zones.json" % stem)).write_text(
        json.dumps(dict({"end": end}, **extra)), encoding="utf-8")


def test_finish_kind_matches_the_trainer_rule(tmp_path):
    (tmp_path / "maps_pool").mkdir()
    _zones(tmp_path, "trig", {"aabb": [[0, 0, 0], [1, 1, 1]]})
    _zones(tmp_path, "tru", {"aabb": [[0, 0, 0], [1, 1, 1]],
                             "true_aabb": [[0, 0, 0], [1, 1, 1]]})
    _zones(tmp_path, "fb", {"aabb": [[0, 0, 0], [1, 1, 1]],
                            "from": "func_button"})
    _zones(tmp_path, "gw", {"aabb": [[0, 0, 0], [1, 1, 1]]}, source="gateway")
    assert L.finish_kind("trig", tmp_path) == "trigger"
    assert L.finish_kind("tru", tmp_path) == "button"
    assert L.finish_kind("fb", tmp_path) == "button"
    assert L.finish_kind("gw", tmp_path) == "button"
    assert L.finish_kind("nope", tmp_path) == "unknown"


def test_finish_kind_of_the_two_real_maps():
    """cannonball and petrus_lite are type 1 - the whole comparison rests
    on it (CLAUDE.md 4b)."""
    assert L.finish_kind("surf_src_cannonball", ROOT) == "trigger"
    assert L.finish_kind("surf_petrus_lite", ROOT) == "trigger"


# --------------------------------------------------------------- the run

def _make_run(tmp_path, cfg, header, rows, name="r1"):
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"config": cfg}),
                                  encoding="utf-8")
    lines = [",".join(header)] + [",".join(r) for r in rows]
    (run / "progress.csv").write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
    return run


def _make_boards(tmp_path, boards):
    d = tmp_path / "boards"
    d.mkdir(exist_ok=True)
    for stem, b in boards.items():
        (d / ("%s.json" % stem)).write_text(json.dumps(b), encoding="utf-8")
    return d


def test_run_finish_times_single_map(tmp_path):
    run = _make_run(tmp_path, {"map": "surf_petrus_lite", "maps": None},
                    ["step", "race/eval_finish_s"],
                    [["1", ""], ["2", "35.28"], ["3", ""], ["4", "31.93"],
                     ["5", "32.40"]])
    got = L.run_finish_times(run)["surf_petrus_lite"]
    assert got["best"] == pytest.approx(31.93)
    assert got["last"] == pytest.approx(32.40)
    assert got["n"] == 3                         # blanks are not finishes
    assert got["col"] == "race/eval_finish_s"


def test_run_finish_times_joint_uses_the_tagged_column(tmp_path):
    run = _make_run(
        tmp_path,
        {"map": "surf_src_cannonball",
         "maps": ["surf_src_cannonball", "surf_petrus_lite"]},
        ["step", "race/eval_finish_s", "race/eval_finish_s.cannonball",
         "race/eval_finish_s.petrus_lite"],
        [["1", "33.14", "", "33.14"], ["2", "32.02", "", "32.02"]])
    got = L.run_finish_times(run)
    assert got["surf_src_cannonball"]["best"] is None      # never finished
    assert got["surf_src_cannonball"]["col"] == \
        "race/eval_finish_s.cannonball"
    assert got["surf_petrus_lite"]["best"] == pytest.approx(32.02)
    assert got["surf_petrus_lite"]["last"] == pytest.approx(32.02)


def _report(run, boards, repo, **kw):
    buf = io.StringIO()
    rows = L.run_report(run, boards, repo, out=buf, **kw)
    return buf.getvalue(), rows


def test_run_report_table_and_summary(tmp_path):
    run = _make_run(
        tmp_path,
        {"map": "surf_a", "maps": ["surf_a", "surf_b", "surf_c"]},
        ["step", "race/eval_finish_s.a", "race/eval_finish_s.b",
         "race/eval_finish_s.c"],
        [["1", "30.0", "", "12.0"], ["2", "28.0", "", "11.0"]])
    _zones(tmp_path, "surf_a", {"aabb": [[0, 0, 0], [1, 1, 1]]})
    _zones(tmp_path, "surf_b", {"aabb": [[0, 0, 0], [1, 1, 1]]})
    _zones(tmp_path, "surf_c", {"aabb": [[0, 0, 0], [1, 1, 1]],
                                "true_aabb": [[0, 0, 0], [1, 1, 1]]})
    boards = _make_boards(tmp_path, {
        "surf_a": dict(BOARD, map="surf_a"),
        "surf_b": dict(BOARD, map="surf_b"),
        # surf_c: we are FASTER than the world record, but it is a button map
        "surf_c": {"map": "surf_c", "total_records": 7, "wr_s": 20.0,
                   "wr_player": "someone",
                   "entries": [{"rank": 1, "player": "someone",
                                "time_s": 20.0, "date": ""}]},
    })
    text, rows = _report(run, boards, tmp_path)

    by = {r["map"]: r for r in rows}
    assert by["surf_a"]["best"] == pytest.approx(28.0)
    assert by["surf_a"]["last"] == pytest.approx(28.0)
    assert by["surf_a"]["rank"] == 4          # 3 of the 4 held are faster
    assert by["surf_b"]["best"] is None       # no finish at all
    assert by["surf_c"]["beats_wr"] is True
    assert by["surf_c"]["comparable"] is False

    assert "maps finished              : 2 of 3" in text
    # surf_a is rank 5 of 48 -> top 100; surf_c is rank 1 -> top 100
    assert "maps ranked in the top 100 : 2" in text
    # the one WR beaten is a BUTTON map, so it does not count by default
    assert "maps beating the WR        : 0" in text
    assert "button" in text
    assert "CLAUDE.md 4b" in text
    assert "SPAWN" in text


def test_run_report_all_counts_button_maps(tmp_path):
    run = _make_run(tmp_path, {"map": "surf_c", "maps": ["surf_c"]},
                    ["step", "race/eval_finish_s.c"], [["1", "11.0"]])
    _zones(tmp_path, "surf_c", {"aabb": [[0, 0, 0], [1, 1, 1]],
                                "true_aabb": [[0, 0, 0], [1, 1, 1]]})
    boards = _make_boards(tmp_path, {
        "surf_c": {"map": "surf_c", "total_records": 7, "wr_s": 20.0,
                   "wr_player": "someone",
                   "entries": [{"rank": 1, "player": "someone",
                                "time_s": 20.0, "date": ""}]}})
    text, _ = _report(run, boards, tmp_path, use_all=True)
    assert "maps beating the WR        : 1" in text


def test_run_report_spawn_offset_shifts_our_time(tmp_path):
    """Our clock is the SPAWN clock; the offset must move OUR column only."""
    run = _make_run(tmp_path, {"map": "surf_a", "maps": ["surf_a"]},
                    ["step", "race/eval_finish_s.a"], [["1", "29.0"]])
    _zones(tmp_path, "surf_a", {"aabb": [[0, 0, 0], [1, 1, 1]]})
    boards = _make_boards(tmp_path, {"surf_a": dict(BOARD, map="surf_a")})
    _, raw = _report(run, boards, tmp_path)
    assert raw[0]["best"] == pytest.approx(29.0) and raw[0]["rank"] == 5
    _, rows = _report(run, boards, tmp_path,
                      spawn_offset=L.SPAWN_CLOCK_EXCESS_S)
    assert rows[0]["best"] == pytest.approx(29.0 - 0.96)
    assert rows[0]["wr_s"] == pytest.approx(27.74)   # the record does NOT move
    assert rows[0]["rank"] == 4              # 28.04 now slips past 28.27


def test_run_report_map_with_no_board(tmp_path):
    run = _make_run(tmp_path, {"map": "surf_a", "maps": ["surf_a"]},
                    ["step", "race/eval_finish_s.a"], [["1", "28.0"]])
    _zones(tmp_path, "surf_a", {"aabb": [[0, 0, 0], [1, 1, 1]]})
    text, rows = _report(run, _make_boards(tmp_path, {}), tmp_path)
    assert "no board" in text
    assert rows[0]["rank"] is None
    assert "maps beating the WR        : 0" in text


def test_map_tag_matches_the_trainer():
    sys.path.insert(0, str(ROOT / "python"))
    from surfgym.mapfleet import map_tag as ref
    for stem in ("surf_src_cannonball", "surf_petrus_lite", "surf_0way",
                 "surf_ctm_guater-ran", "weird_name"):
        assert L.map_tag(stem) == ref(stem)
