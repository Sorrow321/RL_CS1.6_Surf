#!/usr/bin/env python3
"""Human record leaderboards (speedrun.eu) and where our runs would rank.

Three subcommands:

    fetch   download the top-N human records for every map we train on and
            write one small JSON per map under ``docs/leaderboards/``
    rank    what rank a given time would take on one map
    run     read a training run's OWN logs (``run.json`` + ``progress.csv``)
            and print, per map, our best and last eval finish time against
            the human record, with a rank estimate

THE COMPARABILITY CAVEATS ARE LOAD-BEARING, not decoration:

  * our eval clock runs from SPAWN to the finish; a human record runs from
    the START TRIGGER to the finish trigger. On cannonball the spawn clock
    is about 0.96 s longer than the record clock (docs/research-results.md,
    "record clock = spawn clock - ~0.96 s"). ``--spawn-offset`` subtracts a
    constant if you want the corrected number; it defaults to 0.0 because
    the offset has only ever been MEASURED on cannonball.
  * a map whose finish is a BUTTON is not comparable at all. The simulator
    cannot press a button, so arriving inside a padded box is substituted
    for the press (CLAUDE.md section 4b). Such a map is marked "button" and
    is left out of the "beats WR" count unless you pass ``--all``.

No network is needed for ``rank`` or ``run`` - they read the committed
JSONs. ``fetch`` caches raw HTML so a re-run does not re-download.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BOARD_DIR = REPO / "docs" / "leaderboards"
MAP_DIRS = ("maps", "maps_pool")
SITE = "https://speedrun.eu"
RECORDS_URL = SITE + "/maps/{stem}/records"
PER_PAGE = 15          # speedrun.eu renders 15 rows per page; not tunable
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
CONTACT = "RL_Surf research bot - 1 request per 1.5 s"

# The spawn-clock -> record-clock offset MEASURED on cannonball. Not applied
# by default; see the module docstring.
SPAWN_CLOCK_EXCESS_S = 0.96


# ----------------------------------------------------------------- parsing

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TBODY_RE = re.compile(r"<tbody[^>]*>(.*?)</tbody>", re.S | re.I)
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<td\b[^>]*>(.*?)(?=<td\b|</tr>|\Z)", re.S | re.I)
_TOTAL_RE = re.compile(r"([\d,]+)\s*total", re.I)
_PAGES_RE = re.compile(r"Page\s+(\d+)\s+of\s+(\d+)", re.I)
_PLAYER_RE = re.compile(r'href="/players/([^"]+)"', re.I)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NO_RECORDS_RE = re.compile(r"No records found for this map", re.I)


def _text(fragment: str) -> str:
    """Tag-stripped, entity-decoded, whitespace-collapsed cell text."""
    import html as _html
    s = _TAG_RE.sub(" ", fragment)
    s = _html.unescape(s)
    return _WS_RE.sub(" ", s).strip()


def parse_time(s: str):
    """``00:27.74`` / ``1:02:03.45`` / ``27.74`` -> seconds as a float.

    Returns None for anything that is not a clock.
    """
    s = (s or "").strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) > 3:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    total = 0.0
    for v in vals:                      # h:m:s, m:s or s
        total = total * 60.0 + v
    return total


def parse_records_page(page_html: str) -> dict:
    """Parse one speedrun.eu ``/maps/<stem>/records`` page.

    Returns ``{entries, total_records, page, pages, no_records}``. Ranks 1-3
    are rendered as medal emoji rather than numbers, so a row's rank comes
    from its number cell when that cell is numeric and from its position in
    the page otherwise.
    """
    out = {"entries": [], "total_records": None, "page": None,
           "pages": None, "no_records": bool(_NO_RECORDS_RE.search(page_html))}

    m = _TOTAL_RE.search(page_html)
    if m:
        out["total_records"] = int(m.group(1).replace(",", ""))
    m = _PAGES_RE.search(page_html)
    if m:
        out["page"], out["pages"] = int(m.group(1)), int(m.group(2))

    body = _TBODY_RE.search(page_html)
    if not body:
        return out
    rows = _ROW_RE.findall(body.group(1) + "</tr>")
    for i, row in enumerate(rows):
        cells = _CELL_RE.findall(row)
        if len(cells) < 3:
            continue
        rank_txt = _text(cells[0])
        player = _text(cells[1])
        time_s = parse_time(_text(cells[2]))
        date = _text(cells[3]) if len(cells) > 3 else ""
        if not _DATE_RE.match(date):
            date = ""
        if time_s is None:
            continue
        if not player:
            mp = _PLAYER_RE.search(cells[1])
            player = mp.group(1) if mp else ""
        out["entries"].append({"rank": int(rank_txt) if rank_txt.isdigit()
                                       else None,
                               "player": player, "time_s": time_s,
                               "date": date})
    return out


# ------------------------------------------------------------------- fetch

def _default_cache() -> Path:
    env = (os.environ.get("RL_SURF_SCRATCH") or os.environ.get("TMPDIR")
           or os.environ.get("TEMP") or "/tmp")
    return Path(env) / "leaderboard_cache"


def _http_get(url: str, timeout: float = 30.0):
    """GET -> (status, text). Uses requests when present, urllib otherwise."""
    headers = {"User-Agent": USER_AGENT, "From": CONTACT,
               "Accept": "text/html,application/xhtml+xml",
               "Accept-Language": "en"}
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        r = requests.get(url, headers=headers, timeout=timeout)
        return r.status_code, r.text
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def http_get_cached(url: str, cache: Path, key: str, delay: float,
                    refresh: bool, tries: int = 4, log=print) -> str:
    """Polite GET with an on-disk raw-HTML cache and backoff on 429/5xx."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".html")
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    wait, status = 2.0, None
    for attempt in range(tries):
        try:
            status, text = _http_get(url)
        except Exception as exc:                       # network hiccup
            status, text = 0, ""
            log("    ! %s: %s" % (type(exc).__name__, exc))
        if status == 200:
            path.write_text(text, encoding="utf-8")
            time.sleep(delay)
            return text
        if status in (404, 410):
            return ""
        if attempt < tries - 1:
            log("    ! HTTP %s on %s; retry in %.0fs" % (status, url, wait))
            time.sleep(wait)
            wait *= 2
    raise RuntimeError("gave up on %s (last status %s)" % (url, status))


def name_variants(stem: str):
    """Obvious alternative spellings to try ONCE before recording a miss."""
    seen, out = {stem}, []
    for v in (stem.replace("-", "_"), stem.lower()):
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def fetch_map(stem: str, top: int, cache: Path, delay: float, refresh: bool,
              log=print) -> dict:
    """Download up to ``top`` records for one map stem."""
    tried = []
    for name in [stem] + name_variants(stem):
        tried.append(name)
        url0 = RECORDS_URL.format(stem=name)
        entries, total, pages, page = [], None, None, 1
        # An out-of-range page renders an empty table, so `not entries`
        # already ends the walk - but a runaway loop here would hammer a
        # live site, so cap it at the pages `top` could possibly need.
        max_page = top // PER_PAGE + 3
        while page <= max_page:
            url = url0 if page == 1 else "%s?page=%d" % (url0, page)
            body = http_get_cached(url, cache, "%s__p%d" % (name, page),
                                   delay, refresh, log=log)
            if not body:
                break
            got = parse_records_page(body)
            if total is None:
                total = got["total_records"]
            pages = got["pages"] or pages
            if not got["entries"]:
                break
            entries.extend(got["entries"])
            if len(entries) >= top:
                break
            if pages and page >= pages:
                break
            page += 1
        if entries:
            entries = entries[:top]
            for i, e in enumerate(entries):
                if e.get("rank") is None:
                    e["rank"] = i + 1
            return {"map": stem, "url": RECORDS_URL.format(stem=name),
                    "fetched_at": _now(),
                    "total_records": (total if total is not None
                                      else len(entries)),
                    "wr_s": entries[0]["time_s"],
                    "wr_player": entries[0]["player"],
                    "status": "ok", "entries": entries}
    return {"map": stem, "url": RECORDS_URL.format(stem=stem),
            "fetched_at": _now(), "total_records": 0, "wr_s": None,
            "wr_player": None, "status": "not_found",
            "variants_tried": tried, "entries": []}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -------------------------------------------------------------- repo facts

def map_stems(repo: Path = REPO):
    """Union of ``maps/*.bsp`` and ``maps_pool/*.bsp`` (never _removed)."""
    stems = set()
    for d in MAP_DIRS:
        for p in (repo / d).glob("*.bsp"):
            stems.add(p.stem)
    return sorted(stems)


def zones_path(stem: str, repo: Path = REPO):
    for d in MAP_DIRS:
        p = repo / d / ("%s.zones.json" % stem)
        if p.exists():
            return p
    return None


def finish_kind(stem: str, repo: Path = REPO) -> str:
    """One of "trigger", "button", "unknown" - train_fast.py's own rule.

    A type-1 finish is an invisible ``trigger_multiple`` curtain. Types 2/3
    are a ``+use`` button, and THE SIMULATOR CANNOT PRESS A BUTTON: arriving
    inside a 64 u-padded box is substituted for the press, so the times are
    not comparable to a human record (CLAUDE.md 4b). Both the gateway
    service and the in-BSP ``func_button`` fallback emit ``true_aabb``; a
    real trigger brush has no such key.
    """
    p = zones_path(stem, repo)
    if p is None:
        return "unknown"
    try:
        zones = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return "unknown"
    end = zones.get("end") or {}
    return ("button"
            if (end.get("true_aabb") is not None
                or end.get("from") == "func_button"
                or zones.get("source") == "gateway")
            else "trigger")


def map_tag(stem: str) -> str:
    """``surfgym.mapfleet.map_tag``, with a local copy as the fallback."""
    try:
        sys.path.insert(0, str(REPO / "python"))
        from surfgym.mapfleet import map_tag as _mt
        return _mt(stem)
    except Exception:
        for pre in ("surf_src_", "surf_"):
            if stem.startswith(pre):
                return stem[len(pre):]
        return stem


def load_board(stem: str, board_dir: Path = BOARD_DIR):
    p = Path(board_dir) / ("%s.json" % stem)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- ranking

def rank_for_time(board: dict, t: float) -> dict:
    """Rank a time would take: 1 + the number of strictly faster entries.

    ``exact`` is False when the time is slower than every entry we hold AND
    the map has more records than we fetched - then the true rank is only
    known to be past our window.
    """
    entries = (board or {}).get("entries") or []
    total = (board or {}).get("total_records")
    faster = sum(1 for e in entries if e["time_s"] < t)
    exact = not (faster == len(entries)
                 and total is not None and total > len(entries))
    wr = (board or {}).get("wr_s")
    return {"rank": faster + 1, "exact": exact, "n_entries": len(entries),
            "total_records": total, "wr_s": wr,
            "wr_player": (board or {}).get("wr_player"),
            "beats_wr": (wr is not None and t < wr),
            "delta_wr": (t - wr) if wr is not None else None}


def fmt_rank(r: dict) -> str:
    total = (r["total_records"] if r["total_records"] is not None
             else r["n_entries"])
    head = "%d" % r["rank"] if r["exact"] else ">%d" % r["n_entries"]
    return "%s of %s" % (head, total)


# ------------------------------------------------------------------- runs

def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def run_maps(run_dir: Path):
    """``(stems, joint)`` from a run's ``run.json``."""
    cfg = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    cfg = cfg.get("config", cfg)
    maps = cfg.get("maps")
    if isinstance(maps, (list, tuple)) and len(maps) > 0:
        return [str(m) for m in maps], len(maps) > 1
    one = cfg.get("map")
    return ([str(one)] if one else []), False


def run_finish_times(run_dir: Path):
    """``{stem: {best, last, n, col}}`` from ``progress.csv``.

    ``race/eval_finish_s.<tag>`` for a joint run, ``race/eval_finish_s`` for
    a single-map one; an empty cell means that eval had no finish.
    """
    run_dir = Path(run_dir)
    stems, joint = run_maps(run_dir)
    with (run_dir / "progress.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cols = set(rows[0].keys()) if rows else set()
    out = {}
    for stem in stems:
        col = "race/eval_finish_s.%s" % map_tag(stem)
        if col not in cols and not joint and "race/eval_finish_s" in cols:
            col = "race/eval_finish_s"
        vals = [v for v in (_f(r.get(col)) for r in rows) if v is not None]
        out[stem] = {"best": min(vals) if vals else None,
                     "last": vals[-1] if vals else None,
                     "n": len(vals), "col": col}
    return out


CAVEATS = [
    "our clock is SPAWN -> finish; a record is START TRIGGER -> finish",
    "  trigger. On cannonball the spawn clock is ~0.96 s longer",
    "  (docs/research-results.md). --spawn-offset subtracts a constant.",
    "a 'button' map is NOT comparable: the simulator cannot press a button,",
    "  so arriving in a 64 u-padded box is substituted for the press",
    "  (CLAUDE.md 4b). Left out of 'beats WR' unless --all.",
    "a rank shown as '>N' is only known to be past the N entries we hold.",
]


def run_report(run_dir: Path, board_dir: Path = BOARD_DIR,
               repo: Path = REPO, spawn_offset: float = 0.0,
               use_all: bool = False, out=None):
    out = out if out is not None else sys.stdout
    run_dir = Path(run_dir)
    stems, joint = run_maps(run_dir)
    times = run_finish_times(run_dir)
    w = max([len(s) for s in stems] + [4])

    def p(line=""):
        print(line, file=out)

    p("run %s   (%s%d map%s)" % (run_dir, "joint, " if joint else "",
                                 len(stems), "s" if len(stems) != 1 else ""))
    if spawn_offset:
        p("spawn offset applied: -%.2f s on every one of our times"
          % spawn_offset)
    p()
    hdr = ("%-*s  %9s  %9s  %9s  %-20s  %12s  %6s  %s"
           % (w, "map", "our best", "our last", "WR s", "WR holder", "rank",
              "recs", "comparable"))
    p(hdr)
    p("-" * len(hdr))
    n_fin = n_top100 = n_beat = 0
    rows = []
    for stem in stems:
        t = times.get(stem, {})
        best, last = t.get("best"), t.get("last")
        if best is not None:
            best -= spawn_offset
        if last is not None:
            last -= spawn_offset
        board = load_board(stem, board_dir)
        kind = finish_kind(stem, repo)
        comparable = (kind == "trigger")
        wr = board.get("wr_s") if board else None
        holder = (board.get("wr_player") or "") if board else ""
        recs = board.get("total_records") if board else None
        r = None
        if best is None:
            rank_txt = "-"
        elif not board or not board.get("entries"):
            rank_txt = "no board"
        else:
            r = rank_for_time(board, best)
            rank_txt = fmt_rank(r)
        if best is not None:
            n_fin += 1
            if r is not None and r["exact"] and r["rank"] <= 100:
                n_top100 += 1
            if r is not None and r["beats_wr"] and (comparable or use_all):
                n_beat += 1
        p("%-*s  %9s  %9s  %9s  %-20s  %12s  %6s  %s"
          % (w, stem,
             "-" if best is None else "%.2f" % best,
             "-" if last is None else "%.2f" % last,
             "-" if wr is None else "%.2f" % wr,
             _ascii(holder)[:20],
             rank_txt, "-" if recs is None else recs,
             "yes" if comparable else kind))
        rows.append({"map": stem, "best": best, "last": last, "wr_s": wr,
                     "rank": (r or {}).get("rank"), "kind": kind,
                     "comparable": comparable,
                     "beats_wr": bool(r and r["beats_wr"])})
    p()
    p("maps finished              : %d of %d" % (n_fin, len(stems)))
    p("maps ranked in the top 100 : %d" % n_top100)
    p("maps beating the WR        : %d  (%s)"
      % (n_beat, "all maps, comparable or not" if use_all
         else "comparable maps only"))
    p()
    p("caveats:")
    for c in CAVEATS:
        p(("  " + c) if c.startswith(" ") else ("  - " + c))
    return rows


def _ascii(s: str) -> str:
    """The console here is cp1251; a Cyrillic/emoji player name must not
    crash the table."""
    return (s or "").encode("ascii", "replace").decode("ascii")


# --------------------------------------------------------------- commands

def cmd_fetch(args):
    stems = ([s.strip() for s in args.maps.split(",") if s.strip()]
             if args.maps else map_stems(REPO))
    board_dir = Path(args.out) if args.out else BOARD_DIR
    board_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else _default_cache()
    print("%d maps -> %s  (cache %s)" % (len(stems), board_dir, cache))
    index, idx_path = {}, board_dir / "index.json"
    if idx_path.exists():
        try:
            index = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}
    n_ok = n_miss = n_skip = n_entries = 0
    for i, stem in enumerate(stems, 1):
        dest = board_dir / ("%s.json" % stem)
        if dest.exists() and not args.refresh:
            board = json.loads(dest.read_text(encoding="utf-8"))
            n_skip += 1
        else:
            print("[%d/%d] %s" % (i, len(stems), stem))
            board = fetch_map(stem, args.top, cache, args.delay, args.refresh)
            dest.write_text(json.dumps(board, indent=1, ensure_ascii=False)
                            + "\n", encoding="utf-8")
            print("    %s  %d entries  wr %s" %
                  (board["status"], len(board["entries"]),
                   board["wr_s"] if board["wr_s"] is not None else "-"))
        n = len(board.get("entries") or [])
        n_entries += n
        if board.get("status") == "ok":
            n_ok += 1
        else:
            n_miss += 1
        index[stem] = {"wr_s": board.get("wr_s"), "n_entries": n,
                       "total_records": board.get("total_records"),
                       "status": board.get("status")}
    idx_path.write_text(json.dumps(dict(sorted(index.items())), indent=1,
                                   ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print("\nok %d  not_found %d  (already on disk %d)  entries %d"
          % (n_ok, n_miss, n_skip, n_entries))
    return 0


def cmd_rank(args):
    board = load_board(args.map, Path(args.board_dir))
    if board is None:
        print("no leaderboard for %s - run: python tools/leaderboard.py "
              "fetch --maps %s" % (args.map, args.map))
        return 1
    if not board.get("entries"):
        print("%s: no human records on record (%s)"
              % (args.map, board.get("status")))
        return 1
    t = args.time - args.spawn_offset
    r = rank_for_time(board, t)
    kind = finish_kind(args.map, REPO)
    print("%s  (%s finish)" % (args.map, kind))
    print("  our time     %9.2f s%s"
          % (t, ("   (= %.2f - %.2f spawn offset)"
                 % (args.time, args.spawn_offset)) if args.spawn_offset
             else ""))
    print("  WR           %9.2f s   %s" % (r["wr_s"], _ascii(r["wr_player"])))
    print("  rank         %s%s" % (fmt_rank(r),
                                   "" if r["exact"]
                                   else "   (past our fetched window)"))
    if r["beats_wr"]:
        print("  BEATS THE WR by %.2f s" % (-r["delta_wr"]))
    else:
        print("  behind the WR by %.2f s" % r["delta_wr"])
    if kind != "trigger":
        print("  NOT COMPARABLE: %s finish (CLAUDE.md 4b) - the simulator "
              "cannot press a button" % kind)
    print("  note: our clock is spawn -> finish, a record is start trigger "
          "-> finish (~0.96 s longer on cannonball)")
    return 0


def cmd_run(args):
    run_dir = Path(args.run)
    if not run_dir.exists() and not run_dir.is_absolute():
        run_dir = REPO / run_dir
    if not (run_dir / "run.json").exists():
        print("no run.json under %s" % run_dir)
        return 1
    run_report(run_dir, Path(args.board_dir), REPO, args.spawn_offset,
               args.all)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="human record leaderboards and where our runs would rank")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download records from speedrun.eu")
    f.add_argument("--maps", help="comma-separated stems (default: every map "
                                  "in maps/ and maps_pool/)")
    f.add_argument("--top", type=int, default=100,
                   help="entries per map (default 100)")
    f.add_argument("--refresh", action="store_true",
                   help="re-download maps already on disk")
    f.add_argument("--out", help="output dir (default docs/leaderboards)")
    f.add_argument("--cache", help="raw-HTML cache dir")
    f.add_argument("--delay", type=float, default=1.5,
                   help="seconds between requests (default 1.5)")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("rank", help="rank one time on one map")
    r.add_argument("--map", required=True)
    r.add_argument("--time", type=float, required=True, help="seconds")
    r.add_argument("--spawn-offset", type=float, default=0.0,
                   help="subtract this many seconds (cannonball measures "
                        "%.2f)" % SPAWN_CLOCK_EXCESS_S)
    r.add_argument("--board-dir", default=str(BOARD_DIR))
    r.set_defaults(func=cmd_rank)

    u = sub.add_parser("run", help="rank a training run's finish times")
    u.add_argument("--run", required=True, help="runs/<name>")
    u.add_argument("--all", action="store_true",
                   help="count button-finish maps in 'beats WR' too")
    u.add_argument("--spawn-offset", type=float, default=0.0)
    u.add_argument("--board-dir", default=str(BOARD_DIR))
    u.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
