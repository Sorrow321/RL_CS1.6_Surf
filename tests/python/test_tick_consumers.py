"""The follow-up half of --tick-ms: the recorded ``phys`` block, and the
trajectory consumers that still assumed 100 Hz.

F6 - the ``"phys": {"msec": ...}`` block of an episode header was
NONDETERMINISTIC under a tick pattern. ``SurfCore.step`` mirrors the tick it
just ran into ``config.phys.msec`` (a diagnostic that moves with the pattern
phase) and ``record_rollout`` snapshotted it once per file, so three
recordings from ONE 7.63 ms core wrote msec 8, 8, 7. The block now states
the core's NOMINAL tick (``SurfCore.nominal_msec``), and
``set_tick_pattern(None)`` restores that same nominal tick instead of
whichever phase the mirror was sitting in. At any fixed tick the nominal IS
``config.phys.msec``, so a 10 ms recording is byte-for-byte what it was.

F7 - consumers that turned ticks into seconds with a literal 100 / 0.01:

  * ``tools/build_route.py`` (``pick_route(..., tick_ms=10.0)``) picked the
    "fastest" finisher by timing every episode at 10 ms, so a 7.667 ms
    recording read 30% slow and could win a race it lost;
  * ``tools/pick_selfline.py`` printed the trimmed tail as ``ticks * 0.01``
    and stamped ``tick_ms: 10.0`` into the line it wrote, losing the tick
    the line was flown at;
  * ``tools/tas_search.py``'s ``load_episode`` dropped the header entirely,
    so ``tools/tas_chain.py`` timed the recorded run at ``/ 100``;
  * ``tools/build_spine.py`` timed a planner npz at ``/ 100``;
  * ``surfgym/dagger.py``'s ``TICKS_PER_S = 100`` converted every seconds
    flag of ``tools/expert_dagger.py`` (--every, --window, --rollout-secs,
    --spine-secs), so a 3 s planner window was 2.30 s of physics at
    7.667 ms.

Each now reads the tick from the recording's own header (the convention of
tools/finish_times.py and tools/traj_ends.py: ``surfgym.tick.
episode_seconds``, which SUMS the pattern instead of multiplying by its
mean, and refuses a header that carries no tick) or, where no header
exists, from the core that will actually run the ticks.

    python -m pytest tests/python/test_tick_consumers.py -q
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.route import episodes_from_traj                # noqa: E402
from surfgym.tick import (TickClock, episode_seconds,       # noqa: E402
                          header_fields)

CANNONBALL = ROOT / "maps" / "surf_src_cannonball.bsp"
DLL = ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so")
needs_core = pytest.mark.skipif(not (CANNONBALL.exists() and DLL.exists()),
                                reason="needs the built core + cannonball")

PATTERN = [8, 8, 7]                       # --tick-ms 7.63
TICK131 = header_fields(23.0 / 3.0, PATTERN, 0)   # tick_ms 7.666667 + pattern
TICK10 = header_fields(10.0, [10], 0)             # {"tick_ms": 10}


def _write_traj(path, tick_keys, rows, header=True, footer=True):
    """One episode in record_rollout's exact JSONL format."""
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if header:
            hdr = {"map": "m", "phys": {"msec": 8}, "episode": 0}
            hdr.update(tick_keys)
            f.write(json.dumps(hdr, separators=(",", ":")) + "\n")
        for r in rows:
            f.write(json.dumps([float(v) for v in r],
                               separators=(",", ":")) + "\n")
        if footer:
            f.write(json.dumps({"end": "fail", "ticks": len(rows),
                                "best_progress": 0.0}) + "\n")
    return path


def _line_rows(n, dx=10.0):
    """``n`` ticks flying +x at ``dx`` units per tick, 8 recorder columns."""
    return [[t, t * dx, 0.0, 0.0, dx, 0.0, 0.0, 0.0] for t in range(n)]


# ==========================================================================
# F7.1 build_route.py: every episode timed at ITS OWN tick
# ==========================================================================
def test_build_route_times_each_episode_at_its_own_tick(tmp_path):
    """Two recordings of the SAME 300-tick flight: 3.00 s at 10 ms, 2.30 s
    at the [8, 8, 7] pattern. The old 10 ms default called them equal and
    kept the first, which is the wrong "fastest" line."""
    import build_route

    a = _write_traj(tmp_path / "a.jsonl", TICK10, _line_rows(400))
    b = _write_traj(tmp_path / "b.jsonl", TICK131, _line_rows(400))
    mins, maxs = [3000.0, -8.0, -8.0], [3008.0, 8.0, 8.0]

    xyz, secs, n_fin, n_ep = build_route.pick_route([a, b], mins, maxs, pad=0.0)
    assert (n_fin, n_ep) == (2, 2)
    # 300 ticks of [8, 8, 7] from phase 0 = 100 x 23 ms, exactly
    assert secs == pytest.approx(2.3, abs=1e-12)
    assert len(xyz) == 301                      # trimmed at the crossing

    # each file on its own, and the 10 ms one is the legacy arithmetic
    _, s_a, _, _ = build_route.pick_route([a], mins, maxs, pad=0.0)
    _, s_b, _, _ = build_route.pick_route([b], mins, maxs, pad=0.0)
    assert s_a == 3.0 and s_b == pytest.approx(2.3, abs=1e-12)
    assert abs(s_a / s_b - 1.3043) < 1e-3       # the 30% the literal cost

    # --tick-ms still overrides, and reproduces the old default exactly
    _, s_forced, _, _ = build_route.pick_route([b], mins, maxs, tick_ms=10.0,
                                               pad=0.0)
    assert s_forced == 3.0

    # a recording with no header at all refuses rather than assuming 10 ms
    c = _write_traj(tmp_path / "c.jsonl", TICK10, _line_rows(400),
                    header=False)
    with pytest.raises(ValueError):
        build_route.pick_route([c], mins, maxs, pad=0.0)


# ==========================================================================
# F7.2 pick_selfline.py: the tail clock, and the tick the line carries
# ==========================================================================
def test_pick_selfline_times_the_tail_and_stamps_the_source_tick(
        tmp_path, capsys, monkeypatch):
    """The trimmed tail is the exact pattern sum between the cut and the
    end (599 ticks = 4.593 s at [8, 8, 7], not the 5.99 s a 0.01 s literal
    reports), and the written line carries the source recording's tick so
    the next tool times it correctly."""
    import pick_selfline

    ep = pick_selfline.synth(n=900, last_contact=300)
    assert pick_selfline.contact_cut(ep)[0] == 300
    src = _write_traj(tmp_path / "s.jsonl", TICK131, ep)
    out = tmp_path / "line.jsonl"
    monkeypatch.setattr(sys, "argv",
                        ["pick_selfline.py", str(src), "--out", str(out)])
    assert pick_selfline.main() == 0
    printed = capsys.readouterr().out

    # 899 ticks from phase 0 minus the first 300 of them
    tail = episode_seconds(TICK131, 899) - episode_seconds(TICK131, 300)
    assert tail == pytest.approx(4.593, abs=1e-12)
    assert f"{tail:.2f}s of tail" in printed
    assert "5.99s of tail" not in printed         # the 0.01 s literal

    eps, hdrs = episodes_from_traj(out, with_headers=True)
    assert len(eps) == 1 and len(eps[0]) == 301
    assert hdrs[0]["tick_ms"] == TICK131["tick_ms"]
    assert hdrs[0]["tick_pattern_ms"] == PATTERN and hdrs[0]["tick_phase"] == 0
    assert episode_seconds(hdrs[0], 300) == pytest.approx(2.3, abs=1e-12)

    # a source with no tick is the 10 ms reference, as every recording made
    # before the flag was
    assert pick_selfline.tick_keys(None) == {"tick_ms": 10}
    assert pick_selfline.tick_keys({"tick_ms": 10}) == {"tick_ms": 10}
    assert pick_selfline.tick_keys(TICK131) == TICK131


# ==========================================================================
# F7.3 tas_search.load_episode -> tas_chain's clock
# ==========================================================================
def test_tas_search_load_episode_returns_the_header(tmp_path):
    import tas_search

    rows = _line_rows(50)
    src = _write_traj(tmp_path / "t.jsonl", TICK131, rows)
    ep = tas_search.load_episode(src, 1)
    assert ep.shape == (50, 8)
    ep2, hdr = tas_search.load_episode(src, 1, with_header=True)
    assert np.array_equal(ep, ep2)
    assert hdr["tick_pattern_ms"] == PATTERN
    assert episode_seconds(hdr, len(ep2)) == 0.384   # 16x23 + 8 + 8 ms


def test_tas_chain_times_the_recording_from_its_header():
    """The recorded half of tas_chain's report is the recording's clock,
    not ``/ 100``; its own chained ticks stay the search core's."""
    src = (ROOT / "tools" / "tas_chain.py").read_text(encoding="utf-8")
    assert "(len(ep) - t_start) / 100" not in src
    assert "total_ticks / 100" not in src
    assert "episode_seconds" in src and "header_tick_ms" in src


# ==========================================================================
# the whole F7 list, pinned: the literal is gone and the tick is read
# ==========================================================================
FIXED = {
    "tools/build_route.py": (["tick_ms=10.0", "* tick_ms * 1e-3"],
                             "episode_seconds("),
    "tools/pick_selfline.py": (["* 0.01", "tick_ms=10.0"],
                               "episode_seconds("),
    "tools/tas_search.py": (["W/100", "W / 100"], "episode_seconds("),
    "tools/tas_chain.py": (["/ 100:"], "episode_seconds("),
    "tools/build_spine.py": (["fin / 100"], "ticks_to_secs("),
    "tools/beam_campaign.py": (["greedy_ticks / 100"], "ticks_to_secs("),
    "tools/beam_campaign2.py": (["inc_ticks / 100", "s / 100.0"],
                                "episode_seconds("),
    "tools/expert_dagger.py": (["TICKS_PER_S"], "core_clock"),
}


def test_every_fixed_consumer_lost_its_literal_and_reads_a_tick():
    for rel, (banned, wanted) in FIXED.items():
        src = (ROOT / rel).read_text(encoding="utf-8")
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        for b in banned:
            assert b not in code, (rel, b)
        assert wanted in code, (rel, wanted)


# ==========================================================================
# F7.4 surfgym.dagger / tools/expert_dagger.py: no 100 Hz constant
# ==========================================================================
def _assigned(path, names, func):
    """Source text of every assignment to each of ``names`` inside ``func``
    (the static convention of tests/python/test_tick_ms.py)."""
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == func)
    out = {n: [] for n in names}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in out:
                    out[t.id].append(ast.get_source_segment(text, node.value))
    return out


def test_dagger_reads_the_tick_off_the_core_not_a_constant():
    import surfgym.dagger as dg

    assert not hasattr(dg, "TICKS_PER_S")        # the 100 Hz literal is gone
    src = (ROOT / "python" / "surfgym" / "dagger.py").read_text(
        encoding="utf-8")
    assert not any(ln.startswith("TICKS_PER_S") for ln in src.splitlines())

    ref = dg.core_clock(SimpleNamespace())       # a stub core: the reference
    assert ref.is_reference and ref.hz == 100.0
    assert ref.secs_to_ticks(0.5, "round") == 50        # legacy * 100
    assert ref.secs_to_ticks(3.0, "round") == 300
    assert ref.ticks_to_secs(1200) == 12.0

    c = dg.core_clock(SimpleNamespace(tick_ms=23.0 / 3.0))
    assert list(c.pattern) == PATTERN and abs(c.hz - 130.4348) < 1e-3
    assert c.secs_to_ticks(0.5, "round") == 65          # not 50
    assert c.secs_to_ticks(20.0, "round") == 2609       # --spine-secs
    assert abs(c.ticks_to_secs(300) - 2.3) < 1e-12      # the old 3 s window


def test_expert_dagger_converts_every_seconds_flag_at_the_checkpoint_tick():
    p = ROOT / "tools" / "expert_dagger.py"
    src = p.read_text(encoding="utf-8")
    assert "TICKS_PER_S" not in src
    assert "core_clock" in src

    got = _assigned(p, {"every_ticks", "n_ticks", "H"}, "run_relabel")
    for name, srcs in got.items():
        assert srcs, name
        assert all("TICK" in (s or "") for s in srcs), (name, srcs)
    # the episode cap default is a DURATION, converted at that tick
    cap = _assigned(p, {"ep_cap", "ep_120"}, "load_bundle")
    assert any("secs_to_ticks(120" in (s or "") for s in cap["ep_120"]), cap
    assert TickClock(10.0).secs_to_ticks(120.0) == 12000          # unchanged
    assert TickClock(7.63).secs_to_ticks(120.0) == 15652

    # the window arithmetic: identical at 10 ms, a real 3 s at 7.667 ms
    for K in (1, 3, 4):
        assert (int(round(3.0 * TickClock(10.0).hz / K))
                == int(round(3.0 * 100 / K)))
    H = max(1, int(round(3.0 * TickClock(7.63).hz / 4)))
    assert H == 98 and abs(TickClock(7.63).ticks_to_secs(H * 4) - 3.0) < 0.01


# ==========================================================================
# F6 the recorded phys block, and set_tick_pattern(None)
# ==========================================================================
def _core(tick_ms=None, **over):
    from surfgym import SurfCore, default_config
    cfg = default_config(**over)
    return (SurfCore(str(CANNONBALL), cfg, tick_ms=tick_ms)
            if tick_ms is not None else SurfCore(str(CANNONBALL), cfg))


_REC_KW = dict(num_envs=1, spawn_mode=0, max_episode_ticks=100000,
               lidar_w=0, lidar_h=0)


def _legacy_header_line(core, tick_pat):
    """The header record_rollout built BEFORE this fix: config.phys.msec
    copied straight into "tick_ms" and into the phys block."""
    from surfgym.core import phys_to_dict
    h = {"map": Path(core.bsp_path).stem,
         "tick_ms": int(core.config.phys.msec),
         "phys": phys_to_dict(core.config.phys)}
    h.update(header_fields(float(sum(tick_pat)) / len(tick_pat), tick_pat,
                           int(getattr(core, "tick_phase", 0))))
    h["episode"] = 0
    return json.dumps(h, separators=(",", ":"))


@needs_core
def test_recorded_phys_block_is_the_nominal_tick(tmp_path):
    """Three recordings from ONE 7.63 ms core: the mirror cycles 8, 8, 7
    behind them and the header says 8 every time."""
    from surfgym import ScriptedStrafer
    from surfgym.record import record_rollout

    c = _core(7.63, **_REC_KW)
    assert c.nominal_msec == 8 and c.tick_pattern == (8, 8, 7)
    mirrored, headers = [], []
    for i in range(3):
        out = tmp_path / f"r{i}.jsonl"
        record_rollout(c, ScriptedStrafer(c, period_ticks=37), out,
                       episodes=None, max_ticks=100, seed=None)
        headers.append(json.loads(out.read_text(encoding="utf-8")
                                  .splitlines()[0]))
        mirrored.append(int(c.config.phys.msec))
    # the mirror really does move (that is the bug's source) ...
    assert mirrored == [8, 8, 7]
    # ... and the recorded block does not
    assert [h["phys"]["msec"] for h in headers] == [8, 8, 8]
    assert all(h["tick_ms"] == 7.666667 for h in headers)
    assert all(h["tick_pattern_ms"] == PATTERN for h in headers)
    # the phase, which IS per file, still lands each episode exactly
    assert [h["tick_phase"] for h in headers] == [0, 1, 2]
    assert episode_seconds(headers[0], 100) == 0.767   # 33x23 + 8 ms
    assert episode_seconds(headers[2], 100) == 0.766   # 33x23 + 7 ms
    assert c.nominal_msec == 8


@needs_core
def test_ten_ms_header_is_unchanged_byte_for_byte(tmp_path):
    """At a fixed tick the nominal IS config.phys.msec, so the header line
    is byte-identical to the one the pre-fix recorder wrote - before and
    after the change, and before and after any stepping."""
    from surfgym import ScriptedStrafer
    from surfgym.record import record_rollout

    c = _core(10.0, **_REC_KW)
    assert c.nominal_msec == 10 == int(c.config.phys.msec)
    out = tmp_path / "ten.jsonl"
    record_rollout(c, ScriptedStrafer(c, period_ticks=37), out,
                   episodes=1, max_ticks=120, seed=0)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == _legacy_header_line(c, (10,))
    hdr = json.loads(lines[0])
    assert hdr["tick_ms"] == 10 and hdr["phys"]["msec"] == 10
    assert "tick_pattern_ms" not in hdr and "tick_phase" not in hdr
    assert list(hdr) == ["map", "tick_ms", "phys", "episode"]
    # stepping cannot move it (the mirror is only armed under a pattern)
    assert c.nominal_msec == 10 == int(c.config.phys.msec)
    assert lines[0] == _legacy_header_line(c, (10,))
    # and a plain core (no --tick-ms at all) writes the same header
    p = _core(None, **_REC_KW)
    out2 = tmp_path / "plain.jsonl"
    record_rollout(p, ScriptedStrafer(p, period_ticks=37), out2,
                   episodes=1, max_ticks=120, seed=0)
    assert out2.read_bytes() == out.read_bytes()


@needs_core
def test_set_tick_pattern_none_restores_the_config_tick():
    """The docstring's promise, which the phase mirror used to break: a
    7.63 ms core stopped anywhere in [8, 8, 7] goes back to a FIXED 8 ms
    core, not to whichever of 8 / 8 / 7 ran last."""
    from surfgym.core import STATE_DTYPE

    a = np.zeros((1, 6), np.int32)
    c = _core(7.63, num_envs=1, spawn_mode=0, max_episode_ticks=100000,
              lidar_w=0, lidar_h=0)
    c.reset(0)
    for _ in range(3):                     # land on the 7 ms element
        c.step(a)
    assert int(c.config.phys.msec) == 7 and c.tick_phase == 0
    c.set_tick_pattern(None)
    assert c.tick_pattern == (8,) and c.tick_ms == 8.0
    assert c.nominal_msec == 8 and int(c.config.phys.msec) == 8
    assert c.tick_phase == 0

    # and it is a real 8 ms core: identical physics to one built at 8 ms
    st = np.zeros(1, STATE_DTYPE)[0]
    st["origin"] = (1305.057, 11282.656, -12140.515)
    st["velocity"] = (250.0, 0.0, 0.0)
    st["onground"] = -1
    fixed8 = _core(8.0, num_envs=1, spawn_mode=0, max_episode_ticks=100000,
                   lidar_w=0, lidar_h=0)
    outs = []
    for core in (c, fixed8):
        core.reset(0)
        core.set_state(0, st)
        for _ in range(20):
            core.step(np.array([[4, 3, 1, 2, 0, 0]], np.int32))
        outs.append(core.get_states()[0].copy())
    assert np.array_equal(outs[0]["origin"], outs[1]["origin"])
    assert np.array_equal(outs[0]["velocity"], outs[1]["velocity"])

    # a pattern applied to a 10 ms core unwinds to 10 ms, not to the
    # pattern's first element
    ten = _core(None, num_envs=1, spawn_mode=0, max_episode_ticks=100000,
                lidar_w=0, lidar_h=0)
    assert ten.nominal_msec == 10
    ten.set_tick_pattern(PATTERN)
    ten.reset(0)
    ten.step(a)
    assert ten.tick_pattern == (8, 8, 7) and int(ten.config.phys.msec) == 8
    ten.set_tick_pattern(None)
    assert ten.tick_pattern == (10,) and int(ten.config.phys.msec) == 10
    assert ten.nominal_msec == 10 and ten.tick_ms == 10.0
    # setting a FIXED tick is a new config tick, and unwinds to itself
    ten.set_tick_pattern([7])
    assert ten.nominal_msec == 7 and ten.tick_pattern == (7,)
    ten.set_tick_pattern(None)
    assert ten.tick_pattern == (7,) and int(ten.config.phys.msec) == 7
