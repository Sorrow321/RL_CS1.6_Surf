"""Harvesting must not depend on an agent being awake.

2026-09-04: six rented 5090s ran a 4.5 h wave with on-box deadlines at 15:11.
The orchestrating agent had a wake-up at 14:25 to harvest them; an API rate
limit killed every agent from ~14:20 to 15:10, and at 15:11 the six boxes
self-destructed on schedule with every checkpoint and trajectory still on
them. A seventh box lost its results the other way: its expert driver
finished, its pid went away, and the on-box watchdog destroyed the box ten
minutes later.

The self-destruct rules were right. The harvest was a MANUAL step, which is
the same class of failure CLAUDE.md already names for the destroy button:
"the agent's own attention is not a safety mechanism". So the daemon harvests
now, and these are the properties that have to hold while it does:

  * the pull fires ONCE, at deadline - HARVEST_LEAD_S, and is idempotent;
  * a failed pull retries with backoff but NEVER postpones the kill - the
    money rule wins, and the loss is logged loudly;
  * `release` harvests before it destroys, unless told not to;
  * the early-exit path needs the trainer pid reported gone TWICE, and a
    failed ssh is not an observation at all;
  * an entry with no harvest spec behaves exactly as it did before.

Every vastai call, ssh and harvest subprocess is mocked: this test rents
nothing and destroys nothing.
"""
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import fleet_watchdog as fw

NOW = 1.9e9


@pytest.fixture
def sim(monkeypatch):
    """sweep() over a scripted instance list, an in-memory registry, and a
    harvest subprocess that records instead of running."""
    state = {"live": [], "reg": {}, "destroyed": [], "logs": [],
             "harvests": [], "rc": 0, "pid": "PID_ALIVE", "ssh_rc": 0,
             "ssh_calls": []}
    monkeypatch.setattr(fw, "instances", lambda: state["live"])
    monkeypatch.setattr(fw, "load_reg", lambda: state["reg"])
    monkeypatch.setattr(fw, "save_reg", lambda d: state.__setitem__("reg", d))
    monkeypatch.setattr(fw, "log", lambda m: state["logs"].append(str(m)))
    monkeypatch.setattr(fw, "now", lambda: state.get("t", NOW))

    def fake_destroy(iid, why):
        state["destroyed"].append((str(iid), why))
        return 1

    def fake_exec(cmd, env, timeout):
        """Both children go through _exec: the ssh pid probe and the pull."""
        if cmd[0] == "ssh":
            state["ssh_calls"].append(cmd)
            return types.SimpleNamespace(returncode=state["ssh_rc"],
                                         stdout=state["pid"], stderr="")
        state["harvests"].append({"cmd": cmd, "env": env, "timeout": timeout})
        rc = state["rc"]
        return types.SimpleNamespace(returncode=rc, stdout="done." if rc == 0
                                     else "", stderr="" if rc == 0 else "boom")

    # the pull runs in a thread in production; inline here so the assertions
    # are deterministic (and so the registry merge is exercised in the worst
    # order: the worker writes BEFORE the sweep merges its own bookkeeping)
    monkeypatch.setattr(fw, "destroy", fake_destroy)
    monkeypatch.setattr(fw, "_exec", fake_exec)
    monkeypatch.setattr(fw, "_dispatch", lambda fn, *a: fn(*a))
    fw._HARVESTING.clear()
    return state


def _inst(iid, status="running", age_s=600.0, host="ssh5.vast.ai", port=41000):
    return {"id": iid, "actual_status": status, "start_date": NOW - age_s,
            "ssh_host": host, "ssh_port": port}


def _entry(deadline, **kw):
    ent = {"deadline": deadline, "label": "xW1", "ready": True,
           "harvest": {"host": "ssh5.vast.ai", "port": 41000,
                       "runs": ["xW1"]}}
    ent.update(kw)
    return ent


# ---------------------------------------------------------------- the lead --

def test_harvest_fires_at_the_lead_and_only_once(sim):
    """Outside the window: nothing. Inside it: exactly one pull, ever."""
    sim["t"] = NOW
    sim["reg"] = {"7": _entry(NOW + fw.HARVEST_LEAD_S + 600.0)}
    sim["live"] = [_inst(7)]
    assert fw.sweep() == 0
    assert sim["harvests"] == [], "harvested a box 30 min from its deadline"

    sim["t"] = NOW + 601.0                      # now inside the 20 min lead
    assert fw.sweep() == 0
    assert len(sim["harvests"]) == 1
    cmd = sim["harvests"][0]["cmd"]
    assert cmd[0] == "bash" and cmd[1].endswith("harvest_box.sh")
    assert cmd[2:] == ["41000", "ssh5.vast.ai", "xW1"]
    assert sim["harvests"][0]["timeout"] == fw.HARVEST_TIMEOUT_S
    assert sim["reg"]["7"]["harvested_at"]

    for _ in range(3):                          # ...and it does not repeat
        sim["t"] += 60.0
        fw.sweep()
    assert len(sim["harvests"]) == 1, "harvested the same box twice"
    assert sim["destroyed"] == []               # the deadline has not passed


def test_the_harvest_writes_into_this_checkout(sim):
    sim["t"] = NOW
    sim["reg"] = {"7": _entry(NOW + 60.0)}
    sim["live"] = [_inst(7)]
    fw.sweep()
    env = sim["harvests"][0]["env"]
    assert env["LOCAL_REPO"] == fw._posix_repo()
    assert env["LOCAL_REPO"].startswith("/")     # git bash, not C:\...


def test_a_failed_harvest_retries_with_backoff_then_gives_up(sim):
    """2, then 4 minutes - all three attempts fit inside the 20 min lead, on
    purpose: a retry schedule that runs past the deadline is not a retry."""
    sim["t"] = NOW
    sim["rc"] = 1
    sim["reg"] = {"7": _entry(NOW + fw.HARVEST_LEAD_S)}   # the window is open
    sim["live"] = [_inst(7)]
    fw.sweep()
    assert len(sim["harvests"]) == 1
    assert sim["reg"]["7"]["harvest_attempts"] == 1
    assert "rc=1" in sim["reg"]["7"]["harvest_error"]
    assert not sim["reg"]["7"].get("harvested_at")

    sim["t"] += 30.0                            # inside the backoff: no retry
    fw.sweep()
    assert len(sim["harvests"]) == 1

    sim["t"] = NOW + fw.HARVEST_BACKOFF_S + 1.0             # 2 min
    fw.sweep()
    assert len(sim["harvests"]) == 2
    sim["t"] += 2 * fw.HARVEST_BACKOFF_S + 1.0              # +4 min
    fw.sweep()
    assert len(sim["harvests"]) == 3
    sim["t"] += 4 * fw.HARVEST_BACKOFF_S + 1.0              # HARVEST_TRIES
    fw.sweep()
    assert len(sim["harvests"]) == 3
    assert sim["t"] < sim["reg"]["7"]["deadline"], "the retries outlived the box"
    assert sim["destroyed"] == []


def test_a_failed_harvest_still_dies_at_the_deadline_and_says_so(sim):
    """The money rule wins. An unharvested box is still destroyed on time -
    and the log has to make the loss impossible to miss."""
    sim["t"] = NOW
    sim["rc"] = 1
    sim["reg"] = {"7": _entry(NOW + 60.0)}
    sim["live"] = [_inst(7)]
    fw.sweep()
    assert sim["destroyed"] == []

    sim["t"] = NOW + 61.0
    assert fw.sweep() == 1
    assert sim["destroyed"][0][0] == "7"
    assert "deadline passed" in sim["destroyed"][0][1]
    assert any("UNHARVESTED" in m and "rc=1" in m for m in sim["logs"])


def test_a_slow_harvest_does_not_stall_another_boxs_deadline(sim, monkeypatch):
    """The sweep must keep enforcing deadlines while a pull is in flight."""
    sim["t"] = NOW
    sim["reg"] = {"7": _entry(NOW + 60.0), "8": _entry(NOW - 1.0, label="xOLD")}
    sim["live"] = [_inst(7), _inst(8, port=41001)]
    started = []
    monkeypatch.setattr(fw, "_dispatch",           # a pull that never returns
                        lambda fn, *a: started.append(a[0]))
    assert fw.sweep() == 1
    assert started == ["7"]
    assert sim["destroyed"][0][0] == "8"

    sim["t"] += fw.POLL_S                     # ...and the next sweep, and the
    sim["t"] += fw.POLL_S                     # one after, do not start a second
    fw.sweep()
    assert started == ["7"], "a second pull started while one was in flight"


def test_a_box_that_never_came_up_is_never_pid_probed(sim):
    """No `ready` latch = it never ran a trainer; there is nothing to poll for
    and an ssh into a box that is still loading answers nothing anyway."""
    sim["t"] = NOW
    sim["reg"] = {"7": _entry(NOW + 6 * 3600.0, ready=False,
                              harvest={"host": "ssh5.vast.ai", "port": 41000,
                                       "runs": ["xW1"],
                                       "pid_file": "runs/xW1.pid"})}
    sim["live"] = [_inst(7, status="loading", age_s=30.0)]
    fw.sweep()
    assert sim["ssh_calls"] == []
    assert sim["harvests"] == []


# --------------------------------------------------------- the early exit --

def test_the_early_exit_needs_two_dead_polls_not_one(sim):
    """A trainer that finished leaves the on-box watchdog ~10 min. Harvest
    then - but ONE observation is not evidence (CLAUDE.md, the rule that
    exists because a single `offline` poll destroyed a live training box)."""
    sim["t"] = NOW
    sim["pid"] = "PID_DEAD"
    sim["reg"] = {"7": _entry(NOW + 6 * 3600.0,
                              harvest={"host": "ssh5.vast.ai", "port": 41000,
                                       "runs": ["xW1"],
                                       "pid_file": "runs/xW1.pid"})}
    sim["live"] = [_inst(7)]
    fw.sweep()
    assert sim["harvests"] == [], "harvested on ONE dead-pid observation"
    assert sim["reg"]["7"]["pid_dead_polls"] == 1

    sim["t"] += fw.PID_POLL_S                   # the poll is rate limited...
    fw.sweep()
    assert len(sim["harvests"]) == 1, "two dead polls did not trigger a pull"
    assert "pid gone" in sim["reg"]["7"]["harvest_reason"]
    assert sim["reg"]["7"]["harvested_at"]
    assert sim["destroyed"] == []               # harvesting never destroys


def test_a_live_pid_is_never_harvested_and_the_probe_is_rate_limited(sim):
    sim["t"] = NOW
    sim["pid"] = "PID_ALIVE"
    sim["reg"] = {"7": _entry(NOW + 6 * 3600.0,
                              harvest={"host": "ssh5.vast.ai", "port": 41000,
                                       "runs": ["xW1"],
                                       "pid_file": "runs/xW1.pid"})}
    sim["live"] = [_inst(7)]
    for _ in range(4):                          # four sweeps, one ssh
        fw.sweep()
        sim["t"] += fw.POLL_S
    assert len(sim["ssh_calls"]) == 1
    assert sim["harvests"] == []


def test_a_failed_ssh_is_not_evidence_of_a_dead_trainer(sim):
    """A box behind a network blip must not be read as a finished run - the
    same one-observation rule that the readiness latch exists for."""
    sim["t"] = NOW
    sim["ssh_rc"] = 255
    sim["pid"] = ""
    sim["reg"] = {"7": _entry(NOW + 6 * 3600.0,
                              harvest={"host": "ssh5.vast.ai", "port": 41000,
                                       "runs": ["xW1"],
                                       "pid_file": "runs/xW1.pid"})}
    sim["live"] = [_inst(7)]
    for _ in range(3):
        fw.sweep()
        sim["t"] += fw.PID_POLL_S
    assert sim["harvests"] == []
    assert not sim["reg"]["7"].get("pid_dead_polls")
    assert any("inconclusive" in m for m in sim["logs"])


def test_a_missing_pid_file_is_unknown_not_dead(sim):
    sim["t"] = NOW
    sim["pid"] = "PID_NOFILE"
    sim["reg"] = {"7": _entry(NOW + 6 * 3600.0,
                              harvest={"host": "ssh5.vast.ai", "port": 41000,
                                       "runs": ["xW1"],
                                       "pid_file": "runs/xW1.pid"})}
    sim["live"] = [_inst(7)]
    for _ in range(3):
        fw.sweep()
        sim["t"] += fw.PID_POLL_S
    assert sim["harvests"] == []


def test_an_early_pull_does_not_cancel_the_deadline_pull(sim):
    """A pid-triggered pull can be hours early; the box keeps writing until
    the deadline, so the lead window still gets its own attempt."""
    sim["t"] = NOW
    sim["pid"] = "PID_DEAD"
    sim["reg"] = {"7": _entry(NOW + 3600.0,
                              harvest={"host": "ssh5.vast.ai", "port": 41000,
                                       "runs": ["xW1"],
                                       "pid_file": "runs/xW1.pid"})}
    sim["live"] = [_inst(7)]
    fw.sweep()
    sim["t"] += fw.PID_POLL_S
    fw.sweep()
    assert len(sim["harvests"]) == 1
    sim["t"] = NOW + 3600.0 - fw.HARVEST_LEAD_S + 1.0
    fw.sweep()
    assert len(sim["harvests"]) == 2
    sim["t"] += 60.0
    fw.sweep()
    assert len(sim["harvests"]) == 2, "the lead pull repeated"


# -------------------------------------------------------------- the manual --

def test_release_harvests_before_it_destroys(sim):
    sim["t"] = NOW
    sim["reg"] = {"7": _entry(NOW + 3600.0)}
    sim["live"] = [_inst(7)]
    fw.cmd_release(types.SimpleNamespace(id="7", no_harvest=False))
    assert len(sim["harvests"]) == 1
    assert sim["destroyed"] == [("7", "released by owner")]


def test_release_destroys_even_when_the_harvest_fails(sim):
    sim["t"] = NOW
    sim["rc"] = 1
    sim["reg"] = {"7": _entry(NOW + 3600.0)}
    sim["live"] = [_inst(7)]
    fw.cmd_release(types.SimpleNamespace(id="7", no_harvest=False))
    assert sim["destroyed"] == [("7", "released by owner")]
    assert any("harvest FAILED" in m and "Results are LOST" in m
               for m in sim["logs"])


def test_release_no_harvest_skips_the_pull(sim):
    sim["t"] = NOW
    sim["reg"] = {"7": _entry(NOW + 3600.0)}
    sim["live"] = [_inst(7)]
    fw.cmd_release(types.SimpleNamespace(id="7", no_harvest=True))
    assert sim["harvests"] == []
    assert sim["destroyed"] == [("7", "released by owner")]


def test_release_of_a_box_with_no_spec_is_unchanged(sim):
    """The whole point of the compatibility rule: an old-style entry (and
    every entry already in runs/fleet.json) still just gets destroyed."""
    sim["t"] = NOW
    sim["reg"] = {"7": {"deadline": NOW + 3600.0, "label": "xOLD",
                        "ready": True}}
    sim["live"] = [_inst(7)]
    fw.cmd_release(types.SimpleNamespace(id="7", no_harvest=False))
    assert sim["harvests"] == []
    assert sim["destroyed"] == [("7", "released by owner")]


# ------------------------------------------------------- extras and specs --

def test_the_expert_box_spec_reaches_harvest_box_sh(sim):
    """An expert loop has no runs/<run>/progress.csv; it has a summary and one
    checkpoint per round, of which only the newest is wanted."""
    sim["t"] = NOW
    sim["reg"] = {"7": _entry(NOW + 60.0, harvest={
        "host": "ssh5.vast.ai", "port": 41000, "runs": ["exit_scratch"],
        "only_extra": True,
        "extra": ["runs/exit_scratch/expert_summary.jsonl"],
        "newest": ["runs/exit_scratch/round_*/train/ckpt_final.pt"]})}
    sim["live"] = [_inst(7)]
    fw.sweep()
    env = sim["harvests"][0]["env"]
    assert env["HARVEST_ONLY_EXTRA"] == "1"
    assert env["HARVEST_EXTRA"] == "runs/exit_scratch/expert_summary.jsonl"
    assert env["HARVEST_NEWEST"] == \
        "runs/exit_scratch/round_*/train/ckpt_final.pt"


def test_harvest_runs_learns_the_endpoint_from_the_listing(sim):
    """--harvest-runs stores no endpoint; the vast listing supplies it."""
    sim["t"] = NOW
    sim["reg"] = {"7": _entry(NOW + 60.0, harvest={"runs": ["xW1"]})}
    sim["live"] = [_inst(7, host="ssh9.vast.ai", port=12345)]
    fw.sweep()
    assert sim["harvests"][0]["cmd"][2:] == ["12345", "ssh9.vast.ai", "xW1"]


def test_no_endpoint_at_all_is_a_loud_failure_not_a_crash(sim):
    sim["t"] = NOW
    sim["reg"] = {"7": _entry(NOW + 60.0, harvest={"runs": ["xW1"]})}
    sim["live"] = [{"id": 7, "actual_status": "running",
                    "start_date": NOW - 600.0}]
    fw.sweep()
    assert sim["harvests"] == []
    assert any("no ssh endpoint" in m for m in sim["logs"])


def test_a_harvest_that_raises_never_kills_the_sweep(sim, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("scp died")

    monkeypatch.setattr(fw, "_exec", boom)
    sim["t"] = NOW
    sim["reg"] = {"7": _entry(NOW + 60.0), "8": _entry(NOW - 1.0)}
    sim["live"] = [_inst(7), _inst(8)]
    assert fw.sweep() == 1                    # box 8 still destroyed on time
    assert sim["destroyed"][0][0] == "8"


# ------------------------------------------------------------ registration --

def _reg_args(**kw):
    base = dict(id="7", label="xW1", owner="me", minutes=60.0, harvest=None,
                harvest_runs=None, harvest_extra=None, harvest_newest=None,
                harvest_only_extra=False, pid_file=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_register_parses_the_harvest_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(fw, "REG", tmp_path / "fleet.json")
    fw.cmd_register(_reg_args(harvest="41000 ssh5.vast.ai xW1 xW2",
                              pid_file="runs/xW1.pid"))
    spec = fw.load_reg()["7"]["harvest"]
    assert spec == {"port": 41000, "host": "ssh5.vast.ai",
                    "runs": ["xW1", "xW2"], "pid_file": "runs/xW1.pid"}


def test_re_registering_keeps_the_spec_the_latch_and_the_receipt(tmp_path,
                                                                 monkeypatch):
    """wave_launch re-registers to pull the deadline in under the on-box one;
    an agent re-registers to relabel. Neither may lose the harvest state - the
    same reason `ready` is latched."""
    monkeypatch.setattr(fw, "REG", tmp_path / "fleet.json")
    fw.cmd_register(_reg_args(harvest="41000 ssh5.vast.ai xW1"))
    reg = fw.load_reg()
    reg["7"]["ready"] = True
    reg["7"]["harvested_at"] = "2026-09-04T14:51:00Z"
    fw.save_reg(reg)

    fw.cmd_register(_reg_args(label="renamed"))          # no harvest flags
    ent = fw.load_reg()["7"]
    assert ent["label"] == "renamed"
    assert ent["ready"] is True
    assert ent["harvest"]["runs"] == ["xW1"]
    assert ent["harvested_at"] == "2026-09-04T14:51:00Z"

    # a NEW spec is new intent: the receipt goes, the files differ
    fw.cmd_register(_reg_args(harvest="41000 ssh5.vast.ai xW9"))
    ent = fw.load_reg()["7"]
    assert ent["harvest"]["runs"] == ["xW9"]
    assert "harvested_at" not in ent


def test_register_without_a_spec_is_byte_for_byte_the_old_entry(tmp_path,
                                                                monkeypatch):
    """Backward compatibility with the runs/fleet.json that is live right
    now: no harvest flags -> no new keys at all."""
    monkeypatch.setattr(fw, "REG", tmp_path / "fleet.json")
    fw.cmd_register(_reg_args())
    ent = fw.load_reg()["7"]
    assert set(ent) == {"label", "owner", "registered", "ready", "deadline",
                        "deadline_utc"}


def test_a_bad_harvest_string_is_refused_at_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(fw, "REG", tmp_path / "fleet.json")
    with pytest.raises(SystemExit):
        fw.cmd_register(_reg_args(harvest="ssh5.vast.ai xW1"))   # no port
    with pytest.raises(SystemExit):
        fw.cmd_register(_reg_args(harvest="41000 ssh5.vast.ai xW1",
                                  harvest_extra="runs/o'brien.jsonl"))


def test_list_renders_old_and_new_entries_side_by_side(sim, capsys):
    """`list` is the shared view across every agent and session; a registry
    holding both shapes at once (which is what a rollout looks like) must not
    break it."""
    sim["t"] = NOW
    sim["reg"] = {"7": _entry(NOW + 3600.0),
                  "8": {"deadline": NOW + 60.0, "label": "xOLD"},
                  "9": _entry(NOW + 60.0, harvested_at="2026-09-04T14:51:00Z")}
    sim["live"] = [_inst(7), _inst(8), _inst(9)]
    fw.cmd_list(types.SimpleNamespace())
    out = capsys.readouterr().out
    assert "harvest xW1 in +40 min" in out
    assert "harvest: NONE" in out
    assert "harvested 2026-09-04T14:51:00Z" in out


def test_an_entry_with_no_spec_sweeps_exactly_as_before(sim):
    """The compatibility guarantee, at the sweep: no spec, no ssh, no pull,
    and the ready latch and the deadline behave as they always did."""
    sim["t"] = NOW
    sim["reg"] = {"7": {"deadline": NOW + 60.0, "label": "xOLD"}}
    sim["live"] = [_inst(7)]
    assert fw.sweep() == 0
    assert sim["harvests"] == [] and sim["ssh_calls"] == []
    assert sim["reg"]["7"]["ready"] is True
    sim["t"] = NOW + 61.0
    assert fw.sweep() == 1
    assert "deadline passed" in sim["destroyed"][0][1]
