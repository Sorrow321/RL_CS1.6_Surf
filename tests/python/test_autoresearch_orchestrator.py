"""autoresearch/orchestrator.py: the round base keeps the operator's harness (2026-10-20).

A round forks the winner's head. Without the overlay, a fix the operator pushes after round 1
starts never reaches a later round (the winner's branch predates it), and a branch that edits
box.py, the caps or the grader carries that edit forward. harness_base() restores
config.harness_files from the main checkout's HEAD over the winner's head and folds the fleet's
blocklist into tools/bad_hosts.json, as one commit, without touching any working tree.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "autoresearch"))

import orchestrator  # noqa: E402


def _git(repo, *args):
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _write(repo, rel, text):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def _commit(repo, msg):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _bh(*mids):
    return json.dumps({"_README": ["list"], "blocked": [
        {"ip": None, "machine_id": m, "host_id": None, "reason": "unreliable", "detail": str(m)}
        for m in mids]}, indent=2) + "\n"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """base -> win (the agent's work, and an edit to a harness file) and base -> main (the
    operator's fix, checked out as HEAD, as ~/RL_Surf is)."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "core.autocrlf", "false")
    _write(r, "autoresearch/box.py", "v1\n")
    _write(r, "autoresearch/config.json", '{"cap": 0.30}\n')
    _write(r, "python/train.py", "t0\n")
    _write(r, "tools/bad_hosts.json", _bh(1))
    base = _commit(r, "base")
    _git(r, "checkout", "-q", "-b", "win")
    _write(r, "python/train.py", "t1 (the agent's work)\n")
    _write(r, "autoresearch/box.py", "an agent's edit\n")
    _write(r, "tools/bad_hosts.json", _bh(1, 2))
    win = _commit(r, "agent")
    _git(r, "checkout", "-q", "main")
    _write(r, "autoresearch/box.py", "v2 (the operator's fix)\n")
    _write(r, "autoresearch/config.json", '{"cap": 0.40}\n')
    main = _commit(r, "operator")
    ar = tmp_path / "ar"
    monkeypatch.setattr(orchestrator, "AR_HOME", ar)
    monkeypatch.setitem(orchestrator.CFG, "harness_files",
                        ["autoresearch/box.py", "autoresearch/config.json",
                         "autoresearch/not_in_main.py"])
    return {"dir": r, "base": base, "win": win, "main": main, "ar": ar}


def _show(repo, rev, path):
    return _git(repo["dir"], "show", f"{rev}:{path}")


def test_the_operator_harness_over_the_winner(repo, tmp_path):
    slot = tmp_path / "slot_bad_hosts.json"
    slot.write_text(_bh(2, 3), encoding="utf-8")
    fork = orchestrator.harness_base(repo["win"], repo=repo["dir"], blocklists=[slot])
    assert fork != repo["win"]
    assert _git(repo["dir"], "rev-parse", f"{fork}^") == repo["win"]
    # the harness is the operator's, the agent's own work survives
    assert _show(repo, fork, "autoresearch/box.py") == "v2 (the operator's fix)"
    assert _show(repo, fork, "autoresearch/config.json") == '{"cap": 0.40}'
    assert _show(repo, fork, "python/train.py") == "t1 (the agent's work)"
    # the blocklist is the union, the winner's entries first, each machine once
    bh = json.loads(_show(repo, fork, "tools/bad_hosts.json"))
    assert [e["machine_id"] for e in bh["blocked"]] == [1, 2, 3]
    # plumbing only: the main checkout's HEAD, index and working tree are untouched
    assert _git(repo["dir"], "rev-parse", "HEAD") == repo["main"]
    assert _git(repo["dir"], "status", "--porcelain") == ""
    assert not list(repo["ar"].glob(".harness.*"))
    # a second pass over the result changes nothing
    assert orchestrator.harness_base(fork, repo=repo["dir"], blocklists=[slot]) == fork


def test_nothing_to_restore_returns_the_base_itself(repo):
    # the operator's own head already carries its harness and every block
    assert orchestrator.harness_base(repo["main"], repo=repo["dir"]) == repo["main"]
