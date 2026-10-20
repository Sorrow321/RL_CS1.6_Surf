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
    """A blocklist; an entry is a machine id, or (machine id, detail) for a second incident."""
    rows = [m if isinstance(m, tuple) else (m, str(m)) for m in mids]
    return json.dumps({"_README": ["list"], "blocked": [
        {"ip": None, "machine_id": m, "host_id": None, "reason": "unreliable", "detail": d}
        for m, d in rows]}, indent=2) + "\n"


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
    _write(r, "tools/bad_hosts.json", _bh(1, (1, "the same machine, a second incident"), 2))
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
    # the blocklist is the union: the winner's list exactly (both incidents on machine 1),
    # then the slot's one new entry; its verbatim copy of machine 2 is not added twice
    bh = json.loads(_show(repo, fork, "tools/bad_hosts.json"))
    assert [e["machine_id"] for e in bh["blocked"]] == [1, 1, 2, 3]
    theirs = json.loads(_show(repo, repo["win"], "tools/bad_hosts.json"))
    assert bh["blocked"][:3] == theirs["blocked"]
    # plumbing only: the main checkout's HEAD, index and working tree are untouched
    assert _git(repo["dir"], "rev-parse", "HEAD") == repo["main"]
    assert _git(repo["dir"], "status", "--porcelain") == ""
    assert not list(repo["ar"].glob(".harness.*"))
    # a second pass over the result changes nothing
    assert orchestrator.harness_base(fork, repo=repo["dir"], blocklists=[slot]) == fork


def test_nothing_to_restore_returns_the_base_itself(repo):
    # the operator's own head already carries its harness and every block
    assert orchestrator.harness_base(repo["main"], repo=repo["dir"]) == repo["main"]


def test_ckpt_step_reads_the_step_without_unpickling(tmp_path, monkeypatch):
    """The next incumbent's checkpoint is the best run's newest one, usually later than its best
    eval (r1b1_ratnovx: best eval at 1,633M, ckpt_latest ~2.5B), so the round says both steps.
    The file comes from a rented box: its step is read off the pickle's opcodes."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(orchestrator, "AR_HOME", tmp_path / "ar")

    class Boom:
        def __reduce__(self):            # unpickling this would run code
            return (exec, ("raise SystemExit('unpickled')",))

    for step in (749_731_840, 2_500_170_752):        # BININT and LONG1 encodings
        p = tmp_path / f"ckpt_{step}.pt"
        torch.save({"policy": {"w": torch.zeros(3)}, "optimizer": {"state": {}, "param_groups":
                    [{"lr": 1e-4, "step": 7}]}, "global_step": step,
                    "config": {"envs": 1024}, "reservoir": np.arange(4.0), "x": Boom()}, p)
        assert orchestrator.ckpt_step(p) == step
    bad = tmp_path / "not_a_ckpt.pt"
    bad.write_bytes(b"truncated")
    assert orchestrator.ckpt_step(bad) is None
    # a real trainer checkpoint, when this machine has the incumbent's
    real = ROOT / "runs" / "sidEM2cap2surfGR" / "ckpt_latest.pt"
    if real.exists():
        assert orchestrator.ckpt_step(real) == 749_731_840


def test_handon_skips_a_truncated_or_damaged_checkpoint(tmp_path, monkeypatch):
    """A pull that timed out leaves a truncated file under the real name (r1b2, 31 KB/s,
    2026-10-20); the round end must not copy it to incumbent/ckpt.pt."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(orchestrator, "AR_HOME", tmp_path / "ar")
    run = tmp_path / "r1b2_nth"
    run.mkdir()

    def save(name, step):
        torch.save({"policy": {"w": torch.arange(50_000, dtype=torch.float32)},
                    "global_step": step}, run / name)
        return run / name

    whole = save("ckpt_final.pt", 3000).read_bytes()
    (run / "ckpt_final.pt").write_bytes(whole[:len(whole) // 2])        # truncated pull
    bad = bytearray(save("ckpt_latest.pt", 2500).read_bytes())
    bad[len(bad) // 2] ^= 0xFF                                         # one damaged byte
    (run / "ckpt_latest.pt").write_bytes(bytes(bad))
    save("ckpt_1000.pt", 1000)
    save("ckpt_2000.pt", 2000)
    (run / "ckpt_2000.pt.part").write_bytes(b"partial")
    assert not orchestrator.ckpt_complete(run / "ckpt_final.pt")
    assert not orchestrator.ckpt_complete(run / "ckpt_latest.pt")
    assert orchestrator.handon_ckpt(run) == run / "ckpt_2000.pt"
    (run / "ckpt_2000.pt").unlink()
    assert orchestrator.handon_ckpt(run) == run / "ckpt_1000.pt"
    (run / "ckpt_1000.pt").unlink()
    assert orchestrator.handon_ckpt(run) is None
    save("ckpt_final.pt", 3000)
    assert orchestrator.handon_ckpt(run) == run / "ckpt_final.pt"


def test_the_winners_recipe_travels_with_it(tmp_path, monkeypatch):
    """r1b1 (2026-10-20) made its scratch recipe config.json's incumbent_recipe; restoring the
    operator's config.json would have dropped --respawn-novel --race-ratchet from every round-2
    scratch launch while its NOTES.md said box.py prepends them. The lineage's keys travel with
    the winner when it changed them; everything else in config.json is the operator's."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "core.autocrlf", "false")

    def cfg(cap, recipe):
        return json.dumps({"cap": cap, "incumbent_recipe": recipe}, indent=2) + "\n"

    _write(r, "autoresearch/config.json", cfg(0.30, ["--a"]))
    base = _commit(r, "round base")
    _git(r, "checkout", "-q", "-b", "win")
    _write(r, "autoresearch/config.json", cfg(0.99, ["--a", "--ratchet"]))   # its recipe, its cap
    win = _commit(r, "the winner")
    _git(r, "checkout", "-q", "-b", "same", base)
    _write(r, "python/train.py", "work\n")
    same = _commit(r, "a winner that left the recipe alone")
    _git(r, "checkout", "-q", "main")
    _write(r, "autoresearch/config.json", cfg(0.40, ["--b"]))             # the operator's
    _commit(r, "operator")
    monkeypatch.setattr(orchestrator, "AR_HOME", tmp_path / "ar")
    monkeypatch.setitem(orchestrator.CFG, "harness_files", ["autoresearch/config.json"])
    monkeypatch.setitem(orchestrator.CFG, "lineage_keys", ["incumbent_recipe"])

    fork = orchestrator.harness_base(win, repo=r, since=base)
    assert json.loads(_git(r, "show", f"{fork}:autoresearch/config.json")) == {
        "cap": 0.40, "incumbent_recipe": ["--a", "--ratchet"]}
    assert "incumbent_recipe kept" in _git(r, "log", "-1", "--format=%B", fork)
    # a second pass changes nothing
    assert orchestrator.harness_base(fork, repo=r, since=base) == fork
    # a winner that did not touch the recipe gets the operator's, and the operator's cap
    fork = orchestrator.harness_base(same, repo=r, since=base)
    assert json.loads(_git(r, "show", f"{fork}:autoresearch/config.json")) == {
        "cap": 0.40, "incumbent_recipe": ["--b"]}
    # without lineage_keys the whole file is the operator's, as before
    monkeypatch.setitem(orchestrator.CFG, "lineage_keys", [])
    fork = orchestrator.harness_base(win, repo=r, since=base)
    assert json.loads(_git(r, "show", f"{fork}:autoresearch/config.json"))["incumbent_recipe"] == ["--b"]


def test_a_branch_that_wraps_up_early_is_told_once(tmp_path):
    """r1b3 (2026-10-20) wrapped up at 07:57 with wrapup_at at 11:50 and its slot sat idle."""
    slot = {"agent": "r1b3", "dir": str(tmp_path)}
    wrapup_at = 100_000.0
    early = wrapup_at - 4 * 3600

    def result(final):
        (tmp_path / "RESULT.json").write_text(json.dumps({"agent": "r1b3", "final": final}),
                                              encoding="utf-8")

    assert not orchestrator.tell_early_wrapup(slot, early, wrapup_at)          # no RESULT.json
    result(False)
    assert not orchestrator.tell_early_wrapup(slot, early, wrapup_at)          # still working
    result(True)
    assert orchestrator.tell_early_wrapup(slot, early, wrapup_at)              # wrapped up early
    assert not orchestrator.tell_early_wrapup(slot, early + 60, wrapup_at)     # told once
    result(False)                                                              # it carried on
    assert not orchestrator.tell_early_wrapup(slot, early + 120, wrapup_at)
    result(True)                                                               # and went final again
    assert orchestrator.tell_early_wrapup(slot, early + 180, wrapup_at)
    slot["early_told"] = False
    late = wrapup_at - orchestrator.EARLY_S
    assert not orchestrator.tell_early_wrapup(slot, late, wrapup_at)           # a normal Wrap-up
    (tmp_path / "RESULT.json").write_text("[1, 2]", encoding="utf-8")
    assert not orchestrator.tell_early_wrapup(slot, early, wrapup_at)          # not a dict
    assert "never wrap up" in orchestrator.PROMPT
