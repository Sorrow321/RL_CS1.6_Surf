"""--auto-spine (pbr3b2, 2026-10-24): the search -> curriculum loop of pbr2b3 inside one launch.
Unit: the cut (minimum d, clocks zeroed, view along the velocity), the eval reader, the state
machine (quorum -> beam -> accepted spine; a shallow line -> retry backoff). CPU trainer on
princessburglar with a stub beam: the curriculum swaps onto the beam's line and the checkpoint
names it; refusals; flag off is bit-identical to the trainer before the change."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

TRAIN = ROOT / "python" / "train_fast.py"
SRC = ROOT / "autoresearch" / "spines" / "pbr2b3_beam_ge_g16.npy"


class XField:
    """d = x (a field the tests can reason about)."""
    reach_max = 1e9
    cell = 32.0

    def sample(self, p):
        return np.asarray(p, np.float64).reshape(-1, 3)[:, 0].astype(np.float32)


def _line(xs, vy=300.0):
    from surfgym.core import STATE_DTYPE
    s = np.zeros(len(xs), STATE_DTYPE)
    s["origin"][:, 0] = xs
    s["velocity"][:, 1] = vy                    # heading +90 deg
    s["yaw"] = 7.0
    s["tick"] = np.arange(len(xs))
    s["stuck_ticks"] = 3
    return s


def test_cut_line_takes_the_minimum_and_canonicalises():
    from surfgym.autospine import cut_line
    xs = np.r_[np.linspace(1000, 100, 10), np.linspace(150, 900, 5)]
    sp, i, d = cut_line(_line(xs), XField())
    assert i == 9 and d == pytest.approx(100.0) and len(sp) == 10
    assert (sp["tick"] == 0).all() and (sp["stuck_ticks"] == 0).all()
    assert np.allclose(sp["yaw"], 90.0)
    sp2, _, _ = cut_line(_line(xs), XField(), canon=False)
    assert np.allclose(sp2["yaw"], 7.0)
    slow, _, _ = cut_line(_line(xs, vy=50.0), XField())          # under 100 u/s: view kept
    assert np.allclose(slow["yaw"], 7.0)


def _traj(path, eps):
    """A record.py-shaped trajectory: per episode a header, rows [t, x, y, z, ...], a trailer."""
    with open(path, "w", encoding="utf-8") as fh:
        for e, xs in enumerate(eps):
            fh.write(json.dumps({"map": "m", "episode": e}) + "\n")
            for t, x in enumerate(xs):
                fh.write(json.dumps([t, float(x), 0.0, 0.0, 0, 0, 0]) + "\n")
            fh.write(json.dumps({"end": "fail"}) + "\n")


def test_episode_min_d(tmp_path):
    from surfgym.autospine import episode_min_d
    p = tmp_path / "t.jsonl"
    _traj(p, [[500, 400, 450], [300, 200], [900]])
    assert episode_min_d(p, XField()) == [400.0, 200.0, 900.0]


def _stub_cmd(xs_file):
    """A beam_cmd writing beam_frontier.npz from a saved line (no torch, no core)."""
    def argv(ckpt, out_dir, seed, max_ticks):
        code = ("import numpy as np,sys; s=np.load(sys.argv[1]); "
                "np.savez(sys.argv[2], states=s, replay_ok=np.int32(1))")
        return [sys.executable, "-c", code, str(xs_file), str(Path(out_dir) / "beam_frontier.npz")]
    return argv


def _wait(a, end_d, secs=60):
    t0 = time.time()
    while time.time() - t0 < secs:
        r = a.poll(end_d)
        if r is not None or not a.running():
            return r
        time.sleep(0.2)
    raise AssertionError("the stub beam never finished")


def test_state_machine_accepts_a_deeper_line_and_backs_off_a_shallow_one(tmp_path):
    from surfgym.autospine import AutoSpine
    deep = tmp_path / "deep.npy"
    np.save(deep, _line(np.linspace(1000, 100, 50)))
    logs = []
    a = AutoSpine(tmp_path / "run", XField(), _stub_cmd(deep), margin=50.0, quorum=2, retry=3,
                  log=logs.append)
    ev = tmp_path / "ev.jsonl"
    _traj(ev, [[1000, 700], [1000, 650], [1000, 900]])
    # success line 600: nobody there yet -> no beam
    assert "0/3" in a.note_eval(ev, 100, 600.0) and not a.want
    # success line 700: two episodes (quorum 2) -> a beam
    note = a.note_eval(ev, 200, 700.0)
    assert "2/3" in note and a.want and a.q_d == 700.0
    a.start(tmp_path / "ck.pt", 200)
    assert a.running() and not a.want
    r = _wait(a, 700.0)
    assert r is not None and r.k == 1 and r.d_cut == pytest.approx(100.0)
    assert r.path.is_file() and len(np.load(r.path)) == 50
    side = json.loads(r.path.with_suffix(".json").read_text())
    assert "policy's OWN line" in side["provenance"] and side["md5"] == r.md5
    # a line no deeper than the greedy's quorum-th best (+ margin) -> no swap, back off 3 evals
    shallow = tmp_path / "shallow.npy"
    np.save(shallow, _line(np.linspace(1000, 680, 20)))
    a.beam_cmd = _stub_cmd(shallow)
    a.note_eval(ev, 300, 1e9)
    assert a.want
    a.start(tmp_path / "ck.pt", 300)
    assert _wait(a, 1e9) is None and a.retry_at == a.n_eval + 3
    assert "found nothing deeper" in logs[-1]
    assert "next beam no earlier" in a.note_eval(ev, 400, 1e9) and not a.want
    assert [h["result"] for h in a.state_dict()["history"]] == ["accepted", "no_gain"]


def test_state_machine_reports_a_failed_beam(tmp_path):
    from surfgym.autospine import AutoSpine
    a = AutoSpine(tmp_path / "run", XField(),
                  lambda c, o, s, m: [sys.executable, "-c", "import sys; sys.exit(3)"],
                  margin=50.0, quorum=0, retry=2, log=lambda m: None)
    ev = tmp_path / "ev.jsonl"
    _traj(ev, [[1000, 700]])
    a.note_eval(ev, 1, 600.0)
    assert a.want                                  # quorum 0: every eval
    a.start(tmp_path / "ck.pt", 1)
    assert _wait(a, 600.0) is None and a.state_dict()["history"][0]["result"] == "failed"


def test_ge_fallback_splices_a_deeper_chain_onto_the_beam_line(tmp_path):
    """A beam line that is not deep enough -> GE rooted at its last ticks -> the replayed chain
    (rooted at one of those states, going deeper) spliced on -> accepted, source 'ge'."""
    from surfgym.autospine import AutoSpine
    shallow = tmp_path / "shallow.npy"
    line = _line(np.linspace(1000, 600, 41))                 # best d 600 at index 40
    np.save(shallow, line)
    chain = _line(np.r_[line["origin"][30, 0], np.linspace(590, 200, 20)])
    chain[0] = line[30]                                      # rooted at beam state 30 (x 700)
    chain_f = tmp_path / "chain.npy"
    np.save(chain_f, chain)
    seen = {}

    def ge_cmd(roots, out_dir, minutes):
        seen["roots"] = np.load(roots)
        code = ("import os,sys,numpy as np; os.makedirs(sys.argv[1], exist_ok=True); "
                "np.savez(os.path.join(sys.argv[1], 'archive.npz'), x=np.zeros(1)); sys.exit(1)")
        return [sys.executable, "-c", code, str(out_dir)]

    def replay_cmd(archive, field_npz, out_npy):
        seen["field"] = np.load(field_npz)
        code = "import shutil,sys; shutil.copy(sys.argv[1], sys.argv[2])"
        return [sys.executable, "-c", code, str(chain_f), str(out_npy)]

    XField.grid = np.zeros((2, 2, 2), np.float32)
    XField.mins = np.zeros(3)
    logs = []
    a = AutoSpine(tmp_path / "run", XField(), _stub_cmd(shallow), margin=50.0, quorum=0,
                  retry=2, ge_cmd=ge_cmd, replay_cmd=replay_cmd, ge_minutes=1.0,
                  ge_roots_ticks=20, ge_roots_n=5, log=logs.append)
    ev = tmp_path / "ev.jsonl"
    _traj(ev, [[1000, 620]])               # the greedy reaches 620: the beam's 600 + 50 is short
    a.note_eval(ev, 1, 900.0)
    a.start(tmp_path / "ck.pt", 1)
    r = None
    for _ in range(3):                                       # beam, GE, replay
        r = _wait(a, 900.0)
        if r is not None:
            break
    assert r is not None, logs
    # ticks [20, 40] of the line (x 800 .. 600), 5 roots: 20, 25, 30, 35, 40
    assert np.allclose(seen["roots"]["origin"][:, 0], [800, 750, 700, 650, 600])
    assert float(seen["field"]["cell"]) == 32.0
    assert r.d_cut == pytest.approx(200.0) and len(r.states) == 30 + 21   # line[:30] + chain
    side = json.loads(r.path.with_suffix(".json").read_text())
    assert side["source"] == "ge" and "Go-Explore" in side["provenance"]
    h = a.state_dict()["history"][-1]
    assert h["result"] == "accepted" and h["splice_at"] == 30


def test_a_stall_recuts_against_the_greedys_level(tmp_path):
    """No quorum for `stall` evals -> a cycle runs anyway; its line need only pass the greedy's
    quorum-th best (here SHORTER than the success line in force), and the clock restarts."""
    from surfgym.autospine import AutoSpine
    mid = tmp_path / "mid.npy"
    np.save(mid, _line(np.linspace(1000, 400, 61)))         # best d 400
    logs = []
    a = AutoSpine(tmp_path / "run", XField(), _stub_cmd(mid), margin=50.0, quorum=2, retry=1,
                  stall=3, log=logs.append)
    ev = tmp_path / "ev.jsonl"
    _traj(ev, [[1000, 600], [1000, 650], [1000, 900]])      # quorum-th best 650
    end_d = 100.0                                           # a success line nobody reaches
    assert not a.note_eval(ev, 1, end_d).endswith("wanted") and not a.want
    a.note_eval(ev, 2, end_d)
    assert not a.want
    note = a.note_eval(ev, 3, end_d)                        # 3 evals since the start
    assert a.want and a.want_stall and "STALL" in note
    a.start(tmp_path / "ck.pt", 3)
    r = _wait(a, end_d)
    # 400 + 50 = 450 is short of the line in force (100) but past the greedy's 650: accepted
    assert r is not None and r.d_cut == pytest.approx(400.0)
    h = a.state_dict()["history"][-1]
    assert h["stall"] is True and h["bar"] == 650.0 and a.since == a.n_eval
    a.note_eval(ev, 4, 450.0)                               # the new line: still nobody
    assert not a.want


# ---- the trainer (CPU, princessburglar) ----------------------------------------------------
try:
    from test_ge_minutes import _csv, needs_pb  # noqa: E402
    from test_demo_success_d import _run_demo  # noqa: E402
except Exception:                                  # pragma: no cover - the map is not here
    needs_pb = pytest.mark.skip(reason="princessburglar test helpers unavailable")


@needs_pb
def test_the_trainer_swaps_the_curriculum_onto_the_beams_line(tmp_path, monkeypatch):
    run = "asp_on"
    monkeypatch.setenv("AUTOSPINE_STUB_SRC", str(SRC))
    monkeypatch.setenv("AUTOSPINE_STUB_N", "600")
    try:
        r = _run_demo(run, ["--demo-success-d", "64", "--demo-success-bonus", "50",
                            "--auto-spine", "--auto-spine-quorum", "0",
                            "--auto-spine-beam-tool", "tests/python/autospine_stub_beam.py",
                            "--auto-spine-ge-minutes", "0",
                            "--record-every", "2048", "--steps", "10240"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--auto-spine: after a greedy eval" in out
        assert "auto-spine: beam 1 started" in out
        assert "auto-spine: the curriculum now runs on runs/asp_on/autospine_01.npy" in out
        meta = json.loads((ROOT / "runs" / run / "run.json").read_text())
        assert meta["config"]["auto_spine"]["quorum"] == 0
        assert meta["config"]["demo_file"].startswith("runs/asp_on/autospine_")
        assert any(h["result"] == "accepted" for h in meta["auto_spine"]["history"])
        sp = np.load(ROOT / "runs" / run / "autospine_01.npy")
        assert 1 <= len(sp) <= 600 and (sp["tick"] == 0).all()
        import torch
        ck = torch.load(ROOT / "runs" / run / "ckpt_final.pt", map_location="cpu",
                        weights_only=False)
        assert ck["config"]["demo_file"].startswith("runs/asp_on/autospine_")
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


@needs_pb
def test_refused_without_the_success_line(tmp_path):
    bad = _run_demo("asp_bad", ["--auto-spine"], tmp_path)
    try:
        assert bad.returncode != 0 and "no --demo-success-d" in (bad.stdout + bad.stderr)
    finally:
        shutil.rmtree(ROOT / "runs" / "asp_bad", ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--auto-spine" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    import torch
    pre = ROOT / "python" / "_train_fast_pre_asp.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("asp_off_new", TRAIN), ("asp_off_old", pre)):
            r = _run_demo(run, ["--demo-success-d", "64", "--record-every", "2048"], tmp_path,
                          script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "asp_off_new", ROOT / "runs" / "asp_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "auto_spine" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
