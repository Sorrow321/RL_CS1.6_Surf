"""autoresearch/box.py: the rental gate (user, 2026-10-20). The vast CLI is mocked; nothing rents.

* offers: whitelisted single-GPU RTX cards under their cap, blocklisted hosts and the CPUs in
  config.cpu_exclude out, in the user's order (3090-class tier first, cheapest first, then
  4090, then 5090);
* the blocklist is shared: every branch's list folds into $AR_HOME/bad_hosts.json, which
  every agent's box.py reads on top of its own;
* rent refuses a second box for the same agent, a box over the fleet cap, and a box the
  budget pace cannot afford;
* score.py rejects an eval that does not start at the map start.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "autoresearch"))

import box  # noqa: E402


def _offer(i, gpu, dph, n=1, mid=None):
    return {"id": i, "gpu_name": gpu.replace("_", " "), "num_gpus": n, "dph_total": dph,
            "machine_id": mid or i, "host_id": 1000 + i, "public_ipaddr": f"1.2.3.{i}",
            "cpu_cores_effective": 32, "cpu_ram": 64000, "inet_down": 900,
            "geolocation": "X"}


MARKET = {
    "RTX_3090": [_offer(1, "RTX_3090", 0.31), _offer(2, "RTX_3090", 0.21),
                 _offer(3, "RTX_3090", 0.18, mid=777)],
    "RTX_5080": [_offer(4, "RTX_5080", 0.20)],
    "RTX_4090": [_offer(5, "RTX_4090", 0.44), _offer(6, "RTX_4090", 0.50)],
    "RTX_5090": [_offer(7, "RTX_5090", 0.55), _offer(8, "RTX_5090", 0.55, n=2)],
}


@pytest.fixture
def market(monkeypatch, tmp_path):
    state = {"live": [], "credit": 100.0, "created": []}

    def fake_vast(*args, raw=True, timeout=120, check=True):
        if args[:2] == ("search", "offers"):
            gpu = args[2].split()[0].split("=")[1]
            return list(MARKET.get(gpu, []))
        if args[:2] == ("show", "instances"):
            return list(state["live"])
        if args[:2] == ("show", "user"):
            return {"credit": state["credit"]}
        if args[:2] == ("create", "instance"):
            state["created"].append(args)
            return {"new_contract": 999}
        raise AssertionError(f"unexpected vast call {args}")

    monkeypatch.setattr(box, "vast", fake_vast)
    monkeypatch.setattr(box, "blocked_keys", lambda: ({777}, set(), set()))
    monkeypatch.setattr(box, "AR_HOME", tmp_path)
    monkeypatch.setattr(box, "BOXES", tmp_path / "boxes")
    monkeypatch.setattr(box, "_ar_start", lambda: box.time.time())
    return state


def test_offers_are_whitelisted_capped_single_gpu_and_ordered(market):
    got = box.offers()
    ids = [o["id"] for o in got]
    # the 3090 at 0.31 is under its 0.65 cap (raised 2026-10-22, 3a9af7e); machine 777 is
    # blocklisted; the 4090 at 0.50 is over 0.45; the 2-GPU 5090 is out. Tier 0 (3090 / 5080)
    # first, cheapest first
    assert ids == [4, 2, 1, 5, 7], ids
    assert {o["gpu"] for o in got} <= set(box.CFG["max_dph"])


def test_one_box_per_agent(market):
    box.save_rec("r1b1", {"agent": "r1b1", "id": 42, "gpu": "RTX_3090", "dph": 0.2,
                          "created": 0})
    market["live"] = [{"id": 42, "label": "r1b1", "dph_total": 0.2}]
    with pytest.raises(SystemExit) as e:
        box.main(["rent", "--agent", "r1b1", "--dry-run"])
    assert e.value.code == 2
    assert market["created"] == []


def test_the_fleet_cap(market):
    market["live"] = [{"id": 100 + i, "label": f"x{i}", "dph_total": 0.2}
                      for i in range(box.CFG["max_boxes"])]
    with pytest.raises(SystemExit):
        box.main(["rent", "--agent", "r1b2", "--dry-run"])


def test_the_budget_pace(market, capsys):
    # $100, $10 reserve, 72 h: 1.25 $/h. Four 0.30 boxes already burn 1.2 - nothing fits
    market["live"] = [{"id": 100 + i, "label": f"x{i}", "dph_total": 0.30} for i in range(3)]
    market["credit"] = 30.0          # (30 - 10) / 72 = 0.28 $/h allowed, 0.9 already burning
    with pytest.raises(SystemExit) as e:
        box.main(["rent", "--agent", "r1b3", "--dry-run"])
    assert e.value.code == 3
    # with money the dry run picks the best offer and rents nothing
    market["live"] = []
    market["credit"] = 100.0
    capsys.readouterr()
    box.main(["rent", "--agent", "r1b3", "--dry-run"])
    out = json.loads(capsys.readouterr().out)
    assert out["would_rent"]["id"] == 4 and market["created"] == []


def test_the_reserve(market):
    market["credit"] = 9.0
    with pytest.raises(SystemExit):
        box.main(["rent", "--agent", "r1b4", "--dry-run"])


def test_offers_drop_the_excluded_cpus(monkeypatch):
    # vast lists a Haswell E5-2696 v3 at its 3.8 GHz turbo, so cpu_ghz cannot catch it
    xeon = dict(_offer(1, "RTX_3090", 0.18), cpu_name="Xeon® E5-2699 v3 ")
    ryzen = dict(_offer(2, "RTX_3090", 0.25), cpu_name="AMD Ryzen 9 5950X 16-Core Processor")
    monkeypatch.setattr(box, "vast", lambda *a, **k: [xeon, ryzen]
                        if a[:3] == ("search", "offers", f"gpu_name=RTX_3090 {box.CFG['offer_query']}")
                        else [])
    monkeypatch.setattr(box, "blocked_keys", lambda: (set(), set(), set()))
    got = box.offers()
    assert [o["id"] for o in got] == [2], got
    assert got[0]["cpu"].startswith("AMD Ryzen 9 5950X")


def test_launch_declares_self_states_with_its_provenance(monkeypatch, tmp_path):
    """r1b4 had to bypass `launch` (box.py ssh + run_arm.sh) to pass SELF_STATES=1 for a
    policy-rooted Go-Explore spawn file, and the run never reached its box record."""
    monkeypatch.setattr(box, "AR_HOME", tmp_path)
    monkeypatch.setattr(box, "BOXES", tmp_path / "boxes")
    box.save_rec("r1b4", {"agent": "r1b4", "id": 42, "gpu": "RTX_3090", "dph": 0.2,
                          "created": 0, "host": "h", "port": 1})
    sent = []
    monkeypatch.setattr(box, "ssh", lambda rec, cmd, **k: sent.append(cmd) or
                        box.subprocess.CompletedProcess(cmd, 0, "", ""))
    why = "Go-Explore archive rooted at the incumbent's own reservoir, no demo"
    box.main(["launch", "--agent", "r1b4", "--run", "r1b4_gs", "--resume", "--self-states", why,
              "--", "--spawn-states", "runs/ge.npy"])
    assert "SELF_STATES=1 " in sent[-1] and "--spawn-states runs/ge.npy" in sent[-1]
    assert box.load_rec("r1b4")["runs"][-1]["self_states"] == why
    # without the declaration run_arm.sh is left to refuse the file, as it always did
    box.main(["launch", "--agent", "r1b4", "--run", "r1b4_x", "--resume"])
    assert "SELF_STATES" not in sent[-1]
    assert "self_states" not in box.load_rec("r1b4")["runs"][-1]
    # and a declaration has to say something
    with pytest.raises(SystemExit):
        box.main(["launch", "--agent", "r1b4", "--run", "r1b4_y", "--self-states", "mine"])


def _blocklist(*mids, readme="branch"):
    return {"_README": [readme], "blocked": [
        {"ip": None, "machine_id": m, "host_id": None, "reason": "gpu_capped", "detail": f"m{m}"}
        for m in mids]}


def test_the_fleet_blocklist_is_shared(monkeypatch, tmp_path):
    """Two agents rented the same power-capped 3090 Ti (machine 68144) on 2026-10-20 because
    each read only its own branch's list."""
    monkeypatch.setattr(box, "AR_HOME", tmp_path)
    b2, b3 = tmp_path / "b2.json", tmp_path / "b3.json"
    l2, l3 = _blocklist(7, 68144, 11, readme="b2"), _blocklist(7, 68144, 12, readme="b3")
    l3["blocked"][1]["detail"] = "r1b3 rented it too"   # a second incident on the same machine
    b2.write_text(json.dumps(l2), encoding="utf-8")
    b3.write_text(json.dumps(l3), encoding="utf-8")
    assert box.sync_shared_blocklist([b2, b3]) == 5
    shared = json.loads((tmp_path / "bad_hosts.json").read_text(encoding="utf-8"))
    # the first list exactly, then what the second adds: its verbatim copy of machine 7 is
    # dropped, its own report on 68144 is kept (the history is the point of the file)
    assert [e["machine_id"] for e in shared["blocked"]] == [7, 68144, 11, 68144, 12]
    assert shared["_README"] == ["b2"]            # the first source's header is kept
    # idempotent: a second pass rewrites nothing
    before = (tmp_path / "bad_hosts.json").stat().st_mtime_ns
    assert box.sync_shared_blocklist([b2, b3]) == 5
    assert (tmp_path / "bad_hosts.json").stat().st_mtime_ns == before
    # and every agent's box.py skips it on top of its own branch's list
    mids, _, _ = box.blocked_keys()
    own = json.loads((box.REPO / "tools" / "bad_hosts.json").read_text(encoding="utf-8"))
    own_mids = {e["machine_id"] for e in own["blocked"] if e.get("machine_id")}
    assert mids == own_mids | {7, 68144, 11, 12}
    # an unreadable shared list never blocks renting: the branch's own list still applies
    (tmp_path / "bad_hosts.json").write_text("{not json", encoding="utf-8")
    mids, _, _ = box.blocked_keys()
    assert mids == own_mids


FIELD = ROOT / "maps" / "surf_src_sidistic.goal_32.npz"
GR = ROOT / "runs" / "sidEM2cap2surfGR" / "traj_0755499008.jsonl"


@pytest.mark.skipif(not (FIELD.exists() and GR.exists()), reason="needs the sidistic field + an eval")
def test_score_rejects_an_eval_that_does_not_start_at_the_map_start(tmp_path):
    import score
    good = score.score_file(GR)
    # 27.222 on the old goal_32 field; 26.853 on the no-corner-cut field (round 7 on)
    assert good["score"] == pytest.approx(26.853, abs=0.01) and good["max_cp"] == 1
    lines = GR.read_text(encoding="utf-8").splitlines()
    # move every episode's first row 9,000 u along x: not the map start any more
    bad = []
    for ln in lines:
        if ln.startswith("[") and (not bad or bad[-1].startswith("{")):
            row = json.loads(ln)
            row[1] += 9000.0
            ln = json.dumps(row)
        bad.append(ln)
    p = tmp_path / "traj_1.jsonl"
    p.write_text("\n".join(bad) + "\n", encoding="utf-8")
    s = score.score_file(p)
    assert s["score"] is None and "not the map start" in s["error"]


def test_a_truncated_ckpt_pull_keeps_the_copy_already_pulled(monkeypatch, tmp_path):
    """r1b2's box ran at 31 KB/s on 2026-10-20: the scp of an 80 MB checkpoint timed out at its
    30 min with 56 MB and left the truncated file under the real name, over the good copy."""
    import hashlib
    import io
    import subprocess
    import tarfile
    monkeypatch.setattr(box, "AR_HOME", tmp_path)
    monkeypatch.setattr(box, "BOXES", tmp_path / "boxes")
    box.save_rec("r1b2", {"agent": "r1b2", "id": 42, "gpu": "RTX_3090", "dph": 0.2,
                          "created": 0, "host": "h", "port": 1})
    on_box = b"the box's checkpoint " * 50
    dst = tmp_path / "runs" / "r1b2" / "r1b2_nth"
    dst.mkdir(parents=True)
    (dst / "ckpt_latest.pt").write_bytes(b"the good copy pulled earlier")
    monkeypatch.setattr(box, "ssh", lambda rec, cmd, **k: subprocess.CompletedProcess(
        cmd, 0, hashlib.md5(on_box).hexdigest() + "\n", ""))
    scp = {"mode": "drop"}

    def fake_run(args, **k):
        if args[0] == "ssh":                     # the tar stream: one greedy eval
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                data = b'{"header": 1}\n'
                ti = tarfile.TarInfo("r1b2_nth/traj_0000000001.jsonl")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
            k["stdout"].write(buf.getvalue())
            return subprocess.CompletedProcess(args, 0, b"", b"")
        assert args[0] == "scp", args
        Path(args[-1]).write_bytes(on_box if scp["mode"] == "whole" else on_box[:300])
        if scp["mode"] == "timeout":
            raise subprocess.TimeoutExpired(args, 1800)
        return subprocess.CompletedProcess(args, 0 if scp["mode"] == "whole" else 1, b"", b"")

    monkeypatch.setattr(box.subprocess, "run", fake_run)
    for mode in ("drop", "timeout"):
        scp["mode"] = mode
        box.main(["pull", "--agent", "r1b2", "--run", "r1b2_nth", "--ckpt"])
        assert (dst / "ckpt_latest.pt").read_bytes() == b"the good copy pulled earlier", mode
        assert not (dst / "ckpt_latest.pt.part").exists()
    assert (dst / "traj_0000000001.jsonl").read_bytes() == b'{"header": 1}\n'
    scp["mode"] = "whole"
    box.main(["pull", "--agent", "r1b2", "--run", "r1b2_nth", "--ckpt"])
    assert (dst / "ckpt_latest.pt").read_bytes() == on_box
