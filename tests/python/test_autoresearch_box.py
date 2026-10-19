"""autoresearch/box.py: the rental gate (user, 2026-10-20). The vast CLI is mocked; nothing rents.

* offers: whitelisted single-GPU RTX cards under their cap, blocklisted hosts out, in the
  user's order (3090-class tier first, cheapest first, then 4090, then 5090);
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
    # 3090 at 0.31 is over its 0.30 cap; machine 777 is blocklisted; the 4090 at 0.50 is over
    # 0.45; the 2-GPU 5090 is out. Tier 0 (3090 / 5080) first, cheapest first
    assert ids == [4, 2, 5, 7], ids
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


FIELD = ROOT / "maps" / "surf_src_sidistic.goal_32.npz"
GR = ROOT / "runs" / "sidEM2cap2surfGR" / "traj_0755499008.jsonl"


@pytest.mark.skipif(not (FIELD.exists() and GR.exists()), reason="needs the sidistic field + an eval")
def test_score_rejects_an_eval_that_does_not_start_at_the_map_start(tmp_path):
    import score
    good = score.score_file(GR)
    assert good["score"] == pytest.approx(27.222, abs=0.01) and good["max_cp"] == 1
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
