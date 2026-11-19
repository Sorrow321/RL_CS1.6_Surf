"""The launch path of map-sharded DDP (mm/ddp, 2026-11-17), no GPU and no rental:

* tools/run_arm.sh SCRATCH=1 MAPS=... RANKS=N: the same trainer tokens as one rank with
  --run <run> swapped for --ddp-maps shard, started through tools/ddp_launch.sh (a stub
  interpreter records the warm-caches pass and the torchrun call); RANKS unset = the old launch;
  RANKS without MAPS, or with fewer maps than ranks, is refused; the resume branch takes RANKS
  the same way;
* autoresearch/box.py: rent --num-gpus N filters offers on exactly N cards, a per-BOX cap of
  the card's cap x N, >= 16 x N threads and verified hosts (vastai stubbed; --dry-run only);
  launch --multimap passes RANKS from the box's GPU count (or --ranks) and RECORD_GPU=free on a
  multi-GPU box, and a one-GPU record launches exactly as before.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "autoresearch"))
sys.path.insert(0, str(ROOT / "python"))

TAIL = ["--envs", "64", "--keys-hold", "--maxvel", "3500,4000", "--tick-ms", "7.63"]


def _stub(tmp_path):
    """an interpreter that APPENDS each call's argv (one call per block) and exits"""
    p = tmp_path / "stubpy"
    p.write_text("#!/bin/bash\n{ printf '%s\\n' \"$@\"; echo '--- end'; } >> \"$ARGS_OUT\"\n"
                 "exit 0\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _tree(tmp_path):
    d = tmp_path / "tree"
    (d / "tools").mkdir(parents=True)
    for f in ("run_arm.sh", "ddp_launch.sh"):
        (d / "tools" / f).write_text((ROOT / "tools" / f).read_text())
    return d


def _calls(path):
    out, cur = [], []
    for ln in Path(path).read_text().splitlines():
        if ln == "--- end":
            out.append(cur)
            cur = []
        else:
            cur.append(ln)
    return out


def _run(tree, env_extra, args_out, tail=TAIL):
    env = dict(os.environ, SCRATCH="1", BUDGET="1000", RECORD_EVERY="75e6",
               ARGS_OUT=str(args_out), OMP_NUM_THREADS="1", **env_extra)
    return subprocess.run(["bash", str(tree / "tools" / "run_arm.sh"), "ddptest_run"] + tail,
                          capture_output=True, text=True, env=env, timeout=120)


def test_ranks_swaps_run_for_the_shard_flag_and_nothing_else(tmp_path):
    stub, tree = _stub(tmp_path), _tree(tmp_path)
    one, two = tmp_path / "one.txt", tmp_path / "two.txt"
    _run(tree, {"PYTHON": str(stub), "MAPS": "maps/a.bsp,maps/b.bsp"}, one)
    (tree / "runs" / "ddptest_run.pid").unlink()
    r = _run(tree, {"PYTHON": str(stub), "MAPS": "maps/a.bsp,maps/b.bsp", "RANKS": "2"}, two)
    c1, c2 = _calls(one), _calls(two)
    assert len(c1) == 1 and c1[0][:2] == ["-u", "python/train_fast.py"]
    t1 = c1[0][2:]
    # ddp_launch.sh: the single-process warm-caches pass, then torchrun x2 under the SAME python
    assert len(c2) == 2, c2
    warm, trun = c2
    assert warm[:5] == ["-u", "python/train_fast.py", "--warm-caches", "--run",
                        "ddptest_run_warm"]
    assert trun[:6] == ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                        "python/train_fast.py", "--run-name"]
    assert trun[6] == "ddptest_run"
    i = t1.index("--run")
    want = t1[:i] + ["--ddp-maps", "shard"] + t1[i + 2:]
    assert warm[5:] == want and trun[7:] == want
    assert "ranks: 2 (map-sharded DDP" in r.stdout and "tools/ddp_launch.sh 2 ddptest_run" \
        in r.stdout


def test_ranks_dry_run_and_refusals(tmp_path):
    stub, tree = _stub(tmp_path), _tree(tmp_path)
    out = tmp_path / "x.txt"
    r = _run(tree, {"PYTHON": str(stub), "MAPS": "maps/a.bsp,maps/b.bsp", "RANKS": "2",
                    "DRY_RUN": "1"}, out)
    assert r.returncode == 0 and not out.exists()
    assert "DRY_RUN_RANKS 2" in r.stdout
    args = [ln[len("DRY_RUN_ARG "):] for ln in r.stdout.splitlines()
            if ln.startswith("DRY_RUN_ARG ")]
    assert "--run" not in args and args[args.index("--ddp-maps") + 1] == "shard"
    for env, msg in (({"RANKS": "2", "MAP": "maps/a.bsp"}, "give MAPS"),
                     ({"RANKS": "3", "MAPS": "maps/a.bsp,maps/b.bsp"}, "every rank needs a map"),
                     ({"RANKS": "x", "MAPS": "maps/a.bsp,maps/b.bsp"}, "RANKS must be")):
        r = _run(tree, {"PYTHON": str(stub), "DRY_RUN": "1", **env}, out)
        assert r.returncode == 1 and msg in r.stdout, (env, r.stdout[-500:])
    # RANKS=1 is the single-process launch, token for token
    r1 = _run(tree, {"PYTHON": str(stub), "MAPS": "maps/a.bsp,maps/b.bsp", "RANKS": "1",
                     "DRY_RUN": "1"}, out)
    r0 = _run(tree, {"PYTHON": str(stub), "MAPS": "maps/a.bsp,maps/b.bsp", "DRY_RUN": "1"},
              out)
    assert r1.stdout == r0.stdout and "DRY_RUN_RANKS" not in r0.stdout


def test_resume_branch_takes_ranks(tmp_path):
    import torch
    tree = _tree(tmp_path)
    ck = tree / "runs_ckpt.pt"
    torch.save({"global_step": 4096, "config": {}}, ck)
    env = dict(os.environ, CKPT=str(ck), ARM_RESUME="1", BUDGET="1000", RECORD_EVERY="75e6",
               CKF=str(tmp_path / "ck_step"), PYTHON=sys.executable, DRY_RUN="1")
    outs = {}
    for ranks in ("1", "2"):
        r = subprocess.run(["bash", str(tree / "tools" / "run_arm.sh"), "ddptest_r", "--envs",
                            "64"], capture_output=True, text=True,
                           env=dict(env, RANKS=ranks), timeout=300)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        outs[ranks] = [ln[len("DRY_RUN_ARG "):] for ln in r.stdout.splitlines()
                       if ln.startswith("DRY_RUN_ARG ")]
    assert outs["1"][:4] == ["--ckpt", str(ck), "--run", "ddptest_r"]
    assert outs["2"][:4] == ["--ckpt", str(ck), "--ddp-maps", "shard"]
    assert outs["2"][4:] == outs["1"][4:] and outs["1"][5] == str(4096 + 1000)


# --- box.py ---------------------------------------------------------------------------------

def _offer(i, n, gpu="RTX_4080", dph=0.43, cores=48, ver="verified", cpu="AMD EPYC 7532"):
    return {"id": i, "gpu_name": gpu.replace("_", " "), "num_gpus": n, "dph_total": dph,
            "cpu_cores_effective": cores, "verification": ver, "machine_id": 1000 + i,
            "host_id": 2000 + i, "public_ipaddr": f"10.0.0.{i}", "cpu_name": cpu,
            "cpu_ram": 64000, "inet_down": 900.0, "geolocation": "CA"}


@pytest.fixture
def boxmod(monkeypatch, tmp_path):
    import box
    monkeypatch.setattr(box, "AR_HOME", tmp_path)
    monkeypatch.setattr(box, "BOXES", tmp_path / "boxes")
    monkeypatch.setitem(box.CFG, "max_dph", {"RTX_4080": 0.25, "RTX_4090": 0.45})
    monkeypatch.setitem(box.CFG, "gpu_tiers", [["RTX_4080", "RTX_4090"]])
    monkeypatch.setitem(box.CFG, "offer_query",
                        "num_gpus=1 reliability>0.98 cpu_cores_effective>=32 verified=true "
                        "gpu_ram>=15")
    monkeypatch.setattr(box, "blocked_keys", lambda: ({1004}, set(), set()))
    return box


def test_offer_query_for_n_cards(boxmod):
    q1 = boxmod.offer_query(1)
    assert q1 == boxmod.CFG["offer_query"]                       # one card: verbatim
    q2 = boxmod.offer_query(2)
    assert "num_gpus=2" in q2 and "num_gpus=1" not in q2
    assert "cpu_cores_effective>=32" in q2 and "verified=true" in q2
    q3 = boxmod.offer_query(3)
    assert "cpu_cores_effective>=48" in q3                       # 16 threads per card
    boxmod.CFG["offer_query"] = "num_gpus=1 verified=any"
    assert "verified=true" in boxmod.offer_query(2) and "verified=any" not in \
        boxmod.offer_query(2)


def test_offers_filter_n_cards(boxmod, monkeypatch):
    got = [_offer(1, 2, dph=0.43), _offer(2, 2, dph=0.55),            # 0.55 > 2 x 0.25
           _offer(3, 2, cores=24),                                   # 12 threads per card
           _offer(4, 2),                                             # blocklisted machine
           _offer(5, 2, ver="deverified"), _offer(6, 1, dph=0.2),    # deverified / one card
           _offer(7, 2, cpu="Intel Xeon E5-2680 v4"),                # cpu_exclude
           _offer(8, 2, gpu="RTX_4090", dph=0.85)]
    seen = {}

    def fake_vast(*args, **k):
        seen["q"] = args[2]
        return got
    monkeypatch.setattr(boxmod, "vast", fake_vast)
    o2 = boxmod.offers(num_gpus=2)
    assert [o["id"] for o in o2] == [1, 8] and all(o["num_gpus"] == 2 for o in o2)
    assert "num_gpus=2" in seen["q"]
    o1 = boxmod.offers()
    assert [o["id"] for o in o1] == [6] and "num_gpus=1" in seen["q"]


def test_rent_dry_run_n_cards(boxmod, monkeypatch, capsys):
    monkeypatch.setattr(boxmod, "vast", lambda *a, **k: (
        [] if a[:2] == ("show", "instances") else {"credit": 50.0} if a[:2] == ("show", "user")
        else [_offer(1, 2, dph=0.43), _offer(6, 1, dph=0.2)]))
    monkeypatch.setitem(boxmod.CFG, "budget_rate_usd_h", 5.0)
    called = []
    monkeypatch.setattr(boxmod, "watchdog", lambda *a: called.append(a))
    boxmod.main(["rent", "--agent", "ddpdry", "--num-gpus", "2", "--dry-run"])
    out = json.loads(capsys.readouterr().out)
    assert out["would_rent"]["id"] == 1 and out["num_gpus"] == 2
    assert out["cap_per_box"] == 0.5 and "num_gpus=2" in out["query"]
    assert not called and not boxmod.load_rec("ddpdry")          # nothing rented or written


MM = {"maps": ["maps/surf_src_sidistic.bsp", "maps/surf_unitfarmer2.bsp",
               "maps/surf_princessburglar_b26.bsp"], "maxvel": [4000, 3500, 3500], "envs": 1029}


def test_launch_takes_ranks_from_the_box(boxmod, monkeypatch):
    monkeypatch.setitem(boxmod.CFG, "incumbent_recipe", ["--envs", "1024", "--maxvel", "3500"])
    monkeypatch.setitem(boxmod.CFG, "multimap", dict(MM))
    sent = []
    monkeypatch.setattr(boxmod, "ssh", lambda rec, cmd, **k: sent.append(cmd) or
                        boxmod.subprocess.CompletedProcess(cmd, 0, "", ""))
    boxmod.save_rec("dd2", {"agent": "dd2", "id": 5, "gpu": "RTX_4080", "dph": 0.43,
                            "created": 0, "host": "h", "port": 1, "num_gpus": 2})
    boxmod.main(["launch", "--agent", "dd2", "--run", "dd2_a", "--multimap"])
    assert "RANKS=2 " in sent[-1] and "RECORD_GPU=free " in sent[-1] and "SCRATCH=1 " in sent[-1]
    assert boxmod.load_rec("dd2")["runs"][-1]["ranks"] == 2
    boxmod.main(["launch", "--agent", "dd2", "--run", "dd2_b", "--multimap", "--ranks", "1",
                 "--record-gpu", "1"])
    assert "RANKS=" not in sent[-1] and "RECORD_GPU=1 " in sent[-1]
    for argv in (["--multimap", "--ranks", "4"],                 # 4 ranks for 3 maps
                 ["--ranks", "2"],                               # sharding needs --multimap
                 ["--multimap", "--ranks", "3"]):                # 3 ranks on a 2-GPU box
        with pytest.raises(SystemExit):
            boxmod.main(["launch", "--agent", "dd2", "--run", "dd2_x"] + argv)
    # a one-GPU record: the launch it always was
    boxmod.save_rec("dd1", {"agent": "dd1", "id": 6, "gpu": "RTX_4090", "dph": 0.4,
                            "created": 0, "host": "h", "port": 1})
    boxmod.main(["launch", "--agent", "dd1", "--run", "dd1_a", "--multimap"])
    assert "RANKS=" not in sent[-1] and "RECORD_GPU" not in sent[-1]
    assert "ranks" not in boxmod.load_rec("dd1")["runs"][-1]
