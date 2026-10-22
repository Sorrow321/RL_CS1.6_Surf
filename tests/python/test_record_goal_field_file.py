"""record_ckpt mirrors --goal-field-file for what the policy SEES (2026-10-22, r5b4).

The trainer's potential channel, compass, pot-rate and ratchet gap all read the shaping
field --goal-field-file names (slot.reward_field). The recorder listed the key TRAIN_ONLY
and rendered the free-space field instead, so a recording of such a checkpoint showed the
policy an image it never trained on. Now:

* a checkpoint whose config names a field file records with it (and says so);
* a checkpoint whose field file is missing refuses to record instead of falling back.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ez_viewc import _run, _train, needs_run  # noqa: E402


@needs_run
def test_the_recorder_reads_the_field_file_and_refuses_a_missing_one():
    import torch
    run = "gff_rec"
    _train(run, ["--race-ratchet", "--obs-compass", "1", "--obs-pot-rate", "1",
                 "--obs-potential", "norm"], steps="2048")
    d = ROOT / "runs" / run
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    # the shipped field itself, under another name: the lattice matches by construction
    field = d / "copy_goal_32.npz"
    shutil.copy(ROOT / "maps" / "surf_src_sidistic.goal_32.npz", field)
    ck["config"]["goal_field_file"] = str(field)
    torch.save(ck, d / "ck_gff.pt")
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"), str(d / "ck_gff.pt"),
              "--episodes", "1", "--ep-ticks", "120", "--out", str(d / "rec.jsonl")])
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "mirrored: the observation channels read it" in out
    assert "CONFIG MISMATCH" not in out
    ck["config"]["goal_field_file"] = str(d / "missing_goal_32.npz")
    torch.save(ck, d / "ck_missing.pt")
    r = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
              str(d / "ck_missing.pt"), "--episodes", "1", "--ep-ticks", "120",
              "--out", str(d / "rec2.jsonl")])
    assert r.returncode != 0
    assert "which is not here" in r.stdout + r.stderr
    shutil.rmtree(d, ignore_errors=True)
