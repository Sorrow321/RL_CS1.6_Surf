"""--bc-reload (uf2r2b3, 2026-10-31): train_fast re-reads --bc-file when its mtime changes, so an
outer search loop on the box (beam_tas on the newest ckpt_latest -> plan_to_bc -> os.replace of
the file) feeds ONE training run the planner's newest lines - the expert loop without a relaunch,
the run's own greedy evals (the only thing the grader counts) uninterrupted.

(a) the refusal rule (surfgym.bc.reload_refusal) on real BCDataset files: a same-layout file is
    accepted whatever its row count; a file without held-key columns is refused under
    --bc-keys-engine, one without a value target under --bc-value-coef;
(b) the flag surface: default off, dumped only when set, refused without --bc-file, TRAIN_ONLY in
    record_ckpt, checked on the --spawn-states cadence;
(c) off = byte-identical: the first read is the one constructor call with the pre-change seed,
    and every reload line sits behind BC_RELOAD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

from test_expert_targets import _bc_args  # noqa: E402

KW = dict(n_latch=0, obs_reward=False, view_continuous=True, yaw_adaptive=True,
          pitch_rate_max_deg=1.33, view_absolute="velocity", keys_hold=True)


def _ds(path):
    import torch
    from surfgym.bc import BCDataset
    return BCDataset(path, torch.device("cpu"), **KW)


def test_same_layout_is_accepted_whatever_the_row_count(tmp_path):
    from surfgym.bc import reload_refusal, save_bc_dataset
    a9, v9, f9 = _bc_args(9)
    a20, v20, f20 = _bc_args(20)
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    save_bc_dataset(p1, *a9, view=v9, keys=f9)
    save_bc_dataset(p2, *a20, view=v20, keys=f20)
    d1, d2 = _ds(p1), _ds(p2)
    assert d1.keys_col is not None and d1.keys_col == d2.keys_col
    assert (d1.n, d2.n) == (9, 20)
    assert reload_refusal(d2, d1.keys_col, keys_engine=True, value_coef=0.0) is None
    assert reload_refusal(d2, d1.keys_col, keys_engine=False, value_coef=0.0) is None


def test_a_layout_change_is_refused(tmp_path):
    from surfgym.bc import reload_refusal, save_bc_dataset
    a, v, f = _bc_args(9)
    p = tmp_path / "a.npz"
    save_bc_dataset(p, *a, view=v, keys=f)
    d = _ds(p)
    # the compiled term reads the held keys at another column: refused under --bc-keys-engine
    why = reload_refusal(d, d.keys_col + 1, keys_engine=True, value_coef=0.0)
    assert why is not None and "held-key columns" in why
    # ... and irrelevant without it
    assert reload_refusal(d, d.keys_col + 1, keys_engine=False, value_coef=0.0) is None
    # a file with no value target cannot feed --bc-value-coef
    assert not d.has_value
    why = reload_refusal(d, d.keys_col, keys_engine=True, value_coef=0.25)
    assert why is not None and "value target" in why


def test_the_flag_surface_and_off_is_byte_identical():
    tf = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--bc-reload", action="store_const", const=1, default=None' in tf
    assert 'meta["config"]["bc_reload"] = 1' in tf                 # dumped only when set
    assert "--bc-reload needs --bc-file" in tf
    assert "BC_RELOAD = bool(args.bc_reload)" in tf
    # the first read: one constructor, the pre-change seed (no offset)
    assert "seed=args.seed + 31 * D.rank + _seed_off" in tf
    assert "        bc = _make_bc(args.bc_file)\n" in tf
    # every reload line is behind the flag, on the --spawn-states cadence
    assert "if BC_RELOAD and bc is not None and it_no % 20 == 0:" in tf
    assert "_why = bc_reload_refusal(_nb, BC_KC, BCKE, BCV)" in tf
    assert tf.count("if BC_RELOAD") == 4
    # the reload never touches the compiled step or the optimizer
    i0 = tf.index("if BC_RELOAD and bc is not None and it_no % 20 == 0:")
    i1 = tf.index("keeping the previous rows", i0)
    blk = tf[i0:i1]
    assert "torch.compile" not in blk and "opt." not in blk and "bc_step" not in blk
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"bc_reload",' in rc                                     # TRAIN_ONLY
