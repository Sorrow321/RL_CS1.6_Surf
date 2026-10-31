"""--auto-spine-recheck (zlr1b4, 2026-10-31): a search's spine is adopted only if its end still
beats the LATEST greedy frontier by --auto-spine-gain.

zlr1b4_e3: an --auto-spine-async search started on the 711M weights (greedy frontier d 4,511)
returned at ~1,072M a spine ending at d 4,376 (31%) while the greedy was already at d 2,752 (57%);
adopted, its curriculum pulled the greedy back to the spine's dead end (57% -> 34% in 75M steps).

(a) spine_is_fresh: the rule, and None (no eval read yet) keeps the search's verdict.
(b) the trainer: the flag needs --auto-spine, is written to the config only when on, and the
    adoption path checks it before the curriculum is touched; the recorder lists it TRAIN_ONLY.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def test_the_rule():
    from train_fast import spine_is_fresh
    # the zlr1b4_e3 case: a 31% spine against a 57% greedy is stale
    assert not spine_is_fresh(4376.0, 2752.0, 200.0)
    # a spine 200 u past the latest frontier is fresh; 199 u is not
    assert spine_is_fresh(2552.0, 2752.0, 200.0)
    assert not spine_is_fresh(2553.0, 2752.0, 200.0)
    # no eval read yet: the search's own gain check stands
    assert spine_is_fresh(4376.0, None, 200.0)


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--auto-spine-recheck"' in src
    assert 'raise SystemExit("--auto-spine-recheck needs --auto-spine")' in src
    assert 'meta["config"]["auto_spine_recheck"] = 1' in src
    # the check sits between loading the spine and resetting the curriculum on it
    i_load = src.index('sp = np.load(res["spine"])')
    i_chk = src.index("spine_is_fresh(_de_new", i_load)
    i_reset = src.index("demo.reset_spine(sp)", i_load)
    assert i_load < i_chk < i_reset


def test_the_recorder_lists_it_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    block = src[src.index("TRAIN_ONLY = frozenset({"):]
    block = block[:block.index("})")]
    assert re.search(r'"auto_spine_recheck"', block)
