"""--auto-spine-no-retreat (azr1b1, 2026-11-10): a search's spine is adopted only if its end also
beats the CURRENT spine's end by --auto-spine-gain - the curriculum's line never moves backwards.

azr1b1_pool1 (one pool: 90% of the training starts on the curriculum's line): search 17 started
from a bimodal eval whose MEDIAN deepest d read 52,192 while 4/9 episodes were at 84%; its line
ended at d 44,423 (~57%) and replaced line 15 (end d 15,007, ~85%). The greedy fell back from
84% to the CP2 block (~31%) within ~0.3B steps. unr1b1's --ge-line had the same rule for its own
loop ("and the current line's end by as much").

(a) spine_moves_on: the rule, and None (no current spine) adopts.
(b) the trainer: the flag needs --auto-spine, is written to the config only when on, and the
    adoption path checks it after --auto-spine-recheck and before the curriculum is touched; off,
    the adoption path is the old one; the recorder lists it TRAIN_ONLY.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def test_the_rule():
    from train_fast import spine_moves_on
    # the azr1b1_pool1 case: a line to 44,423 does not replace a line to 15,007
    assert not spine_moves_on(44423.0, 15007.0, 200.0)
    # equal ends, or a gain under 200 u, do not replace either
    assert not spine_moves_on(15007.0, 15007.0, 200.0)
    assert not spine_moves_on(14808.0, 15007.0, 200.0)
    # 200 u further does
    assert spine_moves_on(14807.0, 15007.0, 200.0)
    # the first replacement of the scratch spine (the spawn) always moves on
    assert spine_moves_on(97915.0, 103000.0, 200.0)
    # no current spine: adopt
    assert spine_moves_on(44423.0, None, 200.0)


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--auto-spine-no-retreat", action="store_true"' in src
    assert 'raise SystemExit("--auto-spine-no-retreat needs --auto-spine")' in src
    assert 'meta["config"]["auto_spine_no_retreat"] = 1' in src
    # config only when on
    i_cfg = src.index('meta["config"]["auto_spine_no_retreat"] = 1')
    assert "if args.auto_spine_no_retreat:" in src[i_cfg - 80:i_cfg]
    # the check sits after the recheck and before resetting the curriculum on the new spine,
    # and compares with the CURRENT spine (demo.S, before reset_spine replaces it)
    i_load = src.index('sp = np.load(res["spine"])')
    i_recheck = src.index("spine_is_fresh(_de_new", i_load)
    i_chk = src.index("spine_moves_on(_de_new, _de_cur", i_load)
    i_cur = src.index('_de_cur = float(_f.sample(np.asarray(demo.S[-1]["origin"]', i_load)
    i_reset = src.index("demo.reset_spine(sp)", i_load)
    assert i_load < i_recheck < i_cur < i_chk < i_reset
    # guarded by the flag: off, nothing between the recheck and reset_spine runs
    blk = src[i_recheck:i_reset]
    assert "if args.auto_spine_no_retreat:" in blk


def test_the_help_renders():
    # argparse %-formats help text: a bare % in it would crash --help
    import argparse
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    i = src.index('ap.add_argument("--auto-spine-no-retreat"')
    j = src.index("ap.add_argument(", i + 10)
    call = src[i:j].strip()
    ap = argparse.ArgumentParser()
    eval(call, {"ap": ap})                         # the exact add_argument call
    text = " ".join(ap.format_help().split())      # the formatter wraps lines
    assert "never moves backwards" in text
    assert "84% of" not in text and "4/9 episodes were at 84%, and" in text


def test_the_recorder_lists_it_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    block = src[src.index("TRAIN_ONLY = frozenset({"):]
    block = block[:block.index("})")]
    assert re.search(r'"auto_spine_no_retreat"', block)
