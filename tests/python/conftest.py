"""Test-suite environment.

SELF_STATES=1: the trainer (train_fast.py) refuses any spawn-state / imitation
/ route source unless the operator declares it policy-derived (CLAUDE.md
section 0). Every such file the suite uses is synthetic or cut from a test
policy's own recording, so the suite declares it once here.
"""
import os

os.environ.setdefault("SELF_STATES", "1")
