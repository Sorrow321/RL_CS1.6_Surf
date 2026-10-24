"""A stand-in for tools/beam_tas.py in the --auto-spine CPU tests (tests/python/test_auto_spine.py):
it takes beam_tas's arguments, checks the checkpoint the trainer saved for it exists, and writes
<out-dir>/beam_frontier.npz from the first AUTOSPINE_STUB_N states of AUTOSPINE_STUB_SRC (a
policy-derived spine in the repo) - a frontier line far deeper than a smoke policy reaches."""
import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--max-ticks", type=int, default=0)
ap.add_argument("--torch-seed", type=int, default=0)
a, _ = ap.parse_known_args()
if not os.path.isfile(a.ckpt):
    sys.exit(f"stub: no checkpoint at {a.ckpt}")
s = np.load(os.environ["AUTOSPINE_STUB_SRC"])[:int(os.environ.get("AUTOSPINE_STUB_N", "600"))]
np.savez(os.path.join(a.out_dir, "beam_frontier.npz"), states=s, spawn_state=s[:1],
         replay_ok=np.int32(1), act_every=np.int32(4), frontier_tick=np.int32(len(s) - 1))
print(f"stub beam: {len(s)} states, max-ticks {a.max_ticks}, seed {a.torch_seed}")
