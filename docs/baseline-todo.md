# baseline: deferred picks

Two commits were deliberately NOT folded into `baseline`. Both touch
`python/train_fast.py`, which four other agents are editing on feature
branches right now, so picking them here would put a conflict in front of
every one of those merges. The integrator applies them AFTER those branches
land.

| commit | branch | what it adds | why deferred |
|---|---|---|---|
| `621689d` | petrus-chain (also on round18) | `--stall-eps` plumbing (the stall threshold is per-call and scales with `--act-every`; default 32.0, bit-identical) | conflicts with `python/train_fast.py`, under concurrent edit on four feature branches |
| `f22588b` | petrus-chain (also on round18) | `--respawn-vtarget` (respawn velocity dose in u/s, not a multiplier) plus `tools/place_probe.py` and `tests/python/test_respawn_vtarget.py` | conflicts with `python/train_fast.py` (and `python/surfgym/respawn.py`), under concurrent edit on four feature branches |

Both are pure additions in their own commits, so `git cherry-pick -x` should
apply cleanly once `train_fast.py` has settled. `621689d` also carries a
`docs/research-results.md` section; resolve that half the way the rest of
this branch did - keep the ledger text already here and append the incoming
section at the end of the file under its own heading.
