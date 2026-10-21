# `--greedy-tail-frac`: train where the deployed greedy line fails

Built 2026-10-21 (autoresearch r3b2, round 3) after the same mechanism, run with an external
refresh loop, took sidistic's greedy line through the cavity loop and past CP2 (28.84 -> 32.80;
`autoresearch/NOTES.md`, 03:25-04:40 UTC). Default off, byte-identical when off.

## What it does

* At every periodic greedy eval from the true start (`--record-every`), a `record_rollout` tick
  hook (`surfgym.greedy_tail.TailCapture`) keeps env 0's full core state every tick, cut into
  episodes and labelled like `record.py` (fail / done by the finish bonus / trunc).
* `surfgym.greedy_tail.cut_tails` keeps, for every FAILED episode no longer than 120 s, the states
  from `--greedy-tail-secs` (3) to `--greedy-tail-skip` (0.25) seconds before its end, every
  `--greedy-tail-every` (5) ticks, dropping those slower than `--greedy-tail-min-speed` (300 u/s,
  3-D; the `--respawn-min-speed` rule: a stalled state seeds stalled spawns).
* Those rows replace `--greedy-tail-frac` of every training spawn pool (uniform positions, uniform
  rows: the `--spawn-states` code path and slot, clocks zeroed) until the next eval re-cuts them; a
  cut of fewer than 20 rows keeps the previous rows; before the first eval there are none.
* The eval line shows `gtail <rows> rows/<n> fails` (or `gtail kept <n>`).

## Why

The score, and the honest frontier, are read from the greedy line. Its failure point is the
deployed policy's frontier, but the reservoir (states harvested from mostly stochastic, mostly
reservoir-spawned episodes, uniform over states) puts almost nothing there: on sidistic under 1%
of reservoir states lay past the slot where the greedy line died. Returning 15% of episodes to the
greedy line's own last seconds (Go-Explore's "return to the frontier, then explore", with the
frontier defined by the policy that is evaluated) gave, warm from the incumbent at 7.74B and with
`--time-pen 0 --surf-bonus 0`:

* +230M steps: 7/9 greedy cavity entries at the human's entry speed (tp0x: 2-5 slow ones);
* +550M: ramp take-off at the human's strength; the training ceiling past the field route's limit;
* +740M: the greedy loop through the tunnel and the room, CP2 entered, 32.38; +860M: 32.80, 9/9.

A spawn on the tail carries its own ratchet record (the spawn's d), so inside a dip the climb back
toward the record is PAID for those episodes: the dense gradient the start-anchored line lacks.

Generic: a tail length, a share, a speed floor; nothing about the map. Champion-free: the states
are the policy's own greedy recording (no file, so no `SELF_STATES` declaration is involved).

## Flags

| flag | default | meaning |
|---|---|---|
| `--greedy-tail-frac` | 0 (off) | share of every training spawn pool, in (0, 1) |
| `--greedy-tail-secs` | 3.0 | the tail starts this long before a failed episode's end |
| `--greedy-tail-skip` | 0.25 | the last seconds left out (doomed) |
| `--greedy-tail-every` | 5 | keep every k-th tick |
| `--greedy-tail-min-speed` | 300 | drop states slower than this (u/s, 3-D) |

Single-map, single-process; refused with `--spawn-states` (the same slot); a knob without the share
is refused; the config keys are written only when on and a flagless resume restores them;
TRAIN_ONLY in `tools/record_ckpt.py`. Offline twin: `tools/greedy_tail_states.py` (same cut, from a
`record_ckpt.py --dump-states` recording; `autoresearch/analysis/greedy_tail_loop.sh` is the
external refresh loop the first runs used).

## Pinned (`tests/python/test_greedy_tail.py`)

The capture splits and labels episodes and drops an open one; the cut keeps only failed tails in
the window, applies the speed floor and the length cap; the sidistic CPU smoke (eval-stall kills so
a fresh policy's eval episodes fail) cuts tails at every eval, dumps the keys and resumes them; the
refusals; flag off = the pre-flag trainer (config, every non-timing progress.csv column, eval
trajectory bytes).
