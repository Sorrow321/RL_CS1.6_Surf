# GRU probe: is the memory load-bearing? (2026-09-04, CPU)

**Owner: "Did GRU not help at all? Are we sure in implementation?"**

**The implementation is sound and the memory is heavily used. The GRU arms are
not a null mechanism - they are a null RESULT at matched steps.**

## Method

`tools/gru_probe.py` drives `tools/record_ckpt.py` in-process on CPU (no CUDA,
no trainer): render, goal field, spawn pool, stall rule and config audit are the
recorder's own. One spawn seed, one goal draw (identical across conditions: arc
95.6 / 63.4 / 69.3%), three conditions:

* **carry** - the trained policy, h carried, zeroed at episode start.
* **zero** - h forced to 0 entering every decision. The tower still sees
  `gru(f_t, 0)`, so this isolates the RECURRENCE, not the module.
* **lag** - chain normal, tower handed the PREVIOUS decision's GRU output.

Inside **carry** the **zero** action is also computed at every state carry
visits: agreement is measured on ONE state sequence, not two diverged ones.
The probe substitutes `goalsys.eval_hooks`' fixed-route goal draw, which
`record_ckpt` does not mirror; carry reproduces the arm's own eval band
(91-92k against its 92,119 at 2.19B), so the setup is faithful.

## Result (3 greedy eps, cap 3,000 decisions, `eval_honesty --order-only 16`)

| ckpt | cond | corridor mean | MAX | ep len (s) | vs carry |
|---|---|---|---|---|---|
| xW1GRU (shallow) | carry | 91,789 | 92,374 | 39.0-39.8 | - |
| | **zero** | **12,412** | **16,779** | 5.3-11.4 | **-86%** |
| | lag | 38,441 | 59,324 | 10.4-27.2 | -58% |
| xW2GRU (deep) | carry | 87,733 | 118,027 | 19.6-54.7 | - |
| | **zero** | **16,375** | **17,417** | 11.0-11.3 | **-81%** |
| | lag | 65,175 | 106,002 | 18.9-58.8 | -26% |

Paired per-decision disagreement, carry vs zero, on carry's own states:

| ckpt | dec | any | yaw | pitch | fwd | side | jump | duck | logit gap |
|---|---|---|---|---|---|---|---|---|---|
| xW1GRU | 2,958 | **84.4%** | 65.1% | 50.1% | 4.9% | 9.9% | 35.3% | 5.7% | 4.99 |
| xW2GRU | 3,081 | **74.3%** | 42.5% | 47.0% | 11.8% | 5.7% | 13.4% | 6.4% | 4.10 |

The >95%-agreement "GRU unused" branch is nowhere near true: deleting history
changes 3 of every 4 decisions and the frontier collapses 5-7x. Steering
(yaw, pitch) is where the memory lives.

## Sanity

* `run.json` both arms: `rnn=gru`, `rnn_size=256` (xW1 depth 2 / mult 1; xW2
  depth 4 / mult 2).
* Both ckpts: `gru.weight_ih_l0 [768,522]`, `weight_hh_l0 [768,256]`,
  `bias_ih_l0`, `bias_hh_l0` = **599,040 = 3*(256*(522+256+2))**, exact.
* Trained, not at init: |W_ih| 64.6 / 47.8, |W_hh| 50.0 / 46.3 against
  orthogonal init's 16.0; biases 3.9-5.8 against zero init.
* Probe forward == `Policy.forward` to **0.0**. `test_rnn_policy.py`: 14 pass.

## So why did the arms look flat?

Throughput, not mechanism. Median fps 251,322 (xW1GRU) vs 261,200 (xW1CTL),
221,633 (xW2GRU) vs 277,970 (xW2DEEP4): in the same hour the GRU arms consumed
2.27B / 2.26B steps against 2.95B / 3.00B. At **matched steps** best corridor
MAX is 104,101 vs 92,187 (wave 1) and 119,643 vs 117,389 (wave 2) - inside the
27% seed-noise floor either way. The controls' lead is bought with ~30% more
steps.

Artifacts: `probe.json`, `probe.log`, `xW{1,2}GRU_{carry,zero,lag}.jsonl`.
