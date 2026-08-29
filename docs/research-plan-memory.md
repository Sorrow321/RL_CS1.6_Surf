# xMEM - strided frame history + action history (design)

Date: 2026-08-29. Requested by the user; supersedes, for this one arm, the
"frame stacking: dead" row in `docs/ideas-backlog.md`. Base branch: `mmddp`
(newest trainer). Runs LOCALLY on the 5090. Arm branch: `memarm`. Run name:
`xMEM`.

## 1. What was actually tested before, and why it does not close the question

The prior null is sF1 (rounds 10-14, 2026-08-17, branch `framestack`, merged
cf9dd0a, ancestor of `mmddp`):

* `--frame-stack K` with `STACK_STRIDES = (1, 2, 4, 8)` in DECISIONS,
  newest first (`train_fast.py:328`). It was already strided - it stacked
  decisions, not physics ticks - but sF1 ran `--frame-stack 4` = offsets
  (0, 1, 2, 4) decisions at act_every 3, i.e. 0/30/60/120 ms. The whole
  history window was 120 ms, and two of its three past frames sit 30-60 ms
  back, where a 64x32 depth image has barely changed (~90-180 u of travel
  at full speed, much less during early scratch training).
* sF1 was FROM SCRATCH, one seed, 1.5e9 steps, and died on the 25k shelf.
  The ledger itself flags shelf negatives as "partially confounded with
  seed luck", and the rounds 20-21 retraction later established that
  1-hour-scale scratch verdicts on this map are near-binary gate draws.
  The verdict recorded was "lean-negative", which the backlog table
  hardened into "dead" by aggregation.
* Action history has NEVER been tested. `--frame-stack` stacks images only;
  there is no past-action code anywhere in the repo (grepped 2026-08-29).
* The litsurvey citation used to close the topic (Vasco et al., GT stack=1,
  "temporal info belongs in proprioception") argues FOR the proprioceptive
  half of this arm: past actions are pure proprioception.

So the question this arm asks was not answered: history that actually spans
the timescale of the known failure, plus the agent's own recent control
trace, on the standing warm-resume protocol where a single run is readable.

## 2. Hypothesis

The stuck checkpoint's failure is a control-precision event, not map-wide
exploration: it tracks the champion line within 1-2 u for 88% of the map,
then departs the ramp between route vertices 1596 and 1598 after a ~0.45 s
precursor of small, growing error, entering 6% slow. A single depth frame
plus velocity scalars is first-order state only. What it cannot express:

* the TREND - is the error growing or shrinking over the last half second
  (acceleration/derivative information over exactly the precursor window);
* the agent's own recent control - which corrections it already applied,
  so it can stop re-applying them (oscillation damping).

H1: depth history at lags 150/300/450 ms (not 30/60/120) supplies the trend.
H2: the last 15 actions (450 ms) supply the control trace.
One combined arm. This is a mechanism probe, not a factorial; if it moves,
decomposition is a follow-up.

## 3. The arm

* Warm resume of `C:\RL_Surf\runs\sOBSR2\ckpt_latest.pt`
  (md5 `1ba1fd2936af3ae1ad3608e3cd6b1e9e`, step 3,782,737,920) - the
  standing protocol. Control = xCTL, the local-5090 column in CLAUDE.md
  (band 137,972-174,159 over +751M). Same card, so comparable.
* Config: everything restored from the checkpoint, UNCHANGED (act_every 3,
  respawn-margin 10, --obs-reward on - including its known truncation
  bootstrap bug, which stays; do not fix it). The ONLY treatment:
  * `--frame-stack 4` with new `--stack-strides 5,10,15`: offsets
    (0, 5, 10, 15) decisions = 0/150/300/450 ms at act_every 3. 4 conv
    input channels, ring length 16.
  * `--act-hist 15`: the last 15 decisions' actions, newest first, dense
    (proprioception wants the fine-grained trace; no striding here).
* Budget: `--steps 4.6e9` (about +820M over the resume point, ~1 h on the
  5090 at 3090-like rates), plus a hard local deadline kill at 75 min of
  wall time. Eval cadence comes from the checkpoint (rows every ~150M,
  matching xCTL's rows at +150/300/450/600/751M).

### Action encoding

`NVEC = (15, 7, 3, 3, 2, 2)` (yaw, pitch, fwd, side, jump, duck). Encode
each past action as 6 floats, per-dim normalized `(a - c) / c` with
`c = (n-1)/2` - all dims are ordinal or binary, so this is faithful and
compact. 15 decisions x 6 = 90 scalars. Before the episode's first decision
the history is filled with `NEUTRAL_ACT` (the defined hold-everything
action); the history resets on every episode boundary exactly like the
frame ring (age-clamp semantics: pre-start slots read as neutral, early
slots clamp to the oldest real entry - mirror what the frame machinery
does).

### Where the new inputs enter the network - the one real trap

The route block precedent (`Policy.forward_split`) feeds `scal[:,
N_SCALAR:]` to the CRITIC always and to the actor only when not
`--route-critic-only`. Action history MUST reach the ACTOR - that is the
entire point. Layout: `[15 core | 90 act-hist | img]`, image stays the
trailing contiguous slice (the channels_last restride depends on it), block
visible to BOTH pi and vf towers. Do not add core scalar slots
(`Policy.feat_idx` is sorted; a middle insert scrambles the row - see the
race-latch comment near train_fast.py:2323).

### Surgery: start ON the baseline curve

`tools/inflate_ckpt_memory.py` writes a seed checkpoint
(`C:\RL_Surf\runs\research\xMEM\ckpt_seed.pt`) from the sOBSR2 ckpt:

* `conv.0.weight` (16,1,5,5) -> (16,4,5,5): original weights land in
  CHANNEL 0 - the stack is newest-first, channel 0 is the current frame -
  new channels 1..3 zero.
* pi and vf first Linear: +90 trailing input columns, zero
  (`widen_for_route` is the exact template, including Adam
  exp_avg/exp_avg_sq padding with zeros; conv moments need the same
  treatment on the 4-D weight, which widen_for_route's 2-D pad does not
  cover - extend, don't shoehorn).
* ckpt cfg gains `frame_stack=4`, `stack_strides=(5,10,15)`, `act_hist=15`
  so a plain `launch_local.ps1 resume` restores the whole treatment and the
  existing frame_stack mismatch guard (train_fast.py ~2069-2080 on mmddp)
  stays coherent; add the same restore+mismatch guard for the two new keys.
* Refuse chunked checkpoints (code_head present) and in_ch != 1 - assert,
  don't handle.

Zero-init means the seed computes bit-for-bit the baseline function at
step 0; every later divergence is the treatment, not an init shock.

## 4. Implementation notes

What already exists on `mmddp` and must be reused, not rebuilt: the
per-env frame ring outside the CUDA graph, `stack_from_ring` /
`stack_from_buffer` (update side is a GATHER over the existing rollout
buffer plus prologue rows, NOT a bigger buffer), age-clamped gathers,
episode-boundary handling, eval-side `_TorchPolicyBase._push_frame`, cfg
restore + mismatch guard, `check_vision_exclusive` (frame-stack is
exclusive with --surf-mask/--pinhole - baseline uses neither), and the
default-off parity guarantees.

New work:

1. `--stack-strides` CSV flag. Default None = legacy (1,2,4,8) so every
   existing path stays byte-identical. It must reach BOTH the trainer and
   the eval helper (`STACK_STRIDES` is a module global today - make the
   override flow to every consumer; a test must pin train/eval parity).
   Prologue row count and ring length must be DERIVED from max(offsets)
   (goes 8 -> 15), not hardcoded.
2. `--act-hist M`: rollout-side per-env action ring (pushed at decision
   time with the sampled action), update-side gather from the stored
   action buffer + action prologue rows (mirror the image pair), eval-side
   ring in `_TorchPolicyBase` (pushed when a decision is made, not per
   held tick), obs_dim arithmetic (train_fast.py ~2342 region), Policy
   block wiring per section 3.
3. The surgery tool.
4. Tests: stride override correctness incl. offsets/ring/prologue sizing;
   act-hist ring-vs-buffer parity; reset/neutral semantics after forced
   respawn; train-vs-eval assembly parity; default-path (flags off)
   byte-identity with HEAD~ behavior; existing frame-stack test files stay
   green.

## 5. Validation gates - ALL before the 1 h run, all local

1. Full relevant test subset green (frame-stack tests + new tests), CPU.
2. Bit-identity ON THE 5090: seed ckpt + flags vs original ckpt + no
   flags, greedy, >= 300 decisions from identical env states: logits and
   values byte-equal. This is the arm's licence to claim it starts on the
   baseline curve. (Both sides run mmddp code, so mmddp-vs-cpubench drift
   does not enter.)
3. Frame-delta evidence for the ledger: from one greedy episode's frames,
   median per-pixel |delta| at lags 1, 2, 4, 8 vs 5, 10, 15 decisions.
   This retro-quantifies how degenerate sF1's 30-60 ms lags were and
   documents that 150+ ms lags actually differ. Collect during gate 2's
   rollout - no extra GPU session.
4. Smoke train ~3 min on the seed: no goal-field bake line in the first
   minute (absolute map path), finite losses, sane approx_kl, fps within
   ~15% of what the same smoke gives with flags off (measure both, report
   the ratio - the throughput cost IS a result).

## 6. Metrics and decision rules

Primary, per CLAUDE.md section 3: `tools/eval_honesty.py --route
C:/RL_Surf/maps/surf_src_cannonball.route.npz runs/xMEM/traj_*.jsonl
--order-only 16` -> corridor MAX, crossings past 205,440 u, finishes.
Secondary: `race/eval_progress` rows against xCTL's same-card column ONLY
(0 / +150M / +300M / +450M / +600M / +751M), remembering it saturates at
191,812 on-route and is flattered by death dives (check end z vs the
finish box); training ep_len_mean before interpreting eval lengths; win
rate only paired with reservoir min-depth; fps cost.

Reading:

* POSITIVE: any crossing past 205,440 u or any finish. No geodesic-shaping
  arm at the default margin 10 has ever crossed 205,440 (0/333 in round
  18's controls); crossings have only ever come from margin 2 or the arc
  reward, and this arm changes neither. Also positive: sustained eval band
  above ~195k with episodes ending on-route.
* NULL: band inside xCTL's, corridor MAX <= ~205k, 0 finishes. Report as
  null WITH the caveat that 1 h may undertrain zero-init inputs; if the
  late rows trend up, say so - a longer arm is then a fair follow-up ask.
* NEGATIVE: a decaying series vs xCTL (the xEZ pattern).
* "The curves have not separated" is a legitimate result. Do not squint.

Ledger: append a section (Round 28, xMEM) to `docs/research-results.md` on
the arm branch - append-only, never edit others' sections. Include: the
sF1 stride correction (what was actually run in rounds 10-14), the
frame-delta table, bit-identity confirmation, fps ratio, the honesty
numbers, and the verdict.

## 7. Ops (local run, worktree)

* Work in a git worktree; branch `memarm` off `mmddp`. The launcher's map
  path is already absolute (`C:\RL_Surf\maps\surf_src_cannonball.bsp`), so
  the worktree bake trap should not fire - still watch the first minute of
  the log for any bake line and abort if one appears.
* Launch ONLY via the branch's `tools/launch_local.ps1`:
  `resume C:\RL_Surf\runs\research\xMEM\ckpt_seed.pt xMEM --steps 4.6e9`
  (port the resume preset if mmddp's copy lacks it - a small edit to the
  launcher is the sanctioned form of an arm).
* Per the user (2026-08-29): do NOT babysit. One check ~1 min after the
  launcher's own liveness proof (log tail: no bake, no traceback, fps
  line), then a single background waiter until the trainer exits -
  deadline-kill the exact trainer PID at 75 min if still alive. No
  polling in between. Stationarity is judged post-hoc from progress.csv.
* No vast rentals, no fleet_watchdog, no GPU work besides this run (the
  box is the user's desktop; games may be running - never kill any
  process except the trainer PID captured at launch).
* Afterwards copy `progress.csv`, `traj_*.jsonl`, the launch log and
  `ckpt_latest.pt` to `C:\RL_Surf\runs\research\xMEM\`; commit code +
  this doc + ledger on `memarm`; never push, never commit on cpubench or
  main; leave the main checkout's working tree untouched.
