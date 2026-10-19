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

## Part B - xMEMS / xCTLS: the same treatment FROM SCRATCH (2026-08-29)

Requested by the user after Part A's null. Part A resumed a 3.78e9-step
checkpoint whose memory inputs were zero-initialised by surgery - a policy
that never needed history had to grow into it in 817M steps. Part B asks
the other question: does memory change how the task is LEARNED, with the
new inputs normally initialised and used from step 0. It also re-litigates
sF1 directly - sF1 was scratch - under corrected strides plus action
history.

### Protocol

* Baseline = the user's from-scratch ablation config, i.e. the
  `scratch_ablate` preset of `tools/launch_local.ps1` (cannonball, 64x32,
  NO --obs-reward, act_every 4, n-steps 128, epochs 4, minibatches 16,
  respawn-margin 10, --steps 3e9, complete argument set). Port the preset
  from cpubench's launcher if mmddp's copy lacks it; change nothing in it.
* TWO runs, serial, local 5090, same binary, same branch:
  * `xCTLS` - the preset, untouched. The control is not optional: no
    documented scratch control exists on this card or this binary.
  * `xMEMS` - the preset + `--frame-stack 4 --stack-strides 5,10,15
    --act-hist 15`. NO surgery, no seed ckpt: scratch init gives the new
    conv channels and the 90 act-hist columns the same orthogonal init as
    everything else.
* Strides stay defined in DECISIONS. At act_every 4 that is 200/400/600 ms
  - a longer game-time window than Part A's 450 ms. Deliberate: decisions
  are the network's native timebase, and from scratch there is no specific
  0.45 s precursor to target. Say so in the ledger.
* Control first, then treatment. 80-min deadline kill per run (the steps
  budget should land ~60-70 min; the kill is a backstop). Same no-poll
  protocol per run: the launcher's liveness proof, one log check at ~1
  min, then a single background waiter.
* A launch-window crash (first ~2 min, e.g. a preset/flag mismatch on
  mmddp) may be fixed and relaunched once; anything after meaningful
  training has begun is a result - stop and report, never re-roll a seed.

### Gates before either 1 h run

1. Scratch-mode smoke of BOTH configs, a few minutes each: model builds
   with in_ch=4 and the widened towers from scratch, the three cfg keys
   persist into the new run's own checkpoint, no bake line, finite losses.
2. fps ratio at the scratch config (act_every 4, T=128), reported.

### Reading rules - pre-registered, per the RETRACTION section

The end-of-run number of a 1 h scratch arm is nearly binary (gate ladder:
end z ~5,300-6,700 -> ~17k u, ~3,200-5,300 -> ~26k, below 2,000 -> ~48k)
and the seed-noise floor is 27% at 750M. Therefore report, in this order:

1. WHICH gate each run cleared and the STEP at which it cleared it
   (time-to-event, from the eval trajs via the ordered corridor scorer).
2. The matched-step point at 525M (identical runs historically agree to
   1.2% there) - the most sensitive single comparison.
3. End-of-run corridor MAX with the 27% floor stated. A gap inside the
   floor is NOT an effect. MEAN-tracks-MAX is not corroboration.
4. Pace vs sF1's pattern: sF1 sat BELOW base pace on the 25k shelf. xMEMS
   merely matching xCTLS's pace already contradicts "stacking hurts" under
   honest strides; below-pace echoes sF1.
5. ep_len_mean before interpreting any eval episode length; win rate only
   paired with reservoir min-depth; fps cost.

Callable outcomes at n=1+1: a POSITIVE needs the treatment ahead at the
525M matched point AND end-of-run outside the 27% floor AND an earlier or
higher gate; a NEGATIVE needs the mirror image. Everything else is "not
separable at one seed" and must be written up as exactly that.

Ledger: Round 28 addendum on `memarm`. Artifacts to
`C:\RL_Surf\runs\research\xMEMS\` and `...\xCTLS\`.

## Part C - xSEQ10: direct sequence head, no codebook (2026-08-29)

User request: predict H=10 actions per deliberation DIRECTLY - no codebook,
no decoder, just H x sum(NVEC) logits - executed with the standard
act_every repeat (predicted [0, 2] at act_every 3 runs as
[0, 0, 0, 2, 2, 2]). Those execution semantics are exactly what the
`--chunk` machinery already implements (docs/action-chunks-design.md) with
a K=64 codebook + learnable decoder, and per the user that variant
"overall was training bad". Part C swaps the parameterization and changes
NOTHING else, so whatever difference appears is the head, not the chunking.

### Mechanism

* Mode selection: `--chunk 10 --codes 0` = direct mode. `codes > 0` stays
  codebook mode and `chunk 0` stays flat mode, both byte-identical to
  today - guarded by tests.
* Policy: `seq_head = Linear(hidden, chunk * sum(NVEC))`, orthogonal init
  0.01 like action_head, no decoder. Reshape to (B, H, 32); each h-slice
  goes through the same HeadPacker pad/sample path as the flat head.
  Joint log-prob = sum over the H steps of the 6-dim factored log-prob;
  entropy summed the same way. MATCH whatever normalization of
  entropy/ent-coef the codebook mode uses (summed over H heads vs
  averaged) and write it down - a 10x entropy bonus by accident would be
  a second treatment.
* Inherited verbatim from chunk mode: one PPO sample per deliberation,
  one V(s) per deliberation, GAE/gamma handling, NEUTRAL_ACT tail masking
  when an episode ends mid-chunk (design doc 4.3), CUDA-graph static
  shapes, engine action stream at 100/act_every Hz.
* Note the representational trade: the codebook is a mixture over 64
  learned CORRELATED sequences; the direct head is fully factorized -
  each of the 10 steps independent given s. That is the simplest possible
  version, which is the point.

### Comparisons

1. PRIMARY control: xCTLS (flat, scratch_ablate, act_every 4, this card,
   this binary, 1.69e9 steps of curve - already on disk). Arm config =
   scratch_ablate + the chunk flags, with the rollout matched in GAME
   TIME to the control: T = 13 deliberations (13 x 10 = 130 decisions vs
   flat 128; 520 vs 512 ticks), all non-chunk flags identical (act_every
   4, lr, ent, epochs 4, minibatches 16, margin 10, NO --obs-reward,
   record-every 75e6 so eval marks align). If docs/action-chunks-design.md
   or the historical codebook config contradicts any of this for a
   stability reason, follow the design doc and record the deviation.
2. SECONDARY, context not verdict: FIRST locate the historical codebook
   run(s) - ledger + runs/ - and pin down config, card, binary vintage,
   and what "training bad" was numerically. Compare qualitatively, same
   card only.
3. Report BOTH axes: steps AND wall-clock. Chunking's promise is H-fold
   fewer trunk forwards and lidar renders, so fps and gate-vs-wallclock
   matter as much as gate-vs-steps. "Trains at pace with 10x fewer
   deliberations" would itself be the headline result if the level
   matches, given the codebook's history.

### Gates before the run

1. Unit tests: mode guards (flat and codebook paths untouched), direct
   log-prob/entropy vs a hand-computed small case, tail masking in direct
   mode, shapes through HeadPacker.
2. CUDA graph capture + compile with the new head.
3. ~3 min smoke: finite losses, sane kl, no bake line, and the entropy
   MAGNITUDE reported next to a flat smoke (the summed-over-10 scale
   check).
4. fps ratio vs flat, evals off, back-to-back, minimum per-iteration
   time (the Part B measurement lesson).

### Run and reading

* One run, `xSEQ10`, `--steps 3e9`, 80-min deadline kill - the same
  treatment xCTLS received. Standard no-poll protocol (one ~1 min check,
  single waiter).
* Pre-registered reading = Part B's rules (gate + step-at-gate, matched
  marks, 27% floor, MEAN-tracks-MAX is not corroboration, ep_len_mean
  check, win rate only with reservoir min-depth) PLUS the wall-clock
  versions of the gate timings.
* Callable outcomes at n=1+1 as in Part B; anything between is "not
  separable at one seed", written up as exactly that.
* Ledger: Round 29, on branch `seqhead` (created off `memarm` so the
  launcher fixes ride along; memory flags stay OFF - default - in both
  arm and control). Artifacts to `C:\RL_Surf\runs\research\xSEQ10\`.

## Part D - xSEQ10PS: the per-step trust region fix (2026-08-29)

User-ordered follow-up to Part C's callable negative. Part C's diagnosis:
the PPO ratio covered the 10-step JOINT log-prob, so updates moved the
policy ~H times per gradient step against a clip calibrated for one
decision, and - worse - a joint ratio cannot see one step's distribution
going wild while another compensates. Result: kl 8x the control, entropy
collapse to near-deterministic, flatline at 14,900. The fix targets
exactly that mechanism and changes NOTHING else.

### Mechanism

* New `--seq-ratio {joint,per-step}`, default `joint` = Part C behavior,
  byte-identical (guarded). The arm runs `per-step`.
* Per-step clipped surrogate: the chunk's shared advantage A is applied
  to each of the H per-step ratios independently -
  `sum_h min(r_h * A, clip(r_h, 1 +/- 0.2) * A)` - so every step gets its
  own trust region and none can hide behind another. The intended
  semantics: FLAT PPO over the 130 expanded decisions, with the advantage
  shared within each chunk of 10 and the trunk forward shared too. Scale
  the loss so the gradient per ENV STEP matches flat PPO's (13 rows x 10
  terms vs the control's 128 rows - the sum form does this naturally;
  verify, do not assume).
* INVARIANCE GUARD, which is also the correctness test: at `--chunk 1`,
  per-step mode must be BIT-IDENTICAL to flat PPO. Pin it in a test.
* Buffer stores per-step log-probs (T, N, H) instead of only the sum.
  Report `approx_kl` per-step (comparable to flat) and the joint kl as a
  reference column.
* Everything else IDENTICAL to xSEQ10: same head, orthogonal 0.01, meaned
  entropy with --ent 0.005, T=13 deliberations, act_every 4,
  scratch_ablate flags, --steps 3e9, 80-min deadline. One variable.
* Codebook mode untouched, byte-identical.

### Comparisons and pre-registered reading

Three-way at matched marks: xCTLS (flat control), xSEQ10 (joint-ratio,
the failure), xSEQ10PS. Two separable questions, in order:

1. DID THE FIX WORK MECHANICALLY: per-step kl lands near flat's band
   (~0.02-0.03), entropy declines gracefully along xCTLS's shape (-8.2
   toward -4.7) instead of collapsing toward 0, no 14,900-style flatline.
2. DOES CHUNKING THEN COMPETE: corridor at matched marks vs xCTLS (27%
   floor rules), gates on BOTH axes - with ~2x fps retained (re-measure),
   per-step parity within ~2x would already mean a wall-clock WIN.

An arm that fixes (1) but still loses (2) badly is itself decisive: it
says the blocker is the 400 ms open-loop commitment, not the optimizer -
and then no codebook fix rescues H=10 either; the lever becomes H. Say
that explicitly in the ledger if it lands there.

Ledger: Round 29 addendum on `seqhead`. Artifacts to
`C:\RL_Surf\runs\research\xSEQ10PS\`. The codebook gets NO run now; note
in the ledger that the same decomposed-clip fix (code ratio clipped as
one categorical + per-step decoder ratios) is its candidate fix, pending
this arm's outcome.

### Part D2 - the spec-compliant rerun (xSEQ10PS2, 2026-08-29 evening)

The first Part D run was CONFOUNDED: the per-step loss summed over H then
meaned over ROWS, so every decision carried 9.85x flat's policy-loss
weight while value and entropy stayed per-deliberation (effectively
--vf 0.05, --ent 0.0005). The audit also retracted Part D's premise: the
"kl 8x the control" line compared a JOINT (sum-of-10) kl to flat's
per-decision kl; like for like, xSEQ10's per-decision kl was 0.0153 -
INSIDE flat's band. The joint form was naturally scale-matched per env
step; only the botched per-step form was not.

xSEQ10PS2 = xSEQ10PS with the one corrected line (mean over TERMS,
`(x*m).sum()/m.sum()`, handles the masked tail, reduces to flat at H=1 -
now pinned by the corrected invariance test). Everything else identical.

REFRAMED reading, since the trust region was never the proven villain:
* If xSEQ10PS2 lands at xSEQ10's level with healthy diagnostics, two
  independent healthy-optimizer parameterizations agree, and the
  commitment-length/shared-credit conclusion (and the H=10 codebook
  closure) becomes available - properly this time.
* If it clearly beats xSEQ10 (outside the floor), the clipping
  granularity and/or the restored entropy weighting mattered, and H=10
  is not dead yet.
* The entropy-starvation hint (the 10x-weaker-entropy run collapsed
  further and plateaued lower) gets its check for free: D2 restores the
  intended ent weight. Report entropy trajectories side by side.
Four-way tables now: xCTLS / xSEQ10 / xSEQ10PS (confounded, reported not
verdict-bearing) / xSEQ10PS2. Ledger: Round 29 addendum, and this doc
note synced. Artifacts to `C:\RL_Surf\runs\research\xSEQ10PS2\`.

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
