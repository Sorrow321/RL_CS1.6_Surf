# GPT cross-review: exploration research and code

Reviewed branch `petrusnight` at commit `0fa748c`.

## Bottom line

The work has found real unitfarmer2 signal, but it has not yet established one champion-free mechanism across all four maps. The biggest surprise is that cannonball's claimed clean lineage is not clean under the project's no-demo rule.

## Highest-impact findings

### 1. [P0] Cannonball must be disqualified as champion-free evidence

`cannonball_pregate.npy` was cut from `exitABS` round 9, not from the clean failing policy (`docs/research-results.md:20653`). I verified all 163 rows are byte-for-byte a subsequence of `cannonball_finisher_states.npz`. `exitABS` descends from BC on champion rows and a champion spine (`docs/research-results.md:16082`, `docs/research-results.md:16189`), and `gbCAN*` actually trains with this file as `demo_file` (`runs/gbCANfin3/run.json:196`).

That directly violates the "anything derived from a human demo never enters training" rule (`AGENTS.md:8`). The later claim that cannonball was clean (`docs/research-results.md:21541`) is therefore wrong. The mechanism may still work, but cannonball needs a clean rerun using the failing policy's own states.

### 2. [P0] The no-demo launcher guard is bypassable

The shell checks only exact spellings (`tools/run_arm.sh:96`), while argparse permits abbreviations by default (`python/train_fast.py:3561`); for example, `--demo-f` and `--bc-f` evade the shell guard. `--goal-route` and trajectory-fitted `--codebook` are not guarded at all (`python/train_fast.py:4368`, `python/train_fast.py:4054`). Worse, checkpoints can restore demo/route settings after the shell check (`python/train_fast.py:5561`, `python/train_fast.py:5773`), while `ARM_RESUME=1` skips the configuration gate.

Use `allow_abbrev=False` and enforce provenance inside Python. `SELF_STATES=1` is not enough: artifacts and checkpoints need a transitive `human_derived` manifest/hash.

### 3. [P1] Training and evaluation use different spawn-yaw distributions

Training defaults to 8-degree jitter (`python/train_fast.py:3690`); the recorder omits it (`tools/record_ckpt.py:739`) and therefore uses the core's 5-degree default (`python/surfgym/core.py:250`). The simulator also applies jitter to fixed spawn-pool states (`src/env.c:383`). `yaw_jitter` is absent from saved run configuration.

Therefore 6/8 and 2/8 are rates under a different distribution, and the `--yaw-jitter 180` arm was evaluated at 5 degrees. Entry existence still stands; robustness and rate comparisons do not.

### 4. [P1] "Start-anchored" progress is not actually start-anchored on UF2

`AliveReach` identifies start episodes only through `spawn_d >= threshold` (`python/train_fast.py:2866`). Because entering the pit raises geodesic distance, inside-pit reservoir states satisfy this test—the exact contamination already documented for demo-window states (`docs/research-results.md:21341`). Timeouts also include the unverified last three seconds despite the advertised survival hold (`python/train_fast.py:2898`).

Track explicit spawn source and reset identity; geodesic distance cannot identify the platform.

This also changes the interpretation of position-only novelty: it logged 547,529 autonomous pit-box visits and 91.8% nominal gate hits mid-run, then 0/16 final contacts (`runs/research/gate_bench/summary_uf2k.txt:28`). The 91.8% is contaminated by policy-own pit respawns, but the run did autonomously discover pit states. Its failure is at least partly retention/connection, not simply "position-only novelty cannot explore."

### 5. [P1] Several UF2 ablation conclusions are not supported by their configurations

- `uf2SURF` is described as "ratchet + surf reward," but `race_ratchet` is false (`runs/uf2SURF/run.json:159`). The meaningful ratchet+surf combination remains untested.
- The cited pit-return 26 versus slide-return 10 comparison was explicitly retracted as spawn-contaminated (`docs/research-results.md:21341`). It does not show that the platform's south action has higher return.
- The novelty anneal simultaneously removed temperature, extra entropy, count decay, and most intrinsic reward; `--unstuck` changes all of them (`python/train_fast.py:4720`). With KL 0.31, the result means "an abrupt multi-axis switch erased the behavior," not "race reward cannot retain it."
- Several reported `0/16` verdicts were actually 8 sampled + 4 greedy.
- The guide's cannonball comparison uses approximately 81 seconds, while the ledger's human record is 68.60 seconds.

### 6. [P1] The novelty primitive contains two likely performance traps

The key combines position, camera yaw and speed (`python/surfgym/rewards.py:991`), and "movement" means any combined-key change (`python/surfgym/rewards.py:1360`). Turning the camera or crossing a speed bin while remaining in the same spatial cell therefore earns novelty. That naturally supports scanning/camping rather than taking a route.

Additionally, every environment entering the same key on one tick receives reward from the same pre-increment count; duplicates are incremented only afterward (`python/surfgym/rewards.py:1366`). Reward magnitude therefore depends on fleet size and synchronization.

For curiosity-conditioned training specifically, the entire shaping side—including time cost—is multiplied by `1-T/Tmax` (`python/surfgym/rewards.py:1303`). A maximum-temperature explorer pays no clock penalty.

### 7. [P1] Reservoir behavior has several definite silent discontinuities

- Normal race success is used by reward, but only `gmask` is sent to reservoir harvesting (`python/train_fast.py:12097`). With no GoalSystem, true finishes lose the final `respawn_margin` instead of harvesting the whole successful chain.
- Binned fallback sampling can reintroduce sentinel/unreachable states after explicitly filtering them (`python/surfgym/respawn.py:519`).
- Checkpoints save reservoir rows but omit Go-Explore visit counts, Florensa competence state, backward-window state and RNG (`python/surfgym/respawn.py:732`). Continuations are therefore not behaviorally continuous.
- Frontier sampling promises to top up a starved shell/body quota but returns short instead (`python/surfgym/respawn.py:1218`).

### 8. [P1] Gate verdict tooling is not faithful to its declared semantics

`pass_hold` is enforced only when the trailer says `fail`; a late threshold crossing followed by truncation is counted as a pass (`tools/gate_bench.py:240`). Finish detection tests recorded points plus one extrapolated endpoint, unlike the engine's true segment/AABB sweep (`tools/gate_bench.py:205`, `src/env.c:723`). The speed rung is merely maximum speed anywhere inside a broad box and does not require ramp contact (`tools/gate_bench.py:262`).

The entry-bench wrapper also ignores subprocess failure and can load stale prior JSON (`tools/uf2_entry_bench.py:39`).

### 9. [P1/P2] Goal-field construction has two general correctness hazards

An unconverged field only prints a warning and is then cached as normal (`python/surfgym/goalfield.py:406`, `python/surfgym/goalfield.py:512`).

The 26-neighbour relaxation checks only destination occupancy, with no diagonal corner-clearance or player-hull clearance (`python/surfgym/goalfield.py:348`). In a synthetic 32-unit grid, it assigned a 45.25-unit path between two diagonally touching free cells even though both orthogonal routes were solid. A finite player cannot traverse that corner, so the field can contain silent geometric shortcuts beyond its already-known velocity/void problem.

### 10. Other concrete footguns

- A record-gate failure says the run was stopped but never kills the background trainer (`tools/run_arm.sh:265`).
- `--reward-per-decision` undercounts speed/surf/dive rewards and drops intermediate novelty transitions; current UF2 runs do not use this path.
- The curiosity-conditioned test suite is currently red: its `FakeCore` lacks the newly required `config`, so the advertised bit-identity test fails before comparing rewards (`tests/python/test_curiosity_cond.py:137`, `python/surfgym/rewards.py:1030`).

## New, simple mechanisms

### 1. Directed-transition novelty

Count `(previous position cell -> next position cell)`, and award novelty only when the physical position cell changes. Do not include gaze or speed-only changes. North and south exits from the platform become different directed edges, while turning in place and camping do not pay.

Make duplicate accounting batch-safe by assigning within-batch visit ranks, and do not periodically re-novelize already exhausted loops. This is the simplest new mechanism I would test first.

### 2. Survivor-gated predecessor archive

When a policy-own trajectory reaches a rare transition or new true-start alive frontier and survives another H seconds, protect the preceding 2–4 seconds of states/actions in a small separate archive. Sample that archive through a fixed quota.

This saves the causal pre-gate context, rather than merely oversampling the novel pit state. It automatically supplies the manual "find gate and cut pre-gate window" step without map boxes or human data.

### 3. Explicit high-T to T=0 self-distillation

Shared weights do not guarantee that the T=0 member inherits a high-T discovery. On survivor-qualified, policy-generated prefixes, add a small KL/self-imitation loss making the T=0 actor reproduce the successful high-T actions. This directly addresses the observed discover-then-forget behavior and remains fully champion-free.

A free diagnostic comes first: compare high-T and T=0 action distributions on identical platform states.

### 4. Selective long-return credit

Do not globally change PPO yet. For rare survivor-qualified prefixes in the current on-policy rollout, recompute their advantages with lambda approximately 1 or Monte Carlo while ordinary traffic remains at lambda 0.95. This differs from Fable's curiosity weighting: it repairs a potentially wrong-sign advantage instead of scaling it.

First extend `credit_diag` to policy-own UF2 entrants. The existing cannonball result demonstrates that a five-second sign inversion can occur, but the UF2 26-versus-10 evidence is invalid.

The resulting general loop is:

```text
true-start plateau
  -> directed-edge exploration
  -> survivor event
  -> protect causal prefix
  -> long-return credit/self-distill
  -> cool only after T=0 reproduces
```

## Recommended next order

1. Quarantine the `gbCAN*` result and contaminated window.
2. Fix provenance enforcement, yaw parity, explicit spawn-source tracking, and gate semantics before spending another run.
3. Re-evaluate existing UF2 checkpoints under their actual training jitter.
4. Run the corrected ratchet+surf ablation.
5. Perform the two free diagnostics: UF2 lambda/Monte-Carlo sign and high-T versus T=0 action KL.
6. Then test directed-transition novelty with the survivor-gated predecessor archive.

## Verification performed

- 61 targeted goal-field, reach and respawn tests passed.
- The curiosity-conditioned suite has the fixture regression described above.
- No training or rented hardware was launched during this review.
