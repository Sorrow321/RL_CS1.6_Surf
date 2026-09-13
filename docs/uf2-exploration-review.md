# unitfarmer2's start: the exploration benchmark - evidence, ideas, and an honest estimate

Written 2026-09-13 18:05 for a cross-review by another model. Self-contained;
the numbers come from `docs/research-results.md` (entries dated 2026-09-13)
and are reproducible from `runs/research/gate_bench/summary_uf2*.txt`.

**The rule that frames everything (user, 2026-09-13):** human demos never enter
RL training - not as spawn states, windows, spines, BC data, route lines or warm
starts. A demo may be used to *measure* a policy or *analyse* a room. The map is
a benchmark for exploration recipes: the agent has to discover the line itself.

## 1. The problem

`surf_unitfarmer2`, from the true start. The geodesic goal potential (the
shaping reward, ratchet variant) says "go north": a slide down the start ramp
banks 2,777 u of potential in 6.3 s and then the agent dies. The human record
instead dives SOUTH off the start ramp into a pit - a **rise of +2,840 u of
potential** at t 3-5 s, speed 765 -> 1,765 u/s - surfs the pit's face, climbs
out at t 7-9 s, and is at depth 3,667 u by 12 s, 7,818 by 15 s, 13,474 by 20 s
(finish at 38.5 s). Analysis of the record's states (measurement only) fixed
three facts:

* **The first 2.5 s decide.** A policy that can surf the pit still turns north
  from the platform; spawned from the record's t >= 2.5 s states (a diagnostic
  bench, `tools/uf2_entry_bench.py`) the same policy takes the pit 3/4 from the
  entry slice, 4/4 from the pit slice, 4/4 from the exit slice.
* **The dive must be uncharged.** Without `--race-ratchet` the -9 of potential
  makes the policy steer out of the pit even from inside it and die at once.
  With the ratchet only time is paid.
* **Entrants without pit practice die within 2 s.** The pit is a fall unless the
  surf has been learned from states inside it.

**The pit-speed ladder (user, 2026-09-13 18:15, from watching uf2NOV1's video):**
`eval/speed_max` >= 1,400 u/s means the agent surfed the pit's FIRST ramp
down - that is where the record builds its 1,765 u/s - and without that
speed it cannot take the second ramp and fly back out. So the benchmark has
three rungs: pit contact -> in-pit speed >= 1,400 -> exit. The bench prints
the middle rung (`gate_bench.py score --speed-gate 1400`: max horizontal
speed while inside the pit box, per episode); batch 10's campers reach only
878-928 u/s inside the pit (they crawl, they never surf the ramp), which is
why they never leave.

Verdict metric: the **start-line bench** - 8 sampled + 8 greedy episodes from
the true start, 60 s; pit-box contact, and PASS = depth >= 5,600 u reached and
alive 3 s later (the record is there at ~13.5 s). Training-side pit counts are
NOT a verdict (window spawns inside the pit contaminated them once).

## 2. What has been tried (champion-free, 1B steps each unless noted, one seed)

| mechanism | greedy start line | pit contact from the start | note |
|---|---|---|---|
| base recipe (potential shaping, keys held, abs view), 4.4B | north slide, 2,777 u | 0 | stage 1 |
| ratchet (dive uncharged) | 2,777 | 2-5% of tempered training episodes; all die in 2 s | necessary, not sufficient |
| ratchet + keys temperature (plateau-driven sampling T <= 1) | 2,777 | tempered entries, all die | |
| ratchet + temperature on ALL heads | 1,736 | 0/16 in the bench | KL 0.056: yaw noise costs control |
| ratchet + frontier curriculum + `--speed-coef` | 2,756 | 0/16 | |
| ratchet + surf reward (ramp contact paid) | 2,811 | 0 | |
| time-penalty bias (larger per-tick penalty) | 910 | 0 | learned to die at 2 s (death is free) |
| ratchet + `--respawn-random` (uniform random reachable states, random yaw, 1-4k u/s) | 1,138 | 0/16 | learns the pit surf from random pit states, never the entry; view sigma blew up |
| ratchet + `--yaw-jitter 180` (uniform spawn heading) | 1,821 | 0/16 (8 visits in 1B) | spawn view yaw does not set the movement heading |
| **ratchet + count-based novelty 10x (`--int-coef 2.5`)** | 1,920 | **greedy 6/8 enter; 3/8 alive at the 60 s cap crawling in the pit; 0 exit** | the first champion-free entries |
| ratchet + novelty 10x + yaw jitter | 919 | sampled 1/8, greedy 0/8 | |
| ratchet + novelty 10x, then annealed to 0.25 at T = 0 (+600M) | 1,754 | 0/16 - the entry is gone (KL 0.31 when the bonus went) | the race reward alone does not hold the entry |
| **ratchet + novelty 4x (`--int-coef 1.0`)** | 1,725 | greedy 2/8 enter, **both surf the first ramp (in-pit speed 1,616 u/s) and die at the second** | rung 2 of 3, champion-free |
| ratchet + novelty 10x on position-only cells | 1,797 | 0/16 | the yaw/speed keys make the pit novel |
| in flight (18:00): novelty annealed to 0.25 at T = 0 from the 10x checkpoint; novelty 4x; position-only novelty; T-conditioned family at 10x (explorers up to 20x, T = 0 member pure race); Go-Explore spawn bursts; curiosity-cond at base; Go-Explore bin weights; keys T cap 2 | | | |

Demo-assisted arms (disqualified as results, kept as diagnostics): with the
record's approach states in the spawn pool the greedy start line reached
14,049 u (46% of the map), 8/8 through the pit at the record's pace, and a
second gate at 21 s appeared (a line-choice gate: the policy takes the inside
of a turn where the potential points, the record the outside ramps). Without
those states in the pool the pit line was lost within 300M steps twice
(with and without temperature): the entry is a fragile skill that needs
support in the spawn distribution.

## 3. What the evidence says

1. The pit surf and the exit are inside the policy class and are learned in
   under 1B from states inside the pit (shown three ways: demo states,
   random states, the novelty arm's campers).
2. The entry is a heading decision in the first 2.5 s that the race reward
   never pays for (the dive is a rise, the alternative banks +9 first), and
   the only mechanism that produced it from the true start is a novelty
   bonus large enough to dominate the race reward at the platform.
3. Once entered, the same bonus keeps the agent in the pit; the exit needs the
   bonus to fade or to be absent from the member whose line is evaluated.
4. Whatever finds the entry must also keep it: the champion-free analogue of
   the demo window is the policy's OWN reservoir (`--respawn-margin 1`
   harvests the second before death), which fills with pit states as soon as
   entrants survive a second. The novelty arm's campers do.

## 4. Candidate ideas, ranked by my estimate of return per box

1. **Novelty with an anneal** (in flight). 10x to find the entry, then the
   count-based bonus fades by construction (1/sqrt(visits)) if the count
   decay is off, or is switched to the base 0.25 in a T = 0 continuation.
   Cheapest; the risk is that the fade also erases the entry.
2. **The T-conditioned family at 10x** (in flight; user's idea). Explorers
   carry the novelty, the T = 0 member is pure race reward and is the greedy
   line; one network, so the entry found by the explorers is in the shared
   weights. Risk: the T = 0 member's own return still prefers the north slide
   at the platform unless the shared features carry the entry.
3. **Time-windowed or state-triggered curiosity** (user's idea, not built):
   T high for the first W seconds of an episode, or while the episode's
   potential is rising, then 0. Removes camping by construction. ~1 h to
   build (`--cc-window`, per-tick T; the observation column and the reward
   mix already follow the live T).
4. **Curiosity-weighted training** (user's idea, not built): sample as now,
   weight each episode's advantages by its summed novelty (TailRL-style
   groups exist as `--tail-weight`); the advantages themselves stay
   race-reward advantages, so entering-and-exiting is pushed up and camping
   pushed down. Half a day with a diversity filter.
5. **Novelty-weighted reservoir** (not built): Go-Explore proper - the
   respawn reservoir is the archive, cells keyed by position (+ heading),
   selection weight 1/sqrt(visits) so rare pit states are spawned from more
   often. The current `--respawn-mode goex` weights by DEPTH BIN, which
   cannot separate pit from slide (both are shallow).
6. **Two-stage recipe on own states**: any arm whose tempered episodes enter
   and survive > 1 s yields own pit states; then the celestial/cannonball
   stage 2 (window from the own line + keys T) and stage 3 (T = 0
   consolidation) apply unchanged. Needs 1-5 to produce the entrants.
7. Lower priority / evidence against: prediction-error curiosity (RND) was
   validated dead on cannonball; the surf-contact reward and speed reward
   were null here; heading diversity at spawn is null; larger temperatures
   cost control; Go-Explore spawn bursts and depth-bin weights are in
   flight with low expectations.

## 5. Chance of success - my estimate, with the caveats

* **The pit gate, champion-free, greedy start line through it:** 60-70% within
  the next 3-5 batches. The entry has been produced from the true start by a
  mechanism we control (novelty magnitude), the surf and exit are known to be
  learnable from inside, and the remaining problem (camping) is a
  reward-schedule problem with three untried, principled fixes (anneal,
  windowed T, conditioned family). What would falsify this: the annealed and
  conditioned arms both lose the entry when the bonus fades, i.e. the entry
  is never supported by the race reward alone even with the exit learned. If
  the return of the pit route (with exit) is above the north slide's +9 at
  the platform - and the demo-assisted diagnostics say it is, 26 vs 10 for
  the same policy - the entry should survive the fade.
* **The whole map champion-free:** well under 30% in this program's budget.
  Gate 2 at 21 s is a line-choice gate of the cannonball-wall kind and
  needs its own treatment; the record has 17 more seconds after it.
* **Caveats that apply to every number above:** one seed per arm (the
  documented gate-ladder noise, 2.7x between identical runs); the greedy
  line is fragile to spawn jitter even when learned (the trainer's jittered
  eval read 9,089 where the recorder's greedy was 4/4); 1B-step arms on
  this map are 25-35 minutes on a 5090, so "null at 1B" is weak evidence
  against slow mechanisms; and every start-line verdict here is 16
  episodes.

## 6. Questions for a reviewer

1. Is count-based novelty with an anneal the right primitive for a
   one-time heading decision, or is the windowed / conditioned T cleaner
   because it never pays for camping?
2. In the T-conditioned family, is there a reason to expect the T = 0
   member to inherit the entry from the explorers through shared weights,
   or does it need its own support (its spawns drawn from the explorers'
   pit states)?
3. Position-keyed Go-Explore over the own reservoir vs novelty in the
   reward: which fails less badly when the novel region is a death trap?
4. Any known result on "exploration for a decision that is only rewarded
   10 s later through a skill that must be learned first" - the pit is a
   two-skill gate (entry + surf) where the second gates the first's payoff.
