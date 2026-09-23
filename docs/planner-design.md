# The planner: why it exists, what it must do, and the proposed redesign

Living design note, started 2026-09-24 from the user's requirements after the
new mazes (ledger 2026-09-23 22:30). Sections 1-3 are the user's and are
settled; section 4 is a PROPOSAL until the user agrees.

## 1. Mission (user, 2026-09-24)

**The planner exists for macro exploration.** On maps where the potential
field does not describe the correct path, the surf agent does not finish,
because it cannot explore: the policy acts 25 times a second, and nothing
in it can make a long-term commitment to one side, go there, and find out
that something new is there. The labyrinth benchmark exposed this even at
its mild rungs.

The split into a planner (macro) and an executor (micro, the policy) exists
to make exploration possible and convenient at the macro level, and to get
an agent that does not know where to go unstuck. The planner decides an
order of magnitude less often (once every 1 to a few seconds), which lets it
commit to a direction and look ahead by seconds. Everything else about it
serves that goal.

Evidence so far: 34 flat PPO cells never left the corridor (2026-09-21);
with the planner, labyrinth 100/200, easy01 (+25M) and medium01 (+100M) are
solved; hard01 is not (0 of 32,768 training episodes ever finished; the
planner is trapped in the region nearest the finish in a straight line).

## 2. Requirements (user, 2026-09-24)

* **Generic**: one planner for the labyrinths and for surf, same constants
  on every map (CLAUDE.md 0b).
* **No top-down view**: surf maps can be multi-storey; a bird's-eye image is
  ill-defined there.
* **No fixed context window**: the hard maze showed what a 2,048 u window
  costs (a branch left behind looks unexplored again).
* **No predefined trajectory set** in the long run: a fixed vocabulary of
  800 u shapes crosses walls 60-97% of the time in 160-200 u corridors.
* **Velocity is part of the state** and must be part of the plan: on surf,
  what is reachable depends on speed.

## 3. The user's formulation (2026-09-24)

* A trajectory is a **sequence of states**: positions in the maze, position
  plus velocity on surf.
* The planner is a distribution Q(tau | s; theta) over trajectories from the
  current state, concentrated on trajectories that lead toward the goal and
  that the executor can do (or nearly - a moving target, so the executor
  keeps getting slightly harder useful plans).
* Only SAMPLING is needed, not densities (implicit models are fine), with a
  temperature raised when the agent is stuck.
* Open questions he named: how long a trajectory is; whether it must reach
  the goal or only start in a good direction; whether its probability should
  reflect the CURRENT executor's ability.

## 4. Proposal (assistant, 2026-09-24) - not decided

**Factor the distribution.**

    Q(tau | s)  proportional to  P_can(tau | s) * exp( U(end of tau) / T )

* **P_can** - can the current executor do tau from s. Learned by supervised
  learning from outcomes: every executed plan is a labelled example (the
  strict judge already measures it), and every trajectory the executor
  actually produced is a valid, physically possible example for free. No
  physics model: the simulator already enforced physics on everything the
  executor did. This answers the third open question: yes, the current
  executor. The "slightly harder" moving target comes from the tails
  (sampling at temperature, noisy rollouts): stretches that work get
  weighted up, the executor trains on them, P_can widens.
* **U** - how useful the END of the segment is. Before the finish has ever
  been reached, the only honest signal is NOVELTY against a GLOBAL archive
  of the states reached so far (3-D position cell + a coarse velocity bin -
  coarse on purpose: 3,072 keys per cell farmed in place, CLAUDE.md 4).
  After the first finish, a learned time-to-goal value. Never the
  straight-line distance: that is what trapped hard01.
* **T** - the user's temperature, raised automatically when the archive
  stops growing.
* This is the known optimum of KL-regularised RL ("control as inference",
  Levine 2018): the best distribution that stays close to what you can do is
  that prior re-weighted by exp(return / T). Training theta is weighted
  maximum likelihood on the executor's own rollouts (reward/advantage-
  weighted regression) - supervised, no adversarial game. Sampling is draw
  K candidates, re-weight, pick (what MPPI does) - no MCMC.

**Answers to the other open questions.**
* Length: a SEGMENT of a few seconds (the commitment horizon), plus the
  score of where it ends; the score (novelty or value) carries the rest of
  the route. Infinite trajectories are neither needed nor learnable.
* Memory lives in the SCORE (the global archive, queried at decision time),
  not in the network's image - that also removes the top-down view.

**v1, concretely.**
* Candidates: segments of the executor's OWN past motion (a few seconds of
  states with velocity, in its own frame), clustered - the vocabulary is
  learned from its motion; on surf it will hold real flight arcs with
  speeds.
* Network: from the executor's egocentric observation (depth render,
  velocity, finish direction), predict P_can per candidate (and the value).
* Choice: P_can x exp(U / T). No PPO on the planner.
* Executor: also reads the plan's velocities (the fix for "flies past the
  goal and turns back"); trained on progress along the plan, with its own
  segments as plans (hindsight), as today.
* First test: labyrinth_hard01, the case that fails today. Then edgeflow,
  then unitfarmer2.

**Decision still open**: whether the planner may also search with forked
simulators (exact, costs compute); v1 does not need it.
