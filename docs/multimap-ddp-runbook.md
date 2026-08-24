# Multi-map DDP: rent to training, end to end

The plan and design rationale are in `multimap-ddp-plan.md`. This file is the
operational sequence and the numbers you need to tell "working" from "broken".
It was written after a session that burned $3.50 and produced no training
because several of these were not written down.

Every step has a CHECK. If a check fails, fix it; do not proceed hoping.

---

## 1. Rent (target: usable ssh in 60 s)

**Filter on PHYSICAL cores per GPU, not threads.** vast's
`cpu_cores_effective` counts THREADS, and `ddp_launch.sh` sizes each rank at
`quota/(2*ranks)`, so the real filter is

    cpu_cores_effective / (2 * num_gpus) >= 8

For 4 GPUs that means **>= 64 effective cores**. Not academic: a box with 38
usable cores gives 4 ranks only 4 OMP threads each - fewer TOTAL threads than
2 ranks would get - and is the prime suspect for the deadlock in section 7.

Race 3-4 offers at once, keep the first to answer ssh, destroy the rest.

    vastai create instance <id> \
        --image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel --disk 60 --ssh --direct
    python tools/fleet_watchdog.py register <instance> --label <run> --minutes <n>

**CHECK before deploying anything:**

    nvidia-smi --query-gpu=index,power.limit,power.default_limit,utilization.gpu \
               --format=csv,noheader

* `power.limit` < `power.default_limit` -> capped card. One box ran at
  **180 W of 370 W, 43% of reference bf16**. Blacklist, destroy.
* non-zero `utilization.gpu` on a box you have not started -> **foreign
  tenant**. One box had all 4 GPUs pinned at 100% by a PID outside our
  namespace. Blacklist, destroy.
* ssh not up in 60 s -> blacklist, destroy. **No second chances.**
  `fleet_watchdog.py` enforces this via `READY_S`; the `ready` latch means a
  box that ever reached `running` is only killed by its deadline.

## 2. Deploy

    NO_CKPT=1 BRANCH=mmddp bash tools/deploy_box.sh <port> <host>

`NO_CKPT=1` because a multi-map arm trains from scratch - none of these maps
has a checkpoint, and the stuck checkpoint is a cannonball artifact.

**CHECK:** on the box, `git log --oneline -1` shows the branch you wanted and
`build/libsurfcore.so` exists.

## 3. Pool

The pool (107 maps + baked geodesic fields + SDFs + zones) is NOT in git. Its
location is in `runs/pool_source.txt`, which is gitignored - copy that to the
box yourself.

    bash tools/fetch_pool.sh      # download, unpack, RESTAMP, export viewer meshes

**Measure the rate before trusting it.** From a Sichuan box, Google Drive ran
at **221 KB/s = 36 minutes**. Under ~1 MB/s, push the local tarball instead
(`runs/surf_pool.tar.gz`, md5 `266f0855458cbfb2f4bb61fbaa4d55ea`).

**CHECK - not optional:**

    python3 tools/restamp_maps.py --check    # must say "all match"
    python3 tools/check_deps.py              # must exit 0

`tar` does not preserve sub-second mtimes, and every cache keys on
`v2_<size>_<mtime_ns>` of the `.bsp`. A freshly downloaded pool therefore
invalidates its own prebaked fields - measured at **103 of 108 maps**. The
trainer silently rebakes, minutes to hours per map, and a cache miss is
indistinguishable from a cold start in the log. `fetch_pool.sh` restamps
automatically; run the check anyway.

## 4. Arguments

Never hand-type the map list. `--goal-cell` must carry each map's GATED cell
(84 maps at 48, 21 at 32, 2 at 72) or every map misses its cache:

    python3 tools/pool_args.py                  # emits --maps ... --goal-cell ...
    python3 tools/pool_args.py --report         # what it would use
    python3 tools/pool_args.py --only-trigger   # the 42 strong-evidence maps

## 5. Launch

    NMAPS=107 ENVS=54784 RANKS=4 STEPS=500e9 EVAL_EPS=1 RECORD_EVERY=10e9 \
        RUN=<name> bash tools/launch_pool.sh

That is the mmSMOKE config verbatim plus `--act-every 4`, the pool args, and
fleet-scaled `--envs`/`--steps`. `ENVS` must divide by `RANKS * NMAPS`.

**CHECK the first minute:** a map that REBAKES prints a sweep count next to
`goal field:`. If you see sweeps, section 3's check was skipped.

## 5b. Tunnel the dashboard (MANDATORY - the user watches the plots locally)

A run nobody can see is a run nobody is watching. As soon as the launch is
in, before reading throughput:

    ssh -p <port> root@<host> "cd /root/RL_Surf && \
        (setsid nohup python3 tools/dashboard.py --port 8000 \
         > /root/dashboard.log 2>&1 &)"

    # keep this running on the WORKSTATION for the life of the box
    ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
        -L 8000:127.0.0.1:8000 -p <port> root@<host>

Open http://localhost:8000/ locally. The server binds 127.0.0.1 on the box
on purpose - the tunnel is the only exposure, so there is nothing to
firewall. **CHECK:** `curl http://localhost:8000/api/runs` from the
workstation lists the live run. The 3D map panel needs the viewer meshes
that `fetch_pool.sh` exported in step 3; if a map renders NOTHING, that
export failed for it (it is a per-map warning in the fetch log, not an
error).

## 6. What healthy looks like

One RTX 5090, 107 maps, `--envs 10700`:

| | |
|---|---|
| training | **~318,000 fps** marginal, 0 bakes |
| eval, all 107 maps | **~12 min** (8.8 maps/min) |
| slot cost | 1 -> 48 slots at fixed total envs: 914k -> 507k fps (**1.8x**) |

Map count is nearly free. If throughput looks catastrophic you are almost
certainly measuring during an eval, not measuring the trainer.

**`time/fps` in the CSV is CUMULATIVE from process start** - it includes
startup and every eval. For the real rate take two `^step ` lines and compute
`dS / d(S/fps)`. Reading the cumulative column as marginal is how a healthy
run got destroyed.

## 7. Two known problems

**The eval is batch-1.** Every eval core is built `num_envs=1`, so an eval is
N sequential single-env rollouts - ~1,500 launch-bound decisions each, plus a
JSON line per tick. Least parallel code in the trainer, run once per map.

* It fires on iteration 1 by default (`next_record = global_step`). Use
  **`--no-eval-at-start`**, or you pay it before a single gradient step and
  every early throughput reading is meaningless.
* Keep it RARE rather than smaller: at `--record-every 10e9` a 12-minute eval
  is ~2.3% of wall-clock, so there is no need to evaluate a subset and lose
  the per-map metrics `--maps` exists to produce.

The real fix, NOT yet done: evaluate inside the training fleet, which is
already batched - 6,000 ticks x 10,700 envs at 318k fps is **3.4 min for all
107 maps**, with ~100 episodes per map instead of 1.

**The "4-rank DDP deadlock" is SOLVED (2026-08-24) and it was never a
deadlock and never rank count.** It reproduced at 4 AND 2 ranks on a
CPU-adequate box (m16571, 10.7 phys cores/GPU, 13 OMP threads/rank, quota
detected), and PYTHONFAULTHANDLER=1 + SIGABRT stacks showed the ranks
ALIVE inside the rollout (`core.step` / `goalfield.sample` via
`mapfleet.reward`), crawling. Cause: `goalfield._FAST_SAMPLE` is
`@njit(parallel=True)` and numba sizes its pool off HOST cpu_count -
255 threads per rank on that fractional rental, ignoring the cgroup
quota and OMP_NUM_THREADS (294 observed threads/rank = 255 numba + 13
OMP + torch/NCCL). `mapfleet.reward` samples once per slot per decision:
107 syncs of a 255-thread pool per decision on ~256-point batches,
times N ranks spin-waiting on a 42% quota -> iteration 1 alone ran 10+
minutes at 0% GPU, which reads exactly like a hang. Single-map benches
(4/8 GPU), the 5-map smoke and the single-process 107-map run all keep
either the call count or the pool count small, which is why it hid.
`ddp_launch.sh` now exports `NUMBA_NUM_THREADS=$OMP_NUM_THREADS`
(bit-identity unaffected - prange elements are independent). With the
cap, the SAME box trains 4-rank at ~1.0M steps/s marginal, launch to
step lines in ~4 min, and the full 107-map eval-eps-1 eval took 40.4 s
at iteration 1 (untrained policy - trained-policy episodes run longer;
re-measure at the first 10e9 record). **Launch 4 ranks.**

## 8. Reading the metrics

* **`race/map_pct`** - for each map, the share of THAT map's own route the
  greedy eval covered, averaged over episodes, then over maps. A percentage
  of each map's own length, so short and long maps count equally.
* **`race/map_pct_trigger`** - same, restricted to the **42** maps whose
  finish is a real trigger curtain. The other 65 use inflated on-touch button
  boxes the simulator cannot press, so a null there is far weaker evidence.
  This is the honest number.
* **`race/maps_finished`** - fraction of maps where an eval episode's path
  actually crossed the finish AABB. The env's own win test.

**`map_pct` alone lies.** In a local run `shortbox` and `bucetation` both read
100.00% with `maps_finished` = 0: full geodesic progress, never entered the
finish box. Always report it with `maps_finished` beside it.

## 9. Map counts, so the number stops moving

| count | meaning |
|---|---|
| 620 | raw corpus |
| 615 | have start/end from some source |
| 238 | verified reachable |
| **107** | in the pool bundle: bsp + baked field + zones |
| **42** | of those, finish is a real trigger zone - strong evidence |
| 65 | finish is a button, substituted by an inflated on-touch box |

42 + 65 = 107. A box may show 109 `.bsp`: the pool's 107 plus `cannonball`
and `ski_2`, which ship in the repo and have no pool field.
