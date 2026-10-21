"""Trajectory recording — JSONL per docs/04-visualizer.md.

File layout (env 0 only, multi-episode per file):

    {"map": "surf_ski_2", "tick_ms": 10, "phys": {...}, "episode": 0}
    [t, x,y,z, vx,vy,vz, yaw, buttons, onground, progress, reward, pitch,
     fwd, side]                        # fwd/side: raw move bins 0/1/2
    ...
    {"end": "fail|done|trunc", "ticks": 4130, "best_progress": 8121.5}

``tick_ms`` is the core's MEAN tick (``SurfCore.tick_ms``); a ``--tick-ms``
core that cycles an integer pattern (7.63 -> 8,8,7) also writes
``"tick_pattern_ms": [8, 8, 7]`` and ``"tick_phase"`` (the pattern index
of the episode's first row), so ``surfgym.tick.episode_seconds`` times the
episode exactly. Consumers must read ``tick_ms`` from the header and never
assume 10 ms. The ``phys`` block's ``msec`` is the core's NOMINAL tick
(``SurfCore.nominal_msec``: 8 under that pattern) - a fixed number stated
once, not the pattern phase the core happened to be in when the file was
opened; ``tick_ms`` + ``tick_pattern_ms`` are the time base.

Tick-line semantics: line ``t`` holds env 0's state at the *start* of tick
``t`` (the pre-step snapshot from ``surf_get_states``) paired with the action
taken and the reward earned *during* tick ``t`` — a standard (s_t, a_t, r_t)
transition log.  The post-terminal position itself is unrecoverable: the core
autoresets in place the same step (surfcore.h), so after a done step
``get_states`` already shows the new episode.  ``buttons`` is the usercmd
bitmask reconstructed from the action: jump -> 2 (IN_JUMP), duck -> 4
(IN_DUCK).  ``onground`` is written as 0/1 (the core's field is -1 airborne /
solid index otherwise).

Episode-end labels: ``trunc`` when the core truncates (or the recording tick
budget runs out mid-episode); otherwise ``done`` vs ``fail`` is inferred from
the final reward — a completion-bonus-sized reward (+50) means the track was
finished.  The C API itself does not distinguish completion from failure in
its done flag.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .core import SURF_IN_DUCK, SURF_IN_JUMP, SurfCore, phys_to_dict
from .tick import header_fields

__all__ = ["record_rollout", "record_rollout_parallel"]

# Completion heuristic: the +50 completion bonus (the ABI-6 obs no longer
# carries a frac-remaining feature to probe).
_DONE_BONUS_MIN = 25.0


def _round(v: float, nd: int) -> float:
    return round(float(v), nd)


def record_rollout(
    core: SurfCore,
    policy: Any,
    path: Union[str, os.PathLike],
    episodes: Optional[int] = 1,
    max_ticks: Optional[int] = None,
    *,
    seed: Optional[int] = 0,
    on_tick: Optional[Callable[[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]] = None,
    episode_meta: Optional[Callable[[int], Dict[str, Any]]] = None,
    header_extra: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Roll ``policy`` on ``core`` and write env 0's trajectory as JSONL.

    ``episode_meta(episode_index) -> dict`` (optional) is merged into that
    episode's header line. Goal-conditioned recordings use it to carry the
    episode's goal and its reference line so the viewer can draw where the
    agent had to get (``goal: {center: [x, y, z], radius: r}`` and
    ``line: [[x, y, z], ...]`` are the keys viewer/app.js understands).
    Values must be JSON-serialisable (convert numpy arrays first).

    Parameters
    ----------
    core : the batch sim; all envs are stepped, env 0 is recorded.
    policy : object with ``act(obs) -> (N, 6) int32``.
    path : output ``.traj.jsonl`` file (parent dirs created).
    episodes : stop after this many env-0 episodes (``None`` = no episode
        limit; then ``max_ticks`` must be set).
    max_ticks : total tick budget across all episodes (``None`` = unlimited).
        If the budget ends mid-episode the trailer is written with
        ``end="trunc"``.
    seed : passed to ``core.reset``. ``None`` = do NOT reset — roll from the
        sim's current state (e.g. after ``core.set_state``); the policy's first
        ``act`` then receives a zero observation.
    on_tick : optional callback ``on_tick(t, states, rewards, done, trunc)``
        called once per tick with the pre-step state snapshot of *all* envs
        and the post-step reward/flag buffers (views — copy to keep).
    header_extra : optional dict merged into EVERY episode header, for
        conditions of the recording itself rather than of the episode
        (``episode_meta`` is the per-episode channel). ``--eval-stall`` uses
        it to record that the stall rule was armed, so a downstream honesty
        tool can tell a policy that crawled to the tick limit from one that
        was killed for crawling. Values must be JSON-serialisable.

    Returns a list of per-episode summaries: the trailer dict plus
    ``"ep_return"`` (sum of env-0 rewards).
    """
    if episodes is None and max_ticks is None:
        raise ValueError("record_rollout needs episodes and/or max_ticks")
    if episodes is not None and episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    map_name = Path(core.bsp_path).stem
    # the tick the core actually runs: SurfCore.tick_pattern is (10,) by
    # default and e.g. (8, 8, 7) under --tick-ms 7.63; a core without the
    # attribute (a stub in a test) is read off its config as before
    tick_pat = tuple(getattr(core, "tick_pattern", (int(core.config.phys.msec),)))
    # The "phys" block states the core's NOMINAL tick, not the phase it is
    # in: SurfCore.step mirrors each pattern element into config.phys.msec,
    # so snapshotting that wrote msec 8, 8, 7 into three recordings from one
    # 7.63 ms core. The realised sequence is tick_ms / tick_pattern_ms /
    # tick_phase below, which is what a consumer times an episode with
    # (surfgym.tick.episode_seconds). Identical at any fixed tick, so a
    # 10 ms recording is byte-for-byte what it always was.
    nominal_msec = int(getattr(core, "nominal_msec", core.config.phys.msec))
    phys = phys_to_dict(core.config.phys)
    phys["msec"] = nominal_msec          # in place: the key order is fixed
    header_base = {
        "map": map_name,
        "tick_ms": nominal_msec,
        "phys": phys,
    }
    header_base.update(header_fields(float(sum(tick_pat)) / len(tick_pat),
                                     tick_pat, 0))
    if header_extra:
        header_base.update(header_extra)

    summaries: List[Dict[str, Any]] = []
    if seed is not None:
        obs = core.reset(seed)
    else:
        obs = np.zeros((core.num_envs, core.obs_dim), dtype=np.float32)
    total_ticks = 0
    episode = 0

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        while (episodes is None or episode < episodes) and (
            max_ticks is None or total_ticks < max_ticks
        ):
            header = {**header_base, "episode": episode}
            if len(tick_pat) > 1:
                # where in the tick pattern this episode's first row lands
                header["tick_phase"] = int(getattr(core, "tick_phase", 0))
            if episode_meta is not None:
                header.update(episode_meta(episode) or {})
            f.write(json.dumps(header, separators=(",", ":")) + "\n")
            ep_ticks = 0
            ep_return = 0.0
            best_progress = 0.0
            end: Optional[str] = None

            while True:
                states = core.get_states()  # pre-step snapshot (s_t)
                actions = policy.act(obs)
                # --view-continuous: a policy that steers with a float view
                # command exposes it as ``policy.view`` ((N, 2) float32, or
                # None); the discrete call is exactly the one that shipped
                view = getattr(policy, "view", None)
                if view is None:
                    obs, rewards, done, trunc, terminal_obs = core.step(actions)
                else:
                    obs, rewards, done, trunc, terminal_obs = core.step(
                        actions, view=view)
                if on_tick is not None:
                    on_tick(total_ticks, states, rewards, done, trunc)

                s0 = states[0]
                r0 = float(rewards[0])
                ep_return += r0
                best_progress = max(
                    best_progress, float(s0["best_progress"]), float(s0["progress"])
                )
                buttons = (SURF_IN_JUMP if actions[0, 4] else 0) | (
                    SURF_IN_DUCK if actions[0, 5] else 0
                )
                ox, oy, oz = (float(v) for v in s0["origin"])
                vx, vy, vz = (float(v) for v in s0["velocity"])
                line = [
                    ep_ticks,
                    _round(ox, 2), _round(oy, 2), _round(oz, 2),
                    _round(vx, 2), _round(vy, 2), _round(vz, 2),
                    _round(float(s0["yaw"]), 2),
                    int(buttons),
                    int(int(s0["onground"]) >= 0),
                    _round(float(s0["progress"]), 2),
                    _round(r0, 5),
                    _round(float(s0["pitch"]), 2),   # index 12; viewer-safe append
                    int(actions[0, 2]),              # index 13: fwd (0 S, 1 -, 2 W)
                    int(actions[0, 3]),              # index 14: side (0 A, 1 -, 2 D)
                ]
                f.write(json.dumps(line, separators=(",", ":")) + "\n")
                ep_ticks += 1
                total_ticks += 1

                if done[0] or trunc[0]:
                    if done[0]:
                        # a finish on a goal-box core (--goals / --goal-planner recordings)
                        # pays no bonus in the core's own reward - the wrapper adds it - so
                        # the core's per-step goal flag decides too (it labelled every
                        # planner recording's finish "fail")
                        end = ("done" if (r0 >= _DONE_BONUS_MIN
                                          or bool(np.asarray(core.goal_hits)[0]))
                               else "fail")
                    else:
                        end = "trunc"
                    break
                if max_ticks is not None and total_ticks >= max_ticks:
                    end = "trunc"  # recording budget exhausted mid-episode
                    break

            trailer = {
                "end": end,
                "ticks": ep_ticks,
                "best_progress": _round(best_progress, 2),
            }
            # an episode_meta may carry an ``episode_end(episode) -> dict`` attribute: what it
            # returns is merged into the episode's trailer (a re-planning planner's every plan,
            # for the viewer). None on every other recording: the trailer that shipped
            _ee = getattr(episode_meta, "episode_end", None) if episode_meta is not None else None
            if _ee is not None:
                trailer.update(_ee(episode) or {})
            f.write(json.dumps(trailer, separators=(",", ":")) + "\n")
            f.flush()
            summaries.append({**trailer, "ep_return": _round(ep_return, 4)})
            episode += 1

    return summaries


def record_rollout_parallel(
    core: SurfCore,
    policy: Any,
    path: Union[str, os.PathLike],
    episodes: Optional[int] = None,
    max_ticks: Optional[int] = None,
    *,
    seed: Optional[int] = 0,
    on_tick: Optional[Callable[[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]] = None,
    episode_meta: Optional[Callable[[int], Dict[str, Any]]] = None,
    header_extra: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """``record_rollout`` with the episodes run SIDE BY SIDE: env ``i`` of ``core`` plays
    episode ``i`` (its FIRST episode after the reset), all of them at once, and the file
    holds them in env order - the same header / row / trailer format, row for row what
    ``record_rollout`` writes for an episode from the same spawn state.

    Why (``--eval-parallel``, autoresearch r4b4, 2026-10-21): the trainer's greedy eval
    ran its 9 episodes one after another on a 1-env core, so an eval with a few episodes
    alive at the 300 s cap took 90-150 s of a ~125 s eval cycle - 45-49% of the box's
    wall clock was a single env stepping while every other core idled.

    ``episodes`` (default ``core.num_envs``) envs are recorded (the rest, if any, are
    stepped and ignored); an env whose recorded episode has ended keeps stepping (the
    core autoresets it) and is no longer written. ``max_ticks`` is the WALL tick budget
    (``None`` = until every recorded episode ends; the core's own episode cap truncates
    each one); an episode still open when it runs out ends ``trunc``. ``on_tick`` is
    called once per tick with every env's pre-step snapshot, exactly as in
    ``record_rollout``. Returns one summary per recorded episode, in env order."""
    n_env = int(core.num_envs)
    n = n_env if episodes is None else int(episodes)
    if not 1 <= n <= n_env:
        raise ValueError(f"record_rollout_parallel: episodes must be in 1..{n_env} "
                         f"(the core's envs), got {episodes}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    map_name = Path(core.bsp_path).stem
    tick_pat = tuple(getattr(core, "tick_pattern", (int(core.config.phys.msec),)))
    nominal_msec = int(getattr(core, "nominal_msec", core.config.phys.msec))
    phys = phys_to_dict(core.config.phys)
    phys["msec"] = nominal_msec
    header_base = {
        "map": map_name,
        "tick_ms": nominal_msec,
        "phys": phys,
    }
    header_base.update(header_fields(float(sum(tick_pat)) / len(tick_pat),
                                     tick_pat, 0))
    if header_extra:
        header_base.update(header_extra)

    if seed is not None:
        obs = core.reset(seed)
    else:
        obs = np.zeros((core.num_envs, core.obs_dim), dtype=np.float32)
    headers = []
    for i in range(n):
        header = {**header_base, "episode": i}
        if len(tick_pat) > 1:
            # every recorded episode starts on the reset's tick, so on the same phase
            header["tick_phase"] = int(getattr(core, "tick_phase", 0))
        if episode_meta is not None:
            header.update(episode_meta(i) or {})
        headers.append(header)
    lines: List[List[str]] = [[] for _ in range(n)]
    ep_ticks = [0] * n
    ep_return = [0.0] * n
    best_progress = [0.0] * n
    ends: List[Optional[str]] = [None] * n
    open_ = list(range(n))
    total_ticks = 0
    while open_ and (max_ticks is None or total_ticks < max_ticks):
        states = core.get_states()  # pre-step snapshot (s_t)
        actions = policy.act(obs)
        view = getattr(policy, "view", None)
        if view is None:
            obs, rewards, done, trunc, terminal_obs = core.step(actions)
        else:
            obs, rewards, done, trunc, terminal_obs = core.step(actions, view=view)
        if on_tick is not None:
            on_tick(total_ticks, states, rewards, done, trunc)
        still = []
        for i in open_:
            s0 = states[i]
            r0 = float(rewards[i])
            ep_return[i] += r0
            best_progress[i] = max(
                best_progress[i], float(s0["best_progress"]), float(s0["progress"])
            )
            buttons = (SURF_IN_JUMP if actions[i, 4] else 0) | (
                SURF_IN_DUCK if actions[i, 5] else 0
            )
            ox, oy, oz = (float(v) for v in s0["origin"])
            vx, vy, vz = (float(v) for v in s0["velocity"])
            line = [
                ep_ticks[i],
                _round(ox, 2), _round(oy, 2), _round(oz, 2),
                _round(vx, 2), _round(vy, 2), _round(vz, 2),
                _round(float(s0["yaw"]), 2),
                int(buttons),
                int(int(s0["onground"]) >= 0),
                _round(float(s0["progress"]), 2),
                _round(r0, 5),
                _round(float(s0["pitch"]), 2),
                int(actions[i, 2]),
                int(actions[i, 3]),
            ]
            lines[i].append(json.dumps(line, separators=(",", ":")))
            ep_ticks[i] += 1
            if done[i] or trunc[i]:
                if done[i]:
                    ends[i] = ("done" if (r0 >= _DONE_BONUS_MIN
                                          or bool(np.asarray(core.goal_hits)[i]))
                               else "fail")
                else:
                    ends[i] = "trunc"
            else:
                still.append(i)
        open_ = still
        total_ticks += 1
    for i in open_:
        ends[i] = "trunc"  # recording budget exhausted mid-episode

    summaries: List[Dict[str, Any]] = []
    _ee = getattr(episode_meta, "episode_end", None) if episode_meta is not None else None
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            f.write(json.dumps(headers[i], separators=(",", ":")) + "\n")
            for ln in lines[i]:
                f.write(ln + "\n")
            trailer = {
                "end": ends[i],
                "ticks": ep_ticks[i],
                "best_progress": _round(best_progress[i], 2),
            }
            if _ee is not None:
                trailer.update(_ee(i) or {})
            f.write(json.dumps(trailer, separators=(",", ":")) + "\n")
            summaries.append({**trailer, "ep_return": _round(ep_return[i], 4)})
    return summaries
