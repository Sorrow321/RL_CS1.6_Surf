"""Trajectory recording — JSONL per docs/04-visualizer.md.

File layout (env 0 only, multi-episode per file):

    {"map": "surf_ski_2", "tick_ms": 10, "phys": {...}, "episode": 0}
    [t, x,y,z, vx,vy,vz, yaw, buttons, onground, progress, reward, pitch,
     fwd, side]                        # fwd/side: raw move bins 0/1/2
    ...
    {"end": "fail|done|trunc", "ticks": 4130, "best_progress": 8121.5}

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

__all__ = ["record_rollout"]

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
) -> List[Dict[str, Any]]:
    """Roll ``policy`` on ``core`` and write env 0's trajectory as JSONL.

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
    header_base = {
        "map": map_name,
        "tick_ms": int(core.config.phys.msec),
        "phys": phys_to_dict(core.config.phys),
    }

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
            f.write(json.dumps({**header_base, "episode": episode},
                               separators=(",", ":")) + "\n")
            ep_ticks = 0
            ep_return = 0.0
            best_progress = 0.0
            end: Optional[str] = None

            while True:
                states = core.get_states()  # pre-step snapshot (s_t)
                actions = policy.act(obs)
                obs, rewards, done, trunc, terminal_obs = core.step(actions)
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
                        end = "done" if r0 >= _DONE_BONUS_MIN else "fail"
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
            f.write(json.dumps(trailer, separators=(",", ":")) + "\n")
            f.flush()
            summaries.append({**trailer, "ep_return": _round(ep_return, 4)})
            episode += 1

    return summaries
