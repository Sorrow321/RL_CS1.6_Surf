#!/usr/bin/env python3
"""make_sample_traj.py -- synthetic .traj.jsonl for viewer playback testing.

Two episodes of circular flight around the map bounds center, matching the
format from docs/04-visualizer.md exactly:

    {"map":...,"tick_ms":10,"phys":{...},"episode":0}      # header
    [t, x,y,z, vx,vy,vz, yaw, buttons, onground, progress, reward]
    ...
    {"end":"done","ticks":N,"best_progress":P}             # trailer

Usage: python tools/make_sample_traj.py [mesh.json] [out.traj.jsonl]
Defaults: viewer/assets/surf_ski_2.mesh.json -> viewer/assets/sample.traj.jsonl
"""

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "viewer", "assets", "surf_ski_2.mesh.json")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        ROOT, "viewer", "assets", "sample.traj.jsonl")

    map_name = "surf_ski_2"
    cx, cy, cz = 0.0, 0.0, 256.0
    try:
        with open(mesh_path) as f:
            mesh = json.load(f)
        map_name = mesh.get("map", map_name)
        mins, maxs = mesh["bounds"]["mins"], mesh["bounds"]["maxs"]
        cx = (mins[0] + maxs[0]) / 2.0
        cy = (mins[1] + maxs[1]) / 2.0
        cz = (mins[2] + maxs[2]) / 2.0
    except (IOError, OSError, KeyError, ValueError):
        print("note: could not read %s, using default center" % mesh_path)

    tick_ms = 10
    dt = tick_ms / 1000.0
    ticks = 200
    lines = []

    # (radius, height offset, angular speed rad/s, bob amplitude)
    episodes = [(1400.0, 300.0, 0.9, 120.0),
                (900.0, 600.0, -1.4, 200.0)]

    for ep, (radius, zoff, omega, bob) in enumerate(episodes):
        lines.append(json.dumps({
            "map": map_name, "tick_ms": tick_ms,
            "phys": {"sv_airaccelerate": 100, "sv_gravity": 800},
            "episode": ep}, separators=(",", ":")))
        best = 0.0
        for t in range(ticks):
            a = omega * t * dt
            x = cx + radius * math.cos(a)
            y = cy + radius * math.sin(a)
            z = cz + zoff + bob * math.sin(3.0 * a)
            vx = -radius * omega * math.sin(a)
            vy = radius * omega * math.cos(a)
            vz = bob * 3.0 * omega * math.cos(3.0 * a)
            yaw = math.degrees(math.atan2(vy, vx))
            progress = abs(radius * omega) * t * dt
            best = max(best, progress)
            reward = 0.1 * math.hypot(vx, vy) * dt
            lines.append(json.dumps(
                [t, round(x, 2), round(y, 2), round(z, 2),
                 round(vx, 2), round(vy, 2), round(vz, 2),
                 round(yaw, 2), 0, 0, round(progress, 2), round(reward, 4)],
                separators=(",", ":")))
        lines.append(json.dumps(
            {"end": "done", "ticks": ticks, "best_progress": round(best, 2)},
            separators=(",", ":")))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s: %d episodes x %d ticks, center=(%.0f, %.0f, %.0f)"
          % (out_path, len(episodes), ticks, cx, cy, cz))


if __name__ == "__main__":
    main()
