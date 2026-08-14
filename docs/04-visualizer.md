# 04 — Visualizer

Purpose: *watch* agents move through the real map geometry, and author the track spline. Browser-based (three.js), fed by two static inputs: an exported map mesh and trajectory files. No engine, no server round-trips.

```
maps/surf_x.bsp ── tools/export_map.py ──▶ viewer/assets/surf_x.mesh.json
training / surfgym.record ───────────────▶ runs/<name>.traj.jsonl
                    viewer/index.html  (drag-and-drop both)
```

## Mesh export (`tools/export_map.py`, pure Python)

Reads the BSP render lumps (VERTICES, EDGES, SURFEDGES, FACES, TEXINFO, MODELS — see [02](02-bsp-collision.md)) and writes one JSON:

- **Worldspawn** (model 0): triangulate each face as a fan over its surfedge loop. Skip faces whose texture name is `sky`/`aaatrigger`/`{blue`-style null textures. No textures/lightmaps — flat shading tinted by face normal is plenty; specifically **color by `normal.z` with a hard break at 0.7**, so surfable planes (`n.z < 0.7`, never ground — [01](01-physics-spec.md)) render in a distinct hue from walkable floors. That one trick makes every ramp readable at a glance.
- **Brush entities**: each exported as its own mesh, tagged by classname, rendered translucent: `trigger_teleport` blue, `trigger_hurt` red, `trigger_push` yellow, `func_water` cyan, other solids grey.
- **Markers**: spawn points (`info_player_*`) as arrows with yaw, `info_teleport_destination` as rings.
- Format: `{"world": {"positions": [...], "indices": [...], "normals": [...]}, "brushes": [{"classname", "positions", ...}], "markers": [...]}`. Positions in map units, Z-up (viewer converts to three.js Y-up once, at load).
- Typical surf map ⇒ a few MB of JSON; fine. (Binary buffers are a later micro-optimization.)

## Trajectory format (`.traj.jsonl`)

Line 1 header, then one line per tick — dumb on purpose, greppable, streamable, append-while-training:

```jsonl
{"map":"surf_ski_2","tick_ms":10,"phys":{"sv_airaccelerate":100,...},"episode":0}
[t, x,y,z, vx,vy,vz, yaw, buttons, onground, progress, reward]
...
{"end":"fail|done|trunc","ticks":4130,"best_progress":8121.5}   # episode trailer, then next header
```

Producer: `surfgym.record.rollout(policy, env, path)` — pulls `surf_get_states` each tick. Multi-episode per file. `.npz` twin for bulk analysis later; the viewer speaks JSONL only (v1).

## Viewer (`viewer/index.html` + `viewer/app.js`)

Vendored three.js (one file into `viewer/vendor/` — works offline, no CDN). Serve with `python -m http.server` from repo root, or straight `file://` if drag-and-drop is used for all assets.

**Playback**
- Player = wireframe box at the *exact* collision hull size from [02](02-bsp-collision.md) (standing 32×32×72, crouched 32×32×36, origin at hull center — confirmed from engine `player_mins/maxs`) — so "did it clip that edge?" is answerable by eye.
- Velocity arrow (horizontal component + separate vertical tick), live speed readout (u/s), speed sparkline over the last ~10 s.
- Trail polyline colored by speed (slow=blue → fast=red), persistent per episode.
- Transport: play/pause/scrub bar, speed 0.25×–16×, step ±1 tick, jump to episode N (episodes indexed from headers/trailers at load).
- Cameras: orbit (default), follow-cam (chase behind velocity), first-person-ish (at eye height looking along yaw).
- **Ghosts**: load multiple trajectory files → simultaneous playback aligned by tick or by progress; how you *see* training improve across checkpoints.

**Waypoint editor mode** (this is also the spline tool from [03](03-env-design.md))
- Toggle edit mode: click on world mesh → raycast hit + 40u up = new waypoint appended; drag existing to move (on a camera-parallel plane), right-click delete, insert between via midpoint handles.
- Waypoints rendered as numbered spheres joined by a tube; export/import `maps/<map>.waypoints.json` (`{"map": "...", "points": [[x,y,z], ...]}`) via download / drag-in.
- Loading a trajectory in edit mode overlays it — refine the line where the agent actually flies.

**Implementation notes**
- One `requestAnimationFrame` loop; playback time → tick index → lerp between tick states for smoothness at high map speeds.
- Everything in one `app.js` (~500–700 lines); no bundler, no framework. `OrbitControls` from the three.js examples file, also vendored.
- Coordinate note in one place only: GoldSrc is Z-up right-handed; convert to three.js Y-up at load (`(x,y,z) → (x,z,−y)`) and never think about it again.

## Prior art / shortcuts

If the custom exporter stalls: [lewa-j/hlbsp-converter](https://github.com/lewa-j/hlbsp-converter) converts GoldSrc v30 BSP → glTF (textures included) — swap `export_map.py` for it + `GLTFLoader` and lose nothing. [sbuggay/bspview](https://github.com/sbuggay/bspview) is a small three.js BSP explorer to crib parser code from. [hlviewer.js](https://github.com/skyrim/hlviewer.js) renders GoldSrc maps *and plays .dem files* in-browser — overkill tonight, but the natural home for "replay the deployed agent's real-server demo" later. We still prefer the custom exporter: brush entities tagged by classname (translucent trigger volumes) and the waypoint editor need our own scene graph anyway.

## Cut line (tonight)

Must-have: mesh render, JSONL playback, scrub, follow cam, speed readout, waypoint edit + export. Nice-to-have if time remains: ghosts, sparkline, trail-by-speed, first-person cam.
