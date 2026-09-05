"""play.py — first-person playable client for the surfcore physics.

The REAL surfcore.dll ticks at a fixed 100 Hz (msec=10) with live map triggers
(boosters, teleports); rendering interpolates between physics states. Yaw/pitch
are client-authoritative exactly like the GoldSrc client: the mouse moves the
camera instantly and the accumulated view angles go into each usercmd.

Bindings (CS 1.6 style):
  WASD          move                 Mouse        look (m_yaw 0.022 * sens)
  Space         +jump (held)         Wheel up/dn  jump for exactly 1 tick (mwheel bhop)
  Ctrl          +duck (inert v1)     R            teleport to start
  N             cycle spawn points   T / Y        save / load practice checkpoint
  [ / ]         sensitivity -/+ 0.1  F3           third-person   F4 trigger volumes
  Esc           settings menu (resolution, fps limit, sens, teleport mode)

Run:  python python\\play.py            (defaults to surf_ski_2, aa=100)
      python python\\play.py --sens 2.2 --no-stamina
"""
import argparse
import ctypes
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import pyglet
from pyglet.gl import (
    GL_BLEND, GL_DEPTH_TEST, GL_ONE_MINUS_SRC_ALPHA, GL_POLYGON_OFFSET_FILL,
    GL_SRC_ALPHA, GL_TRIANGLES, glBlendFunc, glDepthMask, glDisable, glEnable,
    glPolygonOffset,
)
from pyglet.math import Mat4, Vec3
from pyglet.window import key

from surfgym import SurfCore, default_config
from surfgym.core import SurfState

M_YAW = 0.022
IN_JUMP, IN_DUCK = 2, 4
TICK_MS = 10
TICK_S = TICK_MS / 1000.0
VIEW_OFS_Z = 17.0
SETTINGS_PATH = ROOT / "python" / "play_settings.json"

RESOLUTIONS = [(1280, 800), (1440, 900), (1600, 900), (1920, 1080), (2560, 1440)]
FPS_LIMITS = [60, 120, 144, 165, 240, 300, 500, 0]      # 0 = uncapped

VERT_SRC = """#version 150 core
in vec3 position; in vec3 normal; in vec3 color;
uniform mat4 u_mvp;
out vec3 v_color; out vec3 v_normal;
void main() { gl_Position = u_mvp * vec4(position, 1.0); v_color = color; v_normal = normal; }
"""
FRAG_SRC = """#version 150 core
in vec3 v_color; in vec3 v_normal; out vec4 frag;
void main() {
    vec3 n = normalize(v_normal);
    float l = 0.55 + 0.30 * max(dot(n, normalize(vec3(0.35, 0.20, 0.91))), 0.0)
                   + 0.15 * max(dot(n, normalize(vec3(-0.60, -0.70, 0.40))), 0.0);
    frag = vec4(v_color * l, 1.0);
}
"""
FRAG_FLAT = """#version 150 core
in vec3 v_color; in vec3 v_normal; out vec4 frag;
uniform float u_alpha;
void main() { frag = vec4(v_color, u_alpha); }
"""

OVERLAY_COLORS = {                  # F4-toggleable trigger volumes
    "trigger_push":     (1.00, 0.85, 0.20, 0.30),
    "trigger_hurt":     (1.00, 0.25, 0.25, 0.30),
}
SCENERY_COLORS = {                  # always-on world dressing
    "func_water":       (0.15, 0.65, 0.90, 0.45),
    "func_ladder":      (0.75, 0.60, 0.25, 0.45),
    "func_illusionary": (0.55, 0.58, 0.66, 0.30),
}
WATER_RGBA = SCENERY_COLORS["func_water"]
SOLID_CLASSES = {
    "func_wall", "func_breakable", "func_pushable", "func_door", "func_button",
    "func_train", "func_conveyor", "func_wall_toggle", "func_rotating",
    "func_door_rotating",
}
SOLID_TINT_MUL = (0.94, 0.98, 1.04)   # slight cool cast so brush solids read as such


def face_tint(nz: float):
    if nz >= 0.7:   return (0.62, 0.70, 0.62)
    if nz > 0.0:    return (0.95, 0.70, 0.25)
    if nz > -0.7:   return (0.52, 0.54, 0.58)
    return (0.36, 0.38, 0.42)


def load_mesh(mesh_path: Path, bsp_path: Path) -> dict:
    if not mesh_path.exists():
        print(f"mesh not found, exporting from {bsp_path.name} ...")
        subprocess.check_call([sys.executable, str(ROOT / "tools" / "export_map.py"),
                               str(bsp_path), str(mesh_path)])
    with open(mesh_path, encoding="utf-8") as f:
        return json.load(f)


def defight_coplanar(pos, nrm, idx):
    """Kill world-vs-world z-fighting: reconstruct fan faces (consecutive tris
    sharing the first index), group HORIZONTAL faces by exact plane z, and lift
    each overlapping face by 0.2u per rank (largest face keeps the true plane).
    surf_ski_2's arena floor stacks 118 faces at exactly z=-1856. Sub-unit lifts
    are invisible but decisively separate the depth values."""
    pos = list(pos)
    faces = []                       # (first_tri, ntris, anchor_index)
    t = 0
    ntri = len(idx) // 3
    while t < ntri:
        a = idx[3 * t]
        run = t + 1
        while run < ntri and idx[3 * run] == a:
            run += 1
        faces.append((t, run - t, a))
        t = run
    groups = {}
    for fi, (t0, n, a) in enumerate(faces):
        nz = nrm[3 * a + 2]
        if abs(nz) < 0.999:
            continue
        z = pos[3 * a + 2]
        groups.setdefault((round(z, 2), nz > 0), []).append(fi)
    lifted = 0
    for (z, up), fis in groups.items():
        if len(fis) < 2:
            continue
        def area(fi):
            t0, n, _ = faces[fi]
            s = 0.0
            for tt in range(t0, t0 + n):
                i0, i1, i2 = idx[3*tt], idx[3*tt+1], idx[3*tt+2]
                ax, ay = pos[3*i1] - pos[3*i0], pos[3*i1+1] - pos[3*i0+1]
                bx, by = pos[3*i2] - pos[3*i0], pos[3*i2+1] - pos[3*i0+1]
                s += abs(ax * by - ay * bx)
            return s
        order = sorted(fis, key=area, reverse=True)
        for rank, fi in enumerate(order[1:], start=1):
            dz = 0.2 * min(rank, 8) * (1.0 if up else -1.0)
            t0, n, _ = faces[fi]
            seen = set()
            for tt in range(t0, t0 + n):
                for v in (idx[3*tt], idx[3*tt+1], idx[3*tt+2]):
                    if v not in seen:
                        seen.add(v)
                        pos[3 * v + 2] += dz
            lifted += 1
    return pos, lifted


class PlayApp:
    def __init__(self, args):
        self.args = args
        self.settings = self.load_settings(args)
        bsp = ROOT / args.map
        cfg = default_config(
            num_envs=1, spawn_mode=0,
            water_fail=0,                    # humans swim; the env kills, we don't
            sv_airaccelerate=args.aa, sv_gravity=args.gravity,
            sv_maxvelocity=args.maxvel,
            enable_stamina=0 if args.no_stamina else 1,
            enable_bhop_cap=0 if args.no_bhop_cap else 1,
        )
        self.core = SurfCore(str(bsp), cfg)
        self.spawns = self.core.spawns()
        self.spawn_idx = 0

        self.st = SurfState()
        self.yaw = 0.0
        self.pitch = 0.0
        self.wheel_jump_ticks = 0
        self.checkpoint = None
        self.third_person = False
        # practice tools: H (hold) hook-pull along view, G freeze toggle,
        # J noclip toggle — client-side only, physics/triggers bypassed
        self.frozen = False
        self.noclip = False
        self.msg = ""
        self.msg_until = 0.0
        self.acc = 0.0
        self.prev_pos = [0.0, 0.0, 0.0]
        self.cur_pos = [0.0, 0.0, 0.0]
        self.tick_meter = [0, time.perf_counter(), 0.0]
        self.draw_meter = [0, time.perf_counter(), 0.0]
        self.paused = False
        self.menu_sel = 0
        self.replay = None
        self._ffmpeg = None
        if args.replay and args.dump:
            # deterministic offline render: fixed window, no pacing
            self.settings["display_mode"] = "windowed"
            self.settings["width"], self.settings["height"] = 1280, 720
            self.settings["vsync"] = False
            self.settings["fps_limit"] = 0

        mesh_path = (ROOT / args.mesh) if args.mesh else \
            ROOT / "viewer" / "assets" / (bsp.stem + ".mesh.json")
        self.mesh_raw = load_mesh(mesh_path, bsp)
        self.window = None
        self.mouse_captured = False
        self.create_window()
        if not args.selftest and not args.replay:
            self.capture(True)
        self.respawn(0)
        self.apply_fps_limit()
        if args.selftest:
            pyglet.clock.schedule_once(self.end_selftest, 2.0)
        self.tick_ms = float(TICK_MS)
        if args.replay:
            self.replay, hdr = self._load_replay(args.replay, args.ep)
            # the recording's own tick (7.67 ms at 131 Hz): playback speed,
            # the dump's frame rate and the run clock all follow it
            self.tick_ms = float(hdr.get("tick_ms") or TICK_MS)
            self.replay_i = 0.0
            # run clock. --clock cliff (legacy): start when the agent leaves
            # the start cliff, end at the last frame. --clock zone: the
            # leaderboard's clock - start curtain to finish curtain from
            # maps/<map>.zones.json, sub-tick interpolated.
            z0 = self.replay[0][3]
            self.timer_start = float(next((i for i, r in enumerate(self.replay)
                                           if r[3] < z0 - 100.0), 0))
            self.timer_end = float(len(self.replay) - 1)
            self.clock_mode = "cliff"
            if args.clock == "zone":
                import json as _json
                zp = (ROOT / args.map).with_suffix(".zones.json")
                try:
                    zones = _json.loads(zp.read_text(encoding="utf-8"))
                except OSError:
                    zones = None
                    print(f"[clock] no {zp}, falling back to the cliff clock")
                if zones:
                    t0 = self._curtain_crossing(self.replay, zones.get("start"))
                    t1 = (self._curtain_crossing(self.replay, zones.get("end"), int(t0) + 1)
                          if t0 is not None else None)
                    if t0 is not None:
                        self.timer_start, self.clock_mode = t0, "zone"
                        if t1 is None and zones.get("end"):
                            # the recording stops at the finish TOUCH, a few
                            # ms before the curtain's mid-plane: extrapolate
                            # the last row along its velocity (compare_wr's rule)
                            last = self.replay[-1]
                            yc = 0.5 * (zones["end"]["mins"][1] + zones["end"]["maxs"][1])
                            vy = last[5] if len(last) > 5 else 0.0
                            if vy > 1.0 and 0.0 <= (yc - last[2]) / vy <= 0.1:
                                t1 = (len(self.replay) - 1) + (yc - last[2]) / vy / (self.tick_ms / 1000.0)
                        if t1 is not None:
                            self.timer_end = t1
                        print(f"[clock] zone: start {t0 * self.tick_ms / 1000:.3f} s, "
                              f"finish {(t1 * self.tick_ms / 1000) if t1 is not None else float('nan'):.3f} s "
                              f"-> {(self.timer_end - self.timer_start) * self.tick_ms / 1000:.3f} s")
                    else:
                        print("[clock] start curtain never crossed, cliff clock")
            self._apply_replay(0, 0.0)
            if not args.dump:
                self.say(f"REPLAY {args.replay} ep {args.ep}")
            if args.dump:
                self._start_dump()
                pyglet.clock.unschedule(self.update)
                pyglet.clock.unschedule(self._draw_frame)
                pyglet.clock.schedule(self._dump_step)

    # ---- settings ----------------------------------------------------------
    @staticmethod
    def load_settings(args):
        s = {"width": 1440, "height": 900, "display_mode": "windowed",
             "vsync": False, "fps_limit": 300, "sens": 2.5,
             "teleport_respawn": True, "show_triggers": True}
        try:
            old = json.load(open(SETTINGS_PATH, encoding="utf-8"))
            if old.pop("fullscreen", False):          # migrate pre-display_mode files
                old.setdefault("display_mode", "fullscreen")
            s.update(old)
        except Exception:
            pass
        if s["display_mode"] not in ("windowed", "borderless", "fullscreen"):
            s["display_mode"] = "windowed"
        if args.sens is not None:
            s["sens"] = args.sens
        if args.display is not None:
            s["display_mode"] = args.display
        return s

    def save_settings(self):
        try:
            json.dump(self.settings, open(SETTINGS_PATH, "w", encoding="utf-8"), indent=2)
        except Exception:
            pass

    def apply_fps_limit(self):
        """We own the frame pacing: pyglet.app.run(None) schedules no redraws;
        update+draw are scheduled together at the target rate (update first, so
        each frame renders the freshest physics state)."""
        pyglet.clock.unschedule(self.update)
        pyglet.clock.unschedule(self._draw_frame)
        fps = self.settings["fps_limit"]
        if fps > 0:
            pyglet.clock.schedule_interval(self.update, 1.0 / fps)
            pyglet.clock.schedule_interval(self._draw_frame, 1.0 / fps)
        else:
            pyglet.clock.schedule(self.update)
            pyglet.clock.schedule(self._draw_frame)

    def _draw_frame(self, dt):
        self.window.switch_to()
        self.window.dispatch_event("on_draw")
        if self._ffmpeg is not None:
            buf = pyglet.image.get_buffer_manager().get_color_buffer()
            img = buf.get_image_data()
            self._ffmpeg.stdin.write(img.get_data("RGBA", img.width * 4))
        self.window.flip()
        self.draw_meter[0] += 1
        now = time.perf_counter()
        if now - self.draw_meter[1] >= 0.5:
            self.draw_meter[2] = self.draw_meter[0] / (now - self.draw_meter[1])
            self.draw_meter[0], self.draw_meter[1] = 0, now

    def create_window(self):
        """(Re)create the window for the current display mode and rebuild all
        GL resources into its context. Borderless = undecorated desktop-sized
        window, DWM-composited — the OBS-friendly 'fullscreen (windowed)'."""
        old = self.window
        mode = self.settings["display_mode"]
        cfg_gl = pyglet.gl.Config(depth_size=24, double_buffer=True)
        screen = pyglet.display.get_display().get_default_screen()
        kw = dict(caption="RL_Surf — play", config=cfg_gl,
                  vsync=self.settings["vsync"])
        if mode == "borderless":
            self.window = pyglet.window.Window(
                width=screen.width, height=screen.height,
                style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS, **kw)
            self.window.set_location(screen.x, screen.y)
        elif mode == "fullscreen":
            self.window = pyglet.window.Window(fullscreen=True, **kw)
        else:
            self.window = pyglet.window.Window(
                width=self.settings["width"], height=self.settings["height"], **kw)
        self.keys = key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        self.window.push_handlers(
            on_draw=self.on_draw, on_mouse_motion=self.on_mouse_motion,
            on_key_press=self.on_key_press, on_mouse_scroll=self.on_mouse_scroll,
            on_mouse_press=self.on_mouse_press, on_resize=self.on_resize)
        self.window.switch_to()
        self.build_scene(self.mesh_raw)
        self.build_hud()
        if self.args.replay:
            self._build_keys()
        if old is not None:
            old.close()
        if self.paused:
            self.build_menu()
        self.capture(self.mouse_captured and not self.paused)

    def apply_display(self):
        mode = self.settings["display_mode"]
        if mode == "windowed" and self.window is not None and \
                self.window.style == pyglet.window.Window.WINDOW_STYLE_DEFAULT and \
                not self.window.fullscreen:
            self.window.set_size(self.settings["width"], self.settings["height"])
        else:
            # deferred: never destroy the window from inside its own event dispatch
            pyglet.clock.schedule_once(lambda dt: self.create_window(), 0.0)

    # ---- scene -------------------------------------------------------------
    def build_scene(self, mesh):
        self.prog = pyglet.graphics.shader.ShaderProgram(
            pyglet.graphics.shader.Shader(VERT_SRC, "vertex"),
            pyglet.graphics.shader.Shader(FRAG_SRC, "fragment"))
        self.prog_flat = pyglet.graphics.shader.ShaderProgram(
            pyglet.graphics.shader.Shader(VERT_SRC, "vertex"),
            pyglet.graphics.shader.Shader(FRAG_FLAT, "fragment"))

        w = mesh["world"]
        pos, lifted = defight_coplanar(w["positions"], w["normals"], w["indices"])
        print(f"coplanar de-fight: lifted {lifted} overlapping horizontal faces")
        nrm, idx = w["normals"], w["indices"]
        nvert = len(pos) // 3
        col = []
        for i in range(nvert):
            col.extend(face_tint(nrm[3 * i + 2]))
        self.world_vl = self.prog.vertex_list_indexed(
            nvert, GL_TRIANGLES, idx,
            position=("f", tuple(pos)), normal=("f", tuple(nrm)), color=("f", tuple(col)))

        self.solid_vls = []       # opaque brush entities (jail walls etc.)
        self.overlay_vls = []     # F4-toggleable trigger volumes
        self.teleport_vls = []
        self.scenery_vls = []     # always-on translucent: water, ladders, illusionary
        for b in mesh.get("brushes", []):
            cls = b.get("classname", "")
            bp, bn, bi = b["positions"], b["normals"], b["indices"]
            n = len(bp) // 3
            if cls in SOLID_CLASSES:
                c = []
                for i in range(n):
                    t = face_tint(bn[3 * i + 2])
                    c.extend((t[0] * SOLID_TINT_MUL[0], t[1] * SOLID_TINT_MUL[1],
                              t[2] * SOLID_TINT_MUL[2]))
                self.solid_vls.append(self.prog.vertex_list_indexed(
                    n, GL_TRIANGLES, bi,
                    position=("f", tuple(bp)), normal=("f", tuple(bn)), color=("f", tuple(c))))
            elif cls in SCENERY_COLORS:
                rgba = SCENERY_COLORS[cls]
                if cls == "func_illusionary" and int(b.get("skin", 0)) < -1:
                    rgba = WATER_RGBA            # water-content illusionary (the waterfall)
                c = list(rgba[:3]) * n
                vl = self.prog_flat.vertex_list_indexed(
                    n, GL_TRIANGLES, bi,
                    position=("f", tuple(bp)), normal=("f", tuple(bn)), color=("f", tuple(c)))
                self.scenery_vls.append((rgba[3], vl))
            elif cls in OVERLAY_COLORS or cls == "trigger_teleport":
                rgba = OVERLAY_COLORS.get(cls, (0.25, 0.45, 1.00, 0.22))
                c = list(rgba[:3]) * n
                vl = self.prog_flat.vertex_list_indexed(
                    n, GL_TRIANGLES, bi,
                    position=("f", tuple(bp)), normal=("f", tuple(bn)), color=("f", tuple(c)))
                (self.teleport_vls if cls == "trigger_teleport"
                 else self.overlay_vls).append((rgba[3], vl))

    def build_hud(self):
        self.hud = pyglet.graphics.Batch()
        W, H = self.window.width, self.window.height
        self.lbl_speed = pyglet.text.Label(
            "0", font_size=34, x=W // 2, y=64, anchor_x="center", weight="bold",
            color=(255, 255, 255, 230), batch=self.hud)
        # 3D speed under the 2D one — mirrors the surf-timer HUD pair the
        # user calibrates against on live servers
        self.lbl_speed3d = pyglet.text.Label(
            "0", font_size=18, x=W // 2, y=40, anchor_x="center",
            color=(190, 205, 255, 210), batch=self.hud)
        self.lbl_info = pyglet.text.Label(
            "", font_size=10, x=8, y=H - 16, multiline=True, width=520,
            color=(200, 220, 200, 220), batch=self.hud)
        self.lbl_msg = pyglet.text.Label(
            "", font_size=14, x=W // 2, y=H // 2 + 60, anchor_x="center",
            color=(255, 230, 120, 255), batch=self.hud)
        # surf-timer style run clock (replay only): starts at the cliff
        # jump, freezes green at the finish touch
        self.lbl_timer = pyglet.text.Label(
            "", font_size=28, x=W // 2, y=H - 64, anchor_x="center",
            weight="bold", color=(255, 255, 255, 235), batch=self.hud,
            multiline=True, width=640, align="center")
        self._crosshair = []
        for dx, dy in ((-12, 0), (4, 0), (0, -12), (0, 4)):
            self._crosshair.append(pyglet.shapes.Line(
                W // 2 + dx, H // 2 + dy,
                W // 2 + dx + (8 if dy == 0 else 0), H // 2 + dy + (8 if dx == 0 else 0),
                thickness=2, color=(120, 255, 120), batch=self.hud))
        self.menu_batch = None

    def _adj_display(self, d):
        modes = ["windowed", "borderless", "fullscreen"]
        i = modes.index(self.settings["display_mode"])
        self.settings["display_mode"] = modes[(i + d) % 3]
        self.apply_display()

    def _adj_resolution(self, d):
        if self.settings["display_mode"] != "windowed":
            return
        cur = (self.settings["width"], self.settings["height"])
        i = RESOLUTIONS.index(cur) if cur in RESOLUTIONS else 0
        self.settings["width"], self.settings["height"] = \
            RESOLUTIONS[(i + d) % len(RESOLUTIONS)]
        self.apply_display()

    def _adj_vsync(self, d):
        self.settings["vsync"] = not self.settings["vsync"]
        self.window.set_vsync(self.settings["vsync"])

    def _adj_fps(self, d):
        i = FPS_LIMITS.index(self.settings["fps_limit"]) \
            if self.settings["fps_limit"] in FPS_LIMITS else 5
        self.settings["fps_limit"] = FPS_LIMITS[(i + d) % len(FPS_LIMITS)]
        self.apply_fps_limit()

    def _adj_sens(self, d):
        self.settings["sens"] = max(0.1, round(self.settings["sens"] + 0.1 * d, 2))

    def _adj_teleport(self, d):
        self.settings["teleport_respawn"] = not self.settings["teleport_respawn"]

    def _adj_triggers(self, d):
        self.settings["show_triggers"] = not self.settings["show_triggers"]

    def menu_rows(self):
        s = self.settings
        mode_name = {"windowed": "Windowed", "borderless": "Borderless (windowed FS)",
                     "fullscreen": "Fullscreen"}[s["display_mode"]]
        res = (f'{s["width"]} x {s["height"]}' if s["display_mode"] == "windowed"
               else "— (set by display mode)")
        fps = "Uncapped" if s["fps_limit"] == 0 else str(s["fps_limit"])
        return [
            ("Display", mode_name, self._adj_display),
            ("Resolution", res, self._adj_resolution),
            ("Vsync", "On" if s["vsync"] else "Off", self._adj_vsync),
            ("FPS limit", fps, self._adj_fps),
            ("Sensitivity", f'{s["sens"]:.1f}', self._adj_sens),
            ("Teleports", "Respawn at start" if s["teleport_respawn"] else "Map behavior",
             self._adj_teleport),
            ("Trigger volumes", "Shown" if s["show_triggers"] else "Hidden",
             self._adj_triggers),
            ("Resume", "", None),
            ("Quit", "", None),
        ]

    def build_menu(self):
        self.menu_batch = pyglet.graphics.Batch()
        W, H = self.window.width, self.window.height
        self._menu_bg = pyglet.shapes.Rectangle(0, 0, W, H, color=(8, 10, 14),
                                                batch=self.menu_batch)
        self._menu_bg.opacity = 190
        self._menu_labels = [pyglet.text.Label(
            "SETTINGS", font_size=22, x=W // 2, y=H - 140, anchor_x="center",
            weight="bold", color=(255, 255, 255, 255), batch=self.menu_batch)]
        for i, (name, val, _adj) in enumerate(self.menu_rows()):
            sel = (i == self.menu_sel)
            col = (120, 255, 120, 255) if sel else (210, 210, 210, 220)
            pre = "> " if sel else "  "
            txt = f"{pre}{name}" + (f"   <  {val}  >" if val else "")
            self._menu_labels.append(pyglet.text.Label(
                txt, font_size=15, x=W // 2 - 220, y=H - 200 - i * 34,
                color=col, batch=self.menu_batch))
        nrows = len(self.menu_rows())
        self._menu_labels.append(pyglet.text.Label(
            "Up/Down select    Left/Right change    Enter activate    Esc resume",
            font_size=10, x=W // 2, y=H - 200 - nrows * 34 - 30, anchor_x="center",
            color=(150, 160, 150, 200), batch=self.menu_batch))

    # ---- input -------------------------------------------------------------
    def capture(self, on: bool):
        self.mouse_captured = on
        self.window.set_exclusive_mouse(on)

    def on_mouse_motion(self, x, y, dx, dy):
        if not self.mouse_captured or self.paused:
            return
        self.yaw = (self.yaw - dx * M_YAW * self.settings["sens"]) % 360.0
        self.pitch = max(-89.0, min(89.0, self.pitch - dy * M_YAW * self.settings["sens"]))

    def on_mouse_press(self, *a):
        if not self.paused and not self.mouse_captured:
            self.capture(True)

    def on_mouse_scroll(self, x, y, sx, sy):
        if not self.paused:
            self.wheel_jump_ticks += 1

    def on_resize(self, w, h):
        if hasattr(self, "hud"):
            self.build_hud()
            if self.paused:
                self.build_menu()

    def set_paused(self, on: bool):
        self.paused = on
        self.capture(not on and not self.args.selftest)
        if on:
            self.build_menu()
        else:
            self.menu_batch = None
            self.save_settings()

    def menu_adjust(self, direction: int):
        _name, _val, adj = self.menu_rows()[self.menu_sel]
        if adj is not None:
            adj(direction)
        self.build_menu()

    def on_key_press(self, sym, mods):
        if self.paused:
            n = len(self.menu_rows())
            if sym == key.ESCAPE:
                self.set_paused(False)
            elif sym == key.UP:
                self.menu_sel = (self.menu_sel - 1) % n; self.build_menu()
            elif sym == key.DOWN:
                self.menu_sel = (self.menu_sel + 1) % n; self.build_menu()
            elif sym in (key.LEFT, key.RIGHT):
                self.menu_adjust(1 if sym == key.RIGHT else -1)
            elif sym in (key.ENTER, key.RETURN, key.SPACE):
                name = self.menu_rows()[self.menu_sel][0]
                if name == "Resume":
                    self.set_paused(False)
                elif name == "Quit":
                    self.save_settings(); pyglet.app.exit()
                else:
                    self.menu_adjust(1)
            return True

        if sym == key.ESCAPE:
            self.set_paused(True)
            return True
        if sym == key.R:
            self.respawn(0); self.say("back to start")
        elif sym == key.N:
            self.spawn_idx = (self.spawn_idx + 1) % max(1, len(self.spawns))
            self.respawn(self.spawn_idx); self.say(f"spawn {self.spawn_idx + 1}")
        elif sym == key.T:
            self.checkpoint = (bytes(self.st), self.yaw, self.pitch)
            self.say("checkpoint saved")
        elif sym == key.Y and self.checkpoint:
            raw, cy, cp = self.checkpoint
            ctypes.memmove(ctypes.byref(self.st), raw, ctypes.sizeof(self.st))
            self.yaw, self.pitch = cy, cp
            self.sync_pos(); self.say("checkpoint loaded")
        elif sym == key.BRACKETLEFT:
            self.settings["sens"] = max(0.1, round(self.settings["sens"] - 0.1, 2))
            self.say(f'sens {self.settings["sens"]}')
        elif sym == key.BRACKETRIGHT:
            self.settings["sens"] = round(self.settings["sens"] + 0.1, 2)
            self.say(f'sens {self.settings["sens"]}')
        elif sym == key.F3:
            self.third_person = not self.third_person
        elif sym == key.F4:
            self.settings["show_triggers"] = not self.settings["show_triggers"]
        elif sym == key.G:
            self.frozen = not self.frozen
            if self.frozen:
                self.st.velocity[0] = self.st.velocity[1] = self.st.velocity[2] = 0.0
            self.say("frozen (G to release)" if self.frozen else "unfrozen")
        elif sym == key.J:
            self.noclip = not self.noclip
            self.st.velocity[0] = self.st.velocity[1] = self.st.velocity[2] = 0.0
            self.st.onground = -1
            self.say("noclip on (WASD fly, Space/Ctrl up/down)"
                     if self.noclip else "noclip off")

    def say(self, text):
        self.msg, self.msg_until = text, time.perf_counter() + 2.0

    # ---- sim ---------------------------------------------------------------
    def respawn(self, idx):
        ctypes.memset(ctypes.byref(self.st), 0, ctypes.sizeof(self.st))
        if self.spawns:
            origin, syaw = self.spawns[idx]
            self.st.origin[0], self.st.origin[1], self.st.origin[2] = map(float, origin)
            self.yaw = float(syaw) % 360.0
        self.st.onground = -1
        self.pitch = 0.0
        self.sync_pos()

    def sync_pos(self):
        self.cur_pos = [self.st.origin[0], self.st.origin[1], self.st.origin[2]]
        self.prev_pos = list(self.cur_pos)

    # ---- replay ------------------------------------------------------------
    @staticmethod
    def _load_replay(path, ep):
        import json as _json
        episodes, rows, hdr = [], [], {}
        for line in open(path, encoding="utf-8"):
            o = _json.loads(line)
            if isinstance(o, dict) and "map" in o:
                rows, hdr = [], o
            elif isinstance(o, list):
                rows.append(o)
            elif isinstance(o, dict) and "end" in o and rows:
                episodes.append((rows, hdr))
                rows = []
        if rows:                                   # a file cut before its end stamp
            episodes.append((rows, hdr))
        if not episodes:
            raise SystemExit("no episodes in replay file")
        if not 1 <= ep <= len(episodes):
            raise SystemExit(f"--ep {ep} out of range 1..{len(episodes)}")
        return episodes[ep - 1]

    @staticmethod
    def _curtain_crossing(rows, zone, start=0):
        """fractional row index at which the track crosses a thin zone
        curtain (y plane) inside its x/z rectangle (+64 u pad), or None -
        the same rule tools/demo/compare_wr.py times the record with."""
        if not zone:
            return None
        mn, mx = zone["mins"], zone["maxs"]
        yc, pad = 0.5 * (mn[1] + mx[1]), 64.0
        for i in range(max(1, start), len(rows)):
            a, b = rows[i - 1], rows[i]
            y0, y1 = a[2] - yc, b[2] - yc
            if (y0 < 0 <= y1) or (y0 > 0 >= y1):
                f = y0 / (y0 - y1) if y0 != y1 else 0.0
                x = a[1] + f * (b[1] - a[1])
                z = a[3] + f * (b[3] - a[3])
                if mn[0] - pad <= x <= mx[0] + pad and mn[2] - pad <= z <= mx[2] + pad:
                    return i - 1 + f
        return None

    def _build_keys(self):
        """Input keycaps for replay mode — same cluster as the POV videos."""
        from pyglet import shapes
        self.keys_batch = pyglet.graphics.Batch()
        k, gap = 44, 6
        W = self.window.width
        x0 = W - 3 * k - 2 * gap - 24
        y0 = 96
        defs = {
            "W": (x0 + k + gap, y0 + k + gap, k, k),
            "A": (x0, y0, k, k),
            "S": (x0 + k + gap, y0, k, k),
            "D": (x0 + 2 * (k + gap), y0, k, k),
            "SPACE": (x0 - 74, y0 - 38, 2 * k + gap + 74, 28),
            "DUCK": (x0 + 2 * (k + gap), y0 - 38, k + 22, 28),
        }
        self.key_rects, self.key_labels = {}, {}
        for name, (x, y, w, h) in defs.items():
            r = shapes.Rectangle(x, y, w, h, color=(60, 60, 60),
                                 batch=self.keys_batch)
            r.opacity = 140
            self.key_rects[name] = r
            self.key_labels[name] = pyglet.text.Label(
                name, x=x + w // 2, y=y + h // 2, anchor_x="center",
                anchor_y="center", font_size=13 if len(name) == 1 else 10,
                weight="bold", color=(200, 200, 200, 255), batch=self.keys_batch)

    def _update_keys(self, fwd, side, jump, duck):
        on = {"W": fwd == 2, "S": fwd == 0, "A": side == 0, "D": side == 2,
              "SPACE": jump, "DUCK": duck}
        for name, r in self.key_rects.items():
            if on[name]:
                r.color = (255, 200, 60)
                r.opacity = 235
                self.key_labels[name].color = (25, 25, 25, 255)
            else:
                r.color = (60, 60, 60)
                r.opacity = 140
                self.key_labels[name].color = (200, 200, 200, 255)

    def _apply_replay(self, i, frac):
        r0 = self.replay[i]
        r1 = self.replay[min(i + 1, len(self.replay) - 1)]
        self.prev_pos = [r0[1], r0[2], r0[3]]
        self.cur_pos = [r1[1], r1[2], r1[3]]
        self.acc = frac * TICK_S
        d = ((r1[7] - r0[7] + 540.0) % 360.0) - 180.0
        self.yaw = (r0[7] + d * frac) % 360.0
        p0 = r0[12] if len(r0) > 12 else 0.0
        p1 = r1[12] if len(r1) > 12 else 0.0
        # trajectory pitch is +up; the client camera convention is +down
        self.pitch = -(p0 + (p1 - p0) * frac)
        for k in range(3):
            self.st.origin[k] = r0[k + 1]
            self.st.velocity[k] = r0[k + 4]
        self.st.yaw = r0[7]
        self.st.ducked = 1 if int(r0[8]) & IN_DUCK else 0
        self.st.induck = 0
        self.st.onground = 0 if int(r0[9]) else -1
        self.st.fuser2 = 0.0
        if hasattr(self, "key_rects"):
            btn = int(r0[8])
            fwd = int(r0[13]) if len(r0) > 13 else None
            side = int(r0[14]) if len(r0) > 14 else None
            self._update_keys(fwd, side, bool(btn & IN_JUMP), bool(btn & IN_DUCK))

    def _replay_update(self, dt):
        n = len(self.replay)
        self.replay_i += 1.0 if self.args.dump else dt * (1000.0 / self.tick_ms)
        if self.replay_i >= n - 1:
            if self.args.dump:
                self._apply_replay(n - 2, 1.0)
                pyglet.clock.schedule_once(lambda _dt: self._finish_dump(), 0.0)
                return
            self.replay_i = 0.0            # loop forever in live mode
        i = int(self.replay_i)
        self._apply_replay(i, self.replay_i - i)

    def _dump_step(self, dt):
        self.update(0.0)
        self._draw_frame(0.0)

    def _start_dump(self):
        import shutil
        import subprocess
        ff = shutil.which("ffmpeg")
        if not ff:
            raise SystemExit("--dump needs ffmpeg on PATH")
        w, h = self.window.width, self.window.height
        self._ffmpeg = subprocess.Popen(
            [ff, "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", f"{1000.0 / self.tick_ms:.4f}", "-i", "-",
             "-vf", "vflip", "-c:v", "libx264", "-preset", "fast",
             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             self.args.dump],
            stdin=subprocess.PIPE)

    def _finish_dump(self):
        if self._ffmpeg is not None:
            self._ffmpeg.stdin.close()
            self._ffmpeg.wait()
            self._ffmpeg = None
            print(f"replay video -> {self.args.dump}")
        pyglet.app.exit()

    def update(self, dt):
        if self.replay is not None:
            self._replay_update(dt)
            return
        if self.paused:
            self.acc = 0.0
            return
        self.acc = min(self.acc + dt, 0.1)
        k = self.keys
        fmove = 400.0 * (bool(k[key.W]) - bool(k[key.S]))
        smove = 400.0 * (bool(k[key.D]) - bool(k[key.A]))
        ticks = 0
        while self.acc >= TICK_S and ticks < 12:
            if self.frozen or self.noclip:
                # practice tools: physics and triggers fully bypassed
                if self.noclip:
                    yr = math.radians(self.yaw)
                    pr = math.radians(self.pitch)
                    fx = math.cos(pr) * math.cos(yr)
                    fy = math.cos(pr) * math.sin(yr)
                    fz = -math.sin(pr)                    # client pitch+ = down
                    fw = fmove / 400.0
                    sd = smove / 400.0
                    up = (1.0 if k[key.SPACE] else 0.0) - \
                         (1.0 if (k[key.LCTRL] or k[key.RCTRL]) else 0.0)
                    vx = fx * fw + math.sin(yr) * sd
                    vy = fy * fw - math.cos(yr) * sd
                    vz = fz * fw + up
                    n = math.sqrt(vx * vx + vy * vy + vz * vz)
                    if n > 1e-6:
                        spd = 900.0 * TICK_S / n
                        self.st.origin[0] += vx * spd
                        self.st.origin[1] += vy * spd
                        self.st.origin[2] += vz * spd
                self.st.velocity[0] = self.st.velocity[1] = self.st.velocity[2] = 0.0
                self.st.onground = -1
                self.prev_pos = self.cur_pos
                self.cur_pos = [self.st.origin[0], self.st.origin[1],
                                self.st.origin[2]]
                self.acc -= TICK_S
                ticks += 1
                self.tick_meter[0] += 1
                continue
            buttons = 0
            if k[key.SPACE] or self.wheel_jump_ticks > 0:
                buttons |= IN_JUMP
            if self.wheel_jump_ticks > 0:
                self.wheel_jump_ticks -= 1
            if k[key.LCTRL] or k[key.RCTRL]:
                buttons |= IN_DUCK
            if self.args.selftest:
                self.yaw = (self.yaw + 0.8) % 360.0
                smove = -400.0
            if k[key.H]:
                # hook: pull along the view direction (physics still runs, so
                # walls stop you — release to fly ballistic with the speed)
                yr = math.radians(self.yaw)
                pr = math.radians(self.pitch)
                cp = math.cos(pr)
                self.st.velocity[0] = cp * math.cos(yr) * 1000.0
                self.st.velocity[1] = cp * math.sin(yr) * 1000.0
                self.st.velocity[2] = -math.sin(pr) * 1000.0
                self.st.onground = -1
            flags = self.core.play_step(self.st, self.yaw, self.pitch,
                                        fmove, smove, buttons, TICK_MS)
            self.prev_pos = self.cur_pos
            self.cur_pos = [self.st.origin[0], self.st.origin[1], self.st.origin[2]]
            self.in_water = bool(flags & SurfCore.PLAY_IN_WATER)
            if flags & SurfCore.PLAY_KILLED:
                o = self.st.origin
                self.say(f"killed at ({o[0]:.0f} {o[1]:.0f} {o[2]:.0f})"
                         " — back to start")
                print(f"KILLED at ({o[0]:.1f}, {o[1]:.1f}, {o[2]:.1f}) "
                      f"vel ({self.st.velocity[0]:.0f}, "
                      f"{self.st.velocity[1]:.0f}, {self.st.velocity[2]:.0f})")
                self.respawn(0)
            elif flags & SurfCore.PLAY_TELEPORTED:
                if self.settings["teleport_respawn"]:
                    self.say("teleporter — back to start"); self.respawn(0)
                else:
                    self.pitch = 0.0
                    self.yaw = self.st.yaw
                    self.sync_pos()
            self.acc -= TICK_S
            ticks += 1
            self.tick_meter[0] += 1
        now = time.perf_counter()
        if now - self.tick_meter[1] >= 0.5:
            self.tick_meter[2] = self.tick_meter[0] / (now - self.tick_meter[1])
            self.tick_meter[0], self.tick_meter[1] = 0, now

    # ---- render ------------------------------------------------------------
    def eye_height(self):
        """PM_VEC_VIEW 17 / PM_VEC_DUCK_VIEW 12, with the engine's 400 ms
        spline transition (incl. its authentic below-12 dip) while in-duck."""
        st = self.st
        if st.ducked:
            return 12.0
        if st.induck:
            t = max(0.0, 1.0 - st.duck_time / 1000.0)
            v = min(t * (1.0 / 0.4), 1.0)
            f = 3 * v * v - 2 * v * v * v
            return (12.0 - 18.0) * f + 17.0 * (1.0 - f)
        return 17.0

    def camera_mvp(self):
        a = max(0.0, min(1.0, self.acc / TICK_S))
        px = self.prev_pos[0] + (self.cur_pos[0] - self.prev_pos[0]) * a
        py = self.prev_pos[1] + (self.cur_pos[1] - self.prev_pos[1]) * a
        pz = self.prev_pos[2] + (self.cur_pos[2] - self.prev_pos[2]) * a + self.eye_height()
        yr, pr = math.radians(self.yaw), math.radians(self.pitch)
        f = Vec3(math.cos(pr) * math.cos(yr), math.cos(pr) * math.sin(yr), -math.sin(pr))
        eye = Vec3(px, py, pz)
        if self.third_person:
            eye = eye - f * 160.0 + Vec3(0, 0, 24)
        view = Mat4.look_at(eye, eye + f, Vec3(0.0, 0.0, 1.0))
        aspect = self.window.width / max(1, self.window.height)
        fovy = math.degrees(2.0 * math.atan(math.tan(math.radians(self.args.fov) / 2.0) * 0.75))
        proj = Mat4.perspective_projection(aspect, 4.0, 16384.0, fovy)
        return proj @ view

    def on_draw(self):
        self.window.clear()
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        mvp = self.camera_mvp()
        self.prog.use()
        self.prog["u_mvp"] = mvp
        self.world_vl.draw(GL_TRIANGLES)
        # opaque brush entities slightly offset toward the viewer so entity faces
        # coplanar with world geometry (jail walls) never z-fight
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-0.6, -1.0)
        for vl in self.solid_vls:
            vl.draw(GL_TRIANGLES)
        glDisable(GL_POLYGON_OFFSET_FILL)
        self.prog.stop()

        translucent = list(self.scenery_vls)             # water/ladders always visible
        if self.settings["show_triggers"]:
            translucent += self.overlay_vls + self.teleport_vls
        if translucent:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(False)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-1.2, -2.0)     # water/trigger planes beat coplanar world faces
            self.prog_flat.use()
            self.prog_flat["u_mvp"] = mvp
            for alpha, vl in translucent:
                self.prog_flat["u_alpha"] = alpha
                vl.draw(GL_TRIANGLES)
            self.prog_flat.stop()
            glDisable(GL_POLYGON_OFFSET_FILL)
            glDepthMask(True)

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        v = self.st.velocity
        hs = math.hypot(v[0], v[1])
        self.lbl_speed.text = f"{hs:5.0f}"
        self.lbl_speed3d.text = f"{math.sqrt(v[0]**2 + v[1]**2 + v[2]**2):5.0f}"
        stam = f"  stamina {self.st.fuser2:5.0f}" if self.st.fuser2 > 0 else ""
        self.lbl_info.text = (
            f"fps {self.draw_meter[2]:5.0f}   tick {self.tick_meter[2]:5.1f} Hz\n"
            f"pos ({self.st.origin[0]:7.1f} {self.st.origin[1]:7.1f} {self.st.origin[2]:7.1f})"
            f"   vz {v[2]:7.1f}   {'GROUND' if self.st.onground != -1 else 'air'}"
            f"{'+DUCK' if self.st.ducked else ''}"
            f"{'+WATER' if getattr(self, 'in_water', False) else ''}{stam}\n"
            f'sens {self.settings["sens"]}  aa {self.args.aa:g}  R start  N spawns  Esc settings'
        )
        self.lbl_msg.text = self.msg if time.perf_counter() < self.msg_until else ""
        if self.replay is not None and hasattr(self, "timer_start"):
            i = min(self.replay_i, float(self.timer_end))
            t = max(0.0, i - self.timer_start) * self.tick_ms / 1000.0
            done = self.replay_i >= self.timer_end - (1 if self.clock_mode == "cliff" else 0)
            txt = f"{int(t // 60)}:{t % 60:06.3f}"
            if getattr(self.args, "wr", None):
                wr = float(self.args.wr)
                txt += f"\nWR {int(wr // 60)}:{wr % 60:05.2f}"
                if done:
                    txt += f"  {t - wr:+.2f}"
            self.lbl_timer.text = txt
            self.lbl_timer.color = ((120, 255, 140, 255) if done
                                    else (255, 255, 255, 235))
        else:
            self.lbl_timer.text = ""
        self.window.projection = Mat4.orthogonal_projection(
            0, self.window.width, 0, self.window.height, -1, 1)
        self.window.view = Mat4()
        self.hud.draw()
        if self.replay is not None and hasattr(self, "keys_batch"):
            self.keys_batch.draw()
        if self.paused and self.menu_batch:
            self.menu_batch.draw()

    def end_selftest(self, dt):
        v = self.st.velocity
        hs = math.hypot(v[0], v[1])
        print(f"SELFTEST: ran {self.st.tick} physics ticks in ~2s, "
              f"pos ({self.st.origin[0]:.0f} {self.st.origin[1]:.0f} {self.st.origin[2]:.0f}), "
              f"h-speed {hs:.1f} u/s, fps {self.draw_meter[2]:.0f}")
        ok = 150 < self.st.tick < 260
        print("SELFTEST OK" if ok else "SELFTEST FAIL: tick count off")
        pyglet.app.exit()
        if not ok:
            sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="maps/surf_ski_2.bsp")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--sens", type=float, default=None, help="overrides saved setting")
    ap.add_argument("--fov", type=float, default=90.0, help="horizontal 4:3 fov, CS-style")
    ap.add_argument("--aa", type=float, default=100.0, help="sv_airaccelerate")
    ap.add_argument("--gravity", type=float, default=800.0)
    ap.add_argument("--maxvel", type=float, default=4000.0,
                    help="sv_maxvelocity — GoldSrc default is 2000, but real "
                         "surf servers raise it (the user's reference server "
                         "runs 4000); the per-axis clamp bleeds speed on any "
                         "fast line (measured: ramp2 of cannonball ends at "
                         "1750 u/s capped vs 2250 live)")
    ap.add_argument("--no-stamina", action="store_true")
    ap.add_argument("--no-bhop-cap", action="store_true")
    ap.add_argument("--display", choices=["windowed", "borderless", "fullscreen"],
                    default=None, help="overrides saved display mode")
    ap.add_argument("--fullscreen", action="store_true", help="alias for --display fullscreen")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--replay", default=None,
                    help="play back a recorded trajectory (.jsonl) instead of "
                         "simulating — first-person through the game client")
    ap.add_argument("--ep", type=int, default=1, help="replay episode (1-based)")
    ap.add_argument("--clock", choices=("cliff", "zone"), default="cliff",
                    help="replay run clock: 'cliff' starts at the start-cliff drop (legacy); "
                         "'zone' is the leaderboard's clock, start curtain to finish "
                         "curtain from maps/<map>.zones.json")
    ap.add_argument("--wr", type=float, default=None,
                    help="human record in seconds, shown under the run clock with the delta")
    ap.add_argument("--dump", default=None,
                    help="with --replay: render offline to this .mp4 (100fps)")
    args = ap.parse_args()
    if args.replay and "--map" not in sys.argv:
        # replays must run on the map they were recorded on: take it from the
        # run.json next to the trajectory when present
        rj = Path(args.replay).resolve().parent / "run.json"
        if rj.exists():
            try:
                m = json.loads(rj.read_text(encoding="utf-8")) \
                    .get("config", {}).get("map")
                if m:
                    args.map = f"maps/{m}.bsp"
            except (OSError, ValueError):
                pass
    if args.fullscreen and args.display is None:
        args.display = "fullscreen"
    app = PlayApp(args)
    pyglet.app.run(None)      # no pyglet-scheduled redraws — we pace frames ourselves
    if not args.selftest and not args.replay:   # experiments must not overwrite settings
        app.save_settings()


if __name__ == "__main__":
    main()
