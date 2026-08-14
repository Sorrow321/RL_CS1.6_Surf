#!/usr/bin/env python3
"""export_map.py -- GoldSrc BSP v30 -> viewer mesh JSON.

Reads the render lumps of a v30 BSP (see docs/02-bsp-collision.md section 1)
and writes one JSON file the three.js viewer consumes:

    {"map": name,
     "world":   {"positions": [...], "normals": [...], "indices": [...]},
     "brushes": [{"classname", "model", "targetname", "target",
                  "positions", "normals", "indices"}, ...],
     "markers": [{"classname", "origin", "yaw", "targetname"}, ...],
     "bounds":  {"mins": [...], "maxs": [...]}}

Positions are raw map units, Z-up (the viewer converts to Y-up at load).
Pure stdlib; usage:  python tools/export_map.py maps/surf_ski_2.bsp [out.json]
"""

import json
import os
import struct
import sys

HEADER_LUMPS = 15
LUMP_ENTITIES = 0
LUMP_PLANES = 1
LUMP_TEXTURES = 2
LUMP_VERTEXES = 3
LUMP_TEXINFO = 6
LUMP_FACES = 7
LUMP_EDGES = 12
LUMP_SURFEDGES = 13
LUMP_MODELS = 14

# World faces whose texture name starts with one of these are not rendered.
SKIP_PREFIXES = ("sky", "aaatrigger", "clip", "null", "origin", "skip", "hint")

MARKER_CLASSES = (
    "info_player_start",
    "info_player_deathmatch",
    "info_teleport_destination",
    "info_target",
)


def read_lumps(data):
    version = struct.unpack_from("<i", data, 0)[0]
    if version != 30:
        raise SystemExit("not a BSP v30 file (version=%d)" % version)
    lumps = []
    for i in range(HEADER_LUMPS):
        ofs, ln = struct.unpack_from("<ii", data, 4 + i * 8)
        lumps.append((ofs, ln))
    return lumps


def lump_bytes(data, lumps, idx):
    ofs, ln = lumps[idx]
    return data[ofs:ofs + ln]


def parse_planes(raw):
    n = len(raw) // 20
    out = []
    for i in range(n):
        nx, ny, nz, dist, ptype = struct.unpack_from("<ffffi", raw, i * 20)
        out.append((nx, ny, nz))
    return out


def parse_textures(raw):
    """Miptex directory -> list of lowercase texture names by miptex index."""
    if len(raw) < 4:
        return []
    count = struct.unpack_from("<i", raw, 0)[0]
    names = []
    for i in range(count):
        ofs = struct.unpack_from("<i", raw, 4 + i * 4)[0]
        if ofs < 0 or ofs + 16 > len(raw):
            names.append("")
            continue
        name = raw[ofs:ofs + 16].split(b"\x00", 1)[0]
        names.append(name.decode("latin-1", "replace").lower())
    return names


def parse_vertexes(raw):
    n = len(raw) // 12
    return [struct.unpack_from("<fff", raw, i * 12) for i in range(n)]


def parse_texinfo(raw):
    n = len(raw) // 40
    return [struct.unpack_from("<i", raw, i * 40 + 32)[0] for i in range(n)]  # _miptex


def parse_faces(raw):
    n = len(raw) // 20
    out = []
    for i in range(n):
        planenum, side, firstedge, numedges, texinfo = struct.unpack_from(
            "<hhihh", raw, i * 20)
        out.append((planenum, side, firstedge, numedges, texinfo))
    return out


def parse_edges(raw):
    n = len(raw) // 4
    return [struct.unpack_from("<HH", raw, i * 4) for i in range(n)]


def parse_surfedges(raw):
    n = len(raw) // 4
    return list(struct.unpack_from("<%di" % n, raw, 0))


def parse_models(raw):
    n = len(raw) // 64
    out = []
    for i in range(n):
        vals = struct.unpack_from("<9f4i i ii", raw, i * 64)
        out.append({
            "mins": vals[0:3],
            "maxs": vals[3:6],
            "origin": vals[6:9],
            "headnode": vals[9:13],
            "visleafs": vals[13],
            "firstface": vals[14],
            "numfaces": vals[15],
        })
    return out


def parse_entities(raw):
    """LUMP_ENTITIES text -> list of {key: value} dicts."""
    text = raw.split(b"\x00", 1)[0].decode("latin-1", "replace")
    ents = []
    i = 0
    n = len(text)
    while True:
        i = text.find("{", i)
        if i < 0:
            break
        ent = {}
        i += 1
        while i < n:
            # next token: closing brace or quoted key
            while i < n and text[i] in " \t\r\n":
                i += 1
            if i >= n or text[i] == "}":
                i += 1
                break
            if text[i] != '"':      # malformed junk; skip a char
                i += 1
                continue
            j = text.find('"', i + 1)
            if j < 0:
                i = n
                break
            key = text[i + 1:j]
            i = text.find('"', j + 1)
            if i < 0:
                break
            j = text.find('"', i + 1)
            if j < 0:
                break
            ent[key] = text[i + 1:j]
            i = j + 1
        if ent:
            ents.append(ent)
    return ents


def parse_vec3(s):
    try:
        parts = [float(x) for x in s.split()]
        while len(parts) < 3:
            parts.append(0.0)
        return parts[:3]
    except ValueError:
        return [0.0, 0.0, 0.0]


def entity_yaw(ent):
    if "angles" in ent:
        return parse_vec3(ent["angles"])[1]   # "pitch yaw roll"
    if "angle" in ent:
        try:
            return float(ent["angle"])
        except ValueError:
            return 0.0
    return 0.0


class MeshBuilder(object):
    def __init__(self):
        self.positions = []
        self.normals = []
        self.indices = []
        self.faces = 0
        self.tris = 0

    def add_face(self, poly, normal):
        """poly: list of (x,y,z); fan-triangulated, flat per-vertex normal."""
        if len(poly) < 3:
            return
        base = len(self.positions) // 3
        for p in poly:
            self.positions.extend((round(p[0], 3), round(p[1], 3), round(p[2], 3)))
            self.normals.extend((round(normal[0], 5), round(normal[1], 5),
                                 round(normal[2], 5)))
        for k in range(1, len(poly) - 1):
            self.indices.extend((base, base + k, base + k + 1))
            self.tris += 1
        self.faces += 1

    def to_dict(self):
        return {"positions": self.positions,
                "normals": self.normals,
                "indices": self.indices}


def face_polygon(face, edges, surfedges, vertexes):
    planenum, side, firstedge, numedges, texinfo = face
    poly = []
    for i in range(firstedge, firstedge + numedges):
        se = surfedges[i]
        if se >= 0:
            v = edges[se][0]
        else:
            v = edges[-se][1]
        poly.append(vertexes[v])
    return poly


def defight_coplanar(world):
    """Kill world-vs-world z-fighting IN THE BAKED MESH (mirrors play.py's
    runtime fix): reconstruct fan faces (consecutive tris sharing their first
    index), group horizontal faces by exact plane z, lift each overlapping
    face 0.2u per rank (largest keeps the true plane). surf_ski_2 stacks 118
    coplanar faces on the arena floor. Idempotent: lifted faces leave the
    exact-z group on re-export."""
    pos, nrm, idx = world["positions"], world["normals"], world["indices"]
    faces = []
    t, ntri = 0, len(idx) // 3
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
        groups.setdefault((round(pos[3 * a + 2], 2), nz > 0), []).append(fi)
    lifted = 0
    for (_z, up), fis in groups.items():
        if len(fis) < 2:
            continue

        def area(fi):
            t0, n, _a = faces[fi]
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
            t0, n, _a = faces[fi]
            seen = set()
            for tt in range(t0, t0 + n):
                for v in (idx[3*tt], idx[3*tt+1], idx[3*tt+2]):
                    if v not in seen:
                        seen.add(v)
                        pos[3 * v + 2] = round(pos[3 * v + 2] + dz, 3)
            lifted += 1
    return lifted


def main():
    if len(sys.argv) < 2:
        print("usage: python tools/export_map.py <map.bsp> [out.json]")
        return 1
    bsp_path = sys.argv[1]
    map_name = os.path.splitext(os.path.basename(bsp_path))[0]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        root, "viewer", "assets", map_name + ".mesh.json")

    with open(bsp_path, "rb") as f:
        data = f.read()

    lumps = read_lumps(data)
    planes = parse_planes(lump_bytes(data, lumps, LUMP_PLANES))
    texnames = parse_textures(lump_bytes(data, lumps, LUMP_TEXTURES))
    vertexes = parse_vertexes(lump_bytes(data, lumps, LUMP_VERTEXES))
    texinfos = parse_texinfo(lump_bytes(data, lumps, LUMP_TEXINFO))
    faces = parse_faces(lump_bytes(data, lumps, LUMP_FACES))
    edges = parse_edges(lump_bytes(data, lumps, LUMP_EDGES))
    surfedges = parse_surfedges(lump_bytes(data, lumps, LUMP_SURFEDGES))
    models = parse_models(lump_bytes(data, lumps, LUMP_MODELS))
    entities = parse_entities(lump_bytes(data, lumps, LUMP_ENTITIES))

    def face_normal(face):
        n = planes[face[0]]
        if face[1] != 0:
            return (-n[0], -n[1], -n[2])
        return n

    def face_texname(face):
        ti = face[4]
        if 0 <= ti < len(texinfos):
            mt = texinfos[ti]
            if 0 <= mt < len(texnames):
                return texnames[mt]
        return ""

    # ---- world (model 0), with skip-texture filter ----
    world = MeshBuilder()
    skipped = 0
    m0 = models[0]
    for fi in range(m0["firstface"], m0["firstface"] + m0["numfaces"]):
        face = faces[fi]
        if face_texname(face).startswith(SKIP_PREFIXES):
            skipped += 1
            continue
        world.add_face(face_polygon(face, edges, surfedges, vertexes),
                       face_normal(face))

    # ---- brush entities (model "*N"), all faces rendered ----
    brushes = []
    brush_candidates = 0
    for ent in entities:
        model = ent.get("model", "")
        if not model.startswith("*"):
            continue
        brush_candidates += 1
        classname = ent.get("classname", "")
        if not classname:
            continue
        try:
            mi = int(model[1:])
        except ValueError:
            continue
        if not (0 < mi < len(models)):
            continue
        mb = MeshBuilder()
        m = models[mi]
        for fi in range(m["firstface"], m["firstface"] + m["numfaces"]):
            face = faces[fi]
            mb.add_face(face_polygon(face, edges, surfedges, vertexes),
                        face_normal(face))
        brush = {"classname": classname, "model": mi,
                 "targetname": ent.get("targetname", ""),
                 "target": ent.get("target", ""),
                 "skin": int(float(ent.get("skin", 0) or 0))}
        brush.update(mb.to_dict())
        brush["_faces"] = mb.faces   # stripped below; used for the summary
        brush["_tris"] = mb.tris
        brushes.append(brush)

    # ---- point markers ----
    markers = []
    for ent in entities:
        cn = ent.get("classname", "")
        if cn in MARKER_CLASSES and "origin" in ent:
            markers.append({"classname": cn,
                            "origin": parse_vec3(ent["origin"]),
                            "yaw": entity_yaw(ent),
                            "targetname": ent.get("targetname", "")})

    world_dict = world.to_dict()
    lifted = defight_coplanar(world_dict)
    print("coplanar de-fight: lifted %d overlapping horizontal faces" % lifted)

    out = {
        "map": map_name,
        "world": world_dict,
        "brushes": brushes,
        "markers": markers,
        "bounds": {"mins": list(m0["mins"]), "maxs": list(m0["maxs"])},
    }

    # ---- summary ----
    print("map: %s" % map_name)
    print("world: %d faces (%d skipped by texture), %d tris, %d verts"
          % (world.faces, skipped, world.tris, len(world.positions) // 3))
    per_class = {}
    for b in brushes:
        f, t = per_class.get(b["classname"], (0, 0))
        per_class[b["classname"]] = (f + b["_faces"], t + b["_tris"])
    count_class = {}
    for b in brushes:
        count_class[b["classname"]] = count_class.get(b["classname"], 0) + 1
    print("brush entities: %d emitted of %d candidates"
          % (len(brushes), brush_candidates))
    for cn in sorted(per_class):
        f, t = per_class[cn]
        print("  %-20s x%-3d %4d faces %5d tris"
              % (cn, count_class[cn], f, t))
    print("markers: %d" % len(markers))
    mk = {}
    for m in markers:
        mk[m["classname"]] = mk.get(m["classname"], 0) + 1
    for cn in sorted(mk):
        print("  %-28s x%d" % (cn, mk[cn]))
    print("bounds: mins=%s maxs=%s" % (out["bounds"]["mins"], out["bounds"]["maxs"]))

    for b in brushes:
        del b["_faces"]
        del b["_tris"]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
