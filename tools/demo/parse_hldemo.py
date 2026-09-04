"""parse_hldemo.py - read a GoldSrc / CS 1.6 HLDEMO (.dem) and dump it in the
trajectory format this repo already uses (python/surfgym/record.py).

Layout parsed (demo protocol 5, network protocol 46-48):

    header (544 bytes): "HLDEMO\\0\\0", demo protocol, net protocol,
        map name[260], game dir[260], map CRC, directory offset
    directory: count, then 92-byte entries (type, description[64], flags,
        cd track, track time, frame count, offset, length).  A demo that was
        never closed cleanly has offset 0; then the frame stream is scanned
        from byte 544 and split at NextSection frames.
    frames: type u8, time f32, frame i32, then per type
        0/1 netmsg : demoinfo (436 bytes = timestamp, ref_params_t 232,
                     usercmd_t 52, movevars_t 132, view 12, viewmodel 4),
                     7 sequence ints, int length + network payload
        2 start, 3 console command (64 chars), 4 client data (origin,
        viewangles, weaponbits, fov), 5 next section, 6 event (84 bytes),
        7 weapon anim (8), 8 sound (channel, len+sample, 16 bytes),
        9 demo buffer (len + bytes)

Per netmsg frame we keep ref_params.simorg / simvel (the predicted player
origin and velocity, hull centre like our SurfState.origin), viewangles,
onground, waterlevel, health and the usercmd (msec, viewangles, forward /
side / up move, buttons).  The movevars block is the server's physics.

Outputs (in --out, named after --stem or the demo file):

    <stem>.jsonl         our trajectory format, resampled at --tick-ms:
                         [t, x,y,z, vx,vy,vz, yaw, buttons, onground,
                          progress, reward, pitch, fwd, side]
                         buttons keeps the HL bitmask (IN_JUMP=2, IN_DUCK=4
                         are the bits record.py writes); fwd/side are the
                         0/1/2 move bins of record.py; pitch is converted
                         to the simulator's sign (NEGATIVE = looking down;
                         GoldSrc viewangles have positive = down); progress
                         and reward are 0 (unknown for a demo).
    <stem>.frames.npz    raw per-frame arrays for finer analysis
    <stem>.movevars.json header, movevars, frame statistics and a diff
                         against the simulator's physics defaults

Usage:
    python tools/demo/parse_hldemo.py <demo.dem> --out <dir> [--stem NAME]
                                      [--tick-ms 10] [--section N]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

HEADER_SIZE = 544
DIR_ENTRY_SIZE = 92
DEMOINFO_SIZE = 436          # timestamp + ref_params + usercmd + movevars + view + viewmodel
REF_PARAMS_OFF = 4
USERCMD_OFF = REF_PARAMS_OFF + 232
MOVEVARS_OFF = USERCMD_OFF + 52
NETMSG_TAIL = 7 * 4          # sequence ints between demoinfo and the payload length

MOVEVAR_NAMES = [
    "gravity", "stopspeed", "maxspeed", "spectatormaxspeed", "accelerate",
    "airaccelerate", "wateraccelerate", "friction", "edgefriction",
    "waterfriction", "entgravity", "bounce", "stepsize", "maxvelocity",
    "zmax", "waveHeight",
]

# HL usercmd buttons
IN_ATTACK, IN_JUMP, IN_DUCK, IN_FORWARD, IN_BACK, IN_USE = 1, 2, 4, 8, 16, 32
IN_LEFT, IN_RIGHT, IN_MOVELEFT, IN_MOVERIGHT = 128, 256, 512, 1024

# what python/surfgym/core.py runs with (the trainer's defaults)
SIM_PHYS = {
    "gravity": 800.0, "airaccelerate": 100.0, "accelerate": 5.0,
    "friction": 4.0, "edgefriction": 2.0, "stopspeed": 75.0,
    "maxspeed": 320.0, "maxvelocity": 4000.0, "stepsize": 18.0, "bounce": 1.0,
}


def read_header(data: bytes) -> dict:
    if data[:6] != b"HLDEMO":
        raise ValueError("not an HLDEMO file (magic %r)" % data[:8])
    demo_proto, net_proto = struct.unpack_from("<ii", data, 8)
    map_name = data[16:276].split(b"\0")[0].decode("latin-1")
    game_dir = data[276:536].split(b"\0")[0].decode("latin-1")
    map_crc, dir_off = struct.unpack_from("<Ii", data, 536)
    return {
        "demo_protocol": demo_proto, "net_protocol": net_proto,
        "map": map_name, "game_dir": game_dir, "map_crc": map_crc,
        "directory_offset": dir_off, "file_size": len(data),
    }


def read_directory(data: bytes, dir_off: int) -> list:
    if dir_off <= 0 or dir_off + 4 > len(data):
        return []
    n = struct.unpack_from("<i", data, dir_off)[0]
    entries = []
    for i in range(n):
        e = dir_off + 4 + DIR_ENTRY_SIZE * i
        if e + DIR_ENTRY_SIZE > len(data):
            break
        typ = struct.unpack_from("<i", data, e)[0]
        desc = data[e + 4:e + 68].split(b"\0")[0].decode("latin-1")
        flags, cd, ttime, frames, off, length = struct.unpack_from("<iifiii", data, e + 68)
        entries.append({"type": typ, "description": desc, "flags": flags,
                        "cd_track": cd, "track_time": ttime, "frames": frames,
                        "offset": off, "length": length})
    return entries


def parse_frames(data: bytes, start: int, end: int) -> dict:
    """Walk the frame stream in [start, end).  Returns dict of lists."""
    net = {k: [] for k in ("t", "frame", "timestamp", "vieworg", "viewangles",
                           "frametime", "time", "onground", "waterlevel",
                           "simvel", "simorg", "viewheight", "cl_viewangles",
                           "health", "punchangle", "uc_msec", "uc_lerp",
                           "uc_viewangles", "uc_fsu", "uc_buttons",
                           "uc_impulse", "view", "msglen")}
    clientdata = {k: [] for k in ("t", "frame", "origin", "viewangles",
                                  "weaponbits", "fov")}
    commands = []
    movevars = []
    counts = {}
    off = start
    n = 0
    while off + 9 <= end:
        typ = data[off]
        tm, fr = struct.unpack_from("<fi", data, off + 1)
        off += 9
        counts[typ] = counts.get(typ, 0) + 1
        if typ in (0, 1):
            info = off
            if info + DEMOINFO_SIZE + NETMSG_TAIL + 4 > end:
                break
            rp = info + REF_PARAMS_OFF
            net["t"].append(tm)
            net["frame"].append(fr)
            net["timestamp"].append(struct.unpack_from("<f", data, info)[0])
            net["vieworg"].append(struct.unpack_from("<3f", data, rp))
            net["viewangles"].append(struct.unpack_from("<3f", data, rp + 12))
            ft, tt = struct.unpack_from("<2f", data, rp + 60)
            net["frametime"].append(ft)
            net["time"].append(tt)
            _inter, _paused, _spec, og, wl = struct.unpack_from("<5i", data, rp + 68)
            net["onground"].append(og)
            net["waterlevel"].append(wl)
            net["simvel"].append(struct.unpack_from("<3f", data, rp + 88))
            net["simorg"].append(struct.unpack_from("<3f", data, rp + 100))
            net["viewheight"].append(struct.unpack_from("<3f", data, rp + 112))
            net["cl_viewangles"].append(struct.unpack_from("<3f", data, rp + 128))
            net["health"].append(struct.unpack_from("<i", data, rp + 140)[0])
            net["punchangle"].append(struct.unpack_from("<3f", data, rp + 156))
            uc = info + USERCMD_OFF
            lerp, msec = struct.unpack_from("<hB", data, uc)
            net["uc_lerp"].append(lerp)
            net["uc_msec"].append(msec)
            net["uc_viewangles"].append(struct.unpack_from("<3f", data, uc + 4))
            net["uc_fsu"].append(struct.unpack_from("<3f", data, uc + 16))
            _light, buttons, impulse = struct.unpack_from("<bHb", data, uc + 28)
            net["uc_buttons"].append(buttons)
            net["uc_impulse"].append(impulse)
            mv = info + MOVEVARS_OFF
            vals = struct.unpack_from("<16f", data, mv)
            footsteps = struct.unpack_from("<i", data, mv + 64)[0]
            sky = data[mv + 68:mv + 100].split(b"\0")[0].decode("latin-1")
            tail = struct.unpack_from("<8f", data, mv + 100)
            mvd = dict(zip(MOVEVAR_NAMES, [float(v) for v in vals]))
            mvd.update({"footsteps": footsteps, "skyName": sky,
                        "rollangle": tail[0], "rollspeed": tail[1],
                        "skycolor": list(tail[2:5]), "skyvec": list(tail[5:8])})
            if not movevars or movevars[-1][1] != mvd:
                movevars.append((tm, mvd))
            net["view"].append(struct.unpack_from("<3f", data, info + MOVEVARS_OFF + 132))
            msglen = struct.unpack_from("<i", data, info + DEMOINFO_SIZE + NETMSG_TAIL)[0]
            net["msglen"].append(msglen)
            if msglen < 0 or msglen > (1 << 20):
                raise ValueError("implausible netmsg length %d at frame %d" % (msglen, n))
            off = info + DEMOINFO_SIZE + NETMSG_TAIL + 4 + msglen
        elif typ == 2:
            pass
        elif typ == 3:
            commands.append((tm, fr, data[off:off + 64].split(b"\0")[0].decode("latin-1")))
            off += 64
        elif typ == 4:
            clientdata["t"].append(tm)
            clientdata["frame"].append(fr)
            clientdata["origin"].append(struct.unpack_from("<3f", data, off))
            clientdata["viewangles"].append(struct.unpack_from("<3f", data, off + 12))
            wb, fov = struct.unpack_from("<if", data, off + 24)
            clientdata["weaponbits"].append(wb)
            clientdata["fov"].append(fov)
            off += 32
        elif typ == 5:
            break
        elif typ == 6:
            off += 84
        elif typ == 7:
            off += 8
        elif typ == 8:
            _ch, sl = struct.unpack_from("<ii", data, off)
            off += 8 + sl + 16
        elif typ == 9:
            length = struct.unpack_from("<i", data, off)[0]
            off += 4 + length
        else:
            raise ValueError("unknown frame type %d at byte %d (frame %d)" % (typ, off - 9, n))
        n += 1
    out = {"counts": counts, "frames_walked": n, "end_offset": off,
           "commands": commands, "movevars": movevars}
    for k, v in net.items():
        out["net_" + k] = np.asarray(v)
    for k, v in clientdata.items():
        out["cd_" + k] = np.asarray(v)
    return out


def unwrap_deg(a: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(a)))


def resample(parsed: dict, tick_ms: int) -> tuple:
    """Linear-interpolate the netmsg stream onto a tick_ms grid."""
    t = parsed["net_t"].astype(np.float64)
    if len(t) < 2:
        raise ValueError("no netmsg frames to resample")
    # times must be increasing for interp; drop repeats
    keep = np.concatenate(([True], np.diff(t) > 0))
    t = t[keep]
    org = parsed["net_simorg"][keep].astype(np.float64)
    vel = parsed["net_simvel"][keep].astype(np.float64)
    ang = parsed["net_viewangles"][keep].astype(np.float64)
    og = parsed["net_onground"][keep]
    fsu = parsed["net_uc_fsu"][keep]
    btn = parsed["net_uc_buttons"][keep]
    dt = tick_ms / 1000.0
    grid = np.arange(0.0, t[-1] - t[0] + 1e-9, dt) + t[0]
    cols = [np.interp(grid, t, org[:, i]) for i in range(3)]
    cols += [np.interp(grid, t, vel[:, i]) for i in range(3)]
    yaw = np.interp(grid, t, unwrap_deg(ang[:, 1])) % 360.0
    pitch_hl = np.interp(grid, t, ang[:, 0])
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    fwd = np.where(fsu[idx, 0] > 0, 2, np.where(fsu[idx, 0] < 0, 0, 1))
    side = np.where(fsu[idx, 1] > 0, 2, np.where(fsu[idx, 1] < 0, 0, 1))
    rows = []
    for k in range(len(grid)):
        rows.append([
            k,
            round(float(cols[0][k]), 2), round(float(cols[1][k]), 2), round(float(cols[2][k]), 2),
            round(float(cols[3][k]), 2), round(float(cols[4][k]), 2), round(float(cols[5][k]), 2),
            round(float(yaw[k]), 2),
            int(btn[idx[k]]),
            int(og[idx[k]] != 0),
            0.0,
            0.0,
            round(float(-pitch_hl[k]), 2),   # simulator sign: negative = down
            int(fwd[k]), int(side[k]),
        ])
    return grid, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--stem", default=None, help="output file stem (default: demo file name)")
    ap.add_argument("--tick-ms", type=int, default=10)
    ap.add_argument("--section", type=int, default=None,
                    help="directory entry to dump (default: the Playback entry / the last section)")
    args = ap.parse_args()

    with open(args.demo, "rb") as f:
        data = f.read()
    hdr = read_header(data)
    entries = read_directory(data, hdr["directory_offset"])
    print("header:", json.dumps(hdr))
    for i, e in enumerate(entries):
        print("  dir[%d] %-10s frames=%d time=%.3fs off=%d len=%d" %
              (i, e["description"], e["frames"], e["track_time"], e["offset"], e["length"]))

    sections = []
    if entries:
        for e in entries:
            sections.append((e["description"], e["offset"], e["offset"] + e["length"]))
    else:
        # no directory: scan from the header and split at NextSection
        print("  no directory (demo not closed cleanly): scanning from byte %d" % HEADER_SIZE)
        off = HEADER_SIZE
        k = 0
        while off < len(data):
            p = parse_frames(data, off, len(data))
            sections.append(("section%d" % k, off, p["end_offset"]))
            if p["end_offset"] <= off:
                break
            off = p["end_offset"] + 9    # skip the NextSection frame header
            k += 1

    if args.section is None:
        pick = next((i for i, s in enumerate(sections) if s[0].lower().startswith("playback")), len(sections) - 1)
    else:
        pick = args.section
    name, s0, s1 = sections[pick]
    parsed = parse_frames(data, s0, s1)
    nt = len(parsed["net_t"])
    print("section %d (%s): %d frames walked, %d netmsg frames, types %s" %
          (pick, name, parsed["frames_walked"], nt, parsed["counts"]))
    if nt < 2:
        print("no netmsg frames in this section; nothing to dump")
        return 1

    t = parsed["net_t"]
    dts = np.diff(t)
    msec = parsed["net_uc_msec"]
    org = parsed["net_simorg"]
    vel = parsed["net_simvel"]
    # finite-difference cross-check of simvel.  Alignment (measured, exact):
    #   simorg[k+1] = simorg[k] + simvel[k+1] * uc_msec[k] / 1000
    # i.e. the physics step that produced frame k+1 ran for the msec of the
    # usercmd recorded in frame k, and simvel[k+1] is its post-step velocity.
    step_dt = np.maximum(msec[:-1].astype(np.float64) / 1000.0, 1e-6)
    fd = np.diff(org, axis=0) / step_dt[:, None]
    fd_err = np.linalg.norm(fd - vel[1:], axis=1)
    mv0 = parsed["movevars"][0][1] if parsed["movevars"] else {}

    stem = args.stem or os.path.splitext(os.path.basename(args.demo))[0]
    os.makedirs(args.out, exist_ok=True)
    grid, rows = resample(parsed, args.tick_ms)

    phys = {
        "sv_gravity": mv0.get("gravity"), "sv_airaccelerate": mv0.get("airaccelerate"),
        "sv_accelerate": mv0.get("accelerate"), "sv_friction": mv0.get("friction"),
        "edgefriction": mv0.get("edgefriction"), "sv_stopspeed": mv0.get("stopspeed"),
        "sv_maxspeed": mv0.get("maxspeed"),
        "player_maxspeed": float(np.max(np.abs(parsed["net_uc_fsu"]))) if nt else None,
        "sv_maxvelocity": mv0.get("maxvelocity"), "sv_stepsize": mv0.get("stepsize"),
        "sv_bounce": mv0.get("bounce"), "msec": int(np.median(msec)) if nt else None,
        "enable_stamina": "unknown (not in movevars; CS 1.6 default on)",
        "enable_bhop_cap": "unknown (not in movevars; CS 1.6 default on)",
        "enable_duck": 1,
    }
    header = {
        "map": hdr["map"], "tick_ms": args.tick_ms, "phys": phys, "episode": 0,
        "source": "hldemo", "demo": os.path.basename(args.demo),
        "demo_protocol": hdr["demo_protocol"], "net_protocol": hdr["net_protocol"],
        "game_dir": hdr["game_dir"], "section": name,
        "playback_seconds": float(t[-1] - t[0]), "net_frames": int(nt),
        "t0": float(t[0]),
        "pitch_sign": "simulator convention (negative = down); GoldSrc value negated",
        "buttons": "HL usercmd bitmask (IN_JUMP=2, IN_DUCK=4, IN_FORWARD=8, IN_MOVELEFT=512, IN_MOVERIGHT=1024)",
    }
    jpath = os.path.join(args.out, stem + ".jsonl")
    with open(jpath, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, separators=(",", ":")) + "\n")
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
        f.write(json.dumps({"end": "demo", "ticks": len(rows), "best_progress": 0.0},
                           separators=(",", ":")) + "\n")

    npath = os.path.join(args.out, stem + ".frames.npz")
    np.savez_compressed(npath, **{k[4:]: v for k, v in parsed.items() if k.startswith("net_")},
                        cd_t=parsed["cd_t"], cd_origin=parsed["cd_origin"],
                        cd_viewangles=parsed["cd_viewangles"], cd_fov=parsed["cd_fov"])

    diff = {}
    for k, ours in SIM_PHYS.items():
        theirs = mv0.get(k)
        if theirs is not None and abs(theirs - ours) > 1e-6:
            diff[k] = {"demo": theirs, "sim": ours}
    speed = np.linalg.norm(vel, axis=1)
    summary = {
        "header": hdr, "directory": entries, "section": name,
        "net_frames": int(nt), "playback_seconds": float(t[-1] - t[0]),
        "frame_dt_ms": {"mean": float(dts.mean() * 1000), "min": float(dts.min() * 1000),
                        "max": float(dts.max() * 1000)},
        "usercmd_msec_histogram": {int(k): int(v) for k, v in zip(*np.unique(msec, return_counts=True))},
        "simvel_vs_finite_difference_u_per_s": {"median": float(np.median(fd_err)),
                                                "p95": float(np.percentile(fd_err, 95)),
                                                "max": float(fd_err.max())},
        "movevars": mv0, "movevars_changes": len(parsed["movevars"]) - 1,
        "movevars_vs_sim_defaults": diff,
        "console_commands": [c[2] for c in parsed["commands"]][:200],
        "origin_bounds": {"min": org.min(axis=0).tolist(), "max": org.max(axis=0).tolist()},
        "speed": {"max": float(speed.max()), "mean": float(speed.mean())},
        "abs_velocity_component_max": np.abs(vel).max(axis=0).tolist(),
        "onground_fraction": float(np.mean(parsed["net_onground"] != 0)),
        "health_min": int(parsed["net_health"].min()),
        "outputs": {"jsonl": jpath, "frames_npz": npath},
    }
    mpath = os.path.join(args.out, stem + ".movevars.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    print("netmsg frames %d over %.3f s (mean %.2f ms/frame, usercmd msec %s)" %
          (nt, t[-1] - t[0], dts.mean() * 1000, summary["usercmd_msec_histogram"]))
    print("movevars:", json.dumps({k: mv0.get(k) for k in MOVEVAR_NAMES}))
    print("differs from sim defaults:", json.dumps(diff) if diff else "none")
    print("origin bounds:", summary["origin_bounds"])
    print("speed max %.1f mean %.1f; simvel vs d(org)/dt median %.2f p95 %.2f u/s" %
          (speed.max(), speed.mean(), np.median(fd_err), np.percentile(fd_err, 95)))
    print("wrote", jpath, "(%d rows)" % len(rows), npath, mpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
