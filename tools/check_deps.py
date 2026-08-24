#!/usr/bin/env python3
"""check_deps.py - fail LOUDLY on a box that is missing something, before a
run discovers it hours later.

Written after three separate deployments each surfaced a missing dependency
only at the moment it was needed, and each time the symptom pointed
somewhere else entirely:

  * no `cv2`        -> the dashboard's POV button said "POV x retry", with
                       the real ImportError buried in a .pov.err file
  * no `ffmpeg`     -> render_pov silently fell back to cv2's mp4v writer,
                       which produces a valid file NO BROWSER CAN DECODE, so
                       the video opened and played nothing
  * no mesh exports -> the viewer resolved each map correctly, 404'd on its
                       geometry and rendered an empty scene, which looks
                       exactly like "the dashboard shows the wrong map"

None of those is detectable from the training logs, and all three cost a
round-trip. So: check everything up front, say which of TRAIN / VIEW / POV
each failure breaks, and exit non-zero.

    python3 tools/check_deps.py            # report and exit 1 if broken
    python3 tools/check_deps.py --quiet    # only print failures
"""
import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (module, what it breaks, is it fatal, how to get it)
PY_DEPS = [
    ("numpy",  "TRAIN", True,  "pip install numpy"),
    ("scipy",  "TRAIN", True,  "pip install scipy          # SDF distance transform"),
    ("torch",  "TRAIN", True,  "see deploy_box.sh (cuda wheel)"),
    ("triton", "TRAIN", True,  "ships with torch on linux  # the lidar march"),
    ("cv2",    "POV",   False, "pip install opencv-python-headless"),
    ("numba",  "TRAIN", False, "pip install numba          # 5.1x reward fast path, falls back"),
]


def check_python(rows):
    for mod, area, fatal, how in PY_DEPS:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            rows.append(("ok", area, mod, str(v), ""))
        except Exception as ex:
            rows.append(("FAIL" if fatal else "warn", area, mod,
                         type(ex).__name__, how))


def check_binaries(rows):
    # ffmpeg is NOT optional in practice: without it render_pov falls back to
    # cv2's mp4v writer, which browsers cannot decode - a silent failure that
    # produces a playable-looking file with no picture.
    p = shutil.which("ffmpeg")
    if p:
        try:
            v = subprocess.run([p, "-version"], capture_output=True, text=True,
                               timeout=20).stdout.splitlines()[0][:38]
        except Exception:
            v = "present"
        rows.append(("ok", "POV", "ffmpeg", v, ""))
    else:
        rows.append(("FAIL", "POV", "ffmpeg", "missing",
                     "apt-get install -y ffmpeg  # else POV video has no picture"))


def check_cuda(rows):
    try:
        import torch
        if torch.cuda.is_available():
            rows.append(("ok", "TRAIN", "cuda",
                         f"{torch.cuda.device_count()}x "
                         f"{torch.cuda.get_device_name(0)[:24]}", ""))
        else:
            rows.append(("FAIL", "TRAIN", "cuda", "not available",
                         "wrong torch wheel, or no GPU visible"))
    except Exception as ex:
        rows.append(("FAIL", "TRAIN", "cuda", type(ex).__name__, ""))


def check_core(rows):
    sys.path.insert(0, str(ROOT / "python"))
    try:
        from surfgym import SurfCore  # noqa: F401
        rows.append(("ok", "TRAIN", "surfcore", "importable", ""))
    except Exception as ex:
        rows.append(("FAIL", "TRAIN", "surfcore", f"{type(ex).__name__}: {ex}",
                     "build it: see DEPLOY.md"))


def check_assets(rows):
    """Maps present, and a viewer mesh for each - the dashboard renders an
    EMPTY SCENE without the mesh, which reads as the wrong map."""
    bsps = sorted((ROOT / "maps").glob("*.bsp"))
    if not bsps:
        rows.append(("warn", "TRAIN", "maps", "none in maps/",
                     "bash tools/fetch_pool.sh"))
        return
    missing = [b.stem for b in bsps
               if not (ROOT / "viewer" / "assets" / f"{b.stem}.mesh.json").exists()]
    if missing:
        rows.append(("FAIL", "VIEW", "viewer meshes",
                     f"{len(missing)} of {len(bsps)} maps have none",
                     "python3 tools/export_map.py maps/<m>.bsp "
                     "viewer/assets/<m>.mesh.json  (fetch_pool.sh does this)"))
    else:
        rows.append(("ok", "VIEW", "viewer meshes",
                     f"{len(bsps)} maps, all present", ""))
    # a goal field per map, or the trainer bakes one at startup
    nofield = [b.stem for b in bsps
               if not list((ROOT / "maps").glob(f"{b.stem}.goal_*.npz"))]
    if nofield:
        rows.append(("warn", "TRAIN", "goal fields",
                     f"{len(nofield)} of {len(bsps)} maps unbaked",
                     "they will bake at startup - minutes to hours"))
    else:
        rows.append(("ok", "TRAIN", "goal fields", f"{len(bsps)} maps", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    rows = []
    check_python(rows)
    check_binaries(rows)
    check_cuda(rows)
    check_core(rows)
    check_assets(rows)

    bad = [r for r in rows if r[0] == "FAIL"]
    warn = [r for r in rows if r[0] == "warn"]
    show = bad + warn if a.quiet else rows
    if show:
        print(f"{'':5} {'area':6} {'what':16} {'detail'}")
        for st, area, what, detail, how in show:
            mark = {"ok": "  ok ", "warn": " WARN", "FAIL": " FAIL"}[st]
            print(f"{mark} {area:6} {what:16} {detail}")
            if how and st != "ok":
                print(f"{'':5} {'':6} {'':16} -> {how}")
    print(f"\n{len(rows) - len(bad) - len(warn)} ok, {len(warn)} warnings, "
          f"{len(bad)} FAILURES")
    if bad:
        areas = sorted({r[1] for r in bad})
        print(f"!! broken: {', '.join(areas)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
