"""Render the agent-vs-WR video through the viewer (viewer/vs_render.html, three.js) in headless
Chrome, frame by frame, and encode it with ffmpeg. The dashboard (tools/dashboard.py, port 8600)
serves the page and the frames JSON (tools/demo/vs_wr_frames.py).

usage: python tools/demo/vs_wr_render.py --frames /runs/skWR_cap2000/vs_frames.json --out x.mp4
       [--stills 5,30,60 -> JPEG stills next to --out instead of the video]
"""
import argparse
import base64
import shutil
import subprocess
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="runs/skWR_cap2000/vs_frames.json",
                    help="the frames JSON's URL path on the dashboard")
    ap.add_argument("--host", default="http://localhost:8600")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stills", default=None)
    ap.add_argument("--quality", type=float, default=0.93)
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    o = Options()
    o.add_argument("--headless=new")
    o.add_argument("--window-size=1280,1560")
    o.add_argument("--use-angle=d3d11")
    o.add_argument("--ignore-gpu-blocklist")
    o.add_argument("--hide-scrollbars")
    d = webdriver.Chrome(options=o)
    try:
        t0 = time.time()
        for attempt in range(4):
            # --frames is a URL PATH on the dashboard. Pass it WITHOUT the leading slash from Git
            # Bash: its MSYS path conversion turned '/runs/...' into 'C:/Program Files/Git/
            # runs/...' and every fetch failed. The reload loop is only a guard.
            d.get(f"{args.host}/viewer/vs_render.html?frames=/{args.frames.lstrip('/')}")
            while True:
                st = d.execute_script("return [window.ready === true, window.loadError || null, "
                                      "window.nFrames || 0];")
                if st[0] or st[1] or time.time() - t0 > 120:
                    break
                time.sleep(0.5)
            if st[0]:
                break
            print(f"page load attempt {attempt + 1} failed: {st[1] or 'timeout'}")
            time.sleep(2)
        if not st[0]:
            raise SystemExit(f"page not ready: {st[1] or 'timeout'}")
        n = int(st[2])
        fps = int(d.execute_script("return data.fps;"))
        print(f"page ready in {time.time() - t0:.1f}s: {n} frames at {fps} fps")

        def grab(k):
            url = d.execute_script("return window.captureFrame(arguments[0], arguments[1]);", k,
                                   args.quality)
            return base64.b64decode(url.split(",", 1)[1])

        if args.stills:
            for s in (float(x) for x in args.stills.split(",")):
                p = Path(args.out).with_suffix("").as_posix() + f"_{s:05.1f}s.jpg"
                Path(p).write_bytes(grab(int(round(s * fps))))
                print("wrote", p)
            return
        enc = subprocess.Popen([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "image2pipe",
                                "-framerate", str(fps), "-c:v", "mjpeg", "-i", "-",
                                "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
                               stdin=subprocess.PIPE)
        t1 = time.time()
        for k in range(n):
            enc.stdin.write(grab(k))
            if k % 600 == 0:
                print(f"  frame {k}/{n} ({time.time() - t1:.0f}s)", flush=True)
        enc.stdin.close()
        enc.wait()
        print(f"wrote {args.out} in {time.time() - t1:.0f}s")
    finally:
        d.quit()


if __name__ == "__main__":
    main()
