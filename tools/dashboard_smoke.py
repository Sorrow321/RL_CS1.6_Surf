#!/usr/bin/env python3
"""dashboard_smoke.py - press every dashboard button for one run, headlessly.

Standing rule (user, 2026-09-11): every button must be tested before every
run. Three times in a row a button broke on a freshly launched arm because a
new training flag was not declared in record_ckpt.py's TRAIN_ONLY, and the
failure only surfaced when a human clicked it. This exercises the same HTTP
endpoints the buttons call and reports PASS/FAIL per button.

    python tools/dashboard_smoke.py --run pnFRONT1 [--port 8000]
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.parse, urllib.request


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def poll(base, path, label, budget=420):
    """Kick a job endpoint and poll to a terminal status."""
    t0 = time.time()
    last = None
    while time.time() - t0 < budget:
        try:
            j = get(base, path)
        except Exception as e:
            return label, "FAIL", f"http: {e}"
        st = j.get("status") or ("error" if j.get("error") else "?")
        last = j
        if st in ("done",):
            return label, "PASS", f"{time.time()-t0:.0f}s"
        if st in ("failed", "error") or j.get("error"):
            return label, "FAIL", str(j.get("error") or j)[:160]
        time.sleep(4)
    return label, "FAIL", f"timeout after {budget}s (last {last})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--skip-pov", action="store_true")
    a = ap.parse_args()
    base = f"http://localhost:{a.port}"
    rows = []

    # 1. the runs list itself - a dangling junction used to blank this
    try:
        runs = get(base, "/api/runs")["runs"]
        names = [r["name"] for r in runs]
        rows.append(("runs list", "PASS" if a.run in names else "FAIL",
                     f"{len(runs)} runs, {a.run} "
                     + ("listed" if a.run in names else "MISSING")))
    except Exception as e:
        rows.append(("runs list", "FAIL", str(e)[:160]))
        names = []

    # 2. metrics (the chart)
    try:
        m = get(base, "/api/metrics?run=" + urllib.parse.quote(a.run))
        ser = m.get("series", m) if isinstance(m, dict) else {}
        n = sum(len(v.get("steps", []))
                for v in ser.values() if isinstance(v, dict))
        rows.append(("metrics", "PASS" if n else "FAIL", f"{n} points"))
    except Exception as e:
        rows.append(("metrics", "FAIL", str(e)[:160]))

    # 3. Record, in every spawn mode the UI offers
    q = urllib.parse.quote(a.run)
    for spawn in (None, "reservoir", "mixed"):
        p = f"/api/record?run={q}&mode=greedy" + (f"&spawn={spawn}" if spawn else "")
        rows.append(poll(base, p, f"record[{spawn or 'default'}]"))

    # 4. POV on the newest trajectory, plain and with a forced panel
    if not a.skip_pov:
        try:
            info = next(r for r in get(base, "/api/runs")["runs"] if r["name"] == a.run)
            trajs = info.get("trajs") or []
            traj = trajs[-1]["file"] if trajs else None
        except Exception:
            traj = None
        if not traj:
            rows.append(("render_pov", "FAIL", "no trajectory listed for the run"))
        else:
            t = urllib.parse.quote(traj)
            rows.append(poll(base, f"/api/render_pov?traj={t}", "render_pov"))
            rows.append(poll(base, f"/api/render_pov?traj={t}&panels=mask",
                             "render_pov[+mask]"))

    w = max(len(r[0]) for r in rows)
    bad = 0
    for name, st, note in rows:
        if st != "PASS":
            bad += 1
        print(f"{name:<{w}}  {st:4}  {note}")
    print(("ALL BUTTONS PASS" if not bad else f"{bad} BUTTON(S) BROKEN"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
