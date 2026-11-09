# analysis only (unr1b1): how v2's greedy eval episodes approach the post-CP2 slot. Rows are
# [tick, x, y, z, vx, vy, vz, yaw, ...]. For each episode: where it ends, and the state at which
# it drops below the shelf's underside (z < 10,528 after having been above 10,592, at x < -184):
# speed, -x speed, fall rate and descent angle there (geometry from the goal field, MEASUREMENT).
import json, sys, math
for path in sys.argv[1:]:
    eps = []; cur = None
    for line in open(path):
        d = json.loads(line)
        if isinstance(d, dict):
            cur = []; eps.append(cur)
        else:
            cur.append(d)
    print(path.split('/')[-1], len(eps), "episodes")
    eps = [e for e in eps if e]
    for k, ep in enumerate(eps):
        last = ep[-1]
        above = False; drop = None; apex = None; pre = None
        for r in ep:
            x, y, z = r[1], r[2], r[3]
            if x < -184 and z > 10592:
                above = True
                if apex is None or z > apex[3]:
                    apex = r
            if above and x < -184 and z < 10528 and drop is None:
                drop = r
            if drop is None and x < -184 and z > 10592:
                pre = r
        s = f"  ep {k}: {len(ep)} rows, end x {last[1]:7.0f} y {last[2]:6.0f} z {last[3]:7.0f} |v| {math.hypot(last[4], last[5], last[6]):5.0f}"
        if apex is not None:
            s += f" | apex past x-184 z {apex[3]:6.0f} at x {apex[1]:6.0f}"
        if drop is not None:
            vx, vy, vz = drop[4], drop[5], drop[6]
            vh = math.hypot(vx, vy)
            s += (f" | under shelf at x {drop[1]:6.0f} y {drop[2]:5.0f} z {drop[3]:6.0f}: |v| {math.hypot(vh, vz):5.0f} "
                  f"vx {vx:6.0f} vz {vz:6.0f} descent {math.degrees(math.atan2(-vz, vh)):4.1f} deg")
        if pre is not None and drop is not None:
            vh = math.hypot(pre[4], pre[5])
            s += (f"\n        last state over the shelf: x {pre[1]:6.0f} y {pre[2]:5.0f} z {pre[3]:6.0f} vx {pre[4]:6.0f} vy {pre[5]:6.0f} vz {pre[6]:6.0f}"
                  f" descent {math.degrees(math.atan2(-pre[6], vh)):4.1f} deg; min x after: {min(r[1] for r in ep[ep.index(drop):]):6.0f}")
        print(s)
