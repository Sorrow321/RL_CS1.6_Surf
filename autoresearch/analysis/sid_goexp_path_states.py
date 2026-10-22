#!/usr/bin/env python3
"""path_states.py - rebuild a Go-Explore archive path (root -> target cell) by deterministic replay and
write its states as a STATE_DTYPE .npy (every K ticks). ANALYSIS tool; the states are rooted at the
policy's own greedy-eval state and produced by the archive's random macros (no demo input).

    python3 path_states.py <archive.pkl> <out.npy> [every_ticks] [target: best|x,y,z]
"""
import pickle, sys, math
import numpy as np
sys.path.insert(0, "/home/steam/ar/wt/b4/python")
sys.path.insert(0, "/tmp/claude-1000/-home-steam-ar-wt-b4/8cb3450e-b752-4b21-ab68-c7d97f89943b/scratchpad/work")
import goexp as G
from surfgym.core import STATE_DTYPE
import torch

A = pickle.load(open(sys.argv[1], "rb")); arch = A["arch"]
out = sys.argv[2]; every = int(sys.argv[3]) if len(sys.argv) > 3 else 4
c = min(arch, key=lambda k: arch[k]["d"])
chain = []
while c is not None:
    chain.append(c); c = arch[c]["parent"]
chain = chain[::-1]
rows = []          # (origin, velocity, yaw_view, ducked)
s = arch[chain[0]]["s"]
rows.append((s[0], s[1], math.degrees(math.atan2(s[1][1], s[1][0])), s[3]))
for prev, cur in zip(chain[:-1], chain[1:]):
    s0 = arch[prev]["s"]; mac = arch[cur]["macros"]
    seq = []
    for side, fwd, off, n in mac:
        seq += [off] * n
    got = []
    def on_tick(k, st, got=got):
        got.append((tuple(st.origin), tuple(st.velocity), int(st.ducked)))
    res, k = G.explore(s0, mac, on_tick)
    assert res == "ok", res
    end = got[-1]
    err = math.dist(end[0], arch[cur]["s"][0])
    assert err < 1e-3, ("replay drift", err)
    for i, (o, v, du) in enumerate(got):
        if (i + 1) % every == 0:
            rows.append((o, v, math.degrees(math.atan2(v[1], v[0])) + seq[i], du))
ck = torch.load("/home/steam/ar/incumbent/ckpt.pt", map_location="cpu", weights_only=False)
tmpl = np.asarray(ck["respawn_greedy"]["rows"])[0].copy()
outa = np.zeros(len(rows), STATE_DTYPE)
for j, (o, v, yw, du) in enumerate(rows):
    r = tmpl.copy()
    r["origin"] = o; r["velocity"] = v; r["basevelocity"] = 0; r["yaw"] = yw % 360.0; r["pitch"] = -10.0
    r["onground"] = -1; r["ducked"] = du; r["induck"] = 0; r["oldbuttons"] = 0; r["tick"] = 0; r["stuck_ticks"] = 0
    outa[j] = r
np.save(out, outa)
print(f"{len(outa)} states from a chain of {len(chain)} cells; root {arch[chain[0]].get('root')}; "
      f"first {np.round(outa['origin'][0])} last {np.round(outa['origin'][-1])}")
