"""Go-Explore from the policy's OWN greedy continuation of a spine row (pbr7b4, 2026-10-27): the script that built
spine pbr7b4_c21app01 (an unboosted line through princessburglar's 74.4% passage to 78.14%). Search on the policy's own
states only, never a demo.

1. record_ckpt greedy, one episode, from the spine's row ROW with --dump-states (the policy flies on from there):
       python tools/record_ckpt.py CKPT --map maps/<map>.bsp --spawn-states row.npy --episodes 1 \
           --out OUT/greedy.jsonl --eval-stall 0 --maxvel 3500 --dump-states OUT/greedy_states.npz
2. this script: line = spine[:ROW] + the greedy's states; explore_phase1 --exact-chains through auto_spine.go_explore
   (x1.0 by default), rooted at --n-roots of the greedy's states with progress in [LO, HI] %; the deepest chain spliced
   at its root -> OUT/spine_app.npy, and the held keys -> OUT/spine_app.keys.npy (the spine's keys to ROW, the greedy's
   own fwd / side / duck of the action that produced each state, the chain's link actions translated to the canonical
   views as go_explore does).

The window used, [72.60, 73.44] %, is the last 0.6 s before the greedy's stall by auto_spine's rule (stall_tick, 500 u:
row 6,624 at 73.49%), i.e. auto_spine's roots rule with --roots-secs 0.6.

    python autoresearch/analysis/pb_ge_from_greedy.py --out OUT --spine S.npy --keys S.keys.npy --row 6378 --ckpt CKPT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "python"))
sys.path.insert(0, str(WT / "tools"))
sys.path.insert(0, str(WT / "autoresearch"))
import auto_spine  # noqa: E402
import score  # noqa: E402
from surfgym.core import STATE_DTYPE  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--row", type=int, default=6378)
p.add_argument("--spine", default=str(WT / "autoresearch/spines/pbr7b4_c17e01.npy"))
p.add_argument("--keys", default=str(WT / "autoresearch/spines/pbr7b4_c17e01.keys.npy"))
p.add_argument("--ckpt", default="/home/steam/ar_pb/runs/pbr7b4/pbr7b4_c21/ckpt_final.pt")
p.add_argument("--lo", type=float, default=72.6)
p.add_argument("--hi", type=float, default=73.44)
p.add_argument("--minutes", type=float, default=12.0)
p.add_argument("--envs", type=int, default=1024)
p.add_argument("--threads", type=int, default=3)
p.add_argument("--n-roots", type=int, default=48)
p.add_argument("--heading", type=int, default=1)
c = p.parse_args()
out = Path(c.out)
out.mkdir(parents=True, exist_ok=True)
e01 = np.load(c.spine)
g = np.load(out / "greedy_states.npz")["ep_0000"].astype(STATE_DTYPE)
line = np.concatenate([e01[:c.row], g])
F = score.field()
d = F.sample(line["origin"].astype(np.float64)).astype(np.float64)
prog = 100 * (57149 - d) / 57149
k = np.flatnonzero((np.arange(len(line)) >= c.row) & (prog >= c.lo) & (prog <= c.hi))
idx = np.unique(np.linspace(k[0], k[-1], min(c.n_roots, len(k))).round().astype(int))
print(f"line {len(line)} rows; roots {len(idx)} at rows {idx[0]}-{idx[-1]} "
      f"({prog[idx[0]]:.2f}-{prog[idx[-1]]:.2f}%)", flush=True)
np.save(out / "line.npy", line)
ck = c.ckpt
a = argparse.Namespace(
    map=str(WT / "maps/surf_princessburglar_b26.bsp"), envs=c.envs, ge_cell=64.0, ge_energy=32.0,
    act_every=4, ge_minutes=c.minutes, seed=0, maxvel=3500.0, threads=c.threads,
    tick=auto_spine.ckpt_tick(ck), ge_keep_all=1, ge_cell_heading=c.heading, ge_select_energy=0.0,
    ge_view_follow="-90,-60,-30,-15,0,15,30,60,90",
    field=str(WT / "maps/surf_princessburglar_b26.goalc_32.npz"), roots_secs=4.0, roots_n=48)
auto_spine.TICKS_PER_S = 1000.0 / a.tick
sp, info = auto_spine.go_explore(a, out, line, int(idx[-1]), boosts=[1.0], tag="ge_app", roots=idx)
res = {"info": info}
if sp is not None:
    ds = F.sample(sp["origin"].astype(np.float64)).astype(np.float64)
    pr = 100 * (57149 - ds) / 57149
    res.update(spine_len=int(len(sp)), end_prog=float(pr[-1]), max_prog=float(pr.max()))
    np.save(out / "spine_app.npy", sp)
if sp is not None:
    # the held keys (spine_canon_keys / auto_spine conventions): keys[t] = the keys that produced state t
    import json as _json
    from spine_canon_keys import translate
    from surfgym.record import SURF_IN_DUCK
    rows = [_json.loads(x) for x in open(out / "greedy.jsonl") if x.startswith("[")]
    j = int(info["ge_root_index"])
    gk = np.array([(rows[t - 1][13], rows[t - 1][14], 1 if int(rows[t - 1][8]) & SURF_IN_DUCK else 0)
                   for t in range(1, j - c.row + 1)], np.int32).reshape(-1, 3)
    gl = np.load(out / "ge_app" / "line.npy")
    chk = auto_spine.chain_keys(out / "ge_app" / "archive.npz", auto_spine.load_field(a.field), len(gl))
    if chk is not None:
        v = np.asarray(gl["velocity"][1:], np.float64)
        hd = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
        fast = np.hypot(v[:, 0], v[:, 1]) >= 100.0
        chk = chk.copy()
        chk[fast] = translate(chk[fast], np.asarray(gl["yaw"][1:], np.float64)[fast], hd[fast])
        keys = np.concatenate([np.load(c.keys)[:c.row + 1], gk, chk.astype(np.int32)])
        assert len(keys) == len(sp), (len(keys), len(sp))
        np.save(out / "spine_app.keys.npy", keys)
        res["keys"] = int(len(keys))
print(json.dumps(res, default=str), flush=True)
