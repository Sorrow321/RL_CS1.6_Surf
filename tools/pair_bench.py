"""pair_bench.py - where a ramp-pair checkpoint cannot make the transition (the user, 2026-09-28).

"Maybe you could diagnose the most popular failure cases ... analyze where the model cannot make
the transition. Is it just the hard cases or there are some simple ones?"

For every [T1, T2] pair of a map (the run's own --ramp-pairs file): spawn the checkpoint at that
pair's state, --greedy G and --stoch K episodes, through the recorder's own construction
(record_ckpt.build: the same policy wrapper, target channel and ramp-window hooks as a dashboard
recording). Every episode is classified from its recorded ticks - which target surfaces it
touched (RampVocab.contact_of), in what order - into one outcome:

  done         entered T2 after passing T1 (the pair's success)
  miss_t1      never touched T1
  stuck_t1     touched T1, never left its box (fell off / died on it / timed out on it)
  miss_t2      passed T1, never touched T2 - with its closest approach to T2's contact origins
  wrong_ramp   passed T1, then rode another target instead of T2

and the pass state: exit speed and energy height (z + |v|^2 / 2g) as the window shifted to T2.

    python tools/pair_bench.py runs/jtPAIRS5/ckpt_latest.pt --map surf_unitfarmer2 \\
        --greedy 1 --stoch 16 --out runs/research/pair_bench/jtPAIRS5_unitfarmer2.json

Output JSON: per pair its targets, spawn, per-mode outcome counts, and every episode's outcome /
exit speed / exit energy / closest approach to T2 / end reason. A measurement tool: nothing it
writes enters training.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def read_episodes(path):
    """a record_rollout JSONL -> [(rows (n, k) float64, end dict)]"""
    eps, rows = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if isinstance(r, list):
            rows.append(r)
        elif "end" in r:
            eps.append((np.asarray(rows, np.float64), r))
            rows = []
        elif "episode" in r:
            rows = []
    return eps


def classify_seq(rows, end, seq, voc):
    """--triplets: one episode of the list T1 -> T2 -> T3 -> its outcome - done (entered the
    last), fail_t1 (T1 never passed), fail_hop1 (passed T1, never touched T2), stuck_t2 (touched
    T2, never left its box), fail_hop2 (passed T2, never entered T3)"""
    tg = end.get("targets") or {}
    org = rows[:, 1:4]
    duck = ((rows[:, 8].astype(np.int64) & 4) != 0).astype(np.int64)
    cs = np.asarray(voc.contact_of(org, duck), np.int64)
    touched = [int(s) for s in cs if s >= 0 and voc.is_target[s]]
    order = [s for i, s in enumerate(touched) if i == 0 or s != touched[i - 1]]
    st = int(tg.get("seq_stage", -1))
    out = {"stage": st, "end": str(end.get("end")), "ticks": int(end.get("ticks", len(rows))),
           "touched": order[:12]}
    if tg.get("seq_done"):
        out["outcome"] = "done"
    elif st <= 0:
        out["outcome"] = "fail_t1"
    elif st == 1:
        out["outcome"] = "stuck_t2" if int(seq[1]) in order else "fail_hop1"
    else:
        out["outcome"] = "fail_hop2"
    if not tg.get("seq_done"):
        p_last = voc.points[voc.surf == int(seq[min(st + 1, len(seq) - 1)] if st >= 0 else seq[0])]
        if len(p_last):
            d = np.min(np.linalg.norm(org[:, None, :] - p_last[None, ::max(1, len(p_last) // 400), :],
                                      axis=2), axis=1)
            out["next_closest"] = float(d.min())
    return out


def classify(rows, end, t1, t2, voc, gravity):
    """one episode -> its outcome and the pass state"""
    tg = end.get("targets") or {}
    org = rows[:, 1:4]
    vel = rows[:, 4:7]
    duck = ((rows[:, 8].astype(np.int64) & 4) != 0).astype(np.int64)
    cs = np.asarray(voc.contact_of(org, duck), np.int64)
    touched = [int(s) for s in cs if s >= 0 and voc.is_target[s]]
    order = [s for i, s in enumerate(touched) if i == 0 or s != touched[i - 1]]
    ev = tg.get("events") or []
    # the window's shift off T1: the first event whose T1 is no longer t1 (row, prev, T1, T2, ...)
    k_pass = next((int(e[0]) for e in ev[1:] if int(e[2]) != int(t1)), None)
    out = {"stage": int(tg.get("seq_stage", -1)), "end": str(end.get("end")),
           "ticks": int(end.get("ticks", len(rows))), "touched": order[:12]}
    p2 = voc.points[voc.surf == int(t2)]
    if tg.get("seq_done"):
        out["outcome"] = "done"
    elif int(t1) not in order:
        out["outcome"] = "miss_t1"
    elif k_pass is None:
        out["outcome"] = "stuck_t1"
    else:
        after = [s for s in order[order.index(int(t1)) + 1:] if s != int(t1)]
        out["outcome"] = "wrong_ramp" if after else "miss_t2"
        if after:
            out["instead"] = after[0]
    if k_pass is not None and k_pass < len(rows):
        v = vel[k_pass]
        out["exit_speed"] = float(np.linalg.norm(v))
        out["exit_energy_z"] = float(org[k_pass, 2] + float(v @ v) / (2.0 * gravity))
        out["exit_row"] = int(k_pass)
        if len(p2):
            seg = org[k_pass:]
            d = np.min(np.linalg.norm(seg[:, None, :] - p2[None, ::max(1, len(p2) // 400), :],
                                      axis=2), axis=1)
            out["t2_closest"] = float(d.min())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--map", required=True, help="map stem of the checkpoint's --maps")
    ap.add_argument("--greedy", type=int, default=1, help="greedy episodes per pair")
    ap.add_argument("--stoch", type=int, default=16, help="sampled episodes per pair")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--triplets", action="store_true",
                    help="T1 -> T2 -> T3 from each pair's spawn, T3 = the next pair's T2 (the "
                         "user, 2026-09-28: take the first ramp, go to the next one, then to the "
                         "third); success = entering T3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import record_ckpt
    from surfgym.goalramps import _origin_key, load_ramp_pairs, ramp_file_for_map
    from surfgym.record import record_rollout

    res = {"ckpt": str(a.ckpt), "map": a.map, "pairs": []}
    builds = [("greedy", a.greedy, [])] + ([("stoch", a.stoch, ["--stochastic"])]
                                           if a.stoch > 0 else [])
    tmp = Path(tempfile.mkdtemp(prefix="pair_bench_"))
    for mode, k, extra in builds:
        if k <= 0:
            continue
        b = record_ckpt.build([str(a.ckpt), "--map", a.map, "--episodes", "1"] + extra)
        cfg = b.cfg
        if not cfg.get("ramp_pairs"):
            raise SystemExit(f"{a.ckpt} is not a --ramp-pairs checkpoint")
        pp = ramp_file_for_map(cfg["ramp_pairs"], Path(b.map_path).stem, ROOT)
        pairs = load_ramp_pairs(str(pp))
        if a.triplets:
            # consecutive pairs: pair i + 1's T1 is pair i's T2
            assert (pairs["t1"][1:] == pairs["t2"][:-1]).all(), "pairs are not one chain"
            seqs = np.stack([pairs["t1"][:-1], pairs["t2"][:-1], pairs["t2"][1:]], 1)
            states = pairs["states"][:-1]
        else:
            seqs = np.stack([pairs["t1"], pairs["t2"]], 1)
            states = pairs["states"]
        # the eval windows' per-spawn target lists: this bench's
        table = {_origin_key(o): tuple(int(x) for x in q) for o, q in zip(states["origin"], seqs)}
        b.ramp_planner.eval_windows.pair_lookup = lambda p_, _t=table: _t.get(_origin_key(p_))
        voc = b.vocab
        g = float(getattr(b.core.config.phys, "sv_gravity", 800.0))
        if not res["pairs"]:
            res["pairs_file"] = str(pp)
            res["source"] = pairs["source"]
            res["step"] = int(b.step)
            res["triplets"] = bool(a.triplets)
            for i in range(len(seqs)):
                st = states[i]
                res["pairs"].append({"i": i, "t1": int(seqs[i][0]), "t2": int(seqs[i][1]),
                                     "seq": [int(x) for x in seqs[i]],
                                     "spawn": [float(x) for x in st["origin"]],
                                     "spawn_speed": float(np.linalg.norm(st["velocity"])),
                                     "modes": {}})
        for i, pr in enumerate(res["pairs"]):
            b.core.set_spawn_pool(states[i:i + 1])
            f = tmp / f"{mode}_{i:03d}.jsonl"
            record_rollout(b.core, b.pol, f, episodes=k, max_ticks=k * int(b.ep_ticks),
                           seed=a.seed + 1000 * i, on_tick=b.on_tick,
                           episode_meta=b.episode_meta, header_extra=b.header_extra)
            eps = [(classify_seq(r, e, pr["seq"], voc) if a.triplets
                    else classify(r, e, pr["t1"], pr["t2"], voc, g)) for r, e in read_episodes(f)]
            pr["modes"][mode] = {"n": len(eps), "counts": dict(Counter(e["outcome"] for e in eps)),
                                 "episodes": eps}
            c = pr["modes"][mode]["counts"]
            print(f"{a.map} {'triplet' if a.triplets else 'pair'} {i:2d} {pr['seq']} {mode}: "
                  f"done {c.get('done', 0)}/{len(eps)}  "
                  + " ".join(f"{k_} {v}" for k_, v in sorted(c.items()) if k_ != "done"),
                  flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
