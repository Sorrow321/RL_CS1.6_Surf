"""eps x H receding-horizon campaign for beam_tas --commit mode.

Runs the specified grid of --eps x --commit configs in-process (one
torch import), aggregates the per-config summaries - a DNF is recorded
as dose information (committed depth, windows, reason), not failure -
and for every config that assembles a finish compares its replayed line
against the INCUMBENT beam_best at matched ticks: a divergence over
500 u sustained for over 1 s (100 ticks) counts as a genuinely
different route segment and is reported with its time span and peak.
Promotes to <out-root>/beam_best.* ONLY if a config beats the incumbent
time (the one-time beam_best_v1.* backup from campaign 1 is preserved).

    python tools/beam_campaign2.py                # 0.05/0.15/0.30 x 167/333
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import numpy as np

import beam_tas  # noqa: E402  (torch/surfgym imported once)
from surfgym.tick import (episode_seconds, header_tick_ms,  # noqa: E402
                          ticks_to_secs)

DEF_CKPT = "C:/RL_Surf/runs/frozen/sISV_FINISHER_latest.pt"


def load_traj(path):
    """(xyz rows, first episode header) of a record_rollout .jsonl. The
    header carries the recording's TIME BASE (tick_ms, and the integer
    pattern + phase under --tick-ms), which is what turns a tick index of
    the comparison below into seconds - a /100 literal is only correct at
    the 10 ms tick."""
    out, hdr = [], None
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if isinstance(d, list):
            out.append(d[1:4])
        elif hdr is None and isinstance(d, dict) and "end" not in d:
            hdr = d
    return np.asarray(out, np.float64), hdr


def load_xyz(path):
    return load_traj(path)[0]


def divergence(a, b, thresh=500.0, sustain=100):
    """Max positional distance at matched ticks + sustained segments.

    A segment is (start_tick, end_tick, peak_u) where the distance stays
    over `thresh` u for at least `sustain` ticks - the coordinator's
    definition of a genuinely different route segment. ``sustain`` is 1 s
    AT THE RECORDING'S TICK (100 ticks at 10 ms, 131 at 7.667), which the
    caller converts from the header."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, [], 0
    dist = np.linalg.norm(a[:n] - b[:n], axis=1)
    segs = []
    i = 0
    while i < n:
        if dist[i] > thresh:
            j = i
            while j < n and dist[j] > thresh:
                j += 1
            if j - i >= sustain:
                segs.append((int(i), int(j), float(dist[i:j].max())))
            i = j
        else:
            i += 1
    return float(dist.max()), segs, n


def main():
    ap = argparse.ArgumentParser(
        description="receding-horizon (MPC) beam_tas campaign")
    ap.add_argument("ckpt", nargs="?", default=DEF_CKPT)
    ap.add_argument("--eps-grid", default="0.05,0.15,0.30")
    ap.add_argument("--h-grid", default="167,333")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--commit-frac", type=float, default=0.5)
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--extra", default="",
                    help="extra beam_tas flags for every wave, "
                    "space-separated (e.g. '--boundary-v --dedup')")
    ap.add_argument("--wave-prefix", default="",
                    help="prefix for wave dir names (separates reruns)")
    ap.add_argument("--out-root", default=str(ROOT / "runs" / "beam_tas"))
    args = ap.parse_args()

    eps_grid = [float(x) for x in args.eps_grid.split(",") if x]
    h_grid = [int(x) for x in args.h_grid.split(",") if x]
    out_root = Path(args.out_root)
    inc_sum = json.loads((out_root / "summary.json").read_text(
        encoding="utf-8"))
    inc_ticks = int(inc_sum["best_ticks"])
    inc_path = out_root / "beam_best.jsonl"
    inc_xyz, inc_hdr = load_traj(inc_path)
    # the incumbent's clock: beam_tas already writes best_s in seconds;
    # otherwise the summary's tick keys (10 ms for a pre---tick-ms run)
    inc_s = float(inc_sum["best_s"]) if inc_sum.get("best_s") is not None \
        else ticks_to_secs(inc_ticks, inc_sum.get("tick_ms", 10.0),
                           inc_sum.get("tick_pattern_ms"))
    inc_tick_ms = header_tick_ms(inc_hdr, str(inc_path))
    sustain = max(1, int(round(1000.0 / inc_tick_ms)))   # 1 s at that tick
    print(f"incumbent: {inc_ticks} ticks = {inc_s:.2f}s "
          f"({len(inc_xyz)} traj rows at {inc_tick_ms:g} ms)")

    waves = []
    t0 = time.time()
    for E in eps_grid:
        for H in h_grid:
            tag = f"{args.wave_prefix}e{E:g}_h{H}"
            wdir = out_root / "campaign2" / tag
            print(f"=== wave {tag} ({len(waves) + 1}/"
                  f"{len(eps_grid) * len(h_grid)}) ===")
            old_argv, failed = sys.argv, None
            try:
                sys.argv = (["beam_tas.py", args.ckpt,
                             "--torch-seed", str(args.torch_seed),
                             "--envs", str(args.envs),
                             "--eps", str(E), "--commit", str(H),
                             "--commit-frac", str(args.commit_frac),
                             "--greedy-eps", "1", "--out-dir", str(wdir)]
                            + [x for x in args.extra.split() if x])
                beam_tas.main()
            except SystemExit as e:
                failed = f"SystemExit: {e}"
            except AssertionError as e:
                failed = f"ASSERT: {e}"
            finally:
                sys.argv = old_argv
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            row = {"tag": tag, "eps": E, "H": H}
            sfile = wdir / "summary.json"
            if failed is None and sfile.exists():
                d = json.loads(sfile.read_text(encoding="utf-8"))
                row["greedy_ticks"] = d.get("greedy_ticks")
                row["windows"] = d.get("windows")
                row["wall_s"] = d.get("search_wall_s")
                for k in ("collision_before", "collision_after",
                          "rank_agreement"):
                    if k in d:
                        row[k] = d[k]
                if d.get("dnf"):
                    row["dnf"] = d.get("dnf_reason")
                    row["committed_ticks"] = d.get("committed_ticks")
                else:
                    row["best_ticks"] = d["best_ticks"]
                    row["best_s"] = d["best_s"]
                    row["bit_exact"] = d["replay_bit_exact"]
                    cpath = wdir / "beam_best.jsonl"
                    cxyz, chdr = load_traj(cpath)
                    ctick = header_tick_ms(chdr, str(cpath))
                    if abs(ctick - inc_tick_ms) > 1e-9:
                        raise SystemExit(
                            f"{cpath} ran at {ctick:g} ms and the incumbent "
                            f"at {inc_tick_ms:g} ms - a comparison 'at "
                            f"matched ticks' is not the same instants")
                    mx, segs, n = divergence(cxyz, inc_xyz, sustain=sustain)
                    row["div_max_u"] = round(mx)
                    row["div_segments"] = [
                        {"from_s": episode_seconds(inc_hdr, s),
                         "to_s": episode_seconds(inc_hdr, e),
                         "peak_u": round(p)} for s, e, p in segs]
                    row["div_matched_ticks"] = n
            else:
                row["failed"] = failed or "no summary written"
                print(f"wave {tag} FAILED: {row['failed'][:120]}")
            waves.append(row)

    ok = [w for w in waves if "failed" not in w]
    gset = sorted({w["greedy_ticks"] for w in ok if w.get("greedy_ticks")})
    if len(gset) > 1:
        raise SystemExit(f"waves disagree on the greedy baseline ({gset})")

    print("\ntag             result      windows  depth/notes")
    for w in waves:
        if "failed" in w:
            print(f"{w['tag']:15s} FAILED      ({w['failed'][:60]})")
        elif "dnf" in w:
            print(f"{w['tag']:15s} DNF         {w['windows']:3d}      "
                  f"committed {w['committed_ticks']} ticks")
        else:
            seg = w["div_segments"]
            segtxt = ("none" if not seg else "; ".join(
                f"{s['from_s']:.1f}-{s['to_s']:.1f}s peak {s['peak_u']}u"
                for s in seg))
            print(f"{w['tag']:15s} {w['best_s']:6.2f}s     "
                  f"{w['windows']:3d}      div_max {w['div_max_u']}u, "
                  f"segs: {segtxt}")

    fins = [w for w in ok if "best_ticks" in w]
    best = min(fins, key=lambda w: (w["best_ticks"], w["tag"])) if fins \
        else None
    if best is None:
        print(f"\nno config finished; incumbent stands at "
              f"{inc_s:.2f}s. Deepest DNF: "
              + str(max((w for w in ok if "dnf" in w),
                        key=lambda w: w.get("committed_ticks", 0),
                        default=None)))
    else:
        print(f"\ncampaign best: {best['tag']} {best['best_s']:.2f}s vs "
              f"incumbent {inc_s:.2f}s")
        wsrc = out_root / "campaign2" / best["tag"]
        if best["best_ticks"] < inc_ticks:
            for name in ("beam_best.jsonl", "beam_best.npz",
                         "summary.json"):
                cur = out_root / name
                v1 = out_root / name.replace(
                    "beam_best", "beam_best_v1").replace(
                    "summary", "summary_v1")
                if cur.exists() and not v1.exists():
                    shutil.copy2(cur, v1)
            shutil.copy2(wsrc / "beam_best.jsonl",
                         out_root / "beam_best.jsonl")
            shutil.copy2(wsrc / "beam_best.npz",
                         out_root / "beam_best.npz")
            final = json.loads((wsrc / "summary.json").read_text(
                encoding="utf-8"))
            final["campaign2"] = {"wave": best["tag"],
                                  "beat_incumbent_ticks": inc_ticks}
            (out_root / "summary.json").write_text(
                json.dumps(final, indent=2), encoding="utf-8")
            print(f"PROMOTED {best['tag']} -> {out_root / 'beam_best.jsonl'}")
        else:
            print("does not beat the incumbent - NOT promoted")
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "eval_honesty.py"),
             "--route", "C:/RL_Surf/maps/surf_src_cannonball.route.npz",
             str(wsrc / "beam_best.jsonl")], check=False)

    (out_root / f"campaign2{args.wave_prefix and '_' + args.wave_prefix}"
     ".json").write_text(
        json.dumps({"ckpt": args.ckpt, "envs": args.envs,
                    "incumbent_ticks": inc_ticks, "extra": args.extra,
                    "waves": waves,
                    "wall_s": round(time.time() - t0, 1)}, indent=2),
        encoding="utf-8")
    print(f"campaign wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
