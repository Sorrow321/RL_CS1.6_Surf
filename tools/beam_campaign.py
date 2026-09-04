"""Grid campaign driver for tools/beam_tas.py.

Runs beam_tas.main() IN-PROCESS (one torch import, one CUDA context)
over a grid of --torch-seed x --resample-every waves, each wave writing
into its own out dir, then aggregates the per-wave summaries, promotes
the global-best run's artifacts to <out-root>/beam_best.jsonl/.npz
(backing the previous ones up as beam_best_v1.* once), and writes
<out-root>/campaign_summary.json plus an updated <out-root>/summary.json.

The search itself is untouched: waves vary only flags beam_tas already
has. Every wave re-runs the (deterministic) greedy gate, which doubles
as a cross-wave determinism check - the aggregation refuses to promote
anything if the waves disagree on the greedy baseline. Each wave's own
replay-bit-exact assert has already passed before its artifacts exist,
so promoting a wave promotes a verified run.

    python tools/beam_campaign.py                 # 8 seeds x R {25,100,250}
    python tools/beam_campaign.py CKPT --seeds 0,1 --resamples 25
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import beam_tas  # noqa: E402  (pulls torch/surfgym once for all waves)
from surfgym.tick import ticks_to_secs  # noqa: E402  (never ticks / 100)

DEF_CKPT = "C:/RL_Surf/runs/frozen/sISV_FINISHER_latest.pt"


def main():
    ap = argparse.ArgumentParser(
        description="grid of independent beam_tas searches")
    ap.add_argument("ckpt", nargs="?", default=DEF_CKPT)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7",
                    help="comma list of --torch-seed values")
    ap.add_argument("--resamples", default="25,100,250",
                    help="comma list of --resample-every values")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--out-root", default=str(ROOT / "runs" / "beam_tas"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rs = [int(r) for r in args.resamples.split(",")]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    waves = []
    t0 = time.time()
    for R in rs:
        for s in seeds:
            tag = f"s{s}_r{R}"
            wdir = out_root / "campaign" / tag
            print(f"=== wave {tag} ({len(waves) + 1}/"
                  f"{len(rs) * len(seeds)}) ===")
            old_argv, failed = sys.argv, None
            try:
                sys.argv = ["beam_tas.py", args.ckpt,
                            "--torch-seed", str(s),
                            "--resample-every", str(R),
                            "--envs", str(args.envs),
                            "--greedy-eps", "1",
                            "--out-dir", str(wdir)]
                beam_tas.main()
            except SystemExit as e:     # gate fail / no finisher / diverged
                failed = f"SystemExit: {e}"
            except AssertionError as e:  # replay determinism assert
                failed = f"ASSERT: {e}"
            finally:
                sys.argv = old_argv
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            row = {"tag": tag, "seed": s, "R": R}
            sfile = wdir / "summary.json"
            if failed is None and sfile.exists():
                d = json.loads(sfile.read_text(encoding="utf-8"))
                row.update(best_ticks=d["best_ticks"], best_s=d["best_s"],
                           finishes=d["finishes"],
                           generations=d["generations"],
                           greedy_ticks=d["greedy_ticks"],
                           search_wall_s=d["search_wall_s"],
                           bit_exact=d["replay_bit_exact"],
                           # the wave's own time base (beam_tas.tick_stamp);
                           # absent in a summary written before --tick-ms,
                           # which ran at 10 ms
                           tick_ms=d.get("tick_ms", 10.0),
                           tick_pattern_ms=d.get("tick_pattern_ms"))
            else:
                row["failed"] = failed or "no summary written"
                print(f"wave {tag} FAILED: {row['failed']}")
            waves.append(row)

    ok = [w for w in waves if "failed" not in w]
    if not ok:
        raise SystemExit("campaign: every wave failed")
    # the greedy gate is deterministic; disagreement means the waves were
    # not running the search this campaign thinks they were
    gset = sorted({w["greedy_ticks"] for w in ok})
    if len(gset) != 1:
        raise SystemExit(f"campaign: waves disagree on the greedy baseline "
                         f"({gset}) - refusing to promote a global best")
    greedy_ticks = gset[0]

    print("\ntag        best_s  finishes  gens  wall_s")
    for w in waves:
        if "failed" in w:
            print(f"{w['tag']:9s}  FAILED   ({w['failed'][:50]})")
        else:
            print(f"{w['tag']:9s}  {w['best_s']:6.2f}  {w['finishes']:8d}  "
                  f"{w['generations']:4d}  {w['search_wall_s']:6.1f}")
    print("\nper-R distribution of wave-best times (s):")
    by_r = {}
    for R in rs:
        b = sorted(w["best_s"] for w in ok if w["R"] == R)
        if b:
            med = b[len(b) // 2] if len(b) % 2 else \
                (b[len(b) // 2 - 1] + b[len(b) // 2]) / 2
            by_r[R] = {"n": len(b), "min": b[0], "median": med,
                       "max": b[-1], "distinct_ticks": len({
                           w["best_ticks"] for w in ok if w["R"] == R})}
            print(f"  R={R:3d}: n={len(b)} min={b[0]:.2f} med={med:.2f} "
                  f"max={b[-1]:.2f} distinct={by_r[R]['distinct_ticks']}")

    best = min(ok, key=lambda w: (w["best_ticks"], w["tag"]))
    # seconds at the WAVE's tick, not ticks / 100
    def _secs(t):
        return ticks_to_secs(t, best.get("tick_ms", 10.0),
                             best.get("tick_pattern_ms"))
    print(f"\nglobal best: {best['tag']} at {best['best_s']:.2f}s "
          f"(greedy {_secs(greedy_ticks):.2f}s, margin "
          f"{_secs(greedy_ticks) - _secs(best['best_ticks']):+.2f}s)")

    # promote the winner's verified artifacts; back up v1 exactly once
    for name in ("beam_best.jsonl", "beam_best.npz", "summary.json"):
        cur, v1 = out_root / name, out_root / name.replace(
            "beam_best", "beam_best_v1").replace(
            "summary", "summary_v1")
        if cur.exists() and not v1.exists():
            shutil.copy2(cur, v1)
    wsrc = out_root / "campaign" / best["tag"]
    shutil.copy2(wsrc / "beam_best.jsonl", out_root / "beam_best.jsonl")
    shutil.copy2(wsrc / "beam_best.npz", out_root / "beam_best.npz")
    final = json.loads((wsrc / "summary.json").read_text(encoding="utf-8"))
    final["campaign"] = {"wave": best["tag"], "waves": len(waves),
                         "failed_waves": len(waves) - len(ok),
                         "by_R": by_r,
                         "wall_s": round(time.time() - t0, 1)}
    (out_root / "summary.json").write_text(json.dumps(final, indent=2),
                                           encoding="utf-8")
    (out_root / "campaign_summary.json").write_text(
        json.dumps({"ckpt": args.ckpt, "envs": args.envs,
                    "greedy_ticks": greedy_ticks, "waves": waves,
                    "by_R": by_r, "global_best": best,
                    "wall_s": round(time.time() - t0, 1)}, indent=2),
        encoding="utf-8")
    print(f"promoted {best['tag']} -> {out_root / 'beam_best.jsonl'}; "
          f"campaign wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
