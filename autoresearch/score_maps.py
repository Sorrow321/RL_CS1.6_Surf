"""score_maps.py - grade every map of a --maps run with its own config (mm/s1, 2026-11-16).

autoresearch/score.py grades ONE map: its config names the map, the grader field (goalc_32),
start_d (+ start_d_span), score_episodes and time_cap_s. A --maps run writes each map's greedy
eval as traj_<step>_<tag>.jsonl. This driver loads one config per map, picks that map's files
(score.eval_files with the map's tag) and prints one row per map: the best eval (score.py's
number), its step, best progress, finishes and best finish time, how many evals were valid, and
the first eval with a finish - at the fleet step in its file name and at the map's OWN step (the
fleet step x the map's share of --envs, D2: the number to set against a single-map run).

    python autoresearch/score_maps.py runs/mm7                 # the maps of its run.json
    python autoresearch/score_maps.py runs/mm7 cfg_sid.json cfg_b200.json ... [--root DIR] [--json]

Without configs the maps come from the run's run.json (config "maps") and each map's grading
config from autoresearch/map_configs/<stem>.json (map, field, start_d, start_d_span,
score_episodes, time_cap_s - the v14 single-map runs' values). --root resolves the configs'
relative map / field paths (default the repo; a flat staging directory such as the one
tools/mm_assets.py checks works too).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score  # noqa: E402

MAP_CONFIGS = HERE / "map_configs"


def configs_of_run(run_dir) -> list:
    """autoresearch/map_configs/<stem>.json for every map of the run (run.json config "maps",
    else its single "map")"""
    cfg = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8")).get("config", {})
    stems = [Path(m).stem for m in (cfg.get("maps") or [cfg.get("map")]) if m]
    missing = [s for s in stems if not (MAP_CONFIGS / f"{s}.json").exists()]
    if missing:
        raise SystemExit(f"no grading config for {missing} in {MAP_CONFIGS}: pass the configs")
    return [MAP_CONFIGS / f"{s}.json" for s in stems]


def _step(sc):
    m = re.match(r"traj_(\d+)", Path(sc["file"]).name)
    return int(m.group(1)) if m else None


def own_share(run_dir):
    """(envs, maps) of the run (run.json), for the own-step conversion; None when unknown"""
    try:
        cfg = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))["config"]
        n, m = int(cfg["envs"]), len(cfg.get("maps") or [cfg["map"]])
        return n, m
    except (OSError, ValueError, KeyError, TypeError):
        return None


def grade(run_dir, configs, root=None) -> list:
    keep = dict(score.CFG)
    share = own_share(run_dir)

    def own(step):
        # train_fast._own_step: global x the map's envs (N // maps) // N
        return None if step is None or share is None else step * (share[0] // share[1]) // share[0]

    rows = []
    try:
        for c in configs:
            cfg = json.loads(Path(c).read_text(encoding="utf-8")) if not isinstance(c, dict) else c
            score.use_config(cfg, root=root)
            tag = score.map_tag_of_config()
            scored = [score.score_file(x) for x in score.eval_files([run_dir], tag)]
            good = [sc for sc in scored if sc["score"] is not None]
            best = max(good, key=lambda sc: sc["score"]) if good else None    # = score.best_in
            first = next((sc for sc in good if sc["finishes"]), None)          # files in step order
            rows.append({"map": Path(cfg["map"]).stem, "tag": tag, "n_evals": len(scored),
                         "n_valid": len(good), "best": best,
                         # every valid eval in step order: (fleet step, finishes, episodes,
                         # best progress %, best finish s or None)
                         "evals": [(_step(sc), sc["finishes"], sc["episodes"],
                                    sc["best_progress_pct"], sc["best_finish_s"]) for sc in good],
                         "best_step": _step(best) if best else None,
                         "first_finish_step": _step(first) if first else None,
                         "first_finish_own_step": own(_step(first)) if first else None,
                         "rejected": [sc.get("error") for sc in scored if sc["score"] is None][:5]})
    finally:
        score.use_config(keep)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("configs", nargs="*",
                    help="one score config JSON per map (default: the run's maps, "
                         "autoresearch/map_configs/<stem>.json)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rows = grade(a.run_dir, a.configs or configs_of_run(a.run_dir), a.root)
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    print(f"{'map':28s} {'evals':>9s} {'best':>8s} {'at step':>15s} {'prog %':>7s} "
          f"{'fin':>5s} {'best s':>7s} {'first finish':>15s} {'(own step)':>15s}")
    for r in rows:
        b = r["best"]
        if b is None:
            print(f"{r['map']:28s} {r['n_valid']:>4d}/{r['n_evals']:<4d} {'-':>8s} "
                  f"{'-':>15s}" + (f"  ({r['rejected'][0]})" if r["rejected"] else ""))
            continue
        ff, fo = r["first_finish_step"], r["first_finish_own_step"]
        print(f"{r['map']:28s} {r['n_valid']:>4d}/{r['n_evals']:<4d} {b['score']:8.3f} "
              f"{r['best_step'] or 0:>15,d} {b['best_progress_pct']:7.2f} "
              f"{b['finishes']:>2d}/{b['episodes']:<2d} "
              + (f"{b['best_finish_s']:7.2f}" if b["best_finish_s"] is not None else "      -")
              + (f" {ff:>15,d}" if ff is not None else f" {'-':>15s}")
              + (f" {fo:>15,d}" if fo is not None else f" {'-':>15s}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
