"""Runner for mm_slot_replay.py: the v14 line at CPU smoke scale on MAPS (record), then each
map alone (replay K), then the per-slot comparison. Usage:

    python tests/python/mm_slot_replay_run.py OUTDIR MAP0.bsp,MAP1.bsp,... MAXV0,MAXV1,...
        [ITERS] [--drop FAMILY,...] [--extra "tokens"]

PER envs per map = 21 (63 for three maps), so the eval cores (--eval-eps 2 -> 2 envs) never
share a width with a training core.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v14_smoke as V  # noqa: E402

PER = 21


STRIP = []                                     # --strip: single flags removed from the line


def line(map_args, run, steps, drop, extra, tail=()):
    flags = [f for d in drop for f in V.FAMILIES[d]] + STRIP
    toks = V.strip(V.BASE + V.V14 + V.MCTS + V.smoke(steps), flags)
    # --envs / --maxvel LAST: the line carries its own and argparse keeps the last one
    return list(map_args) + ["--run", run] + toks + list(extra) + list(tail)


def main():
    out = Path(sys.argv[1]).resolve()
    maps = sys.argv[2].split(",")
    maxv = sys.argv[3].split(",")
    iters = int(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else 12
    drop, extra = [], []
    if "--drop" in sys.argv:
        drop = [d for d in sys.argv[sys.argv.index("--drop") + 1].split(",") if d]
    if "--extra" in sys.argv:
        extra = shlex.split(sys.argv[sys.argv.index("--extra") + 1])
    if "--strip" in sys.argv:
        STRIP.extend(f for f in sys.argv[sys.argv.index("--strip") + 1].split(",") if f)
    only = None
    if "--only" in sys.argv:
        only = [int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")]
    out.mkdir(parents=True, exist_ok=True)
    nm = len(maps)
    tick = 4 * 8                                   # act_every x n_steps (smoke)
    steps_multi = iters * PER * nm * tick
    store = out / "rec.pt"
    env = dict(V.env(), PYTHONHASHSEED="0")
    tag = out.name
    runs = []
    if only is None or -1 in only:
        runs.append(("record", 0, line(["--maps", ",".join(maps)], f"{tag}_rec", steps_multi,
                                        drop, extra, ["--envs", str(PER * nm),
                                                      "--maxvel", ",".join(maxv)])))
    for k, m in enumerate(maps):
        if only is not None and k not in only:
            continue
        runs.append(("replay", k, line(["--map", m], f"{tag}_alone{k}", steps_multi // nm, drop,
                                       extra, ["--envs", str(PER), "--maxvel", maxv[k]])))
    for mode, k, argv in runs:
        import shutil
        shutil.rmtree(V.ROOT / "runs" / (f"{tag}_rec" if mode == "record" else f"{tag}_alone{k}"),
                      ignore_errors=True)
        log = out / (f"{mode}{k}.log")
        cmd = (["nice", "-n", "15", sys.executable, "-u", str(HERE / "mm_slot_replay.py"), mode,
                str(store)] + ([str(k)] if mode == "replay" else []) + ["--"] + argv)
        print(" ".join(cmd[:8]), "...", flush=True)
        with open(log, "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                               cwd=str(V.ROOT))
        print(f"{mode} {k}: rc {r.returncode} ({log})", flush=True)
        if r.returncode != 0:
            print(log.read_text()[-3000:])
            sys.exit(r.returncode)
    for k in range(nm):
        if only is not None and k not in only:
            continue
        r = subprocess.run([sys.executable, str(HERE / "mm_slot_replay.py"), "compare",
                            str(store), str(k)], capture_output=True, text=True, env=env)
        print(f"=== slot {k} ({Path(maps[k]).stem}) vs alone: rc {r.returncode}")
        print(r.stdout[-6000:], r.stderr[-3000:])


if __name__ == "__main__":
    main()
