#!/usr/bin/env python3
"""bc_merge.py - concatenate tools/plan_to_bc.py BC files into one --bc-file.

plan_to_bc writes one file per spawn (its lines must share one replay); lines planned from
several of the policy's own states (tools/cem_finish.py roots) need one file for the trainer.
Every per-row column is concatenated in order, line ids are offset per input file, and the
result is written through surfgym.bc.save_bc_dataset (the same writer, so the reader sees the
format it expects); the first file's meta is kept with a "merged_from" list.

    python tools/bc_merge.py --out runs/bc/all.npz runs/bc/bc_*.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from surfgym.bc import save_bc_dataset
    zs = [np.load(f, allow_pickle=False) for f in a.files]
    keys0 = set(zs[0].files)
    for f, z in zip(a.files, zs):
        if set(z.files) != keys0:
            raise SystemExit(f"{f}: columns {sorted(set(z.files) ^ keys0)} differ from {a.files[0]}")
    cat = {k: np.concatenate([z[k] for z in zs]) for k in keys0
           if k not in ("version", "meta", "line_id")}
    off, lid = 0, []
    for z in zs:
        li = np.asarray(z["line_id"], np.int64)
        lid.append(li + off)
        off += int(li.max()) + 1 if len(li) else 0
    meta = json.loads(str(zs[0]["meta"]))
    meta["merged_from"] = [str(f) for f in a.files]
    xcols = {k[2:]: cat[k] for k in keys0 if k.startswith("x_")}
    save_bc_dataset(a.out, cat["states"], cat["scal"], cat["latch"], cat["actions"],
                    cat["weights"], np.concatenate(lid).astype(np.int32), meta,
                    probs=cat.get("probs"), zret=cat.get("zret"), zmask=cat.get("zmask"),
                    view=cat.get("view"), view_zmu=cat.get("view_zmu"),
                    view_zsd=cat.get("view_zsd"), keys=cat.get("keys"),
                    xcols=xcols or None, view_z=cat.get("view_z"))
    print(f"bc_merge: {len(a.files)} files, {len(cat['states'])} rows, {off} lines -> {a.out}")


if __name__ == "__main__":
    main()
