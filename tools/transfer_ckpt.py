#!/usr/bin/env python3
"""transfer_ckpt.py - a CROSS-MAP warm start: another map's checkpoint, made resumable here.

A resume (`--ckpt`) restores the run state next to the weights, and on a different map most of
that state is about the OLD map:

* `respawn` (the reservoir) - RespawnBuffer drops a payload whose map_id differs (safe);
* `int_counts` (the novelty table) - RaceReward re-zeroes a table of the wrong length (safe
  only when the grids happen to differ in size);
* `respawn_novel` (the visit grid) - dropped on a grid mismatch (same caveat);
* `respawn_greedy` (own greedy-death states) - restored WHATEVER the map: raw coordinates of
  the old map, spawned into the new one's walls and voids;
* `unstuck` (the temperature schedule) - its best reading is a distance on the OLD map's
  field, so with `--unstuck-reach eval` the new run starts "stuck" against a number it can
  never beat;
* the config's `map`, which a bare resume restores, and which the launch's record gate
  records on (a map the box does not have).

This writes a checkpoint with the weights and the Adam moments, the config with
`map` set to the target map, the run state dropped, and `global_step` 0, so the run's schedules
start where a scratch run's do and a matched-step comparison against a scratch run is direct.
The provenance is kept in `config["transfer"]`. Nothing here reads the target map: the same
command applies to any map (CLAUDE.md 0b), and the weights carry no demo (check the source's
own `demo_file` / `bc_file` / `route_file`, refused below).

    python tools/transfer_ckpt.py SRC.pt OUT.pt --map surf_princessburglar_b26
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

#: the only top-level entries that are not run state of the source map
KEEP = ("policy", "optimizer", "config", "global_step")
#: config keys that would put demo / record supervision into the resumed run (CLAUDE.md 0)
DEMO_KEYS = ("demo_file", "bc_file", "route_file", "spawn_states")
#: config keys that only describe the source map
DROP_CFG = ("obs_potential_d0", "maps", "heldout_maps")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def transfer(ck: dict, map_stem: str, *, keep_optimizer: bool = True,
             keep_step: bool = False, src: str = "", src_md5: str = "") -> dict:
    """The transferred checkpoint dict (pure: `ck` is not modified)."""
    cfg = dict(ck.get("config") or {})
    bad = [k for k in DEMO_KEYS if cfg.get(k)]
    if bad:
        raise SystemExit(f"the source checkpoint was trained with {bad}: demo / record "
                         "supervision (CLAUDE.md section 0) - not transferred")
    if "policy" not in ck:
        raise SystemExit("the source has no 'policy' entry")
    dropped = sorted(k for k in ck if k not in KEEP)
    src_map = cfg.get("map")
    src_step = int(ck.get("global_step") or 0)
    for k in DROP_CFG:
        cfg.pop(k, None)
    cfg["map"] = str(map_stem)
    cfg["transfer"] = {"src": str(src), "src_md5": str(src_md5), "src_map": src_map,
                       "src_step": src_step, "dropped": dropped,
                       "optimizer": bool(keep_optimizer and "optimizer" in ck),
                       "step_reset": not keep_step}
    out = {"policy": ck["policy"], "config": cfg,
           "global_step": src_step if keep_step else 0}
    if keep_optimizer and "optimizer" in ck:
        out["optimizer"] = ck["optimizer"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--map", required=True, help="the target map's stem (maps/<stem>.bsp)")
    ap.add_argument("--keep-step", action="store_true",
                    help="keep the source's global_step instead of starting at 0")
    a = ap.parse_args(argv)
    import torch
    src = Path(a.src)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    # the Adam moments are kept: train_fast's resume loads ck["optimizer"] unconditionally,
    # and a fresh Adam would move every weight by ~lr per step until its moments warm up
    out = transfer(ck, Path(a.map).stem, keep_step=a.keep_step, src=str(src),
                   src_md5=md5(src))
    torch.save(out, a.out)
    t = out["config"]["transfer"]
    print(f"{src.name} ({t['src_map']}, step {t['src_step']:,}) -> {a.out}: map "
          f"{out['config']['map']}, step {out['global_step']:,}, dropped {t['dropped']}, "
          f"optimizer {'kept' if t['optimizer'] else 'dropped'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
