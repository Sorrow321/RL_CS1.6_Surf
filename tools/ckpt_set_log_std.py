#!/usr/bin/env python3
"""ckpt_set_log_std.py - rewrite ONE view head's log sigma in a train_fast
checkpoint, nothing else (the A/B surgery of the 2026-09-07 cyUNSTUCK
ledger entry: the pitch head of a --pitch-entropy 0 run sits at sigma
0.056 by 1B steps; the plain-seed runs of 12a6a3a had it at 0.58 by 1B and
at the 2.72 clamp by 8B, and --pitch-entropy 1.0 alone recovers it at
~1e-5 per update, far too slowly for a 500M-step arm).

    python tools/ckpt_set_log_std.py IN.pt OUT.pt --head pitch --sigma 0.5
    python tools/ckpt_set_log_std.py IN.pt OUT.pt --head yaw --log-std -2.87

``--head`` is ``yaw`` (0), ``pitch`` (the LAST Gaussian head) or an index;
exactly one of ``--sigma`` / ``--log-std``. ``--reset-adam`` also zeroes
the Adam moments of that one parameter (default: left as they are - the
entry is (n_z,) wide, its step counter is shared with every other
parameter and the moments of the untouched head must survive). Every other
tensor of the policy and the optimizer state, and every other key of the
checkpoint, are checked equal before OUT is written. Under --view-absolute
the pitch head is capped at log 0.5 by Policy.log_std(): a value above the
cap is refused unless --allow-above-cap (project_log_std would pull it
back on the first step anyway)."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

KEY = "view_std.log_std"


def _equal(a, b) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape \
            and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) \
            and a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    try:
        if a != a and b != b:            # NaN == NaN for this purpose
            return True
    except Exception:
        pass
    return a == b


def set_log_std(ck: dict, head, value: float, reset_adam: bool = False,
                allow_above_cap: bool = False) -> dict:
    """-> {"index", "before", "after", "adam_reset"}; mutates ``ck``."""
    from train_fast import PITCH_LOG_STD_MAX_ABS, LOG_STD_MIN, LOG_STD_MAX
    pol = ck["policy"]
    if KEY not in pol:
        raise SystemExit(f"{KEY} not in the policy: not a --view-continuous "
                         "checkpoint")
    ls = pol[KEY]
    n = int(ls.numel())
    idx = (0 if head == "yaw" else n - 1 if head == "pitch" else int(head))
    if not 0 <= idx < n:
        raise SystemExit(f"head index {idx} outside the {n} Gaussian heads")
    if not LOG_STD_MIN <= value <= LOG_STD_MAX:
        raise SystemExit(f"log sigma {value:g} outside the clamp "
                         f"[{LOG_STD_MIN:g}, {LOG_STD_MAX:g}]")
    if (ck.get("config", {}).get("view_absolute") and idx == n - 1
            and value > PITCH_LOG_STD_MAX_ABS + 1e-9 and not allow_above_cap):
        raise SystemExit(f"log sigma {value:g} is above the absolute-mode "
                         f"pitch cap log 0.5 = {PITCH_LOG_STD_MAX_ABS:.4f}; "
                         "--allow-above-cap to write it anyway")
    before = float(ls[idx])
    ls[idx] = value
    adam_reset = False
    if reset_adam:
        # the parameter's optimizer slot: the state is keyed by the
        # parameter's position in Policy.parameters(); view_std.log_std is
        # LAST (Policy.__init__), so it is the highest key present
        st = ck["optimizer"]["state"]
        k = max(int(i) for i in st)
        for m in ("exp_avg", "exp_avg_sq"):
            t = st[k].get(m)
            if t is not None and t.numel() == n:
                t[idx] = 0.0
                adam_reset = True
    return {"index": idx, "before": before, "after": float(ls[idx]),
            "adam_reset": adam_reset}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--head", default="pitch",
                    help="yaw | pitch (the last Gaussian head) | index")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sigma", type=float)
    g.add_argument("--log-std", type=float)
    ap.add_argument("--reset-adam", action="store_true")
    ap.add_argument("--allow-above-cap", action="store_true")
    args = ap.parse_args(argv)
    if args.src == args.dst:
        raise SystemExit("dst must differ from src: the original stays")
    value = math.log(args.sigma) if args.sigma is not None else args.log_std
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    ref = torch.load(args.src, map_location="cpu", weights_only=False)
    r = set_log_std(ck, args.head, value, args.reset_adam,
                    args.allow_above_cap)
    # everything but the one entry (and its moments) is untouched
    for k in ref:
        if k == "policy":
            for kk in ref["policy"]:
                if kk != KEY:
                    assert _equal(ref["policy"][kk], ck["policy"][kk]), kk
            a, b = ref["policy"][KEY].clone(), ck["policy"][KEY].clone()
            a[r["index"]] = b[r["index"]] = 0.0
            assert torch.equal(a, b)
        elif k == "optimizer" and args.reset_adam:
            assert ref[k]["param_groups"] == ck[k]["param_groups"]
        else:
            assert _equal(ref[k], ck[k]), k
    torch.save(ck, args.dst)
    print(f"{KEY}[{r['index']}] ({args.head}): log sigma {r['before']:.4f} "
          f"(sigma {math.exp(r['before']):.4f}) -> {r['after']:.4f} "
          f"(sigma {math.exp(r['after']):.4f})"
          + ("; its Adam moments zeroed" if r["adam_reset"] else
             "; Adam moments left as they were")
          + f"; step {ck.get('global_step')}; wrote {args.dst}")


if __name__ == "__main__":
    main()
