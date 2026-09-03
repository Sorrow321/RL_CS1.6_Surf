"""Collapse a --quantiles checkpoint's distributional critic to the scalar
critic the mainline trainer carries, EXACTLY, so the qr-run arm's finisher
(runs/research/xQR32, 9/9 finishes, 77.86 s) can seed an expert-iteration
loop on a trainer that has no --quantiles flag.

The quantile head is one nn.Linear(hidden, N): quantile i is
W[i] . h + b[i]. Everything the trainer consumes (GAE, the bootstrap, the
actor's advantage) is the MEAN of the quantiles (train_fast.py on the
qr-run branch: "GAE and the bootstrap consume the quantile MEAN"), and
the mean of N affine maps is one affine map:

    mean_i (W[i] . h + b[i]) = (mean_i W[i]) . h + mean_i b[i]

so a (1, hidden) head with the row-mean weights computes the very same
value function for every input, to fp32 rounding. The ACTOR is untouched:
every other tensor is copied byte for byte, so the greedy line and the
planner's proposals are the xQR32 policy exactly.

The Adam moments of the two collapsed tensors are averaged the same way
(row-mean of exp_avg / exp_avg_sq). exp_avg of the mean row is the mean of
the row exp_avgs (linear); the averaged exp_avg_sq over-estimates the
variance of the mean-row gradient, which makes the first steps of the
collapsed critic slightly SMALLER than Adam would take - conservative, and
it washes out within ~1/(1-beta2) = 1000 updates. The step count is kept.

    python tools/ckpt_qr_to_scalar.py runs/research/xQR32/xQR32_final.pt \
        runs/exit/seed_scalar.pt
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

VALUE_KEYS = ("value_head.weight", "value_head.bias")


def collapse_state_dict(sd: dict) -> tuple[dict, int]:
    """(new state dict, N) - N = 0 when the head was scalar already."""
    w, b = sd[VALUE_KEYS[0]], sd[VALUE_KEYS[1]]
    n = int(w.shape[0])
    if n == 1:
        return dict(sd), 0
    out = dict(sd)
    out[VALUE_KEYS[0]] = w.mean(0, keepdim=True).contiguous()
    out[VALUE_KEYS[1]] = b.mean(0, keepdim=True).contiguous()
    return out, n


def collapse_optimizer(opt: dict, param_names: list, n: int) -> dict:
    """Row-mean the Adam moments of the collapsed tensors; param index i is
    the i-th entry of the policy's state_dict order (train_fast builds the
    optimizer from policy.parameters(), the same order)."""
    if n <= 1:
        return opt
    out = {"param_groups": opt["param_groups"], "state": dict(opt["state"])}
    for name in VALUE_KEYS:
        i = param_names.index(name)
        st = dict(out["state"][i])
        for k in ("exp_avg", "exp_avg_sq"):
            v = st[k]
            assert int(v.shape[0]) == n, (name, k, tuple(v.shape))
            st[k] = v.mean(0, keepdim=True).contiguous()
        out["state"][i] = st
    return out


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--check", type=int, default=256,
                    help="random hidden vectors to verify the collapse on")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["policy"]
    names = list(sd.keys())
    new_sd, n = collapse_state_dict(sd)
    cfg = dict(ck.get("config") or {})
    if n == 0:
        print(f"{src.name}: value head is already scalar "
              f"(quantiles={cfg.get('quantiles')!r}) - copying")
    else:
        # exactness check: mean of the N quantiles == the collapsed head
        hid = int(sd[VALUE_KEYS[0]].shape[1])
        g = torch.Generator().manual_seed(0)
        h = torch.randn(args.check, hid, generator=g)
        want = (h @ sd[VALUE_KEYS[0]].T + sd[VALUE_KEYS[1]]).mean(-1)
        got = (h @ new_sd[VALUE_KEYS[0]].T + new_sd[VALUE_KEYS[1]]).squeeze(-1)
        err = float((want - got).abs().max())
        tol = 1e-5 * max(1.0, float(want.abs().max()))
        if err > tol:
            raise SystemExit(f"collapse is not exact: max |dV| {err:g} > {tol:g}")
        print(f"{src.name}: {n} quantiles -> scalar head, max |dV| {err:.2e} "
              f"over {args.check} random features (tol {tol:.1e})")
    ck["policy"] = new_sd
    if "optimizer" in ck and n > 1:
        ck["optimizer"] = collapse_optimizer(ck["optimizer"], names, n)
    # the mainline trainer / recorder must not see the arm's knobs: the
    # recorder's audit refuses any non-None key it does not mirror
    cfg["quantiles"] = None
    cfg["quantile_kappa"] = None
    cfg["qr_source"] = str(src)
    ck["config"] = cfg
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    print(f"wrote {dst}  step {int(ck.get('global_step', 0)):,}  "
          f"md5 {md5(dst)}  (source md5 {md5(src)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
