"""Grow a checkpoint onto --frame-stack + --act-hist, ZERO-INIT (xMEM).

The arm's licence to compare against xCTL is that it starts ON the baseline
curve rather than near it. That needs the widened network to compute, on its
first forward, exactly the function the source checkpoint computed:

* conv.0.weight (16, 1, 5, 5) -> (16, K, 5, 5). The stack is NEWEST FIRST, so
  channel 0 is the current frame: the original filter goes there and channels
  1..K-1 are zero. Until training moves them, the trunk sees one frame.
* pi.0.weight and vf.0.weight gain M*6 TRAILING zero columns - the act-hist
  block is concatenated last in Policy.forward_split, exactly like the route
  block, so this is a pure zero-pad and never a permutation of existing
  columns.
* Adam's exp_avg / exp_avg_sq are padded the same way. The new weights have
  no history, and zero is what "no history" is.

Everything else (global_step, int_counts, the respawn reservoir) rides
through untouched, so the resume keeps the frontier it had.

The cfg gains frame_stack / stack_strides / act_hist, which is what makes a
plain `launch_local.ps1 resume` restore the WHOLE treatment - the trainer's
restore path reads those three keys and its mismatch guards then hold.

    python tools/inflate_ckpt_memory.py            # the xMEM defaults
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (LIDAR_H, LIDAR_W, NACT, N_SCALAR,  # noqa: E402
                        Policy, _parse_strides, frame_offsets,
                        set_stack_strides, widen_for_route)

SRC = r"C:\RL_Surf\runs\sOBSR2\ckpt_latest.pt"
SRC_MD5 = "1ba1fd2936af3ae1ad3608e3cd6b1e9e"
DST = r"C:\RL_Surf\runs\research\xMEM\ckpt_seed.pt"


def file_md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=SRC)
    ap.add_argument("--out", default=DST)
    ap.add_argument("--frame-stack", type=int, default=4)
    ap.add_argument("--stack-strides", default="5,10,15")
    ap.add_argument("--act-hist", type=int, default=15)
    ap.add_argument("--md5", default=SRC_MD5,
                    help="expected md5 of the SOURCE checkpoint ('' skips)")
    a = ap.parse_args(argv)

    src = Path(a.ckpt)
    if a.md5:
        got = file_md5(src)
        if got != a.md5:
            raise SystemExit(f"source md5 {got} != expected {a.md5}: this is "
                             "not the checkpoint the arm is defined against")
        print(f"source md5 {got} OK")

    ck = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ck.get("config") or {})
    sd = ck.get("policy") or {}

    # -- refuse, do not handle --------------------------------------------
    if "code_head.weight" in sd or sd.get("decoder") is not None:
        raise SystemExit("this checkpoint is CHUNKED (--chunk): one decision "
                         "emits a code, not a 6-tuple, so an action history "
                         "is not defined for it. Refusing.")
    w0 = sd.get("conv.0.weight")
    if w0 is None or w0.dim() != 4:
        raise SystemExit("no 4-D conv.0.weight: not the plain trunk")
    if int(w0.shape[1]) != 1:
        raise SystemExit(f"conv.0.weight has {int(w0.shape[1])} input "
                         "channels; this tool only grows a 1-channel trunk "
                         "(a --surf-mask or already-stacked ckpt needs its "
                         "own reasoning about what each channel means)")
    if int(cfg.get("frame_stack") or 0) > 1 or int(cfg.get("act_hist") or 0):
        raise SystemExit("the source checkpoint already carries a frame "
                         "stack or an action history")

    K = max(1, int(a.frame_stack))
    M = max(0, int(a.act_hist))
    strides = _parse_strides(a.stack_strides)
    set_stack_strides(strides)
    offs = frame_offsets(K)
    act_dim = M * NACT

    lw = int(cfg.get("lidar_w") or LIDAR_W)
    lh = int(cfg.get("lidar_h") or LIDAR_H)
    route_dim = 0
    if cfg.get("route_file") or cfg.get("race_latch") or \
            cfg.get("race_latch_frac"):
        raise SystemExit("the source checkpoint carries a route/latch block; "
                         "this tool was written for the plain scalar row")
    in_ch = int(cfg.get("surf_mask") or 0)
    in_ch = 2 if in_ch else 1
    if in_ch != 1:
        raise SystemExit("--surf-mask checkpoints are a different trunk")

    obs_dim = N_SCALAR + route_dim + act_dim + lw * lh * in_ch * K
    policy = Policy(obs_dim, lw, lh,
                    emb=int(cfg.get("emb", 512)),
                    hidden=int(cfg.get("hidden", 448)),
                    gps=bool(cfg.get("gps", False)),
                    trunk=str(cfg.get("trunk") or "plain"),
                    extra_feat=(12,) if cfg.get("obs_reward") else (),
                    in_ch=in_ch * K, route_dim=route_dim, act_dim=act_dim)

    # widen_for_route REBINDS ck["policy"]'s entries in place, so the
    # before-and-after comparison has to hold its own copies
    w0 = w0.clone()
    was = {k: int(v.shape[1]) for k, v in sd.items() if v.dim() >= 2}
    n = widen_for_route(ck, policy)
    print(f"padded {n} tensor(s) (weights + Adam moments)")

    # the pad must be provably a no-op at step 0
    nw = ck["policy"]["conv.0.weight"]
    assert tuple(nw.shape) == (w0.shape[0], K) + tuple(w0.shape[2:]), nw.shape
    assert torch.equal(nw[:, 0:1], w0), "channel 0 is not the original filter"
    assert float(nw[:, 1:].abs().sum()) == 0.0, "new channels are not zero"
    for t in ("pi.0.weight", "vf.0.weight"):
        g = ck["policy"][t]
        assert int(g.shape[1]) == was[t] + act_dim, f"{t} width {g.shape}"
        assert float(g[:, -act_dim:].abs().sum()) == 0.0, f"{t} tail nonzero"
    # and the moments the optimizer will step with
    ost = ((ck.get("optimizer") or {}).get("state")) or {}
    params = list(policy.parameters())
    for i, st in ost.items():
        want = params[int(i)].shape
        for key in ("exp_avg", "exp_avg_sq"):
            assert tuple(st[key].shape) == tuple(want), \
                f"optimizer {key}[{i}] {tuple(st[key].shape)} != {tuple(want)}"

    # a strict load is the real proof that every remaining key already fits
    policy.load_state_dict(ck["policy"])

    cfg["frame_stack"] = K
    cfg["stack_strides"] = list(strides)
    cfg["act_hist"] = M
    ck["config"] = cfg

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    print(f"wrote {out}")
    print(f"  step        {int(ck.get('global_step', 0)):,}")
    print(f"  frame_stack {K}  offsets {list(offs)} decisions "
          f"({', '.join(f'{o * 3 * 10}ms' for o in offs)} at act_every 3)")
    print(f"  act_hist    {M} decisions = {act_dim} scalars")
    print(f"  obs_dim     {obs_dim}")
    print(f"  md5         {file_md5(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
