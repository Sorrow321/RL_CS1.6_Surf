#!/usr/bin/env python3
"""gpu_health.py — is this rented GPU actually delivering its rated speed?

A rented card can be the right model, cool, well under its power limit, show
no other tenants, and still be a third slower than the same model elsewhere.
Measured on a Threadripper 2x5090 instance: sustained bf16 GEMM ran at 166
TFLOPS against 234 on a healthy 5090, because the SM clock sat at 1987 MHz
instead of ~2890 while drawing 392 W of a 575 W limit and sitting at 62 C.
"SW Power Cap" was Active with no way to raise it from inside the container.
That box had the fastest CPU of any measured and was still 21% slower per
GPU, so the CPU spec on the listing tells you nothing about this.

Run it before benchmarking anything on a new box. Two numbers, 30 seconds,
and it fails loudly rather than quietly costing you a fifth of the card:

    python3 tools/gpu_health.py
    python3 tools/gpu_health.py --all          # every visible GPU
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys

import torch

# measured on a healthy RTX 5090 (vast.ai, driver 580.x, torch 2.13+cu130)
REFERENCE = {
    "NVIDIA GeForce RTX 5090": {"gbps": 1524.0, "tflops": 234.0, "sm_mhz": 2890},
    # measured across three healthy vast 4x3090 boxes, 2026-08-23; the
    # power-capped box on the same day read 30-35 TFLOPS
    "NVIDIA GeForce RTX 3090": {"gbps": 840.0, "tflops": 69.0, "sm_mhz": 1695},
}


def _watts(v):
    """'350.00 W' -> 350.0; None when nvidia-smi says [N/A]."""
    try:
        return float(str(v).split()[0])
    except (TypeError, ValueError, IndexError):
        return None
TOLERANCE = 0.90          # below this fraction of reference = flag the box


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def smi(query, idx):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader",
             "-i", str(idx)], capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception:
        return "?"


def check(idx: int) -> bool:
    torch.cuda.set_device(idx)
    name = torch.cuda.get_device_name(idx)
    ref = REFERENCE.get(name)

    # a busy GPU measures low for an honest reason, and reporting that as a
    # capped card would be worse than not checking at all
    busy = smi("memory.used", idx)
    try:
        if int(busy.split()[0]) > 512:
            # NOT healthy - same trap as the missing-reference path. A
            # foreign tenant on the card is precisely the condition a
            # pre-rental check exists to catch (one vast box arrived with
            # all four GPUs already busy, 2026-08-23).
            print(f"GPU{idx}  BUSY ({busy} in use) - cannot be judged; stop "
                  f"other work, or on a rented box treat this as a FOREIGN "
                  f"TENANT and pick another instance.")
            return False
    except (ValueError, IndexError):
        pass

    a = torch.empty(1_000_000_000, dtype=torch.bfloat16, device=f"cuda:{idx}")
    b = torch.empty_like(a)
    gbps = 2 * a.numel() * 2 / 1e9 / (timed(lambda: b.copy_(a)) / 1e3)
    del a, b
    torch.cuda.empty_cache()

    n = 8192
    x = torch.randn(n, n, device=f"cuda:{idx}", dtype=torch.bfloat16)
    y = torch.randn(n, n, device=f"cuda:{idx}", dtype=torch.bfloat16)
    # sustained, so a clock cap shows up rather than a boost window hiding it
    tflops = 2 * n ** 3 / 1e12 / (timed(lambda: torch.mm(x, y), iters=60) / 1e3)
    sm = smi("clocks.sm", idx)
    del x, y
    torch.cuda.empty_cache()

    print(f"GPU{idx}  {name}")
    print(f"  HBM copy    {gbps:8,.0f} GB/s"
          + (f"   ref {ref['gbps']:,.0f}   {gbps / ref['gbps']:.0%}" if ref else ""))
    print(f"  bf16 GEMM   {tflops:8,.0f} TFLOPS"
          + (f"   ref {ref['tflops']:,.0f}   {tflops / ref['tflops']:.0%}" if ref else ""))
    print(f"  under load  sm {sm}, {smi('power.draw', idx)} of "
          f"{smi('power.limit', idx)}, {smi('temperature.gpu', idx)} C")
    # A power cap is visible without any reference: compare the enforced
    # limit to the board's default. Four 3090s capped to 180 W of 350 W
    # passed this script as "healthy" on 2026-08-23 (30-35 TFLOPS against a
    # healthy 69-73) purely because the model was missing from REFERENCE.
    lim = _watts(smi("power.limit", idx))
    dflt = _watts(smi("power.default_limit", idx))
    capped = lim is not None and dflt is not None and lim < 0.85 * dflt
    # A cap only MATTERS if the card also underperforms. Vast 3090s are
    # very often capped and still deliver 100% of reference bandwidth and
    # 98-105% of reference bf16 - two such boxes were blacklisted on
    # 2026-08-24 purely by this heuristic, and bad_hosts.json is near
    # saturation. So the cap is evidence only where there is nothing better:
    # with a reference, the MEASUREMENT decides and the cap is a note.
    if capped:
        print(f"  note: power limit {lim:.0f} W of {dflt:.0f} W default "
              f"({lim / dflt:.0%})"
              + ("" if ref else " - and no reference to check it against, "
                                "so treating it as a defect"))
    if not ref:
        # NOT healthy: unjudged is not the same as fine, and returning True
        # here is what let the capped box through.
        print("  (no reference for this model - UNJUDGED, treat as suspect)")
        return not capped
    # measurement beats the cap heuristic whenever a reference exists
    ok = (tflops >= TOLERANCE * ref["tflops"]
          and gbps >= TOLERANCE * ref["gbps"])
    if not ok:
        why = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE", "-i",
                              str(idx)], capture_output=True, text=True)
        active = [l.strip() for l in why.stdout.splitlines()
                  if ": Active" in l]
        print(f"  *** UNHEALTHY: below {TOLERANCE:.0%} of reference ***")
        for l in active:
            print(f"      clocks event: {l}")
        print("      A capped card cannot be fixed from inside the container "
              "(-lgc needs host privileges). Switch instances.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="check every GPU")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    idxs = range(torch.cuda.device_count()) if args.all else [args.gpu]
    ok = all([check(i) for i in idxs])
    print("\nVERDICT:", "healthy" if ok else "UNHEALTHY - do not benchmark on this box")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
