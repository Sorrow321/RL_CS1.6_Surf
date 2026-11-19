"""gpupick.py - which card a CHILD process runs on (a search, a recording, a POV render).

mm/ddp, 2026-11-17. A child takes bare "cuda" (device 0 of what it sees) and inherits its
parent's CUDA_VISIBLE_DEVICES, so without a choice every search and every recording of a
multi-GPU box lands on card 0 - or, worse, on a training rank's card that is already full. One
SPEC, the same in every caller (the trainer's --search-gpu, the record gate's and the
dashboard's RECORD_GPU), turns into the CUDA_VISIBLE_DEVICES value handed to the child:

  inherit (or empty)  None: the child inherits, exactly as it always did
  own                 the caller's own card (a training rank's: --ddp-maps shard)
  free                the card with the most free memory at the moment the child starts
                      (nvidia-smi): a heavy rank's card is skipped by itself
  <n>                 nvidia-smi's card n
  GPU-<uuid>          that card

A card is always handed over by UUID, so nvidia-smi's numbering (PCI order) and CUDA's device
order can never disagree about which physical card is meant. Without nvidia-smi or CUDA (the
droplet), every spec falls back to inherit - a CPU child is pinned to nothing.

Nothing here reads a map: the choice is the box's memory, never a property of the maps.
"""
from __future__ import annotations

import os
import subprocess

__all__ = ["inventory", "torch_uuid", "visible_value", "child_env", "TORCHRUN_VARS"]

# torchrun's per-process variables: a child of a rank is not a rank
TORCHRUN_VARS = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                 "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
                 "MASTER_ADDR", "MASTER_PORT")


def inventory(timeout: float = 20.0):
    """Every card nvidia-smi lists: [{"index", "uuid", "free_mib", "total_mib"}], in its own
    order; None when nvidia-smi is missing or fails (no GPU here)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    out = []
    for ln in r.stdout.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 4:
            continue
        try:
            out.append({"index": int(parts[0]), "uuid": parts[1],
                        "free_mib": float(parts[2]), "total_mib": float(parts[3])})
        except ValueError:
            continue
    return out or None


def torch_uuid(device_index: int):
    """"GPU-<uuid>" of this process's CUDA device ``device_index`` (its own visible order), or
    None without CUDA / without the property."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        u = getattr(torch.cuda.get_device_properties(int(device_index)), "uuid", None)
        if u is None:
            return None
        u = str(u)
        return u if u.startswith(("GPU-", "MIG-")) else f"GPU-{u}"
    except Exception:
        return None


def visible_value(spec, own_uuid=None, inv=None):
    """(CUDA_VISIBLE_DEVICES value or None, a one-line note) for ``spec`` (see the module
    doc). ``own_uuid``: the caller's own card for "own"; ``inv``: an inventory() result to use
    (tests), else nvidia-smi is asked now."""
    s = str(spec or "").strip()
    if s in ("", "inherit"):
        return None, "inherit"
    if s.startswith(("GPU-", "MIG-")):
        return s, f"card {s}"
    if s == "own":
        if own_uuid:
            return str(own_uuid), f"own card {own_uuid}"
        return None, "own: no CUDA card here - inherit"
    inv = inventory() if inv is None else inv
    if not inv:
        return None, f"{s}: nvidia-smi lists no card here - inherit"
    if s == "free":
        best = max(inv, key=lambda g: (g["free_mib"], -g["index"]))
        return best["uuid"], (f"free: card {best['index']} {best['uuid']} "
                              f"({best['free_mib'] / 1024:.1f} of "
                              f"{best['total_mib'] / 1024:.1f} GiB free; "
                              + ", ".join(f"{g['index']}:{g['free_mib'] / 1024:.1f}"
                                          for g in inv) + ")")
    if s.isdigit():
        hit = [g for g in inv if g["index"] == int(s)]
        if not hit:
            raise ValueError(f"GPU spec {s!r}: nvidia-smi lists cards "
                             f"{[g['index'] for g in inv]}")
        return hit[0]["uuid"], f"card {s} {hit[0]['uuid']}"
    raise ValueError(f"GPU spec {s!r}: inherit, own, free, a card index or GPU-<uuid>")


def child_env(spec, own_uuid=None, base=None, drop_torchrun: bool = False, inv=None):
    """(env for subprocess or None, note): None = inherit (the caller passes env=None, which
    is what it always did). ``drop_torchrun`` strips torchrun's variables (a training rank's
    child is not a rank)."""
    val, note = visible_value(spec, own_uuid, inv)
    if val is None and not drop_torchrun:
        return None, note
    env = dict(os.environ if base is None else base)
    if drop_torchrun:
        for k in list(env):
            if k in TORCHRUN_VARS or k.startswith("TORCHELASTIC_"):
                env.pop(k)
    if val is not None:
        env["CUDA_VISIBLE_DEVICES"] = val
    return env, note
