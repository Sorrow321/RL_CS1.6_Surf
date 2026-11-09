"""az_diag.py - how well a tree-only checkpoint fits its own trees' labels (azr1b4, measurement only).

    python3 tools/az_diag.py <ckpt> <labels.npz> [...]   (run from the repo root on a GPU box)

Builds the network and the lidar the way tools/beam_tas.py does (its own main() is not reusable,
so this re-runs the same construction through record_ckpt's audit-free path), re-renders each
label row's image from its pose, and reports per label kind (0 node rows, 1 the best line's
searched rows, 2 its greedy prefix): the agreement of the argmax with the target's argmax on each
categorical head, the joint agreement, the yaw z error |mu - target| (median / 90th percentile)
and the value error. Reads only; trains nothing; writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def build(ckpt_path: str, device):
    import train_fast as _tf
    from beam_tas import XFeeds, build_sim
    from surfgym.goalfield import build_goal_field
    from surfgym.mapfleet import map_tag
    from surfgym.vision import GpuLidar, LidarPotential, pick_cell
    from surfgym.zones import load_zones
    from surfgym.rewards import map_spawn_pool
    from surfgym.tick import TickClock
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    _tf.set_keys_hold(bool(cfg.get("keys_hold")))
    TICK = TickClock(float(cfg.get("tick_ms") or 10.0))
    map_path = str(ROOT / "maps" / (Path(str(cfg.get("map"))).stem + ".bsp"))
    core1 = build_sim(cfg, map_path, 1, int(cfg.get("ep_ticks", 700)), tick=TICK)
    lw, lh = int(cfg.get("lidar_w", 128)), int(cfg.get("lidar_h", 64))
    cell = float(cfg.get("lidar_cell") or pick_cell(core1))
    zones = load_zones(core1.bsp_path)
    gf = build_goal_field(core1, zones["end"], cell=cell)
    ogf = (build_goal_field(core1, zones["end"], cell=cell, no_corner_cut=True)
           if int(cfg.get("race_no_corner_cut") or 0) else gf)
    raw = map_spawn_pool(core1)
    d0 = float(np.mean(gf.sample(raw["origin"])))
    lidar = GpuLidar(core1, lw, lh, hfov_deg=float(cfg.get("lidar_hfov") or 120.0),
                     vfov_deg=float(cfg.get("lidar_vfov") or 90.0),
                     range_units=float(cfg.get("lidar_range", 2000.0)),
                     near_range=cfg.get("lidar_near"), cell=cell, device=device,
                     potential=LidarPotential.from_cfg(cfg, ogf, core1, device,
                                                       Path(map_path).stem,
                                                       d0=(d0 if ogf is gf else None)),
                     vision_clip=bool(int(cfg.get("vision_clip") or 0)),
                     depth_enc=str(cfg.get("depth_enc") or "legacy"),
                     texture=bool(cfg.get("obs_texture") or cfg.get("obs_normal")
                                  or cfg.get("obs_slope") or 0),
                     texture_mode=("normal" if cfg.get("obs_normal") else "slope"
                                   if cfg.get("obs_slope") else "rgb"),
                     depth_log_d0=float(cfg.get("depth_log_d0") or 200.0))
    XF = XFeeds(cfg, ogf, TICK.ms, int(cfg.get("act_every", 1)))
    n_keys = _tf.keyshold.N_FEATURES if cfg.get("keys_hold") else 0
    VIEWC = bool(cfg.get("view_continuous"))
    policy = _tf.Policy(core1.obs_dim + n_keys + XF.n_columns
                        + int(getattr(lidar, "frame_size", lw * lh * lidar.channels)),
                        lw, lh, emb=int(cfg.get("emb", 256)), hidden=int(cfg.get("hidden", 256)),
                        gps=bool(cfg.get("gps", True)), trunk=str(cfg.get("trunk") or "plain"),
                        tower_depth=int(cfg.get("tower_depth") or 2),
                        conv_mult=int(cfg.get("conv_mult") or 1),
                        in_ch=int(getattr(lidar, "conv_channels", lidar.channels)),
                        simba=bool(cfg.get("simba") or 0), route_dim=n_keys + XF.n_columns,
                        view_continuous=VIEWC, view_absolute=cfg.get("view_absolute") or None,
                        pitch_sigma_min=float(cfg.get("pitch_sigma_min") or 0.0)).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, _tf.HeadPacker(device), lidar, int(ck.get("global_step", 0))


def main():
    import train_fast as _tf
    from az_loop import render_img
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, packer, lidar, step = build(sys.argv[1], dev)
    print(f"ckpt {sys.argv[1]} step {step:,}")
    nv = int(_tf.N_VIEW)
    for f in sys.argv[2:]:
        z = np.load(f, allow_pickle=False)
        kind = z["kind"]
        print(f"labels {f}: {len(kind):,} rows")
        for kd, name in ((0, "node"), (1, "line"), (2, "prefix")):
            ix = np.flatnonzero(kind == kd)
            if not len(ix):
                continue
            agree_h, joint, yerr, verr = [], [], [], []
            for s in range(0, len(ix), 2048):
                j = ix[s:s + 2048]
                scal = torch.as_tensor(z["scal"][j], device=dev)
                pose = torch.as_tensor(z["pose"][j], device=dev)
                with torch.inference_mode():
                    img = render_img(lidar, pose)
                    logits, value = policy.forward_split(scal, img)
                    cat, mu = _tf.split_view(logits.float())
                    padded = packer.pad(cat)
                am = padded[:, nv:].argmax(-1).cpu().numpy()
                tg = z["probs"][j][:, nv:].argmax(-1)
                agree_h.append(am == tg)
                joint.append((am == tg).all(-1))
                yerr.append(np.abs(mu[:, 0].cpu().numpy() - np.clip(z["vmu"][j][:, 0], -3, 3)))
                verr.append(np.abs(value.float().reshape(-1).cpu().numpy() - z["value"][j]))
            ah = np.concatenate(agree_h).mean(0)
            jt = np.concatenate(joint).mean()
            ye = np.concatenate(yerr)
            ve = np.concatenate(verr)
            print(f"  {name:6s} {len(ix):7,} rows: per-head acc fwd/side/jump/duck "
                  + "/".join(f"{x:.3f}" for x in ah) + f", joint {jt:.3f}, yaw |dz| median "
                  f"{np.median(ye):.3f} p90 {np.percentile(ye, 90):.3f}, value |err| median "
                  f"{np.median(ve):.4f}")


if __name__ == "__main__":
    main()
