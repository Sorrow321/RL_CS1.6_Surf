"""Expert iteration (--bc-file, surfgym/bc.py, tools/plan_to_bc.py).

The distillation target has to be the row the policy actually saw when
the planner committed the action, and the trainer must be the same trainer
when the feature is off. Four things are pinned:

  1. **the BC row is the eval wrapper's row.** A planner line replayed by
     surfgym.bc.replay_line, stored through save_bc_dataset, loaded by
     BCDataset and assembled as [scal | latch | render(state)] is
     torch.equal to what train_fast's GreedyTorchPolicy._obs built at the
     same decisions on the same core - scalars (with the --obs-reward
     slot-12 mirror), latch column and depth image alike - and the policy's
     greedy actions on the assembled rows are the recorded actions;
  2. **the loss is the factored cross-entropy**: per-head log-softmax of
     the padded logits gathered at the planner's index, summed over the six
     heads, weight-averaged over rows;
  3. **a zero coefficient contributes zero gradient**, bit for bit, and the
     loop skips the term entirely;
  4. **the flag surface**: the flags exist, land in run.json, are refused by
     nothing in record_ckpt's audit (TRAIN_ONLY), and every BC line in the
     trainer sits behind `if args.bc_file` / `bc is not None`.

Plus the quantile-head collapse tools/ckpt_qr_to_scalar.py relies on: the
row-mean head computes the mean of the quantiles exactly.

    python -m pytest tests/python/test_expert_iteration.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.bc import (BCDataset, N_SCALAR, REWARD_SLOT,   # noqa: E402
                        make_eval_feeds, replay_line, save_bc_dataset)
from train_fast import (NVEC, GreedyTorchPolicy, HeadPacker,   # noqa: E402
                        Policy, logprob_entropy_padded)

# the MAIN checkout's map: a worktree copy has a different mtime and every
# cache (goal field, SDF) keys on it, so it would re-bake for 30 minutes
MAIN_MAP = Path("C:/RL_Surf/maps/surf_src_cannonball.bsp")
MAP = MAIN_MAP if MAIN_MAP.exists() else ROOT / "maps" / "surf_src_cannonball.bsp"
GOAL_CACHE = MAP.with_name("surf_src_cannonball.goal_32.npz")
SDF_CACHE = MAP.with_name("surf_src_cannonball.sdf_32.npz")
W, H = 64, 32
K = 3

# xQR32's observation-relevant config (tools/ckpt_qr_to_scalar.py seeds the
# loop with it): obs-reward slot 12 + the 6,996 u latch column
CFG = {"reward": "race", "map": "surf_src_cannonball", "obs_reward": True,
       "race_latch": 6996.0, "time_pen": 0.01, "race_shaping": 1.0,
       "gamma": 0.9995, "act_every": K, "maxvel": 4000.0,
       "yaw_adaptive": True, "pitch_rate": 1.33, "teleport_fail": True,
       "lidar_w": W, "lidar_h": H, "lidar_range": 11500.0,
       "lidar_near": 2000.0, "lidar_cell": 32.0, "race_dist": "geodesic",
       "ep_ticks": 12000}

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_map = pytest.mark.skipif(
    not (MAP.exists() and GOAL_CACHE.exists() and SDF_CACHE.exists()),
    reason="needs the cannonball map with its baked goal field + SDF")


# --------------------------------------------------------------------------
# 1. the BC row is the eval wrapper's row
# --------------------------------------------------------------------------
class _Recording(GreedyTorchPolicy):
    """GreedyTorchPolicy that keeps every decision's assembled row."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows, self.acts = [], []

    @torch.inference_mode()
    def _decide(self, obs):
        row = self._obs(obs)
        logits, _ = self.policy(row)
        act = self.packer.pad(logits).argmax(-1)
        self.rows.append(row.clone())
        a = act.to("cpu").numpy().astype(np.int32)
        self.acts.append(a[0].copy())
        return a


@gpu
@needs_map
def test_bc_rows_are_the_eval_wrappers_rows(tmp_path):
    from plan_to_bc import open_planner_core
    from surfgym.vision import GpuLidar, HAVE_TRITON
    if not HAVE_TRITON:
        pytest.skip("needs triton")
    device = torch.device("cuda")
    core, gf, d0, _zones = open_planner_core(CFG, str(MAP), 3000)
    lidar = GpuLidar(core, W, H, range_units=11500.0, near_range=2000.0,
                     cell=32.0, device=device)
    torch.manual_seed(3)
    policy = Policy(N_SCALAR + 1 + W * H, W, H, emb=32, hidden=24,
                    extra_feat=(REWARD_SLOT,), route_dim=1).to(device).eval()
    packer = HeadPacker(device)

    # -- reference: the trainer's own eval wrapper along a greedy episode --
    obs = core.reset(5)
    row0 = core.get_states()[0:1].copy()
    o0 = obs[0].copy()
    slot, rf, lf = make_eval_feeds(CFG, gf, d0, K)
    assert slot == REWARD_SLOT and rf is not None and lf is not None
    pol = _Recording(policy, packer, device, lidar, core, K, 1,
                     extra_slot=slot, extra_fn=rf, latch_fn=lf)
    n_ticks = 40 * K
    for _ in range(n_ticks):
        a = pol.act(obs)
        obs, _r, done, trunc, _t = core.step(a)
        if done[0] or trunc[0]:
            break
    ref = torch.cat(pol.rows, 0)
    acts = np.stack(pol.acts).astype(np.int64)
    assert len(acts) >= 5, "the random policy died at once; pick another seed"

    # -- the BC side: replay the same actions open-loop, save, load ------
    core.reset(5)
    _slot, rf2, lf2 = make_eval_feeds(CFG, gf, d0, K)
    rows, tick_states, finished, ticks = replay_line(
        core, row0, o0, acts, K, rf2, lf2, max_ticks=n_ticks)
    assert not finished and ticks == len(acts) * K
    assert len(rows) == len(acts) and len(tick_states) == ticks
    st = np.array([r[0] for r in rows])
    f = tmp_path / "bc.npz"
    save_bc_dataset(f, st, np.array([r[1] for r in rows], np.float32),
                    np.array([r[2] for r in rows], np.float32),
                    np.array([r[3] for r in rows]), np.ones(len(rows), np.float32),
                    np.zeros(len(rows), np.int32),
                    {"obs_reward": True, "n_latch": 1, "act_every": K})
    ds = BCDataset(f, device, n_latch=1, obs_reward=True, seed=0)
    assert ds.n == len(acts)
    full = torch.cat([ds.scal, ds.render(lidar, ds.pose, torch.float32)], 1)
    assert full.shape == ref.shape
    assert torch.equal(full[:, :N_SCALAR + 1], ref[:, :N_SCALAR + 1]), \
        "scalars/latch differ from the eval wrapper's row"
    assert torch.equal(full[:, N_SCALAR + 1:], ref[:, N_SCALAR + 1:]), \
        "the depth rendered from the stored state differs from the live one"
    # the slot-12 mirror is live (not the core's absolute position): after
    # the first decision it is a squashed reward, and the latch is a flag
    assert float(full[1:, REWARD_SLOT].abs().max()) <= 1.0
    assert set(full[:, N_SCALAR].tolist()) <= {0.0, 1.0}
    # and the policy's greedy actions on the assembled rows ARE the
    # recorded actions - the end-to-end statement of "same row"
    with torch.inference_mode():
        got = packer.pad(policy(full)[0]).argmax(-1).cpu().numpy()
    assert np.array_equal(got, acts)
    # the sampler draws from ITS generator and every column round-trips
    s, p, a, w = ds.sample(7)
    assert s.shape == (7, N_SCALAR + 1) and p.shape == (7, 6)
    assert a.shape == (7, 6) and w.shape == (7,)
    assert torch.equal(ds.act, torch.as_tensor(acts, device=device))


@gpu
@needs_map
def test_a_mismatched_layout_is_refused(tmp_path):
    st = np.zeros(3, dtype=np.dtype(BCDataset.__init__.__globals__["STATE_DTYPE"]))
    f = tmp_path / "bc.npz"
    save_bc_dataset(f, st, np.zeros((3, N_SCALAR), np.float32), np.zeros(3),
                    np.zeros((3, 6), np.int64), np.ones(3), np.zeros(3),
                    {"obs_reward": True, "n_latch": 1})
    with pytest.raises(SystemExit):
        BCDataset(f, torch.device("cuda"), n_latch=0, obs_reward=True)
    with pytest.raises(SystemExit):
        BCDataset(f, torch.device("cuda"), n_latch=1, obs_reward=False)
    BCDataset(f, torch.device("cuda"), n_latch=1, obs_reward=True)


# --------------------------------------------------------------------------
# 2. the loss is the factored cross-entropy
# --------------------------------------------------------------------------
def _bc_nll(padded, act, w):
    """train_fast's bc_loss_fn body on padded logits."""
    logp, _ = logprob_entropy_padded(padded, act)
    return -(logp * w).sum() / w.sum().clamp_min(1e-6)


def test_bc_loss_is_the_weighted_factored_cross_entropy():
    torch.manual_seed(0)
    packer = HeadPacker(torch.device("cpu"))
    B = 11
    logits = torch.randn(B, sum(NVEC))
    act = torch.stack([torch.randint(0, n, (B,)) for n in NVEC], 1)
    w = torch.rand(B) + 0.1
    got = _bc_nll(packer.pad(logits), act, w)
    # per head: cross-entropy of that head's own categorical
    off, ce = 0, torch.zeros(B)
    for h, n in enumerate(NVEC):
        ce = ce + F.cross_entropy(logits[:, off:off + n], act[:, h],
                                  reduction="none")
        off += n
    want = (ce * w).sum() / w.sum()
    assert torch.allclose(got, want, atol=1e-6)


# --------------------------------------------------------------------------
# 3. a zero coefficient is zero gradient
# --------------------------------------------------------------------------
def test_zero_coef_is_zero_gradient_bit_for_bit():
    torch.manual_seed(1)
    packer = HeadPacker(torch.device("cpu"))
    p = Policy(N_SCALAR + 8 * 4, 8, 4, emb=16, hidden=12)
    obs = torch.randn(6, N_SCALAR + 32)
    act = torch.stack([torch.randint(0, n, (6,)) for n in NVEC], 1)

    def ppo_like():
        logits, v = p(obs)
        return v.pow(2).mean() - logits.mean()

    p.zero_grad()
    ppo_like().backward()
    g0 = [q.grad.clone() for q in p.parameters()]
    p.zero_grad()
    coef = torch.zeros(())
    lb = _bc_nll(packer.pad(p(obs)[0]), act, torch.ones(6))
    (ppo_like() + coef * lb).backward()
    g1 = [q.grad for q in p.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(g0, g1))
    # and a NON-zero coefficient does move the gradient (the term is live)
    p.zero_grad()
    lb = _bc_nll(packer.pad(p(obs)[0]), act, torch.ones(6))
    (ppo_like() + 0.5 * lb).backward()
    g2 = [q.grad for q in p.parameters()]
    assert not all(torch.equal(a, b) for a, b in zip(g0, g2))


# --------------------------------------------------------------------------
# 4. the flag surface
# --------------------------------------------------------------------------
def test_the_flags_exist_are_recorded_and_are_guarded():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    for flag in ("--bc-file", "--bc-coef", "--bc-coef-final", "--bc-steps",
                 "--bc-batch"):
        assert f'"{flag}"' in src
    for key in ("bc_file", "bc_coef", "bc_coef_final", "bc_steps", "bc_batch"):
        assert f'"{key}":' in src                     # lands in run.json
    assert 'ck_cfg.get("bc_file")' not in src         # never auto-restored
    # every live BC line is guarded: the block, the coefficient, the
    # minibatch term, the diagnostics, the file close
    assert "if args.bc_file:" in src
    assert "if bc is not None and bc_coef_now > 0.0:" in src
    assert src.count("if bc is not None") >= 3
    assert "loss = loss + bc_coef_t * _lb" in src
    # the term is summed BEFORE the one backward, so the PPO minibatch step
    # (mb_step), the clip and the optimizer step are the untouched ones
    i_term = src.index("loss = loss + bc_coef_t * _lb")
    i_back = src.index("loss.backward()", i_term)
    assert "opt.zero_grad(set_to_none=True)" in src[i_term:i_back]
    rec = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    for key in ("bc_file", "bc_coef", "bc_coef_final", "bc_steps", "bc_batch",
                "qr_source"):
        assert f'"{key}"' in rec                      # TRAIN_ONLY: recordable


# --------------------------------------------------------------------------
# 5. the quantile-head collapse is exact
# --------------------------------------------------------------------------
def test_quantile_collapse_is_the_mean_of_the_quantiles():
    from ckpt_qr_to_scalar import collapse_optimizer, collapse_state_dict
    torch.manual_seed(4)
    n, hid = 32, 24
    sd = {"pi.0.weight": torch.randn(3, 3),
          "value_head.weight": torch.randn(n, hid),
          "value_head.bias": torch.randn(n)}
    new, got_n = collapse_state_dict(sd)
    assert got_n == n
    assert new["value_head.weight"].shape == (1, hid)
    assert new["value_head.bias"].shape == (1,)
    assert torch.equal(new["pi.0.weight"], sd["pi.0.weight"])
    h = torch.randn(50, hid)
    want = (h @ sd["value_head.weight"].T + sd["value_head.bias"]).mean(-1)
    got = (h @ new["value_head.weight"].T + new["value_head.bias"]).squeeze(-1)
    assert torch.allclose(got, want, atol=1e-5)
    # a scalar head is left alone
    new1, n1 = collapse_state_dict({"value_head.weight": torch.randn(1, hid),
                                    "value_head.bias": torch.randn(1)})
    assert n1 == 0 and new1["value_head.weight"].shape == (1, hid)
    # the Adam moments follow the same row-mean, indexed by state_dict order
    names = list(sd.keys())
    opt = {"param_groups": [{"params": [0, 1, 2]}],
           "state": {0: {"step": torch.tensor(5.0), "exp_avg": torch.randn(3, 3),
                         "exp_avg_sq": torch.rand(3, 3)},
                     1: {"step": torch.tensor(5.0), "exp_avg": torch.randn(n, hid),
                         "exp_avg_sq": torch.rand(n, hid)},
                     2: {"step": torch.tensor(5.0), "exp_avg": torch.randn(n),
                         "exp_avg_sq": torch.rand(n)}}}
    o2 = collapse_optimizer(opt, names, n)
    assert o2["state"][1]["exp_avg"].shape == (1, hid)
    assert torch.allclose(o2["state"][1]["exp_avg"],
                          opt["state"][1]["exp_avg"].mean(0, keepdim=True))
    assert o2["state"][2]["exp_avg_sq"].shape == (1,)
    assert torch.equal(o2["state"][0]["exp_avg"], opt["state"][0]["exp_avg"])
