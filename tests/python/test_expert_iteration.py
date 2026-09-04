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


# ==========================================================================
# From-scratch expert iteration (--objective progress|auto)
# ==========================================================================
# 6. the progress ranking: finishers first by time, then best arc DESC,
#    ties by the tick the arc was reached at ASC, distinct tables only
# --------------------------------------------------------------------------
def _lin(finish=0, arc=0.0, arc_tick=0, end=0, fill=0, n=4):
    return {"finish_tick": finish, "best_arc": arc, "arc_tick": arc_tick,
            "end_tick": end, "acts": np.full((n, 6), fill, np.int8)}


def test_progress_ranking_finishers_first_then_arc_then_time():
    from surfgym.bc import rank_lineages
    cands = [_lin(arc=100.0, arc_tick=50, end=80, fill=1),
             _lin(arc=200.0, arc_tick=90, end=100, fill=2),
             _lin(finish=700, arc=900.0, arc_tick=699, end=700, fill=3),
             _lin(arc=200.0, arc_tick=70, end=100, fill=4),   # same arc, sooner
             _lin(finish=500, arc=900.0, arc_tick=499, end=500, fill=5),
             _lin(arc=200.0, arc_tick=90, end=100, fill=2)]   # duplicate table
    r = rank_lineages(cands)
    key = [(c["finish_tick"], c["best_arc"], c["arc_tick"]) for c in r]
    assert key == [(500, 900.0, 499), (700, 900.0, 699),
                   (0, 200.0, 70), (0, 200.0, 90), (0, 100.0, 50)]
    assert [int(c["acts"][0, 0]) for c in r] == [5, 3, 4, 2, 1]
    # k truncates AFTER de-duplication; a dead lineage keeps its best arc
    # (the 100 u line died at tick 80 and still ranks on its 100 u)
    assert [int(c["acts"][0, 0]) for c in rank_lineages(cands, 2)] == [5, 3]
    assert rank_lineages([], 3) == []


def test_lineage_hall_keeps_the_k_best_distinct_and_copies_lazily():
    from beam_tas import LineageHall
    hall = LineageHall(2)                       # cap = 4 * k = 8
    calls = {"n": 0}

    def table(v):
        def f():
            calls["n"] += 1
            return np.full((3, 6), v, np.int8)
        return f
    for v in range(12):                         # arcs 0..11, later = better
        hall.offer(float(v), 100 - v, 200, table(v))
    arcs = [c["best_arc"] for c in hall.items]
    assert arcs == [11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
    assert calls["n"] == 12                     # everything entered at first
    # a candidate worse than the worst kept is rejected WITHOUT copying
    assert hall.offer(1.0, 5, 300, table(13)) is False
    assert calls["n"] == 12
    # an equal arc reached LATER loses to the worst kept, sooner wins
    assert hall.offer(4.0, 97, 300, table(14)) is False
    assert hall.offer(4.0, 95, 300, table(14)) is True
    assert [c["arc_tick"] for c in hall.items if c["best_arc"] == 4.0] == [95]
    # a byte-identical table (a clone that died before diverging) enters once
    assert hall.offer(20.0, 1, 2, table(11)) is False
    assert len(hall.items) == 8 and hall.items[0]["best_arc"] == 11.0


# --------------------------------------------------------------------------
# 7. the last-contact trim (pick_selfline's rule over STATE_DTYPE states)
# --------------------------------------------------------------------------
def _states_from_rows(ep):
    from surfgym.core import STATE_DTYPE
    st = np.zeros(len(ep), STATE_DTYPE)
    st["origin"] = ep[:, 1:4]
    st["velocity"] = ep[:, 4:7]
    st["yaw"] = ep[:, 7]
    return st


def test_last_contact_trim_matches_pick_selfline_on_its_own_fixture():
    from pick_selfline import contact_cut, synth
    from surfgym.bc import contact_rows, last_contact_cut
    ep = synth()                                # contacts up to tick 300
    st = _states_from_rows(ep)
    assert np.allclose(contact_rows(st), ep)
    want = contact_cut(ep)
    assert want == (300, -8.0)
    assert last_contact_cut(st) == want         # data-derived gravity step
    assert last_contact_cut(st, gravity_step=-8.0) == want
    # no ballistic tick at all -> nothing trimmed; tiny input -> no raise
    assert last_contact_cut(_states_from_rows(synth(last_contact=0)))[0] == 399
    assert last_contact_cut(st[:2])[0] == 1


def test_last_contact_trim_uses_the_engine_gravity_and_reports_the_median():
    from pick_selfline import contact_cut, synth
    from plan_to_bc import trim_last_contact
    # 80 ticks sliding on a ramp (vz changes by -3/tick: geometry acting),
    # then 20 ticks of free fall (-8/tick). The MEDIAN diff(vz) is the
    # on-ramp -3, so pick_selfline's data-derived rule keeps the fall; the
    # engine step (-800 * 0.01) exposes it and the cut lands at the
    # departure - and the disagreement is reported, not hidden.
    n, ramp = 100, 80
    vz = np.zeros(n)
    for k in range(1, n):
        vz[k] = vz[k - 1] + (-3.0 if k <= ramp else -8.0)
    ep = np.zeros((n, 8))
    ep[:, 0] = np.arange(n)
    ep[:, 1] = np.arange(n) * 10.0
    ep[:, 3] = np.concatenate([[0.0], np.cumsum(vz[:-1])])
    ep[:, 6] = vz
    assert contact_cut(ep)[0] == n - 1                 # the median keeps it
    cut, g, src = trim_last_contact(_states_from_rows(ep), -8.0, 1.0)
    assert (cut, g) == (ramp, -8.0) and src.startswith("physics (median")
    # a mostly-ballistic line: the two rules agree tick for tick
    cut, g, src = trim_last_contact(_states_from_rows(synth()), -8.0, 1.0)
    assert (cut, g, src) == (300, -8.0, "physics=median")


def test_spine_subsample_is_uniform_along_the_path_and_keeps_both_ends():
    from surfgym.bc import subsample_by_path
    from surfgym.core import STATE_DTYPE
    # 50 ticks standing still, then 50 ticks at 40 u/tick: a time-uniform
    # spine puts half its mass on the standstill; a path-uniform one none
    st = np.zeros(100, STATE_DTYPE)
    st["origin"][50:, 0] = np.arange(50) * 40.0
    sel = subsample_by_path(st, 100.0)
    assert sel[0] == 0 and sel[-1] == 99
    assert not np.any((sel > 0) & (sel < 50))          # nothing mid-standstill
    o = st["origin"][sel, 0]
    gaps = np.diff(o)[1:-1]                     # between interior kept rows
    assert np.all(gaps >= 100.0 - 1e-6)         # never closer than spacing
    assert np.all(gaps <= 100.0 + 40.0 + 1e-6)  # never more than one tick over
    assert sel[1] == 53                         # the first tick past 100 u
    assert o[-1] - o[-2] < 100.0 + 40.0         # the forced last row
    assert np.array_equal(subsample_by_path(st, 0.0), np.arange(100))
    assert len(subsample_by_path(st[:0], 10.0)) == 0


# --------------------------------------------------------------------------
# 8. the auto switch: progress until a planner crossed / the policy
#    finishes, then finish time; and the wave ranking behind it
# --------------------------------------------------------------------------
def test_next_objective_auto_switches_once_a_planner_crossed_or_greedy_finishes():
    from expert_loop import next_objective
    assert next_objective("auto") == "auto"
    assert next_objective("auto", crossed_before=False, eval_finishes=0) == "auto"
    assert next_objective("auto", crossed_before=True) == "finish"
    assert next_objective("auto", eval_finishes=1) == "finish"
    assert next_objective("finish") == "finish"
    assert next_objective("progress", crossed_before=True, eval_finishes=9) \
        == "progress"
    with pytest.raises(ValueError):
        next_objective("sometimes")


def test_select_waves_ranks_crossings_by_time_then_arc_waves_by_arc_and_tick():
    from expert_loop import select_waves
    arc = [{"wave": 0, "crossed": False, "best_arc": 5000.0,
            "best_arc_tick": 900, "kept_lines": 3},
           {"wave": 1, "crossed": False, "best_arc": 7000.0,
            "best_arc_tick": 1200, "kept_lines": 2},
           {"wave": 2, "crossed": False, "best_arc": 7000.0,
            "best_arc_tick": 1100, "kept_lines": 2},
           {"wave": 3, "crossed": False, "best_arc": 9000.0,
            "best_arc_tick": 1500, "kept_lines": 0},      # nothing verified
           {"wave": 4, "rc": 1, "crossed": False}]          # crashed wave
    res, best, ranked = select_waves(arc, "auto")
    assert res == "progress" and best["wave"] == 2         # same arc, sooner
    assert [w["wave"] for w in ranked] == [2, 1, 0]
    assert select_waves(arc, "progress")[0] == "progress"
    # a fixed finish objective cannot use arc waves
    assert select_waves(arc, "finish") == (None, None, [])
    # any crossing outranks every arc wave, the fastest crossing first
    fin = [{"wave": 5, "crossed": True, "best_ticks": 7800, "best_s": 78.0},
           {"wave": 6, "crossed": True, "best_ticks": 7650, "best_s": 76.5}]
    res, best, ranked = select_waves(arc + fin, "auto")
    assert res == "finish" and best["wave"] == 6
    assert [w["wave"] for w in ranked] == [6, 5, 2, 1, 0]
    assert select_waves([], "auto") == (None, None, [])


# --------------------------------------------------------------------------
# 9. the summary schema (one expert_summary.jsonl line per round)
# --------------------------------------------------------------------------
def test_summary_row_has_exactly_the_documented_keys_and_is_json():
    import json
    from expert_loop import SUMMARY_KEYS, summary_row
    row = summary_row(round=3, objective="auto", planner_best_arc=12345.5,
                      greedy_in_finishes="0/9", planner_crossed=False)
    assert tuple(row) == SUMMARY_KEYS
    assert row["round"] == 3 and row["planner_best_s"] is None
    assert json.loads(json.dumps(row)) == row
    for key in ("objective", "planner_objective", "planner_best_arc",
                "planner_best_arc_s", "planner_finishes",
                "greedy_in_corridor_max", "greedy_in_corridor_mean",
                "greedy_out_corridor_max", "greedy_out_corridor_mean",
                "greedy_out_finishes", "spine_trim_ticks", "scratch_steps"):
        assert key in SUMMARY_KEYS
    with pytest.raises(KeyError):
        summary_row(round=0, planner_best_arcs=1.0)


def test_a_finished_dry_run_summary_matches_the_schema():
    import json
    from expert_loop import SUMMARY_KEYS
    f = ROOT / "runs" / "dry" / "expert_summary.jsonl"
    if not f.exists():
        pytest.skip("no runs/dry/expert_summary.jsonl (python tools/"
                    "expert_loop.py scratch --name dry --dry-run)")
    rows = [json.loads(ln) for ln in f.read_text(encoding="ascii").splitlines()
            if ln.strip()]
    assert rows
    for row in rows:
        assert tuple(row) == SUMMARY_KEYS
        assert row["objective"] in ("auto", "progress", "finish")
        assert row["planner_crossed"] or row["planner_best_arc"] is not None
        assert row["greedy_in_corridor_max"] is not None
        assert row["greedy_out_corridor_max"] is not None


# --------------------------------------------------------------------------
# 10. replay_line(keep_final=True) appends the post-action state once
# --------------------------------------------------------------------------
CFG_SCRATCH = {"reward": "race", "map": "surf_src_cannonball",
               "act_every": 4, "maxvel": 4000.0, "yaw_adaptive": True,
               "pitch_rate": 1.33, "teleport_fail": True, "lidar_w": W,
               "lidar_h": H, "lidar_cell": 32.0, "race_dist": "geodesic",
               "ep_ticks": 12000}


@needs_map
def test_replay_line_keep_final_appends_the_post_action_state_once():
    from plan_to_bc import open_planner_core
    core, _gf, _d0, _zones = open_planner_core(CFG_SCRATCH, str(MAP), 3000)
    k = 4
    obs = core.reset(5)
    row0 = core.get_states()[0:1].copy()
    o0 = obs[0].copy()
    acts = np.zeros((5, 6), np.int64)            # a few decisions, no death
    rows, ts, fin, ticks = replay_line(core, row0, o0, acts, k, None, None,
                                       max_ticks=3000, keep_final=True)
    assert not fin and ticks == len(acts) * k and len(rows) == len(acts)
    assert len(ts) == ticks + 1
    assert int(ts[-1]["tick"]) == ticks          # the state AFTER the table
    core.reset(5)
    _r2, ts2, _f2, t2 = replay_line(core, row0, o0, acts, k, None, None,
                                    max_ticks=3000)
    assert t2 == ticks and len(ts2) == ticks     # default: byte-identical
    assert np.array_equal(np.array(ts2), np.array(ts[:-1]))
    # the cap ends the replay, so keep_final adds nothing there
    core.reset(5)
    _r3, ts3, _f3, t3 = replay_line(core, row0, o0, acts, k, None, None,
                                    max_ticks=ticks, keep_final=True)
    assert t3 == ticks and len(ts3) == ticks
