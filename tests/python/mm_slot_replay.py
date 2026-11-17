"""Differential replay for the multi-map trainer (mm/bug audit, 2026-11-17).

Question: under ``--maps A,B,C`` does every env's training data (observation row, reward,
done, sampling temperature, entropy weight, spawn pool, value target inputs) equal what ITS
map computes alone, on the single-map code path, for the same state?

How: two kinds of in-process runs of the real ``python/train_fast.py main()`` (CPU):

* ``record`` - the multi-map run. Per iteration it saves the policy weights the rollout uses
  (at ``MapFleet.set_step``), every decision's sampled actions (``sample_view`` called from
  ``step_compute``), every eval's seed and weights (``record_rollout_parallel``), every
  training spawn pool (``SurfCore.set_spawn_pool``) and, at the first optimiser step of the
  iteration (``torch.nn.utils.clip_grad_norm_``, called inline in main), main's rollout
  buffers and per-map state.
* ``replay K`` - map K ALONE (``--map``, ``--envs PER``), i.e. the single-map code path, with
  the recorded weights loaded at every iteration and eval, the recorded actions of slot K's
  rows returned by ``sample_view``, the recorded eval seeds, and slot K's per-slot seed offsets
  (core reset +1013K, reservoir +101K, greedy pool +101K, front-perturb +K; the curriculum's
  seed is 29 on every slot). It saves the same snapshots.

``compare`` then checks slot K's rows of the multi-map buffers against the single-map run's,
bit for bit (values with a tolerance: the forward pass runs on a different batch).

Nothing here changes training behaviour: the wrappers only read, or (replay) substitute the
recorded value for a random draw / the weights.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def _np(x):
    import torch
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().copy()
    return np.array(x, copy=True)


def _h(a) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sd(policy):
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}


SNAP_MAIN = ("b_rew", "b_done", "b_val", "b_logp", "b_act", "b_z", "b_scal", "b_img",
             "b_ftk", "b_ftv", "adv", "ret", "ent_w_env", "ent_t", "demo_idx", "greedy_sp",
             "ep_len", "FRONT_THR_ENV", "FRONT_BASE_ENV", "front_thr", "global_step", "it_no",
             "unstuck_T", "UT_SL", "N", "T", "NMAPS")


def _slot_state(s):
    """per-slot state that later training data depends on (hashable summaries)"""
    rf = s.reward_fn
    out = {"int_coef": float(getattr(rf, "int_coef", float("nan"))),
           "en_front_d": float(getattr(rf, "en_front_d", float("nan")) or float("nan"))}
    c = getattr(rf, "_counts", None)
    if c is not None:
        out["counts_sum"] = int(np.asarray(c).sum())
        out["counts_nz"] = int(np.count_nonzero(c))
        out["counts_hash"] = _h(np.asarray(c))
    r = s.respawn
    if r is not None:
        out["res_size"] = int(r.size)
        out["res_hash"] = _h(np.asarray(r._store[:r.size]))
    return out


def run(mode: str, store: Path, k: int, argv: list) -> None:
    import torch
    import train_fast as TF
    from surfgym.mapfleet import MapFleet
    from surfgym.core import SurfCore

    rec = {"weights": [], "acts": {}, "evals": [], "pools": [], "snaps": {}, "meta": {}}
    ref = torch.load(store, weights_only=False) if mode == "replay" else None
    st = {"it": 0, "t": 0, "eval_i": 0, "snapped": set()}

    # ---- coverage variant (both modes alike): MM_SPINES=<npz of stem -> STATE rows> ----
    # a map's 2-row spawn spine becomes a long line of real states (the curriculum walks it,
    # its success line pays), and each map's reservoir is pre-filled past the 2,000 states the
    # greedy / front-perturb pool rows need. Identical inputs on both sides of the comparison.
    if os.environ.get("MM_SPINES"):
        _sp = np.load(os.environ["MM_SPINES"])
        SP = {s: _sp[s] for s in _sp.files}
        _DC, _RB0 = TF.DemoCurriculum, TF.RespawnBuffer

        def _stem_of(origin):
            o = np.asarray(origin, np.float64)
            for s, a in SP.items():
                if np.linalg.norm(np.asarray(a[0]["origin"], np.float64) - o) < 64.0:
                    return s
            return None

        class DC(_DC):
            def __init__(self, states, *a, **kw):
                st_ = np.asarray(states)
                s = _stem_of(st_[0]["origin"]) if len(st_) <= 2 else None
                super().__init__(SP[s] if s is not None else states, *a, **kw)

        class PRB(_RB0):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                rows = SP.get(str(kw.get("map_id", "")))
                if rows is not None:
                    self.push_many(np.resize(rows, max(2100, len(rows))))
        TF.DemoCurriculum, TF.RespawnBuffer = DC, PRB

    # ---- sensitivity check (record side only): MM_MUTATE=mind_slot0 makes every map's
    # band edges (--front-temp, --energy-pot-front, --unstuck reach) read slot 0's reservoir,
    # the kind of slot-0 leak the port had to remove - compare() must then fail slots 1..
    if mode == "record" and os.environ.get("MM_MUTATE") == "mind_slot0":
        _rmds = MapFleet.reservoir_min_depths

        def reservoir_min_depths(self):
            v = _rmds(self)
            return [v[0]] * len(v)
        MapFleet.reservoir_min_depths = reservoir_min_depths
    # MM_MUTATE=compass_slot0: the rollout's compass block reads slot 0's shaping field for
    # every env (visible from the first decision)
    if mode == "record" and os.environ.get("MM_MUTATE") == "compass_slot0":
        _OA = TF.ObsAux

        class OA(_OA):
            def __init__(self, n_envs, *a, field=None, **kw):
                if isinstance(field, list) and len(field) > 1:
                    field = [(slice(0, int(n_envs)), field[0][1])]
                super().__init__(n_envs, *a, field=field, **kw)
        TF.ObsAux = OA

    # ---- per-slot seed offsets (replay of slot k on the single-map path) ----
    if mode == "replay" and k:
        _reset = MapFleet.reset

        def reset(self, seed=0):
            return _reset(self, int(seed) + 1013 * k)
        MapFleet.reset = reset
        _RB, _GP, _FP = TF.RespawnBuffer, TF.GreedyLinePool, TF.FrontierPerturbRows

        class RB(_RB):
            def __init__(self, *a, **kw):
                kw["seed"] = int(kw.get("seed", 23)) + 101 * k
                super().__init__(*a, **kw)

        class GP(_GP):
            def __init__(self, *a, **kw):
                kw["seed"] = int(kw["seed"]) + 101 * k
                super().__init__(*a, **kw)

        class FP(_FP):
            def __init__(self, *a, **kw):
                kw["seed"] = int(kw["seed"]) + k
                super().__init__(*a, **kw)
        TF.RespawnBuffer, TF.GreedyLinePool, TF.FrontierPerturbRows = RB, GP, FP

    # ---- weights per iteration ----
    _set_step = MapFleet.set_step

    def set_step(self, global_step):
        fl = inspect.currentframe().f_back.f_locals
        pol = fl["policy"]
        st["it"] += 1
        st["t"] = 0
        if mode == "record":
            rec["weights"].append(_sd(pol))
        else:
            pol.load_state_dict(ref["weights"][st["it"] - 1])
        return _set_step(self, global_step)
    MapFleet.set_step = set_step

    # ---- actions per decision ----
    _sv = TF.sample_view

    def sample_view(padded, mu, log_std, temp=None, temp_view=None):
        caller = inspect.currentframe().f_back.f_code.co_name
        if caller != "step_compute":
            return _sv(padded, mu, log_std, temp, temp_view)
        key = (st["it"], st["t"])
        st["t"] += 1
        if mode == "record":
            act, z, logp = _sv(padded, mu, log_std, temp, temp_view)
            rec["acts"][key] = (_np(act), _np(z), _np(logp))
            return act, z, logp
        a, z, lp = ref["acts"][key]
        lo, hi = ref["meta"]["lo"][k], ref["meta"]["hi"][k]
        return (torch.as_tensor(a[lo:hi]), torch.as_tensor(z[lo:hi]),
                torch.as_tensor(lp[lo:hi]))
    TF.sample_view = sample_view

    # ---- evals: seed and weights ----
    _rrp = TF.record_rollout_parallel

    def record_rollout_parallel(core_, pol_, path_, episodes, seed=0, **kw):
        fl = inspect.currentframe().f_back.f_back.f_locals      # _rec_par -> main
        pol = fl["policy"]
        stem = Path(core_.bsp_path).stem
        if mode == "record":
            rec["evals"].append({"it": st["it"], "stem": stem, "seed": int(seed),
                                 "w": _sd(pol)})
        else:
            evs = [e for e in ref["evals"] if e["stem"] == stem]
            e = evs[st["eval_i"]]
            st["eval_i"] += 1
            assert e["it"] == st["it"], (e["it"], st["it"])
            pol.load_state_dict(e["w"])
            seed = e["seed"]
        return _rrp(core_, pol_, path_, episodes, seed=seed, **kw)
    TF.record_rollout_parallel = record_rollout_parallel

    # ---- spawn pools of the TRAINING cores ----
    _ssp = SurfCore.set_spawn_pool

    def set_spawn_pool(self, pool):
        rec["pools"].append({"it": st["it"], "stem": Path(self.bsp_path).stem,
                             "n": int(self.num_envs), "pool": np.array(pool, copy=True)})
        return _ssp(self, pool)
    SurfCore.set_spawn_pool = set_spawn_pool

    # ---- main's buffers at the first optimiser step of each iteration ----
    _cgn = torch.nn.utils.clip_grad_norm_

    def clip_grad_norm_(params, max_norm, *a, **kw):
        fr = inspect.currentframe().f_back
        if fr.f_code.co_name == "main" and st["it"] not in st["snapped"]:
            st["snapped"].add(st["it"])
            fl = fr.f_locals
            snap = {n: _np(fl[n]) if n in fl else None for n in SNAP_MAIN}
            slots = fl["slots"]
            snap["slots"] = [{"tag": s.tag, "lo": s.lo, "hi": s.hi, **_slot_state(s)}
                             for s in slots]
            # the curriculum of every map: DSL[k] under --maps (DSI says which one the
            # names hold), the single names on one map
            dsl, dsi = fl.get("DSL"), fl.get("DSI")
            dem = []
            for j in range(len(slots)):
                d = (fl["demo"] if (dsl is None or (dsi is not None and dsi[0] == j))
                     else dsl[j]["demo"])
                dem.append(None if d is None else
                           {"tau": int(d.tau), "n": int(d.n), "ep": d.ep.copy(),
                            "win": d.win.copy(), "S_hash": _h(np.asarray(d.S))})
            snap["demo"] = dem
            rec["snaps"][st["it"]] = snap
            if not rec["meta"]:
                rec["meta"] = {"lo": [s.lo for s in slots], "hi": [s.hi for s in slots],
                               "tags": [s.tag for s in slots],
                               "stems": [Path(s.core.bsp_path).stem for s in slots]}
        return _cgn(params, max_norm, *a, **kw)
    torch.nn.utils.clip_grad_norm_ = clip_grad_norm_

    sys.argv = ["train_fast.py"] + list(argv)
    try:
        TF.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    finally:
        torch.save(rec, store if mode == "record" else
                   store.with_name(store.stem + f"_replay{k}.pt"))


def compare(store: Path, k: int, val_tol: float = 1e-4) -> list:
    """slot k of the multi-map record vs the single-map replay of map k -> a list of findings
    (empty = identical). Every row of every quantity, every iteration both runs have."""
    import torch
    A = torch.load(store, weights_only=False)
    B = torch.load(store.with_name(store.stem + f"_replay{k}.pt"), weights_only=False)
    lo, hi = A["meta"]["lo"][k], A["meta"]["hi"][k]
    n = hi - lo
    bad = []
    its = sorted(set(A["snaps"]) & set(B["snaps"]))
    if not its:
        return ["no common iterations"]
    t0_temp, boot_noise = [], []
    for it in its:
        a, b = A["snaps"][it], B["snaps"][it]
        assert int(b["N"]) == n, (b["N"], n)
        for name in ("b_rew", "b_done", "b_scal", "b_img", "b_ftk", "b_ftv", "b_act", "b_z",
                     "b_logp"):
            xa, xb = a.get(name), b.get(name)
            if xa is None or xb is None:
                continue
            ra = np.asarray(xa)[:, lo:hi]
            rb = np.asarray(xb)
            if ra.shape != rb.shape:
                bad.append(f"it {it} {name}: shape {ra.shape} vs {rb.shape}")
                continue
            if name in ("b_ftk", "b_ftv"):
                # decision 0 of a rollout: the single-map path's set_unstuck_temp() (end of
                # every iteration) overwrites the per-env tensors with the uniform 1 + T, the
                # --maps path does not call it - reported apart, compared from decision 1 on
                if not np.array_equal(ra[0], rb[0]):
                    t0_temp.append((it, name, float(np.mean(ra[0])), float(np.mean(rb[0]))))
                ra, rb = ra[1:], rb[1:]
            if not np.array_equal(ra, rb):
                d = np.abs(ra.astype(np.float64) - rb.astype(np.float64))
                w = np.argwhere(d > 0)
                if name == "b_rew":
                    # a TRUNCATED row's reward carries gamma x V(s_T), and V is evaluated on
                    # the decision's truncated rows as one batch (other maps' rows too under
                    # --maps): float noise of the forward pass, nothing else
                    dn = np.asarray(a["b_done"])[:, lo:hi]
                    if d.max() <= 1e-5 and all(dn[t, e] > 0 for t, e in w):
                        boot_noise.append((it, len(w), float(d.max())))
                        continue
                t0, e0 = int(w[0][0]), int(w[0][1])
                bad.append(f"it {it} {name}: {len(np.unique(w[:, 1]))}/{n} envs differ, max "
                           f"|d| {d.max():.6g}, first at decision {t0} env {e0} "
                           f"(multi {ra[t0, e0].ravel()[:6]} vs alone {rb[t0, e0].ravel()[:6]})")
        va, vb = np.asarray(a["b_val"])[:, lo:hi], np.asarray(b["b_val"])
        if not np.allclose(va, vb, atol=val_tol, rtol=val_tol):
            bad.append(f"it {it} b_val: max |d| {np.abs(va - vb).max():.6g}")
        # the entropy bonus each row is trained with: ent_t x ent_w_env[env] (multi, ENT_ROW)
        # vs ent_t x 1 (one map: the (1 + T) factor is inside ent_t)
        ea = float(np.asarray(a["ent_t"])) * np.asarray(a["ent_w_env"])[lo:hi]
        eb = float(np.asarray(b["ent_t"])) * np.asarray(b["ent_w_env"])
        if not np.allclose(ea, eb, rtol=1e-6, atol=0):
            bad.append(f"it {it} entropy coef per row: multi {ea[:3]} vs alone {eb[:3]}")
        # N-wide trainer state: the spine row of each env's spawn, the greedy-row flag, the
        # episode clock (all live at the update, i.e. after the rollout's last decision)
        for name in ("demo_idx", "greedy_sp", "ep_len"):
            xa, xb = a.get(name), b.get(name)
            if xa is not None and xb is not None and \
                    not np.array_equal(np.asarray(xa)[lo:hi], np.asarray(xb)):
                bad.append(f"it {it} {name}: multi {np.asarray(xa)[lo:hi][:8]} vs alone "
                           f"{np.asarray(xb)[:8]}")
        # the --front-temp band edge each env is held to: FRONT_THR_ENV (multi) vs front_thr
        # (one map); the out-of-band base T: FRONT_BASE_ENV (multi) vs unstuck_T (one map)
        if a.get("FRONT_THR_ENV") is not None and b.get("front_thr") is not None:
            fa = np.asarray(a["FRONT_THR_ENV"])[lo:hi]
            fb = float(np.asarray(b["front_thr"]))
            if not np.all((fa == fb) | (np.isinf(fa) & np.isinf(fb))):
                bad.append(f"it {it} front band edge: multi {fa[:3]} vs alone {fb}")
        if a.get("FRONT_BASE_ENV") is not None and b.get("unstuck_T") is not None:
            fa = np.asarray(a["FRONT_BASE_ENV"])[lo:hi]
            fb = float(np.asarray(b["unstuck_T"]))
            if not np.all(fa == fb):
                bad.append(f"it {it} out-of-band T: multi {fa[:3]} vs alone {fb}")
        sa, sb = a["slots"][k], b["slots"][0]
        for key in sorted(set(sa) & set(sb) - {"tag", "lo", "hi"}):
            if not (sa[key] == sb[key] or (sa[key] != sa[key] and sb[key] != sb[key])):
                bad.append(f"it {it} slot state {key}: multi {sa[key]} vs alone {sb[key]}")
        da, db = a["demo"][k], b["demo"][0]
        if (da is None) != (db is None):
            bad.append(f"it {it} curriculum present: {da is not None} vs {db is not None}")
        elif da is not None:
            for key in ("tau", "n", "S_hash"):
                if da[key] != db[key]:
                    bad.append(f"it {it} curriculum {key}: {da[key]} vs {db[key]}")
            for key in ("ep", "win"):
                if not np.array_equal(da[key], db[key]):
                    bad.append(f"it {it} curriculum {key} differs")
    # spawn pools of the training core, in order
    stem = A["meta"]["stems"][k]
    pa = [p for p in A["pools"] if p["stem"] == stem and p["n"] == n]
    pb = [p for p in B["pools"] if p["stem"] == stem and p["n"] == n]
    if len(pa) != len(pb):
        bad.append(f"spawn pools: {len(pa)} set on slot {k} vs {len(pb)} alone")
    for i, (p, q) in enumerate(zip(pa, pb)):
        if p["it"] != q["it"] or not np.array_equal(p["pool"], q["pool"]):
            bad.append(f"spawn pool #{i} (it {p['it']} / {q['it']}) differs")
            break
    if t0_temp:
        print(f"note: decision-0 temperatures differ in {len(t0_temp)} (iteration, tensor) "
              f"pairs (single-map set_unstuck_temp quirk), e.g. {t0_temp[:2]}")
    if boot_noise:
        print(f"note: truncation-bootstrap float noise in b_rew (done rows only, max "
              f"{max(x[2] for x in boot_noise):.3g}) at {len(boot_noise)} iteration(s)")
    print(f"compared {len(its)} iterations, {len(pa)} spawn pools, {n} envs of slot {k}")
    return bad


if __name__ == "__main__":
    mode = sys.argv[1]
    store = Path(sys.argv[2])
    if mode == "compare":
        res = compare(store, int(sys.argv[3]))
        print("\n".join(res) if res else "IDENTICAL")
        sys.exit(1 if res else 0)
    k = int(sys.argv[3]) if mode == "replay" else 0
    run(mode, store, k, sys.argv[sys.argv.index("--") + 1:])
